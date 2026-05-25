# uc_decomp_4b.py
# DECOMP (CCG) algorithm for b-parameter counterfactual explanations in UC.
from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Callable, Set
import json
import pathlib
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from uc_pipeline import (
    build_network_uc_model, NetworkUCData, IndexMap, _optimize_with_retry,
)
from uc_master_relax_4b import (
    remove_fixed_fmax_constraints_4b,
    add_variable_fmax_constraints_4b,
)
from uc_branch_sandwich_4b import _GurobiKeepAlive


# ============================================================
# Helpers
# ============================================================

def _compute_vw_from_u(
    u_j: np.ndarray, u_init: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute startup (v) and shutdown (w) matrices from u^j and u_init."""
    nG, T = u_j.shape
    v = np.zeros((nG, T), dtype=float)
    w = np.zeros((nG, T), dtype=float)
    v[:, 0] = np.maximum(0.0, u_j[:, 0] - u_init)
    w[:, 0] = np.maximum(0.0, u_init - u_j[:, 0])
    for t in range(1, T):
        v[:, t] = np.maximum(0.0, u_j[:, t] - u_j[:, t - 1])
        w[:, t] = np.maximum(0.0, u_j[:, t - 1] - u_j[:, t])
    return v, w


def _topology(data: NetworkUCData):
    """Return gens_at_bus, rens_at_bus, in_lines, out_lines lists."""
    nB = int(data.nB)
    gens_at_bus = [[] for _ in range(nB)]
    rens_at_bus = [[] for _ in range(nB)]
    in_lines  = [[] for _ in range(nB)]
    out_lines = [[] for _ in range(nB)]
    for g, gen in enumerate(data.gens):
        gens_at_bus[int(gen.bus)].append(g)
    for r, ren in enumerate(data.rens):
        rens_at_bus[int(ren.bus)].append(r)
    for ell, line in enumerate(data.lines):
        out_lines[int(line.fr)].append(ell)
        in_lines[int(line.to)].append(ell)
    return gens_at_bus, rens_at_bus, in_lines, out_lines


# ============================================================
# Main class
# ============================================================

class UCDecomp4b:
    """
    DECOMP (Column-and-Constraint Generation) for b-parameter CE in UC.

    Each iteration adds a full KKT block for the UC LP at commitment pattern u^j,
    growing the master MILP until convergence or cycle detection.
    """

    def __init__(
        self,
        oracle,
        data: NetworkUCData,
        idx: IndexMap,
        cvec: np.ndarray,
        foil_extra_constr_fn: Callable,
        b0: np.ndarray,
        b_bounds: Tuple[np.ndarray, np.ndarray],
        b_free_idx: List[int],
        big_M_mu: float,
        big_M_mu_others: Optional[Dict[str, float]] = None,
        eps_weak: float = 1e-3,
        eps_obj: float = 1e-3,
        max_iter: int = 15,
        output_flag: int = 0,
        verbose: bool = True,
        w: Optional[np.ndarray] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1,
        keepalive_interval: float = 1200.0,
        big_M_multiplier: float = 1.0,  # set >1 (e.g. 10) to debug big-M tightness
        master_time_limit: Optional[float] = None,  # seconds per master MILP solve
        master_output_flag: int = 0,   # 1 to print Gurobi log for every master solve
        master_mip_gap: float = 1e-4,  # relative MIP gap tolerance
        comp_mode: str = "sos1",  # complementarity mode: "bigM" | "indicator" | "sos1"
        b_hat_hint: Optional[np.ndarray] = None,  # known CE (e.g. from B&S) to warm-start each master
    ):
        self.oracle   = oracle
        self.data     = data
        self.idx      = idx
        self.cvec     = cvec
        self.foil_extra = foil_extra_constr_fn

        self.b0 = np.array(b0, dtype=float)
        self.bL = np.maximum(np.array(b_bounds[0], dtype=float), self.b0)  # expansion-only
        self.bU = np.array(b_bounds[1], dtype=float)
        self.free = list(b_free_idx)
        self._free_set = set(b_free_idx)

        self.big_M_mu         = float(big_M_mu)
        self.big_M_oth        = big_M_mu_others or {}
        self.big_M_multiplier = float(big_M_multiplier)
        self.master_time_limit  = float(master_time_limit) if master_time_limit is not None else None
        self.master_output_flag = int(master_output_flag)
        self.master_mip_gap     = float(master_mip_gap)
        if comp_mode not in ("bigM", "indicator", "sos1"):
            raise ValueError(f"comp_mode must be 'bigM', 'indicator', or 'sos1'; got {comp_mode!r}")
        self.comp_mode = comp_mode

        self.eps_weak = float(eps_weak)
        self.eps_obj  = float(eps_obj)
        self.max_iter = int(max_iter)
        self.output_flag = int(output_flag)
        self.verbose  = bool(verbose)

        nL = len(data.lines)
        if w is None:
            # Default: normalized expansion fraction alpha = (b-b0)/b0
            self.w = np.where(self.b0 > 0, 1.0 / self.b0, 1.0)
        else:
            self.w = np.asarray(w, dtype=float).reshape(-1)

        self.checkpoint_path     = pathlib.Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_interval = int(checkpoint_interval)
        self._keepalive = _GurobiKeepAlive(interval=keepalive_interval)

        self.b_hat_hint  = np.array(b_hat_hint, dtype=float) if b_hat_hint is not None else None

        self.best_F      = float("inf")
        self.best_b      = None
        self.best_v_plain = None
        self.best_v_foil  = None
        self._master_LB  = 0.0

        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def F(self, b: np.ndarray) -> float:
        return float(np.sum(self.w[self.free] * np.abs(b[self.free] - self.b0[self.free])))

    def _pattern_key(self, u: np.ndarray) -> tuple:
        return tuple(u.ravel().astype(int).tolist())

    def _update_incumbent(self, b, Fb, v_plain, v_foil):
        if Fb < self.best_F:
            self.best_F = Fb
            self.best_b = b.copy()
            self.best_v_plain = v_plain
            self.best_v_foil  = v_foil
            if self.verbose:
                print(f"  [DECOMP] New incumbent: F={Fb:.6f}")

    def _save_checkpoint(self, iteration: int, master_LB: float):
        if self.checkpoint_path is None:
            return
        payload = {
            "iteration": iteration, "master_LB": master_LB,
            "best_F": self.best_F,
            "best_b": self.best_b.tolist() if self.best_b is not None else None,
            "best_v_plain": self.best_v_plain,
            "best_v_foil":  self.best_v_foil,
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        tmp.replace(self.checkpoint_path)

    def _load_checkpoint(self):
        try:
            with open(self.checkpoint_path) as fh:
                d = json.load(fh)
            self.best_F = float(d.get("best_F", float("inf")))
            bb = d.get("best_b")
            self.best_b = np.array(bb, dtype=float) if bb is not None else None
            self.best_v_plain = d.get("best_v_plain")
            self.best_v_foil  = d.get("best_v_foil")
            if self.verbose:
                print(f"[DECOMP] Checkpoint loaded: best_F={self.best_F:.6f}")
        except Exception as e:
            if self.verbose:
                print(f"[DECOMP] Checkpoint load failed: {e}")

    # ------------------------------------------------------------------
    # Diagnostic: fix b to a known CE and check if KKT block is feasible
    # ------------------------------------------------------------------

    def debug_fix_b(
        self,
        b_test: np.ndarray,
        window_size: int, per_bus_neutrality: bool,
        u_init, p_init, on_time_init, off_time_init,
        iis_path: str = "debug_fix_b_infeas.ilp",
        u_j_override: Optional[np.ndarray] = None,
        return_values: bool = False,
        verbose: Optional[bool] = None,
    ) -> Dict:
        """
        Build master, add first KKT block for u^j, fix b = b_test, then solve.

        Two modes (controlled by `u_j_override`):
        - Self-consistent (default, u_j_override=None):
            u^j = plain-optimal at b_test.  Tests whether the formulation is
            correct at a known CE.
        - Cross-pattern (u_j_override provided):
            u^j is taken from the override.  Useful for testing whether the
            master at iteration k (with KKT block built for u_1 from Iter 1)
            is feasible at b_test (e.g., b_hat).  This is the actual model
            the DECOMP run() is searching at Iter 2+, with b fixed.

        Returns dict with:
            feasible  : bool — Phase 3 OPTIMAL/SUBOPTIMAL
            obj       : float — L1 objective at solution (≈ F(b_test))
            iis       : str   — IIS file path if infeasible
            cut_slack : float — slack of optimality cut at solution
                                = LP_cost(u^j, b_test) − foil_cost
                                Tiny slack ⇒ cut nearly binding at b_test
                                ⇒ very tight feasible region for B&B.
        """
        # `verbose` parameter overrides instance default for this call (e.g.
        # when invoked silently from _full_integer_warm_start).
        _vb = bool(verbose) if verbose is not None else True
        def _p(*args, **kwargs):
            if _vb:
                kwargs.setdefault("flush", True)
                print(*args, **kwargs)

        if u_j_override is not None:
            u_k = np.asarray(u_j_override, dtype=int)
            _p(f"[debug_fix_b] Using provided u^j override "
               f"(cross-pattern test, sum(u)={int(u_k.sum())})")
        else:
            _p(f"[debug_fix_b] Querying plain oracle at b_test ...")
            vp, _, sol_p = self.oracle.solve_plain(b_test)
            if sol_p is None:
                return {"feasible": None, "obj": None, "iis": None, "cut_slack": None,
                        "error": "plain UC infeasible at b_test"}
            u_k = np.round(sol_p["u"]).astype(int)
            _p(f"[debug_fix_b] v_plain={vp:.4f}  (self-consistent u^j)")

        _p(f"[debug_fix_b] Building master ...")
        m, master_vars = self._build_master_base(
            window_size, per_bus_neutrality,
            u_init, p_init, on_time_init, off_time_init,
        )

        # Phase 1: solve without KKT block (foil + b vars only)
        _optimize_with_retry(m)
        _p(f"[debug_fix_b] Base master status={m.Status}  obj={m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else 'N/A'}")

        # Fix b to b_test
        nL = len(self.data.lines)
        b_fix_constrs = []
        for ell in range(nL):
            if ell in self._free_set:
                c = m.addConstr(master_vars["b"][ell] == float(b_test[ell]),
                                name=f"fix_b[{ell}]")
                b_fix_constrs.append(c)
        m.update()

        # Phase 2: with fixed b, no KKT block
        _optimize_with_retry(m)
        _p(f"[debug_fix_b] Fixed-b, no KKT: status={m.Status}  "
           f"obj={m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else 'N/A'}")
        if m.Status == GRB.INFEASIBLE:
            _p("[debug_fix_b] INFEASIBLE before KKT block — foil block is wrong.")
            try:
                m.computeIIS(); m.write(iis_path)
                _p(f"[debug_fix_b] IIS → {iis_path}")
            except Exception: pass
            m.dispose()
            return {"feasible": False, "obj": None, "iis": iis_path, "cut_slack": None,
                    "values": None, "error": "infeasible before KKT block"}

        # Phase 3: add first KKT block for u_k
        _p(f"[debug_fix_b] Adding KKT block for u_k ...")
        self._add_iteration_block(m, master_vars, 0, u_k, u_init, p_init,
                                   window_size, per_bus_neutrality)
        _optimize_with_retry(m)
        status = m.Status
        obj    = m.ObjVal if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None
        iis_out = None
        cut_slack = None
        values: Optional[Dict[str, float]] = None
        _p(f"[debug_fix_b] Fixed-b + KKT block: status={status}  obj={obj}")
        if status == GRB.INFEASIBLE:
            _p("[debug_fix_b] INFEASIBLE after KKT block — KKT formulation bug.")
            try:
                m.computeIIS(); m.write(iis_path)
                iis_out = iis_path
                _p(f"[debug_fix_b] IIS → {iis_path}")
            except Exception as e:
                _p(f"[debug_fix_b] IIS failed: {e}")
        else:
            try:
                c = m.getConstrByName("opt_cut_0")
                if c is not None:
                    cut_slack = float(c.Slack)
                    _p(f"[debug_fix_b] Optimality cut slack at solution: "
                       f"{cut_slack:.4f}  (tiny ⇒ cut nearly binding)")
            except Exception as e:
                _p(f"[debug_fix_b] cut slack read failed: {e}")
            if return_values and m.SolCount > 0:
                values = {v.VarName: float(v.X) for v in m.getVars()}
        m.dispose()
        return {"feasible": status in (GRB.OPTIMAL, GRB.SUBOPTIMAL),
                "obj": obj, "iis": iis_out, "cut_slack": cut_slack,
                "values": values}

    # ------------------------------------------------------------------
    # Diagnostic helper: replicate Iter 1 to extract (b_1, u_1)
    # ------------------------------------------------------------------

    def iter1_pattern(
        self,
        window_size: int, per_bus_neutrality: bool,
        u_init, p_init, on_time_init, off_time_init,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Replicate Iter 1 of run(): build base master (foil + b + L1, no KKT),
        solve, then call solve_plain at the resulting b_1 to get u_1.

        Returns (b_1, u_1, F_1).  Any element may be None on failure.

        Use case: feed u_1 into debug_fix_b(..., u_j_override=u_1) to test
        whether the actual Iter-2 model (KKT block built for u_1) is feasible
        at a target b (e.g., b_hat from B&S).
        """
        if self.verbose:
            print("[iter1_pattern] Building base master (no KKT) ...")
        m, master_vars = self._build_master_base(
            window_size, per_bus_neutrality,
            u_init, p_init, on_time_init, off_time_init,
        )
        _optimize_with_retry(m)
        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            if self.verbose:
                print(f"[iter1_pattern] Iter 1 master status={m.Status} — abort")
            m.dispose()
            return None, None, None

        nL = len(self.data.lines)
        b_1 = np.array([
            float(master_vars["b"][ell].X) if ell in self._free_set
            else float(self.b0[ell])
            for ell in range(nL)
        ])
        F_1 = float(m.ObjVal)
        m.dispose()

        if self.verbose:
            print(f"[iter1_pattern] b_1: F={F_1:.4f}  Δb={np.sum(np.abs(b_1 - self.b0)):.4f}")
            print("[iter1_pattern] Querying plain oracle at b_1 ...")
        vp, _, sol_p = self.oracle.solve_plain(b_1)
        if sol_p is None:
            if self.verbose:
                print("[iter1_pattern] Plain UC infeasible at b_1 — u_1 unavailable")
            return b_1, None, F_1

        u_1 = np.round(sol_p["u"]).astype(int)
        if self.verbose:
            print(f"[iter1_pattern] v_plain(b_1)={vp:.4f}  sum(u_1)={int(u_1.sum())}")
        return b_1, u_1, F_1

    # ------------------------------------------------------------------
    # Master base: foil block + shared b variables + L1 objective
    # ------------------------------------------------------------------

    def _build_master_base(
        self,
        window_size: int, per_bus_neutrality: bool,
        u_init, p_init, on_time_init, off_time_init,
    ) -> Tuple[gp.Model, Dict]:
        nL = len(self.data.lines)
        T  = int(self.data.T)

        # Foil UC model (binaries kept as INTEGER for u_foil)
        m, var = build_network_uc_model(
            data=self.data,
            window_size=window_size,
            per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init,
            on_time_init=on_time_init, off_time_init=off_time_init,
            output_flag=self.output_flag,
        )

        # Remove fixed flow limits and replace with shared b variables
        remove_fixed_fmax_constraints_4b(m)

        b_vars: Dict[int, object] = {}
        for ell in range(nL):
            if ell in self._free_set:
                b_vars[ell] = m.addVar(
                    lb=float(self.bL[ell]), ub=float(self.bU[ell]),
                    vtype=GRB.CONTINUOUS, name=f"b[{ell}]",
                )
            else:
                b_vars[ell] = float(self.b0[ell])

        add_variable_fmax_constraints_4b(m, var, b_vars, nL=nL, T=T,
                                         name_prefix="foil_fmax")

        # Apply foil constraints (e.g. emissions ≤ threshold)
        self.foil_extra(m, var)

        # L1 linearization: |b[ell] - b0[ell]| = bp[ell] + bm[ell]
        bp, bm = {}, {}
        for ell in self.free:
            bp[ell] = m.addVar(lb=0.0, name=f"bp[{ell}]")
            bm[ell] = m.addVar(lb=0.0, name=f"bm[{ell}]")
            m.addConstr(
                b_vars[ell] - float(self.b0[ell]) == bp[ell] - bm[ell],
                name=f"l1[{ell}]",
            )

        m.setObjective(
            gp.quicksum(float(self.w[ell]) * (bp[ell] + bm[ell])
                        for ell in self.free),
            GRB.MINIMIZE,
        )
        m.update()

        master_vars = {"var": var, "b": b_vars, "bp": bp, "bm": bm, "sos_pairs": []}
        return m, master_vars

    # ------------------------------------------------------------------
    # KKT iteration block
    # ------------------------------------------------------------------

    def _add_iteration_block(
        self,
        m: gp.Model,
        master_vars: Dict,
        j: int,
        u_j: np.ndarray,
        u_init: np.ndarray,
        p_init: np.ndarray,
        window_size: int,
        per_bus_neutrality: bool,
    ):
        """Add primal LP + full KKT conditions for commitment pattern u^j."""
        data   = self.data
        idx    = self.idx
        cvec   = self.cvec
        b_vars = master_vars["b"]
        var_f  = master_vars["var"]   # foil block variables

        nG = len(data.gens)
        nR = len(data.rens)
        nB = int(data.nB)
        nL = len(data.lines)
        T  = int(data.T)
        W  = min(int(window_size), T)

        b0 = self.b0
        bU = self.bU
        free_set = self._free_set

        # Dual big-M floors.  big_M_multiplier > 1 loosens for debugging.
        VOLL  = float(data.voll)
        _mf   = self.big_M_multiplier          # 1.0 in production, e.g. 10.0 for debug
        # _max_c_curt computed first — needed by M_mu, M_shed, M_curt (Bug 7/8/9).
        _max_c_curt = (max(float(np.max(r.curt_cost)) for r in data.rens)
                       if data.rens else 0.0)
        # M_mu floor: μ_p = π_fr − π_to − ν + μ_m; in the radial / simplified case
        # ν ≈ π_to−π_fr and μ_p ≤ π_fr − π_to ≤ π_max − π_min = 2·VOLL + 2·c_curt.
        # Bug 9: was max(big_M_mu, VOLL) — used only π_min (= −VOLL) instead of the
        # full π range, same arithmetic error as Bug 1 for M_shed.
        M_mu    = max(self.big_M_mu, 2.0 * VOLL + 2.0 * _max_c_curt) * _mf
        M_gen   = max(self.big_M_oth.get("gen",   M_mu),    VOLL) * _mf
        M_ramp  = max(self.big_M_oth.get("ramp",  M_mu), 3.0 * VOLL) * _mf
        # M_shed: gsh_lb = VOLL + pi + gsh_ub; pi ≥ −VOLL when only shedding, gsh_ub ≥ 0.
        # Bug 1: was VOLL (off by pi range). Bug 8 (cascade from Bug 7): after M_curt fix,
        # pi_max = max_c_curt + M_curt = 2*max_c_curt + VOLL (not VOLL), so
        # gsh_lb ≤ VOLL + pi_max = 2*VOLL + 2*max_c_curt.
        M_shed  = (2.0 * VOLL + 2.0 * _max_c_curt) * _mf
        # M_curt: gcu_lb = c_curt − pi + gcu_ub; pi ≥ −VOLL when shedding active,
        # so gcu_lb ≤ max_c_curt + VOLL.  Bug 7: was max(c_curt, VOLL) = VOLL,
        # needs a SUM (same root cause as Bug 1 — pi range ignored).
        M_curt  = (_max_c_curt + VOLL) * _mf
        M_shift = max(self.big_M_oth.get("shift", M_mu),    VOLL) * _mf

        s = f"_{j}"   # variable name suffix

        v_j, w_j = _compute_vw_from_u(u_j, u_init)
        gens_at_bus, rens_at_bus, in_lines, out_lines = _topology(data)
        slack_b = int(data.slack_bus)

        # ---- primal LP variables ----
        p_j     = m.addVars(nG, T, lb=0.0,          name=f"p{s}")
        f_j     = m.addVars(nL, T, lb=-GRB.INFINITY, name=f"f{s}")
        th_j    = m.addVars(nB, T, lb=-GRB.INFINITY, name=f"th{s}")
        shed_j  = m.addVars(nB, T, lb=0.0,          name=f"shed{s}")
        curt_j  = m.addVars(nR, T, lb=0.0,          name=f"curt{s}")
        sp_j    = m.addVars(nB, T, lb=0.0,          name=f"sp{s}")
        sm_j    = m.addVars(nB, T, lb=0.0,          name=f"sm{s}")

        # ---- dual variables for equalities (free) ----
        pi_j    = m.addVars(nB, T, lb=-GRB.INFINITY, name=f"pi{s}")    # balance
        nu_j    = m.addVars(nL, T, lb=-GRB.INFINITY, name=f"nu{s}")    # dcflow
        sig_j   = m.addVars(T,    lb=-GRB.INFINITY, name=f"sig{s}")   # slack bus
        # Neutrality duals: (bus_or_0, t0) -> var
        eta_j: Dict = {}
        if per_bus_neutrality:
            for b in range(nB):
                for t0 in range(T - W + 1):
                    eta_j[(b, t0)] = m.addVar(lb=-GRB.INFINITY,
                                              name=f"eta{s}[{b},{t0}]")
        else:
            for t0 in range(T - W + 1):
                eta_j[(0, t0)] = m.addVar(lb=-GRB.INFINITY,
                                          name=f"eta{s}[{t0}]")

        # ---- dual variables for inequalities (≥ 0) ----
        lam_hi    = m.addVars(nG, T, lb=0.0, name=f"lhi{s}")    # pmax
        lam_lo    = m.addVars(nG, T, lb=0.0, name=f"llo{s}")    # pmin (Pmin>=0 so lb=0 is implied)
        rho_up_i  = m.addVars(nG,    lb=0.0, name=f"rui{s}")    # ramp_up_init
        rho_dn_i  = m.addVars(nG,    lb=0.0, name=f"rdi{s}")    # ramp_dn_init
        # ramp_up/dn only for t >= 1
        rho_up    = m.addVars(nG, range(1, T), lb=0.0, name=f"rup{s}")
        rho_dn    = m.addVars(nG, range(1, T), lb=0.0, name=f"rdn{s}")
        mu_p      = m.addVars(nL, T, lb=0.0, name=f"mup{s}")  # fmax (no ub: big-M in complementarity)
        mu_m      = m.addVars(nL, T, lb=0.0, name=f"mum{s}")  # fmin
        gsh_ub    = m.addVars(nB, T, lb=0.0, name=f"gsh_ub{s}")
        gsh_lb    = m.addVars(nB, T, lb=0.0, name=f"gsh_lb{s}")
        gcu_ub    = m.addVars(nR, T, lb=0.0, name=f"gcu_ub{s}")
        gcu_lb    = m.addVars(nR, T, lb=0.0, name=f"gcu_lb{s}")
        gsp_ub    = m.addVars(nB, T, lb=0.0, name=f"gsp_ub{s}")
        gsp_lb    = m.addVars(nB, T, lb=0.0, name=f"gsp_lb{s}")
        gsm_ub    = m.addVars(nB, T, lb=0.0, name=f"gsm_ub{s}")
        gsm_lb    = m.addVars(nB, T, lb=0.0, name=f"gsm_lb{s}")

        # ---- primal feasibility constraints ----
        for g, gen in enumerate(data.gens):
            Pmin, Pmax = float(gen.Pmin), float(gen.Pmax)
            RU, RD    = float(gen.RU),   float(gen.RD)
            p0g = float(p_init[g])
            for t in range(T):
                uj = float(u_j[g, t])
                m.addConstr(p_j[g, t] <= Pmax * uj)
                m.addConstr(p_j[g, t] >= Pmin * uj)
            m.addConstr(p_j[g, 0] - p0g <= RU)
            m.addConstr(p0g - p_j[g, 0] <= RD)
            for t in range(1, T):
                m.addConstr(p_j[g, t] - p_j[g, t-1] <= RU)
                m.addConstr(p_j[g, t-1] - p_j[g, t] <= RD)

        for r, ren in enumerate(data.rens):
            for t in range(T):
                m.addConstr(curt_j[r, t] <= float(ren.avail[t]))

        for b in range(nB):
            for t in range(T):
                m.addConstr(shed_j[b, t] <= float(data.demand[b, t]))
                m.addConstr(sp_j[b, t]   <= float(data.Splus_max[b, t]))
                m.addConstr(sm_j[b, t]   <= float(data.Sminus_max[b, t]))

        # Neutrality
        if per_bus_neutrality:
            for b in range(nB):
                for t0 in range(T - W + 1):
                    m.addConstr(gp.quicksum(
                        sp_j[b, t] - sm_j[b, t] for t in range(t0, t0 + W)) == 0)
        else:
            for t0 in range(T - W + 1):
                m.addConstr(gp.quicksum(
                    sp_j[b, t] - sm_j[b, t]
                    for b in range(nB) for t in range(t0, t0 + W)) == 0)

        # DC flow + variable flow limits (shared b)
        for ell, line in enumerate(data.lines):
            fr, to = int(line.fr), int(line.to)
            bl = float(line.b)
            cap = b_vars[ell] if ell in free_set else float(b0[ell])
            for t in range(T):
                m.addConstr(f_j[ell, t] == bl * (th_j[fr, t] - th_j[to, t]))
                m.addConstr( f_j[ell, t] <= cap)
                m.addConstr(-f_j[ell, t] <= cap)

        # Slack bus
        for t in range(T):
            m.addConstr(th_j[slack_b, t] == 0.0)

        # Nodal balance
        for b in range(nB):
            for t in range(T):
                gen_inj = gp.quicksum(p_j[g, t] for g in gens_at_bus[b])
                ren_inj = gp.quicksum(
                    float(data.rens[r].avail[t]) - curt_j[r, t]
                    for r in rens_at_bus[b])
                m.addConstr(
                    gen_inj + ren_inj + shed_j[b, t]
                    - float(data.demand[b, t]) - sp_j[b, t] + sm_j[b, t]
                    + gp.quicksum(f_j[ell, t] for ell in in_lines[b])
                    - gp.quicksum(f_j[ell, t] for ell in out_lines[b]) == 0)

        # ---- KKT stationarity conditions ----

        # Helper: sum of eta duals for windows containing time t at bus b
        def _eta_sum(bus, t, sign=1.0):
            e = gp.LinExpr()
            lo = max(0, t - W + 1)
            hi = min(t, T - W) + 1
            if per_bus_neutrality:
                for t0 in range(lo, hi):
                    e.add(eta_j[(bus, t0)], sign)
            else:
                for t0 in range(lo, hi):
                    e.add(eta_j[(0, t0)], sign)
            return e

        # ∂L/∂p[g,t]
        for g, gen in enumerate(data.gens):
            bus_g = int(gen.bus)
            cp = float(cvec[idx.p[g, 0]])
            for t in range(T):
                e = gp.LinExpr(cp)
                e += lam_hi[g, t] - lam_lo[g, t]
                e += pi_j[bus_g, t]
                if t == 0:
                    e += rho_up_i[g] - rho_dn_i[g]
                else:
                    e += rho_up[g, t] - rho_dn[g, t]
                if t < T - 1:
                    e -= rho_up[g, t + 1]
                    e += rho_dn[g, t + 1]
                m.addConstr(e == 0.0)

        # ∂L/∂f[ell,t]
        for ell, line in enumerate(data.lines):
            fr, to = int(line.fr), int(line.to)
            cf = float(cvec[idx.f[ell, 0]])
            for t in range(T):
                m.addConstr(
                    cf + nu_j[ell, t] + mu_p[ell, t] - mu_m[ell, t]
                    + pi_j[to, t] - pi_j[fr, t] == 0.0)

        # ∂L/∂theta[b,t]
        for b in range(nB):
            for t in range(T):
                e = gp.LinExpr()
                for ell in out_lines[b]:
                    e.add(nu_j[ell, t], -float(data.lines[ell].b))
                for ell in in_lines[b]:
                    e.add(nu_j[ell, t],  float(data.lines[ell].b))
                if b == slack_b:
                    e.add(sig_j[t], 1.0)
                m.addConstr(e == 0.0)

        # ∂L/∂shed[b,t]  (shed appears +1 in balance)
        for b in range(nB):
            for t in range(T):
                m.addConstr(float(data.voll) + gsh_ub[b, t]
                            - gsh_lb[b, t] + pi_j[b, t] == 0.0)

        # ∂L/∂curt[r,t]  (curt appears as -curt in ren_inj → -1 in balance)
        for r, ren in enumerate(data.rens):
            bus_r = int(ren.bus)
            for t in range(T):
                m.addConstr(float(cvec[idx.curt[r, t]])
                            + gcu_ub[r, t] - gcu_lb[r, t]
                            - pi_j[bus_r, t] == 0.0)

        # ∂L/∂splus[b,t]  (splus appears -1 in balance, +1 in neutrality)
        for b in range(nB):
            for t in range(T):
                e = gp.LinExpr(float(cvec[idx.splus[b, t]]))
                e += gsp_ub[b, t] - gsp_lb[b, t]
                e -= pi_j[b, t]
                e += _eta_sum(b, t, +1.0)
                m.addConstr(e == 0.0)

        # ∂L/∂sminus[b,t]  (sminus appears +1 in balance, -1 in neutrality)
        for b in range(nB):
            for t in range(T):
                e = gp.LinExpr(float(cvec[idx.sminus[b, t]]))
                e += gsm_ub[b, t] - gsm_lb[b, t]
                e += pi_j[b, t]
                e += _eta_sum(b, t, -1.0)
                m.addConstr(e == 0.0)

        # ---- Complementarity ----
        # M_d, M_s are ignored for indicator/sos1 modes (kept in signature for
        # uniform call sites so nothing else changes).
        # z's are named deterministically (zbm{s}_{tag}) so analytic_warm_start
        # can look them up by name across temp/main models.
        def _comp(dual_var, slack_expr, M_d, M_s, tag: str):
            """Enforce dual ⟂ slack (at most one nonzero).  `tag` is a unique
            string per call (per pattern j) used to name the z variable.

            bigM     : dual ≤ M_d*(1-z),  slack ≤ M_s*z            [binary]
            indicator: z=0 → dual=0,       z=1 → slack=0            [binary, M-free]
            sos1     : SOS1({dual, slack})                           [no binary, M-free]
            """
            mode = self.comp_mode
            if mode == "bigM":
                z = m.addVar(vtype=GRB.BINARY, name=f"zbm{s}_{tag}")
                m.addConstr(dual_var <= M_d * (1 - z))
                m.addConstr(slack_expr <= M_s * z)

            elif mode == "indicator":
                z = m.addVar(vtype=GRB.BINARY, name=f"zind{s}_{tag}")
                m.addGenConstrIndicator(z, False, dual_var, GRB.LESS_EQUAL, 0.0)
                if isinstance(slack_expr, gp.Var):
                    m.addGenConstrIndicator(z, True, slack_expr, GRB.LESS_EQUAL, 0.0)
                else:
                    s_aux = m.addVar(lb=-GRB.INFINITY, name=f"saux_ind{s}_{tag}")
                    m.addConstr(s_aux == slack_expr)
                    m.addGenConstrIndicator(z, True, s_aux, GRB.LESS_EQUAL, 0.0)

            elif mode == "sos1":
                if isinstance(slack_expr, gp.Var):
                    m.addSOS(GRB.SOS_TYPE1, [dual_var, slack_expr])
                    master_vars["sos_pairs"].append((dual_var, slack_expr))
                else:
                    s_aux = m.addVar(lb=0.0, name=f"saux_sos{s}_{tag}")
                    m.addConstr(s_aux == slack_expr)
                    m.addSOS(GRB.SOS_TYPE1, [dual_var, s_aux])
                    master_vars["sos_pairs"].append((dual_var, s_aux))

        # Flow limits
        for ell in range(nL):
            for t in range(T):
                if ell in free_set:
                    cap = b_vars[ell]
                    M_s = 2.0 * float(bU[ell])
                else:
                    cap = float(b0[ell])
                    M_s = 2.0 * cap if cap > 0 else 1.0
                _comp(mu_p[ell, t], cap - f_j[ell, t],  M_mu, M_s, tag=f"mup_{ell}_{t}")
                _comp(mu_m[ell, t], cap + f_j[ell, t],  M_mu, M_s, tag=f"mum_{ell}_{t}")

        # Generator bounds
        for g, gen in enumerate(data.gens):
            Pmin, Pmax = float(gen.Pmin), float(gen.Pmax)
            for t in range(T):
                uj = float(u_j[g, t])
                _comp(lam_hi[g, t], Pmax * uj - p_j[g, t], M_gen, Pmax + 1.0, tag=f"lhi_{g}_{t}")
                _comp(lam_lo[g, t], p_j[g, t] - Pmin * uj, M_gen, Pmax + 1.0, tag=f"llo_{g}_{t}")

        # Ramp — use direction-specific big-M since RD may differ from RU
        for g, gen in enumerate(data.gens):
            RU, RD = float(gen.RU), float(gen.RD)
            Pmax   = float(gen.Pmax)
            p0g    = float(p_init[g])
            M_up   = Pmax + RU
            M_dn   = Pmax + RD
            M_up_i = p0g + RU
            M_dn_i = Pmax + RD
            _comp(rho_up_i[g], p0g + RU - p_j[g, 0], M_ramp, max(M_up_i, 1.0), tag=f"rui_{g}")
            _comp(rho_dn_i[g], p_j[g, 0] - p0g + RD, M_ramp, max(M_dn_i, 1.0), tag=f"rdi_{g}")
            for t in range(1, T):
                _comp(rho_up[g, t], p_j[g, t-1] + RU - p_j[g, t], M_ramp, max(M_up, 1.0), tag=f"rup_{g}_{t}")
                _comp(rho_dn[g, t], p_j[g, t]   + RD - p_j[g, t-1], M_ramp, max(M_dn, 1.0), tag=f"rdn_{g}_{t}")

        # Shed
        for b in range(nB):
            for t in range(T):
                d = float(data.demand[b, t])
                _comp(gsh_ub[b, t], d - shed_j[b, t], M_shed, d + 1.0, tag=f"gshub_{b}_{t}")
                _comp(gsh_lb[b, t], shed_j[b, t],     M_shed, d + 1.0, tag=f"gshlb_{b}_{t}")

        # Curtailment
        for r, ren in enumerate(data.rens):
            for t in range(T):
                av = float(ren.avail[t])
                _comp(gcu_ub[r, t], av - curt_j[r, t], M_curt, av + 1.0, tag=f"gcuub_{r}_{t}")
                _comp(gcu_lb[r, t], curt_j[r, t],      M_curt, av + 1.0, tag=f"gculb_{r}_{t}")

        # Shifting
        for b in range(nB):
            for t in range(T):
                spm = float(data.Splus_max[b, t])
                smm = float(data.Sminus_max[b, t])
                _comp(gsp_ub[b, t], spm - sp_j[b, t], M_shift, spm + 1.0, tag=f"gspub_{b}_{t}")
                _comp(gsp_lb[b, t], sp_j[b, t],       M_shift, spm + 1.0, tag=f"gsplb_{b}_{t}")
                _comp(gsm_ub[b, t], smm - sm_j[b, t], M_shift, smm + 1.0, tag=f"gsmub_{b}_{t}")
                _comp(gsm_lb[b, t], sm_j[b, t],       M_shift, smm + 1.0, tag=f"gsmlb_{b}_{t}")

        # ---- Optimality cut: c^T(x_foil) ≤ c^T(x^j, u^j) ----
        # Foil cost (master decision variables — includes binary u_foil)
        foil_cost = gp.LinExpr()
        vf = var_f
        for g in range(nG):
            for t in range(T):
                foil_cost.add(vf["u"][g, t], float(cvec[idx.u[g, t]]))
                foil_cost.add(vf["v"][g, t], float(cvec[idx.v[g, t]]))
                foil_cost.add(vf["w"][g, t], float(cvec[idx.w[g, t]]))
                foil_cost.add(vf["p"][g, t], float(cvec[idx.p[g, t]]))
        for b in range(nB):
            for t in range(T):
                foil_cost.add(vf["shed"][b, t],   float(cvec[idx.shed[b, t]]))
                foil_cost.add(vf["splus"][b, t],  float(cvec[idx.splus[b, t]]))
                foil_cost.add(vf["sminus"][b, t], float(cvec[idx.sminus[b, t]]))
        for r in range(nR):
            for t in range(T):
                foil_cost.add(vf["curt"][r, t], float(cvec[idx.curt[r, t]]))

        # LP block cost: constant part (u_j, v_j, w_j) + variable part (x^j)
        lp_const = 0.0
        lp_var   = gp.LinExpr()
        for g in range(nG):
            for t in range(T):
                lp_const += float(cvec[idx.u[g, t]]) * float(u_j[g, t])
                lp_const += float(cvec[idx.v[g, t]]) * float(v_j[g, t])
                lp_const += float(cvec[idx.w[g, t]]) * float(w_j[g, t])
                lp_var.add(p_j[g, t], float(cvec[idx.p[g, t]]))
        for b in range(nB):
            for t in range(T):
                lp_var.add(shed_j[b, t],  float(cvec[idx.shed[b, t]]))
                lp_var.add(sp_j[b, t],    float(cvec[idx.splus[b, t]]))
                lp_var.add(sm_j[b, t],    float(cvec[idx.sminus[b, t]]))
        for r in range(nR):
            for t in range(T):
                lp_var.add(curt_j[r, t], float(cvec[idx.curt[r, t]]))

        m.addConstr(foil_cost <= lp_var + lp_const, name=f"opt_cut{s}")
        m.update()

    # ------------------------------------------------------------------
    # MIP warm-start injection
    # ------------------------------------------------------------------

    def _inject_mip_start(
        self,
        m: gp.Model,
        master_vars: Dict,
        b_ws: np.ndarray,
        sol_dict: Dict,
        u_init,
    ):
        """Set Gurobi Start values from a known feasible foil solution."""
        var_f = master_vars["var"]
        nG = len(self.data.gens)
        nR = len(self.data.rens)
        nB = int(self.data.nB)
        nL = len(self.data.lines)
        T  = int(self.data.T)

        # b and L1 split vars
        for ell in self.free:
            v = master_vars["b"][ell]
            if isinstance(v, gp.Var):
                v.Start = float(b_ws[ell])
        for ell in self.free:
            d = float(b_ws[ell]) - float(self.b0[ell])
            if ell in master_vars.get("bp", {}):
                master_vars["bp"][ell].Start = max(0.0,  d)
            if ell in master_vars.get("bm", {}):
                master_vars["bm"][ell].Start = max(0.0, -d)

        # binary: u, v, w
        u_ws = np.round(np.asarray(sol_dict.get("u", np.zeros((nG, T))), float)).astype(int)
        v_ws, w_ws = _compute_vw_from_u(u_ws, np.asarray(u_init, float))
        for g in range(nG):
            for t in range(T):
                for key, val in [("u", u_ws[g, t]),
                                  ("v", v_ws[g, t]),
                                  ("w", w_ws[g, t])]:
                    try: var_f[key][g, t].Start = float(val)
                    except Exception: pass

        # continuous variables
        specs = [
            ("p",      nG, T),
            ("f",      nL, T),
            ("theta",  nB, T),
            ("shed",   nB, T),
            ("splus",  nB, T),
            ("sminus", nB, T),
            ("curt",   nR, T),
        ]
        for key, rows, cols in specs:
            arr = np.asarray(sol_dict.get(key, np.zeros((rows, cols))), float)
            if arr.shape != (rows, cols):
                continue
            for i in range(rows):
                for t in range(cols):
                    try: var_f[key][i, t].Start = float(arr[i, t])
                    except Exception: pass

        m.update()

    def _complete_warm_start(
        self,
        m: gp.Model,
        master_vars: Dict,
        b_hint: np.ndarray,
        foil_sol: Dict,
        u_init,
    ):
        """Solve LP at (b_hint, u_foil_hint) and inject ALL variable values as .Start.

        _inject_mip_start only sets foil/b variables.  After KKT blocks are added
        the model also contains LP primal vars (p_j, f_j, ...) and dual vars
        (pi_j, nu_j, mu_j, ...) plus SOS1 pairs — none of which have Start values.
        Gurobi's partial-start completion fails because too many variables are unset.

        This method fixes b=b_hint and u_foil=hint integer values, solves the LP
        relaxation (fast — seconds), and injects every LP variable value as .Start.
        Gurobi then has a complete feasible starting point and only needs to
        round the SOS1 pairs (trivial — most are already near-zero).
        """
        nG = len(self.data.gens)
        T  = int(self.data.T)
        var_f = master_vars["var"]

        # Fix b to b_hint via temporary equality constraints
        b_fix = []
        for ell in self.free:
            v = master_vars["b"][ell]
            if isinstance(v, gp.Var):
                b_fix.append(m.addConstr(v == float(b_hint[ell]),
                                         name=f"_ws_b[{ell}]"))

        # Fix u_foil to integer values via bounds
        u_ws = np.round(np.asarray(foil_sol.get("u", np.zeros((nG, T))), float)).astype(int)
        old_lb: Dict = {}
        old_ub: Dict = {}
        for g in range(nG):
            for t in range(T):
                try:
                    v = var_f["u"][g, t]
                    old_lb[(g, t)] = v.LB
                    old_ub[(g, t)] = v.UB
                    v.LB = float(u_ws[g, t])
                    v.UB = float(u_ws[g, t])
                except Exception:
                    pass

        m.update()

        # Solve LP relaxation (SOS1 constraints dropped, binaries relaxed)
        lp = m.relax()
        lp.Params.OutputFlag   = 0
        lp.Params.NumericFocus = 2
        lp.optimize()

        if lp.Status == GRB.OPTIMAL:
            # Build var-name → LP-value map and inject LP values.
            # Binary variables (u_foil, bigM z) are intentionally skipped:
            #   - u_foil was fixed above → LP value is exactly 0/1 but _inject_mip_start
            #     already handled those; double-injection is harmless but unnecessary.
            #   - bigM z variables: z_LP = 1 - dual/M_d can be close to 1 even when
            #     dual >> 0 (because M_d is large). Injecting z=1 then causes a
            #     warm-start violation on the dual ≤ M*(1-z) constraint. Skipping all
            #     binary injection here avoids the false rounding entirely — Gurobi
            #     completes the MIP start for z variables on its own.
            lp_vals: Dict[str, float] = {v.VarName: v.X for v in lp.getVars()}
            n_injected = 0
            for v in m.getVars():
                val = lp_vals.get(v.VarName)
                if val is None:
                    continue
                if v.VType == GRB.BINARY:
                    pass  # never inject binary Start from LP (see comment above)
                else:
                    v.Start = val
                    n_injected += 1
            m.update()
            if self.verbose:
                print(f"  [WS] LP at b_hint feasible (obj={lp.ObjVal:.4f}) "
                      f"— warm start injected ({n_injected}/{len(lp_vals)} vars)")
        else:
            if self.verbose:
                print(f"  [WS] LP at b_hint status={lp.Status} "
                      f"— skipping full warm start")

        lp.dispose()

        # Remove temporary b constraints
        m.remove(b_fix)

        # Restore u_foil bounds
        for g in range(nG):
            for t in range(T):
                try:
                    v = var_f["u"][g, t]
                    v.LB = old_lb[(g, t)]
                    v.UB = old_ub[(g, t)]
                except Exception:
                    pass

        m.update()

    def _full_integer_warm_start(
        self,
        m: gp.Model,
        master_vars: Dict,
        b_hint: np.ndarray,
        foil_sol: Dict,
        patterns: List[np.ndarray],
        window_size: int,
        per_bus_neutrality: bool,
        u_init: np.ndarray,
        p_init: np.ndarray,
        on_time_init: np.ndarray,
        off_time_init: np.ndarray,
        time_limit: float = 180.0,
    ) -> bool:
        """Public entry point: delegates to _analytic_warm_start.

        Earlier in-place / temp-model / debug_fix_b approaches all failed
        because Gurobi's MIP search couldn't find an integer-feasible z
        assignment in reasonable time (300+ s, 80K+ nodes, zero incumbents
        even though root LP = optimum).  _analytic_warm_start computes the
        warm-start values directly from LP duals — no MIP search needed,
        guaranteed fast and reliable.
        """
        if not patterns:
            return False
        return self._analytic_warm_start(
            m, master_vars, b_hint, foil_sol, patterns,
            window_size, per_bus_neutrality, u_init, p_init,
        )

    def _full_integer_warm_start_inline(
        self,
        m: gp.Model,
        master_vars: Dict,
        b_hint: np.ndarray,
        patterns: List[np.ndarray],
        window_size: int,
        per_bus_neutrality: bool,
        u_init: np.ndarray,
        p_init: np.ndarray,
        on_time_init: np.ndarray,
        off_time_init: np.ndarray,
        time_limit: float = 180.0,
    ) -> bool:
        """Build a FRESH temp master, mimic debug_fix_b's phased build
        (Phase 1: foil only; Phase 2: + b_fix; Phase 3..k+2: + each KKT block),
        solve to integer-feasible, transfer all values by name to the main `m`.

        Why a fresh temp:
        - Reusing `m` carries stale solve state (basis, incumbent, cuts) from
          the prior Iter k-1 solve, which biases Gurobi away from the fixed-b
          region — observed: 120s timeout for what `debug_fix_b` solves in <1s.

        Why we do NOT fix u_foil:
        - In `debug_fix_b`, u_foil is free.  Phase 2 picks SOME foil-feasible
          u_foil (degenerate optima), and Phase 3 retains freedom to repick
          for KKT extensibility.  Pinning u_foil to the B&S hint via bounds
          (earlier attempt) removed that freedom and Phase 3 timed out — the
          B&S u_foil happens not to admit a quick KKT-feasible (z) assignment.

        Why we re-add ALL prior KKT blocks (not just the latest):
        - The main master at Iter k has KKT blocks for u_0..u_{k-1}.  The temp
          must mirror this so variable names align: blocks contributed in the
          same order get suffixes "_0", "_1", ..., "_{k-1}".  Otherwise temp
          would only provide Start values for the latest block and the prior
          blocks' vars would be unset → Gurobi MIP-start completion fails.

        Returns True iff the temp solved and Start was injected into m.
        """
        if self.verbose:
            print(f"  [WS-FULL] Building temp model "
                  f"(b fixed, {len(patterns)} KKT block(s)) ...")
        m_tmp, mv_tmp = self._build_master_base(
            window_size, per_bus_neutrality,
            u_init, p_init, on_time_init, off_time_init,
        )

        nL = len(self.data.lines)

        # Phase 1: solve foil + b + L1 (no fix, no KKT) — establishes initial state.
        # IMPORTANT: do not override MIPFocus / NumericFocus here.  debug_fix_b
        # (which we've proven solves this structure in <1s) runs with Gurobi
        # defaults; setting MIPFocus=1 + NumericFocus=2 on the temp model
        # provoked status=9 timeouts in 180s — same problem.  Keep OutputFlag=0
        # to suppress log, set TimeLimit as a safety net only.
        m_tmp.Params.OutputFlag = 0
        m_tmp.Params.TimeLimit  = float(time_limit)
        _optimize_with_retry(m_tmp)
        if m_tmp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            if self.verbose:
                print(f"  [WS-FULL] temp Phase 1 status={m_tmp.Status} — abort")
            m_tmp.dispose()
            return False

        # Phase 2: add b-fix (u_foil stays binary-free), solve
        for ell in range(nL):
            if ell in self._free_set:
                m_tmp.addConstr(mv_tmp["b"][ell] == float(b_hint[ell]),
                                name=f"_ws_b[{ell}]")
        m_tmp.update()
        _optimize_with_retry(m_tmp)
        if m_tmp.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            if self.verbose:
                print(f"  [WS-FULL] temp Phase 2 status={m_tmp.Status} — abort")
            m_tmp.dispose()
            return False

        # Phase 3..k+2: add each prior KKT block in order, solve after each
        # (one solve per block keeps Gurobi's state similar to a sequential run
        #  and stays fast since b is already pinned)
        for j_idx, u_j in enumerate(patterns):
            self._add_iteration_block(
                m_tmp, mv_tmp, j_idx, u_j,
                u_init, p_init, window_size, per_bus_neutrality,
            )
            _optimize_with_retry(m_tmp)
            if m_tmp.SolCount == 0:
                if self.verbose:
                    print(f"  [WS-FULL] temp KKT-{j_idx} status={m_tmp.Status}, "
                          f"SolCount=0 — abort")
                m_tmp.dispose()
                return False

        # Extract values by VarName
        obj_tmp = float(m_tmp.ObjVal)
        values: Dict[str, float] = {v.VarName: float(v.X) for v in m_tmp.getVars()}
        m_tmp.dispose()

        # Transfer to main master m
        n_injected = 0
        for v in m.getVars():
            val = values.get(v.VarName)
            if val is None:
                continue
            try:
                v.Start = val
                n_injected += 1
            except Exception:
                pass
        m.update()
        if self.verbose:
            print(f"  [WS-FULL] temp solved (obj={obj_tmp:.4f}) — "
                  f"injected {n_injected}/{len(values)} vars into main master")
        return True

    # ------------------------------------------------------------------
    # Analytic warm start (the fix for B&B not finding incumbents)
    # ------------------------------------------------------------------

    def _solve_dispatch_lp(
        self,
        b_hint: np.ndarray,
        u_j: np.ndarray,
        window_size: int,
        per_bus_neutrality: bool,
        p_init: np.ndarray,
    ) -> Optional[Dict]:
        """Solve the UC dispatch LP at fixed (b=b_hint, u=u_j).

        Returns a dict mapping primal variable names and dual variable names
        to values, structured to MATCH the KKT block of `_add_iteration_block`.
        Includes:
          - primals: p, f, th, shed, curt, sp, sm (arrays)
          - free duals: pi (balance), nu (dcflow), sig (slack), eta (neutrality)
          - non-neg duals: lam_hi (pmax), lam_lo (pmin),
                            rho_up_i, rho_dn_i, rho_up, rho_dn,
                            mu_p (fmax), mu_m (fmin),
                            gsh_ub, gsh_lb, gcu_ub, gcu_lb,
                            gsp_ub, gsp_lb, gsm_ub, gsm_lb

        Returns None on LP failure.
        """
        data = self.data
        cvec = self.cvec
        idx  = self.idx
        nG = len(data.gens); nR = len(data.rens)
        nB = int(data.nB); nL = len(data.lines); T = int(data.T)
        W = min(int(window_size), T)
        gens_at_bus, rens_at_bus, in_lines, out_lines = _topology(data)
        slack_b = int(data.slack_bus)

        m = gp.Model()
        m.Params.OutputFlag = 0

        # Primal variables.
        # IMPORTANT: p has lb=-INFINITY here, NOT lb=0, even though master's
        # p_j has lb=0.  Reason: master's KKT block represents the lower bound
        # ONLY via the explicit c_pmin constraint (lam_lo).  If our LP also has
        # an implicit p ≥ 0 from lb=0, the LP optimality splits the lb dual
        # between c_pmin.Pi (lam_lo) and p.RC (implicit) — and master only
        # consumes the c_pmin part, leaving stationarity violated when u_j=0
        # (where both constraints are p ≥ 0 and redundant).  Setting
        # lb=-INFINITY here funnels all of the dual into c_pmin.Pi so master's
        # stationarity is exactly satisfied.
        # Same logic applies to shed/curt/sp/sm IF master had explicit
        # `var ≥ 0` constraints in addition to lb=0 — but it doesn't (gsh_lb
        # etc. are the implicit-lb duals), so those variables keep lb=0.
        p  = m.addVars(nG, T, lb=-GRB.INFINITY, name="p")
        f  = m.addVars(nL, T, lb=-GRB.INFINITY, name="f")
        th = m.addVars(nB, T, lb=-GRB.INFINITY, name="th")
        sh = m.addVars(nB, T, lb=0.0,           name="shed")
        cu = m.addVars(nR, T, lb=0.0,           name="curt")
        sp = m.addVars(nB, T, lb=0.0,           name="sp")
        sm = m.addVars(nB, T, lb=0.0,           name="sm")

        # Track constraints by key for dual extraction
        c_pmax, c_pmin = {}, {}
        c_rui, c_rdi   = {}, {}
        c_rup, c_rdn   = {}, {}
        c_fmax, c_fmin = {}, {}
        c_dcf, c_slk   = {}, {}
        c_bal          = {}
        c_shed_ub      = {}
        c_curt_ub      = {}
        c_sp_ub, c_sm_ub = {}, {}
        c_neutral      = {}

        # Gen pmax/pmin (with u fixed to u_j)
        for g, gen in enumerate(data.gens):
            Pmin, Pmax = float(gen.Pmin), float(gen.Pmax)
            for t in range(T):
                uj = float(u_j[g, t])
                c_pmax[(g, t)] = m.addConstr(p[g, t] <= Pmax * uj, name=f"pmax_{g}_{t}")
                c_pmin[(g, t)] = m.addConstr(p[g, t] >= Pmin * uj, name=f"pmin_{g}_{t}")

        # Ramps
        for g, gen in enumerate(data.gens):
            RU, RD = float(gen.RU), float(gen.RD)
            p0g = float(p_init[g])
            c_rui[g] = m.addConstr(p[g, 0] - p0g <= RU, name=f"rui_{g}")
            c_rdi[g] = m.addConstr(p0g - p[g, 0] <= RD, name=f"rdi_{g}")
            for t in range(1, T):
                c_rup[(g, t)] = m.addConstr(p[g, t] - p[g, t-1] <= RU, name=f"rup_{g}_{t}")
                c_rdn[(g, t)] = m.addConstr(p[g, t-1] - p[g, t] <= RD, name=f"rdn_{g}_{t}")

        # Curt/shed/shift upper bounds
        for r in range(nR):
            for t in range(T):
                c_curt_ub[(r, t)] = m.addConstr(
                    cu[r, t] <= float(data.rens[r].avail[t]), name=f"cub_{r}_{t}")
        for b in range(nB):
            for t in range(T):
                c_shed_ub[(b, t)] = m.addConstr(
                    sh[b, t] <= float(data.demand[b, t]), name=f"shub_{b}_{t}")
                c_sp_ub[(b, t)] = m.addConstr(
                    sp[b, t] <= float(data.Splus_max[b, t]), name=f"spub_{b}_{t}")
                c_sm_ub[(b, t)] = m.addConstr(
                    sm[b, t] <= float(data.Sminus_max[b, t]), name=f"smub_{b}_{t}")

        # Neutrality (equality, free dual)
        if per_bus_neutrality:
            for b in range(nB):
                for t0 in range(T - W + 1):
                    c_neutral[(b, t0)] = m.addConstr(
                        gp.quicksum(sp[b, t] - sm[b, t]
                                    for t in range(t0, t0 + W)) == 0,
                        name=f"neutral_{b}_{t0}")
        else:
            for t0 in range(T - W + 1):
                c_neutral[(0, t0)] = m.addConstr(
                    gp.quicksum(sp[b, t] - sm[b, t]
                                for b in range(nB)
                                for t in range(t0, t0 + W)) == 0,
                    name=f"neutral_g_{t0}")

        # DC flow + fmax/fmin (using b_hint)
        for ell, line in enumerate(data.lines):
            fr, to = int(line.fr), int(line.to)
            bl = float(line.b)
            cap = float(b_hint[ell])
            for t in range(T):
                c_dcf[(ell, t)] = m.addConstr(
                    f[ell, t] - bl * (th[fr, t] - th[to, t]) == 0,
                    name=f"dcf_{ell}_{t}")
                c_fmax[(ell, t)] = m.addConstr(f[ell, t] <= cap, name=f"fmax_{ell}_{t}")
                c_fmin[(ell, t)] = m.addConstr(-f[ell, t] <= cap, name=f"fmin_{ell}_{t}")

        # Slack bus
        for t in range(T):
            c_slk[t] = m.addConstr(th[slack_b, t] == 0.0, name=f"slk_{t}")

        # Nodal balance
        for b in range(nB):
            for t in range(T):
                gen_inj = gp.quicksum(p[g, t] for g in gens_at_bus[b])
                ren_inj = gp.quicksum(
                    float(data.rens[r].avail[t]) - cu[r, t]
                    for r in rens_at_bus[b])
                c_bal[(b, t)] = m.addConstr(
                    gen_inj + ren_inj + sh[b, t]
                    - float(data.demand[b, t]) - sp[b, t] + sm[b, t]
                    + gp.quicksum(f[ell, t] for ell in in_lines[b])
                    - gp.quicksum(f[ell, t] for ell in out_lines[b]) == 0,
                    name=f"bal_{b}_{t}")

        # Objective
        obj = gp.LinExpr()
        for g in range(nG):
            for t in range(T):
                obj.add(p[g, t], float(cvec[idx.p[g, t]]))
        for b in range(nB):
            for t in range(T):
                obj.add(sh[b, t], float(cvec[idx.shed[b, t]]))
                obj.add(sp[b, t], float(cvec[idx.splus[b, t]]))
                obj.add(sm[b, t], float(cvec[idx.sminus[b, t]]))
        for r in range(nR):
            for t in range(T):
                obj.add(cu[r, t], float(cvec[idx.curt[r, t]]))
        m.setObjective(obj, GRB.MINIMIZE)

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            if self.verbose:
                print(f"  [_solve_dispatch_lp] status={m.Status} — abort")
            m.dispose()
            return None

        # Extract primals + duals.
        # Sign convention: Gurobi Pi for "<= b" is <= 0 in min LPs; for ">= b" is >= 0; for "= b" free.
        # We want all non-neg duals to be >= 0 in our convention, matching the KKT block.
        def arr2(d, rows, cols):  # build (rows,cols) ndarray
            return np.array([[d[(i, j)] for j in range(cols)] for i in range(rows)])

        primals = {
            "p":   arr2({(g, t): p[g, t].X  for g in range(nG) for t in range(T)}, nG, T),
            "f":   arr2({(e, t): f[e, t].X  for e in range(nL) for t in range(T)}, nL, T),
            "th":  arr2({(b, t): th[b, t].X for b in range(nB) for t in range(T)}, nB, T),
            "sh":  arr2({(b, t): sh[b, t].X for b in range(nB) for t in range(T)}, nB, T),
            "cu":  arr2({(r, t): cu[r, t].X for r in range(nR) for t in range(T)}, nR, T),
            "sp":  arr2({(b, t): sp[b, t].X for b in range(nB) for t in range(T)}, nB, T),
            "sm":  arr2({(b, t): sm[b, t].X for b in range(nB) for t in range(T)}, nB, T),
        }
        # Free duals — SIGN FLIP from Gurobi convention to optimization-community
        # convention used by master's KKT block.  Verified via shed stationarity:
        #   master:  VOLL + gsh_ub - gsh_lb + pi = 0   (opt-community: pi = LMP)
        #   Gurobi:  c_bal.Pi for a min LP with `=` constraint at a load bus is
        #            negative (relaxing the constraint reduces cost by VOLL),
        #            so opt-community pi = -Gurobi Pi.
        # Same flip applies to nu (dcflow), sig (slack-bus), eta (neutrality).
        pi   = arr2({(b, t): -c_bal[(b, t)].Pi for b in range(nB) for t in range(T)}, nB, T)
        nu   = arr2({(e, t): -c_dcf[(e, t)].Pi for e in range(nL) for t in range(T)}, nL, T)
        sig  = np.array([-c_slk[t].Pi for t in range(T)])
        if per_bus_neutrality:
            eta  = {(b, t0): -c_neutral[(b, t0)].Pi
                    for b in range(nB) for t0 in range(T - W + 1)}
        else:
            eta  = {(0, t0): -c_neutral[(0, t0)].Pi for t0 in range(T - W + 1)}

        # Non-neg duals: flip sign for "<= b" constraints
        lam_hi = arr2({(g, t): max(0.0, -c_pmax[(g, t)].Pi)
                        for g in range(nG) for t in range(T)}, nG, T)
        lam_lo = arr2({(g, t): max(0.0,  c_pmin[(g, t)].Pi)
                        for g in range(nG) for t in range(T)}, nG, T)
        rho_ui = np.array([max(0.0, -c_rui[g].Pi) for g in range(nG)])
        rho_di = np.array([max(0.0, -c_rdi[g].Pi) for g in range(nG)])
        # rho_up/rho_dn only defined for t >= 1; column 0 left at 0.0
        rho_up = np.zeros((nG, T))
        rho_dn = np.zeros((nG, T))
        for g in range(nG):
            for t in range(1, T):
                rho_up[g, t] = max(0.0, -c_rup[(g, t)].Pi)
                rho_dn[g, t] = max(0.0, -c_rdn[(g, t)].Pi)

        mu_p   = arr2({(e, t): max(0.0, -c_fmax[(e, t)].Pi)
                        for e in range(nL) for t in range(T)}, nL, T)
        mu_m   = arr2({(e, t): max(0.0, -c_fmin[(e, t)].Pi)
                        for e in range(nL) for t in range(T)}, nL, T)
        gsh_ub = arr2({(b, t): max(0.0, -c_shed_ub[(b, t)].Pi)
                        for b in range(nB) for t in range(T)}, nB, T)
        gcu_ub = arr2({(r, t): max(0.0, -c_curt_ub[(r, t)].Pi)
                        for r in range(nR) for t in range(T)}, nR, T)
        gsp_ub = arr2({(b, t): max(0.0, -c_sp_ub[(b, t)].Pi)
                        for b in range(nB) for t in range(T)}, nB, T)
        gsm_ub = arr2({(b, t): max(0.0, -c_sm_ub[(b, t)].Pi)
                        for b in range(nB) for t in range(T)}, nB, T)

        # LB duals = variable reduced costs (≥ 0 in min LP)
        gsh_lb = arr2({(b, t): max(0.0, sh[b, t].RC)
                        for b in range(nB) for t in range(T)}, nB, T)
        gcu_lb = arr2({(r, t): max(0.0, cu[r, t].RC)
                        for r in range(nR) for t in range(T)}, nR, T)
        gsp_lb = arr2({(b, t): max(0.0, sp[b, t].RC)
                        for b in range(nB) for t in range(T)}, nB, T)
        gsm_lb = arr2({(b, t): max(0.0, sm[b, t].RC)
                        for b in range(nB) for t in range(T)}, nB, T)

        obj_val = float(m.ObjVal)
        m.dispose()

        return {
            **primals,
            "pi": pi, "nu": nu, "sig": sig, "eta": eta,
            "lam_hi": lam_hi, "lam_lo": lam_lo,
            "rho_up_i": rho_ui, "rho_dn_i": rho_di,
            "rho_up": rho_up, "rho_dn": rho_dn,
            "mu_p": mu_p, "mu_m": mu_m,
            "gsh_ub": gsh_ub, "gsh_lb": gsh_lb,
            "gcu_ub": gcu_ub, "gcu_lb": gcu_lb,
            "gsp_ub": gsp_ub, "gsp_lb": gsp_lb,
            "gsm_ub": gsm_ub, "gsm_lb": gsm_lb,
            "obj": obj_val,
        }

    def _analytic_warm_start(
        self,
        m: gp.Model,
        master_vars: Dict,
        b_hint: np.ndarray,
        foil_sol: Dict,
        patterns: List[np.ndarray],
        window_size: int,
        per_bus_neutrality: bool,
        u_init: np.ndarray,
        p_init: np.ndarray,
    ) -> bool:
        """Analytically compute MIP-start values for the main master m at b=b_hint.

        For each pattern u_j in `patterns`:
          1. Solve the dispatch LP at fixed (b_hint, u_j) to get primal x_j +
             dual values (UNIQUE up to LP degeneracy; LP duals satisfy KKT
             exactly).
          2. For each big-M complementarity pair (dual, slack), choose z based
             on the LP solution:
                z = 0  if slack ≤ tol (constraint binding)
                z = 1  if dual ≤ tol  (constraint slack)
                (degenerate both ≤ tol: pick z = 1, safer.)
          3. Set m.getVarByName(...).Start = computed value for every variable
             in the KKT block of iteration j.

        For the foil block: use values from foil_sol (the cached B&S
        oracle.solve_foil(b_hat) result) — pin u_foil/p_foil/f_foil/... etc.

        Bypasses Gurobi's MIP search entirely — we provide a complete
        integer-feasible point that satisfies every constraint by construction.

        Returns True iff all values were computed and injected.
        """
        TOL = 1e-6
        nG = len(self.data.gens)
        nR = len(self.data.rens)
        nB = int(self.data.nB)
        nL = len(self.data.lines)
        T  = int(self.data.T)
        W  = min(int(window_size), T)
        var_f = master_vars["var"]

        # --- 1) Foil block from cached oracle.solve_foil(b_hint) ----------
        if self.verbose:
            print(f"  [WS-ANALYTIC] Injecting foil block from oracle solution ...")
        u_f = np.round(np.asarray(foil_sol["u"], float)).astype(int)
        v_f, w_f = _compute_vw_from_u(u_f, np.asarray(u_init, float))
        p_f      = np.asarray(foil_sol["p"], float)
        f_f      = np.asarray(foil_sol["f"], float)
        th_f     = np.asarray(foil_sol["theta"], float)
        sh_f     = np.asarray(foil_sol["shed"], float)
        cu_f     = np.asarray(foil_sol["curt"], float)
        sp_f     = np.asarray(foil_sol["splus"], float)
        sm_f     = np.asarray(foil_sol["sminus"], float)

        for g in range(nG):
            for t in range(T):
                for key, arr in [("u", u_f), ("v", v_f), ("w", w_f), ("p", p_f)]:
                    try: var_f[key][g, t].Start = float(arr[g, t])
                    except Exception: pass
        for ell in range(nL):
            for t in range(T):
                try: var_f["f"][ell, t].Start = float(f_f[ell, t])
                except Exception: pass
        for b in range(nB):
            for t in range(T):
                for key, arr in [("theta", th_f), ("shed", sh_f),
                                  ("splus", sp_f), ("sminus", sm_f)]:
                    try: var_f[key][b, t].Start = float(arr[b, t])
                    except Exception: pass
        for r in range(nR):
            for t in range(T):
                try: var_f["curt"][r, t].Start = float(cu_f[r, t])
                except Exception: pass

        # b vars + L1 split
        for ell in self.free:
            try: master_vars["b"][ell].Start = float(b_hint[ell])
            except Exception: pass
            try:
                d = float(b_hint[ell]) - float(self.b0[ell])
                master_vars["bp"][ell].Start = max(0.0, d)
                master_vars["bm"][ell].Start = max(0.0, -d)
            except Exception: pass

        # --- 2) For each KKT block (pattern u_j), solve dispatch LP ------
        for j_idx, u_j in enumerate(patterns):
            if self.verbose:
                print(f"  [WS-ANALYTIC] Solving dispatch LP for pattern j={j_idx} "
                      f"(sum(u)={int(u_j.sum())}) ...")
            sol = self._solve_dispatch_lp(
                b_hint, u_j, window_size, per_bus_neutrality, p_init,
            )
            if sol is None:
                if self.verbose:
                    print(f"  [WS-ANALYTIC] dispatch LP failed for pattern j={j_idx}")
                return False

            s = f"_{j_idx}"

            # Inject KKT primal + dual values by name lookup
            def set_2d(name_prefix, arr):
                rows, cols = arr.shape
                for i in range(rows):
                    for t in range(cols):
                        v = m.getVarByName(f"{name_prefix}{s}[{i},{t}]")
                        if v is not None:
                            try: v.Start = float(arr[i, t])
                            except Exception: pass

            def set_1d(name_prefix, arr):
                for g in range(len(arr)):
                    v = m.getVarByName(f"{name_prefix}{s}[{g}]")
                    if v is not None:
                        try: v.Start = float(arr[g])
                        except Exception: pass

            # Primal vars
            set_2d("p",    sol["p"])
            set_2d("f",    sol["f"])
            set_2d("th",   sol["th"])
            set_2d("shed", sol["sh"])
            set_2d("curt", sol["cu"])
            set_2d("sp",   sol["sp"])
            set_2d("sm",   sol["sm"])
            # Free duals
            set_2d("pi",   sol["pi"])
            set_2d("nu",   sol["nu"])
            for t in range(T):
                v = m.getVarByName(f"sig{s}[{t}]")
                if v is not None:
                    try: v.Start = float(sol["sig"][t])
                    except Exception: pass
            # eta naming: per_bus → eta_j[b,t0]; global → eta_j[t0]
            for key, val in sol["eta"].items():
                if per_bus_neutrality:
                    b, t0 = key
                    v = m.getVarByName(f"eta{s}[{b},{t0}]")
                else:
                    _, t0 = key
                    v = m.getVarByName(f"eta{s}[{t0}]")
                if v is not None:
                    try: v.Start = float(val)
                    except Exception: pass
            # Non-neg duals
            set_2d("lhi", sol["lam_hi"])
            set_2d("llo", sol["lam_lo"])
            set_1d("rui", sol["rho_up_i"])
            set_1d("rdi", sol["rho_dn_i"])
            # rho_up/rho_dn are defined for t >= 1 only — addVars used range(1,T)
            for g in range(nG):
                for t in range(1, T):
                    for nm, arr in [("rup", sol["rho_up"]),
                                     ("rdn", sol["rho_dn"])]:
                        v = m.getVarByName(f"{nm}{s}[{g},{t}]")
                        if v is not None:
                            try: v.Start = float(arr[g, t])
                            except Exception: pass
            set_2d("mup",    sol["mu_p"])
            set_2d("mum",    sol["mu_m"])
            set_2d("gsh_ub", sol["gsh_ub"])
            set_2d("gsh_lb", sol["gsh_lb"])
            set_2d("gcu_ub", sol["gcu_ub"])
            set_2d("gcu_lb", sol["gcu_lb"])
            set_2d("gsp_ub", sol["gsp_ub"])
            set_2d("gsp_lb", sol["gsp_lb"])
            set_2d("gsm_ub", sol["gsm_ub"])
            set_2d("gsm_lb", sol["gsm_lb"])

            # --- 3) Pick z's based on LP slack / dual ---------------------
            # Convention: z = 0 if constraint binding (slack ≤ tol);
            #             z = 1 if constraint slack (dual ≤ tol).
            # Choose by "which side is smaller": if slack > dual ⇒ z=1; else z=0.
            # Degenerate (both small) ⇒ z=1 (favors dual=0).
            def pick_z(dual_val, slack_val):
                return 1 if slack_val > dual_val else 0

            def set_z(tag, z_val):
                v = m.getVarByName(f"zbm{s}_{tag}")
                if v is not None:
                    try: v.Start = float(z_val)
                    except Exception: pass

            # Walk through every _comp call in the SAME order as _add_iteration_block
            free_set = self._free_set
            b0   = self.b0
            # Flow limits
            for ell in range(nL):
                cap_val = float(b_hint[ell]) if ell in free_set else float(b0[ell])
                for t in range(T):
                    set_z(f"mup_{ell}_{t}",
                          pick_z(sol["mu_p"][ell, t], cap_val - sol["f"][ell, t]))
                    set_z(f"mum_{ell}_{t}",
                          pick_z(sol["mu_m"][ell, t], cap_val + sol["f"][ell, t]))
            # Generator bounds
            for g, gen in enumerate(self.data.gens):
                Pmin = float(gen.Pmin); Pmax = float(gen.Pmax)
                for t in range(T):
                    uj = float(u_j[g, t])
                    set_z(f"lhi_{g}_{t}",
                          pick_z(sol["lam_hi"][g, t], Pmax * uj - sol["p"][g, t]))
                    set_z(f"llo_{g}_{t}",
                          pick_z(sol["lam_lo"][g, t], sol["p"][g, t] - Pmin * uj))
            # Ramps
            for g, gen in enumerate(self.data.gens):
                RU = float(gen.RU); RD = float(gen.RD)
                p0g = float(p_init[g])
                set_z(f"rui_{g}",
                      pick_z(sol["rho_up_i"][g], p0g + RU - sol["p"][g, 0]))
                set_z(f"rdi_{g}",
                      pick_z(sol["rho_dn_i"][g], sol["p"][g, 0] - p0g + RD))
                for t in range(1, T):
                    set_z(f"rup_{g}_{t}",
                          pick_z(sol["rho_up"][g, t],
                                  sol["p"][g, t-1] + RU - sol["p"][g, t]))
                    set_z(f"rdn_{g}_{t}",
                          pick_z(sol["rho_dn"][g, t],
                                  sol["p"][g, t] + RD - sol["p"][g, t-1]))
            # Shed
            for b in range(nB):
                for t in range(T):
                    d = float(self.data.demand[b, t])
                    set_z(f"gshub_{b}_{t}",
                          pick_z(sol["gsh_ub"][b, t], d - sol["sh"][b, t]))
                    set_z(f"gshlb_{b}_{t}",
                          pick_z(sol["gsh_lb"][b, t], sol["sh"][b, t]))
            # Curt
            for r in range(nR):
                for t in range(T):
                    av = float(self.data.rens[r].avail[t])
                    set_z(f"gcuub_{r}_{t}",
                          pick_z(sol["gcu_ub"][r, t], av - sol["cu"][r, t]))
                    set_z(f"gculb_{r}_{t}",
                          pick_z(sol["gcu_lb"][r, t], sol["cu"][r, t]))
            # Shift
            for b in range(nB):
                for t in range(T):
                    spm = float(self.data.Splus_max[b, t])
                    smm = float(self.data.Sminus_max[b, t])
                    set_z(f"gspub_{b}_{t}",
                          pick_z(sol["gsp_ub"][b, t], spm - sol["sp"][b, t]))
                    set_z(f"gsplb_{b}_{t}",
                          pick_z(sol["gsp_lb"][b, t], sol["sp"][b, t]))
                    set_z(f"gsmub_{b}_{t}",
                          pick_z(sol["gsm_ub"][b, t], smm - sol["sm"][b, t]))
                    set_z(f"gsmlb_{b}_{t}",
                          pick_z(sol["gsm_lb"][b, t], sol["sm"][b, t]))

        m.update()
        if self.verbose:
            print(f"  [WS-ANALYTIC] Done. {len(patterns)} KKT block(s) populated "
                  f"with LP-optimal primal/dual + analytic z's.")
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        window_size: int,
        per_bus_neutrality: bool,
        u_init: np.ndarray,
        p_init: np.ndarray,
        on_time_init: np.ndarray,
        off_time_init: np.ndarray,
    ) -> Dict:

        self._keepalive.start()
        master_LB: float = 0.0
        seen: Set[tuple] = set()
        patterns: List[np.ndarray] = []   # u^j patterns in iteration order
        j = 0

        # Tolerant CE check: vd ≤ vp + max(eps_weak, eps_rel * |vp|).  The
        # absolute-only check (eps_weak = 1e-3) is too tight when vp is in the
        # 10^5 – 10^6 range — Gurobi MIP-solve tolerances easily produce
        # vd - vp ≈ 0.01% × vp ≈ 100, which exceeds 1e-3 but is still a CE
        # for any practical purpose.  Switch to relative-or-absolute tolerance.
        EPS_REL_CE = 1e-4
        def _ce_ok(vd, vp):
            if vd is None or vp is None:
                return False
            tol = max(self.eps_weak, EPS_REL_CE * abs(vp))
            return vd <= vp + tol

        # Track the master's last incumbent (whether or not it certifies as a CE).
        # Useful for diagnostics when the algorithm stops without finding a
        # better certified CE than the input hint (e.g., cycle detection).
        candidate: Optional[Dict] = None
        termination_reason: str = "max_iter"   # default, overridden on break

        try:
            # Warm-start: check b0
            if self.verbose:
                print("[DECOMP] Warm-start at b0 ...")
            vp, _, sp = self.oracle.solve_plain(self.b0)
            vd, _, sd = self.oracle.solve_foil(self.b0)
            if _ce_ok(vd, vp):
                self._update_incumbent(self.b0, self.F(self.b0), vp, vd)

            # Build base master
            if self.verbose:
                print("[DECOMP] Building master ...")
            m, master_vars = self._build_master_base(
                window_size, per_bus_neutrality,
                u_init, p_init, on_time_init, off_time_init,
            )

            # MIP warm start: solve foil at bU (maximum expansion) to get
            # a feasible incumbent immediately, avoiding 300s+ cold-start struggle.
            b_ws = self.b0.copy()
            b_ws[self.free] = self.bU[self.free]
            vd_ws, _, sd_ws = self.oracle.solve_foil(b_ws)
            if vd_ws is not None and sd_ws is not None:
                self._inject_mip_start(m, master_vars, b_ws, sd_ws, u_init)
                if self.verbose:
                    print(f"[DECOMP] MIP warm start: foil at bU feasible "
                          f"(F_ws={self.F(b_ws):.4f}  v_foil={vd_ws:.4f})")
            else:
                if self.verbose:
                    print("[DECOMP] Foil at bU infeasible — no MIP warm start.")

            # If a known-good CE hint is provided, also solve foil there.
            # This warm start is re-injected after every KKT block so Gurobi
            # always has a feasible incumbent even after the optimality cuts tighten
            # the feasible region (at bU the cut is typically violated).
            _b_hint_sol: Optional[Dict] = None
            if self.b_hat_hint is not None:
                vd_hint, _, sd_hint = self.oracle.solve_foil(self.b_hat_hint)
                if vd_hint is not None and sd_hint is not None:
                    _b_hint_sol = sd_hint
                    self._inject_mip_start(m, master_vars, self.b_hat_hint, sd_hint, u_init)
                    if self.verbose:
                        print(f"[DECOMP] b_hat hint: foil feasible "
                              f"(F_hint={self.F(self.b_hat_hint):.4f}  v_foil={vd_hint:.4f})")

                    # GUARANTEE: register b_hat_hint as the initial incumbent.
                    # The caller passed b_hat_hint asserting it IS a CE (e.g., from
                    # B&S).  We trust that and initialize best_F = F(b_hat) so the
                    # algorithm never reports "no CE" — at worst it returns the
                    # input CE unchanged.  The oracle re-check is informational
                    # only and used to populate v_plain/v_foil for the result.
                    vp_hint, _, _ = self.oracle.solve_plain(self.b_hat_hint)
                    self._update_incumbent(
                        self.b_hat_hint, self.F(self.b_hat_hint),
                        vp_hint if vp_hint is not None else float("nan"),
                        vd_hint,
                    )
                    if self.verbose:
                        ce_gap = (vd_hint - vp_hint) if vp_hint is not None else None
                        print(f"[DECOMP] Initialized incumbent at b_hat_hint  "
                              f"(F={self.F(self.b_hat_hint):.4f}, "
                              f"v_plain={vp_hint}, v_foil={vd_hint:.4f}, "
                              f"oracle CE gap={ce_gap})")
                else:
                    if self.verbose:
                        print("[DECOMP] b_hat hint: foil infeasible — skipping hint warm start.")

            # If a valid hint CE is available, add an explicit objective upper bound.
            # The master LP warm start violates SOS1 stationarity equalities so
            # Gurobi never accepts it as an incumbent — meaning it has no upper bound
            # and explores the entire feasible region.  Adding F ≤ F_hint as a hard
            # constraint has the same pruning effect as having an incumbent at F_hint,
            # without needing a complete feasible integer start.
            _hint_ub_constr = None
            if _b_hint_sol is not None:
                F_ub = self.F(self.b_hat_hint)
                _hint_ub_constr = m.addConstr(
                    gp.quicksum(
                        self.w[ell] * (master_vars["bp"][ell] + master_vars["bm"][ell])
                        for ell in self.free
                    ) <= F_ub,
                    name="_hint_F_ub",
                )
                m.update()
                if self.verbose:
                    print(f"[DECOMP] Added hint upper bound: F ≤ {F_ub:.4f}")

            for k in range(self.max_iter):
                if self.verbose:
                    nv, nc, ni = m.NumVars, m.NumConstrs, m.NumIntVars
                    print(f"\n[DECOMP] Iter {k+1}/{self.max_iter}  "
                          f"model: {nv} vars ({ni} int), {nc} constrs")

                m.Params.OutputFlag   = self.master_output_flag
                m.Params.MIPGap       = self.master_mip_gap
                m.Params.MIPFocus     = 1   # prioritise finding feasible solutions
                m.Params.NumericFocus = 2   # extra care with near-degenerate numerics
                if self.master_time_limit is not None:
                    m.Params.TimeLimit = self.master_time_limit

                _optimize_with_retry(m)

                if m.Status == GRB.INFEASIBLE:
                    if self.verbose:
                        print("  [DECOMP] Master infeasible — no CE exists.")
                    iis_path = f"decomp_master_infeas_iter{k+1}.ilp"
                    try:
                        m.computeIIS()
                        m.write(iis_path)
                        if self.verbose:
                            print(f"  [DECOMP] IIS written to {iis_path}")
                    except Exception as _e:
                        if self.verbose:
                            print(f"  [DECOMP] IIS failed: {_e}")
                    termination_reason = "master_infeasible"
                    break

                # Accept TIME_LIMIT if Gurobi found at least one feasible solution
                tl_hit = (m.Status == GRB.TIME_LIMIT)
                if tl_hit and m.SolCount == 0:
                    if self.verbose:
                        print("  [DECOMP] Time limit hit with no feasible master solution — stopping.")
                    termination_reason = "time_limit_no_incumbent"
                    break

                if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and not tl_hit:
                    if self.verbose:
                        print(f"  [DECOMP] Master status={m.Status}, stopping.")
                    termination_reason = f"master_status_{m.Status}"
                    break

                # Use ObjBound as LB (tighter than ObjVal when time limit hit)
                try:
                    master_LB = float(m.ObjBound)
                except Exception:
                    master_LB = float(m.ObjVal) if not tl_hit else 0.0
                if tl_hit and self.verbose:
                    print(f"  [DECOMP] Time limit hit — using best found solution (gap={m.MIPGap:.2%})")
                nL = len(self.data.lines)
                b_k = np.array([
                    float(master_vars["b"][ell].X) if ell in self._free_set
                    else float(self.b0[ell])
                    for ell in range(nL)
                ])

                if self.verbose:
                    print(f"  master_LB={master_LB:.4f}  "
                          f"Δb={np.sum(np.abs(b_k - self.b0)):.4f}")

                # SP1: plain UC at b_k — must be feasible for the algorithm to proceed
                vp, _, sol_p = self.oracle.solve_plain(b_k)
                if vp is None:
                    if self.verbose:
                        print("  Plain UC infeasible at b_k — stopping.")
                    termination_reason = "sp1_infeasible"
                    break

                # SP2: foil UC at b_k — infeasible here just means b_k is not a CE;
                # we still add the cut for u_k below
                vd, _, sol_d = self.oracle.solve_foil(b_k)

                # Always record b_k as candidate (for the return dict, even if
                # it doesn't certify as a CE).
                Fk = self.F(b_k)
                candidate = {
                    "b":             b_k.copy(),
                    "F":             Fk,
                    "v_plain":       vp,
                    "v_foil":        vd,
                    "oracle_ce_gap": (vd - vp) if (vd is not None and vp is not None) else None,
                    "iter":          k + 1,
                }

                # CE check (with relative+absolute tolerance, see _ce_ok above)
                if _ce_ok(vd, vp):
                    is_new_best = Fk < self.best_F - 1e-9
                    self._update_incumbent(b_k, Fk, vp, vd)
                    if is_new_best and sol_d is not None:
                        # Refresh the warm-start hint to the new best CE.  All
                        # subsequent _analytic_warm_start calls will use this
                        # (b, foil_sol) instead of the original B&S CE, so the
                        # incumbent injected into Iter k+1 is at F(new best)
                        # rather than F(b_hat_BS) — a strictly tighter starting
                        # upper bound (F(new best) ≤ F(b_hat_BS) by definition
                        # of "new best").
                        self.b_hat_hint = b_k.copy()
                        _b_hint_sol = sol_d
                        if self.verbose:
                            print(f"  [HINT] Refreshed warm-start hint to "
                                  f"new best CE (F={Fk:.4f})")
                    if _hint_ub_constr is not None and self.best_F < float("inf"):
                        _hint_ub_constr.RHS = self.best_F
                        if self.verbose:
                            print(f"  [UB] Tightened upper bound: F ≤ {self.best_F:.4f}")

                # Convergence
                gap = self.best_F - master_LB
                if self.verbose:
                    print(f"  best_F={self.best_F:.4f}  gap={gap:.4f}")

                if gap <= self.eps_obj and self.best_F < float("inf"):
                    if self.verbose:
                        print("  [DECOMP] Certified optimal.")
                    termination_reason = "certified_optimal"
                    break

                # Cycle detection
                if sol_p is None:
                    termination_reason = "sp1_infeasible"
                    break
                u_k = np.round(sol_p["u"]).astype(int)
                pk  = self._pattern_key(u_k)
                if pk in seen:
                    if self.verbose:
                        print("  [DECOMP] Cycle detected, stopping.")
                    termination_reason = "cycle"
                    break
                seen.add(pk)

                # Add KKT block (record pattern for temp-model warm starts)
                if self.verbose:
                    print(f"  Adding KKT block {j} ...")
                self._add_iteration_block(
                    m, master_vars, j, u_k, u_init, p_init,
                    window_size, per_bus_neutrality,
                )
                patterns.append(u_k)
                j += 1
                if self.verbose:
                    print(f"  After KKT block {j-1}: {m.NumVars} vars "
                          f"({m.NumIntVars} int), {m.NumConstrs} constrs")

                # Re-inject the CE hint as warm start after every KKT block.
                # _inject_mip_start sets foil/b vars (fast, no solve).
                # _full_integer_warm_start builds a fresh temp master mimicking
                # debug_fix_b's phased build (so Gurobi solves it in seconds),
                # solves to integer-feasible at F(b_hint), then transfers values
                # by name into the main master m as MIP start.  Section 4a showed
                # the cut is nearly binding at b_hat (cut_slack ≈ 1e-11), so
                # without this complete start B&B cannot navigate to the
                # feasible region (see DECOMP_state.md).
                if _b_hint_sol is not None:
                    self._inject_mip_start(m, master_vars, self.b_hat_hint,
                                           _b_hint_sol, u_init)
                    self._full_integer_warm_start(
                        m, master_vars,
                        self.b_hat_hint, _b_hint_sol, patterns,
                        window_size, per_bus_neutrality,
                        u_init, p_init, on_time_init, off_time_init,
                    )

                # LP relaxation feasibility check (fast, seconds, no binaries)
                # Diagnoses whether the combined master is infeasible at the LP level.
                if self.verbose:
                    _m_lp = m.relax()
                    _m_lp.Params.OutputFlag = 0
                    _m_lp.Params.TimeLimit  = 120.0
                    _m_lp.optimize()
                    _lp_s = _m_lp.Status
                    if _lp_s == GRB.INFEASIBLE:
                        print(f"  [LP-relax] INFEASIBLE after KKT block {j-1} — formulation bug!")
                        _lp_iis = (
                            str(self.checkpoint_path).replace(".json", "")
                            + f"_lp_infeas_iter{j}.ilp"
                        ) if self.checkpoint_path else f"decomp_lp_infeas_iter{j}.ilp"
                        try:
                            _m_lp.computeIIS()
                            _m_lp.write(_lp_iis)
                            print(f"  [LP-relax] IIS written to {_lp_iis}")
                        except Exception as _e:
                            print(f"  [LP-relax] IIS write failed: {_e}")
                        _m_lp.dispose()
                        break
                    elif _lp_s in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                        print(f"  [LP-relax] feasible  obj={_m_lp.ObjVal:.4f}")
                    else:
                        print(f"  [LP-relax] status={_lp_s} (unbounded or numeric error)")
                    _m_lp.dispose()

                if (k + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(k + 1, master_LB)

            m.dispose()

        finally:
            self._keepalive.stop()

        gap = self.best_F - master_LB
        return {
            # Best certified CE.  Guaranteed populated iff b_hat_hint was given
            # AND its foil oracle returned a solution (we trust the input was a
            # CE; see "GUARANTEE" block in run() setup).
            "success":       self.best_F < float("inf"),
            "b_hat":         self.best_b,
            "F_opt":         self.best_F,
            "iterations":    j,
            "v_plain":       self.best_v_plain,
            "v_D":           self.best_v_foil,
            "master_LB":     master_LB,
            "gap":           gap,
            "gap_pct":       (gap / max(self.best_F, 1e-9) * 100
                              if self.best_F < float("inf") else float("inf")),
            "certified":     gap <= self.eps_obj and self.best_F < float("inf"),
            "seen_patterns": len(seen),
            # Master's last incumbent — may or may not certify as a CE per the
            # oracle.  Useful when the algorithm stops via cycle detection at
            # a b_k that satisfies the cuts but the oracle's vd > vp + tol.
            # If `candidate["F"] < F_opt` and `oracle_ce_gap` is small, the
            # candidate is plausibly a CE — investigate by hand.
            "candidate":     candidate,
            # Why the loop exited: "certified_optimal" | "cycle" |
            # "time_limit_no_incumbent" | "master_infeasible" | "sp1_infeasible"
            # | "master_status_<N>" | "max_iter".
            "termination_reason": termination_reason,
        }
