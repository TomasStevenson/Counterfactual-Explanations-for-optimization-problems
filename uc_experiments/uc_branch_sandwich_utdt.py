"""
uc_branch_sandwich_utdt.py
──────────────────────────────────────────────────────────────────────────────
Branch-and-Sandwich WCE solver for MUTABLE MIN UP/DOWN TIMES.

Bilevel formulation
───────────────────
  outer:  min  Σ_g (UT_orig_g − UT_g) + (DT_orig_g − DT_g)
          s.t. UT_g ∈ {1, …, UT_orig_g}   for g in free_gens
               DT_g ∈ {1, …, DT_orig_g}   for g in free_gens

  inner:  min  c^T z
          s.t. UC constraints with parameters (UT_g, DT_g)
               Σ curt ≤ (1−α) C_0              ← foil constraint

The outer decision variable is the integer vector (UT, DT).
The outer objective measures total flexibility gained (hours).
A configuration (UT, DT) is a valid WCE if the inner UC with the
foil embedded is FEASIBLE.

Node representation
───────────────────
Each node is a box on the integer lattice:
    UTL_g ≤ UT_g ≤ UTU_g,   DTL_g ≤ DT_g ≤ DTU_g

Outer lower bound (OLB) — exact, no LP needed
──────────────────────────────────────────────
OLB(node) = Σ_g (UT_orig_g − UTU_g) + (DT_orig_g − DTU_g)

This is the minimum objective achievable anywhere in the box
(achieved at the LEAST-flexible corner UT=UTU, DT=DTU).
It is exact because the objective is separable and linear in UT/DT.

Inner bounds
────────────
inner_lb : LP-relaxed UC at (UTU, DTU) — fewest constraints, cheapest
inner_ub : integer UC at (UTL, DTL) — most constraints, most expensive
           used as sandwich cut in OUB solves

OUB (incumbent search)
──────────────────────
For each node, evaluate the MOST-flexible corner (UTL, DTL) with foil.
If feasible, that corner is a valid WCE with objective = OLB(node).
This is best-first: we always try the corner with the lowest OLB first.

Branching
─────────
Split on the free generator with the largest remaining box width
(measured as max(UTU−UTL, DTU−DTL)). Bisect at the midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
from typing import List, Optional, Dict, Tuple
import numpy as np
import heapq
import gurobipy as gp
from gurobipy import GRB

from uc_pipeline import (
    NetworkUCData, IndexMap,
    solve_uc_with_cost_4b,
)
from uc_master_relax_4b import build_uc_relax_master_varfmax_4b


# ─────────────────────────────────────────────────────────────────────────────
# Node dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UTDTNode:
    id: int
    utL: np.ndarray   # lower bound on UT per generator (length nG)
    utU: np.ndarray   # upper bound on UT per generator
    dtL: np.ndarray   # lower bound on DT per generator
    dtU: np.ndarray   # upper bound on DT per generator

    olb: float = 0.0
    inner_lb: float = -np.inf
    inner_ub: float = np.inf
    parent_inner_ub: float = np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Helper: replace UT/DT in a NetworkUCData
# ─────────────────────────────────────────────────────────────────────────────

def _replace_utdt(data: NetworkUCData,
                  ut_vec: np.ndarray,
                  dt_vec: np.ndarray) -> NetworkUCData:
    new_gens = [
        dc_replace(gen, UT=int(ut_vec[g]), DT=int(dt_vec[g]))
        for g, gen in enumerate(data.gens)
    ]
    return dc_replace(data, gens=new_gens)


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

class UCBranchAndSandwichUTDT:
    """
    Branch-and-Sandwich WCE for mutable min up/down times.

    Parameters
    ----------
    data              : NetworkUCData baseline grid
    idx               : IndexMap
    cvec              : cost vector aligned to idx
    foil_extra_fn     : callable(m, var) adding the curtailment foil
    ut_orig           : np.ndarray(nG,) original UT values
    dt_orig           : np.ndarray(nG,) original DT values
    free_gens         : list of generator indices eligible for UT/DT change
    eps_obj           : outer optimality tolerance (hours, default 0.5)
    max_nodes         : node budget
    master_time_limit : per-solve Gurobi time limit (s)
    output_flag       : 0=silent Gurobi
    verbose           : print B&S progress
    """

    def __init__(
        self,
        data: NetworkUCData,
        idx: IndexMap,
        cvec: np.ndarray,
        foil_extra_fn,
        ut_orig: np.ndarray,
        dt_orig: np.ndarray,
        free_gens: List[int],
        eps_obj: float = 0.5,
        max_nodes: int = 300,
        master_time_limit: Optional[float] = None,
        output_flag: int = 0,
        verbose: bool = True,
        plain_metric_fn=None,
        foil_target: Optional[float] = None,
    ):
        self.data    = data
        self.idx     = idx
        self.cvec    = np.asarray(cvec, dtype=float)
        self.foil_fn = foil_extra_fn

        self.ut_orig = np.asarray(ut_orig, dtype=int)
        self.dt_orig = np.asarray(dt_orig, dtype=int)
        self.free    = list(free_gens)
        self.nG      = len(data.gens)

        self.eps_obj           = float(eps_obj)
        self.max_nodes         = int(max_nodes)
        self.master_time_limit = master_time_limit
        self.output_flag       = int(output_flag)
        self.verbose           = bool(verbose)

        # Incumbents
        self.best_F        = float("inf")
        self.best_ut       = None
        self.best_dt       = None
        self.best_sol      = None
        self.best_inner_ub = float("inf")

        self._node_id = 0
        self._history: List[Dict] = []

        # Metric for plain-UC feasibility check.
        # plain_metric_fn(sol) -> float  measures what the foil targets.
        # foil_target float  the threshold (foil satisfied when metric <= target).
        # If not provided, probes the foil model to extract curtailment RHS.
        if plain_metric_fn is not None and foil_target is not None:
            self._plain_metric = plain_metric_fn
            self._curt_target  = float(foil_target)
        else:
            self._plain_metric = lambda sol: float(np.sum(sol["curt"]))
            self._curt_target  = self._extract_curt_target()

    def _extract_curt_target(self) -> float:
        """
        Extract the curtailment RHS from the foil function by building a
        tiny dummy model and reading the constraint RHS.
        Falls back to inf (never satisfied) if extraction fails.
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
            m_dummy = gp.Model("probe")
            m_dummy.Params.OutputFlag = 0
            nR = self.idx.curt.shape[0]
            T  = self.idx.curt.shape[1]
            # Add dummy curt variables
            curt_d = m_dummy.addVars(nR, T, lb=0.0, name="curt")
            var_d  = {"curt": curt_d}
            self.foil_fn(m_dummy, var_d)
            m_dummy.update()
            for c in m_dummy.getConstrs():
                if "foil" in c.ConstrName.lower() or "curt" in c.ConstrName.lower():
                    return float(c.RHS)
            # If no named match, take the first constraint RHS
            constrs = m_dummy.getConstrs()
            if constrs:
                return float(constrs[0].RHS)
        except Exception:
            pass
        return float("inf")

    # ── Outer objective ───────────────────────────────────────────────────

    def F(self, ut_vec: np.ndarray, dt_vec: np.ndarray) -> float:
        return float(np.sum(
            (self.ut_orig[self.free] - ut_vec[self.free]) +
            (self.dt_orig[self.free] - dt_vec[self.free])
        ))

    # ── Exact OLB from box ────────────────────────────────────────────────

    def _olb(self, node: UTDTNode) -> float:
        return float(np.sum(
            (self.ut_orig[self.free] - node.utU[self.free]) +
            (self.dt_orig[self.free] - node.dtU[self.free])
        ))

    # ── Inner LB: LP-relaxed UC at least-constrained corner (UTU, DTU) ───

    def _compute_inner_lb(self, node, ws, pbn, u_init, p_init, on_t, off_t):
        data_mod = _replace_utdt(self.data, node.utU, node.dtU)
        fmax = np.array([L.fmax for L in self.data.lines])
        m, _, _ = build_uc_relax_master_varfmax_4b(
            data=data_mod, idx=self.idx, cvec=self.cvec,
            window_size=ws, per_bus_neutrality=pbn,
            u_init=u_init, p_init=p_init,
            on_time_init=on_t, off_time_init=off_t,
            b0=fmax, node_bL=fmax, node_bU=fmax,
            b_free_idx=[],
            foil_extra_constr_fn=None,
            cost_ub=None,
            output_flag=self.output_flag,
        )
        if self.master_time_limit:
            m.Params.TimeLimit = self.master_time_limit
        m.optimize()
        if m.SolCount == 0:
            return float("inf")
        return float(m.ObjVal)

    # ── Inner UB: integer UC at most-constrained corner (UTL, DTL) ────────

    def _compute_inner_ub(self, node, ws, pbn, u_init, p_init, on_t, off_t):
        data_mod = _replace_utdt(self.data, node.utL, node.dtL)
        _, sol, _ = solve_uc_with_cost_4b(
            data=data_mod, idx=self.idx, cvec=self.cvec,
            window_size=ws, per_bus_neutrality=pbn,
            u_init=u_init, p_init=p_init,
            on_time_init=on_t, off_time_init=off_t,
            extra_constr_fn=None,
            output_flag=self.output_flag,
            time_limit=self.master_time_limit,
        )
        if sol is None:
            return float("inf")
        return float(sol["obj"])

    # ── OUB: solve foil UC at explicit (ut, dt) point ────────────────────

    def _check_oub_at(self, ut_vec, dt_vec, ws, pbn, u_init, p_init, on_t, off_t,
                      cut_ub=None):
        """
        Solve integer UC + foil at (ut_vec, dt_vec).
        Optionally apply sandwich cut: cost <= cut_ub.
        Returns (feasible: bool, sol: dict or None).
        """
        data_mod = _replace_utdt(self.data, ut_vec, dt_vec)

        foil_fn = self.foil_fn
        if cut_ub is not None and np.isfinite(cut_ub):
            def foil_with_cut(m, var):
                foil_fn(m, var)
                from uc_master_relax_4b import build_uc_operating_cost_expr_4b
                cost_expr = build_uc_operating_cost_expr_4b(var, self.idx, self.cvec)
                m.addConstr(cost_expr <= float(cut_ub), name="sandwich_cut")
            extra = foil_with_cut
        else:
            extra = foil_fn

        _, sol, _ = solve_uc_with_cost_4b(
            data=data_mod, idx=self.idx, cvec=self.cvec,
            window_size=ws, per_bus_neutrality=pbn,
            u_init=u_init, p_init=p_init,
            on_time_init=on_t, off_time_init=off_t,
            extra_constr_fn=extra,
            output_flag=self.output_flag,
            time_limit=self.master_time_limit,
        )
        return (sol is not None), sol

    # ── Plain UC check: no foil, just measure curtailment ─────────────────

    def _check_plain_at(self, ut_vec, dt_vec, ws, pbn,
                        u_init, p_init, on_t, off_t):
        """
        Solve plain UC (NO foil) at (ut_vec, dt_vec).
        Returns (curt_mwh, sol). curt_mwh=inf if infeasible.

        This is the correct feasibility check: does cost-minimising
        dispatch NATURALLY produce curtailment <= target?
        """
        data_mod = _replace_utdt(self.data, ut_vec, dt_vec)
        _, sol, _ = solve_uc_with_cost_4b(
            data=data_mod, idx=self.idx, cvec=self.cvec,
            window_size=ws, per_bus_neutrality=pbn,
            u_init=u_init, p_init=p_init,
            on_time_init=on_t, off_time_init=off_t,
            extra_constr_fn=None,   # ← NO foil
            output_flag=self.output_flag,
            time_limit=self.master_time_limit,
        )
        if sol is None:
            return float("inf"), None
        return self._plain_metric(sol), sol

    def _update_incumbent(self, ut_vec, dt_vec, obj, sol, label=""):
        if obj < self.best_F - self.eps_obj:
            self.best_F   = obj
            self.best_ut  = ut_vec.copy()
            self.best_dt  = dt_vec.copy()
            self.best_sol = sol
            if self.verbose:
                chg = [(g, int(self.ut_orig[g]), int(ut_vec[g]),
                           int(self.dt_orig[g]), int(dt_vec[g]))
                       for g in self.free
                       if ut_vec[g] != self.ut_orig[g] or dt_vec[g] != self.dt_orig[g]]
                print(f"  [INC] {label}: F={obj:.1f}h")
                for g, uo, uh, do, dh in chg:
                    print(f"    G{g}: UT {uo}→{uh}  DT {do}→{dh}")
        if sol is not None and float(sol["obj"]) < self.best_inner_ub:
            self.best_inner_ub = float(sol["obj"])
            if self.verbose:
                print(f"  [INNER_UB] {self.best_inner_ub:.2f}")

    # ── Branching ─────────────────────────────────────────────────────────

    def _branch(self, node: UTDTNode):
        best_score, best_g, best_dim = -1.0, None, None

        for g in self.free:
            w_ut = node.utU[g] - node.utL[g]
            w_dt = node.dtU[g] - node.dtL[g]
            score = max(w_ut, w_dt)
            if score > best_score:
                best_score = score
                best_g     = g
                best_dim   = "UT" if w_ut >= w_dt else "DT"

        if best_g is None or best_score < 1:
            return None, None

        if best_dim == "UT":
            mid = int(np.floor((node.utL[best_g] + node.utU[best_g]) / 2))
            utL1, utU1 = node.utL.copy(), node.utU.copy(); utU1[best_g] = mid
            utL2, utU2 = node.utL.copy(), node.utU.copy(); utL2[best_g] = mid + 1
            n1 = UTDTNode(self._node_id+1, utL1, utU1,
                          node.dtL.copy(), node.dtU.copy(),
                          parent_inner_ub=node.inner_ub)
            n2 = UTDTNode(self._node_id+2, utL2, utU2,
                          node.dtL.copy(), node.dtU.copy(),
                          parent_inner_ub=node.inner_ub)
        else:
            mid = int(np.floor((node.dtL[best_g] + node.dtU[best_g]) / 2))
            dtL1, dtU1 = node.dtL.copy(), node.dtU.copy(); dtU1[best_g] = mid
            dtL2, dtU2 = node.dtL.copy(), node.dtU.copy(); dtL2[best_g] = mid + 1
            n1 = UTDTNode(self._node_id+1, node.utL.copy(), node.utU.copy(),
                          dtL1, dtU1, parent_inner_ub=node.inner_ub)
            n2 = UTDTNode(self._node_id+2, node.utL.copy(), node.utU.copy(),
                          dtL2, dtU2, parent_inner_ub=node.inner_ub)

        self._node_id += 2
        return n1, n2

    # ── Node initialisation ───────────────────────────────────────────────

    def _init_node(self, node, ws, pbn, u_init, p_init, on_t, off_t):
        node.olb = self._olb(node)

        node.inner_lb = self._compute_inner_lb(
            node, ws, pbn, u_init, p_init, on_t, off_t)
        node.inner_ub = self._compute_inner_ub(
            node, ws, pbn, u_init, p_init, on_t, off_t)

        if node.inner_ub < self.best_inner_ub:
            self.best_inner_ub = node.inner_ub
            if self.verbose:
                print(f"  [INNER_UB] init node={node.id}: {self.best_inner_ub:.2f}")

        # Inner fathom: LP cost already exceeds sandwich cut
        if node.inner_lb > self.best_inner_ub + self.eps_obj:
            if self.verbose:
                print(f"  [FATHOM-INNER] node={node.id}")
            node.olb = float("inf")

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self, u_init, p_init, on_t, off_t,
            window_size: Optional[int] = None,
            per_bus_neutrality: bool = True) -> Dict:

        self._node_id  = 0
        self._history  = []
        ws  = int(self.data.T) if window_size is None else int(window_size)
        pbn = per_bus_neutrality

        # ── Sanity checks ────────────────────────────────────────────────
        if self.verbose:
            print(f"[UTDT-BS] Sanity checks...")
            print(f"  Foil target: {self._curt_target:.4f}")
            print(f"  Free generators: {self.free}")
            for g in self.free:
                print(f"    G{g}: UT_orig={self.ut_orig[g]}  DT_orig={self.dt_orig[g]}")

        # Check 1: target must be finite and positive
        if not np.isfinite(self._curt_target) or self._curt_target <= 0:
            raise ValueError(
                f"[UTDT-BS] Could not extract foil target "
                f"(got {self._curt_target}). Pass foil_target explicitly.")

        # Check 2: plain UC at original UT/DT must exceed the target
        curt_orig_check, _ = self._check_plain_at(
            self.ut_orig, self.dt_orig, ws, pbn, u_init, p_init, on_t, off_t)
        if self.verbose:
            print(f"  Plain UC at orig UT/DT: metric={curt_orig_check:.4f}  "
                  f"target={self._curt_target:.4f}  "
                  f"need_change={curt_orig_check > self._curt_target + 1e-3}")
        if curt_orig_check <= self._curt_target + 1e-3:
            if self.verbose:
                print(f"[UTDT-BS] WARNING: original UT/DT already achieves "
                      f"metric={curt_orig_check:.4f} <= target={self._curt_target:.4f}. "
                      f"No search needed — returning F=0.")
            return {"success": True, "F_opt": 0.0,
                    "ut_hat": self.ut_orig.copy(), "dt_hat": self.dt_orig.copy(),
                    "sol": None, "global_LB": 0.0, "gap": 0.0,
                    "nodes": 0, "certified": True, "history": [],
                    "already_satisfied": True}

        # Check 3: most-flexible config (all UT=DT=1) must achieve the target
        utL_test = self.ut_orig.copy(); dtL_test = self.dt_orig.copy()
        for g in self.free:
            utL_test[g] = 1; dtL_test[g] = 1
        curt_flex, _ = self._check_plain_at(
            utL_test, dtL_test, ws, pbn, u_init, p_init, on_t, off_t)
        if self.verbose:
            print(f"  Plain UC at all-flexible UT=DT=1: metric={curt_flex:.4f}")
        if curt_flex > self._curt_target + 1e-3:
            if self.verbose:
                print(f"[UTDT-BS] WARNING: even UT=DT=1 gives metric={curt_flex:.4f} "
                      f"> target={self._curt_target:.4f}. "
                      f"Foil may be infeasible for this grid/alpha.")
        # ── End sanity checks ────────────────────────────────────────────

        # Root box: [1, ut_orig] x [1, dt_orig] for free gens
        utL0 = self.ut_orig.copy()   # non-free: fixed at orig
        utU0 = self.ut_orig.copy()
        dtL0 = self.dt_orig.copy()
        dtU0 = self.dt_orig.copy()
        for g in self.free:
            utL0[g] = 1
            dtL0[g] = 1

        root = UTDTNode(id=0, utL=utL0, utU=utU0, dtL=dtL0, dtU=dtU0)

        heap: List = []
        tie = 0
        open_node_lbs: Dict[int, float] = {}

        # Warm-start: check original params with PLAIN UC (no foil embedded)
        # Correct check: does the cost-minimising dispatch at original UT/DT
        # naturally produce curtailment <= target?
        curt0, sol0 = self._check_plain_at(
            self.ut_orig, self.dt_orig, ws, pbn, u_init, p_init, on_t, off_t)
        if self.verbose:
            print(f"[UTDT-BS] Warm-start: plain metric at orig={curt0:.4f}  "
                  f"target={self._curt_target:.4f}  "
                  f"satisfied={curt0 <= self._curt_target + 1e-3}")
        if curt0 <= self._curt_target + 1e-3:
            self._update_incumbent(self.ut_orig, self.dt_orig, 0.0, sol0, "orig")
            if self.verbose:
                print("[UTDT-BS] Original UT/DT naturally satisfies foil — F=0h  "
                      "(no structural change needed)")

        # Also warm-start with most-flexible config (all UT=DT=1)
        curt_flex, sol_flex = self._check_plain_at(
            utL0, dtL0, ws, pbn, u_init, p_init, on_t, off_t)
        if curt_flex <= self._curt_target + 1e-3:
            obj_flex = self.F(utL0, dtL0)
            self._update_incumbent(utL0, dtL0, obj_flex, sol_flex, "full_flex")

        # Initialise root
        self._init_node(root, ws, pbn, u_init, p_init, on_t, off_t)
        if np.isfinite(root.olb):
            heapq.heappush(heap, (root.olb, tie, root.id, root))
            open_node_lbs[root.id] = root.olb
            tie += 1

        nodes_processed = 0
        certified = False

        while heap and nodes_processed < self.max_nodes:

            global_LB = min(open_node_lbs.values()) if open_node_lbs else float("inf")
            if self.best_ut is not None:
                global_LB = min(global_LB, self.best_F)

            if self.best_ut is not None and global_LB >= self.best_F - self.eps_obj:
                if self.verbose:
                    print(f"[UTDT-BS] CERTIFIED: "
                          f"gLB={global_LB:.2f}h >= bestF={self.best_F:.2f}h")
                certified = True
                break

            olb_key, _, _, node = heapq.heappop(heap)
            open_node_lbs.pop(node.id, None)
            nodes_processed += 1

            if self.best_ut is not None and olb_key >= self.best_F - self.eps_obj:
                continue

            gap_pct = ((self.best_F - global_LB) / max(abs(self.best_F), 1e-9) * 100
                       if np.isfinite(self.best_F) else float("nan"))
            if self.verbose:
                print(f"[UTDT-BS] node={node.id:04d}  olb={node.olb:.1f}h  "
                      f"bestF={self.best_F:.1f}h  "
                      f"gLB={global_LB:.1f}h  gap={gap_pct:.1f}%")

            # OUB: solve PLAIN UC at most-flexible corner, check curtailment naturally
            curt_c, sol_c = self._check_plain_at(
                node.utL, node.dtL, ws, pbn, u_init, p_init, on_t, off_t)
            if curt_c <= self._curt_target + 1e-3:
                obj_c = self.F(node.utL, node.dtL)
                self._update_incumbent(node.utL, node.dtL, obj_c, sol_c,
                                       f"node={node.id:04d}")

            # Branch
            n1, n2 = self._branch(node)
            if n1 is None:
                continue

            for child in (n1, n2):
                self._init_node(child, ws, pbn, u_init, p_init, on_t, off_t)
                if not np.isfinite(child.olb):
                    continue
                if self.best_ut is not None and child.olb >= self.best_F - self.eps_obj:
                    continue
                heapq.heappush(heap, (child.olb, tie, child.id, child))
                open_node_lbs[child.id] = child.olb
                tie += 1

            # History
            gLB_now = min(open_node_lbs.values()) if open_node_lbs else float("inf")
            if self.best_ut is not None:
                gLB_now = min(gLB_now, self.best_F)
            self._history.append({
                "nodes":     nodes_processed,
                "best_F":    self.best_F,
                "global_LB": gLB_now,
                "gap_pct":   ((self.best_F - gLB_now) /
                              max(abs(self.best_F), 1e-9) * 100)
                             if np.isfinite(self.best_F) else float("nan"),
            })

        global_LB = (min(open_node_lbs.values()) if open_node_lbs
                     else (self.best_F if self.best_ut is not None else float("inf")))
        if self.best_ut is not None:
            global_LB = min(global_LB, self.best_F)
        gap = (self.best_F - global_LB) if np.isfinite(self.best_F) else float("inf")

        if self.verbose:
            status = "CERTIFIED" if certified else "BUDGET_EXHAUSTED"
            print(f"\n[UTDT-BS] {status}  F={self.best_F:.1f}h  "
                  f"gLB={global_LB:.1f}h  gap={gap:.2f}h  nodes={nodes_processed}")

        return {
            "success":   self.best_ut is not None,
            "F_opt":     self.best_F,
            "ut_hat":    self.best_ut,
            "dt_hat":    self.best_dt,
            "sol":       self.best_sol,
            "global_LB": global_LB,
            "gap":       gap,
            "nodes":     nodes_processed,
            "certified": certified,
            "history":   self._history,
        }
