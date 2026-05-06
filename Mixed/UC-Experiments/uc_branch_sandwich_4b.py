from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable
import json
import pathlib
import threading
import time as _time
from collections import OrderedDict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import heapq
from uc_master_relax_4b import build_uc_relax_master_varfmax_4b


# ============================================================
# LRU cache (bounded dict)
# ============================================================

class _LRUCache:
    """Thread-safe bounded LRU cache."""
    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __contains__(self, key):
        with self._lock:
            return key in self._cache

    def __len__(self):
        with self._lock:
            return len(self._cache)


# ============================================================
# Gurobi keep-alive thread
# ============================================================

class _GurobiKeepAlive:
    """
    Pings Gurobi every `interval` seconds with a trivial 1-variable LP
    to prevent WLS license token expiry during long B&S runs.
    Runs as a daemon thread — dies automatically when the main process exits.
    """
    def __init__(self, interval: float = 1200.0):  # 20 min default
        self._interval = float(interval)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                m = gp.Model()
                m.Params.OutputFlag = 0
                x = m.addVar(lb=0.0, ub=1.0, obj=1.0)
                m.optimize()
                m.dispose()
            except Exception:
                pass  # best-effort; main thread handles real errors


# ============================================================
# Node dataclass
# ============================================================

@dataclass
class BSNode4b:
    id: int
    bL: np.ndarray
    bU: np.ndarray

    olb: float = np.inf
    b_star_lb: Optional[np.ndarray] = None

    inner_lb: float = np.inf
    inner_ub: float = np.inf

    oub_failures: int = 0
    parent_inner_ub: float = np.inf
    status: str = "open"


# ============================================================
# Main solver
# ============================================================

class UCBranchAndSandwichWCE_4b:
    """
    UC Branch-and-Sandwich WCE — v3 (final)

    Changelog vs v2:
      - Lagrangian OLB: relaxed master WITHOUT hard foil, penalty term
        lambda * violation(z,b) added to objective. Valid LB because
        violation=0 at any foil-feasible point.
      - node.olb = max(lp_olb, lagrangian_olb) — strictly tighter.
      - Numerical safety: Lagrangian OLB only accepted at GRB.OPTIMAL;
        stale bounds clipped at pop time and in global_LB computation.
      - MIP lower bound (opt-in): after max_nodes exhausted, solve one
        final MIP over [bL0,bU0] with binaries kept INTEGER to get a
        tight, realistic gap for reporting. Activated via
        run(..., compute_final_mip_lb=True, mip_lb_time_limit=900.0).

    Constructor parameters:
      foil_violation_expr_fn : callable or None
          fn(model, var, idx) -> gurobipy.LinExpr
          Non-negative expression = 0 when foil satisfied, > 0 otherwise.
          Normalise so magnitude is comparable to F(b) (O(10)–O(100) MW).
          If None, Lagrangian OLB is skipped (identical to v2).
      lagrange_penalty : float
          Weight lambda. Rule of thumb:
            lambda ~ (typical F) / (max violation magnitude)
          Curtailment/commitment foils (normalised): 100–500
          Cost-threshold foils:                       50–200
          Ramp/uptime foils:                         200–500
          If Gurobi warns about numerics → halve. If [LAG_OLB]+X ~ 0 → double.
    """

    def __init__(
        self,
        oracle,
        data,
        idx,
        cvec: np.ndarray,
        foil_extra_constr_fn: Callable,
        b0: np.ndarray,
        b_bounds: Tuple[np.ndarray, np.ndarray],
        b_free_idx: List[int],
        eps_b: float = 5.0,
        eps_obj: float = 1e-3,
        eps_weak: float = 1e-3,
        max_nodes: int = 500,
        relax_cost_ub: Optional[float] = None,
        master_time_limit: Optional[float] = None,
        output_flag: int = 0,
        verbose: bool = True,
        w: Optional[np.ndarray] = None,
        b_worst_is_bL: bool = True,
        oub_grid_pts: int = 3,
        foil_violation_expr_fn: Optional[Callable] = None,
        lagrange_penalty: float = 500.0,
        node_selection: str = "best_first",  # "best_first" or "fifo"
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1,        # save every N nodes
        keepalive_interval: float = 1200.0,  # seconds between pings
        inner_ub_cache_size: int = 1000,
    ):
        self.oracle = oracle
        self.data = data
        self.idx = idx
        self.cvec = cvec
        self.foil_extra = foil_extra_constr_fn

        self.b0  = np.array(b0, dtype=float)
        self.bL0 = np.array(b_bounds[0], dtype=float)
        self.bU0 = np.array(b_bounds[1], dtype=float)
        self.free = list(b_free_idx)

        self.w = (np.ones_like(self.b0, dtype=float) if w is None
                  else np.asarray(w, dtype=float).reshape(-1))
        if self.w.shape != self.b0.shape:
            raise ValueError(f"w shape mismatch: expected {self.b0.shape}")

        self.eps_b    = float(eps_b)
        self.eps_obj  = float(eps_obj)
        self.eps_weak = float(eps_weak)
        self.max_nodes = int(max_nodes)

        self.master_time_limit = master_time_limit
        self.output_flag = int(output_flag)
        self.verbose = bool(verbose)
        self.b_worst_is_bL = bool(b_worst_is_bL)
        self.oub_grid_pts  = int(oub_grid_pts)

        self.foil_violation_expr_fn = foil_violation_expr_fn
        self.lagrange_penalty = float(lagrange_penalty)

        self.node_selection = node_selection

        # Checkpoint
        self.checkpoint_path = pathlib.Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_interval = int(checkpoint_interval)

        # Keep-alive
        self._keepalive = _GurobiKeepAlive(interval=keepalive_interval)

        # Incumbents
        self.best_F        = float("inf")
        self.best_b        = None
        self.best_v_plain  = None
        self.best_v_foil   = None
        self.best_inner_ub = float("inf")
        self._node_id      = 0

        # Bounded LRU cache for inner UB evaluations
        self._inner_ub_cache = _LRUCache(maxsize=inner_ub_cache_size)

        # Load checkpoint if it exists
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Objective and box LB
    # ------------------------------------------------------------------

    def F(self, b: np.ndarray) -> float:
        return float(np.sum(self.w[self.free] * (b[self.free] - self.b0[self.free])))

    def lb_box_L1(self, bL: np.ndarray, bU: np.ndarray) -> float:
        lb = 0.0
        for j in self.free:
            wj = float(self.w[j])
            lo = bL[j] - self.b0[j]
            hi = bU[j] - self.b0[j]
            if lo > 0:
                lb += wj * lo
            elif hi < 0:
                lb += wj * abs(hi)
        return lb

    # ------------------------------------------------------------------
    # Weak feasibility check
    # ------------------------------------------------------------------

    def weak_ok(self, b):
        v_plain, _, _ = self.oracle.solve_plain(b)
        v_foil,  _, _ = self.oracle.solve_foil(b)
        if v_plain is None or v_foil is None:
            return False, v_plain, v_foil
        ok = (v_foil <= v_plain + self.eps_weak)
        if not ok and self.verbose:
            print(f"  [weak_ok] FAIL: v_D={v_foil:.6f} > v={v_plain:.6f} + eps")
        return ok, v_plain, v_foil

    # ------------------------------------------------------------------
    # Inner LB: relaxed UC, b free in node box, obj = c^T z
    # ------------------------------------------------------------------

    def _compute_inner_lb(self, node, window_size, per_bus_neutrality,
                          u_init, p_init, on_t, off_t) -> float:
        from uc_master_relax_4b import (
            build_uc_relax_master_varfmax_4b,
            build_uc_operating_cost_expr_4b,
        )
        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data, idx=self.idx, cvec=self.cvec,
            window_size=window_size, per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
            b0=self.b0, node_bL=node.bL, node_bU=node.bU,
            b_free_idx=self.free, foil_extra_constr_fn=None,
            cost_ub=None, output_flag=self.output_flag, w=self.w,
        )
        cost_expr = build_uc_operating_cost_expr_4b(var, self.idx, self.cvec)
        m.setObjective(cost_expr, GRB.MINIMIZE)
        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)
        m.Params.OutputFlag = self.output_flag
        from uc_pipeline import _optimize_with_retry
        _optimize_with_retry(m)
        if m.SolCount == 0 or m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            m.dispose()
            return float("inf")
        result = float(m.ObjVal)
        m.dispose()
        return result

    # ------------------------------------------------------------------
    # Inner UB: integer UC at worst-case b endpoint
    # ------------------------------------------------------------------

    def _compute_inner_ub(self, node, window_size, per_bus_neutrality,
                          u_init, p_init, on_t, off_t) -> float:
        b_worst = node.bL.copy() if self.b_worst_is_bL else node.bU.copy()
        key = tuple(np.round(b_worst[self.free], 4))
        cached = self._inner_ub_cache.get(key)
        if cached is not None:
            return cached
        v_plain, _, _ = self.oracle.solve_plain(b_worst)
        result = float(v_plain) if v_plain is not None else float("inf")
        self._inner_ub_cache.set(key, result)
        return result

    # ------------------------------------------------------------------
    # LP OLB: relaxed master with hard foil + sandwich cut
    # ------------------------------------------------------------------

    def _solve_lp_olb(self, node, window_size, per_bus_neutrality,
                      u_init, p_init, on_t, off_t,
                      warm_start_b=None) -> Tuple[float, Optional[np.ndarray]]:

        cost_ub = min(
            self.best_inner_ub,
            node.inner_ub        if np.isfinite(node.inner_ub)        else float("inf"),
            node.parent_inner_ub if np.isfinite(node.parent_inner_ub) else float("inf"),
        )
        cost_ub = cost_ub if np.isfinite(cost_ub) else None

        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data, idx=self.idx, cvec=self.cvec,
            window_size=window_size, per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
            b0=self.b0, node_bL=node.bL, node_bU=node.bU,
            b_free_idx=self.free, foil_extra_constr_fn=self.foil_extra,
            cost_ub=cost_ub, output_flag=self.output_flag, w=self.w,
        )

        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)
        m.Params.OutputFlag = self.output_flag

        # #1 — warm-start bcap from parent's b_star_lb (clipped to node box)
        if warm_start_b is not None:
            for ell in self.free:
                bcap[ell].Start = float(
                    np.clip(warm_start_b[ell], node.bL[ell], node.bU[ell]))

        from uc_pipeline import _optimize_with_retry
        _optimize_with_retry(m)

        if m.SolCount == 0:
            if m.Status == GRB.INFEASIBLE:
                if self.verbose:
                    print(f"  [LP_OLB] Infeasible node={node.id}")
                try:
                    m.computeIIS(); m.write("master_relax.ilp")
                except gp.GurobiError:
                    pass
            m.dispose()
            return float("inf"), None
        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            m.dispose()
            return float("inf"), None

        b_star = self.b0.copy()
        for ell in self.free:
            b_star[ell] = float(bcap[ell].X)
        result = float(m.ObjVal), b_star
        m.dispose()
        return result

    # ------------------------------------------------------------------
    # Lagrangian OLB: relaxed master WITHOUT hard foil, penalty term added.
    # Valid LB: violation=0 at any foil-feasible point, so
    # penalised obj = ||b-b0||_1 >= true OLB.
    # Only accepted at GRB.OPTIMAL (time-limit gives primal UB, not LB).
    # ------------------------------------------------------------------

    def _solve_lagrangian_olb(self, node, window_size, per_bus_neutrality,
                               u_init, p_init, on_t, off_t,
                               warm_start_b=None) -> float:
        if self.foil_violation_expr_fn is None:
            return float("-inf")

        cost_ub = min(
            self.best_inner_ub,
            node.inner_ub        if np.isfinite(node.inner_ub)        else float("inf"),
            node.parent_inner_ub if np.isfinite(node.parent_inner_ub) else float("inf"),
        )
        cost_ub = cost_ub if np.isfinite(cost_ub) else None

        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data, idx=self.idx, cvec=self.cvec,
            window_size=window_size, per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
            b0=self.b0, node_bL=node.bL, node_bU=node.bU,
            b_free_idx=self.free,
            foil_extra_constr_fn=None,   # NO hard foil
            cost_ub=cost_ub,
            output_flag=self.output_flag, w=self.w,
        )

        try:
            viol_expr = self.foil_violation_expr_fn(m, var, self.idx)
            # #4 — adaptive lambda: scale with current best_F so penalty
            # stays meaningful as the incumbent tightens over the search.
            # Clipped to [50, 2000] to avoid numerical issues.
            if np.isfinite(self.best_F) and self.best_F > self.eps_obj:
                lam = float(np.clip(
                    self.lagrange_penalty * (self.best_F / 10.0), 50.0, 2000.0))
            else:
                lam = self.lagrange_penalty
            m.setObjective(
                m.getObjective() + lam * viol_expr,
                GRB.MINIMIZE,
            )
        except Exception as e:
            if self.verbose:
                print(f"  [LAG_OLB] foil_violation_expr_fn error: {e}")
            return float("-inf")

        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)
        m.Params.OutputFlag = self.output_flag

        # #1 — warm-start bcap from parent's b_star_lb (clipped to node box)
        if warm_start_b is not None:
            for ell in self.free:
                bcap[ell].Start = float(
                    np.clip(warm_start_b[ell], node.bL[ell], node.bU[ell]))

        from uc_pipeline import _optimize_with_retry
        _optimize_with_retry(m)

        # Must be proven optimal — time-limit gives primal value, not a valid LB
        if m.Status != GRB.OPTIMAL or m.SolCount == 0:
            m.dispose()
            return float("-inf")

        result = float(m.ObjVal)
        m.dispose()
        return result

    # ------------------------------------------------------------------
    # Combined OLB: max(LP, Lagrangian) with safety clip
    # ------------------------------------------------------------------

    def _solve_olb_relax(self, node, window_size, per_bus_neutrality,
                         u_init, p_init, on_t, off_t,
                         warm_start_b=None) -> Tuple[float, Optional[np.ndarray]]:

        lp_olb, b_star = self._solve_lp_olb(
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t,
            warm_start_b=warm_start_b)

        if not np.isfinite(lp_olb):
            return float("inf"), None

        lag_olb = self._solve_lagrangian_olb(
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t,
            warm_start_b=warm_start_b)

        if np.isfinite(lag_olb):
            # Safety: discard if numerically corrupt (exceeds known incumbent)
            if self.best_b is not None and lag_olb > self.best_F + self.eps_obj:
                if self.verbose:
                    print(f"  [LAG_OLB] DISCARDED (lag={lag_olb:.4f} > bestF={self.best_F:.4f})")
                lag_olb = lp_olb
            elif lag_olb - lp_olb > self.eps_obj:
                if self.verbose:
                    print(f"  [LAG_OLB] +{lag_olb - lp_olb:.4f}  "
                          f"(lp={lp_olb:.4f} -> lag={lag_olb:.4f})")

        return max(lp_olb, lag_olb), b_star

    # ------------------------------------------------------------------
    # MIP lower bound (opt-in, called after max_nodes exhausted)
    #
    # Solves the full problem over [bL0, bU0] with binaries INTEGER.
    # Uses m.ObjBound (the MIP solver's proven dual bound) as the LB —
    # valid even if the MIP didn't finish, as long as SolCount > 0.
    # Time-limited to mip_lb_time_limit seconds.
    # ------------------------------------------------------------------

    def compute_mip_lower_bound(
        self,
        window_size, per_bus_neutrality, u_init, p_init, on_t, off_t,
        time_limit: float = 900.0,
    ) -> float:
        """
        Tight final lower bound via a single MIP solve over the full domain.

        Keeps UC binaries INTEGER (no relaxation), enforces the hard foil
        constraint, and applies the sandwich cut. Returns m.ObjBound — the
        MIP solver's proven dual bound — which is a valid LB on F_opt
        regardless of whether the MIP solved to optimality.

        Only called when node budget is exhausted and best_b is not None.
        """
        from uc_pipeline import build_network_uc_model
        from uc_master_relax_4b import (
            remove_fixed_fmax_constraints_4b,
            add_variable_fmax_constraints_4b,
            build_uc_operating_cost_expr_4b,
        )

        if self.verbose:
            print(f"\n[MIP_LB] Solving final MIP lower bound "
                  f"(time_limit={time_limit:.0f}s) ...")

        # Build INTEGER UC — binaries stay binary
        m, var = build_network_uc_model(
            data=self.data,
            window_size=window_size,
            per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init,
            on_time_init=on_t, off_time_init=off_t,
            output_flag=self.output_flag,
        )
        # NOTE: no _relax_tupledict_binary_4b call here — that's the whole point

        nL = len(self.data.lines)
        T  = int(self.data.T)

        # Remove fixed line limits, add variable bcap over FULL domain [bL0, bU0]
        remove_fixed_fmax_constraints_4b(m)

        bcap = {ell: float(self.b0[ell]) for ell in range(nL)}
        for ell in self.free:
            bcap[ell] = m.addVar(
                lb=float(self.bL0[ell]),
                ub=float(self.bU0[ell]),
                vtype=GRB.CONTINUOUS,
                name=f"bcap[{ell}]",
            )

        add_variable_fmax_constraints_4b(m, var, bcap, nL=nL, T=T)

        # Hard foil constraint
        if self.foil_extra is not None:
            self.foil_extra(m, var)

        # Sandwich cut: prevents trivially cheap infeasible dispatch
        if np.isfinite(self.best_inner_ub):
            cost_expr = build_uc_operating_cost_expr_4b(var, self.idx, self.cvec)
            m.addConstr(
                cost_expr <= float(self.best_inner_ub),
                name="sandwich_cut",
            )

        # L1 objective over free lines (weighted)
        bp = {ell: m.addVar(lb=0.0, name=f"bp[{ell}]") for ell in self.free}
        bm = {ell: m.addVar(lb=0.0, name=f"bm[{ell}]") for ell in self.free}
        for ell in self.free:
            m.addConstr(
                bcap[ell] - float(self.b0[ell]) == bp[ell] - bm[ell],
                name=f"l1_link[{ell}]",
            )
        m.setObjective(
            gp.quicksum(
                float(self.w[ell]) * (bp[ell] + bm[ell])
                for ell in self.free
            ),
            GRB.MINIMIZE,
        )

        from uc_pipeline import _optimize_with_retry

        m.Params.TimeLimit  = float(time_limit)
        m.Params.OutputFlag = self.output_flag
        m.Params.MIPGap     = 0.005   # stop at 0.5% internal MIP gap

        _optimize_with_retry(m)

        if m.SolCount == 0:
            if self.verbose:
                print(f"  [MIP_LB] No feasible solution found — "
                      f"ObjBound={m.ObjBound:.4f} (may be loose)")
            mip_lb = float(m.ObjBound) if np.isfinite(m.ObjBound) else float("-inf")
        else:
            mip_lb = float(m.ObjBound)

            mip_F = float(m.ObjVal)
            if mip_F < self.best_F - self.eps_obj:
                b_mip = self.b0.copy()
                for ell in self.free:
                    b_mip[ell] = float(bcap[ell].X)
                self.oracle.cache_foil.clear()
                ok, vp, vD = self.weak_ok(b_mip)
                if ok:
                    self._update_incumbent(b_mip, mip_F, vp, vD, "MIP_LB")
                elif self.verbose:
                    print(f"  [MIP_LB] Incumbent rejected after cache clear: "
                          f"v_D={vD:.3f} > v_plain={vp:.3f} + eps")

        if self.verbose:
            status_name = {
                GRB.OPTIMAL:    "OPTIMAL",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.INFEASIBLE: "INFEASIBLE",
            }.get(m.Status, str(m.Status))
            print(f"  [MIP_LB] Status={status_name}  "
                  f"ObjBound={m.ObjBound:.4f}  "
                  f"ObjVal={m.ObjVal if m.SolCount > 0 else float('nan'):.4f}  "
                  f"MIPGap={m.MIPGap*100:.2f}%")

        # Safety: MIP LB can never exceed a known feasible value
        if self.best_b is not None:
            mip_lb = min(mip_lb, self.best_F)

        m.dispose()
        return mip_lb

    # ------------------------------------------------------------------
    # OUB candidates
    # ------------------------------------------------------------------

    def _oub_candidates(self, node: BSNode4b) -> List[np.ndarray]:
        def proj(b): return np.minimum(np.maximum(b, node.bL), node.bU)

        cands = []
        if node.b_star_lb is not None:
            cands.append(np.minimum(np.maximum(proj(node.b_star_lb), self.bL0), self.bU0))
        cands.append(proj(self.b0))
        cands.append(proj(0.5 * (node.bL + node.bU)))
        cands.append(node.bL.copy())
        cands.append(node.bU.copy())

        b_base = proj(self.b0)
        for ell in self.free:
            for alpha in np.linspace(0.0, 1.0, self.oub_grid_pts + 2):
                b_g = b_base.copy()
                b_g[ell] = (1 - alpha) * node.bL[ell] + alpha * node.bU[ell]
                cands.append(b_g)

        uniq, seen = [], set()
        for b in cands:
            key = tuple(np.round(b[self.free], 6))
            if key not in seen:
                seen.add(key); uniq.append(b)
        return uniq

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def _branch(self, node: BSNode4b):
        # #2 — score each free line by: w * width * (0.5 + deviation)
        # where deviation = how far b_star_lb is from the box midpoint.
        # This focuses splits on dimensions where the LP solution is
        # "hugging" a boundary — i.e. where the relaxation is most dishonest.
        scores = {}
        for ell in self.free:
            width = node.bU[ell] - node.bL[ell]
            if width <= self.eps_b:
                continue
            mid = 0.5 * (node.bL[ell] + node.bU[ell])
            if node.b_star_lb is not None:
                deviation = abs(node.b_star_lb[ell] - mid) / max(width, 1e-9)
            else:
                deviation = 0.0
            scores[ell] = float(self.w[ell]) * width * (0.5 + deviation)

        if not scores:
            return None, None

        ell_max = max(scores, key=scores.__getitem__)
        mid = 0.5 * (node.bL[ell_max] + node.bU[ell_max])
        bL1, bU1 = node.bL.copy(), node.bU.copy(); bU1[ell_max] = mid
        bL2, bU2 = node.bL.copy(), node.bU.copy(); bL2[ell_max] = mid
        n1 = BSNode4b(id=self._node_id+1, bL=bL1, bU=bU1, parent_inner_ub=node.inner_ub)
        n2 = BSNode4b(id=self._node_id+2, bL=bL2, bU=bU2, parent_inner_ub=node.inner_ub)
        self._node_id += 2
        return n1, n2

    # ------------------------------------------------------------------
    # Node initialisation (eager: inner bounds + both OLBs before heap push)
    # ------------------------------------------------------------------

    def _init_node(self, node, window_size, per_bus_neutrality,
                   u_init, p_init, on_t, off_t,
                   warm_start_b=None):
        node.inner_lb = self._compute_inner_lb(
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)
        node.inner_ub = self._compute_inner_ub(
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)

        if node.inner_ub < self.best_inner_ub:
            self.best_inner_ub = node.inner_ub
            if self.verbose:
                print(f"  [INNER_UB] best_inner_ub={self.best_inner_ub:.4f}")

        if node.inner_lb > self.best_inner_ub + self.eps_obj:
            if self.verbose:
                print(f"  [FATHOM-INNER] node={node.id} at init")
            node.olb = float("inf")
            return

        olb, b_star    = self._solve_olb_relax(
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t,
            warm_start_b=warm_start_b)
        node.olb       = olb
        node.b_star_lb = b_star

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, nodes_processed: int, global_LB: float) -> None:
        if self.checkpoint_path is None:
            return
        state = {
            "best_F":        self.best_F,
            "best_b":        self.best_b.tolist() if self.best_b is not None else None,
            "best_v_plain":  self.best_v_plain,
            "best_v_foil":   self.best_v_foil,
            "best_inner_ub": self.best_inner_ub,
            "nodes_processed": nodes_processed,
            "global_LB":     global_LB,
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.checkpoint_path)  # atomic on POSIX
        if self.verbose:
            print(f"  [CKPT] saved → {self.checkpoint_path}  "
                  f"nodes={nodes_processed}  bestF={self.best_F:.4f}  "
                  f"gLB={global_LB:.4f}")

    def _load_checkpoint(self) -> None:
        try:
            state = json.loads(self.checkpoint_path.read_text())
            self.best_F        = float(state["best_F"])
            self.best_b        = np.array(state["best_b"], dtype=float) if state["best_b"] is not None else None
            self.best_v_plain  = state.get("best_v_plain")
            self.best_v_foil   = state.get("best_v_foil")
            self.best_inner_ub = float(state.get("best_inner_ub", float("inf")))
            if self.verbose:
                print(f"[CKPT] Resumed from {self.checkpoint_path}  "
                      f"bestF={self.best_F:.4f}  "
                      f"nodes_prev={state.get('nodes_processed', '?')}  "
                      f"gLB={state.get('global_LB', float('nan')):.4f}")
        except Exception as e:
            if self.verbose:
                print(f"[CKPT] Could not load checkpoint ({e}) — starting fresh")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        window_size, per_bus_neutrality, u_init, p_init, on_t, off_t,
        # MIP LB options (opt-in)
        compute_final_mip_lb: bool = False,
        mip_lb_time_limit: float = 900.0,
    ):
        self._node_id = 0
        root = BSNode4b(id=0, bL=self.bL0.copy(), bU=self.bU0.copy())

        heap: list = []
        tie = 0
        open_node_lbs: Dict[int, float] = {}

        # Start keep-alive thread
        self._keepalive.start()

        # Warm-start incumbents
        ok0, vp0, vD0 = self.weak_ok(self.b0)
        if ok0:
            self._update_incumbent(self.b0, self.F(self.b0), vp0, vD0, "b0")
        b_max = self.b0.copy(); b_max[self.free] = self.bU0[self.free]
        okm, vpm, vDm = self.weak_ok(b_max)
        if okm:
            self._update_incumbent(b_max, self.F(b_max), vpm, vDm, "b_max")

        # Initialise root
        self._init_node(root, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)
        if np.isfinite(root.olb):
            root_lb = max(self.lb_box_L1(root.bL, root.bU), root.olb)
            heapq.heappush(heap, (root_lb, tie, root.id, root))
            open_node_lbs[root.id] = root_lb
            tie += 1

        nodes_processed = 0
        certified = False

        try:
            while heap and nodes_processed < self.max_nodes:

                # Global LB: min over open nodes, clipped to never exceed best_F
                global_LB = min(open_node_lbs.values()) if open_node_lbs else float("inf")
                if self.best_b is not None:
                    global_LB = min(global_LB, self.best_F)

                if self.best_b is not None and global_LB >= self.best_F - self.eps_obj:
                    if self.verbose:
                        print(f"[BS] CERTIFIED: gLB={global_LB:.6f} >= bestF={self.best_F:.6f}")
                    certified = True
                    break

                if self.node_selection == "fifo":
                    _, _, _, node = heap.pop(0)
                else:
                    _, _, _, node = heapq.heappop(heap)
                open_node_lbs.pop(node.id, None)
                nodes_processed += 1

                # ── Lazy init: solve OLB now, when node is actually selected ──────────
                self._init_node(
                    node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t,
                    warm_start_b=node.b_star_lb)  # b_star_lb is None for fresh children

                # Re-check after init: OLB may now be inf (infeasible) or exceed incumbent
                if not np.isfinite(node.olb):
                    continue
                node_lb = max(self.lb_box_L1(node.bL, node.bU), node.olb)
                open_node_lbs[node.id] = node_lb  # update with tighter value

                # Prune by incumbent
                if self.best_b is not None and node_lb >= self.best_F - self.eps_obj:
                    continue

                if self.verbose:
                    gLB = min(open_node_lbs.values()) if open_node_lbs else node_lb
                    gLB = min(gLB, self.best_F) if self.best_b is not None else gLB
                    gap_pct = (self.best_F - gLB) / max(abs(self.best_F), 1e-9) * 100
                    print(f"[BS] node={node.id:04d} olb={node.olb:.4f} "
                          f"inner=[{node.inner_lb:.2f},{node.inner_ub:.2f}] "
                          f"bestF={self.best_F:.4f} gLB={gLB:.4f} gap={gap_pct:.1f}%")

                # Inner fathoming (re-checked with latest best_inner_ub)
                if node.inner_lb > self.best_inner_ub + self.eps_obj:
                    if self.verbose:
                        print(f"  [FATHOM-INNER] node={node.id}")
                    continue

                # OUB: test candidate b vectors
                any_oub_ok = False
                for b in self._oub_candidates(node):
                    ok, vp, vD = self.weak_ok(b)
                    if ok:
                        any_oub_ok = True
                        self._update_incumbent(b, self.F(b), vp, vD, f"node={node.id:04d}")

                #if not any_oub_ok:
                #    node.oub_failures += 1
                #else:
                #    node.oub_failures = 0

                # Structural fathom
                #if (node.oub_failures >= 2
                #        and node_lb <= self.eps_obj
                #        and node.inner_lb >= self.best_F - self.eps_obj):
                #    if self.verbose:
                #        print(f"  [FATHOM-STRUCT] node={node.id:04d}")
                #    continue

                # Branch and initialise children
                n1, n2 = self._branch(node)
                if n1 is None:
                    continue

                for child in (n1, n2):
                    child_lb = max(self.lb_box_L1(child.bL, child.bU), 0.0)
                    if self.best_b is not None and child_lb >= self.best_F - self.eps_obj:
                        continue
                    heapq.heappush(heap, (child_lb, tie, child.id, child))
                    open_node_lbs[child.id] = child_lb
                    tie += 1

                # Periodic checkpoint
                if (self.checkpoint_path is not None
                        and nodes_processed % self.checkpoint_interval == 0):
                    gLB_ckpt = min(open_node_lbs.values()) if open_node_lbs else self.best_F
                    if self.best_b is not None:
                        gLB_ckpt = min(gLB_ckpt, self.best_F)
                    self._save_checkpoint(nodes_processed, gLB_ckpt)

        finally:
            self._keepalive.stop()

        # ── Final global LB ──────────────────────────────────────────
        node_budget_exhausted = (not certified) and (nodes_processed >= self.max_nodes) and heap

        global_LB = (min(open_node_lbs.values()) if open_node_lbs
                     else (self.best_F if self.best_b is not None else float("inf")))
        if self.best_b is not None:
            global_LB = min(global_LB, self.best_F)

        # ── MIP LB (opt-in, only when budget exhausted and we have an incumbent) ──
        mip_lb_used = False
        if compute_final_mip_lb and node_budget_exhausted and self.best_b is not None:
            if self.verbose:
                print(f"\n[B&S] Node budget exhausted ({nodes_processed} nodes). "
                      f"LP gap = {(self.best_F - global_LB)/max(abs(self.best_F),1e-9)*100:.1f}%. "
                      f"Computing MIP lower bound ...")
            mip_lb = self.compute_mip_lower_bound(
                window_size, per_bus_neutrality, u_init, p_init, on_t, off_t,
                time_limit=mip_lb_time_limit,
            )
            if mip_lb > global_LB:
                if self.verbose:
                    print(f"  [MIP_LB] global_LB tightened: "
                          f"{global_LB:.4f} -> {mip_lb:.4f}")
                global_LB  = mip_lb
                mip_lb_used = True
            else:
                if self.verbose:
                    print(f"  [MIP_LB] No improvement over LP LB "
                          f"({mip_lb:.4f} <= {global_LB:.4f})")

        gap = ((self.best_F - global_LB)
               if (self.best_b is not None and np.isfinite(global_LB))
               else float("inf"))

        # Final checkpoint
        self._save_checkpoint(nodes_processed, global_LB)

        return {
            "success":        self.best_b is not None,
            "b_hat":          self.best_b,
            "F_opt":          self.best_F,
            "nodes":          nodes_processed,
            "v_plain":        self.best_v_plain,
            "v_D":            self.best_v_foil,
            "global_LB":      global_LB,
            "gap":            gap,
            "best_inner_ub":  self.best_inner_ub,
            "certified":      certified,
            "mip_lb_used":    mip_lb_used,
        }

    # ------------------------------------------------------------------
    def _update_incumbent(self, b, Fb, vp, vD, label=""):
        if Fb < self.best_F - self.eps_obj:
            self.best_F       = Fb
            self.best_b       = b.copy()
            self.best_v_plain = vp
            self.best_v_foil  = vD
            if self.verbose:
                print(f"  [INC] {label}: F={Fb:.4f} v={vp:.3f} v_D={vD:.3f}")
        # #3 — v_plain at ANY accepted foil-feasible point is a valid inner UB:
        # it's the UC cost at a concrete b vector, so c^T z <= v_plain always
        # holds at that b. Using it as sandwich cut tightens all future OLBs.
        if vp is not None and vp < self.best_inner_ub:
            self.best_inner_ub = float(vp)
            if self.verbose:
                print(f"  [INNER_UB] tightened by incumbent: {vp:.4f}")
