from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import heapq
from uc_master_relax_4b import build_uc_relax_master_varfmax_4b


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
    UC Branch-and-Sandwich WCE — v3 (minimal)

    Builds directly on v2. The only new thing is the Lagrangian OLB:

      _solve_lagrangian_olb() solves the relaxed master WITHOUT the hard
      foil constraint, but with a penalty term added to the objective:

          min  ||b - b0||_1  +  lambda * foil_violation(z, b)
          s.t. UC relaxation (binaries in [0,1])
               b in [bL, bU]
               sandwich cut: c^T z <= best_inner_ub

      This is a valid lower bound: at any point where the foil IS satisfied
      the violation = 0, so the penalised objective == ||b - b0||_1 >= true OLB.

      _solve_olb_relax() now returns max(lp_olb, lagrangian_olb).

    New constructor parameters vs v2:
      foil_violation_expr_fn : callable or None
          fn(model, var, idx) -> gurobipy.LinExpr
          Must return a NON-NEGATIVE linear expression that equals 0 when
          the foil is satisfied and > 0 when it is violated.
          If None the Lagrangian OLB is skipped (identical to v2).
      lagrange_penalty : float
          Weight lambda on the violation. Rule of thumb:
            lambda ~ (typical F value) / (max expected violation magnitude)
          Curtailment / commitment foils: 500-1000
          Cost-threshold foils:           50-200
          Ramp / uptime foils:            200-500
          If Gurobi prints numerical warnings -> halve lambda.
          If [LAG_OLB] improvement is consistently ~0 -> double lambda.
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
        # NEW -------------------------------------------------------
        foil_violation_expr_fn: Optional[Callable] = None,
        lagrange_penalty: float = 500.0,
        # -----------------------------------------------------------
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

        # Incumbents
        self.best_F        = float("inf")
        self.best_b        = None
        self.best_v_plain  = None
        self.best_v_foil   = None
        self.best_inner_ub = float("inf")
        self._node_id      = 0

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
    # Inner LB
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
        m.optimize()
        if m.SolCount == 0 or m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return float("inf")
        return float(m.ObjVal)

    # ------------------------------------------------------------------
    # Inner UB
    # ------------------------------------------------------------------

    def _compute_inner_ub(self, node, window_size, per_bus_neutrality,
                          u_init, p_init, on_t, off_t) -> float:
        b_worst = node.bL.copy() if self.b_worst_is_bL else node.bU.copy()
        v_plain, _, _ = self.oracle.solve_plain(b_worst)
        return float(v_plain) if v_plain is not None else float("inf")

    # ------------------------------------------------------------------
    # LP OLB  (unchanged from v2)
    # ------------------------------------------------------------------

    def _solve_lp_olb(self, node, window_size, per_bus_neutrality,
                      u_init, p_init, on_t, off_t) -> Tuple[float, Optional[np.ndarray]]:

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
        m.optimize()

        if m.SolCount == 0:
            if m.Status == GRB.INFEASIBLE:
                if self.verbose:
                    print(f"  [LP_OLB] Infeasible node={node.id}")
                try:
                    m.computeIIS(); m.write("master_relax.ilp")
                except gp.GurobiError:
                    pass
            return float("inf"), None
        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return float("inf"), None

        b_star = self.b0.copy()
        for ell in self.free:
            b_star[ell] = float(bcap[ell].X)
        return float(m.ObjVal), b_star

    # ------------------------------------------------------------------
    # Lagrangian OLB  (NEW in v3)
    # ------------------------------------------------------------------

    def _solve_lagrangian_olb(self, node, window_size, per_bus_neutrality,
                               u_init, p_init, on_t, off_t) -> float:
        if self.foil_violation_expr_fn is None:
            return float("-inf")

        cost_ub = min(
            self.best_inner_ub,
            node.inner_ub        if np.isfinite(node.inner_ub)        else float("inf"),
            node.parent_inner_ub if np.isfinite(node.parent_inner_ub) else float("inf"),
        )
        cost_ub = cost_ub if np.isfinite(cost_ub) else None

        # Build WITHOUT hard foil constraint
        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data, idx=self.idx, cvec=self.cvec,
            window_size=window_size, per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
            b0=self.b0, node_bL=node.bL, node_bU=node.bU,
            b_free_idx=self.free,
            foil_extra_constr_fn=None,   # <-- no hard foil
            cost_ub=cost_ub,
            output_flag=self.output_flag, w=self.w,
        )

        # Append penalty to existing objective
        try:
            viol_expr = self.foil_violation_expr_fn(m, var, self.idx)
            m.setObjective(
                m.getObjective() + self.lagrange_penalty * viol_expr,
                GRB.MINIMIZE,
            )
        except Exception as e:
            if self.verbose:
                print(f"  [LAG_OLB] foil_violation_expr_fn error: {e}")
            return float("-inf")

        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)
        m.Params.OutputFlag = self.output_flag
        m.optimize()

        if m.SolCount == 0 or m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return float("-inf")

        return float(m.ObjVal)

    # ------------------------------------------------------------------
    # Combined OLB: max(LP, Lagrangian)  (NEW in v3)
    # ------------------------------------------------------------------

    def _solve_olb_relax(self, node, window_size, per_bus_neutrality,
                         u_init, p_init, on_t, off_t) -> Tuple[float, Optional[np.ndarray]]:

        lp_olb, b_star = self._solve_lp_olb(
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)

        if not np.isfinite(lp_olb):
            return float("inf"), None   # node is infeasible, skip Lagrangian

        lag_olb = self._solve_lagrangian_olb(
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)

        if self.verbose and np.isfinite(lag_olb):
            improvement = lag_olb - lp_olb
            if improvement > self.eps_obj:
                print(f"  [LAG_OLB] +{improvement:.4f}  "
                      f"(lp={lp_olb:.4f} -> lag={lag_olb:.4f})")

        return max(lp_olb, lag_olb), b_star

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
        widths = [(ell, node.bU[ell] - node.bL[ell]) for ell in self.free]
        ell_max, w_max = max(widths, key=lambda x: x[1])
        if w_max <= self.eps_b:
            return None, None
        mid = 0.5 * (node.bL[ell_max] + node.bU[ell_max])
        bL1, bU1 = node.bL.copy(), node.bU.copy(); bU1[ell_max] = mid
        bL2, bU2 = node.bL.copy(), node.bU.copy(); bL2[ell_max] = mid
        n1 = BSNode4b(id=self._node_id+1, bL=bL1, bU=bU1, parent_inner_ub=node.inner_ub)
        n2 = BSNode4b(id=self._node_id+2, bL=bL2, bU=bU2, parent_inner_ub=node.inner_ub)
        self._node_id += 2
        return n1, n2

    # ------------------------------------------------------------------
    # Node initialisation (eager)
    # ------------------------------------------------------------------

    def _init_node(self, node, window_size, per_bus_neutrality,
                   u_init, p_init, on_t, off_t):
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
            node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)
        node.olb       = olb
        node.b_star_lb = b_star

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t):
        self._node_id = 0
        root = BSNode4b(id=0, bL=self.bL0.copy(), bU=self.bU0.copy())

        heap: list = []
        tie = 0
        open_node_lbs: Dict[int, float] = {}

        # Warm-start
        ok0, vp0, vD0 = self.weak_ok(self.b0)
        if ok0:
            self._update_incumbent(self.b0, self.F(self.b0), vp0, vD0, "b0")
        b_max = self.b0.copy(); b_max[self.free] = self.bU0[self.free]
        okm, vpm, vDm = self.weak_ok(b_max)
        if okm:
            self._update_incumbent(b_max, self.F(b_max), vpm, vDm, "b_max")

        # Root
        self._init_node(root, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)
        if np.isfinite(root.olb):
            root_lb = max(self.lb_box_L1(root.bL, root.bU), root.olb)
            heapq.heappush(heap, (root_lb, tie, root.id, root))
            open_node_lbs[root.id] = root_lb
            tie += 1

        nodes_processed = 0

        while heap and nodes_processed < self.max_nodes:

            global_LB = min(open_node_lbs.values()) if open_node_lbs else float("inf")
            if self.best_b is not None and global_LB >= self.best_F - self.eps_obj:
                if self.verbose:
                    print(f"[BS] CERTIFIED: gLB={global_LB:.6f} >= bestF={self.best_F:.6f}")
                break

            lb_key, _, _, node = heapq.heappop(heap)
            open_node_lbs.pop(node.id, None)
            nodes_processed += 1

            if self.best_b is not None and lb_key >= self.best_F - self.eps_obj:
                continue

            node_lb = max(self.lb_box_L1(node.bL, node.bU), float(node.olb))

            if self.verbose:
                gLB = min(open_node_lbs.values()) if open_node_lbs else node_lb
                gap_pct = (self.best_F - gLB) / max(abs(self.best_F), 1e-9) * 100
                print(f"[BS] node={node.id:04d} olb={node.olb:.4f} "
                      f"inner=[{node.inner_lb:.2f},{node.inner_ub:.2f}] "
                      f"bestF={self.best_F:.4f} gLB={gLB:.4f} gap={gap_pct:.1f}%")

            if node.inner_lb > self.best_inner_ub + self.eps_obj:
                if self.verbose:
                    print(f"  [FATHOM-INNER] node={node.id}")
                continue

            # OUB
            any_oub_ok = False
            for b in self._oub_candidates(node):
                ok, vp, vD = self.weak_ok(b)
                if ok:
                    any_oub_ok = True
                    self._update_incumbent(b, self.F(b), vp, vD, f"node={node.id:04d}")

            if not any_oub_ok:
                node.oub_failures += 1
            else:
                node.oub_failures = 0

            if (node.oub_failures >= 2
                    and node_lb <= self.eps_obj
                    and node.inner_lb >= self.best_F - self.eps_obj):
                if self.verbose:
                    print(f"  [FATHOM-STRUCT] node={node.id:04d}")
                continue

            # Branch
            n1, n2 = self._branch(node)
            if n1 is None:
                continue

            for child in (n1, n2):
                self._init_node(
                    child, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t)
                if not np.isfinite(child.olb):
                    continue
                child_lb = max(self.lb_box_L1(child.bL, child.bU), child.olb)
                if self.best_b is not None and child_lb >= self.best_F - self.eps_obj:
                    continue
                heapq.heappush(heap, (child_lb, tie, child.id, child))
                open_node_lbs[child.id] = child_lb
                tie += 1

        global_LB = (min(open_node_lbs.values()) if open_node_lbs
                     else (self.best_F if self.best_b is not None else float("inf")))
        gap = ((self.best_F - global_LB)
               if (self.best_b is not None and np.isfinite(global_LB))
               else float("inf"))

        return {
            "success":       self.best_b is not None,
            "b_hat":         self.best_b,
            "F_opt":         self.best_F,
            "nodes":         nodes_processed,
            "v_plain":       self.best_v_plain,
            "v_D":           self.best_v_foil,
            "global_LB":     global_LB,
            "gap":           gap,
            "best_inner_ub": self.best_inner_ub,
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
