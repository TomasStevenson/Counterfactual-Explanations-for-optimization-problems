# uc_branch_sandwich_4b.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import heapq
from uc_master_relax_4b import build_uc_relax_master_varfmax_4b


# ============================================================
# Node dataclass — extended with inner bounding fields
# ============================================================

@dataclass
class BSNode4b:
    id: int
    bL: np.ndarray
    bU: np.ndarray

    # OLB (outer lower bound on F = ||b - b0||_1)
    olb: float = np.inf
    b_star_lb: Optional[np.ndarray] = None

    # Inner bounds on v*(b) = min_{z feasible} c^T z over node's b-domain
    inner_lb: float = np.inf   # LB on v*(b): relaxed UC, b free in [bL,bU], obj = c^T z
    inner_ub: float = np.inf   # UB on v*(b): integer UC at b = bU (worst-case network)

    # OUB failure counter (for structural fathoming)
    oub_failures: int = 0

    status: str = "open"


# ============================================================
# Main solver
# ============================================================

class UCBranchAndSandwichWCE_4b:
    """
    UC Branch-and-Sandwich WCE with two-level bounding
    (Kleniati & Adjiman, 2015 structure, UC-specific).

    Outer problem: min ||b - b0||_1  (line capacity perturbation)
    Inner problem: UC optimality + foil feasibility

    Bounding:
      - Inner LB: relaxed UC (binaries in [0,1]), b free in [bL,bU], obj = c^T z
      - Inner UB: integer UC (plain, no foil) at b = bU (worst-case network)
      - Outer LB: relaxed master with cost_ub = best_inner_ub  [SANDWICH COUPLING]
      - Outer UB: integer oracle weak_ok check on candidate b vectors
    """

    def __init__(
        self,
        oracle,
        data,
        idx,
        cvec: np.ndarray,
        foil_extra_constr_fn,
        b0: np.ndarray,
        b_bounds: Tuple[np.ndarray, np.ndarray],
        b_free_idx: List[int],
        eps_b: float = 5.0,
        eps_obj: float = 1e-3,
        eps_weak: float = 1e-3,
        max_nodes: int = 500,
        relax_cost_ub: Optional[float] = None,   # kept for back-compat; overridden internally
        master_time_limit: Optional[float] = None,
        output_flag: int = 0,
        verbose: bool = True,
        w: Optional[np.ndarray] = None,
    ):
        self.oracle = oracle
        self.data = data
        self.idx = idx
        self.cvec = cvec
        self.foil_extra = foil_extra_constr_fn

        self.b0 = np.array(b0, dtype=float)
        self.bL0 = np.array(b_bounds[0], dtype=float)
        self.bU0 = np.array(b_bounds[1], dtype=float)
        self.free = list(b_free_idx)

        if w is None:
            self.w = np.ones_like(self.b0, dtype=float)
        else:
            w = np.asarray(w, dtype=float).reshape(-1)
            if w.shape != self.b0.shape:
                raise ValueError(f"w must have shape {self.b0.shape}, got {w.shape}")
            self.w = w

        self.eps_b    = float(eps_b)
        self.eps_obj  = float(eps_obj)
        self.eps_weak = float(eps_weak)
        self.max_nodes = int(max_nodes)

        self.master_time_limit = master_time_limit
        self.output_flag = int(output_flag)
        self.verbose = bool(verbose)

        # Incumbents
        self.best_F       = float("inf")
        self.best_b       = None
        self.best_v_plain = None
        self.best_v_foil  = None

        # Best inner UB seen across all nodes (used as sandwich coupling)
        self.best_inner_ub = float("inf")

        self._node_id = 0


    # ------------------------------------------------------------------
    # Objective F(b): weighted sum of capacity increases on free lines
    # ------------------------------------------------------------------

    def F(self, b: np.ndarray) -> float:
        d = b[self.free] - self.b0[self.free]
        return float(np.sum(self.w[self.free] * d))

    def lb_box_L1(self, bL: np.ndarray, bU: np.ndarray) -> float:
        """Valid LB on F from box geometry alone."""
        lb = 0.0
        for j in self.free:
            wj = float(self.w[j])
            if self.b0[j] < bL[j]:
                lb += wj * (bL[j] - self.b0[j])
            elif self.b0[j] > bU[j]:
                lb += wj * (self.b0[j] - bU[j])
        return float(lb)


    # ------------------------------------------------------------------
    # Weak feasibility check (OUB oracle)
    # ------------------------------------------------------------------

    def weak_ok(self, b: np.ndarray) -> Tuple[bool, Optional[float], Optional[float]]:
        v_plain, _, _ = self.oracle.solve_plain(b)
        v_foil,  _, _ = self.oracle.solve_foil(b)

        if v_plain is None or v_foil is None:
            if self.verbose:
                print("  [weak_ok] FAIL: oracle returned None")
            return False, v_plain, v_foil

        ok = (v_foil <= v_plain + self.eps_weak)
        if not ok and self.verbose:
            print(f"  [weak_ok] FAIL: v_D={v_foil:.6f} > v={v_plain:.6f} + eps")
        return ok, v_plain, v_foil


    # ------------------------------------------------------------------
    # INNER LOWER BOUND
    # Solve relaxed UC (binaries in [0,1]), b free in [node.bL, node.bU],
    # objective = c^T z  (full operating cost, NOT ||b-b0||_1).
    # This bounds the best UC cost achievable anywhere in the node.
    # ------------------------------------------------------------------

    def _compute_inner_lb(
        self, node: BSNode4b,
        window_size, per_bus_neutrality, u_init, p_init, on_t, off_t
    ) -> float:
        from uc_master_relax_4b import (
            build_uc_relax_master_varfmax_4b,
            build_uc_operating_cost_expr_4b,
        )

        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data,
            idx=self.idx,
            cvec=self.cvec,
            window_size=window_size,
            per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init,
            on_time_init=on_t, off_time_init=off_t,
            b0=self.b0,
            node_bL=node.bL,
            node_bU=node.bU,
            b_free_idx=self.free,
            foil_extra_constr_fn=None,   # NO foil: inner problem is plain UC
            cost_ub=None,                # no sandwich cut here
            output_flag=self.output_flag,
            w=self.w,
        )

        # Override objective: minimize c^T z (full operating cost)
        cost_expr = build_uc_operating_cost_expr_4b(var, self.idx, self.cvec)
        m.setObjective(cost_expr, GRB.MINIMIZE)

        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)
        m.Params.OutputFlag = self.output_flag
        m.optimize()

        if m.SolCount == 0:
            return float("inf")
        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return float("inf")

        return float(m.ObjVal)


    # ------------------------------------------------------------------
    # INNER UPPER BOUND
    # Solve integer UC (plain, no foil) at b = node.bU.
    # Monotonicity: tighter network (higher b = lower capacity) -> higher cost,
    # so b_worst = bU gives a valid UB on v*(b) for all b in [bL, bU].
    # ------------------------------------------------------------------

    def _compute_inner_ub(
        self, node: BSNode4b,
        window_size, per_bus_neutrality, u_init, p_init, on_t, off_t
    ) -> float:
        # Most restrictive = lowest capacity = bL -> highest cost -> valid UB on v*(b)
        b_worst = node.bL.copy()   # <-- FIXED
        v_plain, _, _ = self.oracle.solve_plain(b_worst)
        if v_plain is None:
            return float("inf")
        return float(v_plain)


    # ------------------------------------------------------------------
    # OUTER LOWER BOUND  [SANDWICH COUPLING]
    # Relaxed master with:
    #   objective  = ||b - b0||_1  (line perturbation)
    #   cost_ub    = best_inner_ub  <- THIS IS THE SANDWICH CUT
    # ------------------------------------------------------------------

    def _solve_olb_relax(
        self, node: BSNode4b,
        window_size, per_bus_neutrality, u_init, p_init, on_t, off_t
    ) -> Tuple[float, Optional[np.ndarray]]:

        # Use best_inner_ub as sandwich coupling (tightens OLB)
        cost_ub = self.best_inner_ub if np.isfinite(self.best_inner_ub) else None

        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data,
            idx=self.idx,
            cvec=self.cvec,
            window_size=window_size,
            per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init,
            on_time_init=on_t, off_time_init=off_t,
            b0=self.b0,
            node_bL=node.bL,
            node_bU=node.bU,
            b_free_idx=self.free,
            foil_extra_constr_fn=self.foil_extra,
            cost_ub=cost_ub,             # <- sandwich coupling
            output_flag=self.output_flag,
            w=self.w,
        )

        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)
        m.Params.OutputFlag = self.output_flag

        try:
            m.Params.SymbolicNames = 1
        except Exception:
            pass

        m.optimize()

        if m.SolCount == 0:
            if self.verbose:
                print(f"  [OLB] Status={m.Status}, no solution")
            if m.Status == GRB.INFEASIBLE:
                try:
                    m.computeIIS()
                    m.write("master_relax.ilp")
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
    # Branching: split widest free-line interval
    # ------------------------------------------------------------------

    def _branch(self, node: BSNode4b) -> Tuple[Optional[BSNode4b], Optional[BSNode4b]]:
        widths = [(ell, node.bU[ell] - node.bL[ell]) for ell in self.free]
        ell_max, w_max = max(widths, key=lambda x: x[1])

        if w_max <= self.eps_b:
            return None, None

        mid = 0.5 * (node.bL[ell_max] + node.bU[ell_max])

        bL1, bU1 = node.bL.copy(), node.bU.copy()
        bL2, bU2 = node.bL.copy(), node.bU.copy()
        bU1[ell_max] = mid
        bL2[ell_max] = mid

        n1 = BSNode4b(id=self._node_id + 1, bL=bL1, bU=bU1)
        n2 = BSNode4b(id=self._node_id + 2, bL=bL2, bU=bU2)
        self._node_id += 2
        return n1, n2

    def _project_to_bounds(self, b, bL, bU):
        return np.minimum(np.maximum(b, bL), bU)


    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t):

        self._node_id = 0
        root = BSNode4b(id=0, bL=self.bL0.copy(), bU=self.bU0.copy())

        heap = []
        tie = 0

        root_lb = self.lb_box_L1(root.bL, root.bU)
        heapq.heappush(heap, (root_lb, tie, root.id, root))
        tie += 1

        # --- warm-start incumbents ---
        ok0, vp0, vD0 = self.weak_ok(self.b0)
        if ok0:
            Fb0 = self.F(self.b0)
            self._update_incumbent(self.b0, Fb0, vp0, vD0, label="b0")

        b_max = self.b0.copy()
        b_max[self.free] = self.bU0[self.free]
        okm, vpm, vDm = self.weak_ok(b_max)
        if okm:
            Fb = self.F(b_max)
            self._update_incumbent(b_max, Fb, vpm, vDm, label="b_max")

        nodes_processed = 0

        while heap and nodes_processed < self.max_nodes:

            # --- certified termination ---
            global_LB = heap[0][0]
            if self.best_b is not None and global_LB >= self.best_F - self.eps_obj:
                if self.verbose:
                    print(f"[BS] CERTIFIED: global_LB={global_LB:.6f} >= bestF={self.best_F:.6f}")
                break

            lb_key, _, _, node = heapq.heappop(heap)
            nodes_processed += 1

            if self.best_b is not None and lb_key >= self.best_F - self.eps_obj:
                continue   # prune by incumbent

            # ==================================================
            # STEP 1: Inner bounds (new — two-level bounding)
            # ==================================================
            node.inner_lb = self._compute_inner_lb(
                node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t
            )
            node.inner_ub = self._compute_inner_ub(
                node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t
            )

            if self.verbose:
                print(f"[BS] node={node.id:04d} inner_lb={node.inner_lb:.4f} "
                      f"inner_ub={node.inner_ub:.4f} bestF={self.best_F:.4f}")

            # Update global best_inner_ub (used as sandwich cut in OLB)
            if node.inner_ub < self.best_inner_ub:
                self.best_inner_ub = node.inner_ub
                if self.verbose:
                    print(f"  [INNER_UB] updated best_inner_ub={self.best_inner_ub:.4f}")

            # --- inner fathoming ---
            if node.inner_lb > self.best_inner_ub + self.eps_obj:
                if self.verbose:
                    print(f"  [FATHOM-INNER] inner_lb={node.inner_lb:.4f} "
                          f"> best_inner_ub={self.best_inner_ub:.4f}")
                continue

            # ==================================================
            # STEP 2: Outer LB (relaxed master WITH sandwich cut)
            # ==================================================
            if not np.isfinite(node.olb) or node.b_star_lb is None:
                olb, b_star_lb = self._solve_olb_relax(
                    node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t
                )
                node.olb = olb
                node.b_star_lb = b_star_lb

                if not np.isfinite(olb) or b_star_lb is None:
                    continue   # infeasible node

                new_lb = max(self.lb_box_L1(node.bL, node.bU), float(olb))
                if self.verbose:
                    print(f"  [OLB] olb={olb:.4f} new_lb_key={new_lb:.4f}")

                # Reinsert with tightened LB for correct ordering
                heapq.heappush(heap, (new_lb, tie, node.id, node))
                tie += 1
                continue

            node_lb = max(self.lb_box_L1(node.bL, node.bU), float(node.olb))

            # Prune by tightened OLB
            if self.best_b is not None and node_lb >= self.best_F - self.eps_obj:
                continue

            # ==================================================
            # STEP 3: Outer UB — try candidate b vectors
            # ==================================================
            cand = []
            b_try = self._project_to_bounds(node.b_star_lb, node.bL, node.bU)
            b_try = self._project_to_bounds(b_try, self.bL0, self.bU0)
            cand.append(b_try)
            cand.append(self._project_to_bounds(self.b0, node.bL, node.bU))
            cand.append(0.5 * (node.bL + node.bU))
            b_upper = node.bL.copy(); b_upper[self.free] = node.bU[self.free]
            cand.append(b_upper)
            cand.append(node.bU.copy())

            uniq, seen_keys = [], set()
            for b in cand:
                key = tuple(np.round(b[self.free], 6))
                if key not in seen_keys:
                    seen_keys.add(key)
                    uniq.append(b)

            any_oub_ok = False
            for b in uniq:
                ok, vp, vD = self.weak_ok(b)
                if ok:
                    any_oub_ok = True
                    Fb = self.F(b)
                    self._update_incumbent(b, Fb, vp, vD,
                                           label=f"node={node.id:04d}")

            if not any_oub_ok:
                node.oub_failures += 1
            else:
                node.oub_failures = 0

            # Structural fathom (degenerate node, repeated OUB failures)
            if (node.oub_failures >= 2
                    and node_lb <= self.eps_obj
                    and node.inner_lb >= self.best_F - self.eps_obj):
                if self.verbose:
                    print(f"  [FATHOM-STRUCT] node={node.id:04d}")
                continue

            # ==================================================
            # STEP 4: Branch
            # ==================================================
            n1, n2 = self._branch(node)
            if n1 is None:
                continue

            lb1 = self.lb_box_L1(n1.bL, n1.bU)
            lb2 = self.lb_box_L1(n2.bL, n2.bU)
            heapq.heappush(heap, (lb1, tie, n1.id, n1)); tie += 1
            heapq.heappush(heap, (lb2, tie, n2.id, n2)); tie += 1

        final_global_LB = heap[0][0] if heap else (
            self.best_F if self.best_b is not None else float("inf")
        )
        gap = (
            (self.best_F - final_global_LB)
            if (self.best_b is not None and np.isfinite(final_global_LB))
            else float("inf")
        )

        return {
            "success":    self.best_b is not None,
            "b_hat":      self.best_b,
            "F_opt":      self.best_F,
            "nodes":      nodes_processed,
            "v_plain":    self.best_v_plain,
            "v_D":        self.best_v_foil,
            "global_LB":  final_global_LB,
            "gap":        gap,
            "best_inner_ub": self.best_inner_ub,
        }

    # ------------------------------------------------------------------
    # Helper: update incumbent if improved
    # ------------------------------------------------------------------

    def _update_incumbent(self, b, Fb, vp, vD, label=""):
        if Fb < self.best_F - self.eps_obj:
            self.best_F       = Fb
            self.best_b       = b.copy()
            self.best_v_plain = vp
            self.best_v_foil  = vD
            if self.verbose:
                print(f"  [INC] {label}: F={Fb:.4f} v={vp:.3f} v_D={vD:.3f}")