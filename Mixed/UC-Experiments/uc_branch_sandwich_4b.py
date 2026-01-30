# uc_branch_sandwich_4b.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import heapq
from uc_master_relax_4b import build_uc_relax_master_varfmax_4b


@dataclass
class BSNode4b:
    id: int
    bL: np.ndarray
    bU: np.ndarray
    olb: float = np.inf
    b_star_lb: Optional[np.ndarray] = None
    status: str = "open"


class UCBranchAndSandwichWCE_4b:
    """
    UC Branch-and-Sandwich style:
      - OLB: solve relaxed master in node to get LB on ||b-b0||_1
      - OUB: verify candidate b with UC oracle (plain + foil)
      - branch: split widest coordinate in b_free_idx
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
        eps_b: float = 5.0,            # node width threshold (MW)
        eps_obj: float = 1e-3,         # fathom tolerance on F
        eps_weak: float = 1e-3,        # weak check tolerance
        max_nodes: int = 500,
        relax_cost_ub: Optional[float] = None,
        master_time_limit: Optional[float] = None,
        output_flag: int = 0,
        verbose: bool = True,
        w: Optional[np.ndarray] = None,   # <-- NEW
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
        # -----------------------------
        # Weights for reinforcement cost
        # -----------------------------
        if w is None:
            self.w = np.ones_like(self.b0, dtype=float)
        else:
            w = np.asarray(w, dtype=float).reshape(-1)
            if w.shape != self.b0.shape:
                raise ValueError(f"w must have shape {self.b0.shape}, got {w.shape}")
            self.w = w


        self.eps_b = float(eps_b)
        self.eps_obj = float(eps_obj)
        self.eps_weak = float(eps_weak)
        self.max_nodes = int(max_nodes)

        self.relax_cost_ub = relax_cost_ub
        self.master_time_limit = master_time_limit
        self.output_flag = int(output_flag)
        self.verbose = bool(verbose)

        self.best_F = float("inf")
        self.best_b = None
        self.best_v_plain = None
        self.best_v_foil = None

        self._node_id = 0

        self._master_m = None
        self._master_bcap = None
        self._master_built = False


    def F(self, b: np.ndarray) -> float:
        """
        Weighted reinforcement objective:
        F(b) = sum_{ell in free} w_ell * (b_ell - b0_ell)
        Assumes you enforce reinforcement-only bounds (b >= b0) on free lines.
        """
        d = b[self.free] - self.b0[self.free]
        return float(np.sum(self.w[self.free] * d))


    def lb_box_L1(self, bL: np.ndarray, bU: np.ndarray) -> float:
        """
        Valid LB on weighted distance from b0 to the box [bL,bU]:
        LB = sum_{j in free} w_j * dist(b0_j, [bL_j, bU_j])
        If reinforcement-only (bL >= b0), then dist reduces to (bL-b0).
        """
        lb = 0.0
        for j in self.free:
            wj = float(self.w[j])
            if self.b0[j] < bL[j]:
                lb += wj * (bL[j] - self.b0[j])
            elif self.b0[j] > bU[j]:
                lb += wj * (self.b0[j] - bU[j])
        return float(lb)


    def weak_ok(self, b: np.ndarray) -> Tuple[bool, Optional[float], Optional[float]]:
        v_plain, _, _ = self.oracle.solve_plain(b)
        v_foil,  _, _ = self.oracle.solve_foil(b)

        if v_plain is None or v_foil is None:
            if self.verbose:
                print("  [weak_ok] FAIL: v_plain or v_foil is None")
            return False, v_plain, v_foil

        ok = (v_foil <= v_plain + self.eps_weak)
        if (not ok) and self.verbose:
            print(f"  [weak_ok] FAIL: vD={v_foil:.6f} > v={v_plain:.6f} + eps={self.eps_weak}")
        return ok, v_plain, v_foil

    def _project_to_bounds(self, b: np.ndarray, bL: np.ndarray, bU: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(b, bL), bU)

    def _ensure_master_built(self, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t):
        if self._master_built:
            return

        # Build ONCE using root bounds
        dummy_node = BSNode4b(id=-1, bL=self.bL0.copy(), bU=self.bU0.copy())

        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data,
            idx=self.idx,
            cvec=self.cvec,
            window_size=window_size,
            per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
            b0=self.b0,
            node_bL=dummy_node.bL,
            node_bU=dummy_node.bU,
            b_free_idx=self.free,
            foil_extra_constr_fn=self.foil_extra,
            cost_ub=self.relax_cost_ub,
            output_flag=self.output_flag,
            w=self.w,
        )

        # Set master params once
        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)
        m.Params.OutputFlag = int(self.output_flag)

        self._master_m = m
        self._master_bcap = bcap
        self._master_built = True
        


    def _solve_olb_relax(self, node: BSNode4b, window_size: int, per_bus_neutrality: bool,
                        u_init, p_init, on_t, off_t) -> Tuple[float, Optional[np.ndarray]]:

        m, var, bcap = build_uc_relax_master_varfmax_4b(
            data=self.data,
            idx=self.idx,
            cvec=self.cvec,
            window_size=window_size,
            per_bus_neutrality=per_bus_neutrality,
            u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
            b0=self.b0,
            node_bL=node.bL,
            node_bU=node.bU,
            b_free_idx=self.free,
            foil_extra_constr_fn=self.foil_extra,
            cost_ub=self.relax_cost_ub,
            output_flag=self.output_flag,
            w=self.w,
        )

        if self.master_time_limit is not None:
            m.Params.TimeLimit = float(self.master_time_limit)

        # Ensure names (helps if master does any getConstrByName internally)
        try:
            m.Params.SymbolicNames = 1
        except Exception:
            pass

        m.Params.OutputFlag = int(self.output_flag)

        m.optimize()

        # Diagnostic: no incumbent
        if m.SolCount == 0:
            if self.verbose:
                print(f"[MASTER] Status={m.Status} (no solution) Time={m.Runtime:.2f}s")
            if m.Status == GRB.INFEASIBLE:
                try:
                    m.computeIIS()
                    m.write("master_relax.ilp")
                    if self.verbose:
                        print("[MASTER] Wrote IIS to master_relax.ilp")
                except gp.GurobiError as e:
                    if self.verbose:
                        print(f"[MASTER] IIS failed: {e}")
            return float("inf"), None

        # Accept optimal or time-limit with incumbent
        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return float("inf"), None

        # Extract b*
        b_star = self.b0.copy()
        for ell in self.free:
            b_star[ell] = float(bcap[ell].X)

        return float(m.ObjVal), b_star



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

    def run(self, window_size: int, per_bus_neutrality: bool, u_init, p_init, on_t, off_t):
        # Root node is global bounds
        self._node_id = 0
        root = BSNode4b(id=0, bL=self.bL0.copy(), bU=self.bU0.copy())

        # If you want *provable* optimality: do NOT inject potentially-invalid cost_ub
        # (It can cut off the true optimum b). Leave relax_cost_ub=None.
        # self.relax_cost_ub = None  # optional hard override; you can set this in __init__ too

        # Priority queue elements: (LB_key, tie, node_id, node)
        heap = []
        tie = 0

        root_lb = self.lb_box_L1(root.bL, root.bU)
        heapq.heappush(heap, (root_lb, tie, root.id, root))
        tie += 1

        # Optional: init incumbent from b0 if it is already weak-feasible
        ok0, vp0, vD0 = self.weak_ok(self.b0)
        if ok0:
            Fb0 = self.F(self.b0)
            self.best_F = Fb0
            self.best_b = self.b0.copy()
            self.best_v_plain = vp0
            self.best_v_foil = vD0
            if self.verbose:
                print(f"  [INC] init@b0 F={Fb0:.4f} v={vp0:.3f} vD={vD0:.3f}")

        # Also try a strong reinforcement point once (often finds the first incumbent)
        b_max = self.b0.copy()
        b_max[self.free] = self.bU0[self.free]
        okm, vpm, vDm = self.weak_ok(b_max)
        if okm:
            Fb = self.F(b_max)
            self.best_F = Fb
            self.best_b = b_max.copy()
            self.best_v_plain = vpm
            self.best_v_foil = vDm
            if self.verbose:
                print(f"  [INC] init@b_max F={Fb:.4f} v={vpm:.3f} vD={vDm:.3f}")
            


        nodes_processed = 0

        while heap and nodes_processed < self.max_nodes:
            # Certificate-based termination
            global_LB = heap[0][0]
            if self.best_b is not None and global_LB >= self.best_F - self.eps_obj:
                if self.verbose:
                    print(f"[BS] CERTIFIED: global_LB={global_LB:.6f} >= bestF={self.best_F:.6f} - eps")
                break

            lb_key, _, _, node = heapq.heappop(heap)
            nodes_processed += 1

            # Prune by current LB key
            if self.best_b is not None and lb_key >= self.best_F - self.eps_obj:
                continue

            # If we haven't solved the relaxed master yet, do it now (tightens LB)
            if not np.isfinite(node.olb) or node.b_star_lb is None:
                olb, b_star_lb = self._solve_olb_relax(
                    node, window_size, per_bus_neutrality, u_init, p_init, on_t, off_t
                )
                node.olb = olb
                node.b_star_lb = b_star_lb

                # If master gave no solution, we cannot tighten LB; keep only box-LB.
                if not np.isfinite(olb) or b_star_lb is None:
                    # In optimality-first mode, do NOT prune; we just proceed with box-LB in the queue.
                    # If this happens a lot, increase master_time_limit or remove it entirely.
                    continue

                # Update this node's priority key to a tighter valid LB: max(boxLB, olb)
                new_lb = max(self.lb_box_L1(node.bL, node.bU), float(olb))
                if self.verbose:
                    print(f"[BS] node={node.id:04d} OLB={olb:.4f} LBkey={new_lb:.4f} bestF={self.best_F:.4f}")

                # Reinsert with tightened LB and process later in correct order
                heapq.heappush(heap, (new_lb, tie, node.id, node))
                tie += 1
                continue

            # At this point node has b_star_lb and olb; its true LB key is:
            node_lb = max(self.lb_box_L1(node.bL, node.bU), float(node.olb))

            # Prune by tightened LB
            if self.best_b is not None and node_lb >= self.best_F - self.eps_obj:
                continue

            # 2) OUB verification candidates
            cand = []

            # primary: b from OLB (project to node + global bounds)
            b_try = self._project_to_bounds(node.b_star_lb, node.bL, node.bU)
            b_try = self._project_to_bounds(b_try, self.bL0, self.bU0)
            cand.append(b_try)

            # also try projection of b0 into node
            b_proj = self._project_to_bounds(self.b0, node.bL, node.bU)
            cand.append(b_proj)

            # node center
            cand.append(0.5 * (node.bL + node.bU))
            # aggressive candidate 1: push free lines to node upper bounds (strong reinforcement)
            b_upper = node.bL.copy()
            b_upper[self.free] = node.bU[self.free]
            cand.append(b_upper)

            # aggressive candidate 2 (optional but useful): push ALL lines to upper bound
            # (keeps feasibility if your oracle expects b for all lines)
            cand.append(node.bU.copy())



            # unique candidates
            uniq = []
            seen = set()
            for b in cand:
                key = tuple(np.round(b[self.free], 6))
                if key not in seen:
                    seen.add(key)
                    uniq.append(b)

            for b in uniq:
                ok, vp, vD = self.weak_ok(b)
                if ok:
                    Fb = self.F(b)
                    if Fb < self.best_F - self.eps_obj:
                        self.best_F = Fb
                        self.best_b = b.copy()
                        self.best_v_plain = vp
                        self.best_v_foil = vD
                        if self.verbose:
                            print(f"  [INC] node={node.id:04d} F={Fb:.4f} v={vp:.3f} vD={vD:.3f}")

            # 3) Branch if still splittable
            n1, n2 = self._branch(node)
            if n1 is None:
                continue

            # Push children with box-LB only (valid). Their OLB will be computed lazily on pop.
            lb1 = self.lb_box_L1(n1.bL, n1.bU)
            lb2 = self.lb_box_L1(n2.bL, n2.bU)
            heapq.heappush(heap, (lb1, tie, n1.id, n1)); tie += 1
            heapq.heappush(heap, (lb2, tie, n2.id, n2)); tie += 1

        # final global_LB (for reporting gap)
        final_global_LB = heap[0][0] if heap else (self.best_F if self.best_b is not None else float("inf"))
        gap = (self.best_F - final_global_LB) if (self.best_b is not None and np.isfinite(final_global_LB)) else float("inf")

        return {
            "success": self.best_b is not None,
            "b_hat": self.best_b,
            "F_opt": self.best_F,
            "nodes": nodes_processed,
            "v_plain": self.best_v_plain,
            "v_D": self.best_v_foil,
            "global_LB": final_global_LB,
            "gap": gap,
        }
