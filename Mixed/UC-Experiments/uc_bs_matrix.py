# uc_bs_matrix.py
"""
Matrix-based Branch-and-Sandwich solver for UC line-capacity WCE.

Key design decisions vs. the original uc_branch_sandwich_4b.py:
  - Constraints are extracted ONCE from a Gurobi model via extract_uc_matrices()
    and stored as dense numpy arrays (A_norm, rhs_norm).
  - All constraint directions are normalised to  A x <= rhs  at extraction time.
  - Line-limit rows are identified by constraint name and stored in
    fmax_row_idx / fmin_row_idx  (shape nL x T), so that the same scalar
    variable fmax_var[ell] can drive all 2T rows for that line.
  - The mutable quantity is one scalar per FREE LINE  (same semantics as b0/b_hat
    in the original solver).
  - OUB uses the multi-candidate strategy from the original solver.
"""

from __future__ import annotations

import re
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# 1.  Matrix extraction
# ============================================================

def extract_uc_matrices(
    model: gp.Model,
    nL: int,
    T: int,
) -> dict:
    """
    Extract the LP/MIP matrix from a *solved or updated* Gurobi UC model.

    All constraints are normalised to  A x <= rhs:
        '<' rows  →  kept as-is
        '>' rows  →  row and rhs multiplied by -1
        '=' rows  →  split into two rows  (A x <= rhs)  and  (-A x <= -rhs)

    Returns a dict with keys:
        c            np.ndarray  (n,)          objective coefficients
        A_norm       np.ndarray  (m_norm, n)   normalised constraint matrix
        rhs_norm     np.ndarray  (m_norm,)     normalised RHS
        row_names    list[str]   (m_norm,)     name for each normalised row
        lb           np.ndarray  (n,)          variable lower bounds
        ub           np.ndarray  (n,)          variable upper bounds
        var_names    list[str]   (n,)
        fmax_row_idx np.ndarray  (nL, T) int   row in A_norm for  f[ell,t] <= fmax
        fmin_row_idx np.ndarray  (nL, T) int   row in A_norm for -f[ell,t] <= fmax
        integer_vars list[int]              variable indices that are integer/binary
    """
    model.update()
    gvars   = model.getVars()
    gconstrs = model.getConstrs()
    n = len(gvars)

    c        = np.array([v.Obj for v in gvars],  dtype=float)
    lb       = np.array([v.LB  for v in gvars],  dtype=float)
    ub       = np.array([v.UB  for v in gvars],  dtype=float)
    var_names = [v.VarName for v in gvars]
    integer_vars = [
        i for i, v in enumerate(gvars)
        if v.VType in (GRB.BINARY, GRB.INTEGER)
    ]

    rows_A    = []
    rows_rhs  = []
    row_names = []

    for con in gconstrs:
        row_expr = model.getRow(con)
        a = np.zeros(n, dtype=float)
        for j in range(row_expr.size()):
            a[row_expr.getVar(j).index] = row_expr.getCoeff(j)
        rhs   = float(con.RHS)
        sense = con.Sense        # '<', '>', '='
        name  = con.ConstrName

        if sense == '<':
            rows_A.append(a);   rows_rhs.append(rhs);   row_names.append(name)
        elif sense == '>':
            rows_A.append(-a);  rows_rhs.append(-rhs);  row_names.append(name + "__flipped")
        elif sense == '=':
            rows_A.append(a);   rows_rhs.append(rhs);   row_names.append(name + "__eq_le")
            rows_A.append(-a);  rows_rhs.append(-rhs);  row_names.append(name + "__eq_ge")

    A_norm   = np.vstack(rows_A).astype(float)
    rhs_norm = np.array(rows_rhs, dtype=float)

    # Identify line-limit rows by constraint name
    # Original model uses:  fmax[ell,t]  and  fmin[ell,t]
    fmax_row_idx = np.full((nL, T), -1, dtype=int)
    fmin_row_idx = np.full((nL, T), -1, dtype=int)

    pat_fmax = re.compile(r"^fmax\[(\d+),(\d+)\]$")
    pat_fmin = re.compile(r"^fmin\[(\d+),(\d+)\]__flipped$")

    for i, nm in enumerate(row_names):
        m = pat_fmax.match(nm)
        if m:
            ell, t = int(m.group(1)), int(m.group(2))
            if 0 <= ell < nL and 0 <= t < T:
                fmax_row_idx[ell, t] = i
            continue
        m = pat_fmin.match(nm)
        if m:
            ell, t = int(m.group(1)), int(m.group(2))
            if 0 <= ell < nL and 0 <= t < T:
                fmin_row_idx[ell, t] = i

    missing_fmax = np.sum(fmax_row_idx < 0)
    missing_fmin = np.sum(fmin_row_idx < 0)
    if missing_fmax > 0 or missing_fmin > 0:
        raise ValueError(
            f"extract_uc_matrices: could not find all line-limit rows. "
            f"Missing fmax entries: {missing_fmax}, fmin entries: {missing_fmin}. "
            "Check that the model was built with IgnoreNames=0."
        )

    return dict(
        c=c,
        A_norm=A_norm,
        rhs_norm=rhs_norm,
        row_names=row_names,
        lb=lb,
        ub=ub,
        var_names=var_names,
        fmax_row_idx=fmax_row_idx,
        fmin_row_idx=fmin_row_idx,
        integer_vars=integer_vars,
        n=n,
        m_norm=len(rows_rhs),
        nL=nL,
        T=T,
    )


def get_free_line_rows(mat: dict, free_line_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the sets of row indices in A_norm that belong to free lines.
      fmax_rows  shape (len(free_line_idx) * T,)
      fmin_rows  shape (len(free_line_idx) * T,)
    """
    fmax_row_idx = mat["fmax_row_idx"]
    fmin_row_idx = mat["fmin_row_idx"]
    T  = mat["T"]
    fmax_rows = np.array(
        [fmax_row_idx[ell, t] for ell in free_line_idx for t in range(T)], dtype=int
    )
    fmin_rows = np.array(
        [fmin_row_idx[ell, t] for ell in free_line_idx for t in range(T)], dtype=int
    )
    return fmax_rows, fmin_rows


def fixed_row_mask(mat: dict, free_line_idx: List[int]) -> np.ndarray:
    """Boolean mask over rows: True for rows whose RHS is NOT a free-line fmax."""
    m_norm = mat["m_norm"]
    fmax_rows, fmin_rows = get_free_line_rows(mat, free_line_idx)
    free_row_set = set(fmax_rows.tolist()) | set(fmin_rows.tolist())
    mask = np.ones(m_norm, dtype=bool)
    for r in free_row_set:
        mask[r] = False
    return mask


# ============================================================
# 2.  Node
# ============================================================

@dataclass
class BSMatNode:
    id: int
    bL: np.ndarray          # lower bounds for fmax[ell], shape (nL,)
    bU: np.ndarray          # upper bounds for fmax[ell], shape (nL,)
    olb: float = np.inf
    b_star_lb: Optional[np.ndarray] = None
    inner_lb: float = np.inf
    inner_ub: float = np.inf
    oub_failures: int = 0
    parent_inner_ub: float = np.inf
    status: str = "open"    # "open" | "fathomed"


# ============================================================
# 3.  Low-level LP/MIP builders from matrices
# ============================================================

def _build_model_from_matrix(
    mat: dict,
    free_line_idx: List[int],
    b_vals: np.ndarray,          # scalar fmax per LINE (shape nL,); free lines use b_vals[ell]
    relax_binaries: bool = False,
    foil_constr_fn: Optional[Callable] = None,  # fn(m, x_vars, var_names) -> None
    cost_ub: Optional[float] = None,
    fmax_as_vars: bool = False,   # if True, free lines get Gurobi vars for fmax
    bL_node: Optional[np.ndarray] = None,  # required when fmax_as_vars=True
    bU_node: Optional[np.ndarray] = None,
    output_flag: int = 0,
    model_name: str = "uc_mat",
    time_limit: Optional[float] = None,
) -> Tuple[gp.Model, list, Optional[dict]]:
    """
    Build a Gurobi model from the extracted matrix.

    Returns (model, x_vars, fmax_vars_dict_or_None).
    fmax_vars_dict maps ell -> gurobi Var when fmax_as_vars=True.
    """
    n      = mat["n"]
    A_norm = mat["A_norm"]
    rhs_n  = mat["rhs_norm"]
    c      = mat["c"]
    lb     = mat["lb"]
    ub     = mat["ub"]
    fmax_ri = mat["fmax_row_idx"]
    fmin_ri = mat["fmin_row_idx"]
    T      = mat["T"]

    m = gp.Model(model_name)
    m.Params.OutputFlag = int(output_flag)
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    # --- decision variables ---
    x_vars = []
    for i in range(n):
        vtype = GRB.CONTINUOUS
        if not relax_binaries and i in set(mat["integer_vars"]):
            vtype = GRB.BINARY
        elif relax_binaries and i in set(mat["integer_vars"]):
            vtype = GRB.CONTINUOUS  # relax
        xi = m.addVar(
            lb=float(lb[i]),
            ub=1.0 if (relax_binaries and i in set(mat["integer_vars"])) else float(ub[i]),
            vtype=vtype,
            name=mat["var_names"][i],
        )
        x_vars.append(xi)

    # --- fmax variables for free lines (if requested) ---
    fmax_vars = None
    if fmax_as_vars:
        assert bL_node is not None and bU_node is not None
        fmax_vars = {}
        for ell in free_line_idx:
            fmax_vars[ell] = m.addVar(
                lb=float(bL_node[ell]),
                ub=float(bU_node[ell]),
                vtype=GRB.CONTINUOUS,
                name=f"fmax_var[{ell}]",
            )

    # Precompute which rows are "free-line rows" and map row -> (ell, sign)
    free_row_to_ell: Dict[int, int] = {}
    for ell in free_line_idx:
        for t in range(T):
            free_row_to_ell[int(fmax_ri[ell, t])] = ell
            free_row_to_ell[int(fmin_ri[ell, t])] = ell

    # --- constraints ---
    for j in range(mat["m_norm"]):
        lhs = gp.quicksum(float(A_norm[j, i]) * x_vars[i] for i in range(n) if A_norm[j, i] != 0.0)

        if j in free_row_to_ell:
            ell = free_row_to_ell[j]
            if fmax_as_vars:
                # A x <= fmax_var[ell]
                m.addConstr(lhs <= fmax_vars[ell], name=f"mat_row[{j}]")
            else:
                # A x <= b_vals[ell]  (fixed value)
                m.addConstr(lhs <= float(b_vals[ell]), name=f"mat_row[{j}]")
        else:
            m.addConstr(lhs <= float(rhs_n[j]), name=f"mat_row[{j}]")

    # --- optional foil constraints ---
    if foil_constr_fn is not None:
        foil_constr_fn(m, x_vars, mat["var_names"])

    # --- optional cost upper bound ---
    if cost_ub is not None:
        m.addConstr(
            gp.quicksum(float(c[i]) * x_vars[i] for i in range(n) if c[i] != 0.0)
            <= float(cost_ub),
            name="cost_ub",
        )

    return m, x_vars, fmax_vars


# ============================================================
# 4.  Oracle (plain + foil solves with caching)
# ============================================================

class MatrixUCOracle:
    """
    Replaces UCWeakWCEOracle for the matrix-based solver.
    Caches results keyed on (rounded) b_line vector.
    """

    def __init__(
        self,
        mat: dict,
        free_line_idx: List[int],
        foil_constr_fn: Optional[Callable] = None,
        cache_decimals: int = 3,
        output_flag: int = 0,
        time_limit: Optional[float] = None,
        eps_weak: float = 1e-3,
    ):
        self.mat            = mat
        self.free_line_idx  = free_line_idx
        self.foil_fn        = foil_constr_fn
        self.cache_decimals = cache_decimals
        self.output_flag    = output_flag
        self.time_limit     = time_limit
        self.eps_weak       = eps_weak
        self._cache_plain: dict = {}
        self._cache_foil:  dict = {}

    def _key(self, b: np.ndarray) -> tuple:
        return tuple(np.round(np.asarray(b, float), self.cache_decimals))

    def _solve(self, b: np.ndarray, with_foil: bool) -> Tuple[Optional[float], Optional[np.ndarray]]:
        m, xv, _ = _build_model_from_matrix(
            mat=self.mat,
            free_line_idx=self.free_line_idx,
            b_vals=b,
            relax_binaries=False,
            foil_constr_fn=self.foil_fn if with_foil else None,
            fmax_as_vars=False,
            output_flag=self.output_flag,
            model_name="oracle_foil" if with_foil else "oracle_plain",
            time_limit=self.time_limit,
        )
        c = self.mat["c"]
        m.setObjective(
            gp.quicksum(float(c[i]) * xv[i] for i in range(self.mat["n"]) if c[i] != 0.0),
            GRB.MINIMIZE,
        )
        m.optimize()
        if m.SolCount == 0:
            return None, None
        x_sol = np.array([v.X for v in xv], dtype=float)
        return float(m.ObjVal), x_sol

    def solve_plain(self, b: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        key = self._key(b)
        if key not in self._cache_plain:
            self._cache_plain[key] = self._solve(b, with_foil=False)
        return self._cache_plain[key]

    def solve_foil(self, b: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        key = self._key(b)
        if key not in self._cache_foil:
            self._cache_foil[key] = self._solve(b, with_foil=True)
        return self._cache_foil[key]

    def weak_ok(self, b: np.ndarray) -> Tuple[bool, Optional[float], Optional[float]]:
        v_plain, _ = self.solve_plain(b)
        v_foil,  _ = self.solve_foil(b)
        if v_plain is None or v_foil is None:
            return False, v_plain, v_foil
        return (v_foil <= v_plain + self.eps_weak), v_plain, v_foil


# ============================================================
# 5.  Branch-and-Sandwich (matrix version)
# ============================================================

class UCBranchAndSandwichMatrix:
    """
    Branch-and-Sandwich WCE solver using extracted LP/MIP matrices.

    Objective:  min  sum_{ell in free_line_idx}  w[ell] * |fmax[ell] - b0[ell]|
    Subject to: existence of x satisfying UC constraints with foil + new fmax values.

    Parameters mirror UCBranchAndSandwichWCE_4b where possible.
    """

    def __init__(
        self,
        mat: dict,                          # output of extract_uc_matrices()
        oracle: MatrixUCOracle,
        b0: np.ndarray,                     # original line limits, shape (nL,)
        bL0: np.ndarray,                    # global lower bounds on fmax
        bU0: np.ndarray,                    # global upper bounds on fmax
        free_line_idx: List[int],           # lines allowed to change
        foil_constr_fn: Optional[Callable], # fn(m, x_vars, var_names)
        w: Optional[np.ndarray] = None,     # weights per line
        eps_b: float = 5.0,
        eps_obj: float = 1e-3,
        eps_weak: float = 1e-3,
        max_nodes: int = 500,
        master_time_limit: Optional[float] = None,
        output_flag: int = 0,
        verbose: bool = True,
        oub_grid_pts: int = 3,
        cost_ub_factor: Optional[float] = None,  # if set, add c^T x <= factor * inner_ub
    ):
        self.mat       = mat
        self.oracle    = oracle
        self.b0        = np.array(b0,  dtype=float)
        self.bL0       = np.array(bL0, dtype=float)
        self.bU0       = np.array(bU0, dtype=float)
        self.free      = list(free_line_idx)
        self.foil_fn   = foil_constr_fn

        nL = mat["nL"]
        self.w = np.ones(nL, dtype=float) if w is None else np.asarray(w, dtype=float).reshape(-1)

        self.eps_b    = float(eps_b)
        self.eps_obj  = float(eps_obj)
        self.eps_weak = float(eps_weak)
        self.max_nodes = int(max_nodes)
        self.master_time_limit = master_time_limit
        self.output_flag = int(output_flag)
        self.verbose  = bool(verbose)
        self.oub_grid_pts = int(oub_grid_pts)
        self.cost_ub_factor = cost_ub_factor

        # Incumbents
        self.best_F       = float("inf")
        self.best_b       = None
        self.best_v_plain = None
        self.best_v_foil  = None
        self.best_inner_ub = float("inf")
        self._node_id     = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def F(self, b: np.ndarray) -> float:
        return float(np.sum(self.w[self.free] * np.abs(b[self.free] - self.b0[self.free])))

    def lb_box_L1(self, bL: np.ndarray, bU: np.ndarray) -> float:
        """Geometric lower bound on F from the box [bL, bU]."""
        lb = 0.0
        for ell in self.free:
            wj = float(self.w[ell])
            lo = bL[ell] - self.b0[ell]
            hi = bU[ell] - self.b0[ell]
            if lo > 0:
                lb += wj * lo
            elif hi < 0:
                lb += wj * abs(hi)
        return lb

    # ------------------------------------------------------------------
    # OLB: solve relaxed master with variable fmax, foil constraints
    # ------------------------------------------------------------------

    def _solve_olb(
        self, node: BSMatNode, cost_ub: Optional[float]
    ) -> Tuple[float, Optional[np.ndarray]]:

        m, x_vars, fmax_vars = _build_model_from_matrix(
            mat=self.mat,
            free_line_idx=self.free,
            b_vals=self.b0,           # fixed rows use b0
            relax_binaries=True,      # LP relaxation
            foil_constr_fn=self.foil_fn,
            cost_ub=cost_ub,
            fmax_as_vars=True,
            bL_node=node.bL,
            bU_node=node.bU,
            output_flag=self.output_flag,
            model_name="OLB",
            time_limit=self.master_time_limit,
        )

        # L1 objective over free lines
        c_obj = self.mat["c"]
        bp = {ell: m.addVar(lb=0.0, name=f"bp[{ell}]") for ell in self.free}
        bm = {ell: m.addVar(lb=0.0, name=f"bm[{ell}]") for ell in self.free}
        for ell in self.free:
            m.addConstr(fmax_vars[ell] - float(self.b0[ell]) == bp[ell] - bm[ell])

        m.setObjective(
            gp.quicksum(float(self.w[ell]) * (bp[ell] + bm[ell]) for ell in self.free),
            GRB.MINIMIZE,
        )
        m.optimize()

        if m.SolCount == 0:
            if m.Status == GRB.INFEASIBLE and self.verbose:
                print(f"    [OLB] Infeasible node={node.id}")
                try:
                    m.computeIIS(); m.write("olb_infeas.ilp")
                except Exception:
                    pass
            return float("inf"), None

        b_star = self.b0.copy()
        for ell in self.free:
            b_star[ell] = float(fmax_vars[ell].X)
        return float(m.ObjVal), b_star

    # ------------------------------------------------------------------
    # Inner bounds
    # ------------------------------------------------------------------

    def _compute_inner_lb(self, node: BSMatNode) -> float:
        """min c^T x  s.t. LP-relaxation, fmax in [bL, bU] (no foil)."""
        m, x_vars, fmax_vars = _build_model_from_matrix(
            mat=self.mat,
            free_line_idx=self.free,
            b_vals=self.b0,
            relax_binaries=True,
            foil_constr_fn=None,
            fmax_as_vars=True,
            bL_node=node.bL,
            bU_node=node.bU,
            output_flag=self.output_flag,
            model_name="ILB",
            time_limit=self.master_time_limit,
        )
        c = self.mat["c"]
        m.setObjective(
            gp.quicksum(float(c[i]) * x_vars[i] for i in range(self.mat["n"]) if c[i] != 0.0),
            GRB.MINIMIZE,
        )
        m.optimize()
        if m.SolCount == 0:
            return float("inf")
        return float(m.ObjVal)

    def _compute_inner_ub(self, node: BSMatNode) -> float:
        """Solve plain MIP at b_worst = bL (worst-case line capacities)."""
        b_worst = node.bL.copy()
        v_plain, _ = self.oracle.solve_plain(b_worst)
        return float(v_plain) if v_plain is not None else float("inf")

    # ------------------------------------------------------------------
    # OUB: multi-candidate strategy
    # ------------------------------------------------------------------

    def _oub_candidates(self, node: BSMatNode) -> List[np.ndarray]:
        def proj(b):
            return np.minimum(np.maximum(b, node.bL), node.bU)

        cands = []
        if node.b_star_lb is not None:
            cands.append(proj(node.b_star_lb))
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

        # Deduplicate
        uniq, seen = [], set()
        for b in cands:
            key = tuple(np.round(b[self.free], 6))
            if key not in seen:
                seen.add(key)
                uniq.append(b)
        return uniq

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def _branch(self, node: BSMatNode) -> Tuple[Optional[BSMatNode], Optional[BSMatNode]]:
        widths = [(ell, node.bU[ell] - node.bL[ell]) for ell in self.free]
        ell_max, w_max = max(widths, key=lambda x: x[1])
        if w_max <= self.eps_b:
            return None, None
        mid = 0.5 * (node.bL[ell_max] + node.bU[ell_max])

        bL1, bU1 = node.bL.copy(), node.bU.copy(); bU1[ell_max] = mid
        bL2, bU2 = node.bL.copy(), node.bU.copy(); bL2[ell_max] = mid

        n1 = BSMatNode(id=self._node_id + 1, bL=bL1, bU=bU1, parent_inner_ub=node.inner_ub)
        n2 = BSMatNode(id=self._node_id + 2, bL=bL2, bU=bU2, parent_inner_ub=node.inner_ub)
        self._node_id += 2
        return n1, n2

    # ------------------------------------------------------------------
    # Node initialisation
    # ------------------------------------------------------------------

    def _init_node(self, node: BSMatNode) -> None:
        node.inner_lb = self._compute_inner_lb(node)
        node.inner_ub = self._compute_inner_ub(node)

        if node.inner_ub < self.best_inner_ub:
            self.best_inner_ub = node.inner_ub
            if self.verbose:
                print(f"    [INNER_UB] global best_inner_ub={self.best_inner_ub:.4f}")

        # Inner fathoming
        if node.inner_lb > self.best_inner_ub + self.eps_obj:
            if self.verbose:
                print(f"    [FATHOM-INNER] node={node.id}")
            node.olb = float("inf")
            return

        cost_ub = min(
            self.best_inner_ub,
            node.inner_ub        if np.isfinite(node.inner_ub)        else float("inf"),
            node.parent_inner_ub if np.isfinite(node.parent_inner_ub) else float("inf"),
        )
        cost_ub = cost_ub if np.isfinite(cost_ub) else None

        olb, b_star = self._solve_olb(node, cost_ub=cost_ub)
        node.olb       = olb
        node.b_star_lb = b_star

    # ------------------------------------------------------------------
    # Incumbent update
    # ------------------------------------------------------------------

    def _update_incumbent(self, b: np.ndarray, Fb: float, vp: float, vD: float, label: str = "") -> None:
        if Fb < self.best_F - self.eps_obj:
            self.best_F       = Fb
            self.best_b       = b.copy()
            self.best_v_plain = vp
            self.best_v_foil  = vD
            if self.verbose:
                print(f"  [INC] {label}: F={Fb:.4f}  v_plain={vp:.3f}  v_foil={vD:.3f}")

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        self._node_id    = 0
        self.best_F      = float("inf")
        self.best_b      = None
        self.best_v_plain = None
        self.best_v_foil  = None
        self.best_inner_ub = float("inf")

        # Warm-start: check b0 and b_max
        for b_try, label in [
            (self.b0.copy(), "b0"),
            ({**{}, **{ell: self.bU0[ell] for ell in self.free}}, "b_max"),
        ]:
            if isinstance(b_try, dict):
                b_vec = self.b0.copy()
                for ell, val in b_try.items():
                    b_vec[ell] = val
                b_try = b_vec
            ok, vp, vD = self.oracle.weak_ok(b_try)
            if ok:
                self._update_incumbent(b_try, self.F(b_try), vp, vD, label)

        root = BSMatNode(id=0, bL=self.bL0.copy(), bU=self.bU0.copy())
        self._init_node(root)

        heap: list = []
        tie = 0
        open_node_lbs: Dict[int, float] = {}

        if np.isfinite(root.olb):
            root_lb = max(self.lb_box_L1(root.bL, root.bU), root.olb)
            heapq.heappush(heap, (root_lb, tie, root.id, root))
            open_node_lbs[root.id] = root_lb
            tie += 1

        nodes_processed = 0

        while heap and nodes_processed < self.max_nodes:
            global_LB = min(open_node_lbs.values()) if open_node_lbs else float("inf")
            if self.best_b is not None:
                global_LB = min(global_LB, self.best_F)

            if self.best_b is not None and global_LB >= self.best_F - self.eps_obj:
                if self.verbose:
                    print(f"[BS-MAT] CERTIFIED: gLB={global_LB:.6f} >= bestF={self.best_F:.6f}")
                break

            lb_key, _, _, node = heapq.heappop(heap)
            open_node_lbs.pop(node.id, None)
            nodes_processed += 1

            # Prune by incumbent
            if self.best_b is not None and lb_key >= self.best_F - self.eps_obj:
                continue

            node_lb = max(self.lb_box_L1(node.bL, node.bU), float(node.olb))

            if self.verbose:
                gLB = min(open_node_lbs.values()) if open_node_lbs else node_lb
                gLB = min(gLB, self.best_F) if self.best_b is not None else gLB
                gap_pct = (self.best_F - gLB) / max(abs(self.best_F), 1e-9) * 100 if np.isfinite(self.best_F) else float("nan")
                print(
                    f"[BS-MAT] node={node.id:04d}  olb={node.olb:.4f}  "
                    f"inner=[{node.inner_lb:.2f},{node.inner_ub:.2f}]  "
                    f"bestF={self.best_F:.4f}  gLB={gLB:.4f}  gap={gap_pct:.1f}%"
                )

            # Inner fathoming
            if node.inner_lb > self.best_inner_ub + self.eps_obj:
                if self.verbose:
                    print(f"  [FATHOM-INNER] node={node.id}")
                continue

            # OUB: try all candidates
            any_ok = False
            for b_cand in self._oub_candidates(node):
                ok, vp, vD = self.oracle.weak_ok(b_cand)
                if ok:
                    any_ok = True
                    self._update_incumbent(b_cand, self.F(b_cand), vp, vD, f"node={node.id:04d}")

            if not any_ok:
                node.oub_failures += 1
            else:
                node.oub_failures = 0

            # Structural fathoming (degenerate node)
            if (
                node.oub_failures >= 2
                and node_lb <= self.eps_obj
                and node.inner_lb >= self.best_F - self.eps_obj
            ):
                if self.verbose:
                    print(f"  [FATHOM-STRUCT] node={node.id:04d}")
                continue

            # Branch
            n1, n2 = self._branch(node)
            if n1 is None:
                continue

            for child in (n1, n2):
                self._init_node(child)
                if not np.isfinite(child.olb):
                    continue
                child_lb = max(self.lb_box_L1(child.bL, child.bU), child.olb)
                if self.best_b is not None:
                    child_lb = min(child_lb, self.best_F)
                if self.best_b is not None and child_lb >= self.best_F - self.eps_obj:
                    continue
                heapq.heappush(heap, (child_lb, tie, child.id, child))
                open_node_lbs[child.id] = child_lb
                tie += 1

        global_LB = min(open_node_lbs.values()) if open_node_lbs else float("inf")
        if self.best_b is not None:
            global_LB = min(global_LB, self.best_F)
        gap = (
            float(self.best_F - global_LB)
            if (self.best_b is not None and np.isfinite(global_LB))
            else float("inf")
        )

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
