# uc_master_relax_4b.py
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from uc_pipeline import build_network_uc_model  # your existing builder


def _relax_tupledict_binary_4b(td: gp.tupledict):
    """In-place relax: set binary vars in a tupledict to continuous in [0,1]."""
    for k, v in td.items():
        v.VType = GRB.CONTINUOUS
        v.LB = 0.0
        v.UB = 1.0


def add_variable_fmax_constraints_4b(
    m: gp.Model,
    var: Dict,
    bcap: Dict[int, gp.Var | float],
    nL: int,
    T: int,
    name_prefix: str = "fmax_var"
):
    """
    Replace/augment line limit constraints with variable RHS:
      -bcap[ell] <= f[ell,t] <= bcap[ell]
    Assumes var["f"] exists and is the line flow tupledict (ell,t) -> var.
    """
    f = var["f"]
    for ell in range(nL):
        cap = bcap[ell]
        for t in range(T):
            # f <= cap
            m.addConstr(f[ell, t] <= cap, name=f"{name_prefix}_ub[{ell},{t}]")
            # -f <= cap  <=>  f >= -cap
            m.addConstr(-f[ell, t] <= cap, name=f"{name_prefix}_lb[{ell},{t}]")


def remove_fixed_fmax_constraints_4b(m: gp.Model):
    """
    Robustly remove fixed line-limit constraints created by build_network_uc_model:
      fmax[ell,t]: f[ell,t] <= const
      fmin[ell,t]: f[ell,t] >= -const

    This avoids getConstrByName(), which can fail when Gurobi doesn't build name indices.
    """
    m.update()
    to_remove = []
    for c in m.getConstrs():
        nm = c.ConstrName  # safe even when name indexing isn't available
        if nm.startswith("fmax[") or nm.startswith("fmin["):
            to_remove.append(c)

    for c in to_remove:
        m.remove(c)

    m.update()
    return len(to_remove)



def build_uc_operating_cost_expr_4b(var, idx, cvec: np.ndarray) -> gp.LinExpr:
    """
    Returns UC operating-cost expression: sum_i cvec[i] * z_i
    using the same index map as the UC model.
    """
    expr = gp.LinExpr()

    def add_block(block_idx, vardict, c_block):
        nEnt, T = block_idx.shape
        for i in range(nEnt):
            for t in range(T):
                coeff = float(c_block[i, t])
                if coeff != 0.0:
                    expr.add(vardict[i, t], coeff)

    add_block(idx.u, var["u"], cvec[idx.u])
    add_block(idx.v, var["v"], cvec[idx.v])
    add_block(idx.w, var["w"], cvec[idx.w])
    add_block(idx.p, var["p"], cvec[idx.p])

    if idx.curt.size > 0:
        add_block(idx.curt, var["curt"], cvec[idx.curt])

    add_block(idx.splus, var["splus"], cvec[idx.splus])
    add_block(idx.sminus, var["sminus"], cvec[idx.sminus])

    # theta,f usually zero-cost
    if np.any(cvec[idx.theta] != 0.0):
        add_block(idx.theta, var["theta"], cvec[idx.theta])
    if np.any(cvec[idx.f] != 0.0):
        add_block(idx.f, var["f"], cvec[idx.f])

    return expr


def build_uc_relax_master_varfmax_4b(
    data,
    idx,
    cvec: np.ndarray,
    window_size: int,
    per_bus_neutrality: bool,
    u_init: np.ndarray,
    p_init: np.ndarray,
    on_time_init: np.ndarray,
    off_time_init: np.ndarray,
    b0: np.ndarray,
    node_bL: np.ndarray,
    node_bU: np.ndarray,
    b_free_idx: List[int],
    foil_extra_constr_fn=None,
    cost_ub: Optional[float] = None,
    output_flag: int = 0,
    w: Optional[np.ndarray] = None,
    lp_relax: bool = True,
):
    """
    Master problem inside one node:
      min ||b - b0||_1 over free lines
      s.t. UC constraints + variable line limits (b) + foil constraints
           optional operational cost upper bound: cvec @ z <= cost_ub

    If lp_relax=True, the commitment binaries u/v/w are relaxed to [0,1].
    If lp_relax=False, u/v/w remain binary and the returned model is a MIP.

    Returns: (model, var, bcap_vars)
    """
    nL = len(data.lines)
    T = int(data.T)

    # 1) Build standard UC model 
    m, var = build_network_uc_model(
        data=data,
        window_size=window_size,
        per_bus_neutrality=per_bus_neutrality,
        u_init=u_init,
        p_init=p_init,
        on_time_init=on_time_init,
        off_time_init=off_time_init,
        output_flag=output_flag,
    )

    # 2) Optional LP relaxation of commitment binaries
    if bool(lp_relax):
        _relax_tupledict_binary_4b(var["u"])
        _relax_tupledict_binary_4b(var["v"])
        _relax_tupledict_binary_4b(var["w"])

    # 3) Remove fixed line limits
    removed = remove_fixed_fmax_constraints_4b(m)
    if output_flag:
        print(f"[MASTER] removed fixed fmax/fmin constraints: {removed}")

    # 4) Create bcap vars
    bcap = {ell: float(b0[ell]) for ell in range(nL)}
    for ell in b_free_idx:
        bcap[ell] = m.addVar(lb=float(node_bL[ell]), ub=float(node_bU[ell]),
                            vtype=GRB.CONTINUOUS, name=f"bcap[{ell}]")

    # 5) Add variable constraints
    add_variable_fmax_constraints_4b(m, var, bcap, nL=nL, T=T)

    # 6) Foil constraints (e.g., curtailment/emissions)
    if foil_extra_constr_fn is not None:
        foil_extra_constr_fn(m, var)

    # 7) Optional: keep relaxation near-optimal by bounding operational cost
    #    This is IMPORTANT; otherwise the relaxation can satisfy the foil with arbitrarily costly dispatch.

    uc_cost_expr = build_uc_operating_cost_expr_4b(var, idx, cvec)

    if cost_ub is not None:
        m.addConstr(uc_cost_expr <= float(cost_ub), name="op_cost_ub")


    # weights (default 1)
    if w is None:
        wvec = np.ones(nL, dtype=float)
    else:
        wvec = np.asarray(w, dtype=float).reshape(-1)
        if wvec.shape[0] != nL:
            raise ValueError(f"w must have length nL={nL}, got {wvec.shape}")


    # 8) L1 objective: minimize ||b - b0||_1 over free lines
    bp = {}
    bm = {}
    for ell in b_free_idx:
        bp[ell] = m.addVar(lb=0.0, name=f"bp[{ell}]")
        bm[ell] = m.addVar(lb=0.0, name=f"bm[{ell}]")
        m.addConstr(bcap[ell] - float(b0[ell]) == bp[ell] - bm[ell], name=f"l1_link[{ell}]")

    m.setObjective(gp.quicksum(float(wvec[ell]) * (bp[ell] + bm[ell]) for ell in b_free_idx), GRB.MINIMIZE)

    m.update()
    return m, var, bcap
