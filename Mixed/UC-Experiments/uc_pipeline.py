# uc_pipeline.py
# Standalone, import-safe UC + DC network + shifting pipeline utilities
# Compatible with your b3_ncxplain.py usage.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# 0) Gurobi license-retry helper
# ============================================================

import time as _time

def _optimize_with_retry(m: gp.Model, max_retries: int = 3, wait: float = 30.0) -> None:
    """
    Call m.optimize() with retry logic for transient Gurobi license errors
    (GurobiError errno 6 — cannot reach token.gurobi.com).

    On a license error, checks whether the model object is still usable
    (m.Status accessible) before retrying. If the model is dead, re-raises.
    """
    for attempt in range(max_retries):
        try:
            m.optimize()
            return
        except gp.GurobiError as e:
            if e.errno != 6 or attempt >= max_retries - 1:
                raise
            # Check whether the model object survived the exception
            try:
                _ = m.Status  # will throw if model is internally corrupted
            except gp.GurobiError:
                raise e  # model is dead — re-raise original error
            print(
                f"[Gurobi] License refresh failed (errno=6), "
                f"retry {attempt + 1}/{max_retries} in {wait:.0f}s ..."
            )
            _time.sleep(wait)


# ============================================================
# 0) Dataclasses (data model)
# ============================================================

@dataclass(frozen=True)
class Line:
    fr: int          # from-bus index (0-based)
    to: int          # to-bus index (0-based)
    b: float         # susceptance (DC approx)
    fmax: float      # flow limit (MW)


@dataclass(frozen=True)
class Generator:
    bus: int
    Pmin: float
    Pmax: float
    RU: float
    RD: float
    UT: int
    DT: int
    SU_cost: float
    SD_cost: float
    no_load_cost: float
    fuel_cost: float
    emission_rate: float = 0.0


@dataclass(frozen=True)
class Renewable:
    bus: int
    avail: np.ndarray      # shape (T,)
    curt_cost: np.ndarray  # shape (T,)


@dataclass(frozen=True)
class NetworkUCData:
    nB: int
    T: int
    demand: np.ndarray              # (nB, T)
    lines: List[Line]
    gens: List[Generator]
    rens: List[Renewable]
    Splus_max: np.ndarray           # (nB, T)
    Sminus_max: np.ndarray          # (nB, T)
    pi_plus: np.ndarray             # (nB, T)
    pi_minus: np.ndarray            # (nB, T)
    carbon_price: float = 0.0
    slack_bus: int = 0
    voll: float = 20000.0  # $/MWh (or 10000–50000)


# ============================================================
# 1) Index map utilities (vectorization order)
# ============================================================

@dataclass(frozen=True)
class IndexMap:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    shed: np.ndarray      # NEW
    curt: np.ndarray
    splus: np.ndarray
    sminus: np.ndarray
    theta: np.ndarray
    f: np.ndarray
    n_vars: int
    block_slices: Dict[str, slice]



def _alloc_2d(start: int, n_ent: int, T: int, major: str) -> Tuple[np.ndarray, int]:
    """
    Allocate a contiguous block of indices for a (n_ent, T) matrix.
    major:
      - "t": time-major within block (t outer, entity inner)
      - "e": entity-major within block (entity outer, t inner)
    """
    size = n_ent * T
    idx = np.arange(start, start + size, dtype=int)
    if major == "t":
        mat = idx.reshape((T, n_ent)).T
    elif major == "e":
        mat = idx.reshape((n_ent, T))
    else:
        raise ValueError("major must be 't' or 'e'")
    return mat, start + size


def build_index_map_network_uc(
    nG: int, nR: int, nB: int, nL: int, T: int, major: str = "t"
) -> IndexMap:
    """
    Block order:
    u, v, w, p, shed, curt, splus, sminus, theta, f
    """
    block_slices: Dict[str, slice] = {}
    s = 0

    u, s2 = _alloc_2d(s, nG, T, major); block_slices["u"] = slice(s, s2); s = s2
    v, s2 = _alloc_2d(s, nG, T, major); block_slices["v"] = slice(s, s2); s = s2
    w, s2 = _alloc_2d(s, nG, T, major); block_slices["w"] = slice(s, s2); s = s2
    p, s2 = _alloc_2d(s, nG, T, major); block_slices["p"] = slice(s, s2); s = s2
    shed, s2 = _alloc_2d(s, nB, T, major); block_slices["shed"] = slice(s, s2); s = s2
    curt, s2 = _alloc_2d(s, nR, T, major); block_slices["curt"] = slice(s, s2); s = s2

    splus, s2 = _alloc_2d(s, nB, T, major); block_slices["splus"] = slice(s, s2); s = s2
    sminus, s2 = _alloc_2d(s, nB, T, major); block_slices["sminus"] = slice(s, s2); s = s2

    theta, s2 = _alloc_2d(s, nB, T, major); block_slices["theta"] = slice(s, s2); s = s2
    f, s2 = _alloc_2d(s, nL, T, major); block_slices["f"] = slice(s, s2); s = s2

    return IndexMap(
    u=u, v=v, w=w, p=p, shed=shed, curt=curt,
    splus=splus, sminus=sminus,
    theta=theta, f=f,
    n_vars=s,
    block_slices=block_slices
)


def mutable_cost_mask_commitment_and_prod(idx, allow_prod=True, allow_commit=True):
    """
    Returns a boolean mask over the flattened cost vector cvec (same shape as cvec),
    where True means that coefficient is allowed to change.

    allow_prod   -> p[g,t] coefficients (marginal production price)
    allow_commit -> u[g,t], su[g,t], sd[g,t] coefficients (commitment price parts)
    """
    mask = np.zeros(int(idx.nZ), dtype=bool)

    if allow_prod and hasattr(idx, "p") and idx.p.size > 0:
        mask[idx.p.ravel()] = True

    if allow_commit:
        if hasattr(idx, "u") and idx.u.size > 0:
            mask[idx.u.ravel()] = True
        if hasattr(idx, "su") and idx.su.size > 0:
            mask[idx.su.ravel()] = True
        if hasattr(idx, "sd") and idx.sd.size > 0:
            mask[idx.sd.ravel()] = True

    return mask


# ============================================================
# 2) Initial conditions helper
# ============================================================

def default_initial_conditions(data: NetworkUCData):
    """
    Minimal initial conditions consistent with your earlier usage:
      - all units initially OFF
      - initial dispatch 0
      - on_time_init 0
      - off_time_init = DT (eligible to start at t=0)
    """
    nG = len(data.gens)
    u_init = np.zeros(nG, dtype=int)
    p_init = np.zeros(nG, dtype=float)
    on_time_init = np.zeros(nG, dtype=int)
    off_time_init = np.array([int(gen.DT) for gen in data.gens], dtype=int)
    return u_init, p_init, on_time_init, off_time_init


# ============================================================
# 3) Cost-vector builder (aligned to IndexMap)
# ============================================================

def build_cost_vector_network_uc(
    idx: IndexMap,
    fuel_cost: np.ndarray,         # (nG,)
    emission_rate: np.ndarray,     # (nG,)
    carbon_price: float,
    no_load_cost: np.ndarray,      # (nG,)
    su_cost: np.ndarray,           # (nG,)
    sd_cost: np.ndarray,           # (nG,)
    curt_cost: np.ndarray,         # (nR,T)
    pi_plus: np.ndarray,           # (nB,T)
    pi_minus: np.ndarray,          # (nB,T)
    voll: float = 20000.0,
) -> np.ndarray:
    c = np.zeros(idx.n_vars, dtype=float)

    c_th = fuel_cost + float(carbon_price) * emission_rate  # (nG,)
    

    c[idx.p] = c_th[:, None]
    c[idx.u] = no_load_cost[:, None]
    c[idx.v] = su_cost[:, None]
    c[idx.w] = sd_cost[:, None]

    # load shedding penalty
    c[idx.shed] = float(voll)

    # renewable curtailment penalty
    if idx.curt.size > 0:
        c[idx.curt] = curt_cost

    # shifting prices
    c[idx.splus] = pi_plus
    c[idx.sminus] = pi_minus

    # theta,f default 0 unless you add penalties
    return c


# ============================================================
# 4) UC model builder (no objective here)
# ============================================================

def build_network_uc_model(
    data: NetworkUCData,
    window_size: int,
    per_bus_neutrality: bool,
    u_init: np.ndarray,
    p_init: np.ndarray,
    on_time_init: np.ndarray,
    off_time_init: np.ndarray,
    output_flag: int = 0,
):
    """
    Builds UC + DC network + shifting model, returns (model, var_dict).
    No objective is set here; set with set_objective_from_cvec(...).
    """
    nG = len(data.gens)
    nR = len(data.rens)
    nL = len(data.lines)
    nB = int(data.nB)
    T = int(data.T)

    m = gp.Model("network_uc")
    m.Params.OutputFlag = int(output_flag)

    # IMPORTANT: B&S needs names for getConstrByName(...)
    # If IgnoreNames=1 anywhere, constraint-name lookup will fail.
    try:
        m.Params.IgnoreNames = 0
    except gp.GurobiError:
        pass

    
    # Variables
    u = m.addVars(nG, T, vtype=GRB.BINARY, name="u")
    v = m.addVars(nG, T, vtype=GRB.BINARY, name="v")
    w = m.addVars(nG, T, vtype=GRB.BINARY, name="w")
    p = m.addVars(nG, T, lb=0.0, vtype=GRB.CONTINUOUS, name="p")

    shed = m.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="shed")
    curt = m.addVars(nR, T, lb=0.0, vtype=GRB.CONTINUOUS, name="curt")
    splus = m.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="splus")
    sminus = m.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="sminus")

    theta = m.addVars(nB, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
    f = m.addVars(nL, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f")

    var = {
        "u": u, "v": v, "w": w, "p": p,
        "shed": shed,                      # NEW
        "curt": curt, "splus": splus, "sminus": sminus,
        "theta": theta, "f": f,
    }


    # -----------------------------
    # Generator constraints
    # -----------------------------
    for g, gen in enumerate(data.gens):
        Pmin, Pmax = float(gen.Pmin), float(gen.Pmax)
        RU, RD = float(gen.RU), float(gen.RD)

        for t in range(T):
            m.addConstr(p[g, t] <= Pmax * u[g, t], name=f"pmax[{g},{t}]")
            m.addConstr(p[g, t] >= Pmin * u[g, t], name=f"pmin[{g},{t}]")
            m.addConstr(v[g, t] + w[g, t] <= 1, name=f"no_simul[{g},{t}]")

        # commitment logic
        m.addConstr(u[g, 0] - int(u_init[g]) == v[g, 0] - w[g, 0], name=f"logic_init[{g}]")
        for t in range(1, T):
            m.addConstr(u[g, t] - u[g, t - 1] == v[g, t] - w[g, t], name=f"logic[{g},{t}]")

        # ramping (with initial dispatch)
        m.addConstr(p[g, 0] - float(p_init[g]) <= RU, name=f"ramp_up_init[{g}]")
        m.addConstr(float(p_init[g]) - p[g, 0] <= RD, name=f"ramp_dn_init[{g}]")
        for t in range(1, T):
            m.addConstr(p[g, t] - p[g, t - 1] <= RU, name=f"ramp_up[{g},{t}]")
            m.addConstr(p[g, t - 1] - p[g, t] <= RD, name=f"ramp_dn[{g},{t}]")

        # carryover min down / min up
        if int(u_init[g]) == 0 and int(gen.DT) > 0:
            remaining_off = max(0, int(gen.DT) - int(off_time_init[g]))
            for t in range(min(T, remaining_off)):
                m.addConstr(u[g, t] == 0, name=f"carry_min_dn[{g},{t}]")

        if int(u_init[g]) == 1 and int(gen.UT) > 0:
            remaining_on = max(0, int(gen.UT) - int(on_time_init[g]))
            for t in range(min(T, remaining_on)):
                m.addConstr(u[g, t] == 1, name=f"carry_min_up[{g},{t}]")

        # standard min up/down (only if UT/DT > 1)
        if int(gen.UT) > 1:
            UT = int(gen.UT)
            for t0 in range(0, T - UT + 1):
                m.addConstr(gp.quicksum(u[g, k] for k in range(t0, t0 + UT)) >= UT * v[g, t0],
                            name=f"min_up[{g},{t0}]")

        if int(gen.DT) > 1:
            DT = int(gen.DT)
            for t0 in range(0, T - DT + 1):
                m.addConstr(gp.quicksum(1 - u[g, k] for k in range(t0, t0 + DT)) >= DT * w[g, t0],
                            name=f"min_dn[{g},{t0}]")

    # -----------------------------
    # Renewables: curtailment bounds
    # -----------------------------
    for r, ren in enumerate(data.rens):
        for t in range(T):
            m.addConstr(curt[r, t] <= float(ren.avail[t]), name=f"curt_ub[{r},{t}]")

    # -----------------------------
    # Shifting bounds
    # -----------------------------
    for b in range(nB):
        for t in range(T):
            m.addConstr(splus[b, t] <= float(data.Splus_max[b, t]), name=f"splus_ub[{b},{t}]")
            m.addConstr(sminus[b, t] <= float(data.Sminus_max[b, t]), name=f"sminus_ub[{b},{t}]")

    # -----------------------------
    # Window neutrality
    # -----------------------------
    W = min(int(window_size), T)
    if per_bus_neutrality:
        for b in range(nB):
            for t0 in range(0, T - W + 1):
                m.addConstr(
                    gp.quicksum(splus[b, t] - sminus[b, t] for t in range(t0, t0 + W)) == 0,
                    name=f"neutral_bus[{b},{t0}]"
                )
    else:
        for t0 in range(0, T - W + 1):
            m.addConstr(
                gp.quicksum(splus[b, t] - sminus[b, t] for b in range(nB) for t in range(t0, t0 + W)) == 0,
                name=f"neutral_sys[{t0}]"
            )

    # -----------------------------
    # DC flow + limits
    # -----------------------------
    for ell, line in enumerate(data.lines):
        fr, to = int(line.fr), int(line.to)
        bb = float(line.b)
        fmax = float(line.fmax)

        for t in range(T):
            m.addConstr(f[ell, t] == bb * (theta[fr, t] - theta[to, t]), name=f"dcflow[{ell},{t}]")
            m.addConstr(f[ell, t] <= fmax, name=f"fmax[{ell},{t}]")
            m.addConstr(f[ell, t] >= -fmax, name=f"fmin[{ell},{t}]")

    # slack bus reference angle
    for t in range(T):
        m.addConstr(theta[int(data.slack_bus), t] == 0.0, name=f"slack[{t}]")

    # -----------------------------
    # Nodal balance
    # -----------------------------
    gens_at_bus = [[] for _ in range(nB)]
    rens_at_bus = [[] for _ in range(nB)]
    in_lines = [[] for _ in range(nB)]
    out_lines = [[] for _ in range(nB)]

    for g, gen in enumerate(data.gens):
        gens_at_bus[int(gen.bus)].append(g)
    for r, ren in enumerate(data.rens):
        rens_at_bus[int(ren.bus)].append(r)
    for ell, line in enumerate(data.lines):
        out_lines[int(line.fr)].append(ell)
        in_lines[int(line.to)].append(ell)

    for b in range(nB):
        for t in range(T):
            gen_inj = gp.quicksum(p[g, t] for g in gens_at_bus[b])

            # renewable injection = avail - curt
            ren_inj = gp.LinExpr()
            for r in rens_at_bus[b]:
                ren_inj += float(data.rens[r].avail[t]) - curt[r, t]

            net_load = float(data.demand[b, t]) + splus[b, t] - sminus[b, t]

            flow_in = gp.quicksum(f[ell, t] for ell in in_lines[b])
            flow_out = gp.quicksum(f[ell, t] for ell in out_lines[b])

            m.addConstr( gen_inj + ren_inj + shed[b, t] - net_load + flow_in - flow_out == 0, name=f"balance[{b},{t}]")
            m.addConstr(shed[b, t] <= float(data.demand[b, t]), name=f"shed_ub[{b},{t}]")

    return m, var


# ============================================================
# 5) Objective setter from c-vector
# ============================================================

def set_objective_from_cvec(m, var, idx: IndexMap, cvec: np.ndarray):
    """
    Objective: minimize cvec @ z, where z is the vectorization given by idx.
    """
    expr = gp.LinExpr()

    def add_block(block_idx: np.ndarray, vardict, c_block: np.ndarray):
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
    add_block(idx.shed, var["shed"], cvec[idx.shed])

    if idx.curt.size > 0:
        add_block(idx.curt, var["curt"], cvec[idx.curt])

    add_block(idx.splus, var["splus"], cvec[idx.splus])
    add_block(idx.sminus, var["sminus"], cvec[idx.sminus])

    # theta and f are typically zero-cost; keep here in case you add penalties later
    if np.any(cvec[idx.theta] != 0.0):
        add_block(idx.theta, var["theta"], cvec[idx.theta])
    if np.any(cvec[idx.f] != 0.0):
        add_block(idx.f, var["f"], cvec[idx.f])

    m.setObjective(expr, GRB.MINIMIZE)


# ============================================================
# 6) Solution extraction and flattening
# ============================================================

def extract_sol_and_z(m, var, idx: IndexMap):
    nG, T = idx.u.shape
    nR, _ = idx.curt.shape
    nB, _ = idx.splus.shape
    nL, _ = idx.f.shape

    sol = {
        "obj": float(m.ObjVal),

        "u": np.array([[var["u"][g, t].X for t in range(T)] for g in range(nG)], dtype=float),
        "v": np.array([[var["v"][g, t].X for t in range(T)] for g in range(nG)], dtype=float),
        "w": np.array([[var["w"][g, t].X for t in range(T)] for g in range(nG)], dtype=float),
        "p": np.array([[var["p"][g, t].X for t in range(T)] for g in range(nG)], dtype=float),
        "shed": np.array([[var["shed"][b, t].X for t in range(T)] for b in range(nB)], dtype=float),
        "curt": np.array([[var["curt"][r, t].X for t in range(T)] for r in range(nR)], dtype=float),

        "splus": np.array([[var["splus"][b, t].X for t in range(T)] for b in range(nB)], dtype=float),
        "sminus": np.array([[var["sminus"][b, t].X for t in range(T)] for b in range(nB)], dtype=float),

        "theta": np.array([[var["theta"][b, t].X for t in range(T)] for b in range(nB)], dtype=float),
        "f": np.array([[var["f"][ell, t].X for t in range(T)] for ell in range(nL)], dtype=float),
    }

    z = np.zeros(idx.n_vars, dtype=float)
    z[idx.u] = sol["u"]
    z[idx.v] = sol["v"]
    z[idx.w] = sol["w"]
    z[idx.p] = sol["p"]
    z[idx.shed] = sol["shed"]
    if nR > 0:
        z[idx.curt] = sol["curt"]
    z[idx.splus] = sol["splus"]
    z[idx.sminus] = sol["sminus"]
    z[idx.theta] = sol["theta"]
    z[idx.f] = sol["f"]

    return sol, z


# ============================================================
# 7) Solve wrapper (SP) and metrics
# ============================================================

def solve_uc_with_cost(
    data: NetworkUCData,
    idx: IndexMap,
    cvec: np.ndarray,
    window_size: int,
    per_bus_neutrality: bool,
    u_init: np.ndarray,
    p_init: np.ndarray,
    on_time_init: np.ndarray,
    off_time_init: np.ndarray,
    extra_constr_fn=None,
    output_flag: int = 0,
    time_limit: Optional[float] = None,
):
    """
    Builds model, applies optional extra constraints, sets objective from cvec, solves.
    Returns (m, sol, z); if not optimal returns (m, None, None).
    """
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

    if extra_constr_fn is not None:
        extra_constr_fn(m, var)

    set_objective_from_cvec(m, var, idx, cvec)

    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    _optimize_with_retry(m)

    if m.Status != GRB.OPTIMAL:
        m.dispose()
        return None, None, None

    sol, z = extract_sol_and_z(m, var, idx)
    m.dispose()
    return None, sol, z


def total_curtailment(sol: Dict) -> float:
    return float(np.sum(sol["curt"]))




# ============================================================
# 4b) UC model builder with RHS-mutable line limits
# ============================================================

from typing import Sequence, Union

FmaxType = Union[float, gp.Var, gp.LinExpr]

def build_network_uc_model_4b(
    data: NetworkUCData,
    window_size: int,
    per_bus_neutrality: bool,
    u_init: np.ndarray,
    p_init: np.ndarray,
    on_time_init: np.ndarray,
    off_time_init: np.ndarray,
    output_flag: int = 0,
    fmax_override: Optional[Sequence[FmaxType]] = None,  # NEW
):
    """
    Same as build_network_uc_model(...), but line limits can be overridden by:
      fmax_override[ell] (float or gurobi var/expr), time-invariant across t.
    If fmax_override is None, uses data.lines[ell].fmax (original behavior).
    """
    nG = len(data.gens)
    nR = len(data.rens)
    nL = len(data.lines)
    nB = int(data.nB)
    T = int(data.T)

    if fmax_override is not None:
        if len(fmax_override) != nL:
            raise ValueError("fmax_override must have length equal to number of lines (nL).")

    m = gp.Model("network_uc_4b")
    m.Params.OutputFlag = int(output_flag)
    try:
        m.Params.IgnoreNames = 0
    except gp.GurobiError:
        pass



    # Variables
    u = m.addVars(nG, T, vtype=GRB.BINARY, name="u")
    v = m.addVars(nG, T, vtype=GRB.BINARY, name="v")
    w = m.addVars(nG, T, vtype=GRB.BINARY, name="w")
    p = m.addVars(nG, T, lb=0.0, vtype=GRB.CONTINUOUS, name="p")
    shed = m.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="shed")
    curt = m.addVars(nR, T, lb=0.0, vtype=GRB.CONTINUOUS, name="curt")
    splus = m.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="splus")
    sminus = m.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="sminus")

    theta = m.addVars(nB, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
    f = m.addVars(nL, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f")

    var = {
        "u": u, "v": v, "w": w, "p": p,
        "shed": shed,                   # NEW
        "curt": curt, "splus": splus, "sminus": sminus,
        "theta": theta, "f": f,
    }

    # -----------------------------
    # Generator constraints
    # -----------------------------
    for g, gen in enumerate(data.gens):
        Pmin, Pmax = float(gen.Pmin), float(gen.Pmax)
        RU, RD = float(gen.RU), float(gen.RD)

        for t in range(T):
            m.addConstr(p[g, t] <= Pmax * u[g, t], name=f"pmax[{g},{t}]")
            m.addConstr(p[g, t] >= Pmin * u[g, t], name=f"pmin[{g},{t}]")
            m.addConstr(v[g, t] + w[g, t] <= 1, name=f"no_simul[{g},{t}]")

        # commitment logic
        m.addConstr(u[g, 0] - int(u_init[g]) == v[g, 0] - w[g, 0], name=f"logic_init[{g}]")
        for t in range(1, T):
            m.addConstr(u[g, t] - u[g, t - 1] == v[g, t] - w[g, t], name=f"logic[{g},{t}]")

        # ramping (with initial dispatch)
        m.addConstr(p[g, 0] - float(p_init[g]) <= RU, name=f"ramp_up_init[{g}]")
        m.addConstr(float(p_init[g]) - p[g, 0] <= RD, name=f"ramp_dn_init[{g}]")
        for t in range(1, T):
            m.addConstr(p[g, t] - p[g, t - 1] <= RU, name=f"ramp_up[{g},{t}]")
            m.addConstr(p[g, t - 1] - p[g, t] <= RD, name=f"ramp_dn[{g},{t}]")

        # carryover min down / min up
        if int(u_init[g]) == 0 and int(gen.DT) > 0:
            remaining_off = max(0, int(gen.DT) - int(off_time_init[g]))
            for t in range(min(T, remaining_off)):
                m.addConstr(u[g, t] == 0, name=f"carry_min_dn[{g},{t}]")

        if int(u_init[g]) == 1 and int(gen.UT) > 0:
            remaining_on = max(0, int(gen.UT) - int(on_time_init[g]))
            for t in range(min(T, remaining_on)):
                m.addConstr(u[g, t] == 1, name=f"carry_min_up[{g},{t}]")

        # standard min up/down (only if UT/DT > 1)
        if int(gen.UT) > 1:
            UT = int(gen.UT)
            for t0 in range(0, T - UT + 1):
                m.addConstr(gp.quicksum(u[g, k] for k in range(t0, t0 + UT)) >= UT * v[g, t0],
                            name=f"min_up[{g},{t0}]")

        if int(gen.DT) > 1:
            DT = int(gen.DT)
            for t0 in range(0, T - DT + 1):
                m.addConstr(gp.quicksum(1 - u[g, k] for k in range(t0, t0 + DT)) >= DT * w[g, t0],
                            name=f"min_dn[{g},{t0}]")

    # -----------------------------
    # Renewables: curtailment bounds
    # -----------------------------
    for r, ren in enumerate(data.rens):
        for t in range(T):
            m.addConstr(curt[r, t] <= float(ren.avail[t]), name=f"curt_ub[{r},{t}]")

    # -----------------------------
    # Shifting bounds
    # -----------------------------
    for b in range(nB):
        for t in range(T):
            m.addConstr(splus[b, t] <= float(data.Splus_max[b, t]), name=f"splus_ub[{b},{t}]")
            m.addConstr(sminus[b, t] <= float(data.Sminus_max[b, t]), name=f"sminus_ub[{b},{t}]")

    # -----------------------------
    # Window neutrality
    # -----------------------------
    W = min(int(window_size), T)
    if per_bus_neutrality:
        for b in range(nB):
            for t0 in range(0, T - W + 1):
                m.addConstr(
                    gp.quicksum(splus[b, t] - sminus[b, t] for t in range(t0, t0 + W)) == 0,
                    name=f"neutral_bus[{b},{t0}]"
                )
    else:
        for t0 in range(0, T - W + 1):
            m.addConstr(
                gp.quicksum(splus[b, t] - sminus[b, t] for b in range(nB) for t in range(t0, t0 + W)) == 0,
                name=f"neutral_sys[{t0}]"
            )

    # -----------------------------
    # DC flow + limits (RHS-mutable via fmax_override)
    # -----------------------------
    for ell, line in enumerate(data.lines):
        fr, to = int(line.fr), int(line.to)
        bb = float(line.b)

        # NEW: either constant original or overridden (float or gurobi expr)
        fmax_expr: FmaxType = (fmax_override[ell] if fmax_override is not None else float(line.fmax))

        for t in range(T):
            m.addConstr(f[ell, t] == bb * (theta[fr, t] - theta[to, t]), name=f"dcflow[{ell},{t}]")
            m.addConstr(f[ell, t] <=  fmax_expr,  name=f"fmax[{ell},{t}]")
            m.addConstr(f[ell, t] >= -fmax_expr,  name=f"fmin[{ell},{t}]")

    # slack bus reference angle
    for t in range(T):
        m.addConstr(theta[int(data.slack_bus), t] == 0.0, name=f"slack[{t}]")

    # -----------------------------
    # Nodal balance
    # -----------------------------
    gens_at_bus = [[] for _ in range(nB)]
    rens_at_bus = [[] for _ in range(nB)]
    in_lines = [[] for _ in range(nB)]
    out_lines = [[] for _ in range(nB)]

    for g, gen in enumerate(data.gens):
        gens_at_bus[int(gen.bus)].append(g)
    for r, ren in enumerate(data.rens):
        rens_at_bus[int(ren.bus)].append(r)
    for ell, line in enumerate(data.lines):
        out_lines[int(line.fr)].append(ell)
        in_lines[int(line.to)].append(ell)

    for b in range(nB):
        for t in range(T):
            gen_inj = gp.quicksum(p[g, t] for g in gens_at_bus[b])

            ren_inj = gp.LinExpr()
            for r in rens_at_bus[b]:
                ren_inj += float(data.rens[r].avail[t]) - curt[r, t]

            net_load = float(data.demand[b, t]) + splus[b, t] - sminus[b, t]

            flow_in = gp.quicksum(f[ell, t] for ell in in_lines[b])
            flow_out = gp.quicksum(f[ell, t] for ell in out_lines[b])

            m.addConstr(gen_inj + ren_inj + shed[b, t] - net_load + flow_in - flow_out == 0, name=f"balance[{b},{t}]")
            m.addConstr(shed[b, t] <= float(data.demand[b, t]), name=f"shed_ub[{b},{t}]")
 

    return m, var


# ============================================================
# 7b) Solve wrapper (SP) with RHS-mutable line limits
# ============================================================

def solve_uc_with_cost_4b(
    data: NetworkUCData,
    idx: IndexMap,
    cvec: np.ndarray,
    window_size: int,
    per_bus_neutrality: bool,
    u_init: np.ndarray,
    p_init: np.ndarray,
    on_time_init: np.ndarray,
    off_time_init: np.ndarray,
    extra_constr_fn=None,
    output_flag: int = 0,
    time_limit: Optional[float] = None,
    fmax_override: Optional[Sequence[FmaxType]] = None,  # NEW
):
    """
    Same as solve_uc_with_cost(...), but uses build_network_uc_model_4b and accepts fmax_override.
    Returns (m, sol, z); if not optimal returns (m, None, None).
    """
    m, var = build_network_uc_model_4b(
        data=data,
        window_size=window_size,
        per_bus_neutrality=per_bus_neutrality,
        u_init=u_init,
        p_init=p_init,
        on_time_init=on_time_init,
        off_time_init=off_time_init,
        output_flag=output_flag,
        fmax_override=fmax_override,
    )

    if extra_constr_fn is not None:
        extra_constr_fn(m, var)

    set_objective_from_cvec(m, var, idx, cvec)

    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    _optimize_with_retry(m)

    if m.Status != GRB.OPTIMAL:
        m.dispose()
        return None, None, None

    sol, z = extract_sol_and_z(m, var, idx)
    m.dispose()
    return None, sol, z

def make_curtailment_foil_4b(data: NetworkUCData, alpha: float, C_factual: float):
    rhs = (1.0 - float(alpha)) * float(C_factual)
    nR = len(data.rens); T = int(data.T)

    def extra_constr_fn(m, var):
        m.addConstr(
            gp.quicksum(var["curt"][r, t] for r in range(nR) for t in range(T)) <= rhs,
            name="foil_curtailment"
        )
    return extra_constr_fn


def make_emissions_foil_4b(data: NetworkUCData, alpha: float, E_factual: float):
    rhs = (1.0 - float(alpha)) * float(E_factual)
    nG = len(data.gens); T = int(data.T)
    e = np.array([float(g.emission_rate) for g in data.gens], dtype=float)

    def extra_constr_fn(m, var):
        m.addConstr(
            gp.quicksum(e[g] * var["p"][g, t] for g in range(nG) for t in range(T)) <= rhs,
            name="foil_emissions"
        )
    return extra_constr_fn
