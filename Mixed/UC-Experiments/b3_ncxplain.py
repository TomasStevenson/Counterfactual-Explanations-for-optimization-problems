#b3_ncxplain.py
import importlib, sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from uc_pipeline import (
    build_index_map_network_uc,
    default_initial_conditions,
    build_cost_vector_network_uc,
    solve_uc_with_cost,
    total_curtailment,
    )


def _compute_dTx_uc(
    *,
    c0_fixed: np.ndarray,
    x_fixed: np.ndarray,
    # optional blocks (pass only those you use)
    pi_plus: np.ndarray | None = None,
    pi_minus: np.ndarray | None = None,
    x_splus: np.ndarray | None = None,
    x_sminus: np.ndarray | None = None,
    curt_cost: np.ndarray | None = None,
    x_curt: np.ndarray | None = None,
    fuel_cost: np.ndarray | None = None,
    no_load_cost: np.ndarray | None = None,
    su_cost: np.ndarray | None = None,
    sd_cost: np.ndarray | None = None,
    x_p: np.ndarray | None = None,
    x_u: np.ndarray | None = None,
    x_su: np.ndarray | None = None,
    x_sd: np.ndarray | None = None,
) -> float:
    """
    Compute d^T x = c_new^T x, where:
      - c0_fixed and x_fixed capture all non-mutable indices
      - mutable blocks are added only if provided

    Shapes expected:
      pi_plus, pi_minus, x_splus, x_sminus: (nB, T)
      curt_cost, x_curt: (nR, T)
      fuel_cost: (nG,), x_p: (nG, T)
      no_load_cost: (nG,), x_u: (nG, T)
      su_cost, sd_cost: (nG,), x_su/x_sd: (nG, T)
    """
    val = float(np.dot(c0_fixed, x_fixed))

    # pi blocks
    if (pi_plus is not None) and (x_splus is not None):
        val += float(np.sum(pi_plus * x_splus))
    if (pi_minus is not None) and (x_sminus is not None):
        val += float(np.sum(pi_minus * x_sminus))

    # curt block
    if (curt_cost is not None) and (x_curt is not None):
        val += float(np.sum(curt_cost * x_curt))

    # gen cost blocks
    if (fuel_cost is not None) and (x_p is not None):
        val += float(np.sum(fuel_cost[:, None] * x_p))
    if (no_load_cost is not None) and (x_u is not None):
        val += float(np.sum(no_load_cost[:, None] * x_u))
    if (su_cost is not None) and (x_su is not None):
        val += float(np.sum(su_cost[:, None] * x_su))
    if (sd_cost is not None) and (x_sd is not None):
        val += float(np.sum(sd_cost[:, None] * x_sd))

    return float(val)


import gurobipy as gp
from gurobipy import GRB

def _get_var(var, keys):
    """
    Robustly fetch a tupledict from `var` using multiple possible names.
    Example: _get_var(var, ["u","U","on","commit"]) returns var["u"] or var.u or similar.
    """
    # dict-style
    if isinstance(var, dict):
        for k in keys:
            if k in var:
                return var[k]

    # attribute-style
    for k in keys:
        if hasattr(var, k):
            return getattr(var, k)

    # last resort: scan dict keys if provided as tupledicts directly
    if isinstance(var, dict):
        avail = list(var.keys())
    else:
        avail = [a for a in dir(var) if not a.startswith("_")]

    raise KeyError(f"Could not find variable among {keys}. Available: {avail[:50]}")

def _idx_nvars(idx, cvec=None) -> int:
    """
    Return number of decision variables in z / cvec.
    Supports different IndexMap implementations.
    """
    if hasattr(idx, "n_vars"):
        return int(idx.n_vars)
    if hasattr(idx, "nZ"):
        return int(idx.nZ)
    if cvec is not None:
        return int(len(cvec))
    raise AttributeError("IndexMap has no n_vars/nZ and no cvec was provided.")


#========================================================
#-------------------- FOILS-----------------------------
#========================================================


def foil_force_on(g, times):
    """
    Enforce u[g,t] = 1 for t in times.
    """
    times = _as_list(times)

    def _f(m, var, *args, **kwargs):
        u = _get_var(var, ["u", "U", "on", "commit"])
        for t in times:
            m.addConstr(u[g, t] == 1, name=f"foil_force_on_g{g}_t{t}")
    return _f

def foil_force_off(g, times):
    """
    Enforce u[g,t] = 0 for t in times.
    """
    times = _as_list(times)

    def _f(m, var, *args, **kwargs):
        u = _get_var(var, ["u", "U", "on", "commit"])
        for t in times:
            m.addConstr(u[g, t] == 0, name=f"foil_force_off_g{g}_t{t}")
    return _f


def foil_force_startup(g, t):
    """
    Enforce v[g,t] = 1 (startup event). Requires startup var v.
    """
    def _f(m, var, *args, **kwargs):
        v = _get_var(var, ["v", "V", "startup"])
        m.addConstr(v[g, t] == 1, name=f"foil_force_startup_g{g}_t{t}")
    return _f

def foil_force_shutdown(g, t):
    """
    Enforce w[g,t] = 1 (shutdown event). Requires shutdown var w.
    """
    def _f(m, var, *args, **kwargs):
        w = _get_var(var, ["w", "W", "shutdown"])
        m.addConstr(w[g, t] == 1, name=f"foil_force_shutdown_g{g}_t{t}")
    return _f





def foil_flip_one_commitment(sol_factual, g, t):
    """
    If factual u[g,t]=1 -> enforce u[g,t]=0.
    If factual u[g,t]=0 -> enforce u[g,t]=1.
    """
    uF = sol_factual["u"]
    target = 1 - int(round(uF[g, t]))

    def _f(m, var, *args, **kwargs):
        u = _get_var(var, ["u", "U", "on", "commit"])
        m.addConstr(u[g, t] == target, name=f"foil_flip_u_g{g}_t{t}")
    return _f

def foil_flip_unit_window(sol_factual, g, t0, t1):
    """
    Enforce u[g,t] = 1-uF[g,t] for all t in [t0,t1].
    """
    uF = sol_factual["u"]

    def _f(m, var, *args, **kwargs):
        u = _get_var(var, ["u", "U", "on", "commit"])
        for t in range(t0, t1 + 1):
            target = 1 - int(round(uF[g, t]))
            m.addConstr(u[g, t] == target, name=f"foil_flipwin_u_g{g}_t{t}")
    return _f

def foil_change_on_hours(sol_factual, g, delta_on_hours):
    """
    Enforce sum_t u[g,t] = sum_t uF[g,t] + delta_on_hours.
    Positive delta => more committed hours, negative => fewer.
    """
    uF = sol_factual["u"]
    target = int(round(np.sum(uF[g, :])) + int(delta_on_hours))

    def _f(m, var, *args, **kwargs):
        u = _get_var(var, ["u", "U", "on", "commit"])
        T = u.shape[1]
        m.addConstr(gp.quicksum(u[g, t] for t in range(T)) == target,
                    name=f"foil_onhours_g{g}_d{delta_on_hours}")
    return _f



def foil_at_least_k_on(units, t, k):
    """
    Enforce sum_{g in units} u[g,t] >= k.
    """
    units = _as_list(units)

    def _f(m, var, *args, **kwargs):
        u = _get_var(var, ["u", "U", "on", "commit"])
        m.addConstr(gp.quicksum(u[g, t] for g in units) >= int(k),
                    name=f"foil_kon_t{t}_k{k}")
    return _f

def foil_at_most_k_on(units, t, k):
    """
    Enforce sum_{g in units} u[g,t] <= k.
    """
    units = _as_list(units)

    def _f(m, var, *args, **kwargs):
        u = _get_var(var, ["u", "U", "on", "commit"])
        m.addConstr(gp.quicksum(u[g, t] for g in units) <= int(k),
                    name=f"foil_koff_t{t}_k{k}")
    return _f

def foil_min_startups(t0, t1, min_startups, gens=None):
    """
    Enforce sum_{g in gens, t in [t0,t1]} v[g,t] >= min_startups.
    """
    def _f(m, var, *args, **kwargs):
        v = _get_var(var, ["v", "V", "startup"])
        nG, T = v.shape
        G = range(nG) if gens is None else _as_list(gens)
        m.addConstr(
            gp.quicksum(v[g, t] for g in G for t in range(t0, t1 + 1)) >= int(min_startups),
            name=f"foil_min_startups_{t0}_{t1}_{min_startups}"
        )
    return _f

def foil_max_startups(t0, t1, max_startups, gens=None):
    """
    Enforce sum_{g in gens, t in [t0,t1]} v[g,t] <= max_startups.
    """
    def _f(m, var, *args, **kwargs):
        v = _get_var(var, ["v", "V", "startup"])
        nG, T = v.shape
        G = range(nG) if gens is None else _as_list(gens)
        m.addConstr(
            gp.quicksum(v[g, t] for g in G for t in range(t0, t1 + 1)) <= int(max_startups),
            name=f"foil_max_startups_{t0}_{t1}_{max_startups}"
        )
    return _f





#========================================================
#========================================================


def _fingerprint_solution_uc(sol: dict, decimals_cont: int = 3) -> tuple:
    """
    Create a hashable fingerprint for a UC solution to detect repeats/cycles.
    - Uses binary commitment vars (u,v,w) exactly (rounded).
    - Uses curt + splus + sminus rounded to a few decimals.
    This is usually enough to detect when the SP keeps returning the same result.
    """
    u = np.rint(sol["u"]).astype(int).flatten()
    v = np.rint(sol["v"]).astype(int).flatten()
    w = np.rint(sol["w"]).astype(int).flatten()

    shed  = np.round(sol["shed"].astype(float), decimals_cont).flatten()
    curt  = np.round(sol["curt"].astype(float), decimals_cont).flatten()
    splus = np.round(sol["splus"].astype(float), decimals_cont).flatten()
    sminus= np.round(sol["sminus"].astype(float), decimals_cont).flatten()

    obj_r = round(float(sol["obj"]), 4)

    return (tuple(u), tuple(v), tuple(w), tuple(shed), tuple(curt), tuple(splus), tuple(sminus), obj_r)



def foil_force_unit_commitment(g: int, t: int, on: bool = True):
    """Forces commitment u[g,t] = 1 (on) or 0 (off). g,t 0-based."""
    val = 1 if on else 0

    def _foil(m, var):
        u = var["u"]
        m.addConstr(u[g, t] == val, name=f"foil_u[{g},{t}]")
    return _foil


def foil_curtailment_cap(alpha: float, sol_factual: dict):
    """
    Creates foil: total curtailment <= alpha * total_curtailment(sol_factual).
    alpha in (0,1] typically; if alpha=1, same curtailment as factual.
    """
    C_factual = float(total_curtailment(sol_factual))
    C_bar = float(alpha * C_factual)
    eps_foil = 1e-4

    def _foil(m, var):
        curt = var["curt"]  # curt[r,t]
        # infer nR,T from keys
        nR = max(r for (r, _) in curt.keys()) + 1
        T  = max(t for (_, t) in curt.keys()) + 1
        expr = gp.quicksum(curt[r, t] for r in range(nR) for t in range(T))
        m.addConstr(expr <= C_bar + eps_foil, name="foil_curtailment_cap")
    return _foil


def compose_foils(*foils):
    """Return a foil function that applies all non-None foils."""
    foils = [f for f in foils if f is not None]

    def _foil(m, var):
        for f in foils:
            f(m, var)
    return _foil



def _build_and_solve_b3_mp(
    *,
    nB: int,
    nR: int,
    T: int,
    base_pi_plus: np.ndarray,       # (nB,T)
    base_pi_minus: np.ndarray,      # (nB,T)
    base_curt_cost: np.ndarray,     # (nR,T)
    cuts: list,                     # list[np.ndarray] zK
    fixed_idx: np.ndarray,          # indices into z
    c0_fixed: np.ndarray,           # c0[fixed_idx]
    xFoil_fixed: np.ndarray,        # zFoil[fixed_idx]
    xFoil_splus: np.ndarray,        # zFoil[idx.splus] (nB,T)
    xFoil_sminus: np.ndarray,       # zFoil[idx.sminus] (nB,T)
    xFoil_curt: np.ndarray,         # zFoil[idx.curt] (nR,T)
    idx,                            # IndexMap
    # bounds
    price_lb: float,
    price_ub: float,
    curt_lb: float,
    curt_ub: float,
    # objective weights
    weight_prices: float,
    weight_curt: float,
    # solver controls
    mp_time_limit: float,
    output_flag_mp: int,
    numeric_focus: int = 2,
    infeas_iis_prefix: str = "debug_B3_MP",
    allow_elastic_cuts: bool = False,
    elastic_penalty: float = 1e6,
):
    """
    MP: choose (pi_plus, pi_minus, curt_cost) close (L1) to base values,
    s.t. for every cut zK:  d(theta)^T x_foil <= d(theta)^T xK
    where theta = (pi_plus, pi_minus, curt_cost) and d(theta) is the full cost vector.
    """
    mp = gp.Model("B3_MP")
    mp.Params.OutputFlag = int(output_flag_mp)
    mp.Params.TimeLimit = float(mp_time_limit)
    mp.Params.NumericFocus = int(numeric_focus)
    mp.Params.InfUnbdInfo = 1
    mp.Params.DualReductions = 0  # disambiguate infeasible vs unbounded

    # Decision vars
    Dp = mp.addVars(nB, T, lb=0, ub=float(price_ub), vtype=GRB.CONTINUOUS, name="pi_plus")
    Dm = mp.addVars(nB, T, lb=0, ub=float(price_ub), vtype=GRB.CONTINUOUS, name="pi_minus")

    Kc = None
    if nR > 0:
        Kc = mp.addVars(nR, T, lb=float(curt_lb), ub=float(curt_ub), vtype=GRB.CONTINUOUS, name="curt_cost")

    # L1 distance auxiliaries
    tp = mp.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="t_plus")
    tm = mp.addVars(nB, T, lb=0.0, vtype=GRB.CONTINUOUS, name="t_minus")
    for b in range(nB):
        for t in range(T):
            bp = float(base_pi_plus[b, t])
            bm = float(base_pi_minus[b, t])
            mp.addConstr(tp[b, t] >= Dp[b, t] - bp)
            mp.addConstr(tp[b, t] >= -Dp[b, t] + bp)
            mp.addConstr(tm[b, t] >= Dm[b, t] - bm)
            mp.addConstr(tm[b, t] >= -Dm[b, t] + bm)

    tc = None
    if nR > 0:
        tc = mp.addVars(nR, T, lb=0.0, vtype=GRB.CONTINUOUS, name="t_curt")
        for r in range(nR):
            for t in range(T):
                bc = float(base_curt_cost[r, t])
                mp.addConstr(tc[r, t] >= Kc[r, t] - bc)
                mp.addConstr(tc[r, t] >= -Kc[r, t] + bc)

    # Cuts (optionally elastic)
    slack_cuts = []
    for k, zK in enumerate(cuts):
        xK_fixed = zK[fixed_idx]
        const_k = float(np.dot(c0_fixed, xFoil_fixed - xK_fixed))

        xK_splus  = zK[idx.splus]   # (nB,T)
        xK_sminus = zK[idx.sminus]  # (nB,T)
        xK_curt   = zK[idx.curt] if nR > 0 else None  # (nR,T)

        expr = gp.LinExpr(const_k)
        all_d_zero = True

        # splus/sminus deltas
        for b in range(nB):
            for t in range(T):
                d_sp = float(xFoil_splus[b, t] - xK_splus[b, t])
                d_sm = float(xFoil_sminus[b, t] - xK_sminus[b, t])
                if d_sp != 0.0:
                    expr += Dp[b, t] * d_sp
                    all_d_zero = False
                if d_sm != 0.0:
                    expr += Dm[b, t] * d_sm
                    all_d_zero = False

        # curtailment-cost deltas
        if nR > 0:
            for r in range(nR):
                for t in range(T):
                    d_c = float(xFoil_curt[r, t] - xK_curt[r, t])
                    if d_c != 0.0:
                        expr += Kc[r, t] * d_c
                        all_d_zero = False

        # Structural infeasibility: constant positive cut
        if all_d_zero and const_k > 1e-9:
            return {
                "status": "MP_STRUCTURALLY_INFEASIBLE",
                "message": (
                    f"Cut[{k}] is constant (no delta in mutable blocks) but const_k={const_k:.6g} > 0. "
                    "This cut cannot be satisfied by changing prices/curtailment-penalties."
                ),
                "cut_index": k,
                "const_k": const_k,
            }

        if allow_elastic_cuts:
            s = mp.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"slack_cut[{k}]")
            slack_cuts.append(s)
            mp.addConstr(expr <= s, name=f"cut[{k}]")
        else:
            mp.addConstr(expr <= 0.0, name=f"cut[{k}]")

    # Objective: weighted L1 deviation
    obj = gp.LinExpr()
    obj += float(weight_prices) * gp.quicksum(tp[b, t] + tm[b, t] for b in range(nB) for t in range(T))
    if nR > 0:
        obj += float(weight_curt) * gp.quicksum(tc[r, t] for r in range(nR) for t in range(T))
    if allow_elastic_cuts and slack_cuts:
        obj += float(elastic_penalty) * gp.quicksum(slack_cuts)

    mp.setObjective(obj, GRB.MINIMIZE)
    mp.optimize()

    # Diagnostics for infeasible
    if mp.Status == GRB.INFEASIBLE:

        return {
            "status": "MP_INFEASIBLE",
            "mp_status": int(mp.Status),
            "message": f"MP infeasible.",
        }

    if mp.Status == GRB.TIME_LIMIT and mp.SolCount == 0:

        return {
            "status": "MP_NO_SOLUTION",
            "mp_status": int(mp.Status),
            "message": f"MP hit TIME_LIMIT with no solution.",
        }

    if mp.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):

        return {
            "status": "MP_FAILED",
            "mp_status": int(mp.Status),
            "message": f"MP failed with status={int(mp.Status)}.",
        }

    pi_plus_new = np.array([[Dp[b, t].X for t in range(T)] for b in range(nB)], dtype=float)
    pi_minus_new = np.array([[Dm[b, t].X for t in range(T)] for b in range(nB)], dtype=float)

    curt_cost_new = None
    if nR > 0:
        curt_cost_new = np.array([[Kc[r, t].X for t in range(T)] for r in range(nR)], dtype=float)
    else:
        curt_cost_new = np.zeros((0, T), dtype=float)

    out = {
        "status": "OK",
        "mp_status": int(mp.Status),
        "pi_plus_new": pi_plus_new,
        "pi_minus_new": pi_minus_new,
        "curt_cost_new": curt_cost_new,
        "mp_obj": float(mp.ObjVal),
    }

    if allow_elastic_cuts and slack_cuts:
        out["max_cut_slack"] = float(max(s.X for s in slack_cuts)) if slack_cuts else 0.0
        out["sum_cut_slack"] = float(sum(s.X for s in slack_cuts)) if slack_cuts else 0.0

    return out




def _solve_ncxplain_mp_with_auto_bounds(
    *,
    # dimensions
    nG: int, nB: int, nR: int, T: int,
    # base coefficients
    base_pi_plus: np.ndarray | None,
    base_pi_minus: np.ndarray | None,
    base_curt_cost: np.ndarray | None,
    base_fuel_cost: np.ndarray | None,
    base_no_load_cost: np.ndarray | None,
    base_su_cost: np.ndarray | None,
    base_sd_cost: np.ndarray | None,
    # fixed part (constant across MP)
    fixed_idx: np.ndarray,
    c0_fixed: np.ndarray,
    xFoil_fixed: np.ndarray,
    # foil blocks
    xFoil_splus: np.ndarray | None,
    xFoil_sminus: np.ndarray | None,
    xFoil_curt: np.ndarray | None,
    xFoil_p: np.ndarray | None,
    xFoil_u: np.ndarray | None,
    xFoil_su: np.ndarray | None,
    xFoil_sd: np.ndarray | None,
    # cuts: list of z vectors (full z), but we pass already-extracted blocks for speed
    cuts_blocks: list[dict],
    # mutables selection
    mutables: set[str],
    # bounds dict
    bounds: dict,
    # weights dict for objective
    weights: dict,
    # misc
    mp_time_limit: float = 60.0,
    output_flag_mp: int = 0,
    it: int = 1,
    auto_expand_bounds: bool = True,
    max_expand_rounds: int = 4,
):
    """
    MP (linear) for NCXplain:
      minimize weighted L1 deviation of enabled mutable coefficients from base
      s.t. for every cut k:  c_new^T x_foil <= c_new^T x_cut(k)

    bounds:
      bounds["pi"] = (lb, ub)
      bounds["curt_cost"] = (lb, ub)
      bounds["fuel"] = (lb, ub)
      bounds["no_load"] = (lb, ub)
      bounds["su"] = (lb, ub)
      bounds["sd"] = (lb, ub)

    weights:
      weights["pi"], weights["curt_cost"], weights["gen_costs"]
    """

    mutables = set(mutables)

    # ---------- validation ----------
    def _need(key):
        if key not in bounds:
            raise ValueError(f"bounds must contain key '{key}' when corresponding mutable is enabled.")
        return bounds[key]

    if "pi" in mutables:
        pi_lb, pi_ub = _need("pi")
    if "curt_cost" in mutables:
        curt_lb, curt_ub = _need("curt_cost")
    if "gen_costs" in mutables:
        fuel_lb, fuel_ub = _need("fuel")
        nl_lb, nl_ub = _need("no_load")
        su_lb, su_ub = _need("su")
        sd_lb, sd_ub = _need("sd")

    w_pi = float(weights.get("pi", 1.0))
    w_curt = float(weights.get("curt_cost", 1.0))
    w_gen = float(weights.get("gen_costs", 1.0))

    # ---------- auto-bound expansion loop ----------
    expand_round = 0
    expand_factor = 1.0

    while True:
        # expanded bounds (symmetric scaling around 0)
        def _expand(lb, ub, fac):
            # simple expansion: widen interval away from 0
            if fac <= 1.0:
                return float(lb), float(ub)
            return float(lb) * fac, float(ub) * fac

        m = gp.Model(f"NCXplain_MP_it{it}_ex{expand_round}")
        m.Params.OutputFlag = int(output_flag_mp)
        if mp_time_limit is not None:
            m.Params.TimeLimit = float(mp_time_limit)

        # ---------------- variables ----------------
        # We'll create only variables for enabled mutables.

        # --- pi vars (nB,T) ---
        if "pi" in mutables:
            lb, ub = _expand(pi_lb, pi_ub, expand_factor)
            piP = m.addVars(nB, T, lb=lb, ub=ub, name="pi_plus")
            piM = m.addVars(nB, T, lb=lb, ub=ub, name="pi_minus")

            # abs dev
            dP = m.addVars(nB, T, lb=0.0, name="abs_dpi_plus")
            dM = m.addVars(nB, T, lb=0.0, name="abs_dpi_minus")

            for b in range(nB):
                for t in range(T):
                    baseP = float(base_pi_plus[b, t])
                    baseM = float(base_pi_minus[b, t])
                    m.addConstr(dP[b, t] >=  piP[b, t] - baseP)
                    m.addConstr(dP[b, t] >= -piP[b, t] + baseP)
                    m.addConstr(dM[b, t] >=  piM[b, t] - baseM)
                    m.addConstr(dM[b, t] >= -piM[b, t] + baseM)

        # --- curt_cost vars (nR,T) ---
        if "curt_cost" in mutables:
            lb, ub = _expand(curt_lb, curt_ub, expand_factor)
            cc = m.addVars(nR, T, lb=lb, ub=ub, name="curt_cost")
            dC = m.addVars(nR, T, lb=0.0, name="abs_dcurt")

            for r in range(nR):
                for t in range(T):
                    baseC = float(base_curt_cost[r, t])
                    m.addConstr(dC[r, t] >=  cc[r, t] - baseC)
                    m.addConstr(dC[r, t] >= -cc[r, t] + baseC)

        # --- generator costs (per generator scalar, applied across time) ---
        if "gen_costs" in mutables:
            f_lb, f_ub = _expand(fuel_lb, fuel_ub, expand_factor)
            nl_lb2, nl_ub2 = _expand(nl_lb, nl_ub, expand_factor)
            su_lb2, su_ub2 = _expand(su_lb, su_ub, expand_factor)
            sd_lb2, sd_ub2 = _expand(sd_lb, sd_ub, expand_factor)

            fc = m.addVars(nG, lb=f_lb, ub=f_ub, name="fuel_cost")
            nl = m.addVars(nG, lb=nl_lb2, ub=nl_ub2, name="no_load_cost")
            suc = m.addVars(nG, lb=su_lb2, ub=su_ub2, name="su_cost")
            sdc = m.addVars(nG, lb=sd_lb2, ub=sd_ub2, name="sd_cost")

            df = m.addVars(nG, lb=0.0, name="abs_dfuel")
            dn = m.addVars(nG, lb=0.0, name="abs_dnl")
            dsu = m.addVars(nG, lb=0.0, name="abs_dsu")
            dsd = m.addVars(nG, lb=0.0, name="abs_dsd")

            for g in range(nG):
                bfc = float(base_fuel_cost[g])
                bnl = float(base_no_load_cost[g])
                bsu = float(base_su_cost[g])
                bsd = float(base_sd_cost[g])

                m.addConstr(df[g]  >=  fc[g] - bfc); m.addConstr(df[g]  >= -fc[g] + bfc)
                m.addConstr(dn[g]  >=  nl[g] - bnl); m.addConstr(dn[g]  >= -nl[g] + bnl)
                m.addConstr(dsu[g] >= suc[g] - bsu); m.addConstr(dsu[g] >= -suc[g] + bsu)
                m.addConstr(dsd[g] >= sdc[g] - bsd); m.addConstr(dsd[g] >= -sdc[g] + bsd)

        # ---------------- objective ----------------
        obj = 0.0
        if "pi" in mutables:
            obj += w_pi * (gp.quicksum(dP[b, t] for b in range(nB) for t in range(T)) +
                           gp.quicksum(dM[b, t] for b in range(nB) for t in range(T)))
        if "curt_cost" in mutables:
            obj += w_curt * gp.quicksum(dC[r, t] for r in range(nR) for t in range(T))
        if "gen_costs" in mutables:
            obj += w_gen * (gp.quicksum(df[g] for g in range(nG)) +
                            gp.quicksum(dn[g] for g in range(nG)) +
                            gp.quicksum(dsu[g] for g in range(nG)) +
                            gp.quicksum(dsd[g] for g in range(nG)))

        m.setObjective(obj, GRB.MINIMIZE)

        # ---------------- constraints (foil optimal vs cuts) ----------------
        # For each cut k: c_new^T xFoil <= c_new^T xCut
        # Move fixed part to RHS:
        #   sum_mut c_mut·(xFoil - xCut) <= c0_fixed·(xCut_fixed - xFoil_fixed)

        for k, blk in enumerate(cuts_blocks):
            xCut_fixed = blk["x_fixed"]
            rhs = float(np.dot(c0_fixed, (xCut_fixed - xFoil_fixed)))

            lhs = 0.0

            if "pi" in mutables:
                dSplus  = (xFoil_splus  - blk["x_splus"])   # (nB,T)
                dSminus = (xFoil_sminus - blk["x_sminus"])  # (nB,T)
                lhs += gp.quicksum(piP[b, t] * float(dSplus[b, t]) for b in range(nB) for t in range(T))
                lhs += gp.quicksum(piM[b, t] * float(dSminus[b, t]) for b in range(nB) for t in range(T))

            if "curt_cost" in mutables:
                dCurt = (xFoil_curt - blk["x_curt"])  # (nR,T)
                lhs += gp.quicksum(cc[r, t] * float(dCurt[r, t]) for r in range(nR) for t in range(T))

            if "gen_costs" in mutables:
                # fuel scalar per gen applied across time: sum_t (pFoil - pCut)
                dP = (xFoil_p - blk["x_p"])      # (nG,T)
                dU = (xFoil_u - blk["x_u"])      # (nG,T)
                dSU = (xFoil_su - blk["x_su"])   # (nG,T)
                dSD = (xFoil_sd - blk["x_sd"])   # (nG,T)

                for g in range(nG):
                    lhs += fc[g]  * float(np.sum(dP[g, :]))
                    lhs += nl[g]  * float(np.sum(dU[g, :]))
                    lhs += suc[g] * float(np.sum(dSU[g, :]))
                    lhs += sdc[g] * float(np.sum(dSD[g, :]))

            m.addConstr(lhs <= rhs, name=f"foil_le_cut_{k}")

        # Solve MP
        m.optimize()

        if m.SolCount > 0 and m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            out = {"status": "OK", "mp_obj": float(m.ObjVal), "bounds_used_factor": expand_factor}

            # extract
            if "pi" in mutables:
                pi_plus_new  = np.array([[float(piP[b, t].X) for t in range(T)] for b in range(nB)], dtype=float)
                pi_minus_new = np.array([[float(piM[b, t].X) for t in range(T)] for b in range(nB)], dtype=float)
            else:
                pi_plus_new, pi_minus_new = base_pi_plus.copy(), base_pi_minus.copy()

            if "curt_cost" in mutables:
                curt_cost_new = np.array([[float(cc[r, t].X) for t in range(T)] for r in range(nR)], dtype=float)
            else:
                curt_cost_new = base_curt_cost.copy() if base_curt_cost is not None else np.zeros((0, T), dtype=float)

            if "gen_costs" in mutables:
                fuel_new    = np.array([float(fc[g].X) for g in range(nG)], dtype=float)
                no_load_new = np.array([float(nl[g].X) for g in range(nG)], dtype=float)
                su_new      = np.array([float(suc[g].X) for g in range(nG)], dtype=float)
                sd_new      = np.array([float(sdc[g].X) for g in range(nG)], dtype=float)
            else:
                fuel_new    = base_fuel_cost.copy()
                no_load_new = base_no_load_cost.copy()
                su_new      = base_su_cost.copy()
                sd_new      = base_sd_cost.copy()

            out.update({
                "pi_plus_new": pi_plus_new,
                "pi_minus_new": pi_minus_new,
                "curt_cost_new": curt_cost_new,
                "fuel_cost_new": fuel_new,
                "no_load_cost_new": no_load_new,
                "su_cost_new": su_new,
                "sd_cost_new": sd_new,
            })
            return out

        # infeasible: expand bounds or stop
        if not auto_expand_bounds or expand_round >= max_expand_rounds:
            return {
                "status": "MP_INFEASIBLE",
                "m_status": int(m.Status),
                "bounds_used_factor": expand_factor,
                "message": "MP infeasible (even after auto expansion)." if auto_expand_bounds else "MP infeasible."
            }

        expand_round += 1
        expand_factor *= 2.0



def run_ncxplain_uc(
    data,
    window_size: int,
    per_bus_neutrality: bool,
    foil_fn,
    mutables=None,                 # e.g. {"pi"} or {"gen_costs"} or {"pi","gen_costs"}
    bounds=None,                   # dict; see defaults below
    weights=None,                  # dict; see defaults below
    tol_abs: float = 1e-3,
    tol_rel: float = 1e-6,
    max_iters: int = 30,
    mp_time_limit: float = 60.0,
    output_flag_mp: int = 0,
    output_flag_sp: int = 0,
    verbose: bool = True,
    cycle_patience: int = 3,
    auto_expand_bounds: bool = True,
    seed_with_factual_cut: bool = True,
):
    """
    Generic NCXplain for UC via cutting planes:
      Find minimal cost-perturbation (over selected mutable coefficients)
      such that the provided foil solution becomes optimal.

    You choose:
      - foil_fn: constraint-adder defining the foil (commitment-focused, etc.)
      - mutables: set of coefficient blocks that can change
      - bounds: bounds per block
      - weights: objective weights per block

    Returns dict with status + best explanation found.
    """

    if foil_fn is None:
        raise ValueError("foil_fn must be provided (generic NCXplain requires an explicit foil).")

    if mutables is None:
        mutables = {"pi"}   # default
    mutables = set(mutables)

    # --------------------------
    # defaults: bounds + weights
    # --------------------------
    if bounds is None:
        bounds = {}
    bounds = dict(bounds)

    # sensible defaults (override per application)
    bounds.setdefault("pi", (0.0, 100.0))
    bounds.setdefault("curt_cost", (0.0, 200.0))
    bounds.setdefault("fuel", (0.0, 500.0))
    bounds.setdefault("no_load", (0.0, 5000.0))
    bounds.setdefault("su", (0.0, 50000.0))
    bounds.setdefault("sd", (0.0, 50000.0))

    if weights is None:
        weights = {}
    weights = dict(weights)
    weights.setdefault("pi", 1.0)
    weights.setdefault("curt_cost", 1.0)
    weights.setdefault("gen_costs", 1.0)

    # --------------------------
    # 0) dimensions + index map
    # --------------------------
    nG = len(data.gens)
    nR = len(data.rens)
    nB = int(data.nB)
    nL = len(data.lines)
    T  = int(data.T)

    idx = build_index_map_network_uc(nG=nG, nR=nR, nB=nB, nL=nL, T=T, major="t")
    u_init, p_init, on_time_init, off_time_init = default_initial_conditions(data)

    # --------------------------
    # 1) base cost vector c0
    # --------------------------
    fuel_cost_vec     = np.array([float(g.fuel_cost) for g in data.gens], dtype=float)
    emission_rate_vec = np.array([float(getattr(g, "emission_rate", 0.0)) for g in data.gens], dtype=float)
    no_load_cost_vec  = np.array([float(getattr(g, "no_load_cost", 0.0)) for g in data.gens], dtype=float)
    su_cost_vec       = np.array([float(getattr(g, "SU_cost", 0.0)) for g in data.gens], dtype=float)
    sd_cost_vec       = np.array([float(getattr(g, "SD_cost", 0.0)) for g in data.gens], dtype=float)
    curt_cost_mat     = np.vstack([np.asarray(r.curt_cost, dtype=float) for r in data.rens]) if nR > 0 else np.zeros((0, T))

    c0 = build_cost_vector_network_uc(
        idx=idx,
        fuel_cost=fuel_cost_vec,
        emission_rate=emission_rate_vec,
        carbon_price=float(getattr(data, "carbon_price", 0.0)),
        no_load_cost=no_load_cost_vec,
        su_cost=su_cost_vec,
        sd_cost=sd_cost_vec,
        curt_cost=curt_cost_mat,
        pi_plus=np.asarray(data.pi_plus, dtype=float),
        pi_minus=np.asarray(data.pi_minus, dtype=float),
        voll=float(getattr(data, "voll", 20000.0)),
    )

    # --------------------------
    # 2) factual SP under c0
    # --------------------------
    mF, solF, zF = solve_uc_with_cost(
        data, idx, c0,
        window_size, per_bus_neutrality,
        u_init, p_init, on_time_init, off_time_init,
        extra_constr_fn=None,
        output_flag=output_flag_sp
    )
    if solF is None:
        return {"status": "FACTUAL_FAIL", "m_status": int(mF.Status)}

    # --------------------------
    # 3) foil SP under c0
    # --------------------------
    mFoil, solFoil, zFoil = solve_uc_with_cost(
        data, idx, c0,
        window_size, per_bus_neutrality,
        u_init, p_init, on_time_init, off_time_init,
        extra_constr_fn=foil_fn,
        output_flag=output_flag_sp
    )
    if solFoil is None:
        return {"status": "NO_FOIL", "message": "Foil infeasible under base costs c0."}

    # --------------------------
    # 4) mask mutable / fixed_idx
    # --------------------------
    
    def _build_mutability_mask(idx, data, mutables, c0=None):
        """
        Build boolean mask over z/c0 entries that are mutable.

        mutables: set like {"gen_costs"} or {"pi"} or {"pi","gen_costs"} etc.
        """
        n = _idx_nvars(idx, c0)
        mask_mut = np.zeros(n, dtype=bool)

        # -----------------------
        # Shift prices (pi_plus / pi_minus) live on splus/sminus blocks
        # -----------------------
        if "pi" in mutables or "shift_prices" in mutables:
            if hasattr(idx, "splus") and idx.splus is not None and getattr(idx.splus, "size", 0) > 0:
                mask_mut[idx.splus.flatten()] = True
            if hasattr(idx, "sminus") and idx.sminus is not None and getattr(idx.sminus, "size", 0) > 0:
                mask_mut[idx.sminus.flatten()] = True

        # -----------------------
        # Renewable curtailment penalty (if you model it as cost on curt variables)
        # -----------------------
        if "curt_cost" in mutables or "curtailment_cost" in mutables:
            if hasattr(idx, "curt") and idx.curt is not None and getattr(idx.curt, "size", 0) > 0:
                mask_mut[idx.curt.flatten()] = True

        # -----------------------
        # Generator costs:
        # fuel/no-load/SU/SD affect objective through cvec construction,
        # but in your "cost-vector perturbation" approach, you typically
        # mark the corresponding z-blocks as mutable IF your MP is built on c·x.
        #
        # If your MP is directly changing generator parameters (fuel, SU, etc.)
        # and rebuilding cvec, then mask_mut may be unused for those terms.
        # Still, keep this hook if your implementation uses it.
        # -----------------------
        if "gen_costs" in mutables:
            # If your cost vector places fuel cost on p, no-load on u, SU on v, SD on w:
            if hasattr(idx, "p") and idx.p is not None:
                mask_mut[idx.p.flatten()] = True
            if hasattr(idx, "u") and idx.u is not None:
                mask_mut[idx.u.flatten()] = True
            if hasattr(idx, "v") and idx.v is not None:
                mask_mut[idx.v.flatten()] = True
            if hasattr(idx, "w") and idx.w is not None:
                mask_mut[idx.w.flatten()] = True

        fixed_idx = np.where(~mask_mut)[0]
        return mask_mut, fixed_idx

    # Build mutability mask + fixed index set
    mask_mut, fixed_idx = _build_mutability_mask(idx, data, mutables, c0=c0)

    # (optional safety; helps catch shape issues early)
    mask_mut = np.asarray(mask_mut, dtype=bool)
    fixed_idx = np.asarray(fixed_idx, dtype=int)

    # Sanity check: mask must match the dimension of z/c0
    n = _idx_nvars(idx, c0)
    if mask_mut.shape[0] != n:
        raise ValueError(f"mask_mut length {mask_mut.shape[0]} != n_vars {n}")


    xFoil_fixed = zFoil[fixed_idx]
    c0_fixed    = c0[fixed_idx]

    # foil blocks
    xFoil_splus  = zFoil[idx.splus]   if ("pi" in mutables and hasattr(idx, "splus") and idx.splus.size>0) else None
    xFoil_sminus = zFoil[idx.sminus]  if ("pi" in mutables and hasattr(idx, "sminus") and idx.sminus.size>0) else None
    xFoil_curt   = zFoil[idx.curt]    if ("curt_cost" in mutables and nR>0 and hasattr(idx, "curt") and idx.curt.size>0) else None

    xFoil_p  = zFoil[idx.p]   if ("gen_costs" in mutables and hasattr(idx, "p") and idx.p.size>0) else None
    xFoil_u  = zFoil[idx.u]   if ("gen_costs" in mutables and hasattr(idx, "u") and idx.u.size>0) else None
    xFoil_su = zFoil[idx.su]  if ("gen_costs" in mutables and hasattr(idx, "su") and idx.su.size>0) else None
    xFoil_sd = zFoil[idx.sd]  if ("gen_costs" in mutables and hasattr(idx, "sd") and idx.sd.size>0) else None

    # --------------------------
    # 5) base params for mutables
    # --------------------------
    base_pi_plus   = np.asarray(data.pi_plus, dtype=float)
    base_pi_minus  = np.asarray(data.pi_minus, dtype=float)
    base_curt_cost = curt_cost_mat.copy()

    base_fuel_cost    = fuel_cost_vec.copy()
    base_no_load_cost = no_load_cost_vec.copy()
    base_su_cost      = su_cost_vec.copy()
    base_sd_cost      = sd_cost_vec.copy()

    # helper: build new cvec from selected mutable blocks
    def _build_cvec_from_mutables(
        *,
        pi_plus_new: np.ndarray,
        pi_minus_new: np.ndarray,
        curt_cost_new: np.ndarray,
        fuel_cost_new: np.ndarray,
        no_load_new: np.ndarray,
        su_cost_new: np.ndarray,
        sd_cost_new: np.ndarray,
    ) -> np.ndarray:
        c = c0.copy()

        if "pi" in mutables:
            if hasattr(idx, "splus") and idx.splus.size > 0:
                c[idx.splus] = pi_plus_new
            if hasattr(idx, "sminus") and idx.sminus.size > 0:
                c[idx.sminus] = pi_minus_new

        if "curt_cost" in mutables and nR > 0 and hasattr(idx, "curt") and idx.curt.size > 0:
            c[idx.curt] = curt_cost_new

        if "gen_costs" in mutables:
            # fuel applies to p[g,t]
            if hasattr(idx, "p") and idx.p.size > 0:
                Pidx = idx.p
                for g in range(nG):
                    c[Pidx[g, :]] = fuel_cost_new[g]
            # no-load applies to u[g,t]
            if hasattr(idx, "u") and idx.u.size > 0:
                Uidx = idx.u
                for g in range(nG):
                    c[Uidx[g, :]] = no_load_new[g]
            # SU/SD apply to su[g,t], sd[g,t]
            if hasattr(idx, "su") and idx.su.size > 0:
                SUidx = idx.su
                for g in range(nG):
                    c[SUidx[g, :]] = su_cost_new[g]
            if hasattr(idx, "sd") and idx.sd.size > 0:
                SDidx = idx.sd
                for g in range(nG):
                    c[SDidx[g, :]] = sd_cost_new[g]

        return c

    # --------------------------
    # 6) cutting planes storage
    # --------------------------
    cuts = []
    cuts_fp = set()

    # fallback fingerprint if you don't have _fingerprint_solution_uc
    def _fingerprint(sol, decimals=3):
        # try dict arrays
        if isinstance(sol, dict):
            def _arr(name):
                a = sol.get(name, None)
                if a is None:
                    return ()
                return tuple(np.round(np.asarray(a, dtype=float), decimals).ravel())
            return (
                tuple(np.rint(np.asarray(sol.get("u", 0))).astype(int).ravel()) if "u" in sol else (),
                tuple(np.rint(np.asarray(sol.get("v", 0))).astype(int).ravel()) if "v" in sol else (),
                tuple(np.rint(np.asarray(sol.get("w", 0))).astype(int).ravel()) if "w" in sol else (),
                _arr("shed"), _arr("curt"), _arr("splus"), _arr("sminus"), _arr("p"),
            )
        return ("unknown",)

    if seed_with_factual_cut:
        cuts.append(zF.copy())
        cuts_fp.add(_fingerprint(solF, decimals=3))

    seen = {}
    repeat_hits = 0

    best = {
        "gap": float("inf"),
        "iter": None,
        "params": None,
        "dTx_foil": None,
        "dTx_opt": None,
        "mp_details": None,
        "sol_factual": solF,
        "sol_foil": solFoil,
    }

    # --------------------------
    # 7) main loop
    # --------------------------
    for it in range(1, max_iters + 1):
        # pre-extract cut blocks once for MP
        cuts_blocks = []
        for zcut in cuts:
            blk = {"x_fixed": zcut[fixed_idx]}
            if "pi" in mutables:
                blk["x_splus"]  = zcut[idx.splus]
                blk["x_sminus"] = zcut[idx.sminus]
            if "curt_cost" in mutables and nR > 0:
                blk["x_curt"] = zcut[idx.curt]
            if "gen_costs" in mutables:
                blk["x_p"]  = zcut[idx.p]
                blk["x_u"]  = zcut[idx.u]   if hasattr(idx, "u") and idx.u.size>0 else np.zeros((nG, T))
                blk["x_su"] = zcut[idx.su]  if hasattr(idx, "su") and idx.su.size>0 else np.zeros((nG, T))
                blk["x_sd"] = zcut[idx.sd]  if hasattr(idx, "sd") and idx.sd.size>0 else np.zeros((nG, T))
            cuts_blocks.append(blk)

        mp_out = _solve_ncxplain_mp_with_auto_bounds(
            nG=nG, nB=nB, nR=nR, T=T,
            base_pi_plus=base_pi_plus,
            base_pi_minus=base_pi_minus,
            base_curt_cost=base_curt_cost,
            base_fuel_cost=base_fuel_cost,
            base_no_load_cost=base_no_load_cost,
            base_su_cost=base_su_cost,
            base_sd_cost=base_sd_cost,
            fixed_idx=fixed_idx,
            c0_fixed=c0_fixed,
            xFoil_fixed=xFoil_fixed,
            xFoil_splus=xFoil_splus,
            xFoil_sminus=xFoil_sminus,
            xFoil_curt=xFoil_curt,
            xFoil_p=xFoil_p,
            xFoil_u=xFoil_u if xFoil_u is not None else np.zeros((nG, T)),
            xFoil_su=xFoil_su if xFoil_su is not None else np.zeros((nG, T)),
            xFoil_sd=xFoil_sd if xFoil_sd is not None else np.zeros((nG, T)),
            cuts_blocks=cuts_blocks,
            mutables=mutables,
            bounds=bounds,
            weights=weights,
            mp_time_limit=mp_time_limit,
            output_flag_mp=output_flag_mp,
            it=it,
            auto_expand_bounds=auto_expand_bounds,
        )

        if mp_out.get("status") != "OK":
            mp_out["iter"] = it
            return mp_out

        # new params
        pi_plus_new   = mp_out["pi_plus_new"]
        pi_minus_new  = mp_out["pi_minus_new"]
        curt_cost_new = mp_out["curt_cost_new"]
        fuel_new      = mp_out["fuel_cost_new"]
        no_load_new   = mp_out["no_load_cost_new"]
        su_new        = mp_out["su_cost_new"]
        sd_new        = mp_out["sd_cost_new"]

        # solve SP under new costs
        c_new = _build_cvec_from_mutables(
            pi_plus_new=pi_plus_new,
            pi_minus_new=pi_minus_new,
            curt_cost_new=curt_cost_new,
            fuel_cost_new=fuel_new,
            no_load_new=no_load_new,
            su_cost_new=su_new,
            sd_cost_new=sd_new,
        )

        mOpt, solOpt, zOpt = solve_uc_with_cost(
            data, idx, c_new,
            window_size, per_bus_neutrality,
            u_init, p_init, on_time_init, off_time_init,
            extra_constr_fn=None,
            output_flag=output_flag_sp
        )
        if solOpt is None:
            return {"status": "SP_FAILED", "iter": it, "sp_status": int(mOpt.Status)}

        # stopping test: dTx_foil - dTx_opt
        xOpt_fixed = zOpt[fixed_idx]

        dTx_foil = _compute_dTx_uc(
            c0_fixed=c0_fixed, x_fixed=xFoil_fixed,
            pi_plus=pi_plus_new if "pi" in mutables else None,
            pi_minus=pi_minus_new if "pi" in mutables else None,
            x_splus=xFoil_splus,
            x_sminus=xFoil_sminus,
            curt_cost=curt_cost_new if "curt_cost" in mutables else None,
            x_curt=xFoil_curt,
            fuel_cost=fuel_new if "gen_costs" in mutables else None,
            no_load_cost=no_load_new if "gen_costs" in mutables else None,
            su_cost=su_new if "gen_costs" in mutables else None,
            sd_cost=sd_new if "gen_costs" in mutables else None,
            x_p=xFoil_p,
            x_u=xFoil_u if xFoil_u is not None else None,
            x_su=xFoil_su if xFoil_su is not None else None,
            x_sd=xFoil_sd if xFoil_sd is not None else None,
        )

        # extract opt blocks (only if needed)
        xOpt_splus  = zOpt[idx.splus] if "pi" in mutables else None
        xOpt_sminus = zOpt[idx.sminus] if "pi" in mutables else None
        xOpt_curt   = zOpt[idx.curt] if ("curt_cost" in mutables and nR > 0) else None
        xOpt_p      = zOpt[idx.p] if "gen_costs" in mutables else None
        xOpt_u      = zOpt[idx.u] if ("gen_costs" in mutables and hasattr(idx, "u") and idx.u.size>0) else None
        xOpt_su     = zOpt[idx.su] if ("gen_costs" in mutables and hasattr(idx, "su") and idx.su.size>0) else None
        xOpt_sd     = zOpt[idx.sd] if ("gen_costs" in mutables and hasattr(idx, "sd") and idx.sd.size>0) else None

        dTx_opt = _compute_dTx_uc(
            c0_fixed=c0_fixed, x_fixed=xOpt_fixed,
            pi_plus=pi_plus_new if "pi" in mutables else None,
            pi_minus=pi_minus_new if "pi" in mutables else None,
            x_splus=xOpt_splus,
            x_sminus=xOpt_sminus,
            curt_cost=curt_cost_new if "curt_cost" in mutables else None,
            x_curt=xOpt_curt,
            fuel_cost=fuel_new if "gen_costs" in mutables else None,
            no_load_cost=no_load_new if "gen_costs" in mutables else None,
            su_cost=su_new if "gen_costs" in mutables else None,
            sd_cost=sd_new if "gen_costs" in mutables else None,
            x_p=xOpt_p,
            x_u=xOpt_u,
            x_su=xOpt_su,
            x_sd=xOpt_sd,
        )

        gap = float(dTx_foil - dTx_opt)
        stop_tol = float(tol_abs + tol_rel * max(1.0, abs(dTx_opt)))

        if gap < best["gap"]:
            best.update({
                "gap": gap,
                "iter": it,
                "params": {
                    "pi_plus_new": pi_plus_new,
                    "pi_minus_new": pi_minus_new,
                    "curt_cost_new": curt_cost_new,
                    "fuel_cost_new": fuel_new,
                    "no_load_cost_new": no_load_new,
                    "su_cost_new": su_new,
                    "sd_cost_new": sd_new,
                },
                "dTx_foil": dTx_foil,
                "dTx_opt": dTx_opt,
                "mp_details": mp_out,
            })

        if verbose:
            print(f"[NCX it={it:02d}] gap={gap:.6g} | stop_tol={stop_tol:.3g} | MP_obj={mp_out.get('mp_obj', None)}")

        if gap <= stop_tol:
            return {
                "status": "OPTIMAL",
                "iters": it,
                "mutables": sorted(list(mutables)),
                "bounds": bounds,
                "weights": weights,
                "z_foil": zFoil,
                "z_opt": zOpt,
                "dTx_foil": dTx_foil,
                "dTx_opt": dTx_opt,
                "gap": gap,
                "stop_tol": stop_tol,
                "mp_details": mp_out,
                "sol_factual": solF,
                "sol_foil": solFoil,
                "sol_opt": solOpt,
                **best["params"],
            }

        # cycle detection
        fp = _fingerprint(solOpt, decimals=3)
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] >= 2:
            repeat_hits += 1
        if repeat_hits >= cycle_patience:
            return {
                "status": "CYCLE_OR_NO_SOLUTION",
                "iters": it,
                "message": "SP repeated previously seen solution; likely cycling or explanation does not exist with chosen mutables/bounds.",
                "best_gap": float(best["gap"]),
                "best_iter": best["iter"],
                "best_params": best["params"],
                "best_dTx_foil": best["dTx_foil"],
                "best_dTx_opt": best["dTx_opt"],
                "best_mp_details": best["mp_details"],
            }

        # add cut
        cuts.append(zOpt.copy())

    # max iters
    return {
        "status": "MAX_ITERS",
        "iters": max_iters,
        "best_gap": float(best["gap"]),
        "best_iter": best["iter"],
        "best_params": best["params"],
        "best_dTx_foil": best["dTx_foil"],
        "best_dTx_opt": best["dTx_opt"],
        "best_mp_details": best["mp_details"],
        "message": "No convergence within max_iters. Consider widening bounds or adding additional mutables.",
    }



#=============================================================================
#=============================================================================


def _hourly_emissions_from_sol(sol: dict, emission_rate: np.ndarray) -> np.ndarray:
    """
    Returns hourly emissions E[t] = sum_g emission_rate[g] * p[g,t]
    sol["p"] shape (nG,T)
    emission_rate shape (nG,)
    """
    p = np.asarray(sol["p"], dtype=float)
    er = np.asarray(emission_rate, dtype=float).reshape(-1, 1)
    return (er * p).sum(axis=0)  # (T,)

def _total_emissions_from_sol(sol: dict, emission_rate: np.ndarray) -> float:
    return float(_hourly_emissions_from_sol(sol, emission_rate).sum())

def _add_total_emissions_cap(
    m: gp.Model,
    var: dict,
    emission_rate: np.ndarray,  # (nG,)
    E_cap: float,
    eps: float = 1e-6
):
    """
    Adds: sum_{g,t} er[g] * p[g,t] <= E_cap + eps
    """
    p = var["p"]
    nG = len(emission_rate)
    # infer T from var["p"] keys: assume p[g,t]
    # safest: get max t from keys
    T = max(t for (_, t) in p.keys()) + 1

    expr = gp.quicksum(float(emission_rate[g]) * p[g, t] for g in range(nG) for t in range(T))
    m.addConstr(expr <= float(E_cap) + float(eps), name="foil_total_emissions_cap")

def _add_topk_hour_emissions_caps(
    m: gp.Model,
    var: dict,
    emission_rate: np.ndarray,     # (nG,)
    hours: list[int],              # top-K hours indices
    hourly_caps: np.ndarray,       # (len(hours),) caps for each selected hour
    eps: float = 1e-6
):
    """
    Adds per-hour caps for selected hours:
      for each j: sum_g er[g]*p[g, hours[j]] <= hourly_caps[j] + eps
    """
    p = var["p"]
    nG = len(emission_rate)

    for j, t in enumerate(hours):
        cap_t = float(hourly_caps[j])
        expr_t = gp.quicksum(float(emission_rate[g]) * p[g, t] for g in range(nG))
        m.addConstr(expr_t <= cap_t + float(eps), name=f"foil_emissions_cap_t[{t}]")




def _compute_total_emissions_from_sol(data, sol):
    """
    Total emissions = sum_{g,t} emission_rate[g] * p[g,t]
    Assumes:
      - sol["p"] shape (nG,T)
      - data.gens[g].emission_rate exists
    """
    er = np.array([float(gen.emission_rate) for gen in data.gens], dtype=float)  # (nG,)
    p = np.asarray(sol["p"], dtype=float)                                       # (nG,T)
    return float(np.sum(er[:, None] * p))


def _compute_hourly_emissions_from_sol(data, sol):
    """
    Hourly emissions E[t] = sum_g emission_rate[g] * p[g,t]
    Returns shape (T,)
    """
    er = np.array([float(gen.emission_rate) for gen in data.gens], dtype=float)  # (nG,)
    p = np.asarray(sol["p"], dtype=float)                                       # (nG,T)
    return np.sum(er[:, None] * p, axis=0)


def _compute_dTx_full(
    *,
    c0_fixed: np.ndarray,
    x_fixed: np.ndarray,
    pi_plus: np.ndarray,     # (nB,T)
    pi_minus: np.ndarray,    # (nB,T)
    x_splus: np.ndarray,     # (nB,T)
    x_sminus: np.ndarray,    # (nB,T)
    curt_cost: np.ndarray = None,   # (nR,T) or None
    x_curt: np.ndarray = None,      # (nR,T) or None
) -> float:
    val = float(np.dot(c0_fixed, x_fixed) + np.sum(pi_plus * x_splus) + np.sum(pi_minus * x_sminus))
    if curt_cost is not None and x_curt is not None:
        val += float(np.sum(curt_cost * x_curt))
    return float(val)


def run_E3_ncxplain_shift_and_curt_prices_emissions(
    data,                         # NetworkUCData-like
    window_size: int,
    per_bus_neutrality: bool,
    alpha: float,                 # e.g. 0.10 means "reduce emissions by 10%"
    *,
    # what foil means:
    emissions_mode: str = "total",     # "total" | "top_k_hours"
    top_k: int = 5,                    # only used if emissions_mode="top_k_hours"

    # MP variable bounds
    price_lb: float = 0.0,
    price_ub: float = 500.0,
    curt_lb: float = 0.0,
    curt_ub: float = 200.0,

    # MP objective weights
    weight_prices: float = 1.0,
    weight_curt: float = 0.2,

    # convergence
    tol_abs: float = 1e-3,
    tol_rel: float = 1e-6,
    max_iters: int = 50,

    # solver controls
    mp_time_limit: float = 90.0,
    output_flag_mp: int = 0,
    output_flag_sp: int = 0,

    # robustness
    verbose: bool = True,
    cycle_patience: int = 3,
    auto_expand_bounds: bool = True,
    seed_with_factual_cut: bool = True,
):
    """
    E3 NCXplain:
      "What shift prices (pi_plus, pi_minus) and curtailment penalties (curt_cost)
       are necessary so that a solution achieving an emissions reduction becomes optimal?"

    Foil is defined by emissions constraint:
      - total mode:     sum_{g,t} er_g p_{g,t} <= (1-alpha) * E_factual
      - top_k_hours:    for t in top-k hours (by factual hourly emissions),
                        sum_g er_g p_{g,t} <= (1-alpha) * E_factual_hour[t]
    """

    # Local imports to avoid module-level dependency issues
    from uc_pipeline import (
        build_index_map_network_uc,
        default_initial_conditions,
        build_cost_vector_network_uc,
        solve_uc_with_cost,
        total_curtailment,
    )

    # Need MP helpers already in this module:
    #   _solve_b3_mp_with_auto_bounds (your extended version that supports curt_cost)
    if "_solve_b3_mp_with_auto_bounds" not in globals():
        raise RuntimeError("Missing _solve_b3_mp_with_auto_bounds in b3_ncxplain.py (required).")
    if "_fingerprint_solution_uc" not in globals():
        raise RuntimeError("Missing _fingerprint_solution_uc in b3_ncxplain.py (required).")

    # --------------------------
    # 0) dimensions + index map
    # --------------------------
    nG = len(data.gens)
    nR = len(data.rens)
    nB = data.nB
    nL = len(data.lines)
    T  = data.T

    idx = build_index_map_network_uc(nG=nG, nR=nR, nB=nB, nL=nL, T=T, major="t")
    u_init, p_init, on_time_init, off_time_init = default_initial_conditions(data)

    # Basic check: emissions must exist
    er_vec = np.array([float(g.emission_rate) for g in data.gens], dtype=float)
    if np.all(np.abs(er_vec) <= 1e-12):
        return {
            "status": "EMISSIONS_ZERO",
            "message": "All emission_rate are zero. Set nonzero generator emission rates before running E3."
        }

    # --------------------------
    # 1) base cost vector c0
    # --------------------------
    fuel_cost_vec     = np.array([g.fuel_cost for g in data.gens], dtype=float)
    emission_rate_vec = np.array([g.emission_rate for g in data.gens], dtype=float)
    no_load_cost_vec  = np.array([g.no_load_cost for g in data.gens], dtype=float)
    su_cost_vec       = np.array([g.SU_cost for g in data.gens], dtype=float)
    sd_cost_vec       = np.array([g.SD_cost for g in data.gens], dtype=float)
    curt_cost_mat     = np.vstack([r.curt_cost for r in data.rens]) if nR > 0 else np.zeros((0, T))

    c0 = build_cost_vector_network_uc(
        idx=idx,
        fuel_cost=fuel_cost_vec,
        emission_rate=emission_rate_vec,
        carbon_price=float(getattr(data, "carbon_price", 0.0)),
        no_load_cost=no_load_cost_vec,
        su_cost=su_cost_vec,
        sd_cost=sd_cost_vec,
        curt_cost=curt_cost_mat,
        pi_plus=data.pi_plus,
        pi_minus=data.pi_minus,
        voll=float(getattr(data, "voll", 20000.0)),
    )

    # --------------------------
    # 2) factual SP under c0
    # --------------------------
    mF, solF, zF = solve_uc_with_cost(
        data, idx, c0,
        window_size, per_bus_neutrality,
        u_init, p_init, on_time_init, off_time_init,
        extra_constr_fn=None,
        output_flag=output_flag_sp
    )
    if solF is None:
        return {"status": "FACTUAL_FAIL", "m_status": int(mF.Status)}

    E_factual = _compute_total_emissions_from_sol(data, solF)
    if E_factual <= 1e-9:
        return {
            "status": "EMISSIONS_ZERO_FACTUAL",
            "message": "Factual emissions are ~0, so an emissions-reduction foil is structurally redundant.",
            "E_factual": float(E_factual),
        }

    # emissions target
    if emissions_mode == "total":
        E_bar = float((1.0 - alpha) * E_factual)
        top_hours = None
    elif emissions_mode == "top_k_hours":
        hourly = _compute_hourly_emissions_from_sol(data, solF)  # (T,)
        top_k = int(min(max(1, top_k), T))
        top_hours = np.argsort(-hourly)[:top_k].tolist()
        E_bar_hours = {int(t): float((1.0 - alpha) * hourly[t]) for t in top_hours}
        E_bar = None
    else:
        raise ValueError("emissions_mode must be 'total' or 'top_k_hours'.")

    # --------------------------
    # 3) foil SP: enforce emissions constraint (NOT curtailment)
    # --------------------------
    def foil_constraint(m, var):
        p = var["p"]
        eps = 1e-6

        if emissions_mode == "total":
            expr = gp.quicksum(float(er_vec[g]) * p[g, t] for g in range(nG) for t in range(T))
            m.addConstr(expr <= float(E_bar) + eps, name="foil_emissions_cap_total")
        else:
            # top_k_hours: cap each selected hour
            for t in top_hours:
                expr_t = gp.quicksum(float(er_vec[g]) * p[g, t] for g in range(nG))
                m.addConstr(expr_t <= float(E_bar_hours[int(t)]) + eps, name=f"foil_emissions_cap_hour[{t}]")

    mFoil, solFoil, zFoil = solve_uc_with_cost(
        data, idx, c0,
        window_size, per_bus_neutrality,
        u_init, p_init, on_time_init, off_time_init,
        extra_constr_fn=foil_constraint,
        output_flag=output_flag_sp
    )
    if solFoil is None:
        return {
            "status": "NO_FOIL",
            "message": "Foil infeasible: emissions reduction target is too aggressive for the system constraints.",
            "E_factual": float(E_factual),
            "alpha": float(alpha),
            "emissions_mode": emissions_mode,
            "top_hours": top_hours,
        }

    # --------------------------
    # 4) mutable vs fixed indices (mutable: splus/sminus + curt_cost)
    # --------------------------
    base_pi_plus   = data.pi_plus.copy()
    base_pi_minus  = data.pi_minus.copy()
    base_curt_cost = curt_cost_mat.copy()  # (nR,T)

    mask_mut = np.zeros(idx.n_vars, dtype=bool)
    mask_mut[idx.splus.flatten()]  = True
    mask_mut[idx.sminus.flatten()] = True
    mask_mut[idx.curt.flatten()]   = True
    fixed_idx = np.where(~mask_mut)[0]

    xFoil_fixed  = zFoil[fixed_idx]
    c0_fixed     = c0[fixed_idx]
    xFoil_splus  = zFoil[idx.splus]   # (nB,T)
    xFoil_sminus = zFoil[idx.sminus]  # (nB,T)
    xFoil_curt   = zFoil[idx.curt]    # (nR,T)

    def build_cvec_from_params(pi_plus, pi_minus, curt_cost_new):
        c = c0.copy()
        c[idx.splus]  = pi_plus
        c[idx.sminus] = pi_minus
        if nR > 0:
            c[idx.curt] = curt_cost_new
        return c

    # --------------------------
    # 5) cuts + cycle tracking
    # --------------------------
    cuts = []
    cuts_fp = set()

    if seed_with_factual_cut:
        cuts.append(zF.copy())
        zfp0 = (
            tuple(np.rint(solF["u"]).astype(int).flatten()),
            tuple(np.rint(solF["v"]).astype(int).flatten()),
            tuple(np.rint(solF["w"]).astype(int).flatten()),
            tuple(np.round(solF["shed"], 3).flatten()),
            tuple(np.round(solF["curt"], 3).flatten()),
            tuple(np.round(solF["splus"], 3).flatten()),
            tuple(np.round(solF["sminus"], 3).flatten()),
        )

        cuts_fp.add(zfp0)

    seen = {}
    repeat_hits = 0

    best = {
        "gap": float("inf"),
        "iter": None,
        "pi_plus": None,
        "pi_minus": None,
        "curt_cost": None,
        "dTx_foil": None,
        "dTx_opt": None,
        "mp_details": None,
    }

    # --------------------------
    # 6) cutting-plane loop
    # --------------------------
    for it in range(1, max_iters + 1):
        mp_out = _solve_b3_mp_with_auto_bounds(
            nB=nB, nR=nR, T=T,
            base_pi_plus=base_pi_plus,
            base_pi_minus=base_pi_minus,
            base_curt_cost=base_curt_cost,
            cuts=cuts,
            fixed_idx=fixed_idx,
            c0_fixed=c0_fixed,
            xFoil_fixed=xFoil_fixed,
            xFoil_splus=xFoil_splus,
            xFoil_sminus=xFoil_sminus,
            xFoil_curt=xFoil_curt,
            idx=idx,
            price_lb=price_lb,
            price_ub=price_ub,
            curt_lb=curt_lb,
            curt_ub=curt_ub,
            weight_prices=weight_prices,
            weight_curt=weight_curt,
            mp_time_limit=mp_time_limit,
            output_flag_mp=output_flag_mp,
            it=it,
            auto_expand_bounds=auto_expand_bounds,
        )

        if mp_out.get("status") not in ("OK", "MP_ELASTIC_OK"):
            mp_out["iter"] = it
            mp_out["E_factual"] = float(E_factual)
            mp_out["alpha"] = float(alpha)
            mp_out["emissions_mode"] = emissions_mode
            return mp_out

        pi_plus_new   = mp_out["pi_plus_new"]
        pi_minus_new  = mp_out["pi_minus_new"]
        curt_cost_new = mp_out["curt_cost_new"]

        # ---- SP under new coeffs ----
        c_new = build_cvec_from_params(pi_plus_new, pi_minus_new, curt_cost_new)
        mOpt, solOpt, zOpt = solve_uc_with_cost(
            data, idx, c_new,
            window_size, per_bus_neutrality,
            u_init, p_init, on_time_init, off_time_init,
            extra_constr_fn=None,
            output_flag=output_flag_sp
        )
        if solOpt is None:
            return {"status": "SP_FAILED", "iter": it, "sp_status": int(mOpt.Status)}

        # ---- stopping test ----
        xOpt_fixed  = zOpt[fixed_idx]
        xOpt_splus  = zOpt[idx.splus]
        xOpt_sminus = zOpt[idx.sminus]
        xOpt_curt   = zOpt[idx.curt]

        dTx_foil = _compute_dTx_full(
            c0_fixed=c0_fixed,
            x_fixed=xFoil_fixed,
            pi_plus=pi_plus_new, pi_minus=pi_minus_new,
            x_splus=xFoil_splus, x_sminus=xFoil_sminus,
            curt_cost=curt_cost_new, x_curt=xFoil_curt,
        )
        dTx_opt = _compute_dTx_full(
            c0_fixed=c0_fixed,
            x_fixed=xOpt_fixed,
            pi_plus=pi_plus_new, pi_minus=pi_minus_new,
            x_splus=xOpt_splus, x_sminus=xOpt_sminus,
            curt_cost=curt_cost_new, x_curt=xOpt_curt,
        )

        gap = float(dTx_foil - dTx_opt)
        stop_tol = float(tol_abs + tol_rel * max(1.0, abs(dTx_opt))) # * max(1.0, abs(dTx_opt)))

        if gap < best["gap"]:
            best.update({
                "gap": gap, "iter": it,
                "pi_plus": pi_plus_new, "pi_minus": pi_minus_new,
                "curt_cost": curt_cost_new,
                "dTx_foil": dTx_foil, "dTx_opt": dTx_opt,
                "mp_details": mp_out,
            })

        if verbose:
            E_foil = _compute_total_emissions_from_sol(data, solFoil)
            E_opt  = _compute_total_emissions_from_sol(data, solOpt)
            print(
                f"[E3 it={it:02d}] gap={gap:.6g} | stop_tol={stop_tol:.3g} | "
                f"E_foil={E_foil:.3f} | E_opt={E_opt:.3f} | "
                f"C_foil={total_curtailment(solFoil):.3f} | C_opt={total_curtailment(solOpt):.3f} | "
                f"MP_obj={mp_out.get('mp_obj', None)} | bounds_used={mp_out.get('bounds_used', None)}"
            )



        # ---- foil satisfaction check on induced optimum ----
        foil_ok = True
        if emissions_mode == "total":
            foil_ok = (E_opt <= E_bar + 1e-6)
        else:
            hourly = _compute_hourly_emissions_from_sol(data, solOpt)
            foil_ok = all(hourly[int(t)] <= E_bar_hours[int(t)] + 1e-6 for t in top_hours)

        if gap <= stop_tol and not foil_ok:
            if verbose:
                if emissions_mode == "total":
                    print(f"[E3 it={it:02d}] gap small but foil violated: E_opt={E_opt:.6f} > E_bar={E_bar:.6f}. Continuing.")
                else:
                    print(f"[E3 it={it:02d}] gap small but foil violated (top-k). Continuing.")


        if gap <= stop_tol and foil_ok:
            out = {
                "status": "OPTIMAL",
                "iters": it,
                "alpha": float(alpha),
                "emissions_mode": emissions_mode,
                "top_hours": top_hours,
                "E_factual": float(E_factual),
                "E_foil": float(_compute_total_emissions_from_sol(data, solFoil)),
                "E_opt_under_new_coeffs": float(_compute_total_emissions_from_sol(data, solOpt)),
                "pi_plus_base": base_pi_plus,
                "pi_minus_base": base_pi_minus,
                "curt_cost_base": base_curt_cost,
                "pi_plus_new": pi_plus_new,
                "pi_minus_new": pi_minus_new,
                "curt_cost_new": curt_cost_new,
                "z_foil": zFoil,
                "z_opt": zOpt,
                "dTx_foil": float(dTx_foil),
                "dTx_opt": float(dTx_opt),
                "gap": float(gap),
                "stop_tol": float(stop_tol),
                "mp_details": mp_out,
                "sol_factual": solF,
                "sol_foil": solFoil,
                "sol_opt": solOpt,
            }
            if emissions_mode == "total":
                out["E_bar"] = float(E_bar)
            else:
                out["E_bar_hours"] = E_bar_hours
            return out

        # ---- cycle detection ----
        fp = _fingerprint_solution_uc(solOpt, decimals_cont=3)
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] >= 2:
            repeat_hits += 1
        if repeat_hits >= cycle_patience:
            return {
                "status": "CYCLE_OR_NO_SOLUTION",
                "iters": it,
                "alpha": float(alpha),
                "emissions_mode": emissions_mode,
                "top_hours": top_hours,
                "E_factual": float(E_factual),
                "message": "SP repeated previously seen solution; likely cycling or no explanation exists under chosen mutables/bounds.",
                "best_gap": float(best["gap"]),
                "best_iter": best["iter"],
                "best_pi_plus": best["pi_plus"],
                "best_pi_minus": best["pi_minus"],
                "best_curt_cost": best["curt_cost"],
                "best_dTx_foil": best["dTx_foil"],
                "best_dTx_opt": best["dTx_opt"],
                "best_mp_details": best["mp_details"],
            }

        # ---- add cut (dedup) ----
        zfp = (
            tuple(np.rint(solOpt["u"]).astype(int).flatten()),
            tuple(np.rint(solOpt["v"]).astype(int).flatten()),
            tuple(np.rint(solOpt["w"]).astype(int).flatten()),
            tuple(np.round(solOpt["shed"], 3).flatten()),
            tuple(np.round(solOpt["curt"], 3).flatten()),
            tuple(np.round(solOpt["splus"], 3).flatten()),
            tuple(np.round(solOpt["sminus"], 3).flatten()),
        )
        if zfp not in cuts_fp:
            cuts_fp.add(zfp)
            cuts.append(zOpt.copy())

    # --------------------------
    # 7) max iters
    # --------------------------
    return {
        "status": "MAX_ITERS",
        "iters": int(max_iters),
        "alpha": float(alpha),
        "emissions_mode": emissions_mode,
        "top_hours": top_hours,
        "E_factual": float(E_factual),
        "best_gap": float(best["gap"]),
        "best_iter": best["iter"],
        "best_pi_plus": best["pi_plus"],
        "best_pi_minus": best["pi_minus"],
        "best_curt_cost": best["curt_cost"],
        "best_dTx_foil": best["dTx_foil"],
        "best_dTx_opt": best["dTx_opt"],
        "best_mp_details": best["mp_details"],
        "message": "No convergence within max_iters. Inspect best_gap and best parameters; consider widening bounds or expanding mutable coefficients.",
    }

