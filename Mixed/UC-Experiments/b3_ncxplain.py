import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _compute_dTx(
    *,
    c0_fixed: np.ndarray,
    x_fixed: np.ndarray,
    pi_plus: np.ndarray,   # (nB,T)
    pi_minus: np.ndarray,  # (nB,T)
    x_splus: np.ndarray,   # (nB,T)
    x_sminus: np.ndarray,  # (nB,T)
) -> float:
    return float(np.dot(c0_fixed, x_fixed) + np.sum(pi_plus * x_splus) + np.sum(pi_minus * x_sminus))

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

    curt  = np.round(sol["curt"].astype(float), decimals_cont).flatten()
    splus = np.round(sol["splus"].astype(float), decimals_cont).flatten()
    sminus= np.round(sol["sminus"].astype(float), decimals_cont).flatten()

    # Include objective too (rounded) as extra safety
    obj = float(sol["obj"])
    obj_r = round(obj, 4)

    return (tuple(u), tuple(v), tuple(w), tuple(curt), tuple(splus), tuple(sminus), obj_r)


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
    Dp = mp.addVars(nB, T, lb=float(price_lb), ub=float(price_ub), vtype=GRB.CONTINUOUS, name="pi_plus")
    Dm = mp.addVars(nB, T, lb=float(price_lb), ub=float(price_ub), vtype=GRB.CONTINUOUS, name="pi_minus")

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
        lp_path = f"{infeas_iis_prefix}.lp"
        ilp_path = f"{infeas_iis_prefix}.ilp"
        mp.write(lp_path)
        mp.computeIIS()
        mp.write(ilp_path)
        return {
            "status": "MP_INFEASIBLE",
            "mp_status": int(mp.Status),
            "message": f"MP infeasible. Wrote {lp_path} and IIS {ilp_path}.",
        }

    if mp.Status == GRB.TIME_LIMIT and mp.SolCount == 0:
        lp_path = f"{infeas_iis_prefix}_nosol.lp"
        mp.write(lp_path)
        return {
            "status": "MP_NO_SOLUTION",
            "mp_status": int(mp.Status),
            "message": f"MP hit TIME_LIMIT with no solution. Wrote {lp_path}.",
        }

    if mp.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        lp_path = f"{infeas_iis_prefix}_status{int(mp.Status)}.lp"
        mp.write(lp_path)
        return {
            "status": "MP_FAILED",
            "mp_status": int(mp.Status),
            "message": f"MP failed with status={int(mp.Status)}. Wrote {lp_path}.",
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




def _solve_b3_mp_with_auto_bounds(
    *,
    nB: int,
    nR: int,
    T: int,
    base_pi_plus: np.ndarray,
    base_pi_minus: np.ndarray,
    base_curt_cost: np.ndarray,
    cuts: list,
    fixed_idx: np.ndarray,
    c0_fixed: np.ndarray,
    xFoil_fixed: np.ndarray,
    xFoil_splus: np.ndarray,
    xFoil_sminus: np.ndarray,
    xFoil_curt: np.ndarray,
    idx,
    # bounds
    price_lb: float,
    price_ub: float,
    curt_lb: float,
    curt_ub: float,
    # objective weights
    weight_prices: float,
    weight_curt: float,
    # controls
    mp_time_limit: float,
    output_flag_mp: int,
    it: int,
    auto_expand_bounds: bool = True,
):
    """
    Tries MP with provided bounds; if infeasible, expand:
      - prices: symmetric around 0 (ensures negative allowed)
      - curt_cost: keep lb>=0, increase ub multiplicatively
    Final fallback: elastic-cuts diagnostic solve.
    """
    tried = []

    def _try(lb_p, ub_p, lb_c, ub_c, elastic=False):
        tried.append((lb_p, ub_p, lb_c, ub_c, elastic))
        tag = f"iter{it}_p[{lb_p:g},{ub_p:g}]_c[{lb_c:g},{ub_c:g}]" + ("_elastic" if elastic else "")
        return _build_and_solve_b3_mp(
            nB=nB, nR=nR, T=T,
            base_pi_plus=base_pi_plus, base_pi_minus=base_pi_minus,
            base_curt_cost=base_curt_cost,
            cuts=cuts,
            fixed_idx=fixed_idx,
            c0_fixed=c0_fixed,
            xFoil_fixed=xFoil_fixed,
            xFoil_splus=xFoil_splus,
            xFoil_sminus=xFoil_sminus,
            xFoil_curt=xFoil_curt,
            idx=idx,
            price_lb=lb_p, price_ub=ub_p,
            curt_lb=lb_c, curt_ub=ub_c,
            weight_prices=weight_prices,
            weight_curt=weight_curt,
            mp_time_limit=mp_time_limit,
            output_flag_mp=output_flag_mp,
            infeas_iis_prefix=f"debug_B3_MP_{tag}",
            allow_elastic_cuts=elastic,
        )

    # 1) initial try
    out = _try(float(price_lb), float(price_ub), float(curt_lb), float(curt_ub), elastic=False)
    if out.get("status") == "OK":
        out["bounds_used"] = (float(price_lb), float(price_ub), float(curt_lb), float(curt_ub))
        out["tried"] = tried
        return out

    if out.get("status") == "MP_STRUCTURALLY_INFEASIBLE":
        out["tried"] = tried
        return out

    if not auto_expand_bounds:
        out["tried"] = tried
        return out

    # 2) expand prices symmetrically; expand curt_ub upward; keep curt_lb >= 0
    ubp0 = max(abs(price_lb), abs(price_ub), 1.0)
    ubc0 = max(float(curt_ub), float(np.max(base_curt_cost) if base_curt_cost.size else 1.0), 1.0)

    candidates = [
        (-ubp0,  ubp0,   max(0.0, curt_lb),  2.0 * ubc0),
        (-5*ubp0, 5*ubp0, max(0.0, curt_lb), 10.0 * ubc0),
        (-25*ubp0, 25*ubp0, max(0.0, curt_lb), 50.0 * ubc0),
        (-125*ubp0, 125*ubp0, max(0.0, curt_lb), 250.0 * ubc0),
    ]

    for (lbp, ubp, lbc, ubc) in candidates:
        out2 = _try(lbp, ubp, lbc, ubc, elastic=False)
        if out2.get("status") == "OK":
            out2["bounds_used"] = (lbp, ubp, lbc, ubc)
            out2["tried"] = tried
            return out2
        if out2.get("status") == "MP_STRUCTURALLY_INFEASIBLE":
            out2["tried"] = tried
            return out2

    # 3) elastic diagnostic
    lbp, ubp, lbc, ubc = candidates[-1]
    out3 = _try(lbp, ubp, lbc, ubc, elastic=True)
    out3["tried"] = tried
    if out3.get("status") == "OK":
        out3["status"] = "MP_ELASTIC_OK"
        out3["message"] = (
            "Solved MP only after adding elastic cut slacks. "
            "If max_cut_slack > 0, exact MP is infeasible under all tried bounds."
        )
        out3["bounds_used"] = (lbp, ubp, lbc, ubc)
    return out3


def run_B3_ncxplain_shift_prices(
    data,                         # NetworkUCData
    window_size: int,
    per_bus_neutrality: bool,
    alpha: float,
    price_lb: float = 0.0,
    price_ub: float = 100.0,
    curt_lb: float = 0.0,          # <<< NEW: bounds for curtailment penalty
    curt_ub: float = 100.0,        # <<< NEW
    weight_prices: float = 1.0,    # <<< NEW: weight on |Δpi|
    weight_curt: float = 1.0,      # <<< NEW: weight on |Δcurt_cost|
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
    B3 NCXplain, extended: mutable coefficients are:
      - pi_plus, pi_minus (shift prices)
      - curt_cost (curtailment penalty/compensation)

    Requires existing functions in your pipeline:
      - build_index_map_network_uc
      - default_initial_conditions
      - build_cost_vector_network_uc
      - solve_uc_with_cost
      - total_curtailment
    """

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
        carbon_price=data.carbon_price,
        no_load_cost=no_load_cost_vec,
        su_cost=su_cost_vec,
        sd_cost=sd_cost_vec,
        curt_cost=curt_cost_mat,
        pi_plus=data.pi_plus,
        pi_minus=data.pi_minus,
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

    C_factual = float(total_curtailment(solF))
    C_bar     = float(alpha * C_factual)

    # --------------------------
    # 3) foil SP (curt <= C_bar)
    # --------------------------
    def foil_constraint(m, var):
        curt = var["curt"]
        expr = gp.quicksum(curt[r, t] for r in range(nR) for t in range(T))
        eps_foil = 1e-4
        m.addConstr(expr <= C_bar + eps_foil, name="foil_curtailment_cap")

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
            "C_factual": C_factual,
            "C_bar": C_bar,
            "message": "Foil infeasible (curtailment cap too tight)."
        }

    # --------------------------
    # 4) mutable vs fixed indices
    #    mutable: splus/sminus prices + curt_cost
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
    # 5) cuts + dedup + cycle
    # --------------------------
    cuts = []
    cuts_fp = set()

    if seed_with_factual_cut:
        cuts.append(zF.copy())
        zfp0 = (
            tuple(np.rint(solF["u"]).astype(int).flatten()),
            tuple(np.rint(solF["v"]).astype(int).flatten()),
            tuple(np.rint(solF["w"]).astype(int).flatten()),
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
            mp_out["C_factual"] = C_factual
            mp_out["C_bar"] = C_bar
            return mp_out

        pi_plus_new   = mp_out["pi_plus_new"]
        pi_minus_new  = mp_out["pi_minus_new"]
        curt_cost_new = mp_out["curt_cost_new"]

        # ---- SP under new coefficients ----
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

        dTx_foil = _compute_dTx(
            c0_fixed=c0_fixed,
            x_fixed=xFoil_fixed,
            pi_plus=pi_plus_new, pi_minus=pi_minus_new,
            x_splus=xFoil_splus, x_sminus=xFoil_sminus,
            curt_cost=curt_cost_new, x_curt=xFoil_curt,
        )
        dTx_opt = _compute_dTx(
            c0_fixed=c0_fixed,
            x_fixed=xOpt_fixed,
            pi_plus=pi_plus_new, pi_minus=pi_minus_new,
            x_splus=xOpt_splus, x_sminus=xOpt_sminus,
            curt_cost=curt_cost_new, x_curt=xOpt_curt,
        )

        gap = float(dTx_foil - dTx_opt)
        stop_tol = float(tol_abs + tol_rel * max(1.0, abs(dTx_opt)))

        if gap < best["gap"]:
            best.update({
                "gap": gap, "iter": it,
                "pi_plus": pi_plus_new, "pi_minus": pi_minus_new,
                "curt_cost": curt_cost_new,
                "dTx_foil": dTx_foil, "dTx_opt": dTx_opt,
                "mp_details": mp_out,
            })

        if verbose:
            C_foil = float(total_curtailment(solFoil))
            C_opt  = float(total_curtailment(solOpt))
            bnd = mp_out.get("bounds_used", None)
            print(
                f"[B3+ it={it:02d}] gap={gap:.6g} | stop_tol={stop_tol:.3g} | "
                f"C_foil={C_foil:.3f} | C_opt={C_opt:.3f} | MP_obj={mp_out.get('mp_obj', None)} | bounds_used={bnd}"
            )

        if gap <= stop_tol:
            return {
                "status": "OPTIMAL",
                "iters": it,
                "C_factual": C_factual,
                "C_bar": C_bar,
                "C_foil": float(total_curtailment(solFoil)),
                "C_opt_under_new_coeffs": float(total_curtailment(solOpt)),
                "pi_plus_base": base_pi_plus,
                "pi_minus_base": base_pi_minus,
                "curt_cost_base": base_curt_cost,
                "pi_plus_new": pi_plus_new,
                "pi_minus_new": pi_minus_new,
                "curt_cost_new": curt_cost_new,
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
            }

        # ---- cycle detection ----
        fp = _fingerprint_solution_uc(solOpt, decimals_cont=3)
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] >= 2:
            repeat_hits += 1
        if repeat_hits >= cycle_patience:
            return {
                "status": "CYCLE_OR_NO_SOLUTION",
                "iters": it,
                "C_factual": C_factual,
                "C_bar": C_bar,
                "message": "SP repeated previously seen solution; likely cycling or no explanation exists even with curt_cost mutable.",
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
            tuple(np.round(solOpt["curt"], 3).flatten()),
            tuple(np.round(solOpt["splus"], 3).flatten()),
            tuple(np.round(solOpt["sminus"], 3).flatten()),
        )
        if zfp not in cuts_fp:
            cuts_fp.add(zfp)
            cuts.append(zOpt.copy())

    # --------------------------
    # 7) max iters return
    # --------------------------
    return {
        "status": "MAX_ITERS",
        "iters": max_iters,
        "C_factual": C_factual,
        "C_bar": C_bar,
        "best_gap": float(best["gap"]),
        "best_iter": best["iter"],
        "best_pi_plus": best["pi_plus"],
        "best_pi_minus": best["pi_minus"],
        "best_curt_cost": best["curt_cost"],
        "best_dTx_foil": best["dTx_foil"],
        "best_dTx_opt": best["dTx_opt"],
        "best_mp_details": best["mp_details"],
        "message": (
            "No convergence within max_iters. Inspect best_gap and best parameters; "
            "consider widening bounds or adding more mutable coefficients (e.g., demand, line limits)."
        ),
    }
