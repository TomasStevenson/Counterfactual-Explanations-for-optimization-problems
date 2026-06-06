"""Correctness check for comp_mode='strongdual'.

A valid CE b_BS (from Branch-and-Sandwich) MUST be feasible in the strongdual
master with its own plain-optimal KKT block.  If debug_fix_b reports INFEASIBLE
(or obj >> F(b_BS)), the strong-duality + McCormick formulation is excluding
valid CEs -> ObjBound would be an INVALID (too-high) lower bound.

Usage:  .../ce-env/python.exe _check_strongdual_valid.py \
            [grid] [comp_mode] [bigM] [factor|none] [exact|K] [obbt|noobbt]
"""
import os, sys, json
import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")

import gurobipy as gp
from gurobipy import GRB

from uc_data_loader import quick_setup
from uc_pipeline import solve_uc_with_cost_4b, make_emissions_foil_4b, _optimize_with_retry
from uc_decomp_4b import UCDecomp4b
from _decomp_repro_helpers import (
    get_congested_free_lines, build_b_bounds, make_line_weights, UCWeakWCEOracle,
)

GRID = sys.argv[1] if len(sys.argv) > 1 else "39"
COMP = sys.argv[2] if len(sys.argv) > 2 else "strongdual"
BIGM = float(sys.argv[3]) if len(sys.argv) > 3 else 1e4
FACTOR = (float(sys.argv[4]) if (len(sys.argv) > 4 and sys.argv[4].lower() != "none")
          else None)  # McCormick mu-box factor
_seg_arg = sys.argv[5] if len(sys.argv) > 5 else "1"        # "exact" or K segments
EXACT = (_seg_arg.lower() == "exact")
SEGS = 1 if EXACT else int(_seg_arg)
OBBT = (len(sys.argv) > 6 and sys.argv[6].lower() == "obbt")
OBBT_ITER = int(sys.argv[7]) if len(sys.argv) > 7 else 2   # # root OBBT passes
ALPHA = 0.10
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
_fname = {"14": "ieee14_enhanced.json", "39": "ieee39_newengland.json",
          "57": "ieee57_uc_matpower.json"}[GRID]
# Per-grid quick_setup — MUST MATCH build_decomp_notebook.py (IEEE 57 uses
# carbon_price=50, slack_bus=0, G0 warm-start; 14/39 default).
_setup = {
    "14": dict(carbon_price=None, voll=20_000.0, slack_bus=None),
    "39": dict(carbon_price=None, voll=20_000.0, slack_bus=None),
    "57": dict(carbon_price=50.0,  voll=500.0,   slack_bus=0),
}[GRID]
DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup(
    os.path.join(DATA_DIR, _fname), **_setup,
)
if GRID == "57":
    nG = len(DATA.gens)
    u_init = [0] * nG; p_init = [0.0] * nG
    on_t   = [0] * nG; off_t  = [0] * nG
    u_init[0] = 1; p_init[0] = DATA.gens[0].Pmax; on_t[0] = DATA.gens[0].UT
T = int(DATA.T)
_, solF, _ = solve_uc_with_cost_4b(
    data=DATA, idx=idx, cvec=cvec, window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t, output_flag=0,
)
e = np.array([float(g.emission_rate) for g in DATA.gens])
E_fac = float(np.sum(e[:, None] * solF["p"]))
foil_fn = make_emissions_foil_4b(DATA, alpha=ALPHA, E_factual=E_fac)
free_idx, util = get_congested_free_lines(solF, b0, thr=0.75)
bL, bU = build_b_bounds(b0, free_idx)
w = make_line_weights(DATA, b0, util=util)
oracle = UCWeakWCEOracle(
    data=DATA, cvec=cvec, idx=idx, window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_t=on_t, off_t=off_t,
    foil_extra_constr_fn=foil_fn, output_flag=0,
)

with open(f"bs_{GRID}_checkpoint.json") as fh:
    bs = json.load(fh)
b_bs = np.array(bs["best_b"], float)
F_bs = float(bs["best_F"])
print(f"=== grid=IEEE{GRID} comp={COMP}  B&S F={F_bs:.4f} ===")

# Confirm b_bs is a real CE per the oracle
vp, _, _ = oracle.solve_plain(b_bs)
vd, _, _ = oracle.solve_foil(b_bs)
print(f"oracle at b_BS: v_plain={vp:.2f}  v_foil={vd:.2f}  CE_ok={vd <= vp + 1e-3}")

print(f"big_M_mu={BIGM:g}  mccormick_mu_factor={FACTOR}  segments={SEGS}  "
      f"exact={EXACT}  obbt={OBBT}")
dec = UCDecomp4b(
    oracle=oracle, data=DATA, idx=idx, cvec=cvec, foil_extra_constr_fn=foil_fn,
    b0=b0, b_bounds=(bL, bU), b_free_idx=free_idx,
    big_M_mu=BIGM, verbose=True, w=w, comp_mode=COMP,
    b_hat_hint=b_bs, mccormick_mu_factor=FACTOR, mccormick_segments=SEGS,
    bilinear_exact=EXACT, obbt=OBBT, obbt_iter=OBBT_ITER,
)
# debug_fix_b doesn't call run(), so estimate the mu-box up front (as run() would)
if FACTOR is not None and COMP == "strongdual":
    b_ws = b0.copy(); b_ws[free_idx] = bU[free_idx]
    dec._estimate_mu_box([b0, b_bs, b_ws], T, True, p_init)
F_bs_local = dec.F(b_bs)
print(f"F(b_BS) under local weights = {F_bs_local:.4f}")

if not OBBT:
    res = dec.debug_fix_b(
        b_test=b_bs, window_size=T, per_bus_neutrality=True,
        u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
        iis_path=f"_chk_{GRID}_{COMP}.ilp", verbose=False,
    )
    print(f"debug_fix_b: feasible={res['feasible']}  obj={res['obj']}  "
          f"cut_slack={res.get('cut_slack')}")
    feas, obj_val = res["feasible"], res["obj"]
else:
    # ---- Custom validation path mirroring run()'s OBBT sequence -----------
    # OBBT must run while b is FREE (b ∈ [bL,bU]) — that's the relaxation
    # the OBBT bounds are derived from.  If we fixed b first, the OBBT LP
    # would be a strictly tighter (point) relaxation and any bound it
    # produced would also be valid, but it would NOT exercise the same
    # code path as run().  So: build master + KKT block at b_BS plain-
    # optimal pattern (b free), run OBBT, THEN fix b = b_BS and check.
    print("[validate-obbt] Building master + KKT block (b free) ...")
    vp, _, sol_p = oracle.solve_plain(b_bs)
    assert sol_p is not None, "plain UC infeasible at b_BS"
    u_k = np.round(sol_p["u"]).astype(int)
    m, mv = dec._build_master_base(T, True, u_init, p_init, on_t, off_t)
    dec._add_iteration_block(m, mv, 0, u_k, u_init, p_init, T, True)
    m.update()
    print("[validate-obbt] Running root OBBT ...")
    stats = dec._obbt_root(m, mv, [u_k], T, True, u_init, p_init, on_t, off_t)
    print(f"[validate-obbt] OBBT stats: {stats}")
    print("[validate-obbt] Fixing b = b_BS and solving ...")
    for ell in range(len(DATA.lines)):
        if ell in set(free_idx):
            m.addConstr(mv["b"][ell] == float(b_bs[ell]), name=f"fix_b[{ell}]")
    m.update()
    m.Params.OutputFlag = 0
    if EXACT:
        m.Params.NonConvex = 2
    m.Params.TimeLimit = 600.0
    _optimize_with_retry(m)
    st = m.Status
    obj_val = float(m.ObjVal) if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and m.SolCount > 0 else None
    feas = st in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and m.SolCount > 0
    print(f"[validate-obbt] solve status={st}  obj={obj_val}  SolCount={m.SolCount}")
    if not feas:
        iis_path = f"_chk_{GRID}_{COMP}_obbt.ilp"
        try:
            m.computeIIS(); m.write(iis_path)
            print(f"[validate-obbt] IIS → {iis_path}")
        except Exception as e:
            print(f"[validate-obbt] IIS write failed: {e}")
    m.dispose()

if feas and obj_val is not None:
    ok = abs(obj_val - F_bs_local) <= 1e-2 * max(1.0, F_bs_local)
    print("VALID  (b_BS feasible, obj == F(b_BS))" if ok else
          f"SUSPECT (obj={obj_val:.4f} != F(b_BS)={F_bs_local:.4f})")
else:
    print("INVALID — strongdual master EXCLUDES the known CE b_BS (over-constrained)")
