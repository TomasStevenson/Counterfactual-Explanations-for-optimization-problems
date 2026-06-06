"""Measure actual flow duals (mu_p, mu_m) at b0/b_hat/bU vs the McCormick
mu-box bound M_mu, to see how loose the strongdual McCormick envelope is.

Usage:  .../ce-env/python.exe _measure_flow_duals.py [grid]
"""
import os, sys, json
import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")
from uc_data_loader import quick_setup
from uc_pipeline import solve_uc_with_cost_4b, make_emissions_foil_4b
from uc_decomp_4b import UCDecomp4b
from _decomp_repro_helpers import (
    get_congested_free_lines, build_b_bounds, make_line_weights, UCWeakWCEOracle,
)

GRID = sys.argv[1] if len(sys.argv) > 1 else "39"
ALPHA = 0.10
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
_voll = {"14": 20_000.0, "39": 20_000.0, "57": 500.0}[GRID]
_fname = {"14": "ieee14_enhanced.json", "39": "ieee39_newengland.json",
          "57": "ieee57_uc_matpower.json"}[GRID]
DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup(
    os.path.join(DATA_DIR, _fname), carbon_price=None, voll=_voll, slack_bus=None,
)
T = int(DATA.T)
_, solF, _ = solve_uc_with_cost_4b(
    data=DATA, idx=idx, cvec=cvec, window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t, output_flag=0,
)
e = np.array([float(g.emission_rate) for g in DATA.gens])
foil_fn = make_emissions_foil_4b(DATA, alpha=ALPHA,
                                 E_factual=float(np.sum(e[:, None] * solF["p"])))
free_idx, util = get_congested_free_lines(solF, b0, thr=0.75)
bL, bU = build_b_bounds(b0, free_idx)
w = make_line_weights(DATA, b0, util=util)
oracle = UCWeakWCEOracle(
    data=DATA, cvec=cvec, idx=idx, window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_t=on_t, off_t=off_t,
    foil_extra_constr_fn=foil_fn, output_flag=0,
)
dec = UCDecomp4b(oracle=oracle, data=DATA, idx=idx, cvec=cvec,
                 foil_extra_constr_fn=foil_fn, b0=b0, b_bounds=(bL, bU),
                 b_free_idx=free_idx, big_M_mu=1e4, verbose=False, w=w,
                 comp_mode="strongdual")
VOLL = float(DATA.voll)
mcc = max(float(np.max(r.curt_cost)) for r in DATA.rens) if DATA.rens else 0.0
M_mu = max(1e4, 2*VOLL + 2*mcc)
print(f"=== IEEE{GRID}  M_mu (McCormick mu box) = {M_mu:.0f} ===")

b_bs = None
ckpt = f"bs_{GRID}_checkpoint.json"
if os.path.exists(ckpt):
    b_bs = np.array(json.load(open(ckpt))["best_b"], float)

b_ws = b0.copy(); b_ws[free_idx] = bU[free_idx]
scen = [("b0", b0), ("bU", b_ws)] + ([("b_hat", b_bs)] if b_bs is not None else [])
glob_max = 0.0
for tag, bb in scen:
    vp, _, sp = oracle.solve_plain(bb)
    if sp is None:
        print(f"  {tag}: plain infeasible"); continue
    u_j = np.round(sp["u"]).astype(int)
    sol = dec._solve_dispatch_lp(bb, u_j, window_size=T,
                                 per_bus_neutrality=True, p_init=p_init)
    mp = np.max(sol["mu_p"]); mm = np.max(sol["mu_m"])
    mx = max(mp, mm); glob_max = max(glob_max, mx)
    # only free-line duals matter for McCormick
    mp_f = np.max(sol["mu_p"][free_idx]) if free_idx else 0.0
    mm_f = np.max(sol["mu_m"][free_idx]) if free_idx else 0.0
    print(f"  {tag:6s}: max mu_p={mp:10.3f}  max mu_m={mm:10.3f}  "
          f"(free-line max={max(mp_f, mm_f):10.3f})")
print(f"  -> global max observed mu = {glob_max:.3f}   "
      f"M_mu/observed = {M_mu/max(glob_max,1e-9):.1f}x looser than needed")
