"""Smoke test: run DECOMP comp_mode='strongdual' on IEEE 14, 2 iters.

Checks: master builds, warm start is accepted, LB is reported.
Run with:  .../ce-env/python.exe _smoke_strongdual.py [comp_mode]
"""
import os, sys
import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")

from uc_data_loader import quick_setup
from uc_pipeline import solve_uc_with_cost_4b, make_emissions_foil_4b
from uc_decomp_4b import UCDecomp4b
from _decomp_repro_helpers import (
    get_congested_free_lines, build_b_bounds, make_line_weights, UCWeakWCEOracle,
)

COMP = sys.argv[1] if len(sys.argv) > 1 else "strongdual"
SEED = (sys.argv[2] == "seed") if len(sys.argv) > 2 else False
GRID = sys.argv[3] if len(sys.argv) > 3 else "14"
FACTOR = (float(sys.argv[4]) if (len(sys.argv) > 4 and sys.argv[4].lower() != "none")
          else None)  # McCormick mu-box factor
_seg_arg = sys.argv[5] if len(sys.argv) > 5 else "1"        # "exact" or K segments
EXACT = (_seg_arg.lower() == "exact")
SEGS = 1 if EXACT else int(_seg_arg)
OBBT = (len(sys.argv) > 6 and sys.argv[6].lower() == "obbt")
MTL  = float(sys.argv[7]) if len(sys.argv) > 7 else 120.0   # master time limit (s)
MAXIT = int(sys.argv[8]) if len(sys.argv) > 8 else 4         # CCG max_iter
MOUT = int(sys.argv[9]) if len(sys.argv) > 9 else 0          # master_output_flag (1=Gurobi log)
MFOC = int(sys.argv[10]) if len(sys.argv) > 10 else 1        # master MIPFocus (1=incumbent, 3=bound)
SINT = int(sys.argv[11]) if len(sys.argv) > 11 else 0        # seed_interp: # interior seed points b0+a(bU-b0)
MSTART = int(sys.argv[12]) if len(sys.argv) > 12 else 1      # master_multistart: # Gurobi seeds, keep max ObjBound
OREFR = (len(sys.argv) > 13 and sys.argv[13].lower() == "refresh")  # obbt_refresh
NODE = (len(sys.argv) > 14 and sys.argv[14].lower() == "node")  # node_obbt: spatial-B&B-over-b solver
NMAX = int(sys.argv[15]) if len(sys.argv) > 15 else 16          # node_obbt_max_nodes (boxes to process)
NBUD = float(sys.argv[16]) if len(sys.argv) > 16 else MTL       # node_obbt_budget (per-box TL; default = MTL)
ALPHA = 0.10
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")

_fname = {"14": "ieee14_enhanced.json", "39": "ieee39_newengland.json",
          "57": "ieee57_uc_matpower.json"}[GRID]
# Per-grid quick_setup — MUST MATCH build_decomp_notebook.py exactly, else the
# smoke solves a different instance than the notebook/B&S checkpoint.  IEEE 57
# uses carbon_price=50, slack_bus=0, and an explicit G0 warm-start (as in
# bs_7grids.ipynb); 14/39 use carbon_price=None, slack_bus=None, default init.
_setup = {
    "14": dict(carbon_price=None, voll=20_000.0, slack_bus=None),
    "39": dict(carbon_price=None, voll=20_000.0, slack_bus=None),
    "57": dict(carbon_price=50.0,  voll=500.0,   slack_bus=0),
}[GRID]
DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup(
    os.path.join(DATA_DIR, _fname), **_setup,
)
if GRID == "57":
    # Warm-start G0 exactly as in bs_7grids.ipynb / the notebook (Bug 4 fix).
    nG = len(DATA.gens)
    u_init = [0] * nG; p_init = [0.0] * nG
    on_t   = [0] * nG; off_t  = [0] * nG
    u_init[0] = 1; p_init[0] = DATA.gens[0].Pmax; on_t[0] = DATA.gens[0].UT
T = int(DATA.T)
_, solF, _ = solve_uc_with_cost_4b(
    data=DATA, idx=idx, cvec=cvec, window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
    output_flag=0,
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
# hint: prefer the B&S CE (mirrors the notebook); fall back to bU
import json as _json
_ckpt = f"bs_{GRID}_checkpoint.json"
if os.path.exists(_ckpt):
    b_hint = np.array(_json.load(open(_ckpt))["best_b"], float)
    print(f"[hint] using B&S CE from {_ckpt}")
else:
    b_hint = b0.copy(); b_hint[free_idx] = bU[free_idx]
    print("[hint] no B&S checkpoint; using bU")

print(f"=== grid=IEEE{GRID}  comp_mode={COMP}  seed={SEED}  exact={EXACT}  "
      f"obbt={OBBT}  master_TL={MTL}  max_iter={MAXIT}  "
      f"node_obbt={NODE} (max_nodes={NMAX}, box_budget={NBUD}) "
      f"free_lines={len(free_idx)} ===")
dec = UCDecomp4b(
    oracle=oracle, data=DATA, idx=idx, cvec=cvec,
    foil_extra_constr_fn=foil_fn,
    b0=b0, b_bounds=(bL, bU), b_free_idx=free_idx,
    big_M_mu=1e4, eps_obj=1e-3, max_iter=MAXIT,
    verbose=True, w=w, comp_mode=COMP,
    master_time_limit=MTL, master_output_flag=MOUT, master_mip_gap=1e-4,
    b_hat_hint=b_hint, seed_patterns=SEED, mccormick_mu_factor=FACTOR,
    mccormick_segments=SEGS, bilinear_exact=EXACT, obbt=OBBT,
    master_mip_focus=MFOC, seed_interp=SINT,
    master_multistart=MSTART, obbt_refresh=OREFR, master_seed=0,
    node_obbt=NODE, node_obbt_max_nodes=NMAX, node_obbt_budget=NBUD,
)
res = dec.run(
    window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
)
print(f"\nRESULT comp={COMP}: success={res['success']} F_opt={res['F_opt']:.4f} "
      f"LB={res['master_LB']:.4f} gap%={res['gap_pct']:.2f} "
      f"iters={res['iterations']} term={res['termination_reason']}")
