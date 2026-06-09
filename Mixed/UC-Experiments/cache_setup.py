"""One-shot script: replicate decomp_3grids.ipynb Sections 0-3 for IEEE 14,
solve B&S CE oracle, pickle everything needed by dev/repro_warmstart.py.

Run from this directory:
    python cache_setup.py

Produces _cache_decomp_14.pkl (~5 MB).  Re-run only when DATA changes.
"""
from __future__ import annotations

import os
import json
import pickle
import warnings
import time

import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")
warnings.filterwarnings("ignore")

from uc_pipeline import solve_uc_with_cost_4b, make_emissions_foil_4b
from uc_data_loader import quick_setup

from _decomp_repro_helpers import (
    UCWeakWCEOracle,
    get_congested_free_lines,
    build_b_bounds,
    make_line_weights,
    make_foil_fn_14,
)

# --------- knobs (must match notebook) ---------
DATA_DIR = r"C:\Users\tomas\Documents\GitHub\Counterfactual-Explanations-for-optimization-problems\Mixed\UC-Experiments\Data"
ALPHA = 0.10
UTIL_THR = 0.75
BS_CKPT = "bs_14_checkpoint.json"
CACHE_PATH = "_cache_decomp_14.pkl"

print("=" * 60)
print("DECOMP cache_setup.py  —  IEEE 14")
print("=" * 60)
print(f"GRB_LICENSE_FILE = {os.environ.get('GRB_LICENSE_FILE')}")
print(f"DATA_DIR = {DATA_DIR}")
print()

# --------- Section 1: load dataset ---------
t0 = time.time()
print("[1/6] quick_setup ...")
DATA_14, idx_14, cvec_14, b0_14, u_init_14, p_init_14, on_t_14, off_t_14 = quick_setup(
    os.path.join(DATA_DIR, "ieee14_enhanced.json"),
    carbon_price=None, voll=20_000.0, slack_bus=None,
)
print(f"      OK ({time.time()-t0:.1f}s)  "
      f"buses={DATA_14.nB} lines={len(DATA_14.lines)} gens={len(DATA_14.gens)} T={int(DATA_14.T)}")

# --------- Section 2: factual UC ---------
t0 = time.time()
print("[2/6] factual UC ...")
_, solF_14, zF_14 = solve_uc_with_cost_4b(
    data=DATA_14, idx=idx_14, cvec=cvec_14,
    window_size=int(DATA_14.T), per_bus_neutrality=True,
    u_init=u_init_14, p_init=p_init_14,
    on_time_init=on_t_14, off_time_init=off_t_14, output_flag=0,
)
assert solF_14 is not None, "factual UC infeasible"
e_14 = np.array([float(gen.emission_rate) for gen in DATA_14.gens])
E_factual_14 = float(np.sum(e_14[:, None] * solF_14["p"]))
b_free_idx_14, util_14 = get_congested_free_lines(solF_14, b0_14, thr=UTIL_THR)
bL_14, bU_14 = build_b_bounds(b0_14, b_free_idx_14)
w_14 = make_line_weights(DATA_14, b0_14, util=util_14)
print(f"      OK ({time.time()-t0:.1f}s)  "
      f"E_factual={E_factual_14:.2f}  free_lines={len(b_free_idx_14)}")

# --------- Section 2b: rebuild foil_fn_14 (IEEE 14 has the no-shed wrapper) ---
foil_fn_14 = make_foil_fn_14(DATA_14, E_factual_14, ALPHA)

# --------- Section 3: big_M_mu calibration ---------
t0 = time.time()
print("[3/6] big_M_mu calibration ...")
from uc_master_relax_4b import build_uc_relax_master_varfmax_4b
from uc_pipeline import set_objective_from_cvec
_bL_relax = b0_14 * 0.5
_bU_relax = b0_14 * 2.0
_b_all = list(range(len(DATA_14.lines)))
_m_relax, _var_relax, _bcap_relax = build_uc_relax_master_varfmax_4b(
    data=DATA_14, idx=idx_14, cvec=cvec_14,
    window_size=int(DATA_14.T), per_bus_neutrality=True,
    u_init=u_init_14, p_init=p_init_14, on_time_init=on_t_14, off_time_init=off_t_14,
    b0=b0_14, node_bL=_bL_relax, node_bU=_bU_relax,
    b_free_idx=_b_all, output_flag=0,
)
set_objective_from_cvec(_m_relax, _var_relax, idx_14, cvec_14)
_m_relax.optimize()
_max_pi = 0.0
if _m_relax.Status == 2:
    for c in _m_relax.getConstrs():
        nm = c.ConstrName
        if nm.startswith("foil_fmax") or nm.startswith("fmax") or nm.startswith("fmin"):
            try:
                _max_pi = max(_max_pi, abs(c.Pi))
            except Exception:
                pass
_m_relax.dispose()
big_M_mu_14 = 10.0 * max(_max_pi, 1.0)
print(f"      OK ({time.time()-t0:.1f}s)  max|dual_fmax|={_max_pi:.2f}  big_M_mu={big_M_mu_14:.2f}")

# --------- Section 4: oracle construction (no solve here) ---------
oracle_14 = UCWeakWCEOracle(
    data=DATA_14, cvec=cvec_14, idx=idx_14,
    window_size=int(DATA_14.T), per_bus_neutrality=True,
    u_init=u_init_14, p_init=p_init_14, on_t=on_t_14, off_t=off_t_14,
    foil_extra_constr_fn=foil_fn_14, output_flag=0,
)
print("[4/6] oracle constructed (foil at b0 typically infeasible for IEEE 14 — skipped)")

# --------- Section 5: load B&S checkpoint and cache foil(b_hat) ---------
t0 = time.time()
print("[5/6] B&S checkpoint + oracle.solve_foil at b_hat ...")
assert os.path.exists(BS_CKPT), f"missing {BS_CKPT} — run B&S first"
with open(BS_CKPT) as fh:
    _bs_cp = json.load(fh)
b_hat_14 = np.array(_bs_cp["best_b"], float)
F_bs_14 = float(_bs_cp["best_F"])
vd_hint_14, _, solFoil_bhat_14 = oracle_14.solve_foil(b_hat_14)
assert solFoil_bhat_14 is not None, "foil at b_hat infeasible — B&S CE invalid?"
print(f"      OK ({time.time()-t0:.1f}s)  F_bs={F_bs_14:.4f}  v_foil(b_hat)={vd_hint_14:.2f}")

# --------- Section 6: pickle ---------
t0 = time.time()
print("[6/6] pickling cache ...")
cache = {
    "DATA_14":        DATA_14,
    "idx_14":         idx_14,
    "cvec_14":        cvec_14,
    "b0_14":          b0_14,
    "u_init_14":      u_init_14,
    "p_init_14":      p_init_14,
    "on_t_14":        on_t_14,
    "off_t_14":       off_t_14,
    "solF_14":        solF_14,
    "E_factual_14":   E_factual_14,
    "b_free_idx_14":  b_free_idx_14,
    "util_14":        util_14,
    "bL_14":          bL_14,
    "bU_14":          bU_14,
    "w_14":           w_14,
    "big_M_mu_14":    big_M_mu_14,
    "b_hat_14":       b_hat_14,
    "F_bs_14":        F_bs_14,
    "solFoil_bhat_14": solFoil_bhat_14,
    "vd_hint_14":     vd_hint_14,
    "ALPHA":          ALPHA,
    "UTIL_THR":       UTIL_THR,
}
with open(CACHE_PATH, "wb") as fh:
    pickle.dump(cache, fh)
size_kb = os.path.getsize(CACHE_PATH) / 1024
print(f"      OK ({time.time()-t0:.1f}s)  wrote {CACHE_PATH}  ({size_kb:.1f} KB)")
print()
print("=" * 60)
print("Cache built. Now run dev/repro_warmstart.py for fast iteration.")
print("=" * 60)
