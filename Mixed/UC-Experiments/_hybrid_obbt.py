"""Hybrid certification experiment (Strategy 6, DECOMP_theoretical_strategies.md §6).

Two stages in one process:
  A. bigM master (fast LP-based search) finds a HIGH-QUALITY CE → drives the
     UPPER bound.  On IEEE 14 this historically reaches F=1.9067 (vs B&S 2.3823).
  B. strongdual + bilinear_exact + OBBT, hinted by stage-A's CE, gives a VALID
     tight LOWER bound.  Feeding the better CE as the hint ALSO tightens the
     master's F ≤ F_hint constraint, which can only RAISE ObjBound (smaller
     feasible region ⇒ higher min) — so the gap shrinks from BOTH ends.

This isolates the insight that UB (CE quality) and LB (valid certificate) are
two different axes: bigM wins the UB axis, strongdual+OBBT wins the LB axis.

Usage:
  .../ce-env/python.exe _hybrid_obbt.py [grid] [bigM_TL] [bigM_iter] [sd_TL] [sd_iter]
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

GRID      = sys.argv[1] if len(sys.argv) > 1 else "14"
BIGM_TL   = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
BIGM_ITER = int(sys.argv[3])   if len(sys.argv) > 3 else 10
SD_TL     = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
SD_ITER   = int(sys.argv[5])   if len(sys.argv) > 5 else 6
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

import json as _json
_ckpt = f"bs_{GRID}_checkpoint.json"
if os.path.exists(_ckpt):
    b_bs = np.array(_json.load(open(_ckpt))["best_b"], float)
    print(f"[hint] B&S CE from {_ckpt}", flush=True)
else:
    b_bs = b0.copy(); b_bs[free_idx] = bU[free_idx]
    print("[hint] no B&S checkpoint; using bU", flush=True)

print(f"\n{'='*70}\n=== grid=IEEE{GRID}  free_lines={len(free_idx)}  "
      f"B&S F={float(np.sum(w[free_idx]*np.abs(b_bs[free_idx]-b0[free_idx]))):.4f} ===\n"
      f"{'='*70}", flush=True)

# ---------------------------------------------------------------------------
# Stage A: bigM master — drive the UPPER bound (find the best CE).
# ---------------------------------------------------------------------------
print(f"\n----- Stage A: bigM master (find best CE)  TL={BIGM_TL}s iter={BIGM_ITER} -----",
      flush=True)
dec_bigm = UCDecomp4b(
    oracle=oracle, data=DATA, idx=idx, cvec=cvec, foil_extra_constr_fn=foil_fn,
    b0=b0, b_bounds=(bL, bU), b_free_idx=free_idx,
    big_M_mu=1e4, eps_obj=1e-3, max_iter=BIGM_ITER,
    verbose=True, w=w, comp_mode="bigM",
    master_time_limit=BIGM_TL, master_output_flag=0, master_mip_gap=1e-4,
    b_hat_hint=b_bs,
)
res_a = dec_bigm.run(
    window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
)
b_bigm = np.asarray(res_a["b_hat"], float) if res_a.get("b_hat") is not None else b_bs
F_bigm = float(res_a["F_opt"])
print(f"\n[Stage A] bigM best CE: F={F_bigm:.4f}  (B&S was "
      f"{float(np.sum(w[free_idx]*np.abs(b_bs[free_idx]-b0[free_idx]))):.4f})  "
      f"term={res_a['termination_reason']}", flush=True)

# Sanity: confirm the bigM CE is actually a CE per the oracle.
vp_b, _, _ = oracle.solve_plain(b_bigm)
vd_b, _, _ = oracle.solve_foil(b_bigm)
ce_ok = (vd_b is not None and vp_b is not None and vd_b <= vp_b + max(1e-3, 1e-4*abs(vp_b)))
print(f"[Stage A] oracle check at bigM CE: v_plain={vp_b:.2f} v_foil={vd_b:.2f} "
      f"CE_ok={ce_ok}", flush=True)

hint_for_b = b_bigm if (ce_ok and F_bigm < float("inf")) else b_bs
print(f"[Stage A] hint for stage B: F={float(np.sum(w[free_idx]*np.abs(hint_for_b[free_idx]-b0[free_idx]))):.4f}",
      flush=True)

# ---------------------------------------------------------------------------
# Stage B: strongdual + bilinear_exact + OBBT — valid LOWER bound, hinted by A.
# ---------------------------------------------------------------------------
print(f"\n----- Stage B: strongdual+exact+OBBT (valid LB)  TL={SD_TL}s iter={SD_ITER} -----",
      flush=True)
dec_sd = UCDecomp4b(
    oracle=oracle, data=DATA, idx=idx, cvec=cvec, foil_extra_constr_fn=foil_fn,
    b0=b0, b_bounds=(bL, bU), b_free_idx=free_idx,
    big_M_mu=1e4, eps_obj=1e-3, max_iter=SD_ITER,
    verbose=True, w=w, comp_mode="strongdual",
    master_time_limit=SD_TL, master_output_flag=0, master_mip_gap=1e-4,
    b_hat_hint=hint_for_b, seed_patterns=True,
    bilinear_exact=True, obbt=True,
)
res_b = dec_sd.run(
    window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
UB = min(F_bigm, float(res_b["F_opt"]))
LB = float(res_b["master_LB"])
gap = UB - LB
print(f"\n{'='*70}", flush=True)
print(f"HYBRID RESULT IEEE{GRID}:", flush=True)
print(f"  Stage A (bigM)     UB = {F_bigm:.4f}", flush=True)
print(f"  Stage B (sd+OBBT)  LB = {LB:.4f}   F_opt={res_b['F_opt']:.4f}  "
      f"term={res_b['termination_reason']}", flush=True)
print(f"  COMBINED  UB={UB:.4f}  LB={LB:.4f}  gap={gap:.4f}  "
      f"gap%={100*gap/max(UB,1e-9):.2f}", flush=True)
print(f"{'='*70}", flush=True)
