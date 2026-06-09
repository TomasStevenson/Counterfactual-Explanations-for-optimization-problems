"""Mirror 4a.2 exactly: just iter1_pattern + debug_fix_b on a fresh tester.

If THIS times out, the bug is not in the notebook context at all — it's
something deeper (maybe how the script-loaded uc_decomp_4b differs from the
notebook-loaded one, or Gurobi env subtlety).

If it solves in <1s, the bug is somewhere else in repro_warmstart.py's setup.
"""
from __future__ import annotations
import os, pickle, time

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")

import numpy as np
from uc_decomp_4b import UCDecomp4b
from _decomp_repro_helpers import UCWeakWCEOracle, make_foil_fn_14

with open("_cache_decomp_14.pkl", "rb") as fh:
    c = pickle.load(fh)

DATA = c["DATA_14"]; idx = c["idx_14"]; cvec = c["cvec_14"]; b0 = c["b0_14"]
u_init = c["u_init_14"]; p_init = c["p_init_14"]
on_t = c["on_t_14"]; off_t = c["off_t_14"]
E_factual = c["E_factual_14"]; b_free = c["b_free_idx_14"]
bL = c["bL_14"]; bU = c["bU_14"]; w = c["w_14"]
big_M_mu = c["big_M_mu_14"]; b_hat = c["b_hat_14"]; F_bs = c["F_bs_14"]
ALPHA = c["ALPHA"]; W = int(DATA.T)

foil_fn = make_foil_fn_14(DATA, E_factual, ALPHA)
oracle = UCWeakWCEOracle(
    data=DATA, cvec=cvec, idx=idx,
    window_size=W, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_t=on_t, off_t=off_t,
    foil_extra_constr_fn=foil_fn, output_flag=0,
)

print(f"B&S F_opt={F_bs:.4f}")

# Mimic 4a.2 cell verbatim: minimal constructor args
print("=" * 60)
print("[mimic 4a.2] Constructing fresh tester with NO extras ...")
print("=" * 60)
_tester = UCDecomp4b(
    oracle=oracle, data=DATA, idx=idx, cvec=cvec,
    foil_extra_constr_fn=foil_fn,
    b0=b0, b_bounds=(bL, bU), b_free_idx=b_free,
    big_M_mu=big_M_mu, output_flag=0, verbose=True, w=w,
    big_M_multiplier=1.0, master_output_flag=0,
    comp_mode="bigM",
)

t0 = time.time()
print("\n[mimic 4a.2] iter1_pattern ...")
_b_1, _u_1, _F_1 = _tester.iter1_pattern(
    window_size=W, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
)
print(f"[mimic 4a.2] iter1_pattern done: {time.time()-t0:.2f}s, F_1={_F_1:.4f}, sum(u_1)={int(_u_1.sum())}")

t0 = time.time()
print("\n[mimic 4a.2] debug_fix_b (cross-pattern, b_hat) ...")
_dbg = _tester.debug_fix_b(
    b_test=b_hat, u_j_override=_u_1,
    window_size=W, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init,
    on_time_init=on_t, off_time_init=off_t,
    iis_path="repro_minimal_4a2.ilp",
    return_values=False,    # don't capture values; just test feasibility
    verbose=True,
)
elapsed = time.time() - t0
print(f"\n[mimic 4a.2] debug_fix_b done: {elapsed:.2f}s, feasible={_dbg['feasible']}, obj={_dbg['obj']}")
if _dbg['feasible'] and elapsed < 5.0:
    print("[mimic 4a.2] PASS — script version solves as fast as notebook 4a.2")
elif _dbg['feasible']:
    print(f"[mimic 4a.2] SLOW — feasible but {elapsed:.1f}s vs notebook's <1s")
else:
    print(f"[mimic 4a.2] FAIL — debug_fix_b returned infeasible/timeout even in this minimal context")
