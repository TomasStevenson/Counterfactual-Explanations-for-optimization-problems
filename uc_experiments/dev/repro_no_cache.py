"""Like repro_minimal_4a1.py but builds data FROM SCRATCH (no pickle round-trip).

If this PASSES (<5s) while repro_minimal_4a1.py FAILS, the pickle cache is
corrupting one of DATA / idx / cvec / b0 / b_hat / ...
"""
from __future__ import annotations
import os, time, json, warnings

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")
warnings.filterwarnings("ignore")

import numpy as np
import gurobipy as gp

from uc_pipeline import solve_uc_with_cost_4b, make_emissions_foil_4b
from uc_data_loader import quick_setup
from uc_decomp_4b import UCDecomp4b
from uc_master_relax_4b import build_uc_relax_master_varfmax_4b
from uc_pipeline import set_objective_from_cvec

from _decomp_repro_helpers import (
    UCWeakWCEOracle,
    get_congested_free_lines,
    build_b_bounds,
    make_line_weights,
)

ALPHA = 0.10
UTIL_THR = 0.75
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")

# Build everything fresh (mirror cache_setup.py but no pickle)
print("[fresh] quick_setup ...")
DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup(
    os.path.join(DATA_DIR, "ieee14_enhanced.json"),
    carbon_price=None, voll=20_000.0, slack_bus=None,
)
print("[fresh] factual UC ...")
_, solF, _ = solve_uc_with_cost_4b(
    data=DATA, idx=idx, cvec=cvec,
    window_size=int(DATA.T), per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
    output_flag=0,
)
e = np.array([float(gen.emission_rate) for gen in DATA.gens])
E_factual = float(np.sum(e[:, None] * solF["p"]))
b_free, util = get_congested_free_lines(solF, b0, thr=UTIL_THR)
bL, bU = build_b_bounds(b0, b_free)
w = make_line_weights(DATA, b0, util=util)

# foil_fn with no-shed wrapper (matching notebook IEEE 14)
foil_base = make_emissions_foil_4b(DATA, alpha=ALPHA, E_factual=E_factual)
def foil_fn(m, var):
    foil_base(m, var)
    m.addConstr(
        gp.quicksum(var['shed'][b, t]
                    for b in range(DATA.nB)
                    for t in range(int(DATA.T))) == 0,
        name='foil_no_shed',
    )

# big_M_mu
print("[fresh] big_M_mu calibration ...")
_bL = b0 * 0.5; _bU = b0 * 2.0
_m_relax, _var_relax, _ = build_uc_relax_master_varfmax_4b(
    data=DATA, idx=idx, cvec=cvec,
    window_size=int(DATA.T), per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
    b0=b0, node_bL=_bL, node_bU=_bU,
    b_free_idx=list(range(len(DATA.lines))), output_flag=0,
)
set_objective_from_cvec(_m_relax, _var_relax, idx, cvec)
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
big_M_mu = 10.0 * max(_max_pi, 1.0)
print(f"[fresh] big_M_mu={big_M_mu:.4f}")

# b_hat from B&S
with open("bs_14_checkpoint.json") as fh:
    bs_cp = json.load(fh)
b_hat = np.array(bs_cp["best_b"], float)
print(f"[fresh] B&S F_opt={bs_cp['best_F']:.4f}")

# Oracle
oracle = UCWeakWCEOracle(
    data=DATA, cvec=cvec, idx=idx,
    window_size=int(DATA.T), per_bus_neutrality=True,
    u_init=u_init, p_init=p_init, on_t=on_t, off_t=off_t,
    foil_extra_constr_fn=foil_fn, output_flag=0,
)

# Construct tester with MINIMAL args (mirroring 4a notebook)
_tester = UCDecomp4b(
    oracle=oracle, data=DATA, idx=idx, cvec=cvec,
    foil_extra_constr_fn=foil_fn,
    b0=b0, b_bounds=(bL, bU), b_free_idx=b_free,
    big_M_mu=big_M_mu, output_flag=0, verbose=True, w=w,
    big_M_multiplier=1.0, master_output_flag=0,
    comp_mode="bigM",
)

print()
print("=" * 60)
print("[fresh] debug_fix_b (self-consistent, b_hat)")
print("=" * 60)
t0 = time.time()
_dbg = _tester.debug_fix_b(
    b_test=b_hat, u_j_override=None,
    window_size=int(DATA.T), per_bus_neutrality=True,
    u_init=u_init, p_init=p_init,
    on_time_init=on_t, off_time_init=off_t,
    iis_path="repro_no_cache.ilp",
    return_values=False, verbose=True,
)
elapsed = time.time() - t0
print(f"\n[fresh] debug_fix_b done: {elapsed:.2f}s, feasible={_dbg['feasible']}, obj={_dbg['obj']}")
if _dbg['feasible'] and elapsed < 5.0:
    print("[fresh] PASS — pickle cache was the bug")
elif _dbg['feasible']:
    print(f"[fresh] SLOW — feasible but {elapsed:.1f}s")
else:
    print(f"[fresh] FAIL — pickle is NOT the bug; some other script/Gurobi context issue")
