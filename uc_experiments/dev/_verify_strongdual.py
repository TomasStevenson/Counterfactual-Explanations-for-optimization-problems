"""Scratch: verify the strong-duality dual-objective expression for Fix 2.

Checks that  c^T x*  ==  dual_obj(duals, b, u^j)  on the IEEE 14 dispatch LP,
so the strong-duality equality we wire into the master has correct signs.
Run with the gurobi env:  .../ce-env/python.exe _verify_strongdual.py
"""
import os
import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE",
                      r"C:\Users\tomas\Desktop\gurobi.lic")

from uc_data_loader import quick_setup
from uc_pipeline import solve_uc_with_cost_4b
from uc_decomp_4b import UCDecomp4b

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")

DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup(
    os.path.join(DATA_DIR, "ieee14_enhanced.json"),
    carbon_price=None, voll=20_000.0, slack_bus=None,
)
nG = len(DATA.gens); nR = len(DATA.rens); nB = int(DATA.nB)
nL = len(DATA.lines); T = int(DATA.T)
print(f"IEEE14  nG={nG} nR={nR} nB={nB} nL={nL} T={T}")

# Minimal UCDecomp4b just to call _solve_dispatch_lp (oracle/foil unused there).
free_idx = list(range(nL))
dec = UCDecomp4b(
    oracle=None, data=DATA, idx=idx, cvec=cvec,
    foil_extra_constr_fn=lambda m, v: None,
    b0=b0, b_bounds=(b0.copy(), b0 * 3.0 + 1.0), b_free_idx=free_idx,
    big_M_mu=1e4, verbose=False,
)

# Real commitment pattern from the factual UC solve (dispatch-feasible at b0).
_, solF, _ = solve_uc_with_cost_4b(
    data=DATA, idx=idx, cvec=cvec,
    window_size=T, per_bus_neutrality=True,
    u_init=u_init, p_init=p_init,
    on_time_init=on_t, off_time_init=off_t, output_flag=0,
)
assert solF is not None, "factual UC infeasible"
u_j = np.round(solF["u"]).astype(int)
import sys
_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
b_hint = b0 * _scale                        # expand limits to exercise flow duals
print(f"b scale = {_scale}")

sol = dec._solve_dispatch_lp(
    b_hint, u_j, window_size=T, per_bus_neutrality=True, p_init=p_init,
)
assert sol is not None, "dispatch LP infeasible at (b_hint, u_j)"
cTx = sol["obj"]

# ---- candidate dual objective ------------------------------------------------
W = T
avail = np.array([[float(DATA.rens[r].avail[t]) for t in range(T)]
                  for r in range(nR)]) if nR else np.zeros((0, T))
# avail summed to bus
avail_bus = np.zeros((nB, T))
for r, ren in enumerate(DATA.rens):
    avail_bus[int(ren.bus)] += avail[r]
demand = np.array([[float(DATA.demand[b, t]) for t in range(T)]
                   for b in range(nB)])
Spmax = np.array([[float(DATA.Splus_max[b, t]) for t in range(T)]
                  for b in range(nB)])
Smmax = np.array([[float(DATA.Sminus_max[b, t]) for t in range(T)]
                  for b in range(nB)])

g = 0.0
# gen bounds
for gi, gen in enumerate(DATA.gens):
    Pmin, Pmax = float(gen.Pmin), float(gen.Pmax)
    for t in range(T):
        uj = float(u_j[gi, t])
        g += sol["lam_hi"][gi, t] * (-Pmax * uj)
        g += sol["lam_lo"][gi, t] * (Pmin * uj)
# ramps
for gi, gen in enumerate(DATA.gens):
    RU, RD = float(gen.RU), float(gen.RD)
    p0 = float(p_init[gi])
    g += sol["rho_up_i"][gi] * (-p0 - RU)
    g += sol["rho_dn_i"][gi] * (p0 - RD)
    for t in range(1, T):
        g += sol["rho_up"][gi, t] * (-RU)
        g += sol["rho_dn"][gi, t] * (-RD)
# balance (free dual pi) constant part = avail_bus - demand
for b in range(nB):
    for t in range(T):
        g += sol["pi"][b, t] * (avail_bus[b, t] - demand[b, t])
# flow limits: -cap*(mu_p+mu_m), cap = b_hint
for ell in range(nL):
    cap = float(b_hint[ell])
    for t in range(T):
        g += -cap * (sol["mu_p"][ell, t] + sol["mu_m"][ell, t])
# shed / curt / shift upper bounds
for b in range(nB):
    for t in range(T):
        g += sol["gsh_ub"][b, t] * (-demand[b, t])
        g += sol["gsp_ub"][b, t] * (-Spmax[b, t])
        g += sol["gsm_ub"][b, t] * (-Smmax[b, t])
for r in range(nR):
    for t in range(T):
        g += sol["gcu_ub"][r, t] * (-avail[r, t])

print(f"c^T x*    = {cTx:.6f}")
print(f"dual_obj  = {g:.6f}")
print(f"abs diff  = {abs(cTx - g):.3e}")
print(f"rel diff  = {abs(cTx - g) / max(1.0, abs(cTx)):.3e}")
print("MATCH" if abs(cTx - g) <= 1e-4 * max(1.0, abs(cTx)) else "MISMATCH")
