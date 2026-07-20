"""Probe the IEEE 14 carbon-free CE landscape with the exact weak-feasibility
oracle (v_foil <= v_plain + 1e-3), to answer: is the F*=44.6 all-corners
incumbent genuinely (near-)optimal, or does the pipeline miss a sparser CE?

Tests:
  0. sanity: b0 infeasible, corner bU feasible; factual emissions + curtailment
  1. uniform bisection: minimal t with b(t) = b0 + t*(bU-b0) weak-feasible
  2. leave-one-out: corner with each single line reset to b0 -> feasible?
  3. per-line minimal value holding the other five at corner (bisection)
     -> componentwise-minimal CE + its weighted-L1 F
"""
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
from node_obbt_hpc import build_grid  # noqa: E402

EPS_WEAK = 1e-3

g = build_grid("14")
DATA, b0, oracle = g["DATA"], np.asarray(g["b0"], float), g["oracle"]
free = list(g["free_idx"]); bL, bU = np.asarray(g["bL"], float), np.asarray(g["bU"], float)
w = np.asarray(g["w"], float)
print(f"free lines: {free}")
print(f"b0[free] = {b0[free]}")
print(f"bU[free] = {bU[free]}  (delta = {bU[free]-b0[free]})")
print(f"w[free]  = {w[free]}")
print(f"E_factual = {g['E_factual']:.3f}")

def feas(b):
    vp, _, _ = oracle.solve_plain(b)
    vf, _, _ = oracle.solve_foil(b)
    if vp is None or vf is None:
        return False, vp, vf
    return (vf <= vp + EPS_WEAK), vp, vf

def F_of(b):
    return float(np.sum(w * np.abs(b - b0)))

def with_free(vals):
    b = b0.copy(); b[free] = vals; return b

# --- 0. sanity --------------------------------------------------------------
ok0, vp0, vf0 = feas(b0)
print(f"\n[0] b0:     feasible={ok0}  v_plain={vp0}  v_foil={vf0}")
okU, vpU, vfU = feas(bU)
print(f"[0] corner: feasible={okU}  v_plain={vpU}  v_foil={vfU}  F={F_of(bU):.3f}")

# factual curtailment
_, _, sol0 = oracle.solve_plain(b0)
if DATA.rens and "curt" in sol0:
    avail = np.vstack([r.avail for r in DATA.rens])
    print(f"[0] renewable avail={avail.sum():.1f} MWh, factual curtailment="
          f"{np.asarray(sol0['curt'], float).sum():.1f} MWh")

# --- 1. uniform bisection ----------------------------------------------------
lo, hi = 0.0, 1.0
if not okU:
    print("corner infeasible?! aborting"); sys.exit(1)
for _ in range(12):
    mid = 0.5 * (lo + hi)
    ok, _, _ = feas(with_free(b0[free] + mid * (bU[free] - b0[free])))
    if ok: hi = mid
    else:  lo = mid
b_t = with_free(b0[free] + hi * (bU[free] - b0[free]))
print(f"\n[1] minimal uniform t = {hi:.4f}  ->  F = {F_of(b_t):.3f}")

# --- 2. leave-one-out from corner ---------------------------------------------
print("\n[2] corner minus one line (that line back to b0):")
for i, l in enumerate(free):
    vals = bU[free].copy(); vals[i] = b0[l]
    ok, vp, vf = feas(with_free(vals))
    print(f"    drop L{l}: feasible={ok}  (v_plain={vp}, v_foil={vf})")

# --- 3. componentwise minimal, others at corner --------------------------------
print("\n[3] per-line minimal value, other five at corner (10-step bisection):")
best = bU[free].copy()
for i, l in enumerate(free):
    lo_i, hi_i = b0[l], bU[l]
    # only shrink if dropping fully is feasible; else bisect
    for _ in range(10):
        mid = 0.5 * (lo_i + hi_i)
        vals = bU[free].copy(); vals[i] = mid
        ok, _, _ = feas(with_free(vals))
        if ok: hi_i = mid
        else:  lo_i = mid
    best[i] = hi_i
    print(f"    L{l}: minimal = {hi_i:.3f}  (b0={b0[l]:.3f}, corner={bU[l]:.3f}, "
          f"delta=+{hi_i-b0[l]:.3f})")

# greedy pass: apply ALL componentwise minima simultaneously and check
b_greedy = with_free(best)
okG, vpG, vfG = feas(b_greedy)
print(f"\n[3] all componentwise minima together: feasible={okG}  F={F_of(b_greedy):.3f}")
if not okG:
    print("    (interactions matter — componentwise minima not jointly feasible)")
print(f"\nincumbent F (all corners) = {F_of(bU):.3f}")
print("done.")
