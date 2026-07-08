"""Is {L14, L15} alone a feasible IEEE-14 carbon-free CE under a 2x envelope?
If yes, bisect the minimal joint expansion and componentwise minima; report the
weighted-L1 F. Also print line topology to interpret the result.
"""
import os, sys
import numpy as np

REPO = r"C:\Users\tomas\Documents\GitHub\Counterfactual-Explanations-for-optimization-problems\Mixed\UC-Experiments"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["CE_THR"] = "0.75"
os.environ["CE_SCALE_UP"] = "2.0"
from node_obbt_hpc import build_grid  # noqa: E402

g = build_grid("14")
DATA = g["DATA"]; b0 = np.asarray(g["b0"], float); bU = np.asarray(g["bU"], float)
free = list(g["free_idx"]); oracle = g["oracle"]; w = np.asarray(g["w"], float)

print("\nmutable line topology:")
for l in free:
    ln = DATA.lines[l]
    fb = getattr(ln, "from_bus", getattr(ln, "f", "?"))
    tb = getattr(ln, "to_bus", getattr(ln, "t", "?"))
    print(f"  L{l}: bus {fb} -> bus {tb}   b0={b0[l]:.1f}  w={w[l]:.3f}")
for i, r in enumerate(DATA.rens):
    print(f"  renewable {i}: bus {getattr(r, 'bus', '?')}  cap~{np.max(r.avail):.1f} MW")
for i, gen in enumerate(DATA.gens):
    print(f"  gen {i}: bus {getattr(gen, 'bus', '?')}  Pmax={gen.Pmax:.0f}  emis={gen.emission_rate:.3f}")

def tol(vp): return max(1e-3, 1e-4 * abs(vp))
def feas(b):
    vp, _, _ = oracle.solve_plain(b)
    vf, _, _ = oracle.solve_foil(b)
    if vp is None or vf is None: return False, vp, vf
    return vf <= vp + tol(vp), vp, vf
def F_of(b): return float(np.sum(w * np.abs(b - b0)))

PAIR = [14, 15]
b = b0.copy(); b[PAIR] = 2.0 * b0[PAIR]
ok, vp, vf = feas(b)
print(f"\n[pair] L14+L15 doubled, all else b0: feasible={ok}  vd-vp={vf-vp:.1f}  F={F_of(b):.2f}")

if ok:
    # joint uniform bisection on the pair
    lo, hi = 0.0, 1.0
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        b = b0.copy(); b[PAIR] = b0[PAIR] * (1.0 + mid)
        o, _, _ = feas(b)
        if o: hi = mid
        else: lo = mid
    print(f"[pair] minimal joint scale = +{hi*100:.1f}%  F={F_of(b0 + (b0*(1+hi)-b0)*np.isin(np.arange(len(b0)), PAIR)):.2f}")
    b_joint = b0.copy(); b_joint[PAIR] = b0[PAIR] * (1.0 + hi)
    print(f"[pair] joint point: L14={b_joint[14]:.2f}, L15={b_joint[15]:.2f}  F={F_of(b_joint):.2f}")
    # componentwise minima from the joint point (fix other at doubled)
    best = {}
    for l in PAIR:
        other = PAIR[1] if l == PAIR[0] else PAIR[0]
        lo_l, hi_l = b0[l], 2.0 * b0[l]
        for _ in range(10):
            mid = 0.5 * (lo_l + hi_l)
            b = b0.copy(); b[other] = 2.0 * b0[other]; b[l] = mid
            o, _, _ = feas(b)
            if o: hi_l = mid
            else: lo_l = mid
        best[l] = hi_l
        print(f"[pair] minimal L{l} with L{other} doubled: {hi_l:.2f} (+{hi_l-b0[l]:.2f} MW)")
    b_cw = b0.copy()
    for l in PAIR: b_cw[l] = best[l]
    o, vp, vf = feas(b_cw)
    print(f"[pair] both componentwise minima: feasible={o}  vd-vp={vf-vp if vf else float('nan'):.1f}  F={F_of(b_cw):.2f}")

# singles for completeness
for l in PAIR:
    b = b0.copy(); b[l] = 2.0 * b0[l]
    ok, vp, vf = feas(b)
    print(f"[single] only L{l} doubled: feasible={ok}  vd-vp={vf-vp:.1f}")
print("done.")
