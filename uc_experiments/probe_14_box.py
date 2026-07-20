"""Does a feasible IEEE-14 carbon-free CE exist under a larger box / wider
mutable set? Check weak feasibility (vd <= vp + max(1e-3, 1e-4*|vp|)) at the
box corner for several (CE_THR, CE_SCALE_UP) configs; if feasible, bisect the
minimal uniform expansion to gauge how sparse/cheap a real CE could be.
"""
import os, sys, importlib
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

CONFIGS = [  # (thr, scale_up)
    (0.75, 1.5),
    (0.75, 2.0),
    (0.50, 1.2),
    (0.50, 1.5),
]

def tol(vp):
    return max(1e-3, 1e-4 * abs(vp))

for thr, su in CONFIGS:
    os.environ["CE_THR"] = str(thr)
    os.environ["CE_SCALE_UP"] = str(su)
    import node_obbt_hpc
    importlib.reload(node_obbt_hpc)
    g = node_obbt_hpc.build_grid("14")
    b0 = np.asarray(g["b0"], float); bU = np.asarray(g["bU"], float)
    free = list(g["free_idx"]); oracle = g["oracle"]; w = np.asarray(g["w"], float)

    def feas(b):
        vp, _, _ = oracle.solve_plain(b)
        vf, _, _ = oracle.solve_foil(b)
        if vp is None or vf is None: return False, vp, vf
        return vf <= vp + tol(vp), vp, vf

    def F_of(b): return float(np.sum(w * np.abs(b - b0)))

    ok, vp, vf = feas(bU)
    gap = (vf - vp) if (vp is not None and vf is not None) else float("nan")
    print(f"\n=== thr={thr} su={su}: {len(free)} mutable lines {free}")
    print(f"    corner: feasible={ok}  vd-vp={gap:.1f}  (tol={tol(vp):.1f})  F(corner)={F_of(bU):.2f}")
    if not ok:
        continue
    lo, hi = 0.0, 1.0
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        b = b0.copy(); b[free] = b0[free] + mid * (bU[free] - b0[free])
        o, _, _ = feas(b)
        if o: hi = mid
        else: lo = mid
    b_min = b0.copy(); b_min[free] = b0[free] + hi * (bU[free] - b0[free])
    print(f"    minimal uniform t={hi:.3f}  F={F_of(b_min):.2f}")
    # leave-one-out from corner: which lines are essential?
    ess = []
    for i, l in enumerate(free):
        b = bU.copy(); b[l] = b0[l]
        o, _, _ = feas(b)
        if not o: ess.append(l)
    print(f"    essential lines (corner minus that line infeasible): {ess}")
print("\ndone.")
