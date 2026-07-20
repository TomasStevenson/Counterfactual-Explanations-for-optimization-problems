"""Render fig_decomp_14 from the 2x-envelope run (pipeline_14box2/pipeline_14.json),
in the same paper style as make_ce0_figures.py (black Total, red dashed Demand,
orange 'Changed by CE' bars). Writes fig_decomp_14_box2.png to this dir.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
from node_obbt_hpc import build_grid  # noqa: E402

OUTDIR = os.path.dirname(os.path.abspath(__file__))
GEN_COLORS = plt.get_cmap("tab10").colors
REN_GREEN  = "#2ca02c"


def _p_matrix(sol, nG, T):
    p_raw = sol["p"]
    if isinstance(p_raw, dict):
        return np.array([[float(p_raw[g, t]) for t in range(T)] for g in range(nG)])
    p = np.asarray(p_raw, dtype=float)
    return p.T if p.shape == (T, nG) else p


def _dispatch_panel(ax, DATA, sol, title):
    nG, T = len(DATA.gens), int(DATA.T)
    hrs = np.arange(T)
    p = _p_matrix(sol, nG, T)
    bottom = np.zeros(T)
    for g in range(nG):
        ax.fill_between(hrs, bottom, bottom + p[g], step="mid", alpha=0.85,
                        color=GEN_COLORS[g % len(GEN_COLORS)], label=f"G{g}")
        bottom += p[g]
    if DATA.rens and "curt" in sol:
        avail = np.vstack([r.avail for r in DATA.rens])
        ren = (avail - np.asarray(sol["curt"], float)).sum(axis=0)
        ax.fill_between(hrs, bottom, bottom + ren, step="mid", alpha=0.45,
                        color=REN_GREEN, label="Renewable")
        bottom += ren
    ax.plot(hrs, bottom, color="black", lw=2.0, label="Total")
    ax.plot(hrs, DATA.demand.sum(axis=0), color="#d62728", ls="--", lw=1.8, label="Demand")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour"); ax.set_ylabel("Power (MW)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.9)


rec = json.load(open(os.path.join(REPO, "pipeline_14box2", "pipeline_14.json")))
g = build_grid("14")
DATA, b0, oracle = g["DATA"], g["b0"], g["oracle"]
b_hat = np.asarray(rec["b_hat"], float)
T = int(DATA.T); hrs = np.arange(T); nL = len(DATA.lines)

print("re-solving factual/plain/foil dispatch ...", flush=True)
_, _, sol0     = oracle.solve_plain(b0)
_, _, sol_pl   = oracle.solve_plain(b_hat)
_, _, sol_foil = oracle.solve_foil(b_hat)

fig = plt.figure(figsize=(16, 10.3))
fig.suptitle(
    rf"IEEE 14-bus transmission-limited emissions counterfactual (DECOMP):  "
    rf"$F^\star$={rec['F_opt']:.3f},  gap {rec['gap_pct']:.1f}%",
    fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.24)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1])

_dispatch_panel(ax1, DATA, sol0, "Factual dispatch")
_dispatch_panel(ax2, DATA, sol_foil, "Counterfactual dispatch (foil active)")

def _curt(sol):
    return (np.asarray(sol["curt"], float).sum(axis=0)
            if sol is not None and DATA.rens and "curt" in sol else np.zeros(T))
cF, cP, cX = _curt(sol0), _curt(sol_pl), _curt(sol_foil)
ax3.bar(hrs - 0.28, cF, 0.27, color="#1f77b4", label=f"Factual ({cF.sum():.1f} MWh)")
ax3.bar(hrs,        cP, 0.27, color="#e6a817", label=rf"Plain @ $\hat b$ ({cP.sum():.1f} MWh)")
ax3.bar(hrs + 0.28, cX, 0.27, color="#1d9e75", label=rf"Foil @ $\hat b$ ({cX.sum():.1f} MWh)")
ax3.set_title("Curtailment per hour", fontsize=13, fontweight="bold")
ax3.set_xlabel("Hour"); ax3.set_ylabel("Curtailment (MWh)")
ax3.grid(True, alpha=0.3, axis="y"); ax3.legend(fontsize=9)

pu = b_hat / np.maximum(np.asarray(b0, float), 1e-12)
changed = np.where(np.abs(pu - 1.0) > 1e-4)[0]
ax4.axhline(1.0, color="black", lw=1.2, ls="--", label="Original (1.0 p.u.)")
ax4.bar(np.arange(nL), pu, color="#cccccc", alpha=0.8, label="Unchanged")
if len(changed):
    ax4.bar(changed, pu[changed], color="#d2691e", label="Changed by CE")
ax4.set_title("Line thermal limits: counterfactual vs. original",
              fontsize=13, fontweight="bold")
ax4.set_xlabel(r"Line index $\ell$")
ax4.set_ylabel(r"$\hat b_\ell / b_{0,\ell}$  (p.u.)")
ax4.grid(True, alpha=0.3, axis="y"); ax4.legend(fontsize=9, loc="lower left")
ax4.text(0.98, 0.97, f"{len(changed)} line(s) changed", transform=ax4.transAxes,
         ha="right", va="top", fontsize=11, fontweight="bold")

out = os.path.join(OUTDIR, "fig_decomp_14_box2.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out}", flush=True)
