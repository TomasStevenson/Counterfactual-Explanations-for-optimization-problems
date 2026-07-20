"""Regenerate the Case Study II per-grid figures from the CARBON-FREE results
(pipeline_<grid>_CE0.json), in the paper's visual style (the 06-29 fig_decomp_57
look: black Total line, red dashed Demand, orange 'Changed by CE' bars).

Run with ce-env python. Writes fig_decomp_<grid>_CE0.png to this scratchpad dir.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from node_obbt_hpc import build_grid  # noqa: E402  (carbon-free _SETUP)

RESDIR = os.path.join(REPO, "pipeline_results")
OUTDIR = os.path.dirname(os.path.abspath(__file__))
GRIDS  = ["57"]

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
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("Hour"); ax.set_ylabel("Power (MW)")
    ax.grid(True, alpha=0.3)
    ymax = max(bottom.max(), DATA.demand.sum(axis=0).max())
    ax.set_ylim(0, 1.42 * ymax)
    ax.legend(fontsize=11, ncol=4, loc="upper left", framealpha=0.9)


def make_figure(grid):
    rec = json.load(open(os.path.join(RESDIR, f"pipeline_{grid}_CE0.json")))
    if rec.get("b_hat") is None:
        print(f"[IEEE {grid}] no b_hat — skipping"); return
    g = build_grid(grid)
    DATA, b0, oracle = g["DATA"], g["b0"], g["oracle"]
    b_hat = np.asarray(rec["b_hat"], float)
    T = int(DATA.T); hrs = np.arange(T); nL = len(DATA.lines)

    print(f"[IEEE {grid}] re-solving factual/plain/foil dispatch ...", flush=True)
    _, _, sol0     = oracle.solve_plain(b0)
    _, _, sol_pl   = oracle.solve_plain(b_hat)
    _, _, sol_foil = oracle.solve_foil(b_hat)

    fig = plt.figure(figsize=(13, 8.4))
    fig.suptitle(
        rf"IEEE {grid}-bus transmission-limited emissions counterfactual (DECOMP):  "
        rf"$F^\star$={rec['F_opt']:.3f},  gap {rec['gap_pct']:.3f}%",
        fontsize=20, fontweight="bold")
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
    ax3.set_title("Curtailment per hour", fontsize=18, fontweight="bold")
    ax3.set_xlabel("Hour"); ax3.set_ylabel("Curtailment (MWh)")
    ax3.grid(True, alpha=0.3, axis="y"); ax3.legend(fontsize=12, loc="upper left")

    pu = b_hat / np.maximum(np.asarray(b0, float), 1e-12)
    changed = np.where(np.abs(pu - 1.0) > 1e-4)[0]
    ax4.axhline(1.0, color="black", lw=1.2, ls="--", label="Original (1.0 p.u.)")
    ax4.bar(np.arange(nL), pu, color="#cccccc", alpha=0.8, label="Unchanged")
    if len(changed):
        ax4.bar(changed, pu[changed], color="#d2691e", label="Changed by CE")
    ax4.set_title("Line thermal limits: counterfactual vs. original",
                  fontsize=18, fontweight="bold")
    ax4.set_xlabel(r"Line index $\ell$")
    ax4.set_ylabel(r"$\hat b_\ell / b_{0,\ell}$  (p.u.)")
    ax4.grid(True, alpha=0.3, axis="y"); ax4.legend(fontsize=12, loc="lower left")
    ax4.text(0.98, 0.97, f"{len(changed)} line(s) changed", transform=ax4.transAxes,
             ha="right", va="top", fontsize=15, fontweight="bold")

    out = os.path.join(OUTDIR, f"fig_decomp_{grid}_CE0_big.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    for grid in GRIDS:
        make_figure(grid)
    print("done.")
