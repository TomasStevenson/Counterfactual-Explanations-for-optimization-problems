"""Render result figures for the NLHPC B&S->DECOMP pipeline (IEEE 14/39/57).

Two products, written to pipeline_results/:
  * fig_decomp_<grid>.png  — the 4-panel per-grid figure, ported verbatim from
    decomp_3grids.ipynb's `plot_decomp_results` (factual dispatch, counterfactual
    dispatch, curtailment, line-limit changes). Fed from pipeline_results/pipeline_<grid>.json.
  * fig_pipeline_summary.png — cross-grid summary (optimality gap, runtime, lines changed).

The per-grid setup (data, oracle, foil) is rebuilt with node_obbt_hpc.build_grid — the
SAME code path that produced each b_hat on the cluster — so the re-solved dispatch is
consistent with the certified/incumbent CE. Run with the ce-env python:

    C:/Users/tomas/miniconda3/envs/ce-env/python.exe make_pipeline_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from node_obbt_hpc import build_grid, ALPHA

HERE     = os.path.dirname(os.path.abspath(__file__))
OUTDIR   = os.path.join(HERE, "pipeline_results")
GRIDS    = ["14", "39", "57"]

# Color palette for dispatch stacked-area plots (matches the notebooks).
TECH_COLORS = ["#73726c", "#378ADD", "#BA7517", "#7F77DD", "#1D9E75",
               "#D85A30", "#639922", "#5DCAA5", "#EF9F27", "#E24B4A"]


# --------------------------------------------------------------------------
# 4-panel per-grid figure — ported from decomp_3grids.ipynb plot_decomp_results,
# saving to PNG instead of plt.show(), and reading metrics from the pipeline json.
# --------------------------------------------------------------------------
def plot_decomp_results(DATA, b0, b_hat, sol0, sol_plain_hat, sol_foil_hat,
                        rec, savepath, label="", foil_label="emissions -0%", method="DECOMP"):
    nG = len(DATA.gens); nR = len(DATA.rens); T = int(DATA.T); nL = len(DATA.lines)
    hrs = np.arange(T)
    fig = plt.figure(figsize=(18, 11))
    cert = "CERTIFIED" if rec.get("certified") else "uncertified"
    fig.suptitle(f"{method} - {label}   [foil: {foil_label}]   "
                 f"F*={rec.get('F_opt'):.4f}  LB={rec.get('master_LB'):.4f}  "
                 f"gap={rec.get('gap_pct'):.3f}%  ({cert})",
                 fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.28)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                          fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))

    def _dispatch_panel(ax, sol, title):
        if sol is None:
            ax.set_title(f"{title} (N/A)"); return
        p_raw = sol["p"]
        if isinstance(p_raw, dict):
            p_mat = np.array([[float(p_raw[g, t]) for t in range(T)] for g in range(nG)])
        else:
            p_mat = np.asarray(p_raw, dtype=float)
            if p_mat.shape == (T, nG): p_mat = p_mat.T
        sol = dict(sol, p=p_mat)
        bottom = np.zeros(T)
        for g in range(nG):
            col = TECH_COLORS[g % len(TECH_COLORS)]
            ax.fill_between(hrs, bottom, bottom + sol["p"][g], color=col, alpha=0.75, label=f"G{g}", step="mid")
            bottom += sol["p"][g]
        if nR > 0 and "curt" in sol:
            avail = np.vstack([r.avail for r in DATA.rens])
            ren_inj = (avail - sol["curt"]).sum(axis=0)
            ax.fill_between(hrs, bottom, bottom + ren_inj, color="#1D9E75", alpha=0.5, label="Renewable", step="mid")
            bottom += ren_inj
        ax.plot(hrs, bottom, color="orange", lw=1.8, label="Total production", zorder=5)
        ax.plot(hrs, DATA.demand.sum(axis=0), "k--", lw=1.5, label="Demand")
        ax.set_title(title, fontsize=10); ax.set_xlabel("Hour"); ax.set_ylabel("MW")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=4, loc="upper left")

    _dispatch_panel(ax1, sol0,         "Factual dispatch  (b = b0)")
    _dispatch_panel(ax2, sol_foil_hat, "Counterfactual dispatch  (b = b_hat, foil active)")

    curtF  = sol0["curt"].sum(axis=0) if nR > 0 and "curt" in sol0 else np.zeros(T)
    curtCF = sol_foil_hat["curt"].sum(axis=0) if sol_foil_hat is not None and nR > 0 and "curt" in sol_foil_hat else np.zeros(T)
    curtP  = sol_plain_hat["curt"].sum(axis=0) if sol_plain_hat is not None and nR > 0 and "curt" in sol_plain_hat else np.zeros(T)
    ax3.bar(hrs - 0.27, curtF,  0.26, color="#D85A30", alpha=0.85, label=f"Factual  ({curtF.sum():.1f} MWh)")
    ax3.bar(hrs,        curtP,  0.26, color="#BA7517", alpha=0.85, label=f"Plain@b_hat  ({curtP.sum():.1f} MWh)")
    ax3.bar(hrs + 0.27, curtCF, 0.26, color="#378ADD", alpha=0.85, label=f"Foil@b_hat   ({curtCF.sum():.1f} MWh)")
    ax3.set_title("Curtailment per hour (MWh)", fontsize=10)
    ax3.set_xlabel("Hour"); ax3.set_ylabel("MWh"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

    b_hat_pu = b_hat / np.maximum(b0, 1e-12)
    changed  = np.where(np.abs(b_hat_pu - 1.0) > 1e-4)[0]
    x_all    = np.arange(nL)
    ax4.bar(x_all, b_hat_pu, color="#cccccc", alpha=0.6, label="Unchanged")
    if len(changed):
        ax4.bar(changed, b_hat_pu[changed], color="#378ADD", alpha=0.9, label="Changed")
    ax4.axhline(1.0, color="k", lw=1.0, ls="--", label="Original (1.0 p.u.)")
    ax4.set_title("Line limits - counterfactual vs original  (p.u.)", fontsize=10)
    ax4.set_xlabel("Line index l"); ax4.set_ylabel("b_hat / b0  (p.u.)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="y")
    ax4.text(0.98, 0.97, f"{len(changed)} line(s) changed", transform=ax4.transAxes,
             ha="right", va="top", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {os.path.relpath(savepath, HERE)}")


def make_grid_figure(grid):
    rec = json.load(open(os.path.join(OUTDIR, f"pipeline_{grid}.json")))
    if rec.get("b_hat") is None:
        print(f"[IEEE {grid}] no b_hat in pipeline json — skipping."); return rec
    g = build_grid(grid)
    DATA, b0, oracle = g["DATA"], g["b0"], g["oracle"]
    b_hat = np.array(rec["b_hat"], dtype=float)
    print(f"[IEEE {grid}] buses={DATA.nB} lines={len(DATA.lines)} gens={len(DATA.gens)} T={int(DATA.T)}  "
          f"re-solving factual + counterfactual dispatch ...")
    _, _, sol0       = oracle.solve_plain(b0)      # factual dispatch (no foil) at b0
    _, _, sol_plain  = oracle.solve_plain(b_hat)   # dispatch at b_hat, no foil
    _, _, sol_foil   = oracle.solve_foil(b_hat)    # counterfactual: foil active at b_hat
    plot_decomp_results(
        DATA, b0, b_hat, sol0, sol_plain, sol_foil, rec,
        savepath=os.path.join(OUTDIR, f"fig_decomp_{grid}.png"),
        label=f"IEEE {grid}-bus", foil_label=f"emissions -{ALPHA:.0%}", method="DECOMP",
    )
    return rec


# --------------------------------------------------------------------------
# Cross-grid summary figure
# --------------------------------------------------------------------------
def make_summary_figure(recs):
    grids   = [f"IEEE {r['grid']}" for r in recs]
    gaps    = [float(r["gap_pct"]) for r in recs]
    cert    = [bool(r["certified"]) for r in recs]
    bs_min  = [float(r["bs_time_s"]) / 60.0 for r in recs]
    dec_min = [float(r["decomp_time_s"]) / 60.0 for r in recs]
    nlines  = [int(r["n_lines_changed"]) for r in recs]
    changes = [r.get("ce_changes", "") for r in recs]
    x = np.arange(len(recs))
    GREEN, ORANGE = "#1D9E75", "#D85A30"
    bar_c = [GREEN if c else ORANGE for c in cert]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("NLHPC B&S -> DECOMP pipeline — IEEE 14 / 39 / 57 results",
                 fontsize=14, fontweight="bold")

    # (a) optimality gap, log scale (spans 0.006% .. 24%)
    a.bar(x, gaps, color=bar_c, alpha=0.9)
    a.set_yscale("log")
    a.set_xticks(x); a.set_xticklabels(grids)
    a.set_ylabel("optimality gap  (%, log scale)")
    a.set_title("Certified gap = (F* - LB) / F*")
    a.grid(True, axis="y", alpha=0.3, which="both")
    for xi, gp_, ce in zip(x, gaps, cert):
        a.text(xi, gp_ * 1.15, f"{gp_:.3g}%\n{'CERTIFIED' if ce else 'uncertified'}",
               ha="center", va="bottom", fontsize=9,
               fontweight="bold", color=(GREEN if ce else ORANGE))
    a.set_ylim(min(gaps) * 0.3, max(gaps) * 6)

    # (b) wall-clock: B&S + DECOMP stacked (minutes)
    b.bar(x, bs_min,  0.55, color="#7F77DD", alpha=0.9, label="B&S phase")
    b.bar(x, dec_min, 0.55, bottom=bs_min, color="#378ADD", alpha=0.9, label="DECOMP phase")
    b.set_xticks(x); b.set_xticklabels(grids)
    b.set_ylabel("wall-clock time  (min)")
    b.set_title("Single-job runtime (8 cores)")
    b.grid(True, axis="y", alpha=0.3); b.legend(fontsize=9)
    for xi, t0, t1 in zip(x, bs_min, dec_min):
        b.text(xi, t0 + t1 + max(dec_min) * 0.02, f"{t0 + t1:.0f} min",
               ha="center", va="bottom", fontsize=9, fontweight="bold")

    # (c) lines changed + the actual CE edits
    c.bar(x, nlines, 0.55, color="#BA7517", alpha=0.9)
    c.set_xticks(x); c.set_xticklabels(grids)
    c.set_ylabel("# transmission lines changed")
    c.set_title("Counterfactual sparsity")
    c.grid(True, axis="y", alpha=0.3)
    c.set_ylim(0, max(nlines) + 1.4)
    for xi, n, ce in zip(x, nlines, changes):
        c.text(xi, n + 0.08, f"{n}\n{ce}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUTDIR, "fig_pipeline_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    recs = []
    for grid in GRIDS:
        recs.append(make_grid_figure(grid))
    make_summary_figure(recs)
    print("\nAll figures written to pipeline_results/.")
