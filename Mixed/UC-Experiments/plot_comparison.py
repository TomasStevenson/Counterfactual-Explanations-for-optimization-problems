"""
plot_comparison.py
──────────────────
Comparison plots: Factual vs Counterfactual (CE) solution.

Usage
-----
from plot_comparison import plot_comparison_dashboard, plot_uc_heatmap

plot_comparison_dashboard(data, out)          # 4-panel comparison
plot_uc_heatmap(out, data)                    # UC commitment heatmaps
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────
_C_FACT  = "#2c7bb6"   # blue  – factual
_C_CF    = "#d7191c"   # red   – counterfactual
_C_FOIL  = "#888888"   # grey  – foil reference (dashed)
_ALPHA   = 0.18        # fill alpha for band between curves


# ═════════════════════════════════════════════════════════════
# Internal helpers  (same logic as your original cell)
# ═════════════════════════════════════════════════════════════

def _T(data):
    return int(getattr(data, "T", np.asarray(data.demand).shape[1]))

def _ren_avail(data):
    if not data.rens:
        return np.zeros((0, _T(data)))
    return np.vstack([np.asarray(r.avail, float).reshape(1, -1) for r in data.rens])

def _ren_used(data, sol):
    avail = _ren_avail(data)
    curt  = np.asarray(sol.get("curt", np.zeros_like(avail)), float)
    return avail - curt

def _shed(sol):
    s = np.asarray(sol.get("shed", 0.0), float)
    return s.sum(axis=0) if s.ndim > 1 else np.atleast_1d(s)

def _curt(sol):
    c = np.asarray(sol.get("curt", 0.0), float)
    return c.sum(axis=0) if c.ndim > 1 else np.atleast_1d(c)

def _demand_total(data):
    return np.asarray(data.demand, float).sum(axis=0)

def _net_load(data, sol):
    d  = np.asarray(data.demand, float)
    sp = np.asarray(sol.get("splus",  np.zeros_like(d)), float)
    sm = np.asarray(sol.get("sminus", np.zeros_like(d)), float)
    return (d + sp - sm).sum(axis=0)

def _served(data, sol):
    return _net_load(data, sol) - _shed(sol)

def _gen_total(sol):
    return np.asarray(sol["p"], float).sum(axis=0)

def _ren_total(data, sol):
    return _ren_used(data, sol).sum(axis=0)

def _emissions(data, sol):
    er = np.array([float(g.emission_rate) for g in data.gens]).reshape(-1, 1)
    return (er * np.asarray(sol["p"], float)).sum(axis=0)

def _voll(data):
    return float(getattr(data, "VOLL", getattr(data, "voll", 20_000.0)))

def _curt_cost_total(data, sol):
    if not data.rens:
        return np.zeros(_T(data))
    cc  = np.vstack([np.asarray(r.curt_cost, float).reshape(1, -1) for r in data.rens])
    cu  = np.asarray(sol["curt"], float)
    return (cc * cu).sum(axis=0)

def _valid(out):
    return out.get("status") in ("OK", "OPTIMAL")


# ═════════════════════════════════════════════════════════════
# Shared axis decorator
# ═════════════════════════════════════════════════════════════

def _style(ax, ylabel, title, T):
    ax.set_xlim(0, T - 1)
    ax.set_xlabel("Hour", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")


# ═════════════════════════════════════════════════════════════
# Main comparison dashboard
# ═════════════════════════════════════════════════════════════

def plot_comparison_dashboard(data, out, figsize=(14, 10)):
    """
    2×2 panel comparison:
      [Generation mix]  [Curtailment]
      [Emissions]       [Reliability]

    Each panel overlays Factual (blue) and Counterfactual/CE (red).
    The foil solution is shown as a grey dashed reference where relevant.
    """
    if not _valid(out):
        print(f"[plot_comparison_dashboard] Cannot plot: status = {out.get('status')}")
        return

    solF   = out["sol_factual"]
    solCF  = out["sol_opt"]          # solution under new (CE) cost coefficients
    solFoil = out.get("sol_foil")    # foil reference (may be None)

    T  = _T(data)
    t  = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax_gen, ax_curt), (ax_em, ax_rel) = axes

    # ── 1. Generation mix ───────────────────────────────────────────────────
    for sol, col, lbl in [
        (solF,  _C_FACT, "Factual"),
        (solCF, _C_CF,   "Counterfactual"),
    ]:
        thermal = _gen_total(sol)
        renew   = _ren_total(data, sol)
        total   = thermal + renew
        ax_gen.plot(t, total,   color=col, lw=2,        label=f"{lbl} — total gen")
        ax_gen.plot(t, renew,   color=col, lw=1.2, ls="--", alpha=0.7,
                    label=f"{lbl} — renewables")

    demand = _demand_total(data)
    ax_gen.plot(t, demand, color="black", lw=1.2, ls=":", alpha=0.6, label="Demand")

    # fill between total generation curves
    ax_gen.fill_between(t,
        _gen_total(solF)  + _ren_total(data, solF),
        _gen_total(solCF) + _ren_total(data, solCF),
        alpha=_ALPHA, color="#888888")

    _style(ax_gen, "MW", "Generation Mix", T)

    # ── 2. Curtailment ──────────────────────────────────────────────────────
    avail = _ren_avail(data).sum(axis=0)
    ax_curt.fill_between(t, 0, avail, alpha=0.08, color="green", label="Available (area)")
    ax_curt.plot(t, avail, color="green", lw=1.2, ls=":", alpha=0.6)

    for sol, col, lbl in [
        (solF,  _C_FACT, "Factual"),
        (solCF, _C_CF,   "Counterfactual"),
    ]:
        cu = _curt(sol)
        ax_curt.plot(t, cu, color=col, lw=2, label=f"{lbl} — curtailment")

    ax_curt.fill_between(t, _curt(solF), _curt(solCF),
                         alpha=_ALPHA, color="#888888")
    _style(ax_curt, "MW", "Renewable Curtailment", T)

    # ── 3. Emissions ────────────────────────────────────────────────────────
    emF  = _emissions(data, solF)
    emCF = _emissions(data, solCF)

    ax_em.plot(t, emF,  color=_C_FACT, lw=2, label="Factual")
    ax_em.plot(t, emCF, color=_C_CF,   lw=2, label="Counterfactual")
    ax_em.fill_between(t, emF, emCF, alpha=_ALPHA, color="#888888")

    # cumulative as secondary axis
    ax_em2 = ax_em.twinx()
    ax_em2.plot(t, np.cumsum(emF),  color=_C_FACT, lw=1, ls="--", alpha=0.5)
    ax_em2.plot(t, np.cumsum(emCF), color=_C_CF,   lw=1, ls="--", alpha=0.5)
    ax_em2.set_ylabel("Cumulative emissions", fontsize=8, color="#666666")
    ax_em2.tick_params(axis="y", labelcolor="#666666", labelsize=8)
    ax_em2.spines[["top"]].set_visible(False)

    _style(ax_em, "tCO₂/h", "Emissions", T)

    # ── 4. Reliability ──────────────────────────────────────────────────────
    shedF  = _shed(solF)
    shedCF = _shed(solCF)
    voll   = _voll(data)

    ax_rel.bar(t - 0.2, shedF,  0.38, color=_C_FACT, alpha=0.75, label="Factual shed (MW)")
    ax_rel.bar(t + 0.2, shedCF, 0.38, color=_C_CF,   alpha=0.75, label="CF shed (MW)")

    # curtailment cost overlay on secondary y
    ax_rel2 = ax_rel.twinx()
    ccF  = _curt_cost_total(data, solF)
    ccCF = _curt_cost_total(data, solCF)
    ax_rel2.plot(t, ccF,  color=_C_FACT, lw=1.5, ls="--", alpha=0.7, label="Factual curt cost")
    ax_rel2.plot(t, ccCF, color=_C_CF,   lw=1.5, ls="--", alpha=0.7, label="CF curt cost")
    ax_rel2.set_ylabel("Curtailment cost ($/h)", fontsize=8, color="#666666")
    ax_rel2.tick_params(axis="y", labelcolor="#666666", labelsize=8)
    ax_rel2.spines[["top"]].set_visible(False)

    _style(ax_rel, "MW shed", "Reliability (Shedding + Curt. Cost)", T)

    # ── Totals annotation ───────────────────────────────────────────────────
    def _fmt(label, vF, vCF, unit):
        delta = vCF - vF
        sign  = "+" if delta >= 0 else ""
        return f"{label}: F={vF:.1f}  CF={vCF:.1f}  (Δ{sign}{delta:.1f} {unit})"

    summary_lines = [
        _fmt("Total gen",  (_gen_total(solF)+_ren_total(data,solF)).sum(),
                            (_gen_total(solCF)+_ren_total(data,solCF)).sum(), "MWh"),
        _fmt("Total curt", _curt(solF).sum(),   _curt(solCF).sum(),   "MWh"),
        _fmt("Emissions",  emF.sum(),            emCF.sum(),           "tCO₂"),
        _fmt("Shedding",   shedF.sum(),          shedCF.sum(),         "MWh"),
    ]
    fig.text(0.01, -0.02, "\n".join(summary_lines),
             fontsize=8.5, color="#333333",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc"))

    fig.suptitle("Factual vs Counterfactual — Operational Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════
# Unit Commitment heatmaps
# ═════════════════════════════════════════════════════════════

def plot_uc_heatmap(out, data, gen_labels=None, figsize=(13, 5)):
    """
    Side-by-side heatmaps of u[g,t] for:
      Left  — Factual solution
      Right — Counterfactual (CE) solution

    Cells that DIFFER between the two solutions are outlined in orange.
    Foil-constrained (g, t) pairs are marked with a dot.
    """
    if not _valid(out):
        print(f"[plot_uc_heatmap] Cannot plot: status = {out.get('status')}")
        return

    solF  = out["sol_factual"]
    solCF = out["sol_opt"]

    uF  = np.asarray(solF["u"],  float)   # (nG, T)
    uCF = np.asarray(solCF["u"], float)

    nG, T = uF.shape

    if gen_labels is None:
        gen_labels = [f"Gen {g}" for g in range(nG)]

    # colour map: 0=off (light), 1=on (dark)
    cmap = ListedColormap(["#f0f4f8", "#2c7bb6"])

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    diff = (np.round(uF) != np.round(uCF))   # cells that changed

    for ax, u, title, base_col in [
        (axes[0], uF,  "Factual",         _C_FACT),
        (axes[1], uCF, "Counterfactual",  _C_CF),
    ]:
        im = ax.imshow(np.round(u), aspect="auto", cmap=cmap,
                       vmin=0, vmax=1, origin="upper",
                       interpolation="nearest")

        # outline cells that differ between factual and CE
        for g in range(nG):
            for t in range(T):
                if diff[g, t]:
                    ax.add_patch(plt.Rectangle(
                        (t - 0.5, g - 0.5), 1, 1,
                        fill=False, edgecolor="#ff7f00", lw=2.0, zorder=3
                    ))

        # grid lines
        ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nG, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

        ax.set_xticks(np.arange(0, T, max(1, T // 12)))
        ax.set_xticklabels(np.arange(0, T, max(1, T // 12)), fontsize=8)
        ax.set_yticks(range(nG))
        ax.set_yticklabels(gen_labels, fontsize=9)
        ax.set_xlabel("Hour", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=base_col, pad=6)

        # on-hours count per generator
        for g in range(nG):
            on_h = int(np.round(u[g]).sum())
            ax.text(T - 0.4, g, f"{on_h}h",
                    va="center", ha="right", fontsize=7.5,
                    color="white" if np.round(u[g, -1]) == 1 else "#333333",
                    fontweight="bold")

    # legend
    from matplotlib.patches import Patch, Rectangle as Rect
    legend_els = [
        Patch(facecolor="#2c7bb6", label="ON"),
        Patch(facecolor="#f0f4f8", edgecolor="#cccccc", label="OFF"),
        Patch(facecolor="none",    edgecolor="#ff7f00", lw=2, label="Changed by CE"),
    ]
    axes[1].legend(handles=legend_els, loc="lower right",
                   fontsize=8, framealpha=0.9)

    # summary: how many (g,t) cells flipped
    n_flips = int(diff.sum())
    fig.text(0.5, -0.02,
             f"{n_flips} commitment cell(s) changed  "
             f"({int(((np.round(uF)==0) & (np.round(uCF)==1)).sum())} OFF→ON, "
             f"{int(((np.round(uF)==1) & (np.round(uCF)==0)).sum())} ON→OFF)",
             ha="center", fontsize=9, color="#444444")

    fig.suptitle("Unit Commitment — Factual vs Counterfactual",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
