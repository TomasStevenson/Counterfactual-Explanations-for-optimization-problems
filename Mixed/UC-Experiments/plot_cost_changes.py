import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_cost_changes(out, data, gen_labels=None, figsize=(13, 9)):
    """
    Visualize which generator cost parameters changed and by how much
    after running run_ncxplain_uc with mutables={"gen_costs"}.

    Parameters
    ----------
    out        : dict returned by run_ncxplain_uc
    data       : NetworkUCData used in the experiment
    gen_labels : list of str, optional. If None, uses "Gen 0", "Gen 1", ...
    figsize    : figure size
    """
    if out.get("status") not in ("OK", "OPTIMAL"):
        print(f"[plot_cost_changes] Cannot plot: status = {out.get('status')}")
        return

    nG = len(data.gens)
    if gen_labels is None:
        gen_labels = [f"Gen {g}" for g in range(nG)]

    # ── Extract base and new costs ──────────────────────────────────────────
    fuel_base    = np.array([float(g.fuel_cost)    for g in data.gens])
    nl_base      = np.array([float(g.no_load_cost) for g in data.gens])
    su_base      = np.array([float(g.SU_cost)      for g in data.gens])
    sd_base      = np.array([float(g.SD_cost)      for g in data.gens])

    fuel_new     = np.asarray(out["fuel_cost_new"],    dtype=float)
    nl_new       = np.asarray(out["no_load_cost_new"], dtype=float)
    su_new       = np.asarray(out["su_cost_new"],      dtype=float)
    sd_new       = np.asarray(out["sd_cost_new"],      dtype=float)

    # ── Deltas ──────────────────────────────────────────────────────────────
    d_fuel = fuel_new - fuel_base
    d_nl   = nl_new   - nl_base
    d_su   = su_new   - su_base
    d_sd   = sd_new   - sd_base

    cost_names  = ["Fuel cost",   "No-load cost", "Start-up cost", "Shut-down cost"]
    deltas      = [d_fuel,        d_nl,           d_su,            d_sd]
    base_vals   = [fuel_base,     nl_base,        su_base,         sd_base]
    new_vals    = [fuel_new,      nl_new,         su_new,          sd_new]
    colors_pos  = ["#e84b4b", "#f0904a", "#4a90d9", "#5cb85c"]
    colors_neg  = ["#a00000", "#b05000", "#1a5fa0", "#2d7a2d"]

    x = np.arange(nG)
    bar_w = 0.35
    eps = 1e-6   # threshold below which a change is considered zero

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    any_change_global = False

    for ax, name, delta, base, new, cp, cn in zip(
            axes, cost_names, deltas, base_vals, new_vals, colors_pos, colors_neg):

        changed = np.abs(delta) > eps
        any_change_global |= bool(changed.any())

        # ── bars: base (grey) and new (colour) side by side ─────────────────
        bar_base = ax.bar(x - bar_w/2, base, bar_w, color="#cccccc",
                          edgecolor="white", linewidth=0.5, label="Base", zorder=2)
        bar_new  = ax.bar(x + bar_w/2, new,  bar_w,
                          color=[cp if d >= 0 else cn for d in delta],
                          edgecolor="white", linewidth=0.5, label="CE (new)", zorder=2)

        # ── delta annotation above/below each changed bar ───────────────────
        for g in range(nG):
            if changed[g]:
                sign  = "+" if delta[g] > 0 else ""
                ypos  = max(base[g], new[g]) + 0.01 * max(new.max(), base.max(), 1)
                ax.annotate(
                    f"{sign}{delta[g]:.1f}",
                    xy=(x[g] + bar_w/2, new[g]),
                    xytext=(x[g] + bar_w/2, ypos),
                    ha="center", va="bottom", fontsize=8,
                    color=cp if delta[g] >= 0 else cn,
                    fontweight="bold",
                )

        # ── highlight changed generators ────────────────────────────────────
        for g in range(nG):
            if changed[g]:
                ax.axvspan(g - 0.5, g + 0.5, alpha=0.08,
                           color=cp if delta[g] >= 0 else cn, zorder=0)

        ax.set_title(name, fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(gen_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("$/MWh" if name == "Fuel cost" else "$", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        # ── zero-change note ─────────────────────────────────────────────────
        if not changed.any():
            ax.text(0.5, 0.55, "No change", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13, color="#888888",
                    fontstyle="italic")

        ax.legend(fontsize=8, loc="upper right")

    if not any_change_global:
        fig.text(0.5, 0.5, "No cost parameters changed in this CE",
                 ha="center", va="center", fontsize=16, color="#555555",
                 fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#aaaaaa"))

    fig.suptitle("Counterfactual Explanation — Cost Changes per Generator",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ─── Summary bar chart: total absolute change per generator ─────────────────

def plot_cost_change_summary(out, data, gen_labels=None, figsize=(9, 4)):
    """
    Single horizontal bar chart showing total |Δcost| per generator,
    broken down by cost type (stacked). Good for a quick overview of
    which generators were most affected.
    """
    if out.get("status") not in ("OK", "OPTIMAL"):
        print(f"[plot_cost_change_summary] Cannot plot: status = {out.get('status')}")
        return

    nG = len(data.gens)
    if gen_labels is None:
        gen_labels = [f"Gen {g}" for g in range(nG)]

    fuel_base = np.array([float(g.fuel_cost)    for g in data.gens])
    nl_base   = np.array([float(g.no_load_cost) for g in data.gens])
    su_base   = np.array([float(g.SU_cost)      for g in data.gens])
    sd_base   = np.array([float(g.SD_cost)      for g in data.gens])

    d_fuel = np.abs(np.asarray(out["fuel_cost_new"],    dtype=float) - fuel_base)
    d_nl   = np.abs(np.asarray(out["no_load_cost_new"], dtype=float) - nl_base)
    d_su   = np.abs(np.asarray(out["su_cost_new"],      dtype=float) - su_base)
    d_sd   = np.abs(np.asarray(out["sd_cost_new"],      dtype=float) - sd_base)

    stacks  = [d_fuel, d_nl, d_su, d_sd]
    labels  = ["Fuel",  "No-load", "Start-up", "Shut-down"]
    palette = ["#e84b4b", "#f0904a", "#4a90d9", "#5cb85c"]

    y = np.arange(nG)
    fig, ax = plt.subplots(figsize=figsize)

    left = np.zeros(nG)
    for vals, lbl, col in zip(stacks, labels, palette):
        ax.barh(y, vals, left=left, color=col, label=lbl,
                edgecolor="white", linewidth=0.5, height=0.55)
        left += vals

    # total label at end of each bar
    total = left
    for g in range(nG):
        if total[g] > 1e-6:
            ax.text(total[g] + 0.005 * total.max(), g,
                    f"{total[g]:.1f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(gen_labels, fontsize=10)
    ax.set_xlabel("Total |Δcost| (summed over changed parameters)", fontsize=10)
    ax.set_title("CE Cost Change Magnitude — by Generator",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


# ─── Heatmap: relative % change per generator × cost type ───────────────────

def plot_cost_change_heatmap(out, data, gen_labels=None, figsize=(7, 4)):
    """
    Heatmap of relative cost change (%) per generator × cost type.
    White = no change. Red = increase. Blue = decrease.
    """
    if out.get("status") not in ("OK", "OPTIMAL"):
        print(f"[plot_cost_change_heatmap] Cannot plot: status = {out.get('status')}")
        return

    nG = len(data.gens)
    if gen_labels is None:
        gen_labels = [f"Gen {g}" for g in range(nG)]

    fuel_base = np.array([float(g.fuel_cost)    for g in data.gens])
    nl_base   = np.array([float(g.no_load_cost) for g in data.gens])
    su_base   = np.array([float(g.SU_cost)      for g in data.gens])
    sd_base   = np.array([float(g.SD_cost)      for g in data.gens])

    d_fuel = np.asarray(out["fuel_cost_new"],    dtype=float) - fuel_base
    d_nl   = np.asarray(out["no_load_cost_new"], dtype=float) - nl_base
    d_su   = np.asarray(out["su_cost_new"],      dtype=float) - su_base
    d_sd   = np.asarray(out["sd_cost_new"],      dtype=float) - sd_base

    # relative change: delta / base  (NaN if base == 0)
    def _rel(delta, base):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(np.abs(base) > 1e-9, delta / base * 100, 0.0)
        return r

    mat = np.column_stack([
        _rel(d_fuel, fuel_base),
        _rel(d_nl,   nl_base),
        _rel(d_su,   su_base),
        _rel(d_sd,   sd_base),
    ])  # shape (nG, 4)

    col_labels = ["Fuel", "No-load", "Start-up", "Shut-down"]
    vmax = max(np.abs(mat).max(), 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    # annotations
    for g in range(nG):
        for k in range(4):
            v = mat[g, k]
            txt = f"{v:+.1f}%" if abs(v) > 0.05 else "—"
            ax.text(k, g, txt, ha="center", va="center",
                    fontsize=8.5,
                    color="white" if abs(v) > vmax * 0.55 else "black")

    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(nG))
    ax.set_yticklabels(gen_labels, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relative change (%)", fontsize=9)
    ax.set_title("CE Cost Changes — Relative (%) per Generator × Parameter",
                 fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.show()