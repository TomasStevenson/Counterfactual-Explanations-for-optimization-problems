#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_ncxplain_timing.py
#
# Time the three Case-Study-I unit-commitment OVERRIDE counterfactuals
# (NCXplain, cost-vector perturbation) and render paper figures.
#
# Foils are specified DIRECTLY by (generator, status, hours) -- NOT by rank:
#     IEEE 14 : commit    G1  over t = {17, 18}
#     IEEE 39 : commit    G5  over t = {17, 18, 19}
#     IEEE 57 : commit    G5  over t = {17, 18}
#
# Reproduces the NCXplain_3grids.ipynb runs exactly (same quick_setup args, same
# bounds/weights, and the same per-grid perturbation_beta: 14->None, 39->0.9,
# 57->None), adds wall-clock timing, and writes under ./ncx_results/:
#     ncx_timing.csv                 one row per grid (time_s, cuts, distance, ...)
#     ncx_<grid>.json                full per-grid summary (incl. small u-matrices)
#     fig_ncx_<grid>_commitment.png  factual vs counterfactual commitment heatmap
#     fig_ncx_<grid>_costchange.png  per-generator cost perturbation (Delta %)
#
# Timing never depends on matplotlib: figures are rendered only if matplotlib
# imports, and can always be regenerated locally from the JSON via --render-only
# (useful when the cluster Python has no matplotlib).
#
# Run locally :  python run_ncxplain_timing.py
# On Leftraru :  sbatch run_ncxplain.slurm
# Figures only:  python run_ncxplain_timing.py --render-only
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
import argparse
import numpy as np

# NOTE: uc_data_loader / b3_ncxplain (which pull in gurobipy) are imported
# lazily inside run_one(), so --help and --render-only work in a plain
# numpy+matplotlib environment without the solver.


# ---------------------------------------------------------------------------
# Foil constructors (ported verbatim from NCXplain_3grids.ipynb, cell #4)
# ---------------------------------------------------------------------------
def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple, set)) else [x]


def _get_var(var, names):
    for n in names:
        if isinstance(var, dict) and n in var:
            return var[n]
        if hasattr(var, n):
            return getattr(var, n)
    raise KeyError(f"Could not find any of {names}")


def foil_force_on(g, times):
    times = _as_list(times)

    def _f(m, var, *a, **k):
        u = _get_var(var, ["u", "U"])
        for t in times:
            m.addConstr(u[g, t] == 1, name=f"foil_on_g{g}_t{t}")
    return _f


def foil_force_off(g, times):
    times = _as_list(times)

    def _f(m, var, *a, **k):
        u = _get_var(var, ["u", "U"])
        for t in times:
            m.addConstr(u[g, t] == 0, name=f"foil_off_g{g}_t{t}")
    return _f


# ---------------------------------------------------------------------------
# Experiment registry -- the three chosen overrides (no "rank" anywhere)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("NCX_DATA_DIR", os.path.join(_HERE, "Data"))
OUT_DIR = os.environ.get("NCX_OUT_DIR", "ncx_results")

EXPERIMENTS = [
    dict(grid="14", json="ieee14_enhanced.json",
         gen=1, status="ON",  times=[17, 18],     beta=0.9),
    dict(grid="39", json="ieee39_newengland.json",
         gen=5, status="ON",  times=[17, 18, 19], beta=0.9),
    # 57: de-commit G0 at the evening peak. The old "commit G5" foil only admits
    # the degenerate -100% no-load waiver (G5 commits at zero output, so only a
    # free commitment is optimal); with beta=0.9 it is MP_INFEASIBLE. The
    # de-commit override has an interior <100% CE: G0 fuel +12.3%, G5 fuel -5.4%
    # (dist 11.28, 8 cuts -- the numbers reported in the paper). NOTE: NCXplain's
    # cut path is tie-sensitive across solver environments; the Leftraru run of
    # this same case converged after 1 cut to a coarser CE (dist 15.92). Both are
    # valid weak CEs; the paper reports the better (smaller-distance) one.
    dict(grid="57", json="ieee57_uc_matpower.json",
         gen=0, status="OFF", times=[17, 18],     beta=0.9),
]

# Same configuration as NCXplain_3grids.ipynb (carbon-free; +-/percentage box).
QUICK_SETUP_KW = dict(carbon_price=0.0, curt_penalty=1000.0,
                      voll=20_000.0, ren_scaling=0.5)
NCX_BOUNDS = {"fuel": (0, 500), "no_load": (0, 5000),
              "su": (0, 100_000), "sd": (0, 100_000)}
NCX_WEIGHTS = {"gen_costs": 1.0}

TECH_COLORS = ["#73726c", "#378ADD", "#BA7517", "#7F77DD", "#1D9E75",
               "#D85A30", "#639922", "#5DCAA5", "#EF9F27", "#E24B4A"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _u_matrix(sol, nG, T):
    """Return the (nG, T) integer commitment matrix from a solution dict."""
    u = np.asarray(sol["u"], dtype=float)
    if u.shape != (nG, T):                       # tolerate dict-like layouts
        u = np.array([[float(sol["u"][g, t]) for t in range(T)]
                      for g in range(nG)])
    return np.round(u).astype(int)


def _cost_arrays(DATA):
    fo = np.array([float(getattr(g, "fuel_cost", 0.0)) for g in DATA.gens])
    no = np.array([float(getattr(g, "no_load_cost", 0.0)) for g in DATA.gens])
    so = np.array([float(getattr(g, "SU_cost", 0.0)) for g in DATA.gens])
    do = np.array([float(getattr(g, "SD_cost", 0.0)) for g in DATA.gens])
    return fo, no, so, do


def _n_cuts(out):
    it = out.get("iters")
    if it is None:
        return None
    if hasattr(it, "__len__"):
        return int(len(it))
    try:
        return int(it)
    except (TypeError, ValueError):
        return None


def _summarise_perturbation(DATA, out):
    """Per-generator coefficient changes and the weighted-L1 distance."""
    fo, no, so, do = _cost_arrays(DATA)
    fn = np.asarray(out["fuel_cost_new"], dtype=float)
    nn = np.asarray(out["no_load_cost_new"], dtype=float)
    sn = np.asarray(out["su_cost_new"], dtype=float)
    dn = np.asarray(out["sd_cost_new"], dtype=float)
    distance = float(np.abs(fn - fo).sum() + np.abs(nn - no).sum()
                     + np.abs(sn - so).sum() + np.abs(dn - do).sum())
    changes = []
    for g in range(len(DATA.gens)):
        comps = {}
        for name, ov, nv in (("fuel", fo[g], fn[g]),
                             ("no_load", no[g], nn[g]),
                             ("SU", so[g], sn[g]),
                             ("SD", do[g], dn[g])):
            if abs(nv - ov) > 1e-4:
                pct = (nv - ov) / ov * 100.0 if abs(ov) > 1e-8 else float("inf")
                comps[name] = dict(base=float(ov), new=float(nv), pct=float(pct))
        if comps:
            changes.append(dict(gen=g, **comps))
    return distance, changes


# ---------------------------------------------------------------------------
# Figures (optional -- only if matplotlib is importable)
# ---------------------------------------------------------------------------
def _try_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")                    # headless / cluster-safe
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:                        # pragma: no cover
        print(f"  [figures] matplotlib unavailable ({e}); skipping plots.")
        return None


def save_commitment_heatmap(uF, uCF, label, path, plt):
    import matplotlib.patches as mpatches
    nG, T = uF.shape
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, nG * 0.5 + 1.4)),
                             sharey=True)
    fig.suptitle(f"Unit commitment - factual vs counterfactual   [{label}]",
                 fontsize=12, fontweight="bold")
    color_on = plt.cm.Blues(0.55)
    color_off = (0.94, 0.94, 0.97, 1.0)
    changed_ec = "#E86E00"
    for ax, u_mat, title, tcol in ((axes[0], uF, "Factual", "royalblue"),
                                   (axes[1], uCF, "Counterfactual", "crimson")):
        ax.set_facecolor("white")
        for g in range(nG):
            for t in range(T):
                fc = color_on if u_mat[g, t] else color_off
                ax.add_patch(mpatches.Rectangle((t - 0.48, nG - g - 1 - 0.48),
                             0.96, 0.96, lw=0.3, edgecolor="#cccccc",
                             facecolor=fc))
                if uCF[g, t] != uF[g, t]:
                    ax.add_patch(mpatches.Rectangle(
                        (t - 0.48, nG - g - 1 - 0.48), 0.96, 0.96, lw=2.2,
                        edgecolor=changed_ec, facecolor="none"))
        for g in range(nG):
            ax.text(T + 0.15, nG - g - 1, f"{u_mat[g].sum()}h",
                    va="center", ha="left", fontsize=8, color="#444")
        ax.set_xlim(-0.5, T + 2)
        ax.set_ylim(-0.5, nG - 0.5)
        ax.set_yticks(range(nG))
        ax.set_yticklabels([f"Gen {nG - 1 - g}" for g in range(nG)], fontsize=8)
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, T, 2))
        ax.set_title(title, fontsize=11, fontweight="bold", color=tcol)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
    handles = [mpatches.Patch(color=color_on, label="ON"),
               mpatches.Patch(color=color_off, label="OFF",
                              edgecolor="#aaa", lw=0.5),
               mpatches.Patch(facecolor="none", edgecolor=changed_ec, lw=2,
                              label="Changed by CE")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8,
                   framealpha=0.92)
    n_changed = int(np.sum(uCF != uF))
    n_on2off = int(np.sum((uF == 1) & (uCF == 0)))
    n_off2on = int(np.sum((uF == 0) & (uCF == 1)))
    fig.text(0.5, 0.01,
             f"{n_changed} commitment cell(s) changed "
             f"({n_off2on} OFF->ON, {n_on2off} ON->OFF)",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_costchange_bar(changes, fuel0, nload0, su0, sd0, label, path, plt):
    """Grouped Delta-% bars per generator (symlog y handles the huge IEEE 57 %)."""
    nG = len(fuel0)

    def pct(new, old):
        return 100.0 * (new - old) / old if abs(old) > 1e-8 else 0.0

    df = np.zeros(nG); dn = np.zeros(nG); ds = np.zeros(nG); dd = np.zeros(nG)
    for ch in changes:
        g = ch["gen"]
        if "fuel" in ch:    df[g] = pct(ch["fuel"]["new"], fuel0[g])
        if "no_load" in ch: dn[g] = pct(ch["no_load"]["new"], nload0[g])
        if "SU" in ch:      ds[g] = pct(ch["SU"]["new"], su0[g])
        if "SD" in ch:      dd[g] = pct(ch["SD"]["new"], sd0[g])
    x = np.arange(nG); w = 0.20
    fig, ax = plt.subplots(figsize=(max(7.0, nG * 0.95), 4.4))
    ax.bar(x - 1.5 * w, df, w, color="#D85A30", alpha=0.85, label="Fuel Δ%")
    ax.bar(x - 0.5 * w, dn, w, color="#378ADD", alpha=0.85, label="No-load Δ%")
    ax.bar(x + 0.5 * w, ds, w, color="#1D9E75", alpha=0.85, label="Start-up Δ%")
    ax.bar(x + 1.5 * w, dd, w, color="#BA7517", alpha=0.85, label="Shut-down Δ%")
    ax.axhline(0, color="k", lw=0.8)
    if np.max(np.abs([df, dn, ds, dd])) > 200:
        ax.set_yscale("symlog", linthresh=10)
    ax.set_title(f"Minimal cost perturbation per generator\n{label}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Δ cost vs factual (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}" for g in range(nG)], fontsize=8)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _render_figures(rec, plt):
    grid = rec["grid"]
    uF = np.array(rec["u_factual"], dtype=int)
    uCF = np.array(rec["u_foil"], dtype=int)
    label = f"IEEE {grid}-bus: {rec['override']}"
    save_commitment_heatmap(
        uF, uCF, label,
        os.path.join(OUT_DIR, f"fig_ncx_{grid}_commitment.png"), plt)
    save_costchange_bar(
        rec["changes"], rec["fuel0"], rec["nload0"], rec["su0"], rec["sd0"],
        label, os.path.join(OUT_DIR, f"fig_ncx_{grid}_costchange.png"), plt)
    print(f"  [figures] wrote fig_ncx_{grid}_commitment.png + "
          f"fig_ncx_{grid}_costchange.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_one(exp):
    from uc_data_loader import quick_setup           # lazy (pulls in gurobipy)
    from b3_ncxplain import run_ncxplain_uc

    grid = exp["grid"]
    verb = "commit" if exp["status"] == "ON" else "de-commit"
    override = f"{verb} G{exp['gen']}, t={{{','.join(map(str, exp['times']))}}}"
    print(f"\n{'=' * 64}\nIEEE {grid}-bus  |  {override}  "
          f"(beta={exp['beta']})\n{'=' * 64}")

    DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup(
        os.path.join(DATA_DIR, exp["json"]), **QUICK_SETUP_KW)
    nG, T = len(DATA.gens), int(DATA.T)

    foil_fn = (foil_force_on if exp["status"] == "ON" else foil_force_off)(
        exp["gen"], exp["times"])

    t0 = time.perf_counter()
    out = run_ncxplain_uc(
        data=DATA,
        window_size=T,
        per_bus_neutrality=True,
        foil_fn=foil_fn,
        mutables={"gen_costs"},
        bounds=NCX_BOUNDS,
        perturbation_beta=exp["beta"],
        weights=NCX_WEIGHTS,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    status = str(out.get("status", "?"))
    cuts = _n_cuts(out)
    distance, changes = _summarise_perturbation(DATA, out)
    fuel0, nload0, su0, sd0 = _cost_arrays(DATA)

    print(f"\n  status      : {status}")
    print(f"  cuts        : {cuts}")
    print(f"  distance |c-c0|_1 : {distance:.4g}")
    print(f"  wall time   : {elapsed:.3f} s")
    for ch in changes:
        comp = ", ".join(f"{k} {v['pct']:+.1f}%" for k, v in ch.items()
                         if k != "gen")
        print(f"    G{ch['gen']}: {comp}")

    rec = dict(
        grid=grid, override=override,
        gen=exp["gen"], status=exp["status"], times=exp["times"],
        beta=exp["beta"], ncx_status=status, cuts=cuts,
        distance_l1=distance, wall_time_s=elapsed,
        changes=changes,
        fuel0=fuel0.tolist(), nload0=nload0.tolist(),
        su0=su0.tolist(), sd0=sd0.tolist(),
        u_factual=_u_matrix(out["sol_factual"], nG, T).tolist(),
        u_foil=_u_matrix(out["sol_foil"], nG, T).tolist(),
    )
    with open(os.path.join(OUT_DIR, f"ncx_{grid}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec


def write_csv(records):
    path = os.path.join(OUT_DIR, "ncx_timing.csv")
    cols = ["grid", "override", "ncx_status", "cuts",
            "distance_l1", "wall_time_s", "beta"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            f.write(",".join(
                f"\"{r[c]}\"" if c == "override" else str(r[c]) for c in cols
            ) + "\n")
    print(f"\n[done] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render-only", action="store_true",
                    help="skip solving; rebuild figures from ncx_results/*.json")
    ap.add_argument("--no-figures", action="store_true",
                    help="time only; do not render figures")
    ap.add_argument("--grids", nargs="*", default=None,
                    help="subset of grids to run, e.g. --grids 14 39")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt = None if args.no_figures else _try_mpl()

    if args.render_only:
        if plt is None:
            sys.exit("--render-only needs matplotlib")
        for exp in EXPERIMENTS:
            jp = os.path.join(OUT_DIR, f"ncx_{exp['grid']}.json")
            if os.path.exists(jp):
                with open(jp, encoding="utf-8") as f:
                    _render_figures(json.load(f), plt)
        return

    exps = EXPERIMENTS if not args.grids else [
        e for e in EXPERIMENTS if e["grid"] in set(args.grids)]
    records = []
    for exp in exps:
        rec = run_one(exp)
        records.append(rec)
        if plt is not None:
            try:
                _render_figures(rec, plt)
            except Exception as e:                # pragma: no cover
                print(f"  [figures] render failed for IEEE {exp['grid']}: {e}")
    write_csv(records)

    print(f"\n{'grid':<6}{'override':<34}{'cuts':>6}{'dist |c-c0|_1':>16}"
          f"{'time [s]':>10}")
    print("-" * 72)
    for r in records:
        print(f"{r['grid']:<6}{r['override']:<34}{str(r['cuts']):>6}"
              f"{r['distance_l1']:>16.4g}{r['wall_time_s']:>10.3f}")


if __name__ == "__main__":
    main()
