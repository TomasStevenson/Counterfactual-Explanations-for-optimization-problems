"""Time-limited DECOMP CE benchmark — records CE result, gap, time, and whether
the global time limit was hit, for each (solver, grid).

Two solvers under the SAME global wall-clock budget per instance:
  * "node"        — the in-process node-OBBT spatial-B&B solver (one process,
                    reuses the master; the efficient sequential certifier).
  * "coordinator" — the v3 round-based best-first coordinator (rebuilds per box;
                    the parallel-structured certifier, run sequentially here).

For each run it appends one row to results.csv and one record to results.json with:
    grid, solver, time_limit_s, F_opt (CE cost), master_LB, gap, gap_pct,
    certified, wall_time_s, hit_time_limit, termination_reason,
    n_lines_changed, ce_changes (compact)         [+ full b_hat/b0 in JSON]

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run_benchmark.py \
      --time-limit 3600 --grids 14,39,57 --solvers node,coordinator \
      --per-box-budget 300 --outdir benchmark_results
  # quick smoke (certifies fast):
  python run_benchmark.py --time-limit 120 --grids 39 --solvers node
"""
import os, sys, json, csv, time, argparse, shutil
import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")
from node_obbt_hpc import build_grid, _make_decomp
import node_obbt_coordinator as coord

# Per-grid seed_interp (the documented best: IEEE 39 needs interior seeds to
# close its missing-pattern gap; 14/57 dedupe/bloat, so lean is better).
SEED_INTERP = {"14": 0, "39": 3, "57": 0}
CSV_COLS = ["grid", "solver", "time_limit_s", "F_opt", "master_LB", "gap",
            "gap_pct", "certified", "wall_time_s", "hit_time_limit",
            "termination_reason", "n_lines_changed", "ce_changes"]


def _ce_changes(g, b_hat):
    """Compact human-readable list of the line-limit CHANGES the CE makes."""
    if b_hat is None:
        return [], ""
    b0 = g["b0"]; free = g["free_idx"]
    ch = [(int(e), float(b_hat[e] - b0[e])) for e in free if abs(b_hat[e] - b0[e]) > 1e-6]
    s = "; ".join(f"L{e}:{d:+.3f}" for e, d in ch)
    return ch, s


def _record(outdir, csv_row, json_rec):
    os.makedirs(outdir, exist_ok=True)
    cpath = os.path.join(outdir, "results.csv")
    new = not os.path.exists(cpath)
    with open(cpath, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if new: w.writeheader()
        w.writerow({k: csv_row.get(k) for k in CSV_COLS})
    jpath = os.path.join(outdir, "results.json")
    recs = json.load(open(jpath)) if os.path.exists(jpath) else []
    recs.append(json_rec)
    json.dump(recs, open(jpath, "w"), indent=2)


def _run_node(grid, T, per_box, outdir):
    g = build_grid(grid)
    dec = _make_decomp(g, g["bL"], g["bU"], per_box, SEED_INTERP.get(grid, 0),
                       max_iter=1000, node=True, max_nodes=10**9, time_limit=T)
    res = dec.run(**g["run_kwargs"])
    b_hat = res.get("b_hat")
    b_hat = list(map(float, b_hat)) if b_hat is not None else None
    ch, ch_s = _ce_changes(g, np.array(b_hat) if b_hat is not None else None)
    return g, dict(
        F_opt=res.get("F_opt"), master_LB=res.get("master_LB"),
        gap=res.get("gap"), gap_pct=res.get("gap_pct"),
        certified=bool(res.get("certified")), wall_time_s=res.get("wall_time_s"),
        hit_time_limit=bool(res.get("hit_time_limit")),
        termination_reason=res.get("termination_reason"),
        n_lines_changed=len(ch), ce_changes=ch_s, b_hat=b_hat)


def _run_coordinator(grid, T, per_box, outdir):
    rundir = os.path.join(outdir, f"coord_{grid}")
    if os.path.isdir(rundir): shutil.rmtree(rundir)
    ns = argparse.Namespace
    coord.cmd_init(ns(grid=grid, outdir=rundir, budget=per_box,
                      seed_interp=SEED_INTERP.get(grid, 0), max_iter=1,
                      tol=1e-3, split_k=1, time_limit=T))
    coord.cmd_run_local(ns(outdir=rundir, max_rounds=10**9))
    st = coord._load(rundir); g = build_grid(grid)
    gLB = coord._global_LB(st); gUB = st["global_UB"]
    b_hat = st.get("global_UB_b")
    ch, ch_s = _ce_changes(g, np.array(b_hat) if b_hat is not None else None)
    certified = gUB - gLB <= st["tol"]
    return g, dict(
        F_opt=gUB, master_LB=gLB, gap=gUB - gLB,
        gap_pct=(100 * (gUB - gLB) / abs(gUB) if gUB else float("nan")),
        certified=bool(certified), wall_time_s=st.get("wall_time_s"),
        hit_time_limit=bool(st.get("hit_time_limit")),
        termination_reason=("certified_optimal" if certified else
                            ("global_time_limit" if st.get("hit_time_limit") else "stopped")),
        n_lines_changed=len(ch), ce_changes=ch_s, b_hat=b_hat)


def main():
    ap = argparse.ArgumentParser(description="Time-limited DECOMP CE benchmark")
    ap.add_argument("--time-limit", type=float, default=3600.0, help="global wall-clock budget per (solver,grid), s")
    ap.add_argument("--grids", default="14,39,57")
    ap.add_argument("--solvers", default="node,coordinator")
    ap.add_argument("--per-box-budget", type=float, default=300.0, help="per-box / per-master MIQCP TimeLimit, s")
    ap.add_argument("--outdir", default="benchmark_results")
    args = ap.parse_args()
    grids = [s.strip() for s in args.grids.split(",") if s.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    runner = {"node": _run_node, "coordinator": _run_coordinator}

    summary = []
    for grid in grids:
        for solver in solvers:
            print(f"\n{'#'*70}\n# BENCHMARK  grid=IEEE{grid}  solver={solver}  "
                  f"time_limit={args.time_limit:.0f}s\n{'#'*70}", flush=True)
            t0 = time.time()
            try:
                g, r = runner[solver](grid, args.time_limit, args.per_box_budget, args.outdir)
            except Exception as e:
                import traceback; traceback.print_exc()
                r = dict(F_opt=None, master_LB=None, gap=None, gap_pct=None,
                         certified=False, wall_time_s=time.time() - t0,
                         hit_time_limit=False, termination_reason=f"ERROR:{type(e).__name__}",
                         n_lines_changed=None, ce_changes="", b_hat=None)
            row = dict(grid=grid, solver=solver, time_limit_s=args.time_limit, **r)
            jrec = dict(row); jrec["b0"] = None
            try:
                jrec["b0"] = [float(x) for x in build_grid(grid)["b0"]]
            except Exception:
                pass
            _record(args.outdir, row, jrec)
            summary.append(row)
            print(f"[BENCH] grid=IEEE{grid} solver={solver}: F_opt={r['F_opt']} "
                  f"LB={r['master_LB']} gap%={r['gap_pct']} certified={r['certified']} "
                  f"wall={r['wall_time_s']:.1f}s hit_TL={r['hit_time_limit']} term={r['termination_reason']}",
                  flush=True)

    print(f"\n{'='*70}\nSUMMARY  (full table in {args.outdir}/results.csv)\n{'='*70}")
    print(f"{'grid':>5} {'solver':>12} {'F_opt':>10} {'LB':>10} {'gap%':>7} "
          f"{'cert':>5} {'wall_s':>8} {'hit_TL':>6}")
    for r in summary:
        f = lambda x, p=".3f": (format(x, p) if isinstance(x, (int, float)) else str(x))
        print(f"{r['grid']:>5} {r['solver']:>12} {f(r['F_opt']):>10} {f(r['master_LB']):>10} "
              f"{f(r['gap_pct'],'.2f'):>7} {str(r['certified']):>5} "
              f"{f(r['wall_time_s'],'.0f'):>8} {str(r['hit_time_limit']):>6}")


if __name__ == "__main__":
    main()
