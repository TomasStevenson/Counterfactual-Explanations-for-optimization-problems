"""End-to-end TIMED pipeline for ONE grid, in a SINGLE process:

        Branch-&-Sandwich warm-start cycle   ->   DECOMP (node-OBBT certifier)

and reports the WHOLE-process wall time (B&S + DECOMP) as one number, plus the
per-phase breakdown, the B&S hint value/gap, and the certified CE / gap.

This is the clean experiment the distributed coordinator is NOT: one Slurm job,
one process, one wall clock — no inter-job queueing in the measured time. The
in-process `node` solver IS the full DECOMP (spatial B&B over the b-box), so this
is the same certifier as run_benchmark.py's "node" path, just preceded by a live
B&S phase and timed end to end.

Usage:
  python run_pipeline.py 39 --time-limit 3600 --per-box 600 --bs-nodes 1500
  python run_pipeline.py 57 --time-limit 7200 --per-box 600 --bs-nodes 10
  python run_pipeline.py 14 --time-limit 7200 --per-box 600 --bs-nodes 1200
"""
import os, json, csv, time, argparse, shutil
import numpy as np

from node_obbt_hpc import build_grid, _make_decomp
from uc_branch_sandwich_4b import UCBranchAndSandwichWCE_4b
from run_bs import _make_viol_fn          # the identical emission+shed violation fn

HERE = os.path.dirname(os.path.abspath(__file__))
# Per-grid seed_interp (same as run_benchmark.py): IEEE 39 needs interior seeds;
# 14/57 dedupe so lean is better.
SEED_INTERP = {"14": 0, "39": 3, "57": 0}


def run_bs_phase(grid, max_nodes, fresh=True):
    """Phase 1 — run B&S live, writing bs_<grid>_checkpoint.json (the hint)."""
    g = build_grid(grid)
    ckpt = os.path.join(HERE, f"bs_{grid}_checkpoint.json")
    if fresh and os.path.exists(ckpt):
        shutil.move(ckpt, ckpt.replace(".json", ".prelive.json"))   # new tree
    bs = UCBranchAndSandwichWCE_4b(
        oracle=g["oracle"], data=g["DATA"], idx=g["idx"], cvec=g["cvec"],
        foil_extra_constr_fn=g["foil_fn"],
        b0=g["b0"], b_bounds=(np.array(g["bL"], float), np.array(g["bU"], float)),
        b_free_idx=g["free_idx"],
        eps_b=1.0, eps_obj=1e-3, eps_weak=1e-3, max_nodes=max_nodes,
        output_flag=0, verbose=True, w=g["w"],
        foil_violation_expr_fn=_make_viol_fn(g["DATA"], g["E_factual"]),
        lagrange_penalty=500.0, checkpoint_path=ckpt,
    )
    rk = g["run_kwargs"]
    return bs.run(window_size=rk["window_size"], per_bus_neutrality=rk["per_bus_neutrality"],
                  u_init=rk["u_init"], p_init=rk["p_init"],
                  on_t=rk["on_time_init"], off_t=rk["off_time_init"],
                  compute_final_mip_lb=False)


def run_decomp_phase(grid, time_limit, per_box):
    """Phase 2 — the in-process node-OBBT certifier, warm-started by the fresh
    B&S hint (build_grid re-reads the checkpoint B&S just wrote)."""
    g = build_grid(grid)
    dec = _make_decomp(g, g["bL"], g["bU"], per_box, SEED_INTERP.get(grid, 0),
                       max_iter=1000, node=True, max_nodes=10**9, time_limit=time_limit)
    return g, dec.run(**g["run_kwargs"])


def main():
    ap = argparse.ArgumentParser(description="Timed B&S -> DECOMP pipeline (one process)")
    ap.add_argument("grid", choices=["14", "39", "57"])
    ap.add_argument("--time-limit", type=float, default=3600.0, help="DECOMP global wall budget (s)")
    ap.add_argument("--per-box", type=float, default=600.0, help="per-box MIQCP TimeLimit (s)")
    ap.add_argument("--bs-nodes", type=int, default=300, help="B&S node budget")
    ap.add_argument("--no-fresh-bs", action="store_true",
                    help="resume/extend the existing B&S checkpoint instead of a new tree")
    ap.add_argument("--outdir", default="pipeline_results")
    args = ap.parse_args()

    print(f"{'#'*70}\n# PIPELINE  IEEE{args.grid}   B&S({args.bs_nodes} nodes) -> "
          f"DECOMP({args.time_limit:.0f}s, per-box {args.per_box:.0f}s)\n{'#'*70}", flush=True)

    # ---- Phase 1: B&S warm-start cycle (timed) ----
    t0 = time.time()
    bs = run_bs_phase(args.grid, args.bs_nodes, fresh=not args.no_fresh_bs)
    t_bs = time.time() - t0
    bs_F = bs.get("F_opt"); bs_LB = bs.get("global_LB")
    print(f"[pipeline] B&S done: {t_bs:.1f}s  F={bs_F}  LB={bs_LB}  "
          f"nodes={bs.get('nodes')}  certified={bs.get('certified')}", flush=True)
    if not bs.get("success"):
        print("[pipeline] *** WARNING: B&S found NO CE — DECOMP hint falls back to bU. ***", flush=True)

    # ---- Phase 2: DECOMP certifier (timed) ----
    t1 = time.time()
    g, res = run_decomp_phase(args.grid, args.time_limit, args.per_box)
    t_decomp = time.time() - t1
    t_total = time.time() - t0

    b_hat = res.get("b_hat"); b0 = g["b0"]; free = g["free_idx"]
    ch = [(int(e), float(b_hat[e] - b0[e])) for e in free
          if b_hat is not None and abs(b_hat[e] - b0[e]) > 1e-6]
    ch_s = "; ".join(f"L{e}:{d:+.3f}" for e, d in ch)

    rec = dict(
        grid=args.grid, bs_nodes=args.bs_nodes,
        bs_F=bs_F, bs_LB=bs_LB, bs_certified=bool(bs.get("certified")),
        F_opt=res.get("F_opt"), master_LB=res.get("master_LB"),
        gap=res.get("gap"), gap_pct=res.get("gap_pct"),
        certified=bool(res.get("certified")), hit_time_limit=bool(res.get("hit_time_limit")),
        termination_reason=res.get("termination_reason"),
        bs_time_s=t_bs, decomp_time_s=t_decomp, total_time_s=t_total,
        n_lines_changed=len(ch), ce_changes=ch_s,
        b_hat=(list(map(float, b_hat)) if b_hat is not None else None),
    )
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(rec, open(os.path.join(args.outdir, f"pipeline_{args.grid}.json"), "w"), indent=2)
    cols = ["grid", "bs_nodes", "bs_F", "bs_LB", "bs_certified", "F_opt", "master_LB",
            "gap_pct", "certified", "hit_time_limit", "termination_reason",
            "bs_time_s", "decomp_time_s", "total_time_s", "n_lines_changed", "ce_changes"]
    cpath = os.path.join(args.outdir, "pipeline_results.csv")
    new = not os.path.exists(cpath)
    with open(cpath, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new: w.writeheader()
        w.writerow({k: rec.get(k) for k in cols})

    print(f"{'='*70}")
    print(f"[PIPELINE] IEEE{args.grid}:  B&S {t_bs:.1f}s + DECOMP {t_decomp:.1f}s = "
          f"TOTAL {t_total:.1f}s")
    print(f"   B&S hint F={bs_F}  ->  DECOMP F*={res.get('F_opt')}  LB={res.get('master_LB')}  "
          f"gap={res.get('gap_pct')}%  certified={res.get('certified')}  hit_TL={res.get('hit_time_limit')}")
    print(f"   CE: {ch_s}")
    print(f"   saved -> {args.outdir}/pipeline_{args.grid}.json  (+ pipeline_results.csv)")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
