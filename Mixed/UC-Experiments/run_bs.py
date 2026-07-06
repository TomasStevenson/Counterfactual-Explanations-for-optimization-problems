"""Run the Branch-&-Sandwich (B&S) phase LIVE and write bs_<grid>_checkpoint.json
— the warm-start CE that build_grid / the DECOMP coordinator consume.

This is the self-contained pre-DECOMP B&S phase: instead of trusting a committed
checkpoint, regenerate it on the cluster so the full B&S -> DECOMP pipeline is
reproducible end to end. Construction mirrors `bs_preprocess_cell` in
build_decomp_notebook.py EXACTLY (eps_b=1.0, eps_obj=1e-3, eps_weak=1e-3,
lagrange_penalty=500.0, the emission+shed violation function, per-grid setup via
build_grid), so the regenerated checkpoint matches the published B&S instance.

B&S checkpoints every node (checkpoint_interval=1), so a Slurm-time-limited job
is RESUMABLE: re-run without --fresh to continue from where it stopped. With
--fresh it starts a new tree (any existing checkpoint is backed up to
bs_<grid>_checkpoint.prelive.json).

Usage:
  python run_bs.py 39 --max-nodes 300            # resume / extend bs_39_checkpoint.json
  python run_bs.py 39 --max-nodes 300 --fresh    # start B&S from scratch (backs up old ckpt)
  python run_bs.py 57 --max-nodes 1 --out /tmp/t.json   # validation, leaves the real ckpt alone
"""
import os, json, argparse, shutil, time, csv, socket, datetime
import numpy as np
import gurobipy as gp

from node_obbt_hpc import build_grid, ALPHA
from uc_branch_sandwich_4b import UCBranchAndSandwichWCE_4b

HERE = os.path.dirname(os.path.abspath(__file__))


def _make_viol_fn(DATA, E_factual):
    """Emission-excess + shed violation (normalised) — identical to the notebook's
    bs_preprocess_cell._make_viol_fn."""
    tgt = (1.0 - ALPHA) * E_factual
    e_g = np.array([float(gen.emission_rate) for gen in DATA.gens])
    nB = DATA.nB; T = int(DATA.T)
    total_demand = float(np.sum(DATA.demand))

    def viol_fn(m, var, idx):
        s = m.addVar(lb=0.0, name="emis_viol")
        total_emis = gp.quicksum(e_g[g_] * var["p"][g_, t]
                                 for g_ in range(len(e_g)) for t in range(T))
        total_shed = gp.quicksum(var["shed"][b, t]
                                 for b in range(nB) for t in range(T))
        m.addConstr(s >= (total_emis - tgt) / max(E_factual, 1.0)
                       + total_shed / max(total_demand, 1.0),
                    name="emis_viol_constr")
        m.update(); return s
    return viol_fn


def main():
    ap = argparse.ArgumentParser(description="Run the live Branch-&-Sandwich phase")
    ap.add_argument("grid", choices=["14", "39", "57"])
    ap.add_argument("--max-nodes", type=int, default=300,
                    help="B&S node budget per invocation (checkpoints every node; resumable)")
    ap.add_argument("--fresh", action="store_true",
                    help="start a NEW B&S tree (back up any existing checkpoint)")
    ap.add_argument("--out", default=None,
                    help="checkpoint path (default bs_<grid>_checkpoint.json beside this script)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="global wall-clock budget (s) for the B&S search; D2 compare runs use 7200 "
                         "(2 h). None = bounded only by --max-nodes / tree exhaustion.")
    ap.add_argument("--mip-lb", action="store_true",
                    help="compute a final MIP lower bound to tighten global_LB when the node/time "
                         "budget is exhausted without certifying (opt-in; adds one Gurobi solve).")
    ap.add_argument("--results", default=None,
                    help="directory to write the D2 record: bs_result_<grid>.json + append bs_results.csv")
    args = ap.parse_args()

    ckpt = args.out or os.path.join(HERE, f"bs_{args.grid}_checkpoint.json")

    # Build the per-grid setup FIRST (build_grid only reads the ckpt for an unused
    # hint here), THEN, if --fresh, move the old ckpt aside so B&S starts a new tree.
    g = build_grid(args.grid)
    if args.fresh and os.path.exists(ckpt):
        bak = ckpt.replace(".json", ".prelive.json")
        shutil.move(ckpt, bak)
        print(f"[run_bs] --fresh: backed up existing checkpoint -> {os.path.basename(bak)}")

    viol_fn = _make_viol_fn(g["DATA"], g["E_factual"])
    bs = UCBranchAndSandwichWCE_4b(
        oracle=g["oracle"], data=g["DATA"], idx=g["idx"], cvec=g["cvec"],
        foil_extra_constr_fn=g["foil_fn"],
        b0=g["b0"], b_bounds=(np.array(g["bL"], float), np.array(g["bU"], float)),
        b_free_idx=g["free_idx"],
        eps_b=1.0, eps_obj=1e-3, eps_weak=1e-3,
        max_nodes=args.max_nodes,
        relax_cost_ub=None, master_time_limit=None,
        output_flag=0, verbose=True, w=g["w"],
        foil_violation_expr_fn=viol_fn,
        lagrange_penalty=500.0,
        checkpoint_path=ckpt,
    )
    rk = g["run_kwargs"]
    t0 = time.perf_counter()
    res = bs.run(window_size=rk["window_size"], per_bus_neutrality=rk["per_bus_neutrality"],
                 u_init=rk["u_init"], p_init=rk["p_init"],
                 on_t=rk["on_time_init"], off_t=rk["off_time_init"],
                 compute_final_mip_lb=args.mip_lb, time_limit=args.time_limit)
    wall = res.get("wall_time_s", time.perf_counter() - t0)

    F = res.get("F_opt"); gLB = res.get("global_LB")
    gap_pct = (100.0 * (F - gLB) / abs(F)) if (res.get("success") and F not in (None, 0)
                                               and gLB is not None and np.isfinite(gLB)) else float("nan")

    # Reconstruct the CE (mutable-line perturbations vs b0) so the pure-B&S CE is
    # directly comparable to the pipeline's ce_changes column.
    b0 = np.asarray(g["b0"], float)
    b_hat = res.get("b_hat")
    ce_list = []
    if b_hat is not None:
        b_hat = np.asarray(b_hat, float)
        for ell in list(g["free_idx"]):
            d = float(b_hat[ell] - b0[ell])
            if abs(d) > 1e-6:
                ce_list.append(f"L{ell}:{d:+.3f}")
    ce_changes = "; ".join(ce_list)

    print(f"[run_bs] IEEE{args.grid}: success={res.get('success')}  F_opt={F}  "
          f"LB={gLB}  gap={gap_pct:.2f}%  nodes={res.get('nodes')}  "
          f"certified={res.get('certified')}  stop={res.get('stop_reason')}  "
          f"wall={wall:.1f}s\n          n_lines={len(ce_list)}  CE=[{ce_changes}]  "
          f"checkpoint -> {ckpt}")
    if not res.get("success"):
        print("[run_bs] *** WARNING: B&S found NO counterfactual — DECOMP would fall back "
              "to bU. Increase --max-nodes / --time-limit or check the instance. ***")

    # ── D2 comparison record ──────────────────────────────────────────────────
    if args.results:
        os.makedirs(args.results, exist_ok=True)
        rec = {
            "grid":            args.grid,
            "carbon":          float(g["DATA"].carbon_price),
            "success":         bool(res.get("success")),
            "F_opt":           F,
            "global_LB":       gLB,
            "gap_pct":         gap_pct,
            "certified":       bool(res.get("certified")),
            "stop_reason":     res.get("stop_reason"),
            "nodes":           res.get("nodes"),
            "max_nodes":       args.max_nodes,
            "time_limit_s":    args.time_limit,
            "wall_time_s":     wall,
            "mip_lb_used":     bool(res.get("mip_lb_used")),
            "n_lines_changed": len(ce_list),
            "ce_changes":      ce_changes,
            "host":            socket.gethostname(),
            "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        }
        jpath = os.path.join(args.results, f"bs_result_{args.grid}.json")
        with open(jpath, "w") as f:
            json.dump(rec, f, indent=2)
        cpath = os.path.join(args.results, "bs_results.csv")
        new_csv = not os.path.exists(cpath)
        with open(cpath, "a", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if new_csv:
                wtr.writeheader()
            wtr.writerow(rec)
        print(f"[run_bs] D2 record -> {jpath}  (+ appended {cpath})")


if __name__ == "__main__":
    main()
