"""Parallel node-OBBT for NLHPC — spatial branch-and-bound over the b-box as
independent jobs.

The node-OBBT solver (`UCDecomp4b._solve_master_spatial_obbt`) certifies by
partitioning the shared b-box and taking `global_LB = min over boxes of each
box's Gurobi ObjBound`. That reduction is over INDEPENDENT subproblems, so it
maps directly onto a Slurm job array: each box is one job; an aggregator takes
the min. See `DECOMP_formulation_and_optimality.md` §7.5.

VALIDITY (unchanged from the in-process driver):
  * The boxes are an EXHAUSTIVE partition of the root b-box ⇒
        global_LB = min_box ObjBound_box ≤ min_box (opt in box) = F*   (valid LB).
  * Each job's incumbent best_F is an ORACLE-VERIFIED counterfactual ⇒ ≥ F*
    (valid UB). global_UB = min_box best_F.
  * obbt_iter=1 (the safe default) keeps OBBT valid per box even when the box
    excludes the hint CE (the guard goes inactive — see _obbt_root).

Three subcommands:
    emit      <grid> <n_boxes> <outdir> [--dims D] [--budget S] [--seed-interp K] [--max-iter N]
    solve     <outdir> <box_id>
    aggregate <outdir>

Local end-to-end (no cluster) is just: emit, then a loop of `solve`, then aggregate.
On NLHPC: emit once, submit `node_obbt.slurm` as an array of size n_boxes, then aggregate.

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python node_obbt_hpc.py emit 14 8 runs/ieee14 --budget 300
  ... python node_obbt_hpc.py solve runs/ieee14 0
  ... python node_obbt_hpc.py aggregate runs/ieee14
"""
import os, sys, json, glob, argparse
import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")

from uc_data_loader import quick_setup
from uc_pipeline import solve_uc_with_cost_4b, make_emissions_foil_4b
from uc_decomp_4b import UCDecomp4b
from _decomp_repro_helpers import (
    get_congested_free_lines, build_b_bounds, make_line_weights, UCWeakWCEOracle,
)

ALPHA = 0.10
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
_FNAME = {"14": "ieee14_enhanced.json", "39": "ieee39_newengland.json",
          "57": "ieee57_uc_matpower.json"}
# Per-grid quick_setup — MUST MATCH build_decomp_notebook.py / _smoke_strongdual.py.
_SETUP = {
    "14": dict(carbon_price=None, voll=20_000.0, slack_bus=None),
    "39": dict(carbon_price=None, voll=20_000.0, slack_bus=None),
    "57": dict(carbon_price=50.0,  voll=500.0,   slack_bus=0),
}


def build_grid(grid):
    """Replicate the standard per-grid setup; return everything needed to build
    a UCDecomp4b plus the b-box and the run() kwargs."""
    fn = os.path.join(DATA_DIR, _FNAME[grid])
    DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup(fn, **_SETUP[grid])
    if grid == "57":
        nG = len(DATA.gens)
        u_init = [0] * nG; p_init = [0.0] * nG; on_t = [0] * nG; off_t = [0] * nG
        u_init[0] = 1; p_init[0] = DATA.gens[0].Pmax; on_t[0] = DATA.gens[0].UT
    T = int(DATA.T)
    _, solF, _ = solve_uc_with_cost_4b(
        data=DATA, idx=idx, cvec=cvec, window_size=T, per_bus_neutrality=True,
        u_init=u_init, p_init=p_init, on_time_init=on_t, off_time_init=off_t,
        output_flag=0,
    )
    e = np.array([float(g.emission_rate) for g in DATA.gens])
    E_fac = float(np.sum(e[:, None] * solF["p"]))
    foil_fn = make_emissions_foil_4b(DATA, alpha=ALPHA, E_factual=E_fac)
    free_idx, util = get_congested_free_lines(solF, b0, thr=0.75)
    bL, bU = build_b_bounds(b0, free_idx)
    w = make_line_weights(DATA, b0, util=util)
    oracle = UCWeakWCEOracle(
        data=DATA, cvec=cvec, idx=idx, window_size=T, per_bus_neutrality=True,
        u_init=u_init, p_init=p_init, on_t=on_t, off_t=off_t,
        foil_extra_constr_fn=foil_fn, output_flag=0,
    )
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"bs_{grid}_checkpoint.json")
    if os.path.exists(ckpt):
        b_hint = np.array(json.load(open(ckpt))["best_b"], float)
    else:
        b_hint = b0.copy(); b_hint[free_idx] = bU[free_idx]
    run_kwargs = dict(window_size=T, per_bus_neutrality=True, u_init=u_init,
                      p_init=p_init, on_time_init=on_t, off_time_init=off_t)
    return dict(DATA=DATA, idx=idx, cvec=cvec, b0=b0, foil_fn=foil_fn, w=w,
                oracle=oracle, free_idx=list(free_idx), bL=bL, bU=bU, b_hint=b_hint,
                run_kwargs=run_kwargs)


def _make_decomp(g, bL_box, bU_box, budget, seed_interp, max_iter,
                 node=False, max_nodes=1, b_hint_override=None):
    """Construct a UCDecomp4b on [bL_box, bU_box] with the STANDARDISED config.

    node=False ⇒ one exact MIQCP solve per box (the `solve` path).
    node=True  ⇒ the in-process spatial driver carves a frontier (the `emit
                 --adaptive` path); `budget` is then the cheap per-box carve TL.
    b_hint_override ⇒ a per-box warm CE (must be a genuine CE so run()'s
                 incumbent registration stays valid); the coordinator passes the
                 box's in-box CE here, else the global hint."""
    _threads = os.environ.get("GRB_THREADS")           # set by node_obbt.slurm
    b_hint = g["b_hint"] if b_hint_override is None else np.asarray(b_hint_override, float)
    return UCDecomp4b(
        oracle=g["oracle"], data=g["DATA"], idx=g["idx"], cvec=g["cvec"],
        foil_extra_constr_fn=g["foil_fn"],
        b0=g["b0"], b_bounds=(bL_box, bU_box), b_free_idx=g["free_idx"],
        big_M_mu=1e4, eps_obj=1e-3, max_iter=max_iter,
        verbose=True, w=g["w"], comp_mode="strongdual",
        master_time_limit=budget, master_output_flag=0, master_mip_gap=1e-4,
        b_hat_hint=b_hint, seed_patterns=True, seed_interp=seed_interp,
        bilinear_exact=True, obbt=True, obbt_iter=1,   # iter=1: valid without the guard
        master_mip_focus=3, master_multistart=1, master_seed=0,
        master_threads=(int(_threads) if _threads else None),
        node_obbt=node, node_obbt_max_nodes=max_nodes, node_obbt_budget=budget,
    )


# --------------------------------------------------------------------------- #
# emit — partition the b-box into n_boxes independent sub-boxes
# --------------------------------------------------------------------------- #
def cmd_emit(args):
    g = build_grid(args.grid)
    free = g["free_idx"]; bL = g["bL"]; bU = g["bU"]; bh = g["b_hint"]; b0 = g["b0"]
    mode = "adaptive" if args.adaptive else "grid"
    split_lines = []
    if args.adaptive:
        # v2: run the in-process best-first driver CHEAPLY (small per-box carve
        # budget) to carve a CE-CONCENTRATED frontier, then distribute THOSE
        # leaves.  Each leaf is re-solved at full budget by the Slurm array.  The
        # adaptive partition splits exactly the box pinning global_LB, unlike a
        # uniform grid (where the F-cap leaves most cells infeasible).
        dec = _make_decomp(g, bL, bU, args.emit_budget, args.seed_interp,
                           max_iter=1, node=True, max_nodes=args.n_boxes)
        res = dec.run(**g["run_kwargs"])
        leaves = res.get("node_boxes") or []
        boxes = [{"lo": {int(k): float(v) for k, v in lf["lo"].items()},
                  "hi": {int(k): float(v) for k, v in lf["hi"].items()}}
                 for lf in leaves]
        if not boxes:                       # driver certified at box 1 → 1 box
            boxes = [{"lo": {int(e): float(bL[e]) for e in free},
                      "hi": {int(e): float(bU[e]) for e in free}}]
    else:
        # Static grid: split the `dims` lines the hint CE EXPANDS MOST
        # (largest |b_hat − b0|) into segs**dims sub-boxes — that is where the CE
        # region has extent (uniformly splitting the widest *bound* range tends
        # to put every CE in one segment and leave the rest infeasible).
        ranked = sorted(free, key=lambda e: abs(bh[e] - b0[e]), reverse=True)
        if abs(bh[ranked[0]] - b0[ranked[0]]) <= 1e-9:    # degenerate hint → fall back
            ranked = sorted(free, key=lambda e: bU[e] - bL[e], reverse=True)
        dims = max(1, min(args.dims, len(free)))
        split_lines = ranked[:dims]
        segs = max(2, int(np.ceil(args.n_boxes ** (1.0 / dims))))
        edges = {ell: np.linspace(bL[ell], bU[ell], segs + 1) for ell in split_lines}
        import itertools
        boxes = []
        for combo in itertools.product(range(segs), repeat=dims):
            lo = {int(ell): float(bL[ell]) for ell in free}
            hi = {int(ell): float(bU[ell]) for ell in free}
            for d, ell in enumerate(split_lines):
                lo[int(ell)] = float(edges[ell][combo[d]])
                hi[int(ell)] = float(edges[ell][combo[d] + 1])
            boxes.append({"lo": lo, "hi": hi})
    os.makedirs(args.outdir, exist_ok=True)
    spec = dict(grid=args.grid, n_boxes=len(boxes), mode=mode,
                split_lines=[int(e) for e in split_lines],
                budget=args.budget, seed_interp=args.seed_interp,
                max_iter=args.max_iter, free_idx=[int(e) for e in free],
                bL=[float(x) for x in bL], bU=[float(x) for x in bU])
    json.dump(spec, open(os.path.join(args.outdir, "spec.json"), "w"), indent=2)
    for i, bx in enumerate(boxes):
        json.dump(bx, open(os.path.join(args.outdir, f"box_{i:04d}.json"), "w"))
    print(f"[emit] grid=IEEE{args.grid}  mode={mode}  free={len(free)}  "
          f"split_lines={spec['split_lines']}  -> {len(boxes)} boxes in {args.outdir}")
    print(f"[emit] submit a Slurm array of size {len(boxes)} (0..{len(boxes)-1}); "
          f"each task: python node_obbt_hpc.py solve {args.outdir} $SLURM_ARRAY_TASK_ID")


# --------------------------------------------------------------------------- #
# solve — bound ONE box (one Slurm array task)
# --------------------------------------------------------------------------- #
def cmd_solve(args):
    spec = json.load(open(os.path.join(args.outdir, "spec.json")))
    bx = json.load(open(os.path.join(args.outdir, f"box_{args.box_id:04d}.json")))
    g = build_grid(spec["grid"])
    bL_box = np.array(g["bL"], float).copy()
    bU_box = np.array(g["bU"], float).copy()
    for ell, v in bx["lo"].items():
        bL_box[int(ell)] = float(v)
    for ell, v in bx["hi"].items():
        bU_box[int(ell)] = float(v)
    dec = _make_decomp(g, bL_box, bU_box, spec["budget"],
                       spec["seed_interp"], spec["max_iter"])
    res = dec.run(**g["run_kwargs"])
    term = res.get("termination_reason")
    # An INFEASIBLE box (no CE cheaper than F_hint ≥ F*) cannot contain the
    # global optimum, so its valid LB on F* over that region is +∞ — it must NOT
    # pull global_LB down (the run() placeholder master_LB=0 would).  Mark it so
    # the aggregator excludes it.
    box_feasible = term != "master_infeasible"
    out = dict(
        box_id=args.box_id, grid=spec["grid"],
        feasible=box_feasible,
        master_LB=(float(res.get("master_LB", 0.0)) if box_feasible else None),
        best_F=(float(res["best_F"]) if res.get("best_F") not in (None, float("inf"))
                else None),
        best_b=(list(map(float, res["best_b"])) if res.get("best_b") is not None else None),
        termination=term,
        box_lo=bx["lo"], box_hi=bx["hi"],
    )
    json.dump(out, open(os.path.join(args.outdir, f"result_{args.box_id:04d}.json"), "w"),
              indent=2)
    print(f"[solve] box {args.box_id}: LB={out['master_LB']:.4f}  best_F={out['best_F']}  "
          f"term={out['termination']}")


# --------------------------------------------------------------------------- #
# aggregate — global_LB = min ObjBound, global_UB = min oracle-verified CE
# --------------------------------------------------------------------------- #
def cmd_aggregate(args):
    spec = json.load(open(os.path.join(args.outdir, "spec.json")))
    results = [json.load(open(p)) for p in
               sorted(glob.glob(os.path.join(args.outdir, "result_*.json")))]
    if not results:
        print("[aggregate] no result_*.json found — run `solve` for each box first.")
        return
    n_expected = spec["n_boxes"]
    if len(results) < n_expected:
        print(f"[aggregate] WARNING: {len(results)}/{n_expected} boxes done — the LB is "
              f"still valid (min over a SUBSET is ≤ the full-partition min ≤ F*), but it "
              f"will only RISE as the remaining boxes report.")
    # A box is INFEASIBLE (no CE cheaper than F_hint ≥ F*) when its master is
    # infeasible: its region cannot hold the global optimum, so its valid LB is
    # +∞ and it is EXCLUDED from the min.  (Back-compat: old result files have no
    # "feasible" field — infer it from the termination reason.)
    def _feasible(r):
        if "feasible" in r:
            return bool(r["feasible"]) and r["master_LB"] is not None
        return r.get("termination") != "master_infeasible"
    feas = [r for r in results if _feasible(r)]
    n_infeas = len(results) - len(feas)
    if not feas:
        print("[aggregate] all reported boxes infeasible — global_LB = +inf "
              "(no CE cheaper than F_hint in any reported box).")
        global_LB = float("inf")
    else:
        global_LB = min(r["master_LB"] for r in feas)
    # global_UB: every feasible box's oracle-verified incumbent, AND the hint CE
    # itself (a known GLOBAL counterfactual, F(b_hat) ≥ F*), which floors the UB
    # regardless of which box it falls in.
    g0 = build_grid(spec["grid"])
    bh = g0["b_hint"]; b0 = g0["b0"]; w = g0["w"]; free = g0["free_idx"]
    hint_F = float(sum(w[ell] * abs(bh[ell] - b0[ell]) for ell in free))
    ce = [r for r in feas if r["best_F"] is not None]
    box_UB = min((r["best_F"] for r in ce), default=float("inf"))
    global_UB = min(hint_F, box_UB)
    best = min(ce, key=lambda r: r["best_F"]) if ce else None
    gap = (global_UB - global_LB) / abs(global_UB) if np.isfinite(global_UB) and global_UB != 0 else float("nan")
    # Re-verify the incumbent behind global_UB with the oracle (defence-in-depth):
    # the best box CE if one beats the hint, else the hint b_hat itself.
    b_star = (np.array(best["best_b"], float) if (best is not None and box_UB <= hint_F
              and best["best_b"] is not None) else np.array(bh, float))
    vp, _, _ = g0["oracle"].solve_plain(b_star)
    vd, _, _ = g0["oracle"].solve_foil(b_star)
    verified = (vd is not None and vp is not None and vd <= vp + 1e-3)
    feas_lbs = [r["master_LB"] for r in feas]
    print("=" * 64)
    print(f"[aggregate] grid=IEEE{spec['grid']}  boxes={len(results)}/{n_expected}"
          f"  ({n_infeas} infeasible/pruned)")
    print(f"  global_LB (min feasible-box ObjBound) = {global_LB:.6f}")
    print(f"  global_UB (min CE; hint F={hint_F:.4f})  = {global_UB:.6f}"
          f"  [oracle re-verified: {verified}]")
    print(f"  gap = {100*gap:.2f}%"
          + ("   ✅ CERTIFIED" if np.isfinite(gap) and gap <= 1e-3 else ""))
    if feas_lbs:
        print(f"  feasible-box LB range: [{min(feas_lbs):.4f}, {max(feas_lbs):.4f}]")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Parallel node-OBBT for NLHPC")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit"); e.add_argument("grid"); e.add_argument("n_boxes", type=int)
    e.add_argument("outdir")
    e.add_argument("--dims", type=int, default=2, help="[grid mode] # most-CE-expanded free lines to split (grid = segs**dims boxes)")
    e.add_argument("--adaptive", action="store_true", help="v2: carve a best-first CE-concentrated frontier instead of a uniform grid")
    e.add_argument("--emit-budget", type=float, default=60.0, help="[adaptive] cheap per-box carve time limit (s)")
    e.add_argument("--budget", type=float, default=300.0, help="per-box master time limit (s) for the parallel `solve` step")
    e.add_argument("--seed-interp", type=int, default=3)
    e.add_argument("--max-iter", type=int, default=2)
    e.set_defaults(func=cmd_emit)
    s = sub.add_parser("solve"); s.add_argument("outdir"); s.add_argument("box_id", type=int)
    s.set_defaults(func=cmd_solve)
    a = sub.add_parser("aggregate"); a.add_argument("outdir"); a.set_defaults(func=cmd_aggregate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
