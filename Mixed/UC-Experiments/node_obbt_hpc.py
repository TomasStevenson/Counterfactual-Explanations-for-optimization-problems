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

Four subcommands:
    emit      <grid> <n_boxes> <outdir> [--dims D] [--budget S] [--seed-interp K] [--max-iter N]
    solve     <outdir> <box_id>
    aggregate <outdir>
    drive     <outdir> [--grid G ...]   # unattended campaign: emit -> solve array ->
                                        # harvest/aggregate -> split-worst -> repeat
                                        # until the gap certifies (see cmd_drive).

Local end-to-end (no cluster) is just: emit, then a loop of `solve`, then aggregate.
On NLHPC: emit once, submit `node_obbt.slurm` as an array of size n_boxes, then aggregate —
or let `drive` run the whole loop from a login node:
  CE_ALPHA=0.05 nohup python node_obbt_hpc.py drive runs/nobbt14_a05 > drive14.log 2>&1 &

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

ALPHA = float(os.environ.get("CE_ALPHA", "0.10"))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
_FNAME = {"14": "ieee14_enhanced.json", "39": "ieee39_newengland.json",
          "57": "ieee57_uc_matpower.json"}
# Per-grid quick_setup — MUST MATCH build_decomp_notebook.py / _smoke_strongdual.py.
# carbon_price=0.0 on all grids (carbon-free campaign).
_SETUP = {
    "14": dict(carbon_price=0.0, voll=20_000.0, slack_bus=None),
    "39": dict(carbon_price=0.0, voll=20_000.0, slack_bus=None),
    "57": dict(carbon_price=0.0, voll=500.0,    slack_bus=0),
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
    print(f"[build_grid] IEEE{grid} emissions-reduction foil ALPHA={ALPHA:g}", flush=True)
    # Campaign knobs (env-driven; defaults reproduce the certified baseline exactly):
    #   CE_THR        line-count axis — congestion threshold; LOWER = more mutable lines (default 0.75)
    #   CE_SCALE_DOWN negative perturbation — box lower bound = CE_SCALE_DOWN*b0 (1.0 = positive-only, 0.8 = -20%)
    #   CE_SCALE_UP   box upper bound = CE_SCALE_UP*b0 (default 1.2 = +20%)
    _thr = float(os.environ.get("CE_THR", "0.75"))
    _sd = float(os.environ.get("CE_SCALE_DOWN", "1.0"))
    _su = float(os.environ.get("CE_SCALE_UP", "1.2"))
    free_idx, util = get_congested_free_lines(solF, b0, thr=_thr)
    bL, bU = build_b_bounds(b0, free_idx, scale_up=_su, scale_down=_sd)
    w = make_line_weights(DATA, b0, util=util)
    print(f"[build_grid] IEEE{grid}  CE_THR={_thr:g}  box=[{_sd:g},{_su:g}]*b0  "
          f"carbon={float(DATA.carbon_price):g}  free_lines={len(free_idx)}", flush=True)
    oracle = UCWeakWCEOracle(
        data=DATA, cvec=cvec, idx=idx, window_size=T, per_bus_neutrality=True,
        u_init=u_init, p_init=p_init, on_t=on_t, off_t=off_t,
        foil_extra_constr_fn=foil_fn, output_flag=0,
    )
    # The warm-start hint MUST be the Branch-&-Sandwich CE (the pre-DECOMP B&S
    # phase): load bs_<grid>_checkpoint.json. If it is missing we fall back to bU
    # (max expansion) — NOT a B&S CE — and flag it LOUDLY so a run can never
    # silently skip the B&S phase. hint_source + the B&S scalars are returned so
    # the coordinator can record the hint value and the hint gap.
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"bs_{grid}_checkpoint.json")
    bs_best_F = bs_LB = None
    if os.path.exists(ckpt):
        _bs = json.load(open(ckpt))
        _best_b = _bs.get("best_b")
        bs_best_F = _bs.get("best_F")
        bs_LB = _bs.get("global_LB")
        if _best_b is not None:
            b_hint = np.array(_best_b, float)
            hint_source = "bs_checkpoint"
        else:
            b_hint = b0.copy(); b_hint[free_idx] = bU[free_idx]
            hint_source = "bU_fallback"
            print(f"[build_grid] *** WARNING: bs_{grid}_checkpoint.json has no feasible CE "
                  f"(best_b=null) — hint falls back to bU. ***", flush=True)
    else:
        bs_best_F = bs_LB = None
        b_hint = b0.copy(); b_hint[free_idx] = bU[free_idx]
        hint_source = "bU_fallback"
        print(f"[build_grid] *** WARNING: bs_{grid}_checkpoint.json NOT FOUND — the hint "
              f"fell back to bU (max expansion), which is NOT a Branch-&-Sandwich CE. The "
              f"B&S phase is effectively SKIPPED. Restore the checkpoint before trusting "
              f"this run. ***")
    run_kwargs = dict(window_size=T, per_bus_neutrality=True, u_init=u_init,
                      p_init=p_init, on_time_init=on_t, off_time_init=off_t)
    return dict(DATA=DATA, idx=idx, cvec=cvec, b0=b0, foil_fn=foil_fn, w=w,
                oracle=oracle, free_idx=list(free_idx), bL=bL, bU=bU, b_hint=b_hint,
                hint_source=hint_source, bs_best_F=bs_best_F, bs_LB=bs_LB,
                E_factual=E_fac, run_kwargs=run_kwargs)


def _make_decomp(g, bL_box, bU_box, budget, seed_interp, max_iter,
                 node=False, max_nodes=1, b_hint_override=None, time_limit=None):
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
        time_limit=time_limit,
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
    if args.adaptive:
        # Persist the carve incumbent — the driver may find a better STRICT CE
        # than the hint during the carve (it did: F=1.5110 vs hint 1.7613 on
        # IEEE14 a05) and emit used to throw it away.
        _bF = res.get("F_opt"); _bb = res.get("b_hat")
        json.dump(dict(best_F=(float(_bF) if _bF not in (None, float("inf")) else None),
                       best_b=(list(map(float, _bb)) if _bb is not None else None)),
                  open(os.path.join(args.outdir, "incumbent.json"), "w"), indent=2)
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
        # Driver run() returns F_opt / b_hat (NOT best_F / best_b — reading the
        # wrong keys silently reported every in-box CE as None).
        best_F=(float(res["F_opt"]) if res.get("F_opt") not in (None, float("inf"))
                else None),
        best_b=(list(map(float, res["b_hat"])) if res.get("b_hat") is not None else None),
        termination=term,
        box_lo=bx["lo"], box_hi=bx["hi"],
    )
    json.dump(out, open(os.path.join(args.outdir, f"result_{args.box_id:04d}.json"), "w"),
              indent=2)
    _lb = out["master_LB"]
    print(f"[solve] box {args.box_id}: LB={_lb if _lb is None else format(_lb, '.4f')}  "
          f"best_F={out['best_F']}  term={out['termination']}")


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
    # A carve incumbent saved by emit (incumbent.json) is one more UB candidate;
    # if it wins the min it goes through the same oracle re-verification below.
    _inc_p = os.path.join(args.outdir, "incumbent.json")
    if os.path.exists(_inc_p):
        _inc = json.load(open(_inc_p))
        if _inc.get("best_F") is not None and _inc.get("best_b") is not None:
            ce.append(dict(best_F=float(_inc["best_F"]), best_b=_inc["best_b"]))
    # Oracle re-verification (defence-in-depth): walk the UB candidates in
    # ascending F and report the FIRST one the exact oracle confirms. A
    # candidate that fails (e.g. a hint registered on trust that sits a hair
    # outside the razor-thin CE boundary — seen on IEEE-14 a05, line 15 at
    # 46.79998 vs 46.8) is DISCARDED loudly: it must never certify, and it
    # must not silently sink the reported UB either.
    cands = sorted((r for r in ce if r.get("best_b") is not None),
                   key=lambda r: r["best_F"])[:10]
    cands.append(dict(best_F=hint_F, best_b=[float(x) for x in bh]))  # B&S hint floors the list
    b_star = None
    global_UB = float("inf")
    ub_discarded = []
    for r in cands:
        bb = np.array(r["best_b"], float)
        vp, _, _ = g0["oracle"].solve_plain(bb)
        vd, _, _ = g0["oracle"].solve_foil(bb)
        if vd is not None and vp is not None and vd <= vp + 1e-3:
            b_star = bb
            global_UB = float(r["best_F"])
            break
        ub_discarded.append(float(r["best_F"]))
        print(f"[aggregate] *** UB candidate F={r['best_F']:.6f} FAILED oracle "
              f"re-verification (v_foil > v_plain at that b) — discarded. ***")
    verified = b_star is not None
    if ub_discarded:
        # Purge the rejected CE claims from their result files so the cheap
        # harvest cannot resurrect them (each master_LB stays — it is valid).
        for p in glob.glob(os.path.join(args.outdir, "result_[0-9]*.json")):
            rr = json.load(open(p))
            if rr.get("best_F") in ub_discarded:
                rr["ub_rejected"] = rr["best_F"]
                rr["best_F"] = None
                rr["best_b"] = None
                json.dump(rr, open(p, "w"), indent=2)
    if b_star is None:
        b_star = np.array(bh, float)          # report the hint point, unverified
    elif ub_discarded:
        # The honest UB now lives in incumbent.json so the drive loop and any
        # later cheap aggregate work from a verified incumbent only.
        json.dump(dict(best_F=global_UB, best_b=[float(x) for x in b_star]),
                  open(_inc_p, "w"), indent=2)
        print(f"[aggregate] incumbent.json rewritten to the verified UB "
              f"F={global_UB:.6f}.")
    gap = ((global_UB - global_LB) / abs(global_UB)
           if np.isfinite(global_UB) and global_UB != 0 else float("nan"))
    feas_lbs = [r["master_LB"] for r in feas]
    certified = bool(np.isfinite(gap) and gap <= 1e-3)
    print("=" * 64)
    print(f"[aggregate] grid=IEEE{spec['grid']}  boxes={len(results)}/{n_expected}"
          f"  ({n_infeas} infeasible/pruned)")
    print(f"  global_LB (min feasible-box ObjBound) = {global_LB:.6f}")
    print(f"  global_UB (min CE; hint F={hint_F:.4f})  = {global_UB:.6f}"
          f"  [oracle re-verified: {verified}]")
    print(f"  gap = {100*gap:.2f}%" + ("   ✅ CERTIFIED" if certified else ""))
    if feas_lbs:
        print(f"  feasible-box LB range: [{min(feas_lbs):.4f}, {max(feas_lbs):.4f}]")
    print("=" * 64)
    # Persist the official (oracle-verified) summary so the `drive` orchestrator —
    # which runs this step inside an srun allocation — can read it back.
    json.dump(dict(grid=spec["grid"], n_results=len(results), n_expected=n_expected,
                   n_infeasible=n_infeas, global_LB=global_LB, global_UB=global_UB,
                   hint_F=hint_F, gap=gap, certified=certified,
                   oracle_verified=bool(verified), ub_discarded=ub_discarded,
                   b_star=[float(x) for x in b_star]),
              open(os.path.join(args.outdir, "aggregate_summary.json"), "w"), indent=2)


# --------------------------------------------------------------------------- #
# drive — unattended split-worst orchestration of the whole campaign
#
# Automates the loop that certified IEEE-14 a05 manually over 12 rounds:
#   emit (once) -> sbatch --wait the open boxes -> harvest incumbents ->
#   cheap aggregate -> certified? official oracle aggregate via srun :
#   else split every box pinning the bound and repeat.
#
# The drive process itself only does file bookkeeping and blocking sbatch/srun
# calls (the run_coordinator_leftraru.sh pattern) — it is safe on a login node.
# Run it detached:  nohup python node_obbt_hpc.py drive runs/<dir> ... &
#
# VALIDITY: identical to the manual recipe. Splitting replaces a parent box by
# an exhaustive quartering of itself, so the leaf set stays an exhaustive
# partition of the root b-box; the parent's stale bound is retired
# (result_NNNN.json -> retired_result_NNNN.json, excluded from every glob) and
# its oracle-verified incumbent is folded into incumbent.json BEFORE retiring,
# so no UB information is ever lost.
# --------------------------------------------------------------------------- #

def _drive_boxes(outdir):
    """All box ids present as box_NNNN.json."""
    return sorted(int(os.path.basename(p)[4:8])
                  for p in glob.glob(os.path.join(outdir, "box_[0-9]*.json")))


def _drive_results(outdir):
    """id -> result dict, live results only (retired parents excluded by glob)."""
    out = {}
    for p in glob.glob(os.path.join(outdir, "result_[0-9]*.json")):
        r = json.load(open(p))
        out[int(r["box_id"])] = r
    return out


def _drive_retired(outdir):
    return {int(os.path.basename(p)[len("retired_result_"):len("retired_result_") + 4])
            for p in glob.glob(os.path.join(outdir, "retired_result_[0-9]*.json"))}


def _harvest_incumbent(outdir, results):
    """Fold every oracle-verified box CE into incumbent.json; return (F, b)."""
    inc_p = os.path.join(outdir, "incumbent.json")
    cands = []
    if os.path.exists(inc_p):
        inc = json.load(open(inc_p))
        if inc.get("best_F") is not None and inc.get("best_b") is not None:
            cands.append((float(inc["best_F"]), inc["best_b"]))
    for r in results.values():
        if r.get("best_F") is not None and r.get("best_b") is not None:
            cands.append((float(r["best_F"]), r["best_b"]))
    if not cands:
        return float("inf"), None
    best_F, best_b = min(cands, key=lambda t: t[0])
    json.dump(dict(best_F=best_F, best_b=[float(x) for x in best_b]),
              open(inc_p, "w"), indent=2)
    return best_F, best_b


def _cheap_aggregate(outdir, tol):
    """Round-decision bookkeeping from the result files alone — NO build_grid, NO
    Gurobi. UB = min oracle-verified CE seen so far (incumbent + box CEs); LB and
    the below-bar set come from the live per-box master_LB values. The official
    certificate still goes through cmd_aggregate's oracle re-verification."""
    boxes = _drive_boxes(outdir)
    results = _drive_results(outdir)
    retired = _drive_retired(outdir)
    open_ids = [b for b in boxes if b not in results and b not in retired]
    UB, _ = _harvest_incumbent(outdir, results)
    feas = {i: r for i, r in results.items()
            if r.get("feasible") and r.get("master_LB") is not None}
    LB = min((r["master_LB"] for r in feas.values()), default=float("inf"))
    bar = UB * (1.0 - tol) if np.isfinite(UB) else float("inf")
    below_bar = sorted((i for i, r in feas.items() if r["master_LB"] < bar),
                       key=lambda i: feas[i]["master_LB"])
    gap = ((UB - LB) / abs(UB)
           if np.isfinite(UB) and np.isfinite(LB) and UB != 0 else float("nan"))
    return dict(open_ids=open_ids, n_boxes=len(boxes), n_results=len(results),
                n_retired=len(retired), UB=UB, LB=LB, gap=gap, below_bar=below_bar)


def split_box(outdir, box_id, n_dims=2):
    """The manual round recipe: quarter box `box_id` along its `n_dims` widest
    dimensions (absolute width), append the children as new boxes, retire the
    parent's result. Caller must harvest incumbents first."""
    import itertools
    bx = json.load(open(os.path.join(outdir, f"box_{box_id:04d}.json")))
    lo, hi = bx["lo"], bx["hi"]
    widths = {k: float(hi[k]) - float(lo[k]) for k in lo}
    dims = sorted(widths, key=widths.get, reverse=True)[:max(1, min(n_dims, len(widths)))]
    if widths[dims[0]] <= 1e-9:
        raise RuntimeError(f"box {box_id} is degenerate (max width "
                           f"{widths[dims[0]]:.2e}) — cannot split further.")
    mids = {k: 0.5 * (float(lo[k]) + float(hi[k])) for k in dims}
    spec_p = os.path.join(outdir, "spec.json")
    spec = json.load(open(spec_p))
    child_ids = []
    nb = spec["n_boxes"]
    for combo in itertools.product((0, 1), repeat=len(dims)):
        clo, chi = dict(lo), dict(hi)
        for k, h in zip(dims, combo):
            if h == 0:
                chi[k] = mids[k]
            else:
                clo[k] = mids[k]
        json.dump({"lo": clo, "hi": chi},
                  open(os.path.join(outdir, f"box_{nb:04d}.json"), "w"))
        child_ids.append(nb)
        nb += 1
    spec["n_boxes"] = nb
    json.dump(spec, open(spec_p, "w"), indent=1)
    res_p = os.path.join(outdir, f"result_{box_id:04d}.json")
    if os.path.exists(res_p):
        os.replace(res_p, os.path.join(outdir, f"retired_result_{box_id:04d}.json"))
    print(f"[drive] split box {box_id} along dims {dims} -> children {child_ids}")
    return child_ids


def _hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _round_resources(budget, max_iter, deep=False):
    """Memory/walltime heuristic measured on the IEEE-14 a05 campaign: boxes
    solved at deep budgets (>=1800 s master TL) peak well above 12G — 12G OOMs,
    24G holds. Walltime covers max_iter CCG iterations at full budget + margin."""
    deep = deep or budget >= 1800
    if deep:
        return "24G", "05:00:00"      # the proven IEEE-14 a05 deep-round envelope
    secs = min(int(1.3 * max_iter * budget) + 900, 5 * 3600)
    return "12G", _hms(max(secs, 3000))


def _run(cmd, dry):
    """Run a blocking cluster command; in --dry-run print it and stop the drive."""
    import subprocess
    print(f"[drive] $ {' '.join(cmd)}", flush=True)
    if dry:
        print("[drive] --dry-run: stopping before the cluster command above.")
        sys.exit(0)
    return subprocess.run(cmd, check=False).returncode


def _export_arg():
    a = os.environ.get("CE_ALPHA")
    return "--export=ALL" + (f",CE_ALPHA={a}" if a else "")


def _sbatch_solve(outdir, ids, mem, walltime, maxconc, slurm_script, dry):
    idlist = ",".join(str(i) for i in ids)
    return _run(["sbatch", "--wait", f"--array={idlist}%{maxconc}",
                 f"--mem={mem}", f"--time={walltime}", _export_arg(),
                 slurm_script, outdir], dry)


def _drive_log(outdir, entry):
    import datetime
    p = os.path.join(outdir, "drive_log.json")
    log = json.load(open(p)) if os.path.exists(p) else []
    entry["at"] = datetime.datetime.now().isoformat(timespec="seconds")
    log.append(entry)
    json.dump(log, open(p, "w"), indent=1)


def cmd_drive(args):
    import datetime, shutil
    outdir = args.outdir
    t0 = __import__("time").time()
    here = os.path.dirname(os.path.abspath(__file__))
    slurm_script = os.path.join(here, "node_obbt.slurm")

    # ---- emit (first run only): the adaptive carve is a real solve — submit it
    # as its own job, never run it on the login node. ------------------------
    if not os.path.exists(os.path.join(outdir, "spec.json")):
        if not args.grid:
            sys.exit("[drive] no spec.json in outdir and no --grid given: pass "
                     "--grid and --n-boxes for the initial emit.")
        os.makedirs(outdir, exist_ok=True)
        emit_sh = os.path.join(outdir, "drive_emit.slurm")
        with open(emit_sh, "w", newline="\n") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name=nobbt_emit
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output={outdir}/emit_%j.out
#SBATCH --error={outdir}/emit_%j.err
module load gurobi
export PYTHONPATH="$GUROBI_HOME/lib/python3.13/site-packages:${{PYTHONPATH:-}}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export GRB_THREADS=$SLURM_CPUS_PER_TASK
srun python {os.path.join(here, 'node_obbt_hpc.py')} emit {args.grid} {args.n_boxes} \\
    {outdir} --adaptive --emit-budget {args.emit_budget} --budget {args.budget} \\
    --seed-interp {args.seed_interp} --max-iter {args.max_iter}
""")
        rc = _run(["sbatch", "--wait", _export_arg(), emit_sh], args.dry_run)
        if rc != 0 or not os.path.exists(os.path.join(outdir, "spec.json")):
            sys.exit(f"[drive] emit job failed (rc={rc}) or produced no spec.json — "
                     f"see {outdir}/emit_*.err")

    spec = json.load(open(os.path.join(outdir, "spec.json")))
    print(f"[drive] campaign {outdir}: grid=IEEE{spec['grid']}  budget={spec['budget']:g}s"
          f"  max_iter={spec['max_iter']}  tol={args.tol:g}  CE_ALPHA="
          f"{os.environ.get('CE_ALPHA', '(default 0.10)')}", flush=True)

    rounds = 0
    nonconfirm = 0
    while True:
        agg = _cheap_aggregate(outdir, args.tol)
        print(f"[drive] round {rounds}: boxes={agg['n_boxes']} ({agg['n_retired']} retired)"
              f"  open={len(agg['open_ids'])}  LB={agg['LB']:.6f}  UB={agg['UB']:.6f}"
              f"  gap={100 * agg['gap']:.3f}%  below-bar={agg['below_bar']}", flush=True)

        # 1) open boxes -> solve them (this is a "round").
        if agg["open_ids"]:
            if rounds >= args.max_rounds:
                print(f"[drive] max-rounds={args.max_rounds} reached with boxes still "
                      f"open — stopping. Resume with the same command.")
                return
            rounds += 1
            mem, wt = _round_resources(spec["budget"], spec["max_iter"])
            _sbatch_solve(outdir, agg["open_ids"], mem, wt, args.maxconc,
                          slurm_script, args.dry_run)
            missing = _cheap_aggregate(outdir, args.tol)["open_ids"]
            if missing:
                # OOM/failure recovery: one retry at deep resources.
                print(f"[drive] {len(missing)} box(es) returned no result "
                      f"({missing}) — resubmitting once at deep memory.")
                mem, wt = _round_resources(spec["budget"], spec["max_iter"], deep=True)
                _sbatch_solve(outdir, missing, mem, wt, args.maxconc,
                              slurm_script, args.dry_run)
                missing = _cheap_aggregate(outdir, args.tol)["open_ids"]
                if missing:
                    sys.exit(f"[drive] boxes {missing} failed twice — inspect "
                             f"runs/node_obbt_*_{missing[0]}.err before resuming.")
            _drive_log(outdir, dict(event="solved", round=rounds,
                                    ids=agg["open_ids"], mem=mem, time=wt))
            nonconfirm = 0
            continue

        # 2) nothing open and the bound clears the bar -> official certificate.
        if np.isfinite(agg["gap"]) and agg["gap"] <= args.tol:
            print(f"[drive] cheap gap {100 * agg['gap']:.3f}% <= tol — running the "
                  f"official oracle-verified aggregate.")
            _run(["srun", "--partition=main", "--cpus-per-task=4", "--mem=8G",
                  "-t", "00:25:00", "python",
                  os.path.join(here, "node_obbt_hpc.py"), "aggregate", outdir],
                 args.dry_run)
            summ_p = os.path.join(outdir, "aggregate_summary.json")
            summ = json.load(open(summ_p)) if os.path.exists(summ_p) else None
            if summ and summ.get("certified") and summ.get("oracle_verified"):
                cert = dict(summ, tol=args.tol, rounds=rounds,
                            wall_s=round(__import__("time").time() - t0, 1),
                            finished=datetime.datetime.now().isoformat(timespec="seconds"))
                json.dump(cert, open(os.path.join(outdir, "certificate.json"), "w"),
                          indent=2)
                print(f"[drive] ✅ CERTIFIED: F*={summ['global_UB']:.6f}  "
                      f"LB={summ['global_LB']:.6f}  gap={100 * summ['gap']:.3f}%  "
                      f"({rounds} rounds, {_hms(__import__('time').time() - t0)}) — "
                      f"certificate.json written.")
                _drive_log(outdir, dict(event="certified", round=rounds,
                                        F=summ["global_UB"], LB=summ["global_LB"]))
                return
            # The aggregate discarded unverifiable UB candidate(s) and rewrote
            # incumbent.json to the best ORACLE-VERIFIED one — re-enter the loop
            # so pruning/splitting continue against the honest bar. Two
            # consecutive non-confirmations without progress = a real problem.
            nonconfirm += 1
            if nonconfirm >= 2:
                sys.exit("[drive] official aggregate failed to confirm twice in a "
                         "row — inspect aggregate_summary.json (ub_discarded) and "
                         f"{outdir}/incumbent.json; not writing a certificate.")
            print("[drive] official aggregate did not confirm (unverifiable UB "
                  "candidate discarded); continuing against the verified "
                  "incumbent.")
            continue

        # 3) bound below the bar -> split the pinning box(es) and loop.
        if not agg["below_bar"]:
            sys.exit(f"[drive] no open boxes, gap {100 * agg['gap']:.3f}% > tol, but no "
                     f"feasible box sits below the bar — inconsistent state (all boxes "
                     f"infeasible with a finite UB?). Inspect {outdir} manually.")
        to_split = agg["below_bar"][:args.max_splits_per_round]
        for bid in to_split:
            split_box(outdir, bid)
        _drive_log(outdir, dict(event="split", round=rounds, parents=to_split))
        nonconfirm = 0


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
    d = sub.add_parser("drive", help="unattended split-worst campaign to certification")
    d.add_argument("outdir")
    d.add_argument("--grid", choices=list(_FNAME), help="required for the initial emit "
                   "(omit to resume an existing outdir)")
    d.add_argument("--n-boxes", type=int, default=64, help="[emit] adaptive carve target")
    d.add_argument("--emit-budget", type=float, default=60.0)
    d.add_argument("--budget", type=float, default=3600.0, help="[emit] per-box master TL (s)")
    d.add_argument("--seed-interp", type=int, default=0)
    d.add_argument("--max-iter", type=int, default=3)
    d.add_argument("--tol", type=float, default=1e-3, help="relative certification gap")
    d.add_argument("--max-rounds", type=int, default=20, help="max solve submissions")
    d.add_argument("--maxconc", type=int, default=11, help="array throttle (88-core grant / 8)")
    d.add_argument("--max-splits-per-round", type=int, default=8)
    d.add_argument("--dry-run", action="store_true",
                   help="do all local bookkeeping but stop at the first cluster command")
    d.set_defaults(func=cmd_drive)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
