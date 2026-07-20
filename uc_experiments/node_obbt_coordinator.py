"""v3 — iterative best-first coordinator for parallel node-OBBT (NLHPC).

The one-shot static partition (node_obbt_hpc.py) is validity-correct but its
global_LB is pinned by the WEAKEST box and it never refines that box. The
in-process driver fixes this by best-first splitting the weakest box, but is
sequential. This coordinator makes the best-first refinement PARALLEL via rounds:

    round r:  solve every OPEN box (an independent Slurm-array task)  ->
              collect ObjBound + in-box CE  ->  prune (lb >= UB - tol)  ->
              SPLIT the non-pruned leaf with the smallest lb (the one pinning
              global_LB)  ->  its children are the OPEN boxes of round r+1.

Two design points carried over from the findings:
  * VALIDITY: global_LB = min over non-pruned leaf boxes of each box's Gurobi
    ObjBound <= F*, at every round. global_UB = min(hint F, in-box CEs). Pruned
    boxes (lb >= UB - tol) cannot hold the optimum and are dropped.
  * PER-BOX WARM START: each box carries `warm`, a b that is ALWAYS A GENUINE CE
    (the global hint, or an in-box CE once one is found) so run()'s incumbent
    registration stays valid (no invalid UB). A child that contains the parent's
    in-box CE inherits it (a real warm start in that box); the other child falls
    back to the global CE (valid UB, solves cold but still bounds the region).

Subcommands:
    init       <grid> <outdir> [--budget S] [--seed-interp K] [--max-iter N] [--tol T]
    solve-box  <outdir> <box_id>        # one Slurm-array task
    step       <outdir>                 # coordinator: collect + prune + split
    run-local  <outdir> <max_rounds>    # solve+step loop on one machine (testing)
    status     <outdir>

NLHPC loop:  init  ->  repeat { sbatch --array=<open ids> node_obbt_round.slurm <outdir>;  step }  until certified.
"""
import os, sys, json, glob, argparse, time
import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")
from node_obbt_hpc import build_grid, _make_decomp

NEG_INF = -1e18   # JSON-safe stand-in for an unsolved box's lower bound


def _spath(outdir):  return os.path.join(outdir, "state.json")
def _rpath(outdir, i):  return os.path.join(outdir, f"result_{i:04d}.json")
def _load(outdir):  return json.load(open(_spath(outdir)))
def _save(outdir, st):  json.dump(st, open(_spath(outdir), "w"), indent=2)


def _box_bounds(g, box):
    bL = np.array(g["bL"], float).copy(); bU = np.array(g["bU"], float).copy()
    for ell, v in box["lo"].items(): bL[int(ell)] = float(v)
    for ell, v in box["hi"].items(): bU[int(ell)] = float(v)
    return bL, bU


def _warm_full(g, warm):
    b = np.array(g["b0"], float).copy()
    for ell, v in warm.items(): b[int(ell)] = float(v)
    return b


def _leaves(st):       return [b for b in st["boxes"] if b["status"] in ("open", "solved")]
def _open(st):         return [b for b in st["boxes"] if b["status"] == "open"]
def _global_LB(st):
    lv = _leaves(st)
    if not lv: return st["global_UB"]               # all pruned -> certified
    return min((b["lb"] if b["lb"] is not None else NEG_INF) for b in lv)


# --------------------------------------------------------------------------- #
def cmd_init(args):
    g = build_grid(args.grid)
    free = g["free_idx"]; bL = g["bL"]; bU = g["bU"]; bh = g["b_hint"]; b0 = g["b0"]
    hint_F = float(sum(g["w"][e] * abs(bh[e] - b0[e]) for e in free))
    root = {"id": 0,
            "lo": {int(e): float(bL[e]) for e in free},
            "hi": {int(e): float(bU[e]) for e in free},
            "lb": None, "warm": {int(e): float(bh[e]) for e in free},
            "status": "open", "best_F": None, "best_b": None}
    # B&S hint provenance (record the hint value + B&S's own gap, and prove the
    # hint really is the Branch-&-Sandwich CE rather than the bU fallback).
    bs_best_F = g.get("bs_best_F"); bs_LB = g.get("bs_LB")
    bs_gap_pct = (100.0 * (bs_best_F - bs_LB) / abs(bs_best_F)
                  if (bs_best_F not in (None, 0) and bs_LB is not None) else None)
    st = {"grid": args.grid, "tol": args.tol, "next_id": 1, "round": 0,
          "budget": args.budget, "seed_interp": args.seed_interp,
          "max_iter": args.max_iter, "split_k": args.split_k,
          "time_limit": args.time_limit,   # GLOBAL wall-clock budget for run-local
          "free_idx": [int(e) for e in free],
          "hint_F": hint_F, "global_UB": hint_F,
          "hint_source": g.get("hint_source"),   # "bs_checkpoint" | "bU_fallback"
          "bs_best_F": bs_best_F, "bs_LB": bs_LB, "bs_gap_pct": bs_gap_pct,
          "global_UB_b": [float(x) for x in bh],
          "boxes": [root]}
    os.makedirs(args.outdir, exist_ok=True)
    _save(args.outdir, st)
    src = g.get("hint_source")
    print(f"[init] grid=IEEE{args.grid}  free={len(free)}  hint_F={hint_F:.4f}  "
          f"hint_source={src}"
          + (f"  (B&S: F={bs_best_F:.4f} LB={bs_LB:.4f} gap={bs_gap_pct:.2f}%)"
             if bs_gap_pct is not None else "")
          + f"\n       root box id=0 (open).  Next: solve-box 0, then step.")
    if src != "bs_checkpoint":
        print("[init] *** WARNING: hint is NOT the Branch-&-Sandwich CE — the B&S phase "
              "was skipped (missing bs checkpoint). ***")


def cmd_solve_box(args):
    st = _load(args.outdir); g = build_grid(st["grid"])
    box = next(b for b in st["boxes"] if b["id"] == args.box_id)
    bL_box, bU_box = _box_bounds(g, box)
    warm = _warm_full(g, box["warm"])
    # per-box global cap: run-local passes the REMAINING global budget via args.tl;
    # standalone (Slurm) falls back to the per-box budget.
    tl = getattr(args, "tl", None) or st["budget"]
    dec = _make_decomp(g, bL_box, bU_box, st["budget"], st["seed_interp"],
                       st["max_iter"], b_hint_override=warm, time_limit=tl)
    res = dec.run(**g["run_kwargs"])
    term = res.get("termination_reason")
    feasible = term != "master_infeasible"
    out = {"id": args.box_id, "feasible": feasible,
           "lb": (float(res.get("master_LB", 0.0)) if feasible else None),
           "best_F": (float(res["best_F"]) if res.get("best_F") not in (None, float("inf")) else None),
           "best_b": (list(map(float, res["best_b"])) if res.get("best_b") is not None else None),
           "term": term}
    json.dump(out, open(_rpath(args.outdir, args.box_id), "w"), indent=2)
    print(f"[solve-box] {args.box_id}: lb={out['lb']}  best_F={out['best_F']}  term={term}")


def cmd_step(args):
    st = _load(args.outdir); tol = st["tol"]
    # 1) ingest results for every OPEN box that has one
    for box in st["boxes"]:
        if box["status"] != "open": continue
        rp = _rpath(args.outdir, box["id"])
        if not os.path.exists(rp):
            print(f"[step] WARNING: box {box['id']} has no result yet — skipping."); continue
        r = json.load(open(rp))
        if not r["feasible"]:
            box["status"] = "pruned"; box["lb"] = None       # +inf region
            continue
        box["lb"] = max(box["lb"] if box["lb"] is not None else NEG_INF, float(r["lb"]))
        if r["best_F"] is not None and r["best_F"] < st["global_UB"] - 1e-12:
            st["global_UB"] = float(r["best_F"]); st["global_UB_b"] = r["best_b"]
        box["best_F"] = r["best_F"]; box["best_b"] = r["best_b"]
        box["status"] = "solved"
    # 2) prune leaves that cannot beat the incumbent
    for box in st["boxes"]:
        if box["status"] == "solved" and box["lb"] is not None and box["lb"] >= st["global_UB"] - tol:
            box["status"] = "pruned"
    # 3) certificate
    gLB = _global_LB(st); gUB = st["global_UB"]
    gap = (gUB - gLB) / abs(gUB) if gUB not in (0, None) else float("nan")
    st["round"] += 1
    if gUB - gLB <= tol:
        _save(args.outdir, st)
        print(f"[step] round {st['round']}: global_LB={gLB:.4f}  global_UB={gUB:.4f}  "
              f"gap={100*gap:.2f}%  ✅ CERTIFIED"); return
    # 4) split the `split_k` non-pruned SOLVED leaves with the smallest lb (the
    #    ones pinning / nearest to pinning global_LB).  Splitting only the single
    #    weakest is the purest best-first; split_k>1 widens each round for better
    #    parallel (Slurm-array) utilisation at a small efficiency cost.
    cand = [b for b in st["boxes"] if b["status"] == "solved"]
    if not cand:
        _save(args.outdir, st)
        print(f"[step] round {st['round']}: no splittable leaf (open boxes pending or all "
              f"pruned). global_LB={gLB:.4f} global_UB={gUB:.4f} gap={100*gap:.2f}%"); return
    free = st["free_idx"]
    gUB_b = st["global_UB_b"]
    def _g(d, e): return float(d[str(e)]) if str(e) in d else float(d[e])  # JSON keys are str
    k = max(1, int(st.get("split_k", 1)))
    weakest = sorted(cand, key=lambda b: b["lb"] if b["lb"] is not None else NEG_INF)[:k]
    split_ids = []
    for weak in weakest:
        ell = max(free, key=lambda e: _g(weak["hi"], e) - _g(weak["lo"], e))
        lo = _g(weak["lo"], ell); hi = _g(weak["hi"], ell); mid = 0.5 * (lo + hi)
        pbest = weak["best_b"]
        for clo, chi in ((lo, mid), (mid, hi)):
            cid = st["next_id"]; st["next_id"] += 1
            child = {"id": cid,
                     "lo": {**{int(kk): float(v) for kk, v in weak["lo"].items()}, int(ell): float(clo)},
                     "hi": {**{int(kk): float(v) for kk, v in weak["hi"].items()}, int(ell): float(chi)},
                     "lb": weak["lb"], "status": "open",
                     "best_F": None, "best_b": None, "warm": None}
            # per-box warm CE: parent's in-box CE if it falls in this child, else global CE
            if pbest is not None and clo - 1e-9 <= pbest[int(ell)] <= chi + 1e-9:
                child["warm"] = {int(e): float(pbest[int(e)]) for e in free}
            else:
                child["warm"] = {int(e): float(gUB_b[int(e)]) for e in free}
            st["boxes"].append(child)
            rp = _rpath(args.outdir, cid)
            if os.path.exists(rp): os.remove(rp)
        weak["status"] = "internal"; split_ids.append(weak["id"])
    _save(args.outdir, st)
    new_open = [b["id"] for b in st["boxes"] if b["status"] == "open"]
    print(f"[step] round {st['round']}: split {len(split_ids)} weakest box(es) {split_ids}; "
          f"global_LB={gLB:.4f} global_UB={gUB:.4f} gap={100*gap:.2f}%. "
          f"Next round OPEN boxes: {new_open}")


def cmd_run_local(args):
    t0 = time.time()
    tlim = _load(args.outdir).get("time_limit")     # GLOBAL wall-clock budget
    hit = False
    def _rem():
        return (tlim - (time.time() - t0)) if tlim else float("inf")
    for _ in range(args.max_rounds):
        st = _load(args.outdir); open_ids = [b["id"] for b in _open(st)]
        if not open_ids:
            print("[run-local] no open boxes — done."); break
        for i in open_ids:
            if _rem() <= 0:
                hit = True
                print(f"[run-local] global time limit ({tlim:.0f}s) reached — stopping."); break
            cmd_solve_box(argparse.Namespace(outdir=args.outdir, box_id=i,
                                             tl=min(st["budget"], _rem())))
        if hit: break
        cmd_step(argparse.Namespace(outdir=args.outdir))
        st = _load(args.outdir)
        if st["global_UB"] - _global_LB(st) <= st["tol"]:
            break
        if _rem() <= 0:
            hit = True; print(f"[run-local] global time limit ({tlim:.0f}s) reached — stopping."); break
    # record wall-clock + hit-time-limit on the state
    st = _load(args.outdir)
    st["wall_time_s"] = time.time() - t0
    certified = st["global_UB"] - _global_LB(st) <= st["tol"]
    st["hit_time_limit"] = bool(hit or (tlim and not certified and st["wall_time_s"] >= tlim - 5.0))
    _save(args.outdir, st)
    cmd_status(argparse.Namespace(outdir=args.outdir))


def cmd_status(args):
    st = _load(args.outdir); gLB = _global_LB(st); gUB = st["global_UB"]
    gap = (gUB - gLB) / abs(gUB) if gUB not in (0, None) else float("nan")
    n = {k: sum(1 for b in st["boxes"] if b["status"] == k)
         for k in ("open", "solved", "pruned", "internal")}
    print("=" * 64)
    print(f"[status] grid=IEEE{st['grid']}  round={st['round']}  boxes={len(st['boxes'])} "
          f"(open={n['open']} solved={n['solved']} pruned={n['pruned']} internal={n['internal']})")
    print(f"  global_LB = {gLB:.6f}   global_UB = {gUB:.6f} (hint_F={st['hint_F']:.4f})")
    print(f"  gap = {100*gap:.2f}%" + ("   ✅ CERTIFIED" if gUB - gLB <= st["tol"] else ""))
    # Hint provenance + the hint's own optimality gap vs the proven LB (how far the
    # Branch-&-Sandwich CE sits above F* — the publishable "B&S suboptimality").
    hint_F = st.get("hint_F"); src = st.get("hint_source", "?")
    hint_gap_str = ("n/a (no box solved yet)" if (hint_F in (None, 0) or gLB < -1e17)
                    else f"{100 * (hint_F - gLB) / abs(hint_F):.2f}%")
    print(f"  hint: F={hint_F:.6f}  source={src}  gap_vs_LB={hint_gap_str}"
          + (f"   [B&S own: F={st['bs_best_F']:.4f} LB={st['bs_LB']:.4f} gap={st['bs_gap_pct']:.2f}%]"
             if st.get("bs_gap_pct") is not None else ""))
    if src != "bs_checkpoint":
        print("  *** WARNING: hint is NOT the B&S CE (B&S phase skipped) ***")
    if "wall_time_s" in st:
        print(f"  wall_time = {st['wall_time_s']:.1f}s   hit_time_limit = {st.get('hit_time_limit')}"
              + (f"  (limit {st['time_limit']:.0f}s)" if st.get("time_limit") else ""))
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="v3 iterative best-first node-OBBT coordinator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("grid"); i.add_argument("outdir")
    i.add_argument("--budget", type=float, default=300.0)
    i.add_argument("--seed-interp", type=int, default=0)
    i.add_argument("--max-iter", type=int, default=1)
    i.add_argument("--tol", type=float, default=1e-3)
    i.add_argument("--split-k", type=int, default=1, help="# weakest leaves to split per round (wider rounds = more parallel)")
    i.add_argument("--time-limit", type=float, default=None, help="GLOBAL wall-clock budget (s) enforced by run-local")
    i.set_defaults(func=cmd_init)
    s = sub.add_parser("solve-box"); s.add_argument("outdir"); s.add_argument("box_id", type=int)
    s.set_defaults(func=cmd_solve_box)
    st_ = sub.add_parser("step"); st_.add_argument("outdir"); st_.set_defaults(func=cmd_step)
    rl = sub.add_parser("run-local"); rl.add_argument("outdir"); rl.add_argument("max_rounds", type=int)
    rl.set_defaults(func=cmd_run_local)
    ss = sub.add_parser("status"); ss.add_argument("outdir"); ss.set_defaults(func=cmd_status)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
