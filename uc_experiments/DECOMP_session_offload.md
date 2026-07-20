# DECOMP — session offload (cold-pickup handoff)

*Refreshed 2026-06-06. Drop into a new chat and say: "Read DECOMP_session_offload.md and
DECOMP_state.md, then continue."  Authoritative current state + full bug history:
`DECOMP_state.md`.  Theory/optimality proof: `DECOMP_formulation_and_optimality.md`.
(Earlier 2026-05-24 offload content is in git history; the full bug history is in DECOMP_state.md.)*

Repo: `github.com/TomasStevenson/Counterfactual-Explanations-for-optimization-problems`,
dir `Mixed/UC-Experiments`. Branch `main` (pushed). Python env:
`C:\Users\tomas\miniconda3\envs\ce-env\python.exe` (run with `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`).

---

## 1. What this is

Counterfactual explanations (CE) for Unit Commitment: find the **minimum L1 change to transmission
line flow limits** `b=fmax` so a **foil** (a 10%-emissions-reduced schedule) becomes the
**cost-optimal** UC outcome — `min F(b) s.t. v_foil(b) ≤ v_plain(b)`. Method = **DECOMP**
(Column-and-Constraint Generation) with a strong-duality master. Grids: IEEE 14 / 39 / 57.

**The non-negotiable invariants** (this project has been bitten by invalid LBs three times):
`LB = ObjBound ≤ F*` always, and every reported CE is **oracle-verified** (`vd ≤ vp`) so `≥ F*`.
Certified ε-optimal when `UB − LB ≤ ε`. **Run `_validate_all.py` after ANY formulation change** —
it asserts the master never excludes the known B&S CE `b_BS` on all 3 grids.

## 2. Current standing (one standardized solver, all grids)

| grid | result (standardized config) |
|------|------------------------------|
| **IEEE 39** | **CERTIFIED 0.00%**, F=0.7060 (better than B&S 0.7138) |
| IEEE 57 | valid, ~1.5-7% gap depending on budget; strict CE ~10.36 (beats B&S 11.68) |
| IEEE 14 | valid, exact-MIQCP-hard; gap closes slowly; needs parallel budget |

The solver is `node-OBBT`: an explicit spatial branch-and-bound over the shared `b`-box (Gurobi
callbacks can't do node-local OBBT, so we drive the tree and call Gurobi only to bound each box).
`global_LB = min over leaf boxes of ObjBound ≤ F*`. One config runs all grids; it certifies where
easy (39 at box 1) and adds boxes only where the MIQCP is hard (14/57).

## 3. What was built this session (2026-06-05 → 06-06), all committed + pushed

- **OBBT validity FIX** — iterated OBBT's 2nd pass over-tightened μ and excluded `b_BS` on IEEE 39
  (model-state staleness). Fix: `obbt_iter` default 1 + corrected iterate + self-validating
  rollback guard (inactive when the hint is outside a sub-box). `_validate_all.py` all VALID.
- **Theory note** `DECOMP_formulation_and_optimality.md` — formulation, optimality proof, and why
  each upgrade is valid + faster.
- **Parallel port** `node_obbt_hpc.py` (emit/solve/aggregate, `emit --adaptive`) + `node_obbt.slurm`
  — one-shot static partition; validity-correct (infeasible boxes excluded; hint floors UB).
- **v3 coordinator** `node_obbt_coordinator.py` (init/solve-box/step/run-local/status) +
  `node_obbt_round.slurm` — ITERATIVE parallel best-first: solve OPEN boxes (Slurm array) → prune →
  split the `--split-k` weakest → repeat. Per-box `warm` is always a genuine CE (valid b_hat_hint).
  Validated: `global_LB` rises monotonically round-over-round on IEEE 14.
- **Global time limit + benchmark** — `UCDecomp4b(time_limit=…)` (whole-run wall budget, checked in
  the loop AND the preamble; no overshoot; returns `wall_time_s`, `hit_time_limit`). `run_benchmark.py`
  records CE / gap / time / hit-limit to `results.csv` + `results.json`.

Commits: `44f7777` (node-OBBT + OBBT fix), `38350ab` (theory + port + scaling), `299a97d` (v2 emit
+ cold-box fix), `1538556` (v3 coordinator), `ba3f535` (time limit + benchmark).

## 4. ✅ DONE — the 1-hour benchmark RESULTS (single machine, time_limit=3600 s, per-box 600 s)

`run_benchmark.py --time-limit 3600 --grids 39,57,14 --solvers node,coordinator --per-box-budget 600`
completed 2026-06-06. Full data in **`benchmark_results/results.csv`** (+ `.json` with full `b_hat`):

| grid | solver | CE F_opt | CE changes | LB | gap% | certified | wall s | hit_TL |
|------|--------|---------:|------------|-----:|-----:|-----------|-------:|--------|
| **IEEE 39** | **node** | **0.7060** | L13:+2.819 | 0.7060 | **0.0008** | ✅ **yes** | 599 | no |
| IEEE 39 | coordinator | 0.7138 | L13:+2.850 | 0.7060 | 1.09 | no | 3612 | yes |
| **IEEE 57** | **node** | **10.390** | L20:+19.9; L69:+33.7 | 9.874 | **4.97** | no | 3604 | yes |
| IEEE 57 | coordinator | 11.680 | L20:+13.1; L69:+41.5 | 8.475 | 27.44 | no | 3638 | yes |
| **IEEE 14** | **node** | **2.382** | L14:+6.8; L15:+7.6; L18:+2.9 | 1.633 | **31.44** | no | 3605 | yes |
| IEEE 14 | coordinator | 2.382 | (same) | 1.625 | 31.77 | no | 3607 | yes |

**Findings:** IEEE 39 (node) **certifies 0% in ~10 min** (minimal CE = expand line 13 by +2.82). The
**`node` solver is the sequential winner** (39 cert; 57 ~5%; 14 ~31%); `coordinator` is worse on one
machine because it rebuilds the master (incl. OBBT ~480 s on 39) PER box — it only pays off with
CONCURRENT boxes on HPC. IEEE 14 & 57 hit the 1 h limit ⇒ they need the parallel Leftraru budget to
certify. `--per-box-budget` must exceed the per-box OBBT (~480 s on 39) → keep ≥600 s.

## 5. Key levers / facts (so you don't re-derive them)

- **Per-box budget dominates box count** for the LB (`global_LB = min over leaves`; one undersolved
  box pins the floor). IEEE 14: 120 s/box → 1.35, 300 s/box → 1.58, F*≈2.37.
- **One-shot static distribution is pinned by the weakest box and never refines it** → v3 iterative
  refinement is the fix; gap-closing scales with per-box budget × parallel width.
- IEEE 14's feasible region is a thin low-`b` sliver (each split → 1 feasible + 1 infeasible child);
  most of the box certifies near F* easily, a small hard region is the bottleneck.
- node-OBBT internals: `bilinear_exact` (exact `b·μ` MIQCP), root OBBT (μ + b bound tightening),
  `seed_interp` (interior pattern seeding — needed for 39, lean for 14/57), analytic warm start
  (the master is hostile to cold incumbent search), running-max LB + multistart.

## 6. Recommended next steps (evaluate, then choose)

1. **Read the benchmark** (§4) and tabulate CE/gap/time/hit-limit per grid+solver. This is the
   immediate deliverable the user asked for.
2. **Run the gap-closer on NLHPC Leftraru** (partition `main`, gurobipy on PYTHONPATH, token-server
   license). The v3 coordinator parallelizes across rounds:
   `init` → loop {`sbatch --array=<open ids> node_obbt_round.slurm <dir>`; `step`} until certified.
   High per-box budget (≥600 s) × wide rounds is what should certify IEEE 14/57. (The assistant
   cannot submit on the cluster — the user runs this.)
3. **v3 refinements** (optional): skip solving infeasible-side children; per-box OBBT re-tightening
   only the split line's μ; warm-start children from the carve's per-box incumbents.
4. If a longer single-machine answer is wanted: re-run the benchmark at a larger `--time-limit`, or
   the in-process node solver with more `node_obbt_max_nodes` / per-box budget.

## 7. Files (all in Mixed/UC-Experiments)

- `uc_decomp_4b.py` — `UCDecomp4b` (the solver; node-OBBT driver, OBBT + guard, time_limit).
- `node_obbt_hpc.py` — one-shot parallel port (emit/solve/aggregate).
- `node_obbt_coordinator.py` + `node_obbt_round.slurm` — v3 iterative coordinator.
- `run_benchmark.py` — time-limited CE benchmark recorder.
- `_validate_all.py` / `_check_strongdual_valid.py` — the validity gate (slot 7 = obbt_iter).
- `_smoke_strongdual.py` — single-grid smoke (slots 14-16 = node/max_nodes/box_budget).
- `DECOMP_state.md` (authoritative), `DECOMP_formulation_and_optimality.md` (theory),
  `DECOMP_OBBT_offload.md`, `DECOMP_theoretical_strategies.md`.

## 8. Read order for a cold pickup
1. This file. 2. `benchmark_results/results.csv` (the live results). 3. `DECOMP_state.md`
§"Time-limited benchmark" + §"v3 iterative best-first coordinator". 4.
`DECOMP_formulation_and_optimality.md` for the why. Then proceed.
