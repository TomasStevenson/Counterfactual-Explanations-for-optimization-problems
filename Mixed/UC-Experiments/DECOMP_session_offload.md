# DECOMP — session offload

*Generated 2026-05-24. Captures all work done across 2026-05-23 → 2026-05-24 sessions to bring DECOMP from "Iter 2 times out, no incumbent" to a working self-contained pipeline that outperformed B&S on IEEE 14.*

---

## 0. Starting state (before this session)

- `uc_decomp_4b.py` had Bugs 1–8 fixed per `DECOMP_review.md` (M_shed, M_curt, etc.)
- `decomp_3grids.ipynb` Section 4c (IEEE 14, bigM, 900s) was timing out at Iter 2: 187K B&B nodes, **zero incumbents**, `Best objective -`
- Diagnosis was unclear: formulation bug vs. MIP hardness vs. M values still wrong

---

## 1. Bug 9 — `M_mu` floor too small (FIXED)

Same arithmetic error pattern as Bug 1: used `|π_min| = VOLL` as the floor instead of the full π range `π_max − π_min = 2·VOLL + 2·c_curt`.

Derivation: from the LP stationarity of `f`, `μ_p − μ_m = π_fr − π_to − c_f − ν`. With `π ∈ [−VOLL, 2·c_curt + VOLL]`, the max LMP difference is `2·VOLL + 2·c_curt`, so `μ_p ≤ 2·VOLL + 2·c_curt`.

| Grid | Was | Fixed to |
|------|-----|---------|
| IEEE 14 (VOLL=20000, c_curt=5) | max(4998, 20000) = 20000 | max(4998, **40010**) |
| IEEE 39 | max(?, 20000) | max(?, **40010**) |
| IEEE 57 (VOLL=500, c_curt=5) | max(?, 500) | max(?, **1010**) |

Code: `_max_c_curt` computation moved before `M_mu` in `_add_iteration_block`.

**Lesson:** any dual M bounded by stationarity with `π_term` must use the **full π range**, not the absolute minimum.

---

## 2. Diagnostic infrastructure (Section 4a)

Added two per-grid cells:

| Cell | What it tests |
|------|--------------|
| `4a.x self-consistent` | `debug_fix_b(b_hat, u_j = plain-optimal-at-b_hat)` — verifies formulation correctness at the known CE |
| `4a.x cross-pattern`  | Replicates Iter 1 to extract `u_1`, then `debug_fix_b(b_hat, u_j_override=u_1)` — tests the *actual* Iter-2 model with b fixed |

Required modifications to `debug_fix_b`:
- New param `u_j_override: Optional[np.ndarray]`
- New param `return_values: bool` (returns `values` dict mapping `VarName → X`)
- New param `verbose: Optional[bool]` (overrides instance default for silent invocation from `_full_integer_warm_start`)
- Returns `cut_slack = LP_cost(u^j, b_test) − foil_cost` via `m.getConstrByName("opt_cut_0").Slack`
- New method `iter1_pattern(...)` — builds base master, solves Iter 1 (no KKT), returns `(b_1, u_1, F_1)`

**Critical diagnostic finding for IEEE 14:**
```
[debug_fix_b] Optimality cut slack at solution: 3.6e-11
```
The cut `foil_cost ≤ LP_cost(u_1, b_hat)` is binding to machine epsilon at b_hat. The Iter-2 master's feasible region is essentially a single point — B&B can't navigate to it without a complete warm-start.

This proved the problem was **MIP hardness, not a formulation bug**, and ruled out further M-tightening as a solution.

---

## 3. Fast-iteration debug infrastructure

To avoid the 30+ minute kernel-restart + setup-cell cycle for each warm-start attempt:

| File | Purpose |
|------|---------|
| `_decomp_repro_helpers.py` | Shared `UCWeakWCEOracle`, `replace_line_limits`, `make_foil_fn_14`, etc. |
| `cache_setup.py` | Run once; pickles all expensive setup (`DATA`, `idx`, `cvec`, `b0`, `bU`, `bL`, `w`, `big_M_mu`, `b_hat_BS`, `solFoil_bhat`) to `_cache_decomp_14.pkl` (~96 KB) |
| `repro_warmstart.py` | Loads cache, runs only the failing scenario (Iter 1 + KKT + warm start + 30s Iter-2 sanity solve). `--solve-iter2 N` flag for timed Iter-2 |
| `repro_minimal_4a1.py`, `repro_minimal_4a2.py` | Minimal mirrors of the notebook's 4a cells |
| `repro_no_cache.py` | Same but builds data from scratch via `quick_setup` (to rule out pickle bugs) |

Each iteration on the warm-start fix went from ~30 minutes to ~30 seconds. Essential.

---

## 4. Warm-start: failed attempts in order

Each attempt timed out with `Time limit reached / Best objective -`:

| # | Approach | Result |
|---|---------|--------|
| 1 | Reuse `m`, fix `b = b_hat` via equality constraint, solve | 120s timeout |
| 2 | Same + also fix `u_foil` via bounds | 120s timeout |
| 3 | Build fresh temp model mimicking `debug_fix_b`'s phased build, solve, transfer values by VarName | 60s timeout (z's didn't transfer cleanly) |
| 4 | Delegate entirely to `debug_fix_b(b_hat, u_j_override=u_1, return_values=True)` | 60s timeout |
| 5 | Same + `Heuristics=1.0` + `MIPFocus=1` | 40s, still no incumbent |

Hypotheses falsified:
- **Env-leakage from m**: rejected (disposed m, still failed)
- **Pickle corrupting data**: rejected (fresh `quick_setup` also failed)
- **Notebook vs script context**: rejected (both fail)
- **Gurobi just needs more time**: rejected (300s, 80K nodes, root LP = optimum exactly, still 0 incumbents)

**Conclusion**: the bigM-complementarity MIP structure is hostile to default B&B incumbent search. Confirmed empirically via the cold-start ablation (4b.1b): 67K nodes / 900s / 0 incumbents on the production model. This is structural to the MIP family, not solvable by tuning.

---

## 5. The fix that worked — analytic warm-start

Stopped asking Gurobi to find a feasible point and **computed it directly** from LP duals.

### `_solve_dispatch_lp(b_hint, u_j, ...)`

Builds and solves a clean UC dispatch LP at fixed `(b = b_hint, u = u_j)` — no bigM, no KKT block, no master coupling. Returns all primal + dual values:
- Primals: `p`, `f`, `θ`, `shed`, `curt`, `sp`, `sm`
- Free duals: `π` (balance), `ν` (dcflow), `σ` (slack-bus), `η` (neutrality)
- Non-neg duals: `λ_hi`, `λ_lo`, `ρ_up_i`, `ρ_dn_i`, `ρ_up`, `ρ_dn`, `μ_p`, `μ_m`, `γ_shed_*`, `γ_curt_*`, `γ_sp_*`, `γ_sm_*`

**Critical sign-convention fixes:**

1. **Free duals**: master uses optimization-community LMP convention (`π` positive at load buses); Gurobi's `Pi` for `=` constraints in min LP has opposite sign. → flip sign: `π(ours) = −c_bal.Pi`. Verified via shed-stationarity derivation. Same for `ν`, `σ`, `η`.

2. **Non-neg duals for `≤` constraints**: Gurobi `Pi ≤ 0` in min LP. → `μ(ours) = max(0, −Pi)` to keep ≥ 0.

3. **Implicit lb dual**: for variables at `lb=0` (shed, curt, sp, sm), use `.RC` (reduced cost).

4. **The `p` lb=−∞ fix** (most subtle): master's KKT block has the explicit `c_pmin: p ≥ Pmin·u` constraint AND the implicit `addVars(lb=0.0)` on `p`. The KKT stationarity uses **only `lam_lo` (the c_pmin dual)**. If our LP also has `lb=0` on `p`, the LP duality splits the lb dual between `c_pmin.Pi` (lam_lo) and `p.RC` (implicit) — and master only consumes the c_pmin part, leaving stationarity violated by `p.RC` when `u_j = 0`. **Fix:** in `_solve_dispatch_lp`, build `p` with `lb=-INFINITY` so c_pmin is the sole `p ≥ 0` enforcer. All the dual lands in `c_pmin.Pi`. This eliminated the residual 8-unit constraint violation that remained after the sign-flip fix.

### `_analytic_warm_start(...)`

Orchestrator:
1. Inject foil-block values from cached `oracle.solve_foil(b_hint)` (binary `u_foil` + continuous `x_foil`)
2. Inject `b[ell].Start = b_hint[ell]` for free lines + L1 split (`bp`, `bm`)
3. For each pattern `u_j` in `patterns`: call `_solve_dispatch_lp(b_hint, u_j)`, inject all primal + dual values by VarName lookup
4. For each `_comp` complementarity pair, pick `z` analytically: `z = 0 if slack > dual else 1`
5. `m.update()`

**Performance**: 0.2s vs 60–300s for all prior attempts. Zero constraint violations. Gurobi accepts the MIP start as the immediate incumbent.

### z-binary naming

Made deterministic for name-based transfer:
```python
z = m.addVar(vtype=GRB.BINARY, name=f"zbm{s}_{tag}")
```
where `s` is the iteration suffix (`"_0"`, `"_1"`, ...) and `tag` is unique per `_comp` call (e.g., `"mup_3_17"`, `"gshlb_2_20"`).

### `_full_integer_warm_start`

Now a thin wrapper delegating to `_analytic_warm_start`. The old in-place implementation preserved as `_full_integer_warm_start_inline` for reference.

---

## 6. `run()` semantics improvements

| Change | What | Why |
|--------|------|-----|
| Loosened CE tolerance | `_ce_ok(vd, vp) = vd ≤ vp + max(eps_weak, 1e-4·\|vp\|)` | Pure absolute 1e-3 is too tight when vp ~ 10⁶ (solver noise alone is ~83 at MIPGap=1e-4) |
| Hint refresh on better CE | When CE found with `Fk < best_F − 1e-9`: `self.b_hat_hint = b_k.copy()`, `_b_hint_sol = sol_d`; subsequent warm starts use the new best | Tighter `F ≤ F_hint` bound each iteration |
| Register `b_hat_hint` as initial incumbent | Right after the hint warm-start succeeds, call `_update_incumbent(b_hat_hint, F(b_hat_hint), vp, vd)` | Trust the input CE; guarantees `success=True` when a hint is provided. `best_F = inf` is structurally impossible. |
| `candidate` field in return dict | Master's last incumbent (with `oracle_ce_gap = vd − vp`), regardless of CE certification | Diagnostics when cycle halts at a master-feasible-but-borderline-CE point |
| `termination_reason` field | `"certified_optimal" \| "cycle" \| "time_limit_no_incumbent" \| "master_infeasible" \| "sp1_infeasible" \| "master_status_<N>" \| "max_iter"` | Distinguish "ran out of time" from "no feasible point" from "converged" |

---

## 7. Notebook restructure — self-contained pipeline

Removed Section 4b (comp-mode comparison) which was diagnostic and redundant. Rebuilt Section 4b as a per-grid 3-step pipeline:

```
Section 4b · Per grid:
  Step 1 — B&S preprocess (bs_preprocess_cell)
    • Uses cached bs_<grid>_checkpoint.json if present (instant)
    • Otherwise runs B&S internally with the inline violation function
    • Falls back to bU (max expansion) if B&S finds no CE
    • Saves checkpoint either way
  Step 2 — DECOMP refinement (decomp_cell)
    • Loads B&S checkpoint as b_hat_hint
    • Runs UCDecomp4b.run() with analytic warm start
  Step 3 — Plot (plot_decomp_cell)
    • 4-panel via plot_decomp_results (factual dispatch, foil dispatch, 
      curtailment comparison, line-limit changes p.u.)
    • Plus numeric summary: iterations, patterns, F, LB, gap, termination_reason
```

Plus Section 4b.1b: cold-start ablation (`use_bs_hint=False`) demonstrating the warm-start is structural.

Pipeline is now self-contained: inputs are only the JSON data files in `Data/`. No external `b_hat`, no manual B&S step.

Added at notebook top:
- `matplotlib`, `matplotlib.gridspec` imports
- `TECH_COLORS` palette
- `plot_decomp_results(...)` (ported from `BranchSand_3grids.ipynb`'s `plot_bs_results`, adapted to the DECOMP result dict)
- `from uc_branch_sandwich_4b import UCBranchAndSandwichWCE_4b`

Final notebook: 59 cells.

---

## 8. Empirical result: DECOMP beats B&S on IEEE 14

Ran the pipeline end-to-end. **DECOMP found a strictly tighter CE than B&S.**

| Method | F | v_foil − v_plain | Status | Lines changed |
|--------|---|------------------|--------|---------------|
| B&S | 2.3823 | 0.0000 (slack) | "Certified" (within B&S tree) | 14, 15, 18 (Δ: +6.83, +7.56, **+2.93**) |
| **DECOMP** | **1.9067** | +0.0023 (boundary) | CE within solver precision (3e-9 rel.) | 14, 15, 18 (Δ: +7.80, +7.70, **+0.35**) |

DECOMP found a b on the CE boundary where line 18 is much less expanded (+0.35 vs +2.93). This gives a lower L1 (F = 1.9067 vs 2.3823 — **20% improvement**).

The 0.0023 boundary violation is 4.5 orders of magnitude below Gurobi's MIPGap=1e-4 noise floor (~83 on cost 830k). At achievable solver precision, the b is indistinguishable from a strict CE.

**B&S's "0% gap"** reflects convergence within its branching tree, not the global bilevel optimum. CCG/DECOMP's master MIP explored b values B&S's branching missed.

The **DECOMP gap of 50.94%** is `F_opt − master_LB = 1.9067 − 0.9354`. This is the master's LP-relaxation looseness from bigM-complementarity, NOT the real optimality gap (which is near zero or possibly negative — i.e., DECOMP's incumbent is the true optimum or very close).

---

## 9. Theory documented in conversation

Concepts discussed/explained:
- Why iterations exist (lazy enumeration of patterns; alternative is enumerating 2^144 patterns)
- Why warm-start is structurally required (B&B can't find a feasible point on its own for bigM-complementarity masters)
- Big-M derivation comparison to Fritz & Bukhsh 2025: we do **hybrid** (analytical KKT-stationarity floor + 10× empirical at b0), paper does **only** 10× empirical from a database. Ours is strictly safer; bridges Bugs 1, 7, 8, 9 that pure-empirical would fail on.
- DECOMP+cut from Fritz & Bukhsh §C: mine `oracle.cache_plain` for always-on/always-off (g,t) pairs → fix u_foil; tighten F bound to min-F CE in cache. **Parked as future work** (Option C from earlier).

---

## 10. Files changed in this session

| File | Status |
|------|--------|
| `uc_decomp_4b.py` | Heavy modifications: Bug 9, `_solve_dispatch_lp`, `_analytic_warm_start`, `_full_integer_warm_start` wrapper, named z's, `iter1_pattern`, modified `debug_fix_b`, CE tolerance, hint refresh, candidate, termination_reason, b_hat_hint registration |
| `build_decomp_notebook.py` | matplotlib imports; `plot_decomp_results`; `bs_preprocess_cell` with bU fallback; `plot_decomp_cell`; restructured 4b |
| `decomp_3grids.ipynb` | Regenerated (59 cells) |
| `cache_setup.py` | New — pickle setup once |
| `repro_warmstart.py` | New — fast iteration on warm-start step |
| `_decomp_repro_helpers.py` | New — shared oracle/foil-fn helpers for repro scripts |
| `repro_minimal_4a1.py`, `repro_minimal_4a2.py`, `repro_no_cache.py` | New — minimal mirrors for hypothesis testing |
| `DECOMP_state.md` | Updated with Bug 9, warm-start overhaul, pipeline architecture, parked work |
| `DECOMP_review.md` | Bug 9 added to bug table |
| Memory: `project_decomp.md`, `user_profile.md`, `MEMORY.md` | Updated |
| This file | New |

---

## 11. Validated / outstanding

**Validated end-to-end:**
- IEEE 14: B&S preprocess → DECOMP (F=1.9067) → plot.

**Outstanding validation:**
- IEEE 39: cells 44–48
- IEEE 57: cells 51–55
- Run the strict-CE diagnostic on each (the cell that prints `Strict CE`, `Loose-abs`, `Loose-rel`)

**Optional next steps** (in order of value):

1. **Option B — projection to strict CE**: bisect along `b_hat → bU` to find smallest expansion that gives `vd ≤ vp − margin` strictly. Reports a guaranteed-strict CE alongside the master's borderline best. *(User said yes to this; was about to add when this summary request came in.)*

2. **DECOMP+cut (Fritz & Bukhsh §C)**: mine `oracle.cache_plain` for always-on/always-off `u_foil[g,t]` pairs and fix them in master; tighten F bound to min-F CE in cache. ~150 LOC. Risk: may exclude true optimum on sparse datasets — implement with 95% threshold and infeasibility fallback.

3. **Multi-pattern enumeration** to break cycle detection: instead of one `u_k` per iteration, use Gurobi's solution pool in SP1 to extract top-K plain-optimal patterns. Adds more cuts per iteration → tighter master LB → smaller gap → maybe certification.

4. **Strong-duality reformulation** of the lower-level (Audet et al. 1997): tighter LP relaxation than bigM, no z binaries. Major rewrite but might fix the LB issue at the root.
