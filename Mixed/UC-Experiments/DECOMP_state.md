# DECOMP — Current Debug State
*Last updated: 2026-05-23 (session 2 — state verified, IEEE 57 B&S result confirmed)*

## How to use this file

Drop into a new Claude Code session and say: **"Read DECOMP_state.md and continue."**

Also read: `DECOMP_summary.md` (algorithm design) and `build_decomp_notebook.py` (notebook generator).

---

## Goal

Implement **DECOMP** (CCG, Yue et al. 2019) for b-parameter counterfactual explanations in UC. Finds minimum change to line flow limits `b[ell]` (fmax) so a 10% emissions-reduction foil becomes UC-optimal.

Key files:
- `uc_decomp_4b.py` — `UCDecomp4b` class (main implementation)
- `build_decomp_notebook.py` — generates `decomp_3grids.ipynb`
- `decomp_3grids.ipynb` — experiments on ieee14, ieee39, ieee57

Grids + foil: same as `bs_7grids.ipynb` — IEEE 14/39/57 with ALPHA=0.10 emissions reduction.

B&S reference results (from `bs_14/39/57_checkpoint.json`):
- IEEE 14: F_opt=2.3823 (certified), best_b known
- IEEE 39: F_opt=0.9016 (NOT certified, LB=0.3713)
- IEEE 57: result unknown

---

## Code state (uc_decomp_4b.py)

### Constructor signature (complete)
```python
def __init__(
    self,
    oracle,
    data: NetworkUCData,
    idx: IndexMap,
    cvec: np.ndarray,
    foil_extra_constr_fn: Callable,
    b0: np.ndarray,
    b_bounds: Tuple[np.ndarray, np.ndarray],
    b_free_idx: List[int],
    big_M_mu: float,
    big_M_mu_others: Optional[Dict[str, float]] = None,
    eps_weak: float = 1e-3,
    eps_obj: float = 1e-3,
    max_iter: int = 15,
    output_flag: int = 0,
    verbose: bool = True,
    w: Optional[np.ndarray] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 1,
    keepalive_interval: float = 1200.0,
    big_M_multiplier: float = 1.0,
    master_time_limit: Optional[float] = None,
    master_output_flag: int = 0,
    master_mip_gap: float = 1e-4,
    comp_mode: str = "sos1",   # "bigM" | "indicator" | "sos1" | "hybrid"
    b_hat_hint: Optional[np.ndarray] = None,  # known-good CE from B&S
)
```

### Key methods
- `run(window_size, per_bus_neutrality, u_init, p_init, on_time_init, off_time_init) -> Dict`
- `debug_fix_b(b_test, ...)` — fixes b, adds one KKT block, checks feasibility
- `_build_master_base(...)` — foil UC (binary u_foil) + shared b vars + L1 objective
- `_add_iteration_block(...)` — adds primal LP + full KKT block for one u^j pattern
- `_inject_mip_start(m, master_vars, b_ws, sol_dict, u_init)` — sets Gurobi Start from oracle solution (foil vars + b vars only)
- `_complete_warm_start(m, master_vars, b_hint, foil_sol, u_init)` — solves LP at b_hint and injects ALL variable values as .Start (raw LP values, no SOS1 post-processing)
- `_comp(dual_var, slack_expr, M_d, M_s)` — complementarity closure (inside `_add_iteration_block`)

### comp_mode implementations
- `"bigM"`: `dual ≤ M_d*(1-z)`, `slack ≤ M_s*z` — binary per pair, LP bound stays meaningful ← **current production mode**
- `"indicator"`: `z=1 → dual=0`, `z=0 → slack=0` — binary per pair, M-free, enforced via B&B branching (does NOT loosen LP). Same z-convention as `"bigM"` so `_analytic_warm_start` works for either mode.
- `"sos1"`: `SOS1({dual, slack})` — no binary, no M ← **DOES NOT WORK** (see Bug 5)
- `"hybrid"`: bigM for pairs with bilinear slack (free-line flow limits, where `slack = b[ell] ± f^j` involves master variable `b[ell]`); indicator for everything else (constant-RHS pairs). See `DECOMP_lb_stagnation.md` — addresses CCG LB stagnation by tightening the LP relaxation on non-bilinear KKT pairs.

### run() flow (current)
```
1. oracle.solve_plain/foil(b0)  → warm-start incumbent if CE found
2. _build_master_base(...)       → m, master_vars
3. oracle.solve_foil(bU)         → inject MIP start via _inject_mip_start (b at max expansion)
4. oracle.solve_foil(b_hat_hint) → inject MIP start via _inject_mip_start (B&S CE hint)
5. Add F ≤ F_hint constraint     → explicit objective UB, gives Gurobi pruning power
6. For k in range(max_iter):
   a. Solve master MIP → b_k, LB
   b. If INFEASIBLE: write IIS, break
   c. If TIME_LIMIT and SolCount==0: break
   d. SP1: oracle.solve_plain(b_k) → u_k
   e. SP2: oracle.solve_foil(b_k)
   f. CE check: if foil ≤ plain + eps → update incumbent; tighten F ≤ F_hint RHS
   g. Convergence: if best_F - LB < eps_obj → certified, break
   h. Cycle: if pattern(u_k) in seen → break
   i. _add_iteration_block(u_k)
   j. Re-inject hint warm start: _inject_mip_start + _complete_warm_start
   k. LP relaxation check (if verbose): m.relax() → write IIS if infeasible
   l. Checkpoint
```

### master_vars dict
```python
master_vars = {
    "var": var,          # foil UC variable dict (u, v, w, p, f, theta, shed, curt, splus, sminus)
    "b": b_vars,         # b[ell] master variables
    "bp": bp,            # bp[ell] = max(0, b-b0) for L1
    "bm": bm,            # bm[ell] = max(0, b0-b) for L1
    "sos_pairs": [],     # list of (dual_var, slack_var) tuples for SOS1 tracking
}
```

---

## Debug history

### Bug 1: M_shed = VOLL (FIXED)
`M_shed = 2.0 * VOLL * _mf` — was VOLL, caused infeasibility for all grids.

### Bug 2: ALPHA mismatch (FIXED)
DECOMP used ALPHA=0.20, B&S uses ALPHA=0.10. Fixed in `build_decomp_notebook.py`.

### Bug 3: IEEE 14 missing foil_no_shed (FIXED)
B&S wraps the foil with a no-shedding constraint for IEEE 14.

### Bug 4: IEEE 57 warm-start mismatch (FIXED)
`on_t_57[0]` set to 1 instead of `DATA_57.gens[0].UT`.

### Bug 5: SOS1 LP relaxation collapses (CONFIRMED, SWITCHED TO bigM)

**Symptom**: All runs stop at Iter 2 with 0 feasible solutions (300s timeout, 86k+ nodes).

**Root cause**: After adding KKT block 0 with `comp_mode="sos1"`, the LP relaxation drops all SOS1 constraints entirely. Both dual and slack can be simultaneously nonzero, making b=b0 (F=0) trivially LP-feasible. The LP bound collapses from **0.9463 → 0.0000**, giving Gurobi zero pruning power. B&B is blind.

```
Observed in output:
[LP-relax] feasible  obj=0.0000   ← should be ~0.9 for IEEE 14
```

**Fix**: Section 4c now uses `comp_mode="bigM"`. The big-M constraints `dual ≤ M*(1-z)` and `slack ≤ M*z` remain active in the LP relaxation, giving a tight bound.

### Bug 7: M_curt too small — master integer-infeasible (FIXED)

**Symptom**: Iter 2 explores 101K nodes in 900s with 0 feasible solutions. Also "User MIP start violates R10319 by 140."

**Root cause**: `M_curt = max(c_curt_max, VOLL) = 500`. From KKT stationarity `gcu_lb = c_curt − pi + gcu_ub`. When shedding is active at the same bus (pi = −VOLL = −500), `gcu_lb = c_curt + VOLL = 505 > 500 = M_curt`. The bigM constraint `gcu_lb ≤ M_curt*(1−z)` then allows max gcu_lb = 500 for z=0 and 0 for z=1 — neither permits the required 505. Every integer assignment of z is infeasible → master is integer-infeasible for all b values. Same root cause as Bug 1 (M_shed was VOLL, needed 2×VOLL = VOLL + VOLL).

**Fix**: `M_curt = (max_c_curt + VOLL) * mf` (sum, not max). For IEEE systems with c_curt ≈ 5: M_curt = 505.

**Second fix** (warm start): `_complete_warm_start` was injecting fractional LP z values (e.g. z=0.972) as `.Start`. Gurobi rounds to z=1, making `rho_up ≤ 0` violated by 140 → warm start rejected. Fix: skip fractional binary Start values (leave at `GRB.UNDEFINED`); only inject near-integer (< 0.01 or > 0.99) binary values.

### Issue 10: B&B cannot navigate near-binding optimality cut (FIXED 2026-05-23 via full integer warm start)

**Symptom (post Bug 9 fix)**: 4b.1 (was 4c.1) still fails — Iter 2 explored 1.2M nodes in 1h with indicator mode (same as bigM), Solution count 0.

**Root cause (diagnostic 4a.2 cross-pattern)**: At b = b_hat, the optimality cut `foil_cost ≤ LP_cost(u_1, b_hat)` is **binding to ~1e-11** (machine epsilon). The feasible region of the Iter-2+ master is essentially a single point (b_hat itself). Default B&B branching on 4080 binaries cannot land on this needle in `[bL, bU]^|free|`.

**Diagnostic output (4a.2, IEEE 14)**:
```
[iter1_pattern] b_1: F=0.9463
[iter1_pattern] v_plain(b_1)=869826.5200  sum(u_1)=95
[debug_fix_b] Fixed-b + KKT block: status=2  obj=2.3823
[debug_fix_b] Optimality cut slack at solution: 0.0000  (cut nearly binding)
[4a-cross] cut_slack=3.6e-11  → b_hat is a corner of feasible region
```

**Fix**: New method `_full_integer_warm_start` (uc_decomp_4b.py). After each KKT block addition, temporarily fix b = b_hint, solve the full MIP (fast since b is fixed → no b-search), extract ALL variable values (continuous AND binary), inject as `.Start`. Gurobi gets a complete integer-feasible incumbent at F = F(b_hint) on entering Iter 2+. Replaces the `_complete_warm_start` call inside `run()`. The old method is kept for reference but no longer called.

### Bug 9: M_mu floor too small — same error pattern as Bug 1 (FIXED 2026-05-23)

**Symptom**: Iter 2 finds 0 solutions despite LP bound = 0.8648 (bigM working). After Bugs 7+8 fixes, the master is still not finding feasible integer solutions.

**Root cause**: From LP stationarity: `μ_p = π_fr − π_to − ν + μ_m`. In the simplified (radial) case, `μ_p ≤ π_max − π_min = (2·c_curt + VOLL) − (−VOLL) = 2·VOLL + 2·c_curt`. This is the same bound as M_shed. The code used `VOLL` as the floor for M_mu — i.e., the absolute value of `π_min`, NOT the full π range. Same arithmetic error as Bug 1.

- IEEE 14: M_mu was `max(4998, VOLL=20000) = 20000`, bound is `40010` → 2× too small
- IEEE 57: M_mu was `max(big_M_mu, VOLL=500) ≤ ~1000`, bound is `1010` → potentially too small
- IEEE 39: same as IEEE 14

If the actual flow dual at any `b ∈ [bL, bU]` exceeds M_mu, both z=0 and z=1 become infeasible for that complementarity → master integer-infeasible at that b value.

**Fix**: `M_mu = max(self.big_M_mu, 2.0*VOLL + 2.0*_max_c_curt) * _mf`. Also moved `_max_c_curt` computation before M_mu so it's available (previously it was computed after M_mu).

### Bug 6: _complete_warm_start violated R8031 (FIXED)

**Symptom**: "User MIP start violates constraint R8031 by 181.47"

**Root cause**: `_complete_warm_start` was post-processing SOS1 pairs by zeroing the smaller member. For flow constraints, `s_aux` is linked by `m.addConstr(s_aux == b[ell] - f_j[ell,t])`. Zeroing `s_aux` while keeping b and f_j at LP values violated this equality by exactly the LP value of `s_aux`.

**Fix**: Removed SOS1 post-processing from `_complete_warm_start`. Raw LP values are injected directly. Gurobi warns about SOS1 violations but those are harmless — it still uses the LP values for node LP warm-starting. Linear constraints are satisfied.

---

## Current state

### What was just run
Section 4c.1 (IEEE 14, sos1, 300s) → confirmed Bug 5. Output:
```
[DECOMP] Added hint upper bound: F ≤ 2.3823
[WS] LP at b_hint feasible (obj=2.3823) — full warm start injected (11576 vars)
[LP-relax] feasible  obj=0.0000
[DECOMP] Iter 2/15 — Time limit hit with no feasible master solution — stopping.
IEEE 14-bus | comp=sos1 success=False F_opt=inf iters=1 cert=False
```

### What was run (2026-05-23 session 2) — FAILED
Cell 37 (IEEE 14, bigM, 900s): iter 1 OK (LP bound=0.9463), iter 2 → 900s timeout, 0 solutions.
- LP relaxation root bound at iter 2: **0.8648** (bigM IS working for LP, not a Bug 5 repeat)
- "User MIP start violates R10319 by 140" — warm start rejected
- Root cause: Bug 7 (M_curt too small, see above)

### Session 5+ (2026-05-23 → 2026-05-24): warm-start architecture overhaul

After Bug 9 (M_mu floor), the per-iteration master MIP was still infeasible to *find an incumbent* for, even though the LP root bound was at the optimum. Documented this via cell 41 (cold-start ablation): 67K B&B nodes, 900s, zero incumbents — proving the bigM-complementarity structure is hostile to Gurobi's default heuristics.

Architectural changes added to `uc_decomp_4b.py` to address this:

1. **`_solve_dispatch_lp`** — standalone UC dispatch LP solver at fixed (b, u_j). Returns all primal + dual values for the KKT block, with correct sign conventions (free duals π/ν/σ/η: flip sign from Gurobi's `Pi`; non-neg duals: `max(0, -Pi)` for ≤ constraints, `RC` for implicit lb=0). Critical fix: `p` variable uses `lb=-INFINITY` (not 0) so all the lb dual goes into `c_pmin.Pi` (matches master's KKT structure which has no separate implicit-lb dual for `p`).
2. **`_analytic_warm_start`** — orchestrates `_solve_dispatch_lp` per pattern, derives z's analytically (`z=0 if slack < dual else 1`), injects every variable by name into the main master `m` as `.Start`. Runs in ~0.2s for IEEE 14. Replaces all prior attempts (in-place fix, temp-model, debug_fix_b delegation — all failed for Gurobi-state reasons).
3. **`_full_integer_warm_start`** — now a thin wrapper delegating to `_analytic_warm_start`.
4. **z binaries named deterministically** (`zbm{j}_{tag}`) so name-based transfer works across temp/main models.
5. **Hint refresh on better CE**: when iter k finds a CE with F < best_F, `self.b_hat_hint` and `_b_hint_sol` update to (b_k, foil_sol_k). Subsequent warm-starts use the tighter incumbent.
6. **CE-tolerance loosened**: `_ce_ok(vd, vp)` uses `max(eps_weak, 1e-4 * |vp|)` (relative + absolute) instead of pure absolute 1e-3 — required when vp ~ 10^6.
7. **b_hat_hint registered as initial incumbent**: if a hint is provided AND its foil oracle returns a solution, `_update_incumbent` is called immediately with `(b_hat_hint, F(b_hat_hint), ...)`. Guarantees `best_F ≤ F(b_hat_hint)` at termination — `success=False` is now structurally impossible when a hint is provided.
8. **`candidate` field in return dict**: master's last incumbent (with `oracle_ce_gap`) tracked separately from certified result. Useful when cycle-detection halts at a master-feasible-but-oracle-not-CE point.
9. **`termination_reason` field**: `"certified_optimal" | "cycle" | "time_limit_no_incumbent" | "master_infeasible" | "sp1_infeasible" | "master_status_<N>" | "max_iter"`.

### Pipeline architecture (current)

`decomp_3grids.ipynb` is now self-contained (59 cells). Per grid, Section 4b runs:
- **Step 1**: B&S preprocessing (`bs_preprocess_cell`) — runs B&S if no `bs_<grid>_checkpoint.json`; otherwise no-op. Falls back to `bU` (max expansion, always foil-feasible) if B&S finds no CE.
- **Step 2**: DECOMP refinement (`decomp_cell`) — warm-started by the B&S CE.
- **Step 3**: Plot (`plot_decomp_cell`) — 4-panel summary via `plot_decomp_results` (ported from BranchSand_3grids.ipynb).

Plus 4b.1b: cold-start ablation (no warm-start, no F upper bound, no analytic warm-start) demonstrating the structural requirement.

### Future work (parked, not yet implemented)

**LB stagnation — Fix 1 IMPLEMENTED 2026-05-25**: `comp_mode="hybrid"` added to `uc_decomp_4b.py`. Routes constant-RHS complementarity pairs (gen bounds, ramps, shed, curt, shift) through `addGenConstrIndicator` while keeping bigM for free-line flow pairs (where `slack = b[ell] ± f^j` is bilinear). Indicator pairs are enforced by B&B branching, NOT linearised via big-M, so they don't loosen the LP relaxation. Also added per-iteration LB diagnostic in `run()` (verbose only): prints `ObjBound`, `ObjVal`, `MIPGap`, and a freshly solved `m.relax()` Root LP value so it's easy to see whether the LP bound is tracking new KKT blocks or stagnating at the root. **Next: validate on IEEE 14 by re-running 4b.1 with `comp_mode="hybrid"`** and confirming `ObjBound` grows across iterations.

**LB stagnation — Fix 2 (parked)**: McCormick linearisation of `w = b[ell] · μ_p^j[ell,t]` + strong-duality equality replacing the flow z-binaries (~100–150 lines). Pursue if Fix 1 doesn't close the gap on IEEE 14.

**LB stagnation — Fix 3 (parked)**: Yue et al. §4.2 projection reformulation (P4). Major rewrite. Only if Fix 2 isn't enough on IEEE 39/57.

**DECOMP+cut** (Fritz & Bukhsh 2025 §C, "Data-Driven Heuristic Cuts"):
- Mine `oracle.cache_plain` for (g,t) pairs where `u_plain[g,t]` is constant across all cached samples → fix `u_foil[g,t]` accordingly in the master
- Use min-F CE from the cache as a tighter `F ≤ F_data` bound (generalizes the current `F ≤ F_hint`)
- Risk acknowledged in paper: may exclude the true optimum if dataset coverage is sparse. Implement with 95% threshold (not 100% unanimity) + fallback to plain DECOMP on infeasibility.

### Bug 9: M_mu floor too small — same error pattern as Bug 1 (FIXED 2026-05-23)
`uc_decomp_4b.py`:
- Bugs 7, 8, 9: M_curt / M_shed / M_mu floors corrected (see history above)
- `debug_fix_b`: added `u_j_override` param + `cut_slack` reporting
- `iter1_pattern`: new method — replicates Iter 1 to extract (b_1, u_1, F_1)
- `_full_integer_warm_start`: new method — solves fixed-b MIP, injects complete integer-feasible Start (replaces `_complete_warm_start` call in `run()`)

`build_decomp_notebook.py`:
- Removed Section 4b (redundant comp-mode comparison)
- Section 4a now has 6 cells: self-consistent + cross-pattern tests for each grid (comp_mode="bigM")
- Section 4c renamed to Section 4b
- 40 cells total (down from 45)

### Next step
**Restart kernel → run Cells 0–16 → run Cell 32 (Section 4b.1, IEEE 14, 900s)**

Expected output with full integer warm start:
- `[WS-FULL] Fixed-b MIP solved (status=2, obj=2.3823) — injected N vars as Start`
- Iter 2 starts B&B with incumbent at F=2.3823 (immediately)
- Either certifies F=2.3823 as optimal (gap=0%) or finds smaller F
- Total wall time per iteration: tens of seconds to a few minutes (not 900s timeout)

If 4b.1 converges, run 4b.2 (IEEE 39) and 4b.3 (IEEE 57).

---

## Notebook structure (decomp_3grids.ipynb, 45 cells)

| Section | Cells | Description |
|---------|-------|-------------|
| 0 | 1–5 | Imports, helpers, oracle |
| 1 | 6–9 | Dataset loading (14/39/57) |
| 2 | 10–14 | Factual UC + emissions baseline |
| 3 | 15–16 | Big-M calibration |
| 4a | 17–23 | debug_fix_b with B&S checkpoint (feasibility test) |
| 4b | 24–34 | Comp mode comparison (diagnostic, skip) |
| 4c | 35–41 | **Full DECOMP (bigM, max_iter=15, 900s)** |
| 5 | 42–44 | Results summary + CE verification |

### To regenerate the notebook
```bash
python build_decomp_notebook.py
```

---

## Gurobi parameters in master solve (current)

```python
m.Params.MIPFocus     = 1       # find feasible solutions first
m.Params.NumericFocus = 2
m.Params.MIPGap       = 1e-4
m.Params.TimeLimit    = 900     # seconds (was 300, increased for bigM)
m.Params.OutputFlag   = 1       # visible for monitoring
```

---

## Model sizes (IEEE 14, after Iter 1 KKT block)

| Mode | Vars | Int vars | Constrs |
|------|------|----------|---------|
| sos1 | 11,576 | 432 | 11,951 + 3,648 SOS1 |
| bigM | ~11,576 + ~3,600 | 432 + ~3,600 | ~15,600 |

---

## Expected performance

| Grid | B&S F_opt | Expected DECOMP F_opt | B&S certified? |
|------|-----------|----------------------|----------------|
| IEEE 14 | 2.3823 | ≤ 2.3823 | Yes |
| IEEE 39 | 0.9016 | ≤ 0.9016 | No (LB=0.37) |
| IEEE 57 | 11.6796 | ≤ 11.6796 | No |

DECOMP is exact (certified when gap < eps_obj=1e-3). B&S is heuristic.
