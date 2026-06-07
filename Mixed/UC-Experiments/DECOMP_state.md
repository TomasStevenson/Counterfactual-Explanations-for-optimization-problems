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
    comp_mode: str = "sos1",   # "bigM" | "indicator" | "sos1" | "hybrid" | "strongdual"
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
- `"hybrid"`: bigM for pairs with bilinear slack (free-line flow limits, where `slack = b[ell] ± f^j` involves master variable `b[ell]`); indicator for everything else (constant-RHS pairs). See `DECOMP_lb_stagnation.md` Fix 1. **TESTED 2026-05-28 — did NOT close the gap** (IEEE 14: 50.94% bigM → 51.00% hybrid; IEEE 39/57 stay at LB=0). The binding looseness is the bilinear flow complementarity, which hybrid leaves as bigM.
- `"strongdual"`: Fix 2 — drops ALL complementarity binaries; enforces dispatch-LP optimality via one strong-duality equality `lp_var = dual_obj` + primal feas + stationarity. Bilinear dual term `−Σ b[ell]·(μ_p+μ_m)` (free lines) linearised with McCormick `w = b·μ`; `μ` capped at `M_mu`. Only integer vars left are `u_foil`. **CRITICAL — the equality is `lp_var == dual_obj`, NOT `lp_var + lp_const`.** `dual_obj` is the dual optimum of the *dispatch LP*, whose objective is the variable cost only (`p/shed/curt/sp/sm` = `lp_var`); `lp_const` is the commitment cost (`u,v,w`), a *constant* of the dispatch LP and absent from its objective. First implementation wrongly used `lp_var+lp_const`, which over-constrained the master by the (nonzero) commitment cost and **excluded valid CEs → invalid (too-high) LB / false certification** (IEEE 39 smoke falsely certified F=2.106 when B&S CE=0.714). Caught by `_check_strongdual_valid.py` (`debug_fix_b` at the known CE `b_BS` returned INFEASIBLE). Fixed 2026-05-28. Post-fix validation: `b_BS` feasible with obj=F(b_BS) on IEEE 14 & 39; IEEE 39 smoke LB=0.665 ≤ 0.714 (valid, tight). dual_obj signs verified to 1e-16 (`_verify_strongdual.py`).

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

**LB stagnation — Fix 1 IMPLEMENTED 2026-05-25, TESTED 2026-05-28 — INSUFFICIENT**: `comp_mode="hybrid"` (indicators for constant-RHS pairs, bigM for bilinear flow pairs). Also added per-iteration LB diagnostic in `run()` (verbose only): prints `ObjBound`, `ObjVal`, `MIPGap`, and a freshly solved `m.relax()` Root LP. Ran the full notebook 2026-05-28: hybrid did NOT close the gap (IEEE 14 50.94%→51.00%; IEEE 39/57 stay LB=0). Confirmed the doc's prediction — the binding looseness is the bilinear flow complementarity, which hybrid leaves as bigM. Note the diagnostic `m.relax()` reads 0 everywhere because it relaxes `u_foil` too, not just the `z`s — it conflates two integrality sources but the `ObjBound` trace is authoritative.

**LB stagnation — Fix 2 IMPLEMENTED + correctness-fixed 2026-05-28**: `comp_mode="strongdual"`. Strong-duality equality `lp_var = dual_obj` (NOT `lp_var+lp_const` — see comp_mode note above for why) + primal feas + stationarity + McCormick on flow bilinears. **Bug found and fixed same day** via `_check_strongdual_valid.py`: the first version used `lp_var+lp_const`, excluding valid CEs → invalid LB (IEEE 39 falsely certified 2.106 vs true ≤0.714). Post-fix: VALID + tight (IEEE 39 LB 0.665 vs F* 0.714 → ~7% gap with real hint). Warm start: `_analytic_warm_start` injects `w = b_hint·μ` for free lines; `set_z` calls are harmless no-ops (no z vars). **Lesson: ALWAYS run the `debug_fix_b`-at-known-CE feasibility check before trusting a new LB — a tighter LB is worthless (worse: dangerous) if it certifies above F\*.**

**Notebook 4d/4e results pending re-run with the FIX.** The 4d numbers the user saw on 2026-05-28 (IEEE14 gap 58.9%, IEEE39 100%, IEEE57 16%) were from the BUGGY version — IEEE 57's "16%" and the certifications are NOT trustworthy. Re-run 4d (strongdual) and 4e (strongdual+seed) after the fix.

**Technique 2 (seeding) IMPLEMENTED 2026-05-28**: `seed_patterns: bool = False` ctor arg. When True, `run()` pre-loads KKT blocks for plain-optima at {b0, b_hat, bU} before the loop (de-duped via `seen`), then warm-starts them. Front-loads the LB so it plateaus at iter 1 instead of cycle-halting after 1-2 discovered patterns. Notebook Section 4e uses it. Smoke (IEEE 39): reaches LB=0.665 at iter 1 (2 seeds) vs iter 2 (1 discovered) for no-seed — same plateau, fewer iters.

**LB stagnation — Fix 1 `comp_mode="hybrid"`**: see comp_mode note — tested, insufficient (leaves flow pairs bigM).

**Technique 3 / Option A (McCormick mu-box tightening) — IMPLEMENTED + TESTED 2026-05-28 — INEFFECTIVE**: ctor arg `mccormick_mu_factor` (None ⇒ provable M_mu; else McCormick mu-box = factor × max observed |μ| at {b0,b_hat,bU}, via `_estimate_mu_box`). Motivation: measured μ-box (40010) is 16-81× larger than observed flow duals (IEEE14 498, IEEE39 2511, IEEE57 123). Validated factor=3 keeps b_BS feasible (boxes 1493/7534/370). **BUT LB barely moved**: IEEE14 0.9355→0.9355 (0%), IEEE57 6.266→6.328 (+1%), same smoke harness. **Why**: at the LB-optimal master b (b expanded just enough to minimise F), the free-line flows are NOT binding ⇒ μ≈0 ⇒ McCormick terms inactive ⇒ box width irrelevant THERE. The looseness was measured where flows bind (b0/b_hat/bU), not at the LB point. Knob retained (default off, harmless); not a useful lever. **Real bottleneck = CCG stalling**: all grids cycle-halt after 1 KKT block whose cut sits at the base-master floor; LB ≈ base floor. Lesson: measure looseness AT the LB-binding solution, not at arbitrary b.

**STALL DIAGNOSTIC (`_diagnose_stall`, added 2026-05-28) — VERDICT: RELAXATION GAP (not missing pattern)**. Fires at the cycle-halt (verbose). Compares, at the optimistic master b_k, each pattern's master dispatch value (via `opt_cut_{j}.Slack` + master foil cost) against the TRUE dispatch cost (`_solve_dispatch_lp` + commitment). Result on IEEE 14 & 39: the master **inflates** the dispatch cost of the cycle pattern by EXACTLY the oracle CE gap (IEEE14 +44399 = vd−vp; IEEE39 +37718 = vd−vp). The opt_cut binds in the master (master_foil = master_disp) but true_disp < master_foil, so b_k passes a cut it should violate. By LP strong duality the only source is the McCormick approx of b·μ (w ≠ b·μ); tightening the μ-box didn't help because the envelope still has enough w-slack (free_lines × T terms) to absorb the needed inflation. **⇒ the McCormick relaxation itself is the LB ceiling. Pattern count is irrelevant (adding cuts can't fix an inflated-RHS cut).**

**Next lever (LB) = Fix 3 (projection, Yue §4.2)** — eliminates the b·μ bilinearity entirely (project out the dispatch LP) ⇒ exact LP relaxation, no McCormick gap. Lighter alternative to consider first: **piecewise-McCormick on b** (partition [bL,bU], few binaries) to shrink the bilinear envelope. Seeding (Technique 2), hybrid (Fix 1), and μ-box tightening (Option A) are all confirmed ineffective for the LB.

**Piecewise-McCormick IMPLEMENTED 2026-05-29 (`mccormick_segments: int = 1` ctor arg; plan `distributed-swinging-willow.md`)**: disaggregated piecewise McCormick on the shared `b[ell]` — global segment binaries `bdelta[ell,k]` (nFree×K total, tiny) + `bseg`, with per-segment μ/w disaggregation in the strongdual block (`_flow_w` helper). K=1 is byte-identical to the single-envelope code. Warm start sets the active segment of `b_hint`. **Validated CORRECT** (`_check_strongdual_valid.py`, b_BS feasible + obj=F(b_BS) at K=4,8 on IEEE 14 & 39). **Inflation falls ~1/K (the envelope tightens as designed) BUT the cycle-halt LB is NON-MONOTONE in K** (IEEE 14, smoke, B&S hint): K=1 LB 0.9355/infl 44399; K=4 0.9460/18478; K=8 **1.0355**/10097; K=16 **0.8542**/1791; K=32 0.8540/2005. LB peaks at K≈8 then DROPS below the K=1 value. Cause (either/both): (a) the bigger disaggregated master (μ/w grow `nFree×T×K×4`) doesn't converge in the per-iter budget at K≥16 → weaker ObjBound; (b) the CCG path shifts — K=16 halts at 2 iters vs 3 for K=8, so fewer cuts. Either way the cycle-halt is the real ceiling and piecewise does NOT reliably raise it. F_opt stuck at hint 2.3823 throughout. **VERDICT 2026-05-29: piecewise is valid + tightens the envelope but is NOT a reliable certification lever** — best (coincidental) at K≈8 (0.94→1.04 on IEEE14), useless/worse beyond, far short of F*≈1.9. Notebook Section 4f uses K=8. **Endgame for tight certification = Fix 3 (dual-vertex/Benders, plan Part 2, exact-at-integrality).** Also worth investigating: the **cycle-halt itself** (all modes halt after 1-2 patterns) may be the binding limit, independent of the relaxation. **Publishable now: corrected strongdual (4d) — VALID certificates, IEEE 39 ~6.8%.**

**Fix 3 = exact-bilinear NonConvex MIQCP (IMPLEMENTED 2026-05-29 / 2026-05-30 as `bilinear_exact: bool = False` ctor flag).** Drops McCormick entirely: writes the flow term in `dual_obj` as the true `b[ell]*mu_p[ell,t]` product (QuadExpr); the strong-duality equality becomes a non-convex quadratic constraint; `m.Params.NonConvex = 2` engages Gurobi's spatial branch-and-bound. EXACT and VALID: `_check_strongdual_valid.py 14 strongdual 1e4 none exact` confirms b_BS feasible with obj == F(b_BS). **But slow** — the IEEE 39 fixed-b feasibility check (`debug_fix_b`) did not complete in the smoke time window (spatial B&B on `b·μ` is intrinsically expensive on the larger grid). So `bilinear_exact` is a *proof-of-concept of rigorous certification at small scale* (IEEE 14), not a practical certification route for 39/57. Knob retained, default off.

**RETRACTED: dual-vertex / Benders projection (was Plan Part 2).** When designing the implementation I realized this approach **does not give a valid LB in our problem**. In two-stage stochastic LP / classical Benders the inner value `Q(x)` is in the *objective*, partial vertex set under-estimates Q → master under-estimates total cost → valid LB. Here the dispatch value `v(u^j,b)` is in a *constraint* `foil ≤ v`: partial vertex set under-estimates v → cut `foil ≤ v_partial` is **stronger** than `foil ≤ v_true` → over-restricts the master → master min goes UP → **invalid (too-high) LB**, the same hazard class as the `lp_const` bug. Even disjunctive vertex-selection has the same problem — picking the best from a partial set never reaches the true max. Valid-LB representations of `v` must over-approximate it (primal-feasible x cost ≥ v), which is exactly the strong-duality structure we already have; the only tightening lever is the bilinear `b·μ` representation (McCormick / piecewise / exact). Plan Part 2 (`distributed-swinging-willow.md`) is marked retracted; the rigorous Fix 3 is `bilinear_exact`.

## OBBT validity FIX + node-OBBT standardized solver (2026-06-05)

Two things this session: (1) found + fixed a **validity bug in iterated OBBT** that was
silently producing INVALID LBs on IEEE 39; (2) implemented **node-OBBT** — an explicit
spatial branch-and-bound over the shared b-box that is ONE solver/config for all three
grids (the standardization target) and is embarrassingly parallel for NLHPC.

### OBBT validity bug — iterated 2nd pass excluded the known CE (FIXED)
`_validate_all.py` reported **IEEE 39 INVALID** (the exact+OBBT master is INFEASIBLE at
b=b_BS → any LB it proves can certify ABOVE F*). IIS = all 480 OBBT-tightened μ UBs vs the
strong-duality dual at b_BS. Bisection: `obbt_iter=1` VALID, `obbt_iter=2` INVALID — the
**2nd OBBT pass** over-tightened. Root cause: **model-state staleness** — the original
iterate's pass 2 re-derived bounds over a not-properly-refreshed relaxation and applied the
SAME shrink again (logs showed pass1 and pass2 both "tightened 480, shrink 1.8289e7",
exactly double), pushing μ below the b_BS dual. 14/57 happened to survive but were at risk.
Fix (3 layers, user chose "both"):
1. **`obbt_iter` default 2 → 1** (conservative; the 2nd pass never adds value anyway).
2. **Corrected iterate** — per-pass `m.update()` + a pre-pass bound snapshot via
   `getVarByName` force the model state to refresh, so pass 2 now correctly sees the
   fixpoint and tightens **0** (verified) instead of double-tightening.
3. **Self-validating guard** in `_obbt_root`: precompute the hint CE's exact per-pattern
   dispatch duals once (`_solve_dispatch_lp`); after each pass, if any tightened μ UB <
   that dual (or b_hint leaves the b-box), **roll the pass back and stop iterating**. OBBT
   can now only ever stay conservative — safe for unattended HPC runs with no human gate.
Post-fix `_validate_all.py`: **IEEE 14 / 39 / 57 ALL VALID** (and `obbt_iter=2` now also
VALID — the guard + corrected iterate make the 2nd pass a harmless no-op). **NOTE: the
published pre-06-05 IEEE 39 "0.00%" numbers were produced with the buggy `obbt_iter=2` and
must be regarded as obtained under an unsound OBBT; the 06-05 re-run below re-certifies
IEEE 39 rigorously with the FIXED OBBT.**

### node-OBBT — spatial B&B over the b-box (`node_obbt: bool = False`)
Strategy 5 ("per-node OBBT"). Gurobi's callbacks CANNOT tighten node-local bounds inside
its spatial B&B, so we drive the tree ourselves and call Gurobi only to BOUND each box:
- New ctor args: `node_obbt`, `node_obbt_budget` (per-box TL), `node_obbt_max_nodes` (box
  cap), `node_obbt_tol`, `node_obbt_per_box` (per-box OBBT μ-sweep, default OFF). Single
  switch — auto-enables strongdual+bilinear_exact+obbt prerequisites.
- New method `_solve_master_spatial_obbt`: best-first spatial B&B. Per box: set b-box →
  (optional per-box OBBT) → solve the exact MIQCP with `node_obbt_budget` → update
  incumbent / prune / split the widest free b[ell] at its midpoint.
- **Validity**: boxes are an exhaustive partition ⇒ `global_LB = min over leaf boxes of
  each box's Gurobi ObjBound ≤ F*`, valid at ANY stop point. Best-first on the smallest
  box-LB is the engine that RAISES global_LB. Two guards in `run()`: take the driver's
  global_LB (NOT `max` with the last box's `m.ObjBound` — that single box can exceed the
  true min ⇒ invalid); use the driver's returned b_k (m.X reflects only the last box).
  m's b/μ bounds are snapshotted+restored so CCG iterations are unaffected.
- **`node_obbt_per_box` is OFF by default** because a single-dim b-split barely moves the
  μ-max (μ[ell',t] depends on b[ell'], not the split line) → per-box OBBT measured 0
  tightening at ~876 LPs/box of pure overhead. Root μ.UB is already valid for any sub-box
  (sub-box ⊆ root ⇒ root μ.UB ≥ max over sub-box), so per box we keep root μ and only
  tighten b — Gurobi's internal spatial relaxation of b·μ benefits directly from the
  tighter b-box.
- `_smoke_strongdual.py` slots 14-16 = `node` / `max_nodes` / `box_budget`;
  `_check_strongdual_valid.py` slot 7 = `obbt_iter`.

### Standardized 3-grid results (ONE config: strongdual+exact+obbt[iter1]+seed+seed_interp=3+node_obbt)
Only runtime budgets differ per grid (TL/box count scale with size); the ALGORITHM is identical.

| Grid | budget | valid global_LB | strict CE (UB) | gap% | term | mechanism |
|------|--------|----------------:|---------------:|-----:|------|-----------|
| **IEEE 39** | 1 box @300s | **0.7060** | **0.7060** | **0.00** | **certified** | certifies at box 1 (no split needed) |
| IEEE 57 | 4 boxes @300s | 9.7798 | 10.5578 | 7.37 | budget | box1 9.70 → box2 9.78 → box3 10.25; box4 PRUNED |
| IEEE 14 | 3 boxes @300s | 1.5756 | 2.3724 | 33.59 | budget | box1 1.559 → box2 1.576; box3 PRUNED |

- **IEEE 39 re-certified rigorously (0.00%) with the FIXED OBBT** — node-OBBT reduces to a
  single box where spatial branching isn't needed, then stops. "Doesn't hurt where not
  needed."
- **The lever works on 14 + 57**: a half-box proves a HIGHER bound than its parent at equal
  budget (box2 > box1 on both), and infeasible half-regions are PRUNED entirely — both
  spatial-B&B mechanisms fire. `global_LB` is pinned by the lowest unsplit leaf, so it
  rises monotonically as boxes are added.
- At small single-machine budgets the gap on 14/57 is still wide (33.6% / 7.37%) and 57 is
  behind the documented multistart 1.52% — **expected**: gap-closing scales with box count,
  and the boxes are INDEPENDENT (one Slurm-array job each, aggregate min-ObjBound
  externally). This is the NLHPC parallelization story, not a single-box-machine result.
- All VALID (b_BS feasibility gate green on all 3).

### Files modified (2026-06-05)
- `uc_decomp_4b.py`: `obbt_iter` default 1; OBBT self-guard (`_hint_ok`, per-pass snapshot
  + rollback) + corrected iterate; `sync_bounds_from_m` arg on `_obbt_root`; node-OBBT ctor
  args; `_solve_master_spatial_obbt`; `run()` node-mode branch with the two validity guards.
- `_smoke_strongdual.py`: node/max_nodes/box_budget slots + announce line.
- `_check_strongdual_valid.py`: `obbt_iter` slot (used to bisect the validity bug).

### Scaling study + parallel-box HPC port — RESULTS (2026-06-05)

**Theory note added:** `DECOMP_formulation_and_optimality.md` — formulation, the optimality
guarantee (LB≤F*≤UB invariants, subset-relaxation + strong-duality-direction lemmas, the
spatial-partition validity theorem), and why each upgrade is valid + faster.

**LB-vs-#boxes scaling (IEEE 14, node-OBBT, MFOC=3):**
| per-box budget | #boxes | global_LB | note |
|---|---|---|---|
| 120 s | 16 | 1.3484 | plateaus at box 7; 2 pruned |
| 300 s | 3 | 1.5756 | |
| (monolithic) | 1 | ~1.68 @1800 s | documented |
| F* (strict CE) | — | ~2.37 | target |

**Key finding: per-box budget DOMINATES box count.** `global_LB = min over leaf boxes`, so a
single undersolved box pins the floor — 16 boxes @120 s (1.35) is WORSE than 3 boxes @300 s
(1.58). Boxes only help once each is solved tightly enough to clear the floor. ⇒ certifying
IEEE 14 needs BOTH high per-box budget (≥300 s) AND many boxes ⇒ a large *parallel* budget
(the HPC case). IEEE 57 is closer (4 boxes @300 s → 7.37 %), so a modest array (8–16 boxes
@300 s) should bring it to ~1–2 % or certify.

**Parallel-box HPC port:** `node_obbt_hpc.py` (subcommands `emit` / `solve` / `aggregate`) +
`node_obbt.slurm` (Leftraru array template: partition `main`, gurobipy PYTHONPATH, token-server
license, `GRB_THREADS`). Each box = one independent Slurm task; `global_LB = min over boxes`.
**Validity-correct:** an INFEASIBLE box (no CE cheaper than F_hint ≥ F*) cannot hold the global
optimum → reported as `+∞` and **excluded** from the LB min (the run() placeholder
`master_LB=0` would otherwise crash the bound to 0); `global_UB` is floored at the known hint CE
`F(b_hat)` and oracle-re-verified. Tested end-to-end on IEEE 14 (4 boxes): `global_LB=1.4316`,
`global_UB=2.3823` (verified), gap 39.9 %, 3 boxes pruned.

**Port finding (partition quality is the lever, not validity):** a *uniform* `b`-split is
correct but weak here — the `F ≤ F_hint` cap concentrates all CEs into one sub-region, so most
boxes prune as infeasible and one box (undersolved) holds the CEs → no parallel speedup.
Mitigations: (a) split the **CE-expanded** lines `argmax|b_hat−b0|` (now the `emit` default,
`--dims 2`); (b) **v2 = adaptive-frontier emit** — run the in-process best-first driver a few
levels to generate a *good* box frontier (concentrated where CEs live), then distribute those.
The best-first split rule + snapshot/restore are already in place; v2 only needs to serialize
the driver's open `leaves` instead of a static grid.

### v2 adaptive-frontier emit + the cold-start / weakest-box finding (2026-06-06)

Implemented **`emit --adaptive`**: the in-process best-first driver carves a CE-concentrated
frontier (small `--emit-budget`), and its open `leaves` (now returned via `run()`’s
`node_boxes`) become the parallel boxes. Also fixed a real **missed-bound bug in `run()`**: a box
whose MIQCP times out with NO incumbent used to return `master_LB=0.0` (placeholder), discarding
the valid `ObjBound`; now it keeps `max(master_LB, ObjBound)` — essential for cold parallel boxes.

**IEEE 14, 4 adaptive boxes @180 s (post-fix):** box bounds `[1.0231, 1.4516, 2.2194, 2.2753]`
→ `global_LB = 1.0231`. **Two findings:**
1. The `ObjBound` fix works — cold boxes contribute REAL bounds; boxes 2 & 3 proved ≈ 2.22 / 2.28
   (**near `F*≈2.37`**) even without an incumbent. **Most of IEEE 14’s b-box certifies near
   optimal easily; only a small hard region is the bottleneck.**
2. **Static distribution is pinned by the WEAKEST box.** Adaptive `global_LB` (1.02) is *worse*
   than the uniform grid (1.43) because `global_LB = min over boxes` and box 0 solved weakly
   (cold: the hint CE lives in only one box; the rest solve cold, and the master is hostile to
   cold incumbent search). One-shot static distribution does NOT refine the weakest box, so it
   loses the best-first driver’s key advantage.

**⇒ The parallel design that actually closes the gap is ITERATIVE, not one-shot:**
distribute the frontier → solve → split the box pinning `global_LB` → redistribute (a coordinator
over Slurm rounds), and/or give each box a **per-box warm start** (carry the carve’s per-box
incumbent — the driver finds it — into the parallel re-solve). The current `node_obbt_hpc.py`
is the validity-correct building block (emit/solve/aggregate, min-ObjBound reduction); the
coordinator loop + per-box hints are the v3 efficiency layer.

**Boxes-to-certify estimate (revised, honest):** IEEE 14 is *mostly* easy (most boxes → ~2.2-2.3
near `F*`); certification is gated by a small hard sub-region that needs deep best-first
refinement (the in-process driver, or iterative parallel rounds) — NOT a uniform many-box sweep.
IEEE 57 (4-box adaptive gap 7.37 %) is closer and a modest iterative refinement should certify.

### v3 iterative best-first coordinator — IMPLEMENTED + VALIDATED (2026-06-06)

`node_obbt_coordinator.py` (init / solve-box / step / run-local / status) + `node_obbt_round.slurm`.
The parallel best-first refinement the one-shot static port lacked:

```
round r:  solve every OPEN box (independent Slurm-array task)  →  collect ObjBound + in-box CE
       →  prune (lb ≥ UB−tol)  →  SPLIT the split_k non-pruned leaves with the smallest lb
       →  their children are round r+1’s OPEN boxes.        (repeat until UB−LB ≤ tol)
```

- **Validity (every round):** `global_LB = min over non-pruned leaf boxes of ObjBound ≤ F*`;
  `global_UB = min(hint F, in-box CEs)`; pruned boxes (lb ≥ UB−tol) dropped. Children inherit the
  parent’s (valid) lb, so unsolved leaves still carry a valid LB.
- **Per-box warm start:** each box’s `warm` is ALWAYS a genuine CE (the global hint, or an in-box
  CE once found) ⇒ `b_hat_hint` registration stays valid (no invalid UB). The child containing the
  parent’s in-box CE inherits it (a real warm start there); the other falls back to the global CE.
  (`_make_decomp` gained `b_hint_override`.)
- **`--split-k`:** split the K weakest leaves per round (wider rounds = more parallel array width);
  caps at the #non-pruned candidates.

**Validation (IEEE 14 run-local):**
| round | global_LB (120s/box) | note |
|------:|---------------------:|------|
| 1 | 1.3067 | solve root → split |
| 2 | 1.3205 | box pruned + split (1 infeasible child auto-pruned) |
| 3 | 1.3444 | |
| 4 | 1.3458 | |

`global_LB` rises **monotonically** (best-first refinement), infeasible children auto-pruned,
`global_UB=2.3823` (hint, valid) throughout — the mechanism the one-shot static port could not do.
Convergence rate is governed by **per-box budget × parallel width**: 60s/box → ~0.98, 120s/box →
~1.35 (budget dominates, consistent with the scaling study); IEEE 14’s feasible region is a thin
low-`b` sliver (each split → one feasible + one infeasible child), so best-first naturally tracks
it. **Production certification = run on Leftraru with high per-box budget (≥600 s) and many
rounds** (the array submit loop is in `node_obbt_round.slurm`); the local runs validate the
mechanism, not the final gap.

### Time-limited benchmark + global time limit (2026-06-06)

Added a real **global wall-clock time limit** and a results recorder so the CE solve can be run
as a proper time-limited experiment (record CE, gap, time, hit-limit).

- **`UCDecomp4b(time_limit=…)`** — global budget (s) for the whole `run()`. Checked between CCG
  iterations AND inside the preamble (seeding loop + root-OBBT pass, per-pattern/per-line) so the
  run never overshoots; each internal MIQCP solve's `TimeLimit` is also capped by the remaining
  budget. On hit: `termination_reason="global_time_limit"`. `run()` now returns **`wall_time_s`**
  and **`hit_time_limit`**. (A partial OBBT cut by the deadline is still valid — OBBT is monotone.)
- **Coordinator** (`node_obbt_coordinator.py init --time-limit …`) — `run-local` enforces the
  global budget across rounds, caps each box by the remaining budget, and records
  `wall_time_s`/`hit_time_limit` in the state.
- **`run_benchmark.py`** — runs each (solver ∈ {node, coordinator}) × (grid) under the budget and
  appends a row to `results.csv` + a record to `results.json`:
  `grid, solver, time_limit_s, F_opt (CE cost), master_LB, gap, gap_pct, certified, wall_time_s,
  hit_time_limit, termination_reason, n_lines_changed, ce_changes` (+ full `b_hat`/`b0` in JSON).

**Validated:** overshoot fixed (IEEE 39 @120 s limit: wall 127 s, was 514 s; `hit_TL=True`); realistic
IEEE 39 @700 s: **certified** F=0.7060, gap≈0, wall 542 s, `hit_TL=False`, CE `L13:+2.819`; IEEE 14
@150 s: `hit_TL=True`, F=2.3823, LB 0.885, gap 62.9 %, CE `L14:+6.825; L15:+7.556; L18:+2.925`.

**Calibration:** `node` pays root OBBT ONCE then reuses the master — efficient sequentially.
`coordinator` rebuilds the master (incl. OBBT) PER BOX (the price of independent parallel jobs), so
`--per-box-budget` must exceed the per-box OBBT (~480 s on IEEE 39) — use **600 s**. Full run:
`run_benchmark.py --time-limit 3600 --grids 14,39,57 --solvers node,coordinator --per-box-budget 600`.

**1-hour benchmark RESULTS (2026-06-06, single machine, time_limit=3600 s, per-box 600 s):**
| grid | solver | CE F_opt | CE changes | LB | gap% | certified | wall s | hit_TL |
|------|--------|---------:|------------|-----:|-----:|-----------|-------:|--------|
| **IEEE 39** | **node** | **0.7060** | L13:+2.819 | 0.7060 | **0.0008** | ✅ **yes** | 599 | no |
| IEEE 39 | coordinator | 0.7138 | L13:+2.850 | 0.7060 | 1.09 | no | 3612 | yes |
| **IEEE 57** | **node** | **10.390** | L20:+19.9; L69:+33.7 | 9.874 | **4.97** | no | 3604 | yes |
| IEEE 57 | coordinator | 11.680 | L20:+13.1; L69:+41.5 | 8.475 | 27.44 | no | 3638 | yes |
| **IEEE 14** | **node** | **2.382** | L14:+6.8; L15:+7.6; L18:+2.9 | 1.633 | **31.44** | no | 3605 | yes |
| IEEE 14 | coordinator | 2.382 | (same) | 1.625 | 31.77 | no | 3607 | yes |

Takeaways: **IEEE 39 (node) certifies 0% in ~10 min** (minimal CE = expand line 13 by +2.82). The
**`node` solver is the sequential winner** (39 cert; 57 ~5%; 14 ~31%); the `coordinator` is worse on
one machine (rebuilds OBBT per box) — it only pays off with concurrent boxes on HPC. IEEE 14/57 hit
the 1 h limit ⇒ need the parallel Leftraru budget to certify. Full data: `benchmark_results/results.csv`.

### Next
1. **Run the v3 coordinator on Leftraru** at high per-box budget — the gap-closer for IEEE 14/57.
2. Optional: warm-start the *infeasible-side* children less wastefully (skip solving a child whose
   box the carve already proved CE-free).
3. Optional: per-box OBBT re-tightening only the SPLIT line’s μ.

---

## Root OBBT — IMPLEMENTED + VALIDATED + CERTIFIES IEEE 39 (2026-05-30)

**Strategy 1 (root OBBT on free-line flow duals μ_p/μ_m) — IMPLEMENTED** as `obbt: bool = False` ctor arg + companion `obbt_safety`, `obbt_iter`. New method `_obbt_root(m, master_vars, patterns, ...)`:
- Builds a **McCormick LP relaxation** R of the (exact-bilinear) master m by calling `_build_master_base` with `bilinear_exact=False` and re-adding the same seeded KKT blocks (so name suffixes `_j` align).
- For each free `(ell, t)` and each pattern j, maximises `μ_p[j,ell,t]` and `μ_m[j,ell,t]` over `R.relax()` (LP — drops u_foil integrality, segment binaries).
- Tightens m's μ-variable UB if the OBBT bound improves on `M_box`. Provably valid (S_exact ⊆ S_R; bound from R cannot exclude exact-feasible points).
- Optional iterated passes (default `obbt_iter=2`) — each pass uses the previous pass's tightened UBs in R for further tightening.
- Call site: `run()` after seeding loop, before main loop. No-op when `comp_mode != "strongdual"` or no patterns are seeded.
- Hand-off doc + pseudocode: `DECOMP_OBBT_offload.md`.

### Validation (`_check_strongdual_valid.py … exact obbt`) — all VALID
Custom path in the script: build master + 1 KKT block at b_BS plain-optimal pattern (b free), apply OBBT, then fix b=b_BS, solve, check obj == F(b_BS).
- IEEE 14: 286/288 bounds tightened/pass, fix-b solve status=2, obj=2.3823 = F(b_BS) ✓
- IEEE 39: 480/480 bounds tightened/pass, fix-b solve status=2, obj=0.7138 = F(b_BS) ✓
  - **Breakthrough**: previously the IEEE 39 fix-b feasibility check with `bilinear_exact` did NOT complete in the smoke window (spatial B&B on b·μ). With OBBT-tightened μ first, it OPTIMAL in seconds.
- IEEE 57: 192/192 bounds tightened/pass, fix-b solve status=2, obj=11.359 = F(b_BS) ✓

### Smoke results (`_smoke_strongdual.py strongdual seed <grid> none exact obbt`)
Full DECOMP loop, 4 iters, 120s master TL, seed_patterns=True (so OBBT has patterns to tighten):
| Grid | F_opt | master_LB | gap% | iters | term | stall verdict |
|------|-------|-----------|------|-------|------|---------------|
| IEEE 14 | 2.3794 | 0.9373 | 60.61 | 3 | cycle | **MISSING PATTERN, inflation=0.00** |
| **IEEE 39** | **0.7060** | **0.7060** | **0.01** | 2 | **certified_optimal** | n/a |
| IEEE 57 | **10.3623** (NEW BEST) | 6.2095 | 40.08 | 4 | cycle | **MISSING PATTERN, inflation=0.00** |

**IEEE 39 IS CERTIFIED**: gap 0.01%, far exceeds the offload doc's 5% target. Headline result.

**The diagnostic flip**: prior to OBBT the stall verdict on all grids was "RELAXATION GAP" (McCormick inflation ~oracle CE gap — see `_diagnose_stall`). With `bilinear_exact + OBBT` the inflation drops to **0.00 on every pattern on every grid** — the LB ceiling that had blocked IEEE 14/57 is now the **pattern set**, not the relaxation. This is a different bottleneck class and is addressable via pattern-source / multi-pattern work, not by tighter relaxations.

**IEEE 57**: F_opt=10.3623 is strictly better than the B&S hint (11.36) and the bigM result (11.36) — a NEW best CE. LB raised from 0 (bigM) / 6.27 (strongdual loose) to 6.21 (bilinear_exact, exact).

**IEEE 14**: cycle halts after 3 patterns at LB 0.94. Inflation = 0, so the relaxation is exact at the cycle point; the master's optimistic b_k satisfies every existing pattern's TRUE dispatch ≤ master_foil. Need a new pattern source (Technique 1 / multi-pattern) to break the cycle.

### Cost / runtime
- OBBT pass: `2 × nFree × T × n_patterns` LPs over the McCormick LP relaxation. IEEE 14: 864 LPs/pass (~seconds total). IEEE 39: 960 LPs/pass. IEEE 57: 576 LPs/pass.
- Per-LP TimeLimit = 30s (safety; LPs solve in <1s typically). NumericFocus = 2 for robustness.
- Default `obbt_iter=2` — pass 2 reuses the tightened bounds on R for further tightening. In practice both passes shrink similarly (suggests iterated OBBT has incremental value; could try 3+ on hard grids).

### Master-MIP scaling is the lever (2026-05-30, IEEE 14 TL sweep)
Re-ran IEEE 14 (strongdual+exact+OBBT+seed) at **600s** master TL instead of 120s
(`_smoke_strongdual.py strongdual seed 14 none exact obbt 600 6`). Both bounds improved
substantially — the earlier "cycle / MISSING PATTERN" verdict was an **undersolving
artifact**:

| master TL | LB (ObjBound) | best strict CE (UB) | gap% | iter-1 behavior |
|-----------|--------------:|--------------------:|-----:|-----------------|
| 120s | 0.9373 | 2.379 | 60.6 | b_k pinned at hint → SP1 returns seeded pattern → premature cycle |
| **600s** | **1.139** | **1.9563** | **41.78** | b_k navigates to NEW region → SP1 returns NEW pattern (sum_u=90) → loop progresses |

- At 600s the master itself navigates to a strict CE at **F=1.9563** (oracle gap=0,
  accepted via `_ce_ok`) and discovers a genuinely new plain pattern (sum_u=90, not a
  seed) — no premature cycle. **Pattern coverage and master-MIP scaling are the SAME
  root cause**: solve the exact MIQCP master better and it both raises ObjBound and
  finds new patterns. The 120s cycle-halt was the master stuck at the warm-start hint.
- The minimal **strict** CE on IEEE 14 is **~1.956** (found directly by the exact
  master), NOT 2.379 (that was undersolving). bigM's 1.906 is *sub-strict* (see strict-
  vs-tolerant CE note below).
- iter 2 (4 patterns) hit `time_limit_no_incumbent` even at 600s — **ROOT CAUSE FOUND +
  FIXED** (see "Strict-CE gate fix" below). The Gurobi log showed
  `User MIP start violates constraint opt_cut_3 by 14.97`: the iter-1 b_k (F=1.9563) was
  accepted as a CE by the LOOSE `_ce_ok` (tol ~82) and used to refresh the warm-start
  hint + F-cap, but its oracle gap is 14.97 (NOT strict) — pattern 90, just added, cuts
  it off by exactly 14.97. So the warm start was infeasible for the iter-2 master and the
  F≤1.9563 cap excluded the true strict optimum. The fix gates state refresh on a STRICT
  CE check.

### Strict vs tolerant CE (2026-05-30, `_hybrid_obbt.py`)
Tested Strategy 6 (hybrid: bigM UB + strongdual/OBBT LB). bigM found F=1.9061 but the
oracle shows **v_foil − v_plain = +0.22 > 0** → it is NOT a strict CE (foil not optimal
at that b; passes only because 0.22 ≪ rel-tol ~83). Fed as a hard `F ≤ 1.9061` cap to
the exact master, the feasible region (strict CEs that cheap) is empty/tiny → master
returns no incumbent → LB lost. **The bigM UB and the exact LB are not directly
combinable** (tolerant vs strict CE). Part of the docs' "20% improvement (1.91 vs 2.38)"
is a CE-tolerance artifact; the honest minimal strict CE is ~1.956. Full analysis:
`DECOMP_theoretical_strategies.md` §7.

### Strict-CE gate fix + honest cycle labeling (2026-05-31, user chose STRICT mode)
Two coupled correctness fixes in `uc_decomp_4b.py:run()`:

1. **`_ce_strict` gate** (new `eps_ce_strict: float = 1.0` ctor arg). The loose `_ce_ok`
   (tol `max(eps_weak, 1e-4·|vp|)` ≈ 82) is still used for the informational b0 / hint
   CE checks, but **best_F, the `F ≤ F_hint` cap, and the warm-start hint are now
   refreshed ONLY when `vd − vp ≤ eps_ce_strict`** (a true strict CE). A tolerant-but-
   not-strict b_k (gap up to ~82) has a missing pattern beating the foil by that gap;
   refreshing state to it broke the next exact-cut warm start (the 14.97 violation) and
   over-tightened the cap. Now such a b_k is reported (`[CE] b_k tolerant-CE only …`),
   its pattern is still added, and the loop keeps the previous strict-feasible hint
   (B&S) so iter k+1's warm start stays valid. The LB still lower-bounds the minimal
   STRICT CE (the exact-cut feasible-region min), so no false certification.
2. **Honest cycle labeling**. A repeated plain pattern is only a TRUE cycle when the
   master solved to (near) optimality. When the master hit its TIME LIMIT (undersolved),
   a repeat just means "the exact MIQCP couldn't reach the next optimistic point in
   budget" → now reported as `termination_reason="time_limit_undersolved"` (a
   master-MIP-scaling limit), NOT `"cycle"` (a pattern-source limit). Stops conflating
   the two.

**Validation (IEEE 14, exact+OBBT+seed, 600s×6, strict mode):** iter 2 now RUNS
(previously `time_limit_no_incumbent`). iter 1: b_k=1.9563 correctly flagged tolerant-CE-
only (gap 14.97), pattern added, best_F stays 2.3823. iter 2: b_k=2.3729 is a STRICT CE
(gap 1.3e-07) → best_F=2.3729; then repeated-pattern-under-TL → `time_limit_undersolved`.
**But the LB went 1.139 → 1.008 when pattern 90 was added** — the bigger master proved a
WEAKER bound in the same 600s. ⇒ On IEEE 14 the exact MIQCP scaling is the hard wall:
accumulating patterns is counterproductive under a fixed budget. Contrast IEEE 39, which
certifies at iter 1 because its master solves to optimality fast. **Why IEEE 14's exact
MIQCP is harder than IEEE 39's (fewer free lines, fewer int vars, yet doesn't converge)
is the open question.**

### MIPFocus=3 (bound-focused) confirms the scaling lever (2026-05-31)
New ctor arg `master_mip_focus: int = 1` (1=incumbent, 3=bound), threaded into the master
solve. Tested IEEE 14 with the SMALL seed-only master (3 patterns, no accumulation),
`MIPFocus=3`, 900s, `max_iter=1`:

| config | master | ObjBound (LB) | strict-CE UB | gap% |
|--------|--------|--------------:|-------------:|-----:|
| MIPFocus=1, 600s | 3 seeds → +pattern90 | 1.139 → 1.008 (patterns HURT) | 2.373 | 57 |
| **MIPFocus=3, 900s** | **3 seeds only** | **1.60–1.63** (two runs) | 2.3823 (B&S strict) | **~32** |

**Bound-focused solving of a SMALL master beats incumbent-focused pattern-accumulation**
for the IEEE 14 LB: ObjBound climbed 1.139 → ~1.60 (1.6313 then 1.6044 on a clean re-run;
NonConvex B&B is nondeterministic), gap 60.6% → ~32%. Confirms the "patterns hurt the LB
under a fixed budget" observation — the right tactic is a small exact master solved
bound-focused, NOT accumulating cuts. The clean re-run's incumbent (F=2.339) was correctly
flagged tolerant-CE-only (gap 6.57 > eps_ce_strict) and did NOT corrupt best_F, which
stayed at the B&S strict CE 2.3823 — so the reported certificate is honest:
**LB 1.604, UB 2.3823 (strict), gap 32.65%** (best strict CE found in other runs ~2.373).
(First 900s run died on a power-off, not a code error — WLS phone-home failed mid-oracle;
re-run was clean. `_optimize_with_retry` could still be hardened against network drops.)

### Enriched seeding CERTIFIES IEEE 39 rigorously (2026-06-01)
New ctor arg `seed_interp: int = 0`: with `seed_patterns=True`, also seed plain-optima at
`seed_interp` INTERIOR points `b0 + α(bU−b0)`, `α=k/(seed_interp+1)` (Strategy 4: pattern
diversification). Targets the "missing pattern" stall where the exact master solves to
optimality but its optimum `b_k` isn't a strict CE because the thin corner-only cut set
lets `b_k` sit between patterns.

**IEEE 39 result** (`_smoke_strongdual.py strongdual seed 39 none exact obbt 900 4 0 3 3`,
i.e. exact+OBBT+MIPFocus=3+`seed_interp=3`): seeds 5 patterns (b0, bU, interp@0.25/0.50/
0.75; b_hat deduped). Master ObjBound=0.7060, `b_k=0.7060` is now a **STRICT CE
(oracle gap 1.16e-10)**, `best_F=0.7060`, **gap 0.00%, `term=certified_optimal`**. This is
the rigorous version of the earlier "0.01%" (which had certified a non-strict 0.706 with
gap 8.98). Bonus: **F=0.7060 < B&S 0.7138** — a better CE too. The interior seeds
(sum_u=48, 46) were exactly the missing patterns.

**IEEE 57: enriched seeding does NOT apply.** Same config with `seed_interp=3`: the
interior points DEDUPE (plain-optimal commitment is ~constant along the b0→bU path), so no
new patterns; the 3rd corner pattern even bloated the model and the NonConvex bound came
out WEAKER (LB 8.57, gap 17.3%) than the 2-pattern notebook run (LB 10.09, gap 3.91%). ⇒
**IEEE 57 is a pure master-scaling problem** (small master + more time), the OPPOSITE of
IEEE 39 (which needed more patterns). 39 = missing-pattern; 57 = undersolved MIQCP.

### IEEE 14 ceiling + running-max LB fix (2026-06-02)
- IEEE 14 with `seed_interp=3`: interior points DO generate new patterns (sum_u=90,89,76,
  distinct from corners — unlike 57), BUT the 6-pattern master then UNDERSOLVES (hit TL at
  ~35% internal gap), LB 1.54 — worse than the lean 3-pattern run. ⇒ 14 wants a LEAN master.
- IEEE 14 lean (3-pattern) ceiling: even at **1800s** the master stays at **~28% internal
  Gurobi gap**; LB climbs only 0.94→1.58→1.68 for 120s→900s→1800s (**diminishing returns**).
  ⇒ IEEE 14 is genuinely exact-MIQCP-bound; brute-force time won't certify it. Real lever =
  **per-node OBBT** (tighten μ inside the B&B tree, not just root) or structural work.
- **Running-max LB fix** (`uc_decomp_4b.py:run()`): report `master_LB = max_k ObjBound_k`,
  not the last iter's. Each iter's master is a relaxation of the full problem ⇒ each
  ObjBound ≤ F*; the exact MIQCP is non-monotone under a TL (iter1 1.681 → iter2 1.631 on
  IEEE 14, the better bound was being discarded). Max is valid and free. IEEE 39 regression:
  still certifies 0.00%.

### HPC-readiness upgrades (2026-06-03) — multistart, iterated OBBT, resilience, repro
Four upgrades added before moving to HPC, all validity-gated:

1. **Multi-start max-ObjBound** (`master_multistart: int = 1`, `master_seed`). Solves each
   per-iteration master with N Gurobi Seeds, keeps `max_k ObjBound_k` (every ObjBound is a
   valid LB ⇒ max is valid). Beats NonConvex-MIQCP bound variance. Cheap: stops after the
   first seed that hits OPTIMAL or finds no incumbent (so IEEE 39 pays for 1 seed).
   HPC-parallelizable (independent seeds). **WIN: IEEE 57 1.52% → 0.57%** (3 seeds at 600s:
   seed0 LB 10.07, seed1 10.20, max 10.331 vs single-run 10.232; UB 10.3904 → gap 0.57%).
2. **Iterated OBBT** (`obbt_refresh: bool = False`). Re-runs `_obbt_root` after `best_F`
   drops (the tighter F-cap shrinks b/μ bounds further). Valid; gated off by default.
3. **Network resilience**: `_optimize_with_retry` now uses exponential backoff (8 retries,
   15s→…→600s, env-tunable `GRB_RETRY_MAX`/`GRB_RETRY_WAIT`) so unattended HPC runs survive
   transient WLS license/network blips. (A full power-off is still unrecoverable.)
4. **Reproducibility + Threads** (`master_seed`, `master_threads`) + **validity gate**
   `_validate_all.py` — one command asserting VALID at b_BS on all 3 grids (exit 0/1, CI).

Bug fixed along the way: `m.MIPGap` is not always retrievable for a NonConvex MIQCP after
multi-start re-optimizes — replaced with `_safe_mipgap()` (falls back to
|ObjVal−ObjBound|/|ObjVal| then nan). IEEE 39 regression: still certifies with
`multistart=3 + obbt_refresh`.

**Updated standing with multistart: IEEE 39 CERTIFIED 0%; IEEE 57 ~0.57% (multistart);
IEEE 14 ~30% (still MIQCP-hard).**

### b-OBBT + IEEE 57 setup fix (2026-06-03)
Two next-step items done:

1. **IEEE 57 smoke/notebook mismatch RESOLVED** — root cause was the smoke scripts
   (`_smoke_strongdual.py`, `_check_strongdual_valid.py`) using `carbon_price=None,
   slack_bus=None`, default init for 57, while the notebook (and the B&S checkpoint) use
   `carbon_price=50, slack_bus=0` + a G0 warm-start (as in `bs_7grids.ipynb`). Fixed the
   smoke scripts with per-grid setup. Validator now reports `F(b_BS)=11.6796` (was
   11.3587) and VALID. 14/39 were already matching.
2. **b-OBBT** (`_obbt_root` now also tightens the SHARED `b[ell]` bounds, both UB and LB,
   over the same relaxation+F-cap). `b[ell]` is the spatial-B&B's MAIN branching variable,
   so shrinking its box is the most direct speedup for `bilinear_exact`. Valid by the same
   relaxation argument as the μ bounds (validated: b_BS feasible on IEEE 14; IEEE 39 still
   certifies 0.00%). Cost: +2·nFree LPs per pass.

**b-OBBT results (exact+OBBT+MIPFocus=3, lean, smoke):**
| Grid | before (μ-only OBBT) | after (+b-OBBT) |
|------|----------------------|-----------------|
| IEEE 14 | LB 1.68 needed 1800s | **LB 1.678 at 900s** (~2× speedup); gap 29.5% |
| IEEE 39 | certified 0.00% | certified 0.00% (no regression) |
| IEEE 57 | 3.91% (notebook, μ-only) | **gap 1.52%**, LB 10.23, strict CE 10.39 (beats B&S 11.68 by 11%) — undersolved, near-certifying |

b-OBBT is internal to `_obbt_root`, so the notebook picks it up automatically via
`obbt=True` (no notebook code change).

**IEEE 57 near-certification + NonConvex bound variance (2026-06-03):** the 900s run gave
LB 10.2321 (gap 1.52%); a 1800s rerun gave LB 10.1627 (gap 2.19%) — WORSE. This is
NonConvex-MIQCP ObjBound *variance* (the spatial-B&B proves a different bound each draw),
not a real regression: more wall-clock didn't help. **A valid LB is valid regardless of
which run produced it**, so IEEE 57's best certificate stands at **LB 10.2321, strict CE
10.3904, gap 1.52%** (strict CE beats B&S 11.6796 by ~11%). Formal 0% certification on 57
is blocked by bound variance, not a real ceiling; a multi-start (keep the max ObjBound
across runs) or tighter OBBT would close it. Practically 1.52% is a strong valid result.

### Per-grid lever summary (2026-06-02) — they differ
| Grid | bottleneck | lever | best result |
|------|-----------|-------|-------------|
| IEEE 39 | missing pattern | `seed_interp=3` (interior seeds) | **CERTIFIED 0.00%**, F=0.7060 (< B&S 0.7138) |
| IEEE 57 | undersolved MIQCP | lean master + time (interp DEDUPES) | notebook 3.91% (valid); smoke instance differs |
| IEEE 14 | exact MIQCP hard | lean master; per-node OBBT next (time→diminishing) | LB~1.68, strict CE 2.37, ~29% gap |

### KNOWN ISSUE — IEEE 57 smoke vs notebook mismatch
`_smoke_strongdual.py` and the notebook solve DIFFERENT IEEE 57 instances: smoke
F(b_hat)=11.3587 (own weights) vs notebook/checkpoint 11.6796. IEEE 14 & 39 MATCH
(2.3823, 0.7138) — only 57 diverges. Likely `make_line_weights`/`util` differs for 57
(the B&S 57 checkpoint was built with different weights than the smoke recomputes). The
**notebook is the consistent artifact** (matches the checkpoint); trust its 57 numbers.
TODO: reconcile the 57 weight computation between the two paths.

### Next levers (revised 2026-06-01, superseded by per-grid table above)
0. **Per-grid lever now differs**: 39 SOLVED (enriched seeding). 57 = small master + more
   time (NOT more patterns — they dedupe and bloat). 14 = both hard master + needs check
   whether interp generates new patterns.

### Next levers (superseded 2026-05-31)
1. **MIPFocus=3 + small master + more time / tighter OBBT** — the confirmed lever for the
   IEEE 14/57 LB. Default seed-only (don't accumulate). Next: longer TL (1800–3600s) or
   `obbt_iter`>2 to see if ObjBound reaches best_F (~2.34) and certifies IEEE 14.
2. **Per-node OBBT** (Strategy 1 escalation) if root OBBT + MIPFocus=3 plateau.
3. **Investigate why IEEE 14 ≫ IEEE 39 in MIQCP difficulty** (IEEE 39 certifies at iter 1;
   IEEE 14 doesn't despite fewer free lines / int vars — numerics? degeneracy? F-scale?).
4. **Shared-`b` RLT** — demoted; only if a relaxation gap re-emerges (inflation is 0).

### Files modified
- `uc_decomp_4b.py`: added `obbt`, `obbt_safety`, `obbt_iter`, `eps_ce_strict` ctor args; added `_obbt_root` method (wired into `run()` after seeding); added `_ce_strict` gate for best_F/cap/hint refresh; honest cycle vs `time_limit_undersolved` labeling.
- `_check_strongdual_valid.py`: added slot-6 "obbt" arg + custom validation path (OBBT runs with b free, then b fixed for feasibility check).
- `_smoke_strongdual.py`: added slot-6 "obbt", slot-7 master TL, slot-8 max_iter, slot-9 master_output_flag args.
- `_hybrid_obbt.py`: NEW — Strategy 6 (bigM UB + strongdual/OBBT LB) two-stage experiment.
- `build_decomp_notebook.py`: threaded `bilinear_exact` and `obbt` kwargs through `decomp_cell`.

---

## Continuing this work (POST-OBBT, 2026-05-30)

**Root OBBT done** — IEEE 39 certified at 0.01%; IEEE 14/57 still uncertified but
diagnostic verdict flipped from "RELAXATION GAP" to "MISSING PATTERN" with **inflation=0
on every pattern**. The McCormick slack is gone; the cap on IEEE 14/57 is the CCG
cycle-halt after 3–4 patterns. See §"Root OBBT — IMPLEMENTED + VALIDATED + CERTIFIES
IEEE 39" above and `DECOMP_theoretical_strategies.md` §6 for the diagnostic flip.

### Next levers — bottleneck class changed, lever set changed too

1. **Verify cycle-halt cause first** (cheap, ~minutes). `_diagnose_stall` reports
   "MISSING PATTERN" but the master MIPs also hit time limits at 40–60% gap. Grep
   smoke logs for `cycle detected` vs `time limit hit` to confirm the halt is on
   `_pattern_key(u_k) ∈ seen` and not on a master-MIP timeout that aborted pattern
   discovery prematurely. If it's the latter, the right move is to raise
   `master_time_limit`, not to add pattern sources.
2. **Pattern-source diversification** (Strategy 4 in `DECOMP_theoretical_strategies.md`
   §6, NEW). At cycle-halt, query `oracle.solve_plain` at 2–3 perturbed `b` points
   (or use k-shortest commitments, or perturbed-cost solves) and seed the resulting
   `u_j` as additional KKT blocks. If LB rises, the bottleneck is confirmed
   pattern-side and we iterate the loop.
3. **Per-node OBBT inside spatial B&B** (Strategy 1 escalation,
   `DECOMP_theoretical_strategies.md` §6 "Strategy 5"). Sharpens μ bounds using
   node-local fixings. Useful if the master MIP itself (not the CCG loop) is the
   slowdown — but root OBBT already certified IEEE 39, so this is only worth it on
   harder grids than 57.
4. **Demoted — Strategy 2 (shared-`b` RLT) and Strategy 3 (Balas disjunctive hull)**
   in `DECOMP_theoretical_strategies.md`. Both were motivated by McCormick relaxation
   slack. With inflation=0 they have nothing to tighten. Move back up only if a
   future change re-introduces a relaxation gap (e.g. dropping `bilinear_exact` for
   runtime on a larger grid).

### Companion docs (read in this order for a cold pickup)

- **`DECOMP_state.md`** (this file) — current state + results table. **Authoritative.**
- **`DECOMP_OBBT_offload.md`** — §7 has the post-OBBT results and revised escalation
  (was the implementation handoff doc; now also records what happened).
- **`DECOMP_theoretical_strategies.md`** — §1–§5 is the original
  relaxation-tightening landscape; **§6 is the post-OBBT pivot** that introduces
  Strategies 4 (pattern-source diversification), 5 (per-node OBBT), and 6 (hybrid
  bigM + bilinear_exact verifier), and demotes RLT/Balas.
- To resume in a fresh session: *"Read DECOMP_state.md §'Root OBBT' and
  DECOMP_theoretical_strategies.md §6, then continue."*

## Theoretical research directions for tighter valid LB (summary; details in companion doc)

The slowness of `bilinear_exact` on IEEE 39/57 is intrinsic to general-purpose spatial B&B on `b·μ`. Beating it requires problem-structure-specific tightening. Candidate research directions, roughly ordered by promise:

1. **RLT (Reformulation-Linearization Technique, Sherali-Adams) on the bilinear set with shared `b`.** Multiplying valid constraints (e.g. `b[ell] − bL[ell] ≥ 0` with `μ ≥ 0`, `M_box − μ ≥ 0` with `bU − b ≥ 0`, etc.) produces valid inequalities tighter than McCormick. Higher-order RLT (triple products) yields convex hulls of multi-term bilinear sets. Because our flow term is `Σ_t b[ell]·μ_p[ell,t]` with `b[ell]` **shared across t and across both `μ_p`/`μ_m`**, the joint convex hull is strictly tighter than the sum of per-term McCormicks (Anstreicher–Burer, "convex envelopes of bilinear with shared variables"). Likely the cheapest concrete improvement.

2. **Disjunctive convex hull of `foil ≤ v(u^j,b) = max_y y·rhs(b)` (Balas).** The set is a union of half-spaces, one per dual vertex. Balas's disjunctive convex hull gives an exact extended-formulation polyhedron with one disaggregated copy per disjunct — *valid* (unlike the naive vertex-selection that fails for our LB direction), and generated on demand via column generation. Theoretically the cleanest "exact valid" representation; non-trivial to derive because the disjuncts are `b`-affine half-spaces and the disaggregation must preserve the affine structure.

3. **Optimization-based bound tightening (OBBT) inside spatial B&B.** Before each branch, solve an auxiliary LP to maximize / minimize each `μ_p[ell,t]` over the current master feasible set. Tighter per-`(ell,t)` `μ` bounds shrink the McCormick envelope at each node → fewer branches → faster certification. Cheap to implement (LPs only) and orthogonal to the relaxation choice. Could be combined with `bilinear_exact` to speed it up on 39/57.

4. **Time-decomposed convex hulls.** The dispatch LP couples weakly across time (only ramp constraints span t–1↔t). For each line `ell` separately, the per-time `b[ell]·μ_t[ell]` terms could be aggregated via a tighter "many small bilinear with shared b" convex hull (Tawarmalani-Sahinidis convex extensions). Less explored in the bilevel CE literature but well-studied in MINLP.

5. **Value-function reformulations (Mitsos et al.).** Skip the KKT/strong-duality machinery entirely; represent `v(u^j,b)` parametrically via its known piecewise-linear convex structure in `b` and enforce the constraint using semi-infinite or parametric optimization techniques. Theoretical-grade, but bypasses both bilinearity and complementarity.

6. **Outer approximation of `v(u^j,b)` from above.** Solve the dispatch LP at a finite grid of `b` values, collect the optimal primals `x_b`, and use each `c·x_b` as a "supporting upper bound" on `v` over a neighborhood of `b`. Then `foil ≤ min_b' c·x_{b'}` is a valid (loose) cut. Cheap, but the upper bound is only sharp at the sampled `b`'s — would need adaptive sampling.

For practical recommendation: **start with OBBT (3) layered on top of `bilinear_exact`** — it's the smallest change with the most likely speedup. RLT (1) is the next step if more tightening is needed. Disjunctive convex hull (2) is the rigorous endgame but a real research project.

**Caveat on McCormick (strongdual)**: `w = b·μ` is only exact at the box corners of `(b, μ)`; the envelope is a relaxation, so the master stays a valid *lower-bound* relaxation (ObjBound ≤ F_true, no true CE excluded) but a returned `b_k` may not be an exact CE — the oracle re-check still gates incumbents, so correctness of the final answer is unaffected.

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
