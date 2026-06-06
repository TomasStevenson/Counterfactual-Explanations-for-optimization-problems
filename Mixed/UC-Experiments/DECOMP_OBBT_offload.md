# DECOMP — OBBT inside `bilinear_exact`: session offload

*Generated 2026-05-30. Pick this up cold in a new Claude Code session by saying:*
**"Read `DECOMP_OBBT_offload.md` and continue."**

*Also read first:* `DECOMP_state.md` (authoritative current state),
`DECOMP_theoretical_strategies.md` (why OBBT is the chosen next lever),
`DECOMP_lb_stagnation.md` (original Fix 1/2/3 diagnosis).

---

## 1. Where things stand

`comp_mode="strongdual"` (strong-duality + McCormick on `b·μ`) is correct and gives a
**valid** but loose LB (IEEE 39 ~6.8% gap, IEEE 14/57 ~60%/37%). The exact NonConvex
formulation (`bilinear_exact=True`) is correct on IEEE 14 (validated, b_BS feasible) but
**Gurobi's spatial branch-and-bound on `b·μ` doesn't scale** — IEEE 39 fixed-`b`
feasibility check (`debug_fix_b`) did not complete in a smoke window (killed 2026-05-30).

Other ideas that have been ruled out:
- Hybrid mode (Fix 1): non-flow indicators — leaves the binding bilinear flow pairs as
  bigM, no LB lift.
- Seeding (Technique 2): pattern count is not the bottleneck.
- μ-box heuristic tightening (`mccormick_mu_factor`, Option A): the box is loose where
  `μ` binds (b0/b_hat/bU), but at the LB-optimal `b` the free flows are slack ⇒ box
  inactive there. Knob kept default off.
- Piecewise McCormick (`mccormick_segments`): inflation falls ~1/K but cycle-halt LB is
  non-monotone and the disaggregated model doesn't scale past K≈8.
- Dual-vertex/Benders: invalid LB direction in our problem (`v` is in a *constraint*, not
  the objective). RETRACTED.

**OBBT is the next lever** because it is (a) **provably valid** (unlike Option A's
heuristic factor), (b) **orthogonal** to the bilinear representation (combines with
strongdual *or* bilinear_exact), and (c) cheap to implement and likely to speed up the
spatial B&B that's killing IEEE 39.

---

## 2. Goal of this session

Tighten `μ_p[ell,t]`, `μ_m[ell,t]` upper bounds (and possibly lower bounds) per
free `(ell, t)` by solving auxiliary LPs *over a provably valid relaxation* of the
master MIQCP. With tighter `μ` bounds:

- The McCormick envelope on `b·μ` shrinks at every spatial-B&B node.
- Gurobi's `bilinear_exact` solve on IEEE 39 should become tractable.
- The strong-duality LP relaxation (used as the master's root LP) tightens too.

**Target empirical outcome:** `bilinear_exact=True` + OBBT solves IEEE 39 to certified
optimality (gap ≤ 5%) in under 15 min wall, and IEEE 14 in seconds.

---

## 3. Algorithm — root OBBT

Start with **root OBBT** (called once before the main solve loop). If insufficient,
escalate to per-node OBBT later.

```
INPUT:  master model m (after _build_master_base + any seeded KKT blocks)
        free line set, T, current mu_p / mu_m variables

1. Build R = LP relaxation of m:
     - Drop integrality (m.relax()).
     - Disable NonConvex (Params.NonConvex = 0) AND replace the exact bilinear
       b·μ in the strong-duality equality with its STANDARD McCormick LP
       envelope (the same one used when bilinear_exact=False).
     This R is a provably valid relaxation of the exact MIQCP master.

2. For each free line ell, each t in 1..T:
     For each dual_var d in (mu_p[ell,t], mu_m[ell,t]):
         R.setObjective(d, GRB.MAXIMIZE)
         R.optimize()
         if R.Status == OPTIMAL:
             new_ub = R.ObjVal * (1 + safety_margin)   # e.g. 1e-6 safety
             d.UB = min(d.UB, new_ub)
         # Optionally also minimize → tighten LB (rarely improves > 0)

3. m.update()  # propagate new bounds
```

**Cost.** `2 × nFree × T` LPs per call. For IEEE 39 with ~10 free lines and T=24:
~480 LPs. Each LP is the size of the master relaxation (thousands of vars), modest
runtime — single-digit minutes total for root OBBT.

**Optional refinement: iterated OBBT.** After the first OBBT pass tightens bounds, the
LP relaxation R itself becomes tighter, so a second pass may further tighten bounds.
Iterate until the largest bound improvement < tolerance (or for a fixed number of
passes, e.g. 2–3).

---

## 4. Validity argument (why this is provably safe)

Let `S_exact` = feasible set of the exact MIQCP master (with `b·μ` exact). Let
`S_R` = feasible set of relaxation `R` (with `b·μ` replaced by McCormick `w`,
integrality relaxed). By construction `S_exact ⊆ S_R` (every exact-feasible point
satisfies McCormick and the dropped-integrality constraints).

Define `μ̄_p^{OBBT}[ell,t] = max{ μ_p[ell,t] : (b, μ, w, x, …) ∈ S_R }`. Then for any
`(b, μ, …) ∈ S_exact ⊆ S_R`:

```
μ_p[ell,t]  ≤  μ̄_p^{OBBT}[ell,t]
```

So replacing the original `μ_p[ell,t].UB` (which was the heuristic/provable `M_box`) with
`min(M_box, μ̄_p^{OBBT}[ell,t])` **cannot exclude any exact-feasible point**, hence cannot
exclude any optimal MIQCP solution. The master min stays ≤ F*, LB stays valid. ✓

**Contrast with Option A's heuristic `mccormick_mu_factor`:** that one bounds `μ` by
`factor × observed_max_at_{b0,b_hat,bU}`, which is *not* derived from the master's
constraints and could exceed the true bound at unobserved `b`'s. OBBT is the principled
version of the same intuition.

---

## 5. Implementation plan

### 5.1 Files to modify

- `Mixed/UC-Experiments/uc_decomp_4b.py` — new method + run() hook.
- `Mixed/UC-Experiments/_check_strongdual_valid.py` — extend args (already accepts FACTOR,
  SEGS, EXACT; add OBBT toggle if useful).
- `Mixed/UC-Experiments/_smoke_strongdual.py` — same.
- `Mixed/UC-Experiments/build_decomp_notebook.py` — thread `obbt: bool = False`
  through `decomp_cell`; add Section 4h or extend 4d.
- `Mixed/UC-Experiments/DECOMP_state.md` — record results.

### 5.2 Constructor change

```python
# in UCDecomp4b.__init__, after mccormick_segments / bilinear_exact:
obbt: bool = False,   # root OBBT pass on mu bounds before main loop (Strategy 1)
obbt_safety: float = 1e-6,
obbt_iter: int = 2,   # number of OBBT passes
# in __init__ body:
self.obbt = bool(obbt)
self.obbt_safety = float(obbt_safety)
self.obbt_iter = max(1, int(obbt_iter))
```

### 5.3 New method

```python
def _obbt_root(self, m, master_vars):
    """Root OBBT: tighten mu_p / mu_m UBs for free lines via valid LP relaxation.

    Provably valid: bounds derived from the master's own constraint set under a
    McCormick LP relaxation; cannot exclude any exact-feasible point. See
    DECOMP_OBBT_offload.md §4.
    """
    if not self.obbt:
        return
    nT = int(self.data.T)
    # Build R = standard McCormick LP relaxation of m.
    # NOTE: if comp_mode=="strongdual" with bilinear_exact=True, the master has
    # a quadratic constraint. R must use the McCormick LP envelope of b·μ instead,
    # NOT the exact product. Two options:
    #   (a) clone m, drop the strongdual{j} quadratic constraints, add McCormick
    #       w vars + envelope + replace flow term with -Σw; relax integrality.
    #   (b) build R from scratch using _build_master_base with a temporary
    #       comp_mode="strongdual" + bilinear_exact=False and re-add seeded patterns.
    # (b) is cleaner but pays the rebuild cost; (a) is faster if you can edit
    # the constraint list robustly. Recommend (b) for correctness initially.
    #
    # Pseudocode for (b):
    saved_exact = self.bilinear_exact
    self.bilinear_exact = False
    R, rv = self._build_master_base(...)         # need same window/init args
    for u_j in patterns_so_far:                  # if any seeded
        self._add_iteration_block(R, rv, ..., u_j, ...)
    R = R.relax()
    R.Params.OutputFlag = 0
    R.Params.NonConvex = 0
    R.Params.TimeLimit = 5.0   # per-LP time limit
    self.bilinear_exact = saved_exact

    n_tightened = 0
    total_shrink = 0.0
    for _ in range(self.obbt_iter):
        any_change = False
        for ell in self.free:
            for t in range(nT):
                for tag, src in (("mup", master_vars... ), ("mum", ...)):
                    # find the variable in R by name (matches m's naming
                    # — same _add_iteration_block builds them).
                    var_R = R.getVarByName(f"{tag}_0[{ell},{t}]")  # adjust suffix
                    if var_R is None:
                        continue
                    R.setObjective(var_R, GRB.MAXIMIZE)
                    R.optimize()
                    if R.Status == GRB.OPTIMAL:
                        new_ub = R.ObjVal * (1.0 + self.obbt_safety) + self.obbt_safety
                        var_m = m.getVarByName(...)   # same name on m
                        if new_ub < var_m.UB - 1e-9:
                            total_shrink += var_m.UB - new_ub
                            var_m.UB = new_ub
                            # mirror on R too for iterated OBBT
                            var_R.UB = new_ub
                            any_change = True
                            n_tightened += 1
        if not any_change:
            break
    m.update()
    if self.verbose:
        print(f"[OBBT] tightened {n_tightened} bounds; "
              f"total shrink {total_shrink:.2e}")
    R.dispose()
```

### 5.4 Call site in `run()`

Place the OBBT call **after** the seeded KKT blocks are added (if any) and **before** the
main loop, so root OBBT sees the strongest formulation:

```python
# after seeded patterns block, before "for k in range(self.max_iter):"
if self.obbt:
    self._obbt_root(m, master_vars)
```

### 5.5 Validation (REQUIRED before measuring LB)

```bash
# IEEE 14 + 39 + 57, exact bilinear + OBBT, b_BS feasibility:
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
  /c/Users/tomas/miniconda3/envs/ce-env/python.exe \
  _check_strongdual_valid.py 14 strongdual 1e4 none exact obbt
```

(Extend `_check_strongdual_valid.py` to accept an `obbt` arg in slot 6.) Must report
**VALID** (b_BS feasible, obj == F(b_BS)) on all grids. If any grid reports INVALID,
**stop** — the OBBT LP construction has a bug (most likely it's not actually a valid
relaxation; double-check that NonConvex=0 + McCormick envelope is used, NOT the exact
bilinear).

### 5.6 LB measurement (after validation passes)

```bash
# IEEE 14: should converge faster; same valid LB.
_smoke_strongdual.py strongdual noseed 14 none 1 exact obbt
# IEEE 39: the target — see if it now certifies in budget.
_smoke_strongdual.py strongdual noseed 39 none 1 exact obbt
# IEEE 57: stretch goal.
_smoke_strongdual.py strongdual noseed 57 none 1 exact obbt
```

Compare:
- `[OBBT]` log: how many bounds tightened, total shrink → sanity check OBBT did something.
- `master_LB`: should be ≥ the no-OBBT LB (or equal if OBBT doesn't help).
- Solve time per iter: should *decrease* on IEEE 39 thanks to tighter spatial-B&B nodes.
- `[STALL]` diagnostic at cycle-halt: inflation should drop (tighter μ → less McCormick
  slack → less inflation under bilinear_exact).

---

## 6. Risks and stop conditions

| Risk | Symptom | Action |
|------|---------|--------|
| OBBT LP isn't a valid relaxation (e.g. accidentally uses exact bilinear) | `_check_strongdual_valid.py` reports INVALID | Debug: verify NonConvex=0, McCormick `w` vars in R, no exact `b*μ` |
| OBBT LP is too loose (no tightening) | `[OBBT] tightened 0 bounds` | OBBT can't beat McCormick at the root; try per-node OBBT or pivot to RLT |
| OBBT pass is slow (>>1 LP/sec) | Wall time per call > 10 min on IEEE 39 | Add parallel OBBT (Gurobi `threads` param + concurrent objectives), or limit to lines with largest observed μ |
| LB stops being valid after OBBT (rare bug) | `master_LB > F*` (where F* known) | Roll back; the LP relaxation R has a bug |
| OBBT helps formulation but cycle-halt still caps reported LB | Same as piecewise's non-monotone issue | Combine with strongdual K=1 (no piecewise) so the comparison is clean |

---

## 7. After OBBT — RESULTS + revised next escalation

**Implemented + measured 2026-05-30** (see `DECOMP_state.md` §"Root OBBT" for details):

| Grid | F_opt | LB | gap% | term | `_diagnose_stall` |
|------|------:|---:|-----:|------|-------------------|
| IEEE 14 | 2.3794 | 0.9373 | 60.61 | cycle | **MISSING PATTERN, inflation=0** |
| **IEEE 39** | **0.7060** | **0.7060** | **0.01** | **certified_optimal** | n/a |
| IEEE 57 | **10.36** (new best) | 6.21 | 40.08 | cycle | **MISSING PATTERN, inflation=0** |

IEEE 39 hit the target (gap ≤ 5% → actual 0.01%). IEEE 14 and 57 did NOT certify, but
the `_diagnose_stall` verdict on both flipped from "RELAXATION GAP" (the pre-OBBT
diagnosis) to "MISSING PATTERN" with **0.00 inflation on every pattern**. The McCormick
slack that had been the LB ceiling is gone; the remaining cap is the CCG cycle-halt
after 3–4 patterns.

**Revised next escalation** (the bottleneck class changed — relaxation tightening is no
longer the right lever for IEEE 14/57):

1. **Pattern-source diversification** (new — was not in original plan, see
   `DECOMP_theoretical_strategies.md` §6 "Strategy 4"). The relaxation is now exact; the
   master just needs to see more dispatch patterns. Cheap experiment: at cycle-halt,
   query `oracle.solve_plain` at 2–3 perturbed `b` points and add those `u_j` as KKT
   blocks. If LB rises, the bottleneck is confirmed pattern-side and we iterate.
2. **Iterated / per-node OBBT** (Strategy 1 escalation, `DECOMP_theoretical_strategies.md`
   §6 "Strategy 5"). Sharpens μ bounds inside the spatial B&B using node-local fixings.
   Useful if root OBBT misses tightening visible only at deeper nodes — pairs well
   with Strategy 4 for cases where added patterns introduce new μ vars to OBBT.
3. **Verify cycle-halt cause first** (cheap sanity check). `_diagnose_stall` reports
   "MISSING PATTERN" but the master MIP also hit its time limit at 60–40% gap. Confirm
   the halt is on `_pattern_key(u_k) ∈ seen` and not on a master-MIP timeout that
   stopped pattern discovery prematurely — quick `grep` on the verbose log.
4. **Shared-`b` RLT** (`DECOMP_theoretical_strategies.md` §3) and **Balas disjunctive
   hull** (§4) are **demoted** to "needed only if a relaxation gap re-emerges" — e.g.
   if `bilinear_exact` is dropped for runtime reasons on a larger grid, or if per-node
   OBBT regresses. With inflation=0 they have nothing to tighten.

**Companion docs to read in order before resuming**: `DECOMP_state.md` (current state +
results table), this offload doc (§7 above for the pivot), then
`DECOMP_theoretical_strategies.md` §6 for the post-OBBT strategy list.

---

## 8. Hand-off summary (paste into a fresh session)

> Strongdual produces VALID LBs (IEEE39 ~6.8%) but is loose. `bilinear_exact=True`
> (Gurobi NonConvex MIQCP) is exact + valid but spatial B&B doesn't scale on IEEE 39+.
> Next step: implement **root OBBT** on `μ_p`/`μ_m` to provably tighten their UBs via
> McCormick-LP-relaxation auxiliary LPs. Cheap, valid by construction, orthogonal to
> bilinear handling. Spec + code sketch in `DECOMP_OBBT_offload.md`. Validate first
> via `_check_strongdual_valid.py` at known CE `b_BS` (must stay feasible), then
> measure LB on IEEE 14/39/57. Estimated 1–2 weeks. If OBBT alone isn't enough,
> escalate to shared-`b` RLT (`DECOMP_theoretical_strategies.md` §3).

---

## 9. Files / scripts to reuse

- `_check_strongdual_valid.py` — extend args slot 6 to accept "obbt". Runs `debug_fix_b`
  at a known CE; correctness gate.
- `_smoke_strongdual.py` — extend args slot 6 likewise. Full smoke run with B&S hint.
- `_diagnose_stall` (already in `uc_decomp_4b.py`) — fires at cycle-halt; should show
  reduced inflation post-OBBT.
- `_verify_strongdual.py` — dual_obj algebra regression; unchanged.
- `_measure_flow_duals.py` — re-run after OBBT to compare observed `μ` vs the new
  OBBT-derived UB.

---

## 10. Commit-message template (when done)

```
feat(decomp): root OBBT on flow duals for tighter valid LB

- Add `obbt`, `obbt_safety`, `obbt_iter` ctor args to UCDecomp4b.
- Implement `_obbt_root(m, master_vars)`: builds McCormick LP relaxation,
  maximizes each μ_p / μ_m per free (ell, t), tightens UBs.
- Called once after seeding, before main loop, when obbt=True.
- Provably valid: bounds derived from master's own constraint set.
- Validated at b_BS on IEEE 14/39/57 via _check_strongdual_valid.py.
- Smoke results (IEEE 39, exact+OBBT): LB ..., time ..., certified ...

See: DECOMP_OBBT_offload.md, DECOMP_theoretical_strategies.md §2.
```
