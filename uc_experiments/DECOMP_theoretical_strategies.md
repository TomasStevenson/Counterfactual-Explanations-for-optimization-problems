# DECOMP — Three theoretical strategies for a tighter valid lower bound

*Generated 2026-05-30. Companion to `DECOMP_state.md` ("Theoretical research directions").*

## 1. Context — what we need and why these three

`DECOMP_state.md` records the LB-stagnation diagnosis: the strong-duality reformulation
(`comp_mode="strongdual"`) gives a *valid* lower bound (`master_LB ≤ F*`) but loose, because
the McCormick envelope on the bilinear term `b·μ` lets the master *inflate* a pattern's
dispatch cost by exactly the oracle CE gap (`vd − vp`) and pass the optimality cut at a
non-CE `b_k`. The exact NonConvex formulation (`bilinear_exact=True`) is correct but Gurobi's
generic spatial branch-and-bound on `b·μ` does not scale to IEEE 39+.

A **valid** LB requires the master to be a *relaxation* of "foil ≤ v(u^j, b)". Equivalently,
the master must represent `v(u^j, b)` from **above** (an over-approximation). Standard
Benders/dual-vertex generation gives a *lower* approximation of `v` and is therefore
**invalid in our LB direction** (see `DECOMP_state.md` for the retraction). The only valid
tightening lever is the way the bilinear `b·μ` is represented. The three strategies below
are the most actionable concrete improvements over single-envelope McCormick that **preserve
the valid-LB direction**:

| # | Strategy | Validity preserved? | Effort | Likely gain |
|---|---|---|---|---|
| 1 | **OBBT** (Optimization-Based Bound Tightening) inside spatial B&B | Yes — bounds derived from master's own constraints | Low (~1–2 wk) | Speeds up `bilinear_exact`; modest envelope tightening |
| 2 | **RLT** (Reformulation–Linearization Technique) with shared `b` | Yes — multiplies *valid* constraints | Medium (~2–4 wk) | Tighter than single McCormick by construction |
| 3 | **Balas disjunctive convex hull** of `foil ≤ v(u^j,b)` | Yes — convex hull of union (NOT vertex selection) | High (~1–3 mo) | Theoretically the tightest valid LB short of exact |

The three layer naturally: OBBT is cheap and orthogonal (combines with anything); RLT adds
static valid inequalities that tighten the formulation everywhere; Balas is the rigorous
endgame. The recommended sequence is bottom-up.

---

## 2. Strategy 1 — OBBT (Optimization-Based Bound Tightening)

**Idea.** Before (or during) the master's spatial B&B, tighten the variable bounds on each
`μ_p[ell,t]`, `μ_m[ell,t]` by *solving auxiliary LPs* that maximize/minimize each `μ` over
the master's current constraint set. The tighter `μ` box ⇒ tighter McCormick envelope at
every B&B node ⇒ fewer branches needed for `bilinear_exact` to certify.

**Why valid.** The OBBT LP is constructed as a **valid relaxation** of the master MIQCP
(e.g., the McCormick LP relaxation with integrality dropped). Any feasible point of the
exact MIQCP is also feasible in this relaxation, so

```
μ_p[ell,t]_optimal  ≤  max{ μ_p[ell,t] : (b, μ, x, …) feasible in LP relaxation }
```

The right-hand side is a **provable upper bound** on `μ_p[ell,t]` at any optimum.
Tightening `UB ← that max` cannot exclude any optimal MIQCP solution ⇒ LB stays valid.
This is the same logic that makes spatial B&B preprocessing safe in MINLP solvers
(BARON, Couenne, SCIP, Gurobi internals).

**Critical distinction vs the heuristic `mccormick_mu_factor` (Option A).** The heuristic
factor uses *observed* dual values at a few `b` samples × a safety multiplier — not provably
valid, requires re-validation at known CEs after every change. OBBT is the *principled*
version: the bound is derived from the master's own constraints, so it's provably valid
*by construction*.

**Variants.**

- **Root OBBT (recommended start)** — run once before the main solve. Cheap (one pass of
  `2 × nFree × T` LPs). For IEEE 39: ~720 LPs of moderate size, minutes total.
- **Per-node OBBT** — re-tighten at each spatial B&B node. Much more expensive, but each
  bound is tighter (uses node-local cuts/fixings). Reserve for cases where root OBBT
  isn't enough.
- **Probing OBBT** — fix `b[ell] = bL[ell]` or `bU[ell]` and re-OBBT `μ`; useful for
  understanding the boundary behavior.

**Expected effect.** The McCormick width at b in segment of width `Δb` is bounded by
`Δb · (μ_UB − μ_LB) / 4`. Halving `μ_UB` halves the envelope (and the spatial-B&B search
volume per node). For our problem `dev/_measure_flow_duals.py` showed observed `μ` is 16–81×
smaller than the provable `M_mu`; OBBT should recover most of that gap *validly* — i.e.
what `mccormick_mu_factor` *almost* did, but provably correct.

**Implementation sketch** (`uc_decomp_4b.py`):

```python
def _obbt_tighten_mu(self, m, master_vars, time_limit_per_lp=2.0):
    """Tighten mu_p / mu_m bounds per free (ell, t) via root OBBT.
    Uses the McCormick LP relaxation of the master (NonConvex disabled,
    integrality relaxed) as a VALID relaxation."""
    # 1. Build LP relaxation (m.relax() with NonConvex=0 + McCormick of b·μ
    #    explicitly added — DO NOT use the exact bilinear constraint here).
    # 2. For each free ell, t:
    #       max μ_p[ell,t] → new UB
    #       min μ_p[ell,t] → new LB (≥ 0)
    #       same for μ_m
    # 3. Apply tighter bounds to the master m.
    # 4. (Optional) iterate until no bound improves significantly.
```

**References.**

- Belotti, Cafieri, Lee, Liberti (2010), "Feasibility-based bounds tightening via fixed
  points" — foundational OBBT theory.
- Puranik, Sahinidis (2017), "Domain reduction techniques for global NLP and MINLP
  optimization" — survey including OBBT variants.
- Gleixner et al. (SCIP papers) on propagation + OBBT for MINLP.
- The `bilinear_exact` knob already in `uc_decomp_4b.py` is what OBBT plugs into.

---

## 3. Strategy 2 — RLT (Reformulation–Linearization Technique) with shared `b`

**Idea.** Generate valid inequalities by multiplying pairs of valid bound constraints and
substituting bilinear products with auxiliary variables. The classical first-order RLT
on `{b ∈ [bL, bU], μ ∈ [0, M]}` reproduces McCormick. **Higher-order RLT** and
**RLT exploiting shared variables** give strictly tighter relaxations.

**Why it matters here.** Our flow term in `dual_obj` is

```
−Σ_{ell free} Σ_t b[ell] · (μ_p[ell,t] + μ_m[ell,t])
```

with `b[ell]` **shared across all `t`** and across both `μ_p` and `μ_m`. The convex hull
of the *joint* bilinear set

```
{ (b[ell], μ_p[ell,1..T], μ_m[ell,1..T], w_p, w_m) :
      w_·[ell,t] = b[ell] · μ_·[ell,t]  ∀ t,
      b[ell] ∈ [bL,bU], μ_·[ell,t] ∈ [0, M] }
```

is **strictly tighter** than the sum of `2T` independent rectangular McCormick envelopes.
For a fixed `b[ell] = b*`, each `w_·[ell,t] = b* · μ_·[ell,t]` is a rotated half-line —
the joint hull captures the *linear combination across t* (with the same scaling factor
`b*`) that single-term McCormick misses.

**Validity.** All RLT inequalities are obtained by multiplying valid constraints (e.g.,
`(b − bL) ≥ 0` with `μ ≥ 0` yields `b·μ − bL·μ ≥ 0`, i.e. `w ≥ bL·μ`, which is McCormick).
Higher-order products give new valid inequalities — never invalid.

**Concrete shared-`b` RLT inequalities to add (sketch).**

1. *Single-`b` aggregated McCormick* per free line ell:
   `Σ_t w_p[ell,t] ≥ bL · Σ_t μ_p[ell,t]` (and the three sister McCormick faces, summed
   over `t`). This is a single inequality per line per face, but it captures the shared
   `b` aggregation that per-`t` McCormick doesn't.
2. *Two-term RLT* on `b · (μ_p + μ_m)` per `(ell, t)`: define `u = μ_p + μ_m`, multiply
   `(b − bL)·u`, etc.
3. *RLT on auxiliary symmetric terms*: e.g., `(b − bL)·(b − bU) ≤ 0` (since `b` is between
   bounds) gives `b² ≤ (bL+bU)·b − bL·bU` — useful if any `b²` terms appear after
   substitution.

The full convex hull of the shared-`b` bilinear set has a known semidefinite description
(Anstreicher–Burer 2010) and a finite linear description via RLT for special cases
(Tawarmalani–Sahinidis 2002).

**Expected effect.** Tighter LP relaxation at the root and at every node ⇒ smaller LB gap
even *without* spatial branching, and faster `bilinear_exact` if used in combination.

**Implementation sketch.**

- Add a helper `_rlt_shared_b_inequalities(m, master_vars, j)` called from the strongdual
  block (alongside the existing per-(ell,t) McCormick faces).
- Inequalities are static (don't depend on master solution), added once per pattern j.

**References.**

- Sherali, Adams (1990, 1999) — original RLT papers.
- Sherali, Tuncbilek (1992) — RLT for polynomial programming.
- Anstreicher, Burer (2010), "Computable representations for convex hulls of low-dimensional
  quadratic forms".
- Tawarmalani, Sahinidis (2002) "Convexification and Global Optimization in Continuous and
  Mixed-Integer Nonlinear Programming" — convex envelopes of bilinear with shared
  variables.

---

## 4. Strategy 3 — Balas disjunctive convex hull of `foil ≤ v(u^j, b)`

**Idea.** The set `{(b, foil) : foil ≤ v(u^j, b)}` is a **union of half-spaces**, one per
dual vertex `y_v` of the dispatch LP:

```
{(b, foil) : foil ≤ v(u^j,b)} = ⋃_v { (b, foil) : foil ≤ y_v · rhs(b) }
```

Balas's disjunctive programming framework provides a **convex hull of a union of polyhedra**
via an extended formulation with one disaggregated copy per disjunct. This gives a
*convex* (LP-representable) over-approximation of the non-convex union — which is exactly
what a valid LB needs.

**Why this is different from (and avoids) the failed dual-vertex selection.** Naive
"pick one vertex" disjunction with `Σ δ_v = 1` and `foil ≤ Σ δ_v · g_v(b)` would
*restrict* the master to the *intersection* of feasibilities of subsets — wrong direction,
invalid LB. Balas's convex hull is the *union*'s convex hull = the smallest convex set
containing all disjuncts. The master then operates in this convex hull, which CONTAINS the
true non-convex set ⇒ relaxation ⇒ valid LB. The key is that the disaggregated `b^v`,
`foil^v` per disjunct allow *any convex combination* of disjunct points, not just selection
of one.

**Extended formulation (per pattern j).**

```
foil = Σ_v foil^v
b    = Σ_v b^v
1    = Σ_v λ_v                          (λ_v ≥ 0)

For each disjunct v:
    foil^v ≤ y_v · rhs(b^v)                  (the disjunct's inequality, in b^v)
    bL · λ_v ≤ b^v[ell] ≤ bU · λ_v           (b^v lives in segment scaled by λ_v)
    foil^v ≥ −Big · λ_v                       (basic disaggregation bound)
```

Project (foil, b) out — the projection is the convex hull of the union. Tractable when
the number of *generated* vertices is small (column generation on demand). Note `y_v·rhs(b^v)`
is **affine in `b^v`** because `y_v` is a constant vertex.

**Vertex generation on demand (Benders-style outer loop, BUT in the right direction
because we're describing the convex hull, not selecting from a partial set).**

```
loop:
    solve master with current vertex set V  → b*, foil*
    solve dispatch LP at (u^j, b*) → get optimal dual y*  → new vertex
    if y* ∈ V: stop                          (convex hull is sufficient)
    else: add y* to V; rebuild Balas hull; repeat
```

Finite termination: dispatch LP has finitely many dual vertices.

**Validity at every iteration.** Unlike the failed approach: each finite `V` gives the
convex hull of the *vertices in V's* half-spaces, which **contains** the convex hull of
the true union (the master sees a *larger* feasible region than truth) ⇒ relaxation ⇒
LB valid. As `V` grows, the convex hull shrinks toward the truth. So even with partial
`V`, LB stays valid (just loose).

Wait — re-derivation: the convex hull of the union over `V` is the convex hull of fewer
disjuncts, which is *smaller* than the convex hull of the union over all vertices (the
truth). Smaller relaxation = *fewer* feasible points = stronger constraints = master min
≥ truth = invalid?? Need to think carefully here.

**Resolution (the subtle point):** the union over fewer disjuncts is a subset of the
union over all (each removed disjunct removes some half-space, and the union loses points
covered only by removed disjuncts). The convex hull of a smaller set is smaller. So
Balas with partial `V` gives a *smaller* feasible region than truth → restriction →
invalid LB direction. **The same hazard as naive dual-vertex.**

**Therefore Balas only gives a valid LB once the vertex set is provably complete.**
The on-demand generation needs to certify completeness *before* using the LB. In
practice: keep generating until the dispatch dual at `b*` returns a `y* ∈ V` (current
vertex already in set), which proves `v(u^j, b*) = max_{v∈V} y_v · rhs(b*)` *at b**;
LB is then valid *at that solution*. This is similar to Bender's-style convergence
proofs but stricter. Theoretical work needed to verify this gives a valid LB at every
master iteration (or only at convergence).

**Alternative (and simpler): use Balas as an *upper bound* on `v`** (over-approximate v
by the convex hull from above) — but constructing such an upper bound from vertices is
harder than the standard lower one, and may degenerate to McCormick.

**Honest takeaway.** Balas is the rigorous endgame for an *exact* representation, but
the validity-at-partial-V hazard is real and matches the dual-vertex retraction. This
strategy is a research project: derive a Balas variant whose *partial generation*
preserves the over-approximation property (e.g., by adding a "rest of the polyhedron"
slack disjunct).

**References.**

- Balas (1985, 1998), "Disjunctive programming: properties of the convex hull of feasible
  points".
- Balas, Ceria, Cornuéjols (1993, 1996) — lift-and-project for 0-1 IPs.
- Vielma (2015), "Mixed integer linear programming formulation techniques" — modern
  survey including disjunctive formulations.
- Conforti, Cornuéjols, Zambelli, "Integer Programming" — textbook treatment of
  disjunctive cuts.

---

## 5. Recommended sequence

1. **Implement OBBT first** (Strategy 1, see `DECOMP_OBBT_offload.md`). Cheap, orthogonal,
   provably valid. Layers on top of `bilinear_exact` to speed up IEEE 39 spatial B&B.
2. **Add shared-`b` RLT inequalities** (Strategy 2). Static valid cuts; tighten the
   formulation directly.
3. **Re-evaluate the gap.** If OBBT + RLT bring IEEE 14/39/57 close to certification, stop.
4. **Only if needed: Balas disjunctive hull** (Strategy 3). Resolve the
   validity-at-partial-V issue first; this is months of research-grade work.

Cross-cutting discipline (from `DECOMP_state.md`): every reformulation change must pass
`_check_strongdual_valid.py` at a known CE `b_BS` before its LB can be trusted. A tighter
LB that certifies above F* is worse than a loose one.

---

## 6. Status after Strategy 1 (root OBBT, 2026-05-30) — diagnostic flip

Root OBBT was implemented + validated + smoke-tested in one session
(`DECOMP_state.md` §"Root OBBT — IMPLEMENTED + VALIDATED + CERTIFIES IEEE 39"). Headline:

| Grid | F_opt | LB | gap% | term | `_diagnose_stall` verdict |
|------|------:|---:|-----:|------|---------------------------|
| IEEE 14 | 2.3794 | 0.9373 | 60.61 | cycle | **MISSING PATTERN, inflation=0** |
| **IEEE 39** | **0.7060** | **0.7060** | **0.01** | **certified_optimal** | n/a |
| IEEE 57 | **10.36** (new best) | 6.21 | 40.08 | cycle | **MISSING PATTERN, inflation=0** |

**The bottleneck class changed.** Before OBBT, `_diagnose_stall` on all three grids
returned **RELAXATION GAP** — the master inflated a pattern's dispatch cost by exactly
the oracle CE gap (`vd − vp`), passing a cut it should violate. That diagnosis motivated
this whole document (RLT, Balas — all relaxation tighteners).

After `bilinear_exact + OBBT`, inflation is **0.00 on every pattern on every grid**.
IEEE 14/57 still don't certify, but **not because the relaxation is loose** — because
the CCG loop cycle-halts after 3–4 patterns. The master's optimistic `b_k` satisfies
every existing pattern's TRUE dispatch ≤ master_foil; the pattern that would beat the
foil at `b_k` is simply not in the cut set.

**Implications for Strategies 2 and 3.**

- **Strategy 2 (shared-`b` RLT)**: was motivated by the McCormick inflation. With
  `bilinear_exact + OBBT` there *is* no McCormick — RLT against the exact bilinear set
  is still mathematically valid but has nothing to tighten. **Demoted from "next step"
  to "optional fallback"** — useful only if a future relaxation-gap reappears (e.g., if
  per-node OBBT regresses or `bilinear_exact` is dropped for runtime reasons on a
  larger grid).
- **Strategy 3 (Balas disjunctive hull)**: same demotion. The validity-at-partial-V
  hazard makes this research-grade anyway; with the relaxation no longer the
  bottleneck, the urgency drops further.

**Where the urgency moves instead.** Pattern coverage / cycle-halt escape. New strategies
(not in §§2–4 above, because the original document was scoped to relaxation tightening):

- **Strategy 4 — Pattern-source diversification.** Generate additional KKT patterns
  beyond the plain-optimum at the current `b_k`: e.g., k-shortest commitments, perturbed
  cost solves, or multi-objective tilts on the plain UC. The relaxation is now exact;
  the master simply needs to know about a wider set of dispatch behaviors. Cheap
  experiment first: at cycle-halt, fire `solve_plain` at 2–3 nearby `b` perturbations
  and add those `u_j` patterns; see whether LB rises.
- **Strategy 5 — Per-node OBBT inside spatial B&B.** Strategy 1's escalation rung:
  re-tighten `μ` UBs at each spatial-B&B node using local fixings/cuts. Sharper than
  root OBBT, but more expensive. Useful if root OBBT misses tightening that only
  becomes visible at deeper nodes (e.g., after `b` is branched into a smaller range).
- **Strategy 6 — Hybrid: bigM master + bilinear_exact verifier.** If cycle halt blocks
  pattern coverage, an alternative is to run a bigM `comp_mode` master (which doesn't
  cycle-halt the same way because z's give Gurobi more branching flexibility) and use
  `bilinear_exact` only as a final certificate at the candidate CE. Loses the LB but
  potentially closes the gap from above. Complementary to the LB direction.

**Updated recommended sequence (post-OBBT).**

1. **Strategy 4 first** (pattern-source diversification) — cheap, directly attacks the
   bottleneck the diagnostic flip identified.
2. **Strategy 5** (per-node OBBT) if Strategy 4 stalls — tightens the spatial-B&B
   search inside the master MIP itself.
3. **Strategies 2 / 3** demoted to "needed only if a relaxation gap re-emerges".
4. **`/code-review` cycle-halt logic** in `uc_decomp_4b.py:run()` to confirm it's
   really the cycle-detection-on-`u_k` that's halting and not e.g. the master MIP
   timing out without a certified incumbent — a quick check before investing in
   Strategy 4.

---

## 7. Strict vs tolerant CE — the UB axis is not free (2026-05-30, hybrid experiment)

Tested **Strategy 6 (hybrid)** on IEEE 14 (`dev/_hybrid_obbt.py`): Stage A runs bigM to
drive the UB, Stage B runs strongdual+bilinear_exact+OBBT (hinted by Stage A's CE) for
the valid LB. The intent: a better incumbent both lowers the UB *and* tightens the
master's `F ≤ F_hint` cap, which can only raise ObjBound (smaller feasible region ⇒
higher min). Result was instructive in an unexpected way:

- **Stage A (bigM)** found **F=1.9061** as expected (vs B&S 2.3823) — the docs' "20%
  improvement". ✓
- **Stage B (strongdual+OBBT)** returned **`time_limit_no_incumbent`, LB lost (0.0)**. ✗

**Root cause — bigM's CE is sub-strict.** Oracle at the bigM `b`:
`v_plain=830689.71`, `v_foil=830689.94` ⇒ **`v_foil − v_plain = +0.22 > 0`**. A CE
requires `v_foil ≤ v_plain` (the foil must be the cheapest commitment at `b`). bigM
accepts this `b` because `0.22 ≪` the relative CE tolerance (`max(eps_weak,
1e-4·|v_plain|) ≈ 83`), but it is **not a strict CE** — the plain optimum is 0.22
cheaper, so the foil is not optimal there. The exact strongdual master enforces
`foil ≤ dispatch` *exactly*, so under a hard `F ≤ 1.9061` cap the feasible region is
"strict CEs with F ≤ 1.9061", which is empty/tiny (the minimal strict CE is > 1.906).
The MIQCP correctly finds no incumbent, and the LB attempt collapses.

**Consequence — the two bounds measure different objects and are NOT combinable:**

| Quantity | Definition | IEEE 14 |
|----------|-----------|--------:|
| bigM "CE" | tolerant CE (allows ~0.22 boundary slack) | 1.906 |
| strongdual exact CE | **strict** CE (`v_foil ≤ v_plain` exactly) | 2.379 |
| OBBT LB | valid lower bound | 0.937 |

The minimal **strict** CE lies in `(1.906, 2.379]`; the valid certified interval is
`[0.937, 2.379]`. **Part of bigM's "20% improvement" is a CE-tolerance artifact**, not a
real gain against the strict problem. This must be stated honestly in any writeup: the
publishable certified result uses strict CEs.

**Implications for Strategy 6.** The hybrid only works if Stage A returns a *strict* CE.
Options:
- (a) **Tighten bigM's CE tolerance** (`eps_weak`, `EPS_REL_CE`) so it returns an honest
  strict CE. Expected effect: bigM's UB rises toward the strict minimum (its 1.906 edge
  shrinks), but the result becomes a valid UB feedable to Stage B.
- (b) **Project the bigM CE onto strict feasibility** before Stage B: small local solve
  to push `v_foil ≤ v_plain` exactly, then use that `b` (slightly higher F) as the hint.
- (c) **Abandon the hard `F ≤ F_hint` coupling** for the UB and only use the bigM CE as a
  warm-start incumbent (not a cap). Loses the LB-tightening benefit but keeps the master
  solvable.

**Bottom line.** The UB and LB are coupled through the `F ≤ F_hint` cap, but exploiting
that coupling requires a *strict* hint, and the exact MIQCP master is the binding
constraint either way (it can't find a strict-CE incumbent under a tight cap in budget).
This points back at **master-MIP scaling** (Strategy 5 / Gurobi tuning / better
warm-starts) as the real lever, with strict-CE discipline on the UB side.
