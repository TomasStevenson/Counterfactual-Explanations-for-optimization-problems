# DECOMP — Formulation, Optimality Guarantee, and Why the Upgrades Work

*Authoritative theory note for the DECOMP (Column-and-Constraint Generation) method for
counterfactual explanations of Unit Commitment. Companion to `DECOMP_state.md` (current
state + results), `DECOMP_OBBT_offload.md` (OBBT design), and
`DECOMP_theoretical_strategies.md` (relaxation-tightening landscape). Last updated 2026-06-05.*

This document answers two questions precisely:

1. **Why does the method provably certify the globally optimal counterfactual?**
2. **Why does each upgrade actually make it faster (or make it correct)?**

The guiding principle throughout is a single pair of invariants:

> **(LB)** every number we report as a *lower bound* is `≤ F*` (the true optimum), and
> **(UB)** every number we report as a *feasible counterfactual* is an oracle‑verified
> point with cost `≥ F*`.
>
> When `UB − LB ≤ ε` the counterfactual is **certified ε‑optimal**. Every upgrade below is
> justified by (a) it preserves these invariants, and (b) it shrinks `UB − LB` faster.

---

## 1. The problem: a minimal counterfactual explanation for Unit Commitment

### 1.1 The factual model

A Unit Commitment (UC) instance over a horizon `t = 1..T` chooses binary commitments
`u_{g,t} ∈ {0,1}` and a continuous dispatch `x` (generation `p`, flows `f`, phase angles
`θ`, load shed, renewable curtailment, slacks) to minimise total cost subject to the network
+ generation constraints. The decision-relevant parameters here are the **transmission line
flow limits** `b_ℓ = fmax_ℓ`. Write the UC optimal value as a function of the limits:

```
v_plain(b) = min_{u, x}  c(u, x)        s.t.   (u, x) ∈ UC(b).               (PLAIN)
```

### 1.2 The foil and the counterfactual question

A **foil** is a desired qualitative outcome expressed as an extra constraint on the UC — in
our experiments, *"total emissions ≤ (1 − α)·E_factual"* (`make_emissions_foil_4b`, α = 0.10).
The foil‑constrained optimum is

```
v_foil(b) = min_{u, x}  c(u, x)   s.t.   (u, x) ∈ UC(b),  Emissions(u,x) ≤ (1−α)E_fac.  (FOIL)
```

Because (FOIL) is (PLAIN) with an extra constraint, `v_foil(b) ≥ v_plain(b)` for every `b`.
The foil outcome is **optimal for the UC** — i.e. the system operator would actually choose
it — exactly when the extra constraint is *not binding on cost*:

```
foil is UC-optimal at b   ⟺   v_foil(b) = v_plain(b)   ⟺   v_foil(b) ≤ v_plain(b).   (CE)
```

The **counterfactual explanation** is the *smallest change to the line limits* that makes the
foil optimal. With per‑line weights `w_ℓ > 0`, an expansion‑only, free‑line index set `L_free`
(`b_ℓ ≥ b0_ℓ`, others fixed at `b0`):

```
F* = min_b  F(b) := Σ_{ℓ∈L_free} w_ℓ (b_ℓ − b0_ℓ)        (L1 change)
        s.t.   v_foil(b) ≤ v_plain(b).                                          (CE-OPT)
```

This is the object we certify. The answer *"increase these few lines by these amounts and the
emissions‑reduced schedule becomes cost‑optimal"* is the explanation.

### 1.3 Why it is hard: a bilevel problem with an inner minimisation in a constraint

`v_plain(b) = min_u min_x c(u,x)` is a **minimisation inside a constraint**. Expand (CE):

```
v_foil(b) ≤ v_plain(b) = min_{u feasible} v(u, b),
   where  v(u, b) = min_x c(u, x) s.t. (u,x) ∈ UC(b)    is the dispatch LP at fixed u.
```

Equivalently, the foil must be no more expensive than **every** commitment:

```
v_foil(b) ≤ v(u, b)     for ALL feasible commitments u.                          (∀u)
```

There are `2^{(#gens)(T)}` commitments (e.g. `2^{144}` on IEEE 14), so (∀u) is an enormous
family of constraints. Two facts make it tractable and are the seeds of the whole method:

* **F.1 (relaxation by subset).** Enforcing (∀u) only for a *subset* `J = {u^1,…,u^k}` of
  commitments yields a problem with *fewer* constraints, hence a *larger* feasible set, hence
  a *smaller* minimum. **The subset problem's optimum is a valid lower bound on `F*`.** Adding
  patterns can only raise it. This is Column‑and‑Constraint Generation (Yue et al. 2019).
* **F.2 (each inner value is a convex LP).** For *fixed* `u`, `v(u,b)` is a linear program in
  `x`; its value is a well‑behaved (piecewise‑linear concave) function of `b` that we can
  encode exactly with LP duality. That encoding is the strong‑duality master block (§3).

---

## 2. The CCG master and the certification loop

### 2.1 The master (relaxation with a finite pattern set `J`)

```
M(J):   min_{b, (u_f,x_f), y}   F(b)
        s.t.   (u_f, x_f) ∈ UC(b) with the foil constraint               (foil block)
               c(u_f, x_f) ≤ v(u^j, b)        for each u^j ∈ J            (optimality cuts)
               b0_ℓ ≤ b_ℓ ≤ bU_ℓ  (ℓ∈L_free),  b_ℓ = b0_ℓ otherwise.
```

`u_f, x_f` are the foil's *own* commitment and dispatch (master variables). Because the master
minimises `F` and is free to pick the cheapest feasible foil, the foil block contributes
`c(u_f,x_f) = v_foil(b)`, and the cuts enforce `v_foil(b) ≤ v(u^j,b)` for the sampled `u^j`.

**Lemma 1 (valid LB).** `opt M(J) ≤ F*` for every finite `J`.
*Proof.* The feasible set of (CE‑OPT) is `{b : v_foil(b) ≤ v(u,b) ∀u}`; the feasible set of
`M(J)` drops all constraints with `u ∉ J`, so it is a superset. Minimising the same objective
over a superset gives a value `≤ F*`. ∎

### 2.2 The loop (lazy pattern generation)

```
LB ← 0;  UB ← +∞;  J ← seeds (§5.1)
repeat:
    (b_k, foilcost_k) ← solve M(J)              # LB_k = ObjBound of M(J)
    LB ← max(LB, LB_k)                          # running max — see §6.3
    vp ← v_plain(b_k);  vd ← v_foil(b_k)        # ORACLE: exact UC solves at b_k
    if vd ≤ vp (oracle-verified CE):  UB ← min(UB, F(b_k))   # feasible -> upper bound
    if UB − LB ≤ ε:  return  CERTIFIED (b*, F*≈UB)
    u_k ← argmin_u v(u, b_k)  =  plain-optimal commitment at b_k     # the violated pattern
    J ← J ∪ {u_k}                               # add the cut that b_k violates
```

`u_k` is the commitment that is *cheaper than the foil at `b_k`* — precisely the constraint of
(∀u) that `b_k` violated. Adding it cuts `b_k` off and raises the next `LB_k`.

### 2.3 Optimality guarantee

**Theorem (finite ε‑certification).**
*The loop maintains `LB ≤ F* ≤ UB` at all times, and terminates with `UB − LB ≤ ε`.*

* `LB ≤ F*`: Lemma 1 (each `M(J)` is a relaxation) + the bound representation in §3 is itself a
  valid lower‑bounding relaxation of each cut, so `ObjBound(M(J)) ≤ opt M(J) ≤ F*`. The
  running max preserves the tightest proven LB (§6.3).
* `F* ≤ UB`: every `UB` update is an **oracle‑verified** CE (`vd ≤ vp` from exact UC solves),
  i.e. a genuinely feasible point of (CE‑OPT), whose cost is `≥ F*`.
* Termination: there are finitely many commitments `u`; each iteration that does not certify
  adds a *new* `u_k` (it was violated, so not already in `J`). After at most `2^{(#gens)T}`
  additions `J` contains every binding pattern and `opt M(J) = F*`. In practice the binding set
  is tiny (a handful), and seeding (§5.1) front‑loads it.

The gap `UB − LB` has two distinct sources, and **the entire upgrade program is about driving
each to zero efficiently**:

* **pattern gap** — `J` is missing a commitment that binds at the optimum (`LB` too low because
  `M(J)` is too loose a *outer* relaxation). Fixed by *pattern coverage* (§5).
* **bound gap** — `M(J)` is correct but its non‑convex `b·μ` term and integer `u_f` are not
  solved to optimality in budget (`ObjBound` below `opt M(J)`). Fixed by *relaxation tightness*
  and *search* upgrades (§4, §6, §7).

---

## 3. Encoding the inner value exactly: strong duality (and the validity direction)

Each cut needs `c(u_f,x_f) ≤ v(u^j,b)` where `v(u^j,b) = min_x { c_x·x : A x ≥ r(b), x ≥ 0 }`
is the dispatch LP at fixed commitment `u^j` (the RHS `r(b)` is affine in `b`: the flow limits
appear as `−b_ℓ ≤ f_ℓ ≤ b_ℓ`). We represent `v` by **LP strong duality**:

```
v(u^j, b) = max_y { y · r(b) : yᵀA ≤ c_x, y ≥ 0 (on inequality rows) } = dual_obj(y, b).
```

The master adds the **dual variables `y`** (the LMPs `π`, flow duals `μ_p, μ_m`, etc.), the
**dual‑feasibility** constraints, and the cut in the form

```
c(u_f, x_f)  ≤  dual_obj(y, b)        with y dual-feasible.                         (CUT)
```

**Lemma 2 (strong duality is the *valid* representation; vertex selection is not).**
For any dual‑feasible `y`, weak duality gives `dual_obj(y,b) ≤ v(u^j,b)`. Hence any master
point satisfying (CUT) satisfies `c(u_f,x_f) ≤ dual_obj(y,b) ≤ v(u^j,b)` — the cut is **at
least as strong** as the true constraint, so it never admits an infeasible `b`: **the LB stays
valid.** And because strong duality is *attained* (`max_y dual_obj = v`), the master can always
reach the true value, so it excludes no genuine CE. ∎

> **Why not the cheaper "pick the best dual vertex" (Benders‑style) representation?** In
> two‑stage stochastic programming the inner value sits in the *objective*; a partial vertex
> set under‑estimates it and still yields a valid LB. **Here the inner value sits in a
> *constraint* `foil ≤ v`.** A partial vertex set *under‑estimates* `v`, making the cut
> `foil ≤ v_partial` *stronger* than truth → it over‑restricts the master → the master minimum
> goes *up* → an **invalid (too‑high) LB**. This is why we use the full strong‑duality block
> (an over‑approximation of `v` in the right direction), not vertex selection. (Recorded as the
> "RETRACTED dual‑vertex" result; it is the same hazard class as the `lp_const` and the
> over‑tight‑OBBT bugs — all three are "the LB silently certified above `F*`".)

The only nonconvexity left in (CUT) is the **bilinear product** `b_ℓ · μ_ℓ,t` (a *flow limit*
times *its own dual*), since both are master variables. Everything else is linear. §4 is about
this single term.

---

## 4. Handling the bilinear `b·μ`: McCormick (valid, loose) vs exact (valid, tight)

### 4.1 `strongdual` + McCormick — a valid lower‑bounding relaxation

Replace each product `w_{ℓ,t} = b_ℓ·μ_ℓ,t` by its **McCormick envelope** over the box
`b_ℓ ∈ [b0,bU]`, `μ_ℓ,t ∈ [0, M]`. The envelope is the convex hull of the bilinear surface on
the box; it **contains** the true product set, so the master feasible set grows, so `opt` drops
— still a valid LB (Lemma 1's relaxation argument extends). It is an LP/MILP (with `u_f`
integer), fast, but the envelope is loose where `b` and `μ` are both large → a *bound gap*.

### 4.2 `bilinear_exact` — the exact product, no relaxation gap

Write `w_{ℓ,t} = b_ℓ·μ_ℓ,t` as the **exact quadratic equality** and set Gurobi
`NonConvex=2` (spatial branch‑and‑bound). This is **exact**: the cut equals the true
strong‑duality constraint, the relaxation gap is **0**, the LB is the tightest possible for the
given `J`. The cost is that spatial B&B on `b·μ` is expensive on the larger grids. Upgrades §5–§7
exist to make this exact solve tractable.

> **Validity note.** McCormick keeps the master a valid *relaxation* (a returned `b_k` may not
> be an exact CE — the oracle re‑check in §2.2 still gates every UB, so the final answer is
> unaffected). `bilinear_exact` makes the master exact, so `ObjBound` is a tight valid LB.
> Both are validated by `_check_strongdual_valid.py` at the known CE `b_BS` (the master must
> stay feasible there with objective `F(b_BS)`).

---

## 5. Upgrade A — pattern coverage (attacking the *pattern gap*)

### 5.1 Seeding (`seed_patterns`) and interior seeding (`seed_interp`)

The LB only rises when a cut for a binding pattern is present. Instead of discovering patterns
one‑per‑iteration (and cycle‑halting after 1–2), we **seed** the plain‑optimal commitments at
the key line‑capacity vectors before the loop:

* corners `{b0, b_hat (a known CE, e.g. from Branch‑and‑Sandwich), bU}`, and
* `seed_interp = K` **interior** points `b0 + (k/(K+1))(bU−b0)`, `k=1..K`.

*Why it works (validity + speed).* Every seeded cut is a genuine constraint of (∀u), so adding
it only tightens a valid relaxation — never invalid. Interior seeds cover the `b`‑path between
"no change" and "max expansion", which is exactly where the optimum lives, so the master can no
longer settle at a non‑CE `b` that sits *between* the corner patterns' cuts (the "missing
pattern" stall). **Measured:** on IEEE 39 the two interior patterns (`sum_u = 48, 46`) were the
exact missing patterns — adding them moved IEEE 39 from a stalled gap to **certified 0.00%**.
Seeds dedupe automatically (`seen` set), so on grids where the plain commitment is constant
along the path (IEEE 57) interior seeding is a harmless no‑op.

---

## 6. Upgrade B — relaxation tightness and search (attacking the *bound gap*)

### 6.1 Root OBBT (`obbt`) — provably valid bound tightening

Optimization‑Based Bound Tightening shrinks each dual's box `μ_ℓ,t ∈ [0, M]` by *maximising
`μ_ℓ,t` over a valid relaxation `R` of the master* and replacing `M` with that maximum (plus a
safety margin). Tighter `μ` boxes shrink the McCormick envelope and Gurobi's spatial‑B&B box on
`b·μ` at every node → fewer branches.

**Validity (Lemma 3).** Let `S_exact` be the master's exact feasible set and `R ⊇ S_exact` the
McCormick‑LP relaxation. Then `max_R μ_ℓ,t ≥ max_{S_exact} μ_ℓ,t ≥ μ_ℓ,t` at every exact point.
Replacing the UB by `max_R μ + margin` therefore **cannot exclude any exact‑feasible point** —
the LB stays valid. The same argument tightens the *shared* `b_ℓ` bounds (b‑OBBT), the most
direct lever because `b_ℓ` is the spatial‑B&B's main branching variable. **Measured:** OBBT was
the breakthrough that made the IEEE 39 exact fixed‑`b` solve finish in seconds and certify; with
`bilinear_exact + OBBT` the McCormick inflation drops to **0 on every pattern** (the
diagnostic flipped from "relaxation gap" to "pattern/scaling‑bound").

### 6.2 The OBBT validity fix (a correctness upgrade, 2026-06-05)

OBBT is valid *in exact arithmetic*, but the **iterated** version (`obbt_iter > 1`) had a
**model‑state staleness** bug: the 2nd pass re‑derived bounds over a relaxation whose pending
tightenings had not been flushed, and applied the same shrink twice — pushing a `μ` UB *below*
the strong‑duality dual that the known CE `b_BS` requires. Result: the master became infeasible
at `b_BS` → an **invalid LB that could certify above `F*`** (caught by `_validate_all.py`,
IEEE 39). The fix is three‑layered, and the design lesson is general:

1. **`obbt_iter` default 1** — the conservative default; the 2nd pass never adds value once the
   iterate is correct (it correctly tightens 0).
2. **Corrected iterate** — force a model refresh each pass so OBBT operates on the truly
   tightened relaxation (no double shrink).
3. **Self‑validating guard** — compute the hint CE's *exact* per‑pattern dispatch duals once
   (`_solve_dispatch_lp`); after every pass, if any tightened `μ` UB drops below that dual (or
   `b_hint` leaves its box), **roll the pass back and stop iterating**. OBBT can then only ever
   stay conservative, with **no human gate** — essential for unattended HPC runs.

This is the concrete instance of the project‑wide rule: *a tighter LB is worthless — worse,
dangerous — if it can certify above `F*`; always re‑validate against a known CE.*

### 6.3 Running‑max LB and multistart (NonConvex bound variance)

A NonConvex MIQCP under a time limit proves a *different* (always valid) `ObjBound` on different
random seeds and on different (bigger) masters. Two cheap, provably valid tricks:

* **Running‑max LB:** report `LB = max_k ObjBound_k` across iterations. Each `ObjBound_k ≤ F*`
  individually (Lemma 1), so the max is `≤ F*` and never discards a better proven bound (a later,
  bigger master can prove a *weaker* bound in the same budget — observed on IEEE 14).
* **Multistart:** solve each master with several Gurobi seeds, keep the max `ObjBound`. Valid for
  the same reason; cheap (stops at the first seed that proves optimality); and **parallelisable**.

### 6.4 Analytic warm start (why the master needs it at all)

The bigM/strong‑duality master is *hostile to Gurobi's default incumbent heuristics*: at the
optimality cut's binding face the feasible region is effectively a single point, and cold B&B
explored 10⁵–10⁶ nodes with **zero** incumbents. We instead **compute** a feasible point: solve
the dispatch LP at the hint CE per pattern (`_solve_dispatch_lp`), read off the exact primal +
dual values (with the sign conventions matched to the master's KKT block), derive the
complementarity binaries analytically, and inject the whole thing as a MIP start (~0.2 s). This
is not a relaxation change — it just hands Gurobi the incumbent it cannot find itself, turning
"no feasible solution in 900 s" into "incumbent at `t=0`". The hint CE is also registered as the
initial `UB` (so `success=False` is structurally impossible when a hint is supplied).

### 6.5 Strict‑vs‑tolerant CE gating

UC costs are `~10⁶`, so a relative CE tolerance `~10⁻⁴·|v|` is `~80` in absolute terms — fine for
*reporting* a CE, but a `b_k` whose oracle gap is `≤ 80` may still hide a missing pattern that
beats the foil by that much. Refreshing the incumbent / objective cap / warm‑start hint to such a
`b_k` corrupts the next exact‑cut warm start and can push the cap below the true strict optimum.
So state is refreshed **only** on a *strict* CE (`vd − vp ≤ eps_ce_strict`), while tolerant `b_k`
are still reported and their patterns still added. This keeps the LB a lower bound on the minimal
**strict** CE (no false certification) and keeps warm starts valid.

---

## 7. Upgrade C — node‑OBBT: spatial branch‑and‑bound over the `b`‑box (the standardised solver)

### 7.1 The idea, and why we drive the tree ourselves

The single remaining bottleneck on the hard grids (IEEE 14/57) is solving the **exact** master
MIQCP's spatial B&B on `b·μ` to a tight `ObjBound`. "Per‑node OBBT" — re‑tightening `μ` using a
node's local `b`‑box — is the textbook escalation, but **Gurobi's callback API cannot tighten a
node's local variable bounds inside its own spatial B&B** (you can read the node relaxation,
not shrink `μ` within a subtree). So we **run the spatial branch‑and‑bound over the `b`‑box
ourselves** and call Gurobi only to *bound* each box. The "nodes" are our `b`‑boxes.

```
_solve_master_spatial_obbt:
   leaves ← { root box = current b-box };  UB ← incumbent F;  LB ← −∞
   repeat (best-first on the smallest box LB):
       box ← argmin_{open leaves} box.LB
       set b ∈ box;  (optional per-box OBBT);  ObjBound_box ← solve exact MIQCP (budget)
       box.LB ← max(box.LB, ObjBound_box)
       if box has a feasible incumbent cheaper than UB: update UB, b*
       if box.LB ≥ UB − ε:  prune
       else: split the widest free b_ℓ at its midpoint into two child boxes (inherit box.LB)
   global_LB ← min over leaf boxes of box.LB
```

### 7.2 Why it is valid (the load‑bearing proof)

**Theorem (spatial LB validity).** *At every stopping point,
`global_LB := min_{leaf boxes} ObjBound_box ≤ F*`.*
*Proof.* The boxes are an **exhaustive partition** of the root `b`‑box (every split divides a box
into two covering sub‑boxes; nothing is discarded except by pruning, which only removes regions
proven to contain no CE cheaper than the incumbent `UB ≥ F*`). The master optimum over the whole
box is `min over boxes of (optimum within the box)`, and each `ObjBound_box ≤ optimum within the
box` (Gurobi's guarantee for a minimisation). Taking the min over an exhaustive partition,
`min_box ObjBound_box ≤ min_box (optimum in box) = opt M(J) ≤ F*`. The bound holds for *any* set
of per‑box budgets, so it is valid whether the run certifies, exhausts its node cap, or is killed
early. ∎

Two implementation guards make this airtight in code:
* In node mode `run()` takes `iter_LB = global_LB` and **does not** `max` it with the last box's
  `m.ObjBound` (that single box's bound can exceed the true partition minimum → invalid).
* `m`'s `b/μ` bounds are snapshotted and restored, and the incumbent `b_k` is returned
  explicitly (Gurobi's `.X` reflects only the last box solved).

### 7.3 Why it speeds up (and where the speed actually comes from)

* **A sub‑box is *easier* to bound.** With `b_ℓ` confined to half its range, Gurobi's spatial
  relaxation of `b_ℓ·μ_ℓ` is tighter, so the box proves a *higher* `ObjBound` in the same time.
  **Measured (IEEE 14, 300 s/box):** the full root box proved `1.5588`; its half‑box child proved
  `1.5756` at the same budget. (IEEE 57: child `9.78` vs root `9.70`, and a deeper child `10.25`.)
* **Pruning eliminates whole regions.** A half‑box with no CE cheaper than the incumbent returns
  `ObjBound = +∞` and is removed, lifting the floor to the surviving boxes' level. **Measured:**
  one IEEE 14 half and one IEEE 57 half were pruned outright.
* **Best‑first raises the floor.** `global_LB` is pinned by the *lowest* unsplit leaf; best‑first
  always splits exactly that leaf, so `global_LB` rises monotonically with box count.
* **Per‑box OBBT is OFF by default**, by measurement: a single‑dimension `b`‑split barely moves
  the `μ`‑max (`μ_{ℓ',t}` depends on `b_{ℓ'}`, not the split line), so re‑running the full OBBT
  per box tightened **0** bounds at ~876 LPs of pure overhead. The root `μ` bounds are already
  valid for any sub‑box (`sub‑box ⊆ root ⇒ root μ.UB ≥ max over sub‑box`), so per box we keep
  root `μ` and only feed Gurobi the tighter `b`‑box — which is where the speed actually comes
  from.

### 7.4 Why this is the *standardised* solver

The same `node_obbt` switch + config runs on all three grids and adapts automatically:

| grid | behaviour | result (one config) |
|------|-----------|----------------------|
| IEEE 39 | certifies at box 1 — no split needed | **0.00 % (certified)**, F = 0.7060 |
| IEEE 57 | a few boxes; split + prune | gap 7.37 % (4 boxes), valid, CE 10.5578 |
| IEEE 14 | hardest; split + prune | gap 33.6 % (3 boxes), valid, CE 2.3724 |

It **certifies where the master is easy** (39) and **adds boxes only where the MIQCP is hard**
(14/57) — "doesn't hurt where not needed". And because the boxes are **independent**, the floor
can be raised by solving more of them *in parallel*, which is the cluster story.

### 7.5 Parallelism (the NLHPC argument)

`global_LB = min over boxes of ObjBound_box` is an **associative reduction over independent
subproblems**. Each box can be solved by a separate process with no communication; an external
aggregator takes the min of the returned `ObjBound`s and the min‑cost returned incumbent. On a
cluster (NLHPC Leftraru) this is a Slurm **job array**: N boxes → N jobs → `global_LB = min`.
The single‑machine results above understate the method precisely because they are sequential;
the certificate strength scales with the number of boxes you can afford to solve, and that is a
parallel resource. The best‑first split rule and the snapshot/restore are already in place; the
only new seam is *serialize a box → solve one box → collect its `ObjBound`*.

---

## 8. The validity invariants, in one place

Every component above is admissible only because it preserves these:

1. **LB ≤ F\*** — guaranteed by: subset‑relaxation (Lemma 1), strong‑duality over‑approximation
   in the correct direction (Lemma 2), McCormick/exact relaxations of `b·μ` (§4), OBBT bounds
   derived from a valid relaxation (Lemma 3) **with the self‑validating rollback guard** (§6.2),
   running‑max + multistart over valid `ObjBound`s (§6.3), and the spatial‑partition minimum
   (§7.2 Theorem).
2. **UB ≥ F\*** — every reported counterfactual is **oracle‑verified** by exact UC solves
   (`vd ≤ vp`), and only *strict* CEs refresh state (§6.5).
3. **Certificate = `UB − LB`** — reported honestly; "certified" is claimed only when
   `UB − LB ≤ ε` with both invariants intact. The regression gate `_validate_all.py` asserts the
   master never excludes a known CE on any grid — run it after *any* change to the formulation.

The three historical near‑misses (the `lp_const` over‑constraint, the dual‑vertex direction
error, and the iterated‑OBBT over‑tightening) were all violations of invariant (1) — *a tighter
LB that certified above `F*`*. Each is now blocked structurally (corrected formula, retraction,
self‑guard) and caught by the regression gate.

---

## 9. Summary: why it's correct, and why it's fast

* **Correct (optimality).** DECOMP is a Column‑and‑Constraint Generation scheme whose master is
  always an *outer relaxation* of the bilevel counterfactual problem (valid LB) and whose
  incumbents are always *oracle‑verified* feasible points (valid UB). The strong‑duality block
  encodes the inner UC value in the only direction that keeps the LB valid, and the bilinear
  term is handled by relaxations that are valid lower bounds (McCormick) or exact
  (`bilinear_exact`). The gap is a genuine optimality certificate.
* **Fast (the upgrades).** Speed = closing `UB − LB` per unit time, decomposed into:
  *pattern coverage* (seeding / `seed_interp`) removes the missing‑pattern gap; *relaxation
  tightness* (`bilinear_exact` + OBBT, with the validity fix) removes the McCormick gap and
  shrinks the spatial‑B&B box; *search* (analytic warm start, multistart, running‑max,
  strict‑CE gating) extracts a tight, valid `ObjBound` from each solve; and *node‑OBBT* turns
  the remaining exact‑MIQCP hardness into an **embarrassingly parallel** spatial branch‑and‑bound
  whose floor rises with the number of boxes — the lever that scales on NLHPC.

## See also
- `DECOMP_state.md` — authoritative current state + full results and bug history.
- `DECOMP_OBBT_offload.md` — OBBT design, validity proof, and the per‑node escalation spec.
- `DECOMP_theoretical_strategies.md` — the relaxation‑tightening landscape (RLT, Balas,
  per‑node OBBT) and the post‑OBBT diagnostic flip.
- `DECOMP_summary.md` — concise implementation summary (master structure, algorithm flow).
- Yue, Gao, You (2019) — Column‑and‑Constraint Generation for bilevel/robust optimisation.
