# DECOMP Algorithm — Implementation Review Document

**Purpose**: This document is a complete specification of the DECOMP algorithm as implemented, for independent correctness review. It includes the original problem formulation, the mathematical derivation of every constraint, the current code state, and a list of open questions.

**Key file**: `Mixed/UC-Experiments/uc_decomp_4b.py` — `UCDecomp4b` class.

---

## 1. Problem Statement (Original Formulation)

**Goal**: Find the *minimum-cost change* to DC line flow limits `b[ell]` (fmax values) such that a specific "foil" UC solution — one requiring a 10% reduction in emissions compared to the factual UC — becomes the globally optimal UC solution.

**Bilevel MILP** (outer = b choice, inner = UC dispatch):

```
min_{b}   ||b - b0||_1  (weighted L1 distance)
s.t.
  bL[ell] ≤ b[ell] ≤ bU[ell]    for all ell
  v_foil(b) ≤ v_plain(b)         [foil objective ≤ factual objective at b]
```

where:
- `b0[ell]` = original line flow limits (fmax)
- `bL[ell] = b0[ell]`, `bU[ell] = 1.2 * b0[ell]` for "free" lines (expansion-only)
- **Free lines** = lines congested ≥ 75% utilization in the factual UC solution
- `v_plain(b)` = optimal UC cost at limits `b` (no extra constraints)
- `v_foil(b)` = optimal UC cost at `b` with emissions ≤ (1 - 0.10) × E_factual + no load shedding

**Algorithm basis**: Yue et al. 2019 — Column-and-Constraint Generation (CCG) for mixed-integer bilevel LPs.

**Key idea**: The lower-level problem `v_plain(b)` is a MIP. By fixing the commitment binary `u`, the dispatch subproblem becomes an LP. CCG iteratively discovers optimal commitment patterns `u^j` and adds their full KKT optimality conditions as constraints to a growing master MILP.

---

## 2. UC Problem Structure

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| `G, R, B, L, T` | # generators, renewables, buses, lines, time periods |
| `u[g,t]` | binary commitment (1 = on) |
| `v[g,t], w[g,t]` | startup, shutdown binaries |
| `p[g,t]` | generation dispatch (MW) |
| `f[ell,t]` | line flow (MW) |
| `θ[b,t]` | voltage angle (rad) |
| `shed[b,t]` | load shedding (MW) |
| `curt[r,t]` | renewable curtailment (MW) |
| `s+[b,t], s-[b,t]` | energy storage shift in/out (MW) |
| `b[ell]` | line flow limit (fmax), the CE parameter |
| `b_line[ell]` | line susceptance (fixed physical parameter) |
| `VOLL` | value of lost load (cost per MWh shed) |
| `c_p[g], c_curt[r], c_s` | generation, curtailment, shift costs |

### 2.2 Factual UC (reference)

```
min_{u,v,w,p,f,θ,shed,curt,s+,s-}  Σ c_p[g]*p[g,t] + VOLL*shed + c_curt*curt + c_s*(s+ + s-)
s.t. [all constraints below with b = b0]
```

### 2.3 UC LP given fixed commitment u^j

Given a fixed binary commitment pattern `u^j[g,t]`, the dispatch is an LP:

**Power balance** (equality, dual `π[b,t]` free):
```
Σ_{g∈bus_b} p[g,t]  +  Σ_{r∈bus_b} (avail[r,t] - curt[r,t])
  + shed[b,t]  -  demand[b,t]  -  s+[b,t]  +  s-[b,t]
  +  Σ_{ell∈in[b]} f[ell,t]  -  Σ_{ell∈out[b]} f[ell,t]  =  0
```

**DC flow** (equality, dual `ν[ell,t]` free):
```
f[ell,t]  =  b_line[ell] * (θ[fr,t] - θ[to,t])
```

**Slack bus** (equality, dual `σ[t]` free):
```
θ[slack_bus, t]  =  0
```

**Energy shift neutrality** (equality, dual `η[b,t0]` free, per window `[t0, t0+W-1]`):
```
Σ_{t=t0}^{t0+W-1} (s+[b,t] - s-[b,t])  =  0   for each b, t0
```

**Generator bounds** (inequality, duals `λ_hi, λ_lo ≥ 0`):
```
p[g,t]  ≤  Pmax[g] * u^j[g,t]
p[g,t]  ≥  Pmin[g] * u^j[g,t]
```

**Ramp limits** (inequality, duals `ρ_up, ρ_dn, ρ_up_i, ρ_dn_i ≥ 0`):
```
p[g,0] - p0[g]  ≤  RU[g]        (init ramp-up)
p0[g] - p[g,0]  ≤  RD[g]        (init ramp-dn)
p[g,t] - p[g,t-1]  ≤  RU[g]     (ramp-up, t ≥ 1)
p[g,t-1] - p[g,t]  ≤  RD[g]     (ramp-dn, t ≥ 1)
```

**Flow limits** (inequality, duals `μ_p, μ_m ≥ 0`):
```
f[ell,t]   ≤  b[ell]    (fmax, dual μ_p)
-f[ell,t]  ≤  b[ell]    (fmin, dual μ_m)
```
> **Note**: `b[ell]` is the master decision variable (not a constant) for "free" lines.

**Variable upper bounds** (inequality, `γ_... ≥ 0` duals, each has a LB dual too since lb=0):
```
shed[b,t]  ≤  demand[b,t]         (γ_shed_ub, γ_shed_lb)
curt[r,t]  ≤  avail[r,t]          (γ_curt_ub, γ_curt_lb)
s+[b,t]   ≤  Splus_max[b,t]       (γ_sp_ub, γ_sp_lb)
s-[b,t]   ≤  Sminus_max[b,t]      (γ_sm_ub, γ_sm_lb)
```

---

## 3. DECOMP / CCG Master MILP

### 3.1 Master structure

After `k` iterations the master MILP is:

```
min_{b, bp, bm, u_foil, x_foil, {x^j, duals^j, z^j}_{j<k}}
    Σ_{ell∈free} w[ell] * (bp[ell] + bm[ell])         [L1 objective]

s.t.
[B block]   bL[ell] ≤ b[ell] ≤ bU[ell]
            b[ell] - b0[ell] = bp[ell] - bm[ell]      [L1 split]
            bp[ell], bm[ell] ≥ 0

[Foil UC]   full network UC model with BINARY u_foil,
            shared b[ell] in fmax constraints,
            + emissions constraint: Σ e[g]*p_foil ≤ (1-α)*E_factual

[∀j<k, KKT block for u^j]
    primal LP vars:  p^j, f^j, θ^j, shed^j, curt^j, s+^j, s-^j
    dual vars:       π^j, ν^j, σ^j, η^j,
                     λ_hi^j, λ_lo^j, ρ^j, μ_p^j, μ_m^j, γ^j_...
    binary z vars:   one per complementarity pair
    KKT conditions:  (Section 3.2 below)

[Optimality cut for u^j]
    c^T(u_foil, x_foil) ≤ c^T(u^j, v^j, w^j) + c^T(x^j)
```

### 3.2 KKT stationarity conditions (added per iteration)

These are derived from ∂L/∂variable = 0 for the LP given u^j:

**∂L/∂p[g,t] = 0**:
```
c_p[g]  +  λ_hi[g,t]  -  λ_lo[g,t]  +  π[bus_g, t]
  +  ρ_up[g,t] (or ρ_up_i[g] if t=0)
  -  ρ_dn[g,t] (or ρ_dn_i[g] if t=0)
  -  ρ_up[g, t+1]  +  ρ_dn[g, t+1]  (terms absent at t=T-1)
  =  0
```

**∂L/∂f[ell,t] = 0**:
```
c_f[ell,t]  +  ν[ell,t]  +  μ_p[ell,t]  -  μ_m[ell,t]
  +  π[to[ell], t]  -  π[fr[ell], t]
  =  0
```

> `c_f` is the flow cost from cvec (usually 0 but can be non-zero).

**∂L/∂θ[b,t] = 0**:
```
Σ_{ell∈out[b]} (-b_line[ell] * ν[ell,t])
  +  Σ_{ell∈in[b]} (b_line[ell] * ν[ell,t])
  +  σ[t]  (only if b == slack_bus)
  =  0
```

**∂L/∂shed[b,t] = 0**:
```
VOLL  +  γ_shed_ub[b,t]  -  γ_shed_lb[b,t]  +  π[b,t]  =  0
```

**∂L/∂curt[r,t] = 0**:
```
c_curt[r,t]  +  γ_curt_ub[r,t]  -  γ_curt_lb[r,t]  -  π[bus_r, t]  =  0
```

**∂L/∂s+[b,t] = 0**:
```
c_sp[b,t]  +  γ_sp_ub[b,t]  -  γ_sp_lb[b,t]  -  π[b,t]
  +  Σ_{t0: t∈window(t0)} η[b,t0]
  =  0
```

**∂L/∂s-[b,t] = 0**:
```
c_sm[b,t]  +  γ_sm_ub[b,t]  -  γ_sm_lb[b,t]  +  π[b,t]
  -  Σ_{t0: t∈window(t0)} η[b,t0]
  =  0
```

### 3.3 Complementarity conditions (bigM linearization)

For each primal inequality `x ≤ x_ub` (slack `s = x_ub - x ≥ 0`) with dual `μ ≥ 0`:

```
μ  ≤  M_d * (1 - z)        [dual is 0 when z=0; free when z=1]
s  ≤  M_s * z              [slack is 0 when z=1; free when z=0]
z ∈ {0, 1}
```

Full list of complementarity pairs:

| Dual | Slack expression | M_d | M_s |
|------|-----------------|-----|-----|
| `μ_p[ell,t]` | `b[ell] - f^j[ell,t]` | `M_mu` | `2*bU[ell]` |
| `μ_m[ell,t]` | `b[ell] + f^j[ell,t]` | `M_mu` | `2*bU[ell]` |
| `λ_hi[g,t]` | `Pmax*u^j - p^j[g,t]` | `M_gen` | `Pmax+1` |
| `λ_lo[g,t]` | `p^j[g,t] - Pmin*u^j` | `M_gen` | `Pmax+1` |
| `ρ_up_i[g]` | `p0[g] + RU[g] - p^j[g,0]` | `M_ramp` | `p0+RU` |
| `ρ_dn_i[g]` | `p^j[g,0] - p0[g] + RD[g]` | `M_ramp` | `Pmax+RD` |
| `ρ_up[g,t]` | `p^j[g,t-1] + RU - p^j[g,t]` | `M_ramp` | `Pmax+RU` |
| `ρ_dn[g,t]` | `p^j[g,t] + RD - p^j[g,t-1]` | `M_ramp` | `Pmax+RD` |
| `γ_shed_ub[b,t]` | `demand[b,t] - shed^j[b,t]` | `M_shed` | `demand+1` |
| `γ_shed_lb[b,t]` | `shed^j[b,t]` | `M_shed` | `demand+1` |
| `γ_curt_ub[r,t]` | `avail[r,t] - curt^j[r,t]` | `M_curt` | `avail+1` |
| `γ_curt_lb[r,t]` | `curt^j[r,t]` | `M_curt` | `avail+1` |
| `γ_sp_ub[b,t]` | `Splus_max - s+^j` | `M_shift` | `Splus_max+1` |
| `γ_sp_lb[b,t]` | `s+^j[b,t]` | `M_shift` | `Splus_max+1` |
| `γ_sm_ub[b,t]` | `Sminus_max - s-^j` | `M_shift` | `Sminus_max+1` |
| `γ_sm_lb[b,t]` | `s-^j[b,t]` | `M_shift` | `Sminus_max+1` |

### 3.4 Optimality cut

For each pattern `u^j`, the cut enforces that the foil UC objective is ≤ the UC objective at pattern `u^j`:

```
Σ_{g,t} [c_u*u_foil + c_v*v_foil + c_w*w_foil + c_p*p_foil + ...]
  ≤
  Σ_{g,t} [c_u*u^j + c_v*v^j + c_w*w^j + c_p*p^j]  +  c_shed*shed^j  + ...
```

Left side = foil block variables (master decision variables, binary u_foil).
Right side = LP block variables for pattern j + constant (u^j, v^j, w^j are constants).

---

## 4. Big-M Bound Derivations

This is the most error-prone part. Each `M_d` must upper-bound the maximum value of the corresponding dual variable across all feasible LP solutions (given any b ∈ [bL, bU]).

### 4.1 M_mu (flow duals μ_p, μ_m)

Calibrated externally: `M_mu = 10 × max|dual_fmax|` from solving the LP relaxation of the UC at `b = b0`. For IEEE 14: `M_mu ≈ 4998`.

**Risk**: This is calibrated at `b0`, but at `b_hat` (the CE) with different flows, the flow duals may be larger. If `M_mu` is too small for some `b ∈ [bL, bU]`, the master will be integer-infeasible.

### 4.2 M_gen (generator duals λ_hi, λ_lo)

```
M_gen = max(M_mu, VOLL)
```

From stationarity: `λ_hi = -(c_p + λ_lo + π + ramp_terms)`. Bounded by generation cost plus balance dual.

### 4.3 M_ramp (ramp duals ρ_up, ρ_dn)

```
M_ramp = max(M_mu, 3 × VOLL)
```

### 4.4 M_shed (shed duals γ_shed_ub, γ_shed_lb)

From stationarity: `VOLL + γ_shed_ub - γ_shed_lb + π = 0`
→ `γ_shed_lb = VOLL + γ_shed_ub + π`

Since `γ_shed_ub ≥ 0` and `γ_shed_lb ≥ 0`:
- `γ_shed_lb` is maximized when π is maximized and γ_shed_ub is maximized
- π_max = ? (see below)

**Bug 1** (fixed): Original code used `M_shed = VOLL`. Correct derivation requires `π ≥ -VOLL` (minimum balance dual), so `γ_shed_lb ≤ VOLL + 0 + VOLL = 2*VOLL`. Fixed to `M_shed = 2*VOLL`.

**Bug 8** (recently fixed): After fixing M_curt (Bug 7 below), π_max increased. See cascades.

**Current formula**:
```python
M_shed = (2.0 * VOLL + 2.0 * max_c_curt) * big_M_multiplier
```

### 4.5 M_curt (curtailment duals γ_curt_ub, γ_curt_lb)

From stationarity: `c_curt + γ_curt_ub - γ_curt_lb - π = 0`
→ `γ_curt_lb = c_curt + γ_curt_ub - π`

Maximized when π is minimized (most negative) and γ_curt_ub is maximized:
- π_min from shed stationarity: `VOLL + γ_shed_ub - γ_shed_lb + π = 0` → `π = γ_shed_lb - γ_shed_ub - VOLL ≥ -VOLL` (since γ_shed_lb, γ_shed_ub ≥ 0)
- Therefore `γ_curt_lb ≤ c_curt + M_curt_ub - (-VOLL) = c_curt + VOLL`

**Bug 7** (fixed): Original code used `M_curt = max(c_curt, VOLL) = VOLL`. Should be the SUM: `M_curt = c_curt + VOLL`.

**Current formula**:
```python
max_c_curt = max(r.curt_cost for r in data.rens)   # = 5.0 for IEEE 14
M_curt = (max_c_curt + VOLL) * big_M_multiplier
```

For IEEE 14: `M_curt = 5 + 20000 = 20005`.

### 4.6 Cascade chain (M_curt → π_max → M_shed)

After fixing M_curt, the maximum π can be derived from curtailment stationarity:
```
π = c_curt + γ_curt_ub - γ_curt_lb
```
- Maximized when γ_curt_ub = M_curt and γ_curt_lb = 0: `π_max = c_curt + M_curt = c_curt + (c_curt + VOLL) = 2*c_curt + VOLL`

Then from shed stationarity: `γ_shed_lb = VOLL + γ_shed_ub + π`
- Maximized when π = π_max and γ_shed_ub = max: `γ_shed_lb ≤ VOLL + (some M) + (2*c_curt + VOLL) = 2*VOLL + 2*c_curt`

**Therefore the correct M_shed is**: `M_shed = 2*VOLL + 2*max_c_curt` ← currently implemented.

### 4.7 M_shift (energy shift duals)

```
M_shift = max(M_mu, VOLL)
```

Derivation similar to shed/curt. Not yet stress-tested.

---

## 5. Algorithm Flow (run method)

```
1. Warm-start: solve_plain(b0), solve_foil(b0) → check if b0 is already a CE

2. Build base master:
   - Foil UC with BINARY u_foil + shared b[ell] + L1 objective
   - Apply foil extra constraints (emissions ≤ threshold, no-shed for IEEE 14)

3. MIP warm starts:
   a. solve_foil(bU) → inject Start values for foil/b vars via _inject_mip_start
   b. solve_foil(b_hat) → inject Start for foil/b vars via _inject_mip_start
   c. Add hard upper bound: F ≤ F(b_hat)    [F_hint constraint]

4. For k = 1..max_iter:
   a. Solve master MILP (900s timeout) → b_k, master_LB
   b. If INFEASIBLE: write IIS, stop
   c. If TIME_LIMIT and SolCount==0: stop
   d. SP1: solve_plain(b_k) → u_k, v_plain
   e. SP2: solve_foil(b_k) → v_foil
   f. CE check: v_foil ≤ v_plain + eps → update incumbent; tighten F ≤ F_hint
   g. Convergence: best_F - master_LB ≤ eps_obj → certified, stop
   h. Cycle: if u_k seen before → stop
   i. _add_iteration_block(u_k): add LP primal + KKT + optimality cut
   j. Re-inject warm start: _inject_mip_start(b_hat) + _complete_warm_start(b_hat)
   k. LP-relax check (diagnostic, verbose): m.relax() → obj (expected 0.0 for bigM)
   l. Checkpoint

5. Return: success, b_hat, F_opt, certified, gap, iterations, seen_patterns
```

### 5.1 Warm start methods

**`_inject_mip_start(m, master_vars, b_ws, sol_dict, u_init)`**:
Sets Gurobi `Start` for foil UC variables (`u_foil`, `v_foil`, `w_foil`, `p_foil`, `f_foil`, ...) and `b[ell]`, `bp[ell]`, `bm[ell]` from an oracle solution at `b_ws`. Does NOT touch KKT block variables.

**`_complete_warm_start(m, master_vars, b_hint, foil_sol, u_init)`**:
- Temporarily fixes `b[ell] = b_hint[ell]` and `u_foil = u_foil_hint` (via bounds)
- Solves LP relaxation (`m.relax()`, which drops all binary/SOS1 constraints)
- Injects every LP continuous variable value as `.Start` in the original MIP
- Skips all binary variables (u_foil already handled by `_inject_mip_start`; bigM z variables are skipped to avoid false-rounding violations — see Section 7.2)
- Restores b and u_foil bounds

---

## 6. Known Bugs and Fixes

| Bug | Description | Root Cause | Fix Applied |
|-----|-------------|------------|-------------|
| Bug 1 | M_shed too small (= VOLL) | gsh_lb ≤ 2*VOLL, not VOLL | M_shed = 2*VOLL |
| Bug 2 | ALPHA mismatch (0.20 vs 0.10) | Wrong α in notebook generator | Fixed in build_decomp_notebook.py |
| Bug 3 | IEEE 14 missing foil no-shed constraint | B&S adds it; DECOMP didn't | Added foil_no_shed manually |
| Bug 4 | IEEE 57 warm-start: on_t[0]=1 not UT | Hardcoded 1 instead of gen.UT | Fixed |
| Bug 5 | SOS1 LP relaxation collapses to obj=0 | relax() drops SOS1 constraints → b=b0 LP-feasible | Switched Section 4c to bigM |
| Bug 6 | _complete_warm_start violated R8031 | SOS1 post-processing zeroed s_aux linked by equality | Removed SOS1 post-processing |
| Bug 7 | M_curt = max(c_curt, VOLL) = VOLL too small | Should be c_curt + VOLL (sum, not max) | M_curt = max_c_curt + VOLL |
| Bug 8 | M_shed cascade from Bug 7 fix | π_max increased from VOLL to 2*c_curt+VOLL | M_shed = 2*VOLL + 2*max_c_curt |
| Bug 9 | M_mu floor = VOLL, should be 2*VOLL+2*c_curt | μ_p ≤ π_fr−π_to ≤ π_max−π_min = 2*VOLL+2*c_curt; same error as Bug 1 | M_mu = max(big_M_mu, 2*VOLL+2*c_curt) |

---

## 7. Current Status and Open Issues

### 7.1 Test case: IEEE 14-bus (VOLL=20000, c_curt=5, M_shed=40010, M_mu≈4998)

After Bugs 7+8 fixes, the Gurobi output for Iter 2 shows:

```
Iter 1 (no KKT block):
  Solved optimally in 0.44s
  F* = 0.9463 (root LP = 0.8648)
  MIP start loaded: obj = 2.3823 (from B&S hint)

Iter 2 (after 1 KKT block — 12632 vars, 4080 int, 16655 constrs):
  User MIP start violates constraint R10716 by 107.570033197
  Root relaxation: 0.8648 (same as Iter 1 — LP bound is meaningful)
  [LP-relax] feasible obj=0.0000  ← known artifact, see §7.3
  Explored 187K+ nodes in 900–1500s → Solution count 0
```

**Iter 1 succeeds quickly** because the master at that point is just the foil UC with b variables — no KKT block yet. The foil UC has a warm start at F=2.3823 and finds F=0.9463 quickly.

**Iter 2 has never found a solution.** This is the core problem.

### 7.2 Warm start violation issue

The violation of 107.57 on R10716 is:
- R10716 is the bigM constraint `γ_shed_lb ≤ M_shed * (1 - z)`
- LP has: `γ_shed_lb_LP = 107.57`
- LP has: `z_LP = 1 - 107.57 / 40010 = 0.99731`
- Previous code threshold: inject z=1 if z_LP > 0.99 → z=1 injected
- MIP enforces: `γ_shed_lb ≤ 40010 * (1-1) = 0` → violated by 107.57

**Current fix (applied)**: Never inject any binary Start from `_complete_warm_start`. Binary starts from `_inject_mip_start` (u_foil from oracle) are unaffected.

**However**: the warm start violation is NOT causing integer infeasibility — it causes Gurobi to reject the warm start and start cold. The MIP is still solved by B&B from the root LP bound (0.8648). The problem is that B&B explores 187K+ nodes in 1500s without finding a solution. This suggests the MIP **may be integer-infeasible** even after the M fixes, or it is **extremely hard**.

### 7.3 LP relaxation shows obj=0 (known bigM artifact)

The `[LP-relax] feasible obj=0.0000` diagnostic is expected and uninformative for bigM mode:
- `m.relax()` drops binary constraints on z, allowing z ∈ [0,1]
- Both dual > 0 and slack > 0 are now simultaneously LP-feasible (fractional z absorbs both)
- This makes b=b0, F=0 LP-feasible → obj=0
- This does NOT mean B&B is blind: Gurobi's presolved LP bound (0.8648) is derived after applying logical implications to the binary z variables

The [LP-relax] check is a holdover diagnostic that is misleading for bigM mode.

### 7.4 Open questions for review

The following questions are the core of what needs to be verified:

**Q1: Is the optimality cut direction correct?**

The cut added per pattern `u^j`:
```
c^T(u_foil, x_foil) ≤ c^T(u^j, v^j, w^j) + c^T(x^j)
```
Left side uses master binary `u_foil`. Right side uses constant `u^j` (fixed pattern) plus LP variables `x^j` (master continuous variables). 

**Is this the correct direction for making the foil a CE?** The bilevel condition is `v_foil(b) ≤ v_plain(b)`. The cut says "foil obj ≤ LP obj at pattern u^j" which is v_foil(b) ≤ LP(u^j, b). Since LP(u^j, b) ≥ v_plain(b) (u^j may not be optimal at b), this cut is a RELAXATION. Is this the intended formulation from Yue et al. 2019?

**Q2: Should the optimality cut use v_plain or LP(u^j)?**

In standard CCG for bilevel problems: at convergence, if u^j is optimal at b*, then LP(u^j, b*) = v_plain(b*), making the cut exact. Between iterations, the cut is a valid relaxation. However, if u^j is NOT optimal at b for any j accumulated, the constraint set may be feasible even when v_foil(b) > v_plain(b). Is the convergence argument still valid?

**Q3: Is there a coupling issue with b[ell] in the KKT block?**

For "free" lines, the fmax constraint is `f[ell,t] ≤ b[ell]` where b[ell] is a master decision variable (shared across the foil block and all KKT blocks). The complementarity constraint `μ_p[ell,t] ≤ M_mu * (1 - z)` and `b[ell] - f^j[ell,t] ≤ 2*bU[ell] * z` involves b[ell] as a variable.

**Is this linearization valid?** The product `μ_p * (b[ell] - f^j)` is bilinear, but with binary z the bigM reformulation avoids explicit bilinearity. Are both directions correct?:
- `z=0 → μ_p ≤ M_mu` (dual unconstrained) AND `b[ell] - f^j ≤ 0` (impossible, so z≠0 when constraint is non-binding)
- `z=1 → μ_p ≤ 0` (forces μ_p=0) AND `b[ell] - f^j ≤ 2*bU` (always satisfied since f is bounded)

Actually wait — when z=0, `b[ell] - f^j ≤ 0` should NOT be enforced (it would say f^j ≥ b[ell] which violates fmax). The bigM says `b[ell] - f^j ≤ 2*bU[ell] * z`, so z=0 gives `b[ell] - f^j ≤ 0`. **This seems wrong**: z=0 means dual=0 and slack is unconstrained, but it's actually constraining the slack to be ≤ 0, which forces f^j = b[ell] (flow at capacity). Is z=0 ↔ "slack=0" or "dual=0"?

**Convention check**: In `_comp(dual, slack, M_d, M_s)`:
```python
m.addConstr(dual <= M_d * (1 - z))  # z=1 → dual ≤ 0 → dual=0
m.addConstr(slack <= M_s * z)       # z=0 → slack ≤ 0 → slack=0
```
So z=0 → slack=0 (constraint is tight), z=1 → dual=0 (constraint is slack).

With `slack = b[ell] - f^j[ell,t]` (line at capacity when slack=0):
- `z=0`: `b[ell] - f^j ≤ 0` → f^j ≥ b[ell] → BUT fmax says f^j ≤ b[ell] → f^j = b[ell] ✓ (tight)
- `z=1`: `b[ell] - f^j ≤ 2*bU[ell]` → always true ✓ (slack)

This is correct. **z=0 means the line is at capacity (flow = fmax), z=1 means the dual is zero (line not congested)**.

**Q4: Are M_s bounds for the flow slack tight enough?**

`M_s = 2 * bU[ell]` for free lines. The slack is `b[ell] - f^j[ell,t]` where `b[ell] ≤ bU[ell]` and `f^j ≥ -bU[ell]` (fmin). Maximum slack = `bU[ell] - (-bU[ell]) = 2*bU[ell]`. ✓

**Q5: Are the big-M bounds globally valid or only local?**

The bounds derived in Section 4 assume:
- π ≥ -VOLL from shed stationarity
- π ≤ 2*c_curt + VOLL from curtailment stationarity

But there may be additional constraints on π from generator stationarity, ramp stationarity, and flow stationarity that further tighten or extend π. For example:
- From generator stationarity: `π = -c_p - λ_hi + λ_lo - ρ_up + ρ_dn ± ...`
  → π could be very large positive if λ_lo is large (generator at minimum → penalty for being below min)

**Is the range -VOLL ≤ π ≤ 2*c_curt + VOLL correct and tight?**

For IEEE 14: `-20000 ≤ π ≤ 2*5 + 20000 = 20010`. Given VOLL=20000 >> c_curt=5, the upper bound is essentially 20010 ≈ VOLL.

For IEEE 57: VOLL=500, c_curt=5 → `-500 ≤ π ≤ 510`. The old M_shed=1000 was computed assuming π ≤ 500, so the cascade from M_curt fix would push π_max to 510 → M_shed needed = 1010. This matches the previous session's diagnosis.

**Q6: Why does Iter 2 find 0 feasible solutions despite a meaningful LP bound?**

The root LP at Iter 2 is 0.8648 (meaningful). The master has 4080 binary variables and 16655 constraints. The B&B explores 187K+ nodes in 1500s with 0 solutions. Possible causes:
1. The MIP is still integer-infeasible (some M is still too small)
2. The MIP is integer-feasible but very hard (many fractional LP solutions, poor branching)
3. There is a subtle formulation error making the feasible set empty

**To distinguish**: run `debug_fix_b(b_hat)` with `comp_mode="bigM"` on IEEE 14. If this is FEASIBLE, the KKT formulation is correct at b_hat and the issue is MIP hardness. If INFEASIBLE, the KKT block has a remaining formulation error.

**Q7: Is the GomoryPasses=0 parameter correct?**

Gurobi output for Iter 2 shows `GomoryPasses 0` as a non-default parameter. This disables Gomory cuts. This was not explicitly set in the visible code. It may be set by `_optimize_with_retry` in `uc_pipeline.py` (not reproduced here). Disabling Gomory cuts with MIPFocus=1 can significantly slow finding the first feasible solution. **Check if GomoryPasses=0 is intentional.**

---

## 8. Grids and Parameters

| Grid | Buses | Lines | Gens | VOLL | M_shed | M_curt | M_mu | B&S F_opt |
|------|-------|-------|------|------|--------|--------|------|-----------|
| IEEE 14 | 14 | 20 | 5 | 20000 | 40010 | 20005 | ≈4998 | 2.3823 (certified) |
| IEEE 39 | 39 | 46 | 10 | 20000 | 40010 | 20005 | ? | 0.9016 (LB=0.37) |
| IEEE 57 | 57 | 80 | 7 | 500 | 1010 | 505 | ? | 11.6796 |

Free lines: congested ≥ 75% at factual UC optimum. bU = 1.2 × b0 for free lines.

---

## 9. Suggested Diagnostic Runs

1. **`debug_fix_b(b_hat, comp_mode="bigM")`** for IEEE 14: If FEASIBLE → formulation is correct at b_hat, problem is MIP hardness. If INFEASIBLE → remaining formulation bug.

2. **`big_M_multiplier=10`**: Run with M values inflated 10×. If Iter 2 now finds solutions, there is a remaining M too small (the inflated M covers the gap).

3. **Check π range at b_hat**: After `debug_fix_b` succeeds, print the LP values of `π^j[b,t]` to verify they stay within the claimed range `[-VOLL, 2*c_curt+VOLL]`.

4. **Check `uc_pipeline._optimize_with_retry`**: Verify if it sets `GomoryPasses=0` and whether this is intentional.

5. **Alternative: try `comp_mode="indicator"`**: Indicator constraints are M-free (no big-M violations possible). If indicator mode finds solutions quickly while bigM doesn't, the remaining bug is a big-M bound being too small.

---

## 10. Original Paper: Yue et al. 2019 (Algorithm Source)

**Reference**: Yue, D., Gao, J., Zeng, B., & You, F. (2019). A projection-based reformulation and decomposition algorithm for global optimization of a class of mixed integer bilevel linear programs. *Journal of Global Optimization*, 73, 27–57.

This paper provides the theoretical foundation for the DECOMP algorithm (CCG = Column-and-Constraint Generation).

### 10.1 General MIBLP Formulation (P0)

The paper considers the following class of mixed-integer bilevel linear programs:

```
(P0)  min_{x^u, y^u}  c^u_x x^u + c^u_y y^u + q^T x^l(x^u, y^u)
      s.t.
        A^u x^u + B^u y^u ≥ b^u           [upper-level constraints]
        x^u ≥ 0,  y^u ∈ {0,1}^{n^u_y}

        where x^l(x^u, y^u) solves:
        min_{x^l}  q^T x^l
        s.t.
          A^l x^l ≥ b^l - B^l_x x^u - B^l_y y^u    [lower-level constraints]
          G^l y^l + H^l x^l ≥ h^l                    [coupling with lower binaries]
          x^l ≥ 0,  y^l ∈ {0,1}^{n^l_y}
```

**Key structure**:
- Upper level: continuous `x^u`, binary `y^u`; lower level: continuous `x^l`, binary `y^l`
- Lower-level binaries `y^l` appear in its own constraints but NOT in lower-level objective
- The lower-level problem is a MILP (not a pure LP) — the challenge that makes standard bilevel reformulation fail

### 10.2 Projection-Based Reformulation (P4)

The paper exploits the fact that for any fixed `(x^u, y^u, y^l)`, the lower-level problem reduces to an LP. Define `Y^L = {0,1}^{n^l_y}` (all possible lower-level binary patterns). The bilevel problem is reformulated as:

```
(P4)  min_{x^u, y^u, {x^l_j}_j}  c^u_x x^u + c^u_y y^u + θ
      s.t.
        A^u x^u + B^u y^u ≥ b^u
        θ ≥ q^T x^l_j                            [for each y^l_j ∈ Y^L]
        A^l x^l_j ≥ b^l - B^l_x x^u - B^l_y y^u  [LP feasibility for y^l_j]
        G^l y^l_j + H^l x^l_j ≥ h^l             [LP coupling for y^l_j]
        KKT conditions for x^l_j (LP duality)     [optimality of x^l_j given y^l_j]
        x^l_j ≥ 0
```

This is a single-level MILP if all lower-level binary patterns `Y^L` are enumerated — but `|Y^L|` is exponential.

### 10.3 CCG Algorithm (P5, P6, P7)

The CCG algorithm incrementally builds (P4) by adding one pattern per iteration:

**Master MILP at iteration k (P5)**:
```
min_{x^u, y^u, θ, {x^l_j, duals_j, z_j}_{j ≤ k}}
    c^u_x x^u + c^u_y y^u + θ
s.t.
    A^u x^u + B^u y^u ≥ b^u
    [For each j ≤ k, add:]
      θ ≥ q^T x^l_j                    [objective cut]
      LP feasibility for x^l_j at (x^u, y^u, y^l_j)
      KKT conditions for x^l_j (complementarity via big-M)
      x^l_j ≥ 0
```

**Subproblem 1 (P6)** — find optimal lower-level binary at current upper:
```
Given (x^u_k, y^u_k) from master:
min_{y^l, x^l}  q^T x^l
s.t.  LP(x^u_k, y^u_k, y^l)
      y^l ∈ {0,1}^{n^l_y}
→ Returns y^l_k = optimal lower-level binary pattern
```

**Subproblem 2 (P7)** — verify upper-level objective:
```
Given (x^u_k, y^u_k, y^l_k) from P6:
min_{x^l}  q^T x^l
s.t.  LP(x^u_k, y^u_k, y^l_k)   [fixed y^l]
→ Returns x^l_k = LP-optimal dispatch at (x^u_k, y^u_k, y^l_k)
```

### 10.4 KKT-Tightening (P8)

After solving P6, the paper adds a tighter formulation by incorporating LP duality (KKT conditions) for each discovered `y^l_j` pattern. Complementarity conditions are linearized via big-M. The bound on big-M depends on dual variable ranges — the paper does not give explicit formulas, only that M must be "sufficiently large."

### 10.5 Convergence

**Theorem**: The CCG algorithm converges in at most `|Y^L|` iterations (finite, since each iteration adds a distinct pattern and `Y^L` is finite). In practice convergence is much faster because the optimality cuts eliminate most patterns early.

**Lower bound**: The master (P5) objective is a valid lower bound on the bilevel optimum at each iteration (all KKT cuts are necessary conditions).

**Upper bound**: Any feasible solution `(x^u_k, y^u_k)` to the master that also satisfies the lower-level SP1 is bilevel-feasible → gives an upper bound.

---

## 11. Application Paper: Fritz & Bukhsh 2025 (CE Framework)

**Reference**: Fritz, M., & Bukhsh, W. (2025). Counterfactual Explanations for Optimization Problems. arXiv:2512.04833v1.

This paper defines the CE problem for optimization problems and applies it to DCOPF and Unit Commitment, using the DECOMP algorithm from Yue et al. 2019 as one of three CE methods.

### 11.1 CE Problem Definition (Equations 2a–2d)

The general CE framework:

```
(CE)  min_{b}   d(b, b0)        [distance from original parameters]
      s.t.
        b ∈ B                   [parameter feasibility set]
        x* = argmin_x f(x, b)   [foil is optimal at b]
                 s.t.  g(x, b) ≤ 0
                       h(x, b) = 0
```

where:
- `b0` = original parameter values (the "factual" parameters)
- `b` = perturbed parameters (the CE)
- `x*` = the desired "foil" solution
- `d(·, ·)` = some distance metric (L1 in our implementation)

The CE finds the **minimum parameter change** that makes a target solution globally optimal.

### 11.2 UC Formulation in Fritz & Bukhsh (Equations 4a–4g)

**Important**: Fritz & Bukhsh use an **aggregate UC formulation** (no network, no nodal balance):

```
(UC-Fritz)  min_{u,p,v,w,r}  Σ_{g,t} [c_p p_{g,t} + c_u u_{g,t} + c_v v_{g,t} + c_w w_{g,t}]
            s.t.
              Σ_g p_{g,t} = d_t              [aggregate demand balance — NO bus/network]
              Pmin_g u_{g,t} ≤ p_{g,t} ≤ Pmax_g u_{g,t}
              p_{g,t} - p_{g,t-1} ≤ RU_g     [ramp up]
              p_{g,t-1} - p_{g,t} ≤ RD_g     [ramp down]
              v_{g,t} - w_{g,t} = u_{g,t} - u_{g,t-1}   [commitment logic]
              u_{g,t}, v_{g,t}, w_{g,t} ∈ {0,1}
```

**No line flows, no DC network, no voltage angles, no transmission constraints.**

### 11.3 Mutable Parameter

Fritz & Bukhsh use **demand** `d_t` as the mutable CE parameter (not line flow limits). The CE finds the minimum change to demand that makes a target commitment `u*` globally optimal.

The distance is `Σ_t |d_t - d0_t|` (L1 on demand perturbations).

### 11.4 Solution Region and CE Condition

The foil (target) solution region is defined as:

```
X = {x : u_{g',t'} = 1  for all (g', t') ∈ target_set}
```

where `target_set` specifies which generators must be committed at which time periods. The CE condition is that the foil objective `v*(b)` at the perturbed parameters must be ≤ the optimal objective over all other solutions.

### 11.5 DECOMP Method (as used in Fritz & Bukhsh)

Fritz & Bukhsh call their implementation of Yue et al. 2019 "DECOMP." The key steps in their application:

1. **Upper level**: `b = d_t` (demand), `y^u` = none (Fritz's upper level has no binaries — demand is continuous)
2. **Lower level**: `y^l = u_{g,t}` (commitment pattern), `x^l = p_{g,t}, v_{g,t}, w_{g,t}`
3. **Pattern enumeration**: Each CCG iteration discovers one commitment pattern `u^j`
4. **KKT block**: For each `u^j`, add LP+KKT conditions for the dispatch subproblem (which becomes an LP given `u^j`)
5. **Optimality cut**: Foil objective ≤ LP objective at pattern `u^j`

**Key difference from Yue et al. 2019**: In Fritz & Bukhsh, the upper-level has NO binary variables (demand `d_t` is continuous). The lower-level binaries are `u_{g,t}`. This simplifies the master MILP significantly (no product of upper and lower binaries).

---

## 12. Mapping: Papers → Our Implementation

| Concept | Yue et al. 2019 (P0) | Fritz & Bukhsh 2025 | Our Implementation |
|---------|---------------------|--------------------|--------------------|
| Mutable param (upper `x^u`) | generic continuous | demand `d_t` | fmax `b[ell]` |
| Upper binary (`y^u`) | present | none | `u_foil[g,t]` |
| Lower binary (`y^l`) | present | commitment `u_{g,t}` | commitment `u^j[g,t]` |
| Lower continuous (`x^l`) | present | dispatch `p, v, w` | `p, f, θ, shed, curt, s+, s-` |
| Network model | none specified | aggregate (no network) | DC network (nodal balance, flows) |
| CE distance | generic | L1 on demand | L1 on fmax change |
| Foil condition | lower obj ≤ foil obj | u_{g',t'} = 1 forced | emissions ≤ (1-α)×E_factual |
| LP structure | generic | 7 constraints (4a-4g) | 14+ constraint types (Section 2.3) |

### 12.1 Critical Differences from Fritz & Bukhsh

1. **Upper-level binary `u_foil`**: Our master MILP includes binary `u_foil[g,t]` (the foil commitment is a master variable). Fritz & Bukhsh have no upper-level binary. This means our master is a **bilevel MILP with binary on both levels**, while theirs has binary only on the lower level. The KKT blocks for `x^l` (LP dispatch given `u^j`) are the same, but the foil UC block on top is an additional MILP.

2. **DC network**: We have DC line flows, nodal power balance, and fmax constraints. Fritz & Bukhsh have only aggregate demand balance. This adds the `f[ell,t]`, `θ[b,t]`, `ν[ell,t]` dual variables and their stationarity/complementarity conditions to every KKT block.

3. **Shared `b[ell]`**: In our implementation, `b[ell]` (fmax) appears in both the foil UC block and in every KKT block's fmax complementarity. This creates a coupling between the foil block and all KKT blocks through the shared `b[ell]` variable. Fritz & Bukhsh's demand `d_t` appears only in the demand balance constraint — similar coupling but without the bilinear complementarity issue.

4. **Bilinear complementarity**: The flow complementarity `μ_p × (b[ell] - f^j[ell,t]) = 0` is bilinear in the master decision `b[ell]` and dual `μ_p`. We linearize this via big-M with binary `z`. Fritz & Bukhsh avoid this because their mutable parameter (demand) appears linearly in equality constraints, not in inequality slack expressions.

### 12.2 Mapping to Yue et al. 2019 Variables

| Yue et al. | Our implementation | Notes |
|------------|-------------------|-------|
| `x^u` | `b[ell], bp[ell], bm[ell]` | fmax and L1 split vars |
| `y^u` | `u_foil[g,t], v_foil, w_foil` | foil commitment binaries |
| `y^l` | `u^j[g,t]` | commitment pattern (fixed per iteration) |
| `x^l` | `p^j, f^j, θ^j, shed^j, curt^j, s+^j, s-^j` | LP dispatch vars |
| `q` | `c_p, VOLL, c_curt, c_s` | objective coefficients |
| `A^l x^l ≥ b^l - B^l_x x^u` | fmax constraints with `b[ell]` | key bilinear term |
| dual block | `π, ν, σ, η, λ_hi, λ_lo, ρ, μ_p, μ_m, γ_...` | LP duals (Section 3.2) |
| `z` (bigM binary) | `z_p, z_m, z_hi, z_lo, ...` | complementarity binaries |
| `θ` (value fn) | LHS of optimality cut | foil UC objective |
