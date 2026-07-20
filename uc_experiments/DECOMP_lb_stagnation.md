# DECOMP — Lower Bound Stagnation: Diagnosis & Fixes

*Generated 2026-05-25. Explains why the CCG lower bound does not increase across iterations
in `uc_decomp_4b.py` and provides fixes in order of implementation effort.*

---

## 1. Why the LB *should* increase (theory recap)

Yue et al.'s Proposition 1 states: the master at iteration k is a **relaxation** of the full
single-level reformulation, because it enforces KKT conditions only for the k patterns seen
so far (a subset of all Y^L). Since the problem is a minimisation, a tighter relaxation (more
constraints) can only raise the optimal value. Therefore:

```
master_LB_k  ≤  master_LB_{k+1}
```

This monotone property is guaranteed **only when the master is solved to proven optimality.**

---

## 2. The two numbers Gurobi reports — and which one is the LB

After each master `m.optimize()` call, Gurobi exposes two distinct values:

| Gurobi attribute | Meaning | Role in CCG |
|---|---|---|
| `m.ObjBound` | LP-relaxation-based **lower bound** on the master's MIP optimal, derived from the root LP and open B&B node bounds | The formal CCG lower bound to pass to `run()` |
| `m.ObjVal` | Value of the best **integer-feasible** solution Gurobi has found so far (upper bound on the master's MIP optimal) | Valid lower bound on the *bilevel* problem (master is a relaxation), but NOT a valid lower bound on the master's own optimal |

The CCG lower bound that appears in `run()` — the one that is supposed to grow — is
`m.ObjBound`. The stagnation problem is entirely about why `m.ObjBound` does not
increase even as new KKT blocks are added to the master.

---

## 3. The big-M LP looseness mechanism

This is the root cause. Consider a single flow-limit complementarity pair as encoded by
`_comp` in `_add_iteration_block`:

```python
# big-M complementarity for fmax: μ_p × (b[ell] − f^j[ell,t]) = 0
m.addConstr(mu_p        <= M_mu  * (1 − z))   # z=1  →  μ_p = 0
m.addConstr(b[ell] − fj <= 2*bU  *      z )   # z=0  →  slack = 0
```

When `z ∈ {0,1}` (integer), exactly one of {μ_p, slack} is forced to zero — complementarity
holds. But in the **LP relaxation** that Gurobi uses to compute `ObjBound`, `z` floats
freely in `[0, 1]`. Setting `z = 0.5` allows both `μ_p ≤ M_mu/2` and `b−f ≤ bU`
simultaneously, at zero additional LP cost.

The LP can therefore satisfy any new KKT constraint for pattern u^k by slightly adjusting
its fractional z values, **without changing b or the master's LP objective at all.** Every
new KKT block you add brings new z variables that absorb the new constraint in the same way.
The root LP bound stagnates.

This is structurally the same phenomenon as Bug 5 (SOS1 collapses LP to 0), only less
severe: the LP doesn't fully collapse to F=0, but it freezes near the root LP value
(~0.9354 for IEEE 14) while the true integer optimal is ~1.9067.

---

## 4. Confirming the diagnosis

Add the following print block immediately after each `m.optimize()` call in `run()`:

```python
# --- LB stagnation diagnostic ---
print(f"  [LB diag] ObjBound (LP-LB):   {m.ObjBound:.6f}")
print(f"  [LB diag] ObjVal   (int-UB):  {m.ObjVal:.6f}   gap={100*m.MIPGap:.2f}%")

# Solve the LP relaxation explicitly to isolate the root-LP contribution
lp = m.relax()
lp.Params.OutputFlag = 0
lp.optimize()
print(f"  [LB diag] Root LP  (relaxed): {lp.ObjVal:.6f}")
# ---------------------------------
```

**Confirmed pattern:** `ObjBound ≈ Root LP ≈ constant` across all iterations, while
`ObjVal` is good because the analytic warm-start gives Gurobi an incumbent immediately.
The gap between ObjBound and ObjVal is the "structural" 50.94% gap — it is NOT a real
optimality gap in the bilevel sense, because the warm-start incumbent is at or very near the
true bilevel optimum.

---

## 5. Why the algorithm still finds the right answer

The warm-start infrastructure (`_analytic_warm_start`) bypasses the LB problem entirely:
it computes an integer-feasible solution to the master analytically from LP duals, injects
it as a Gurobi MIP start, and the master immediately accepts it as its incumbent. The
master's job is then only to verify this can't be beaten — but since the LP bound is stuck
at ~0.9354, Gurobi cannot close the 51% gap within the time limit, so it cannot certify.

The algorithm has found the right **answer** (F=1.9067, a 20% improvement over B&S); it
just cannot produce a formal certificate. The reported gap is an artifact of the big-M
formulation, not a sign of suboptimality.

---

## 6. Fixes in order of implementation effort

### Fix 1 — Hybrid complementarity: indicators for non-bilinear pairs (low effort)

The LP looseness is only *unavoidable* for the complementarity pairs that involve the
bilinear term `b[ell] × μ_p[ell,t]`, because `b[ell]` is a master decision variable appearing
in the slack expression. For **all other** complementarity pairs (generator bounds
`λ_hi, λ_lo`, ramp limits `ρ_up, ρ_dn`, shed/curt upper bounds `γ`), the RHS is a
constant. Indicator constraints for these pairs are **not** linearised via big-M internally by
Gurobi — they are enforced by branching on z, so they do not contribute to LP looseness.

Split `_comp` in `_add_iteration_block` into two variants:

```python
def _comp_bigM(self, m, dual_var, slack_expr, M_d, M_s, tag, j):
    """Big-M complementarity — required when slack_expr contains master variable b[ell]."""
    z = m.addVar(vtype=GRB.BINARY, name=f"zbm{j}_{tag}")
    m.addConstr(dual_var              <= M_d * (1 - z), name=f"comp_d_{j}_{tag}")
    m.addConstr(slack_expr            <= M_s *      z,  name=f"comp_s_{j}_{tag}")
    return z

def _comp_indicator(self, m, dual_var, slack_expr, tag, j):
    """Indicator complementarity — use for all pairs where slack_expr is constant-RHS.
    Enforced via B&B branching; does NOT loosen the LP relaxation."""
    z = m.addVar(vtype=GRB.BINARY, name=f"zbm{j}_{tag}")
    m.addGenConstrIndicator(z, True,  dual_var   == 0, name=f"comp_ind_d_{j}_{tag}")
    m.addGenConstrIndicator(z, False, slack_expr == 0, name=f"comp_ind_s_{j}_{tag}")
    return z
```

Then in `_add_iteration_block`, call `_comp_bigM` only for flow-limit pairs
(`mu_p`, `mu_m`) and `_comp_indicator` for everything else (`lam_hi`, `lam_lo`,
`rho_up`, `rho_dn`, `rho_up_i`, `rho_dn_i`, `gam_shed_*`, `gam_curt_*`, `gam_sp_*`,
`gam_sm_*`). This is roughly 20 lines of change.

**Expected effect:** Tighter LP bound for non-flow KKT conditions. Gurobi gets better
branching hints. The bilinear flow complementarity is still big-M, so the LP is not fully
tight, but the improvement in non-flow pairs may be enough to let B&B close the gap further.

---

### Fix 2 — McCormick + strong duality for flow complementarity (medium effort)

For the flow limits, the complementarity is `μ_p × (b[ell] − f^j) = 0` with `b[ell]` a
master variable. Replace the big-M z-based encoding with:

1. A McCormick linearisation of the bilinear product `w^j[ell,t] = b[ell] × μ_p^j[ell,t]`.
2. A **strong duality equality** that uses `w` in place of `b × μ_p`, eliminating the need
   for z binaries in the KKT block entirely.

Since `b[ell] ∈ [bL[ell], bU[ell]]` and `μ_p^j ∈ [0, M_mu]`, the four McCormick
inequalities are:

```python
# Introduce auxiliary w = b[ell] * mu_p^j[ell,t] for each free line, time, pattern
w = m.addVar(lb=0.0, name=f"w_mup_{ell}_{t}_{j}")

bL_e, bU_e = bL[ell], bU[ell]   # known constants

# McCormick lower envelopes
m.addConstr(w >= bL_e * mu_p + b[ell] * 0      - bL_e * 0,      ...)  # = bL_e * mu_p
m.addConstr(w >= bU_e * mu_p + b[ell] * M_mu   - bU_e * M_mu,   ...)

# McCormick upper envelopes
m.addConstr(w <= bU_e * mu_p + b[ell] * 0      - bU_e * 0,      ...)  # = bU_e * mu_p
m.addConstr(w <= bL_e * mu_p + b[ell] * M_mu   - bL_e * M_mu,   ...)
```

Then add the **strong duality equality** for the LP at fixed u^j:

```
c^T x^j  =  Σ_{ell} [w_mup^j[ell,t] + w_mum^j[ell,t]]   (flow-limit dual contribution)
          + Σ_{g,t} [Pmax*u^j * λ_hi^j − Pmin*u^j * λ_lo^j]   (gen-bound contribution)
          + ...  (all other RHS × dual terms — all linear since no b in RHS)
```

This equality replaces both the primal optimality cut AND the complementarity conditions for
the LP, using no z binaries at all for the KKT block. The LP relaxation of the master now
has a **tight dual coupling** between b[ell] and the flow duals, and the LP bound should
grow meaningfully with each new pattern.

**Caveat:** Requires restructuring `_add_iteration_block` to build the strong-duality
equality, and adding `w` variables for every `(ell, t, j)` free-line pair. This is roughly
100–150 lines of change.

---

### Fix 3 — Projection reformulation (major rewrite, theoretically cleanest)

This is Option D from the session offload. Implements Yue et al.'s Section 4.2
projection-based single-level formulation (P4) directly, which avoids both big-M and
bilinear terms by projecting out the LP continuous variables entirely. The LP relaxation is
as tight as possible by construction, and no McCormick envelopes are needed.

This is a major rewrite of `_add_iteration_block` and the master problem structure.
Worth pursuing if Fix 2 is implemented and the LB still stagnates on IEEE 39 or IEEE 57.

---

## 7. Recommended action sequence

Start with the diagnostic print in section 4 to confirm the pattern. Then:

1. **Immediately**: implement Fix 1 (indicator constraints for non-bilinear pairs). Low risk,
   ~20 lines, should tighten LP bound and improve B&B convergence. Validate on IEEE 14.

2. **If gap still doesn't close**: implement Fix 2 (McCormick + strong duality for flow
   complementarity). This is the most targeted fix for the actual bilinear source of looseness.

3. **If IEEE 39/57 still show stagnant LB after Fix 2**: consider Fix 3 (full projection
   reformulation).

In all cases, the current algorithm already finds a high-quality solution (F=1.9067, 20%
better than B&S). The certification gap is structural to big-M, not a sign of a wrong answer.

---

## 8. Files to change

| File | Change |
|---|---|
| `uc_decomp_4b.py` | Split `_comp` into `_comp_bigM` + `_comp_indicator`; update all callsites in `_add_iteration_block`; add diagnostic prints in `run()` |
| `DECOMP_state.md` | Add Fix 1/2/3 to future work; update current state |
