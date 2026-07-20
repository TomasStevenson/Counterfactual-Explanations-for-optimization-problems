# DECOMP Algorithm — Implementation Summary

## Goal
Implement a third CE method (**DECOMP**) for modifying line flow limits `b[ell]` (fmax) in Unit Commitment problems, alongside the existing B&S and NCXplain methods.

## Mathematical Basis
- **Paper**: Yue et al. 2019 — Column-and-Constraint Generation (CCG) for mixed-integer bilevel LPs.
- **Problem**: `min ||b - b0||_1` s.t. the foil solution is UC-optimal at `b` (bilevel MILP).
- **Key idea**: Iteratively add LP-feasibility + full KKT conditions for each discovered commitment pattern `u^j` directly into the master MILP.

## Existing Code Analysed

| File | Key utilities extracted |
|---|---|
| `uc_pipeline.py` | `build_network_uc_model`, `NetworkUCData`, `IndexMap`, `_optimize_with_retry`, constraint naming convention |
| `uc_master_relax_4b.py` | `remove_fixed_fmax_constraints_4b`, `add_variable_fmax_constraints_4b` |
| `uc_branch_sandwich_4b.py` | `_GurobiKeepAlive`, `_LRUCache`, oracle interface (`solve_plain`, `solve_foil`) |

## Constraint Names (from `build_network_uc_model`)
`pmax[g,t]`, `pmin[g,t]`, `ramp_up[g,t]`, `ramp_dn[g,t]`, `ramp_up_init[g]`, `ramp_dn_init[g]`,
`dcflow[ell,t]`, `fmax[ell,t]`, `fmin[ell,t]`, `balance[b,t]`, `slack[t]`,
`shed_ub[b,t]`, `curt_ub[r,t]`, `splus_ub[b,t]`, `sminus_ub[b,t]`, `neutral_bus[b,t0]`

## Master Problem Structure (per iteration k)
```
min  sum_ell w[ell] * (bp[ell] + bm[ell])        [L1 objective]
s.t.
  [b block]         bL ≤ b[ell] ≤ bU, L1 linearization

  [foil block]      full UC(b) with BINARY u_foil, shared b in fmax
                    + foil_extra_constr_fn (emissions inequality)

  [for each u^j in U^k, j=0..k-1]
    primal LP vars:  p^j, f^j, theta^j, shed^j, curt^j, splus^j, sminus^j
    dual vars:       pi (balance), nu (dcflow), sigma (slack bus), eta (neutrality)
                     lam_hi, lam_lo (gen bounds), rho_up/dn (ramp), mu_p/m (flow),
                     gam_shed/curt/splus/sminus (ub and lb=0)
    binary vars:     z_p[ell,t,j], z_m[ell,t,j] (flow compl.), + others per ineq.
    KKT conditions:  stationarity, dual feasibility, complementarity (big-M)
    optimality cut:  c^T(x_foil, u_foil) ≤ c^T(x^j, u^j)
```

## KKT Stationarity Conditions (LP given u^j)

| Variable | Stationarity (= 0) |
|---|---|
| p[g,t] | c_p[g] + lam_hi − lam_lo − lam_lb_p + pi[bus_g,t] + ramp_terms |
| f[ell,t] | c_f + nu[ell,t] + mu_p − mu_m + pi[to,t] − pi[fr,t] |
| theta[b,t] | Σ_{out} −b_line·nu + Σ_{in} b_line·nu + sigma[t]·1{b=slack} |
| shed[b,t] | voll + gam_shed_ub − gam_shed_lb + pi[b,t] |
| curt[r,t] | c_curt[r,t] + gam_curt_ub − gam_curt_lb − pi[bus_r,t] |
| splus[b,t] | pi_plus[b,t] + gam_sp_ub − gam_sp_lb − pi[b,t] + Σ eta[b,t0] |
| sminus[b,t] | pi_minus[b,t] + gam_sm_ub − gam_sm_lb + pi[b,t] − Σ eta[b,t0] |

## Complementarity (Big-M Linearization)
- **Flow limits (free lines)**: bilinear (mu × variable b) → binary z per (ell, t, j)
  - `mu_p[ell,t] ≤ M_mu*(1-z_p)`;  `b[ell] - f^j[ell,t] ≤ 2*bU[ell]*z_p`
- **Fixed lines**: standard big-M (constant RHS)
- **Generator bounds, ramp, shed, curt, shift**: standard big-M

## Algorithm Flow (`run`)
```
1. Warm-start: check b0 via oracle
2. Build base master (foil block + b vars + L1)
3. For k = 1..max_iter:
   a. Solve master → b_k, master_LB
   b. SP1: oracle.solve_plain(b_k) → u_k, v_plain
   c. SP2: oracle.solve_foil(b_k)  → v_foil
   d. CE check: v_foil ≤ v_plain + eps_weak → update incumbent
   e. Convergence: best_F - master_LB ≤ eps_obj → certified, stop
   f. Cycle check: if u_k pattern already seen → stop
   g. Add KKT block for u_k, record pattern
4. Return result dict
```

## Files to Create
- `uc_decomp_4b.py` — `UCDecomp4b` class (in progress)
- `decomp_3grids.ipynb` — experiments on ieee14, ieee39, ieee57

## Grids / Foil
Same setup as `bs_7grids.ipynb`: **ieee14, ieee39, ieee57** with **20% emissions reduction** foil.
