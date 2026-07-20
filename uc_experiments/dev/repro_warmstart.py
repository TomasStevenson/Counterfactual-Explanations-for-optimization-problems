"""Fast iteration script for debugging the DECOMP warm-start step.

Loads the pickled cache from cache_setup.py, replicates run()'s flow up to
and including Iter-1 solve + KKT block 0 + the warm-start call, then prints
timing/success info.  Optionally runs Iter 2 for `--solve-iter2 SECONDS` to
verify Gurobi accepts the injected incumbent.

Run from this directory:
    python repro_warmstart.py                  # warm-start only (~10s)
    python repro_warmstart.py --solve-iter2 30 # also burn 30s on Iter 2 to check incumbent

Exits 0 on warm-start success, 1 on warm-start failure.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\tomas\Desktop\gurobi.lic")

import gurobipy as gp
from gurobipy import GRB

from uc_pipeline import _optimize_with_retry
from uc_decomp_4b import UCDecomp4b

from _decomp_repro_helpers import UCWeakWCEOracle, make_foil_fn_14

CACHE_PATH = "_cache_decomp_14.pkl"


def banner(s):
    print()
    print("-" * 72)
    print(s)
    print("-" * 72)


def main(args):
    # --------------- load cache ----------------------------------------
    t0 = time.time()
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: {CACHE_PATH} not found — run cache_setup.py first")
        return 2
    with open(CACHE_PATH, "rb") as fh:
        c = pickle.load(fh)
    print(f"[load] cache ({os.path.getsize(CACHE_PATH)/1024:.0f} KB) "
          f"in {time.time()-t0:.2f}s")

    DATA      = c["DATA_14"]
    idx       = c["idx_14"]
    cvec      = c["cvec_14"]
    b0        = c["b0_14"]
    u_init    = c["u_init_14"]
    p_init    = c["p_init_14"]
    on_t      = c["on_t_14"]
    off_t     = c["off_t_14"]
    E_factual = c["E_factual_14"]
    b_free    = c["b_free_idx_14"]
    bL        = c["bL_14"]
    bU        = c["bU_14"]
    w         = c["w_14"]
    big_M_mu  = c["big_M_mu_14"]
    b_hat     = c["b_hat_14"]
    F_bs      = c["F_bs_14"]
    solFoil_bhat = c["solFoil_bhat_14"]
    ALPHA     = c["ALPHA"]
    W         = int(DATA.T)

    # --------------- reconstruct oracle + foil fn ----------------------
    foil_fn_14 = make_foil_fn_14(DATA, E_factual, ALPHA)
    oracle = UCWeakWCEOracle(
        data=DATA, cvec=cvec, idx=idx,
        window_size=W, per_bus_neutrality=True,
        u_init=u_init, p_init=p_init, on_t=on_t, off_t=off_t,
        foil_extra_constr_fn=foil_fn_14, output_flag=0,
    )
    # Prime the cache with the b_hat foil solution
    oracle.cache_foil[oracle._key(b_hat)] = (
        float(solFoil_bhat["obj"]),
        None,             # z not needed
        solFoil_bhat,
    )

    # --------------- construct UCDecomp4b ------------------------------
    decomp = UCDecomp4b(
        oracle=oracle, data=DATA, idx=idx, cvec=cvec,
        foil_extra_constr_fn=foil_fn_14,
        b0=b0, b_bounds=(bL, bU), b_free_idx=b_free,
        big_M_mu=big_M_mu,
        eps_weak=1e-3, eps_obj=1e-3, max_iter=2,
        output_flag=0, verbose=True, w=w,
        big_M_multiplier=1.0,
        master_time_limit=900.0,
        master_output_flag=0,
        master_mip_gap=1e-4,
        comp_mode="bigM",
        b_hat_hint=b_hat,
    )

    # --------------- replicate run() up to Iter-1 solve ----------------
    banner("PHASE A — build master + Iter 1 solve")
    t_phase = time.time()
    m, master_vars = decomp._build_master_base(
        W, True, u_init, p_init, on_t, off_t,
    )

    # warm-start at bU (matches run())
    b_ws = b0.copy()
    b_ws[b_free] = bU[b_free]
    vd_ws, _, sd_ws = oracle.solve_foil(b_ws)
    if sd_ws is not None:
        decomp._inject_mip_start(m, master_vars, b_ws, sd_ws, u_init)
        print(f"  bU warm-start injected (F_ws={decomp.F(b_ws):.4f})")

    # b_hat hint
    decomp._inject_mip_start(m, master_vars, b_hat, solFoil_bhat, u_init)
    print(f"  b_hat hint injected (F_hint={decomp.F(b_hat):.4f})")

    # F upper bound
    m.addConstr(
        gp.quicksum(w[ell] * (master_vars["bp"][ell] + master_vars["bm"][ell])
                    for ell in b_free) <= decomp.F(b_hat),
        name="_hint_F_ub",
    )
    m.update()

    # Iter 1 solve
    m.Params.OutputFlag   = 0
    m.Params.MIPGap       = 1e-4
    m.Params.MIPFocus     = 1
    m.Params.NumericFocus = 2
    m.Params.TimeLimit    = 120.0
    _optimize_with_retry(m)
    print(f"  Iter 1 status={m.Status} obj={m.ObjVal:.4f}  "
          f"({time.time()-t_phase:.2f}s)")
    assert m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL), "Iter 1 didn't solve"

    nL = len(DATA.lines)
    b_1 = np.array([
        float(master_vars["b"][ell].X) if ell in set(b_free)
        else float(b0[ell])
        for ell in range(nL)
    ])
    vp1, _, sp1 = oracle.solve_plain(b_1)
    u_1 = np.round(sp1["u"]).astype(int)
    print(f"  b_1: F={decomp.F(b_1):.4f}  d|b|={np.sum(np.abs(b_1-b0)):.4f}")
    print(f"  v_plain(b_1)={vp1:.4f}  sum(u_1)={int(u_1.sum())}")

    # --------------- replicate KKT block addition ----------------------
    banner("PHASE B — add KKT block 0 for u_1")
    t_phase = time.time()
    decomp._add_iteration_block(
        m, master_vars, 0, u_1, u_init, p_init, W, True,
    )
    print(f"  KKT block added: m has {m.NumVars} vars ({m.NumIntVars} int), "
          f"{m.NumConstrs} constrs  ({time.time()-t_phase:.2f}s)")

    # --------------- THE TEST: analytic warm-start --------------------
    banner("PHASE C — _full_integer_warm_start (analytic warm start)")
    decomp._inject_mip_start(m, master_vars, b_hat, solFoil_bhat, u_init)

    t_ws = time.time()
    success = decomp._full_integer_warm_start(
        m, master_vars,
        b_hat, solFoil_bhat, [u_1],
        W, True,
        u_init, p_init, on_t, off_t,
    )
    elapsed = time.time() - t_ws
    print(f"\n  [REPRO] _full_integer_warm_start: success={success}  "
          f"time={elapsed:.2f}s")
    if not success:
        print("  [REPRO] FAIL — analytic warm start did not produce values")
        m.dispose()
        return 1

    # --------------- Verify Gurobi accepts the start ------------------
    if args.solve_iter2 > 0:
        banner(f"PHASE D — Iter 2 solve for {args.solve_iter2}s  "
               f"(verify Gurobi accepts the incumbent)")
        m.Params.OutputFlag = 1
        m.Params.TimeLimit  = float(args.solve_iter2)
        t_iter2 = time.time()
        _optimize_with_retry(m)
        print(f"\n  Iter 2 done in {time.time()-t_iter2:.2f}s")
        print(f"  Status={m.Status}  SolCount={m.SolCount}")
        if m.SolCount > 0:
            print(f"  Incumbent ObjVal={m.ObjVal:.4f}  ObjBound={m.ObjBound:.4f}  "
                  f"MIPGap={m.MIPGap:.4%}")
        else:
            print(f"  No incumbent found")

    m.dispose()
    banner("DONE")
    return 0

    # --------------- optionally: verify Gurobi accepts the start ------
    if args.solve_iter2 > 0:
        banner(f"PHASE D — Iter 2 solve for {args.solve_iter2}s  "
               f"(verify Gurobi accepts the incumbent)")
        m.Params.OutputFlag = 1
        m.Params.TimeLimit  = float(args.solve_iter2)
        t_iter2 = time.time()
        _optimize_with_retry(m)
        print(f"\n  Iter 2 done in {time.time()-t_iter2:.2f}s")
        print(f"  Status={m.Status}  SolCount={m.SolCount}")
        if m.SolCount > 0:
            print(f"  Incumbent ObjVal={m.ObjVal:.4f}  ObjBound={m.ObjBound:.4f}  "
                  f"MIPGap={m.MIPGap:.4%}")
        else:
            print(f"  No incumbent found")

    m.dispose()
    banner("DONE")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solve-iter2", type=float, default=0.0,
        help="Seconds to spend solving Iter 2 after warm start (0 = skip)"
    )
    args = parser.parse_args()
    sys.exit(main(args))
