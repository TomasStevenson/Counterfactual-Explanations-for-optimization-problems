"""Helpers shared by cache_setup.py and dev/repro_warmstart.py.

Mirrors the inline helpers in decomp_3grids.ipynb (cell 5) so the cache /
repro scripts don't depend on the notebook runtime.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional
import numpy as np
import gurobipy as gp

from uc_pipeline import solve_uc_with_cost_4b


def replace_line_limits(data, b_line):
    b_line = np.asarray(b_line, dtype=float).reshape(-1)
    new_lines = [replace(L, fmax=float(b_line[ell])) for ell, L in enumerate(data.lines)]
    return replace(data, lines=new_lines)


def get_congested_free_lines(sol0, b0, thr=0.75):
    f_abs_max = np.max(np.abs(sol0['f']), axis=1)
    util = f_abs_max / np.maximum(b0, 1e-9)
    idx_cong = [ell for ell in range(len(b0)) if util[ell] >= thr]
    if not idx_cong:
        idx_cong = [int(np.argmax(util))]
    return idx_cong, util


def build_b_bounds(b0, b_free_idx, scale_up=1.2):
    bL = b0.copy(); bU = b0.copy()
    for ell in b_free_idx:
        bU[ell] = scale_up * b0[ell]
    return bL, bU


def make_line_weights(DATA, b0, util=None, alpha=1.0, beta=1.0, gamma=0.5, eps_u=0.05):
    b_susc = np.array([float(L.b) for L in DATA.lines], dtype=float)
    x_proxy = 1.0 / np.maximum(np.abs(b_susc), 1e-9)
    w = (x_proxy / max(np.median(x_proxy), 1e-9)) ** alpha
    w *= (b0 / max(np.median(b0), 1e-9)) ** beta
    if util is not None:
        w *= (1.0 / (eps_u + util)) ** gamma
    w = w / np.mean(w)
    return np.clip(w, 0.1, 10.0)


class UCWeakWCEOracle:
    """LRU-cached UC oracle. Same as the notebook's inline class."""

    def __init__(self, data, cvec, idx, window_size, per_bus_neutrality,
                 u_init, p_init, on_t, off_t,
                 foil_extra_constr_fn=None, output_flag=0,
                 time_limit=None, cache_decimals=3):
        self.data = data
        self.cvec = np.asarray(cvec, float)
        self.idx = idx
        self.window_size = int(window_size)
        self.per_bus_neutrality = bool(per_bus_neutrality)
        self.u_init = u_init; self.p_init = p_init
        self.on_t = on_t;     self.off_t = off_t
        self.foil_extra_constr_fn = foil_extra_constr_fn
        self.output_flag = int(output_flag)
        self.time_limit = time_limit
        self.cache_decimals = int(cache_decimals)
        self.cache_plain = {}
        self.cache_foil = {}

    def _key(self, b):
        return tuple(np.round(np.asarray(b, float), self.cache_decimals))

    def _solve(self, b, extra_fn, cache):
        key = self._key(b)
        if key in cache:
            return cache[key]
        data_b = replace_line_limits(self.data, b)
        _, sol, z = solve_uc_with_cost_4b(
            data=data_b, idx=self.idx, cvec=self.cvec,
            window_size=self.window_size,
            per_bus_neutrality=self.per_bus_neutrality,
            u_init=self.u_init, p_init=self.p_init,
            on_time_init=self.on_t, off_time_init=self.off_t,
            extra_constr_fn=extra_fn,
            output_flag=self.output_flag, time_limit=self.time_limit,
        )
        out = (None, None, None) if sol is None else (float(sol['obj']), np.array(z, float), sol)
        cache[key] = out
        return out

    def solve_plain(self, b):
        return self._solve(b, None, self.cache_plain)

    def solve_foil(self, b):
        return self._solve(b, self.foil_extra_constr_fn, self.cache_foil)


def make_foil_fn_14(DATA_14, E_factual_14, alpha):
    """Rebuild the IEEE-14 foil_fn (emissions cap + no-shed wrapper).

    foil_fn_14 is a closure that cannot be pickled directly; this constructor
    reproduces it from the cached scalars (E_factual_14, alpha) and DATA_14.
    """
    from uc_pipeline import make_emissions_foil_4b
    base = make_emissions_foil_4b(DATA_14, alpha=float(alpha), E_factual=float(E_factual_14))

    def foil_fn(m, var):
        base(m, var)
        m.addConstr(
            gp.quicksum(var['shed'][b, t]
                        for b in range(DATA_14.nB)
                        for t in range(int(DATA_14.T))) == 0,
            name='foil_no_shed',
        )
    return foil_fn
