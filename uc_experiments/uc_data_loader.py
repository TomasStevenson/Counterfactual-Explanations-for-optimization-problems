# uc_data_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Load any of the generated JSON grid files into a NetworkUCData object
# that is immediately usable by your existing pipeline.
#
# Usage
# -----
#   from uc_data_loader import load_network_from_json, quick_setup
#
#   # Minimal — returns (DATA, idx, cvec, b0, u_init, p_init, on_t, off_t)
#   DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup("ieee14_enhanced.json")
#
#   # Then straight into your notebook pattern:
#   v0, z0, sol0 = oracle.solve_plain(b0)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import numpy as np

from uc_pipeline import (
    Line, Generator, Renewable, NetworkUCData,
    IndexMap, build_index_map_network_uc,
    build_cost_vector_network_uc,
    default_initial_conditions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────────────────────────────────────

def load_network_from_json(
    path: str | Path,
    *,
    carbon_price: Optional[float] = None,   # overrides JSON value if given
    curt_penalty: float = 5.0,              # $/MWh curtailment cost (if not in JSON)
    voll: float = 20_000.0,                 # $/MWh value of lost load
    slack_bus: Optional[int] = None,        # overrides JSON value if given
    demand_scaling: Optional[float] = None,  # if given, scales demand and S+/S- limits
    ren_scaling: Optional[float] = None,     # if given, scales renewable availability and curtailment costs
) -> NetworkUCData:
    """
    Load a grid JSON file and return a ready-to-use NetworkUCData.

    The JSON schema expected (matches your generated files):
      nB, T, slack_bus, carbon_price
      demand         : list[list[float]]  shape (nB, T)
      Splus_max      : list[list[float]]  shape (nB, T)
      Sminus_max     : list[list[float]]  shape (nB, T)
      pi_plus        : list[list[float]]  shape (nB, T)
      pi_minus       : list[list[float]]  shape (nB, T)
      lines          : list[{fr, to, b, fmax}]
      gens           : list[{bus, Pmin, Pmax, RU, RD, UT, DT,
                              SU_cost, SD_cost, no_load_cost,
                              fuel_cost, emission_rate}]
      rens           : list[{bus, avail: list[float],
                              curt_cost: list[float]}]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")

    with open(path) as f:
        d = json.load(f)

    nB = int(d["nB"])
    T  = int(d["T"])

    cp = float(carbon_price) if carbon_price is not None else float(d.get("carbon_price", 0.0))
    sb = int(slack_bus)       if slack_bus   is not None else int(d.get("slack_bus", 0))

    # ── Arrays ──────────────────────────────────────────────────────────────
    demand     = np.array(d["demand"],     dtype=float)   # (nB, T)
    demand = demand * float(demand_scaling) if demand_scaling is not None else demand   
    Splus_max  = np.array(d["Splus_max"],  dtype=float)
    Sminus_max = np.array(d["Sminus_max"], dtype=float)
    pi_plus    = np.array(d["pi_plus"],    dtype=float)
    pi_minus   = np.array(d["pi_minus"],   dtype=float)

    # ── Lines ────────────────────────────────────────────────────────────────
    lines: List[Line] = [
        Line(fr=int(l["fr"]), to=int(l["to"]),
             b=float(l["b"]), fmax=float(l["fmax"]))
        for l in d["lines"]
    ]

    # ── Generators ──────────────────────────────────────────────────────────
    gens: List[Generator] = [
        Generator(
            bus=int(g["bus"]),
            Pmin=float(g["Pmin"]),
            Pmax=float(g["Pmax"]),
            RU=float(g["RU"]),
            RD=float(g["RD"]),
            UT=int(g["UT"]),
            DT=int(g["DT"]),
            SU_cost=float(g["SU_cost"]),
            SD_cost=float(g["SD_cost"]),
            no_load_cost=float(g["no_load_cost"]),
            fuel_cost=float(g["fuel_cost"]),
            emission_rate=float(g.get("emission_rate", 0.0)),
        )
        for g in d["gens"]
    ]

    # ── Renewables ───────────────────────────────────────────────────────────
    rens: List[Renewable] = []
    for r in d.get("rens", []):
        avail = np.array(r["avail"], dtype=float)
        cc = r.get("curt_cost", None)
        if cc is None:
            curt_cost = np.full(T, float(curt_penalty))
        else:
            curt_cost = np.array(cc, dtype=float)
        rens.append(Renewable(bus=int(r["bus"]), avail=avail, curt_cost=curt_cost))
    if ren_scaling is not None:
        s = float(ren_scaling)
        rens = [Renewable(bus=r.bus, avail=r.avail * s, curt_cost=r.curt_cost) for r in rens]

    
    return NetworkUCData(
        nB=nB, T=T,
        demand=demand, lines=lines, gens=gens, rens=rens,
        Splus_max=Splus_max, Sminus_max=Sminus_max,
        pi_plus=pi_plus, pi_minus=pi_minus,
        carbon_price=cp,
        slack_bus=sb,
        voll=voll,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick-setup helper — mirrors your notebook Cell 3 pattern exactly
# ─────────────────────────────────────────────────────────────────────────────

def quick_setup(
    path: str | Path,
    *,
    carbon_price: Optional[float] = None,
    curt_penalty: float = 5.0,
    voll: float = 20_000.0,
    slack_bus: Optional[int] = None,
    demand_scaling: Optional[float] = None,  # if given, scales demand and S+/S- limits
    ren_scaling: Optional[float] = None,     # if given, scales renewable availability and curtailment costs
):
    """
    One-call setup that returns everything needed to run your pipeline.

    Returns
    -------
    DATA       : NetworkUCData
    idx        : IndexMap
    cvec       : np.ndarray   — cost vector aligned to idx
    b0         : np.ndarray   — baseline line capacities (fmax from JSON)
    u_init     : np.ndarray
    p_init     : np.ndarray
    on_t       : np.ndarray   — on_time_init
    off_t      : np.ndarray   — off_time_init

    Example
    -------
    >>> DATA, idx, cvec, b0, u_init, p_init, on_t, off_t = quick_setup("ieee14_enhanced.json")
    >>> oracle = UCWeakWCEOracle(data=DATA, cvec=cvec, idx=idx, ...)
    >>> v0, z0, sol0 = oracle.solve_plain(b0)
    """
    DATA = load_network_from_json(
        path,
        carbon_price=carbon_price,
        curt_penalty=curt_penalty,
        voll=voll,
        slack_bus=slack_bus,
        demand_scaling=demand_scaling,
        ren_scaling=ren_scaling,
    )

    nG = len(DATA.gens)
    nR = len(DATA.rens)
    nB = int(DATA.nB)
    nL = len(DATA.lines)
    T  = int(DATA.T)

    idx = build_index_map_network_uc(nG=nG, nR=nR, nB=nB, nL=nL, T=T, major="t")

    fuel_cost_vec     = np.array([g.fuel_cost     for g in DATA.gens], dtype=float)
    emission_rate_vec = np.array([g.emission_rate for g in DATA.gens], dtype=float)
    no_load_cost_vec  = np.array([g.no_load_cost  for g in DATA.gens], dtype=float)
    su_cost_vec       = np.array([g.SU_cost       for g in DATA.gens], dtype=float)
    sd_cost_vec       = np.array([g.SD_cost       for g in DATA.gens], dtype=float)
    curt_cost_mat     = (np.vstack([r.curt_cost for r in DATA.rens])
                         if nR > 0 else np.zeros((0, T), dtype=float))

    cvec = build_cost_vector_network_uc(
        idx=idx,
        fuel_cost=fuel_cost_vec,
        emission_rate=emission_rate_vec,
        carbon_price=float(DATA.carbon_price),
        no_load_cost=no_load_cost_vec,
        su_cost=su_cost_vec,
        sd_cost=sd_cost_vec,
        curt_cost=curt_cost_mat,
        pi_plus=DATA.pi_plus,
        pi_minus=DATA.pi_minus,
        voll=float(DATA.voll),
    )

    b0 = np.array([float(L.fmax) for L in DATA.lines], dtype=float)

    u_init, p_init, on_t, off_t = default_initial_conditions(DATA)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_peak = float(np.max(np.sum(DATA.demand, axis=0)))
    ren_peak   = sum(float(np.max(r.avail)) for r in DATA.rens)
    print(f"Loaded: {Path(path).name}")
    print(f"  Buses: {nB}  |  Lines: {nL}  |  T: {T}h")
    print(f"  Generators: {nG}  |  Renewables: {nR}")
    print(f"  System peak demand: {total_peak:.1f} MW")
    print(f"  Total renewable capacity: {ren_peak:.1f} MW")
    print(f"  Carbon price: {DATA.carbon_price:.1f} $/tCO2")

    return DATA, idx, cvec, b0, u_init, p_init, on_t, off_t


# ─────────────────────────────────────────────────────────────────────────────
# Grid summary printer (optional diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def print_grid_summary(DATA: NetworkUCData):
    """Print a compact human-readable summary of the grid."""
    nG = len(DATA.gens)
    nR = len(DATA.rens)
    nL = len(DATA.lines)

    print(f"\n{'─'*55}")
    print(f"  Grid: {DATA.nB} buses, {nL} lines, T={DATA.T}h")
    print(f"{'─'*55}")

    print(f"\n  Generators ({nG}):")
    print(f"  {'#':>3}  {'Bus':>4}  {'Pmin':>6}  {'Pmax':>6}  "
          f"{'RU':>5}  {'UT':>3}  {'DT':>3}  {'FuelCost':>9}  {'Emit':>5}")
    for i, g in enumerate(DATA.gens):
        print(f"  {i:>3}  {g.bus:>4}  {g.Pmin:>6.0f}  {g.Pmax:>6.0f}  "
              f"{g.RU:>5.0f}  {g.UT:>3}  {g.DT:>3}  "
              f"{g.fuel_cost:>9.2f}  {g.emission_rate:>5.2f}")

    if nR > 0:
        print(f"\n  Renewables ({nR}):")
        print(f"  {'#':>3}  {'Bus':>4}  {'PeakMW':>8}")
        for i, r in enumerate(DATA.rens):
            print(f"  {i:>3}  {r.bus:>4}  {float(np.max(r.avail)):>8.1f}")

    print(f"\n  Lines ({nL}):")
    print(f"  {'#':>3}  {'From':>5}  {'To':>4}  {'b(1/xl)':>9}  {'fmax':>7}")
    for i, l in enumerate(DATA.lines):
        print(f"  {i:>3}  {l.fr:>5}  {l.to:>4}  {l.b:>9.3f}  {l.fmax:>7.1f}")

    peak_by_bus = np.max(DATA.demand, axis=1)
    print(f"\n  Peak demand by bus (MW):")
    nonzero = [(b, peak_by_bus[b]) for b in range(DATA.nB) if peak_by_bus[b] > 0.1]
    for b, p in nonzero:
        print(f"    Bus {b:>3}: {p:>8.2f} MW")
    print(f"  System peak: {float(np.max(np.sum(DATA.demand, axis=0))):.2f} MW")
    print(f"{'─'*55}\n")
