# uc_experiment_builders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Data model (keep it Gurobi-free)
# ============================================================

@dataclass(frozen=True)
class Line:
    fr: int          # from-bus index (0-based)
    to: int          # to-bus index (0-based)
    b: float         # susceptance (DC approx)
    fmax: float      # flow limit (MW)


@dataclass(frozen=True)
class Generator:
    bus: int
    Pmin: float
    Pmax: float
    RU: float
    RD: float
    UT: int
    DT: int
    SU_cost: float
    SD_cost: float
    no_load_cost: float
    fuel_cost: float
    emission_rate: float = 0.0  # tCO2/MWh


@dataclass(frozen=True)
class Renewable:
    bus: int
    avail: np.ndarray      # shape (T,)
    curt_cost: np.ndarray  # shape (T,)


@dataclass(frozen=True)
class NetworkUCData:
    nB: int
    T: int
    demand: np.ndarray              # (nB, T)
    lines: List[Line]
    gens: List[Generator]
    rens: List[Renewable]
    Splus_max: np.ndarray           # (nB, T)
    Sminus_max: np.ndarray          # (nB, T)
    pi_plus: np.ndarray             # (nB, T)
    pi_minus: np.ndarray            # (nB, T)
    carbon_price: float = 0.0
    slack_bus: int = 0


# ============================================================
# Defaults (override from notebook if you already have these)
# ============================================================

# Fill these with the same values you use in the notebook.
# Structure expected:
# REP_PARAMS[tech] = {"Pmin_rep": .., "RU": .., "RD": .., "UT": .., "DT": .., "SU_cost": .., "SD_cost": .., "fuel_cost": ..}
DEFAULT_REP_PARAMS: Dict[str, Dict[str, float]]  = {
    "Carbon": {  # Coal
        "fuel_cost": 66.62,
        "SU_cost": 53133.0,
        "SD_cost": 3749.0,
        "UT": 48,
        "DT": 48,
        "RU": 132.61,
        "RD": 132.61,
        "Pmin_rep": 175.0,
    },
    "Gas": {  # Natural gas (OCGT-like)
        "fuel_cost": 77.38,
        "SU_cost": 0.0,
        "SD_cost": 0.0,
        "UT": 1,
        "DT": 1,
        "RU": 54.95,
        "RD": 54.95,
        "Pmin_rep": 0.0,
    },
    "Diesel": {  # Diesel/Oil
        "fuel_cost": 254.37,
        "SU_cost": 168.0,
        "SD_cost": 44.45,
        "UT": 1,
        "DT": 1,
        "RU": 11.29,
        "RD": 11.29,
        "Pmin_rep": 5.0,
    },
}

# Emission rate in tCO2/MWh for each tech string (same tech key used in GenConv sheet)
DEFAULT_EMISSION_RATE_T_PER_MWH: Dict[str, float] = {
    "Carbon": 1000.0 / 1000.0,  # 1.00 tCO2/MWh
    "Diesel": 760.0 / 1000.0,   # 0.76 tCO2/MWh
    "Gas": 500.0 / 1000.0,      # 0.50 tCO2/MWh
}


# ============================================================
# Excel utils
# ============================================================

def _read_raw(xlsx_path: str, sheet: str) -> pd.DataFrame:
    """
    Reads a sheet as a raw DataFrame with no header inference.
    Your notebook indexing like iloc[2:26, ...] assumes this style.
    """
    return pd.read_excel(xlsx_path, sheet_name=sheet, header=None, engine="openpyxl")


def _norm_label(x: Any) -> str:
    """Normalize labels (trim + lower) to avoid ' Alto ' vs 'Alto' mismatches."""
    return str(x).strip().lower()


# ============================================================
# Main loader
# ============================================================

def load_network_uc_from_excel_fixed(
    xlsx_path: str,
    wind_scenario: int = 2,          # 1/2/3
    solar_scenario: str = "Alto",    # Bajo/Medio/Alto (case/space-insensitive)
    curt_penalty: float = 1000.0,    # $/MWh
    allow_shifting: bool = False,
    carbon_price: float = 0.0,
    slack_bus: int = 0,
    rep_params: Optional[Dict[str, Dict[str, float]]] = None,
    emission_rates: Optional[Dict[str, float]] = None,
    skip_hydro: bool = True,
) -> NetworkUCData:
    """
    Loads the 'Gabriel' Excel UC dataset into a NetworkUCData object.

    You MUST provide rep_params/emission_rates unless you populate defaults above.
    """

    REP_PARAMS = rep_params if rep_params is not None else DEFAULT_REP_PARAMS
    EMISSION_RATE_T_PER_MWH = emission_rates if emission_rates is not None else DEFAULT_EMISSION_RATE_T_PER_MWH

    # -------------------------
    # 1) Demand (nB=14, T=24)
    # -------------------------
    dem = _read_raw(xlsx_path, "Demanda")

    # Your notebook assumes:
    # row 1 contains bus labels in columns 1..14
    # rows 2..25 contain hourly blocks 1..24
    bus_labels = dem.iloc[1, 1:15].astype(int).values  # 1..14
    blocks = dem.iloc[2:26, 0].astype(int).values      # 1..24

    T = int(len(blocks))
    nB = int(len(bus_labels))

    demand_bt = dem.iloc[2:26, 1:15].astype(float).to_numpy()  # (T, nB)
    demand = demand_bt.T  # (nB, T)

    # -------------------------
    # 2) Lines
    # -------------------------
    lin = _read_raw(xlsx_path, "Lineas")
    lin_data = lin.iloc[2:, 1:6].dropna(how="all")
    lin_data.columns = ["id", "From", "To", "xl", "Fmax"]

    lines: List[Line] = []
    for _, row in lin_data.iterrows():
        fr = int(row["From"]) - 1
        to = int(row["To"]) - 1
        x = float(row["xl"])
        if abs(x) < 1e-12:
            raise ValueError(f"Line reactance xl is ~0 for line row={row.to_dict()}")
        fmax = float(row["Fmax"])
        b = 1.0 / x  # DC susceptance
        lines.append(Line(fr=fr, to=to, b=b, fmax=fmax))

    # -------------------------
    # 3) Conventional generators (optionally skip Hydro)
    # -------------------------
    gc = _read_raw(xlsx_path, "GenConv")
    gc_data = gc.iloc[2:, 1:6].dropna(how="all")
    gc_data.columns = ["id", "Tech", "Bus", "CV", "Pmax"]

    gens: List[Generator] = []
    unknown_techs: List[str] = []

    for _, row in gc_data.iterrows():
        tech_raw = row["Tech"]
        if pd.isna(tech_raw):
            continue

        tech = str(tech_raw).strip()

        if skip_hydro and tech.lower() == "hidro":
            continue

        if tech not in REP_PARAMS:
            unknown_techs.append(tech)
            continue

        bus = int(row["Bus"]) - 1
        Pmax = float(row["Pmax"])

        rp = REP_PARAMS[tech]
        Pmin = min(float(rp["Pmin_rep"]), 0.5 * Pmax)  # safety cap
        RU = min(float(rp["RU"]), Pmax)
        RD = min(float(rp["RD"]), Pmax)

        UT = int(rp["UT"])
        DT = int(rp["DT"])

        # Avoid horizon-infeasible rigidity:
        UT = min(UT, T)
        DT = min(DT, T)

        gens.append(
            Generator(
                bus=bus,
                Pmin=Pmin,
                Pmax=Pmax,
                RU=RU,
                RD=RD,
                UT=UT,
                DT=DT,
                SU_cost=float(rp["SU_cost"]),
                SD_cost=float(rp["SD_cost"]),
                no_load_cost=float(rp.get("no_load_cost", 0.0)),
                fuel_cost=float(rp["fuel_cost"]),
                emission_rate=float(EMISSION_RATE_T_PER_MWH.get(tech, 0.0)),
            )
        )

    if unknown_techs:
        missing = sorted(set(unknown_techs))
        raise KeyError(
            "Some generator techs from sheet 'GenConv' are missing in REP_PARAMS.\n"
            f"Missing tech keys: {missing}\n"
            "Fix: pass rep_params=... with these keys (or populate DEFAULT_REP_PARAMS in this module)."
        )

    # -------------------------
    # 4) Renewables: Wind
    # -------------------------
    eol = _read_raw(xlsx_path, "GenEol")
    eol_meta = eol.iloc[2:26, 1:4].dropna()
    eol_meta.columns = ["id", "Bus", "Pmax"]
    wind_units = [(int(r.Bus) - 1, float(r.Pmax)) for r in eol_meta.itertuples(index=False)]

    wind_profile = eol.iloc[2:26, 6:9].astype(float).to_numpy()  # (T,3)
    if wind_profile.shape[0] != T:
        raise ValueError(f"Wind profile rows != T. got {wind_profile.shape[0]} vs T={T}")

    w_idx = int(wind_scenario) - 1
    if w_idx not in (0, 1, 2):
        raise ValueError("wind_scenario must be 1, 2, or 3")

    wind_cf = wind_profile[:, w_idx]  # (T,)

    rens: List[Renewable] = []
    for (bus, Pmax) in wind_units:
        avail = Pmax * wind_cf
        curt_cost = np.full(T, float(curt_penalty))
        rens.append(Renewable(bus=bus, avail=avail, curt_cost=curt_cost))

    # -------------------------
    # 5) Renewables: Solar
    # -------------------------
    sol = _read_raw(xlsx_path, "GenSol")
    sol_meta = sol.iloc[2:26, 1:4].dropna()
    sol_meta.columns = ["id", "Bus", "Pmax"]
    solar_units = [(int(r.Bus) - 1, float(r.Pmax)) for r in sol_meta.itertuples(index=False)]

    solar_labels_raw = sol.iloc[1, 6:9].tolist()  # e.g., ["Bajo","Medio","Alto"]
    solar_labels = [_norm_label(x) for x in solar_labels_raw]

    solar_profile = sol.iloc[2:26, 6:9].astype(float).to_numpy()  # (T,3)
    if solar_profile.shape[0] != T:
        raise ValueError(f"Solar profile rows != T. got {solar_profile.shape[0]} vs T={T}")

    s_key = _norm_label(solar_scenario)
    if s_key not in solar_labels:
        raise ValueError(
            f"solar_scenario='{solar_scenario}' not found in GenSol labels {solar_labels_raw}"
        )
    s_idx = solar_labels.index(s_key)
    solar_cf = solar_profile[:, s_idx]

    for (bus, Pmax) in solar_units:
        avail = Pmax * solar_cf
        curt_cost = np.full(T, float(curt_penalty))
        rens.append(Renewable(bus=bus, avail=avail, curt_cost=curt_cost))

    # -------------------------
    # 6) Shifting (disabled by default)
    # -------------------------
    if allow_shifting:
        # You can implement real shifting parameters here later.
        Splus_max = np.full((nB, T), 0.0)
        Sminus_max = np.full((nB, T), 0.0)
        pi_plus = np.full((nB, T), 0.0)
        pi_minus = np.full((nB, T), 0.0)
    else:
        Splus_max = np.zeros((nB, T))
        Sminus_max = np.zeros((nB, T))
        pi_plus = np.zeros((nB, T))
        pi_minus = np.zeros((nB, T))

    return NetworkUCData(
        nB=nB,
        T=T,
        demand=demand,
        lines=lines,
        gens=gens,
        rens=rens,
        Splus_max=Splus_max,
        Sminus_max=Sminus_max,
        pi_plus=pi_plus,
        pi_minus=pi_minus,
        carbon_price=float(carbon_price),
        slack_bus=int(slack_bus),
    )


# ============================================================
# Demand archetypes (energy-preserving diversification)
# ============================================================

def _smooth(x, k=3):
    k = max(1, int(k))
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")


def make_archetypes_from_system(demand_bt: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    demand_bt: (nB,T)
    returns archetypes (3,T): [residential, commercial, industrial]
    """
    base = demand_bt.sum(axis=0)
    base = base / (base.mean() + 1e-12)

    res = _smooth(np.roll(base, +2), 3)   # later peak
    com = _smooth(np.roll(base, -2), 3)   # earlier peak
    ind = _smooth(np.roll(base,  0), 7)   # flatter

    eps = 1e-6
    res = np.clip(res, eps, None)
    com = np.clip(com, eps, None)
    ind = np.clip(ind, eps, None)

    return np.vstack([res, com, ind])


def ipf_preserve_rows_cols(
    X: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    iters: int = 2000,
    tol: float = 1e-7
) -> np.ndarray:
    """
    Iterative proportional fitting (RAS) to match:
      sum_t X[b,t] = row_targets[b]
      sum_b X[b,t] = col_targets[t]
    while keeping X >= 0 and preserving the initial pattern.
    """
    X = np.clip(X.astype(float), 1e-12, None)

    for _ in range(iters):
        row_sums = X.sum(axis=1) + 1e-12
        X *= (row_targets / row_sums)[:, None]

        col_sums = X.sum(axis=0) + 1e-12
        X *= (col_targets / col_sums)[None, :]

        r_err = np.max(np.abs(X.sum(axis=1) - row_targets) / (row_targets + 1e-12))
        c_err = np.max(np.abs(X.sum(axis=0) - col_targets) / (col_targets + 1e-12))
        if max(r_err, c_err) < tol:
            break

    return X


def diversify_demand_archetypes_preserve_C(
    demand_bt: np.ndarray,
    seed: int = 0,
    p=(0.45, 0.35, 0.20)
):
    """
    Preserves:
      - each bus daily energy (rows)
      - total system per hour (cols)

    demand_bt: (nB,T)
    returns: (demand_new, types)
    """
    rng = np.random.default_rng(seed)
    nB, T = demand_bt.shape

    row_targets = demand_bt.sum(axis=1)  # bus energy
    col_targets = demand_bt.sum(axis=0)  # system hourly totals

    arche = make_archetypes_from_system(demand_bt, seed=seed)  # (3,T)

    types = rng.choice(3, size=nB, p=p)  # 0=res,1=com,2=ind

    X0 = np.zeros((nB, T), dtype=float)
    for b in range(nB):
        shape = arche[types[b]]
        X0[b, :] = row_targets[b] * shape / (shape.sum() + 1e-12)

    X = ipf_preserve_rows_cols(X0, row_targets=row_targets, col_targets=col_targets)
    X = np.clip(X, 0.0, None)
    return X, types


# ============================================================
# Renewable scalability
# ============================================================

def scale_renewables(data: NetworkUCData, ren_scale: float = 1.0) -> NetworkUCData:
    new_rens = []
    for r in data.rens:
        avail = np.asarray(r.avail, float) * float(ren_scale)
        new_rens.append(Renewable(bus=r.bus, avail=avail, curt_cost=np.asarray(r.curt_cost, float)))
    return NetworkUCData(
        nB=data.nB, T=data.T, demand=np.asarray(data.demand, float),
        lines=data.lines, gens=data.gens, rens=new_rens,
        Splus_max=data.Splus_max, Sminus_max=data.Sminus_max,
        pi_plus=data.pi_plus, pi_minus=data.pi_minus,
        carbon_price=data.carbon_price, slack_bus=data.slack_bus
    )

def make_scenario_data(base_data, archetype_seed: int, ren_scale: float):
    from dataclasses import replace
    from uc_experiment_builders import diversify_demand_archetypes_preserve_C, scale_renewables

    D_new, types = diversify_demand_archetypes_preserve_C(base_data.demand, seed=archetype_seed)
    d = replace(base_data, demand=D_new)
    d = scale_renewables(d, ren_scale=ren_scale)
    return d, types
