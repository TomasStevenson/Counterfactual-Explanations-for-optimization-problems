# -*- coding: utf-8 -*-
# ============================================================
# PRELUDE batch Leftraru (NLHPC) - agregado automaticamente.
# No modifica el problema: shim de display() fuera de IPython
# y medicion uniforme de tiempos ([TIMING]).
# Threads de Gurobi se limitan via gurobi.env en el cwd.
# ============================================================
import atexit as _atexit
import time as _time

_JOB_T0 = _time.perf_counter()

try:
    from IPython.display import display  # noqa: F401
except Exception:
    def display(*args, **kwargs):
        for _a in args:
            print(_a)

def _print_total_time():
    print("\n[TIMING] total_script_seconds={:.2f}".format(
        _time.perf_counter() - _JOB_T0), flush=True)
_atexit.register(_print_total_time)

try:
    import gurobipy as _gp
    _orig_optimize = _gp.Model.optimize

    def _timed_optimize(self, *args, **kwargs):
        _t0 = _time.perf_counter()
        _res = _orig_optimize(self, *args, **kwargs)
        _wall = _time.perf_counter() - _t0
        try:
            _rt = float(self.Runtime)
            _name = self.ModelName
        except Exception:
            _rt, _name = float("nan"), "?"
        print("[TIMING] optimize '{}': wall={:.2f}s gurobi_runtime={:.2f}s".format(
            _name, _wall, _rt), flush=True)
        return _res

    _gp.Model.optimize = _timed_optimize
    print("[TIMING] gp.Model.optimize instrumentado (wall + Runtime por solve)")
except Exception as _e:
    print("[TIMING] aviso: no se pudo instrumentar optimize(): {!r}".format(_e))
# ============================== fin prelude =================



# %% ================== celda 53 del notebook ==================
import os
import pandas as pd
import numpy as np

# ============================================================
# 0) CONFIG: usar CSV IEEE39 finales
# ============================================================
PATH = "."

PATH_GEN   = os.path.join(PATH, "generadores_procesados_ieee39_final.csv")
PATH_LINE  = os.path.join(PATH, "lineas_procesadas_ieee39.csv")
PATH_LOAD  = os.path.join(PATH, "demanda_nodal_ieee39_final.csv")
PATH_SOLAR = os.path.join(PATH, "perfil_solar_ieee39_final.csv")

df_gen   = pd.read_csv(PATH_GEN)
df_lines = pd.read_csv(PATH_LINE)
df_load  = pd.read_csv(PATH_LOAD)

df_wind = pd.DataFrame(columns=["Gen_ID", "Period", "P_Available_MW"])

df_solar = pd.read_csv(PATH_SOLAR) if os.path.exists(PATH_SOLAR) else pd.DataFrame(
    columns=["Gen_ID", "Period", "P_Available_MW"]
)

# ============================================================
# 1) LIMPIAR NOMBRES DE COLUMNAS
# ============================================================
df_gen.columns   = df_gen.columns.str.strip()
df_lines.columns = df_lines.columns.str.strip()
df_load.columns  = df_load.columns.str.strip()
df_solar.columns = df_solar.columns.str.strip()

# ============================================================
# 2) RENOMBRAR COLUMNAS PARA DEJARLAS COMPATIBLES
# ============================================================

# ----- DEMANDA -----
df_load = df_load.rename(columns={
    "bus": "Bus ID",
    "bus_id": "Bus ID",
    "periodo": "Period",
    "period": "Period",
    "demanda_mw": "Nodal_Load_MW",
    "load_mw": "Nodal_Load_MW"
})

# ----- LÍNEAS -----
# Esta parte acepta tanto el formato viejo:
#   desde, hacia, x, fmax
# como el formato estilo IEEE30:
#   from_bus, to_bus, resistance_pu, reactance_pu, line_charging_pu, tap_ratio, phase_shift_deg

df_lines = df_lines.rename(columns={
    "desde": "from_bus",
    "hacia": "to_bus",
    "x": "Reactancia_X",
    "fmax": "F_max",
    "reactance_pu": "Reactancia_X"
})

# Si no viene F_max, se crea constante
if "F_max" not in df_lines.columns:
    df_lines["F_max"] = 100.0

# ----- PERFIL SOLAR -----
df_solar = df_solar.rename(columns={
    "id_gen": "Gen_ID",
    "periodo": "Period",
    "period": "Period",
    "disponibilidad_mw": "P_Available_MW"
})

# ============================================================
# 3) CHEQUEO DE COLUMNAS OBLIGATORIAS
# ============================================================
cols_gen_req = [
    "id_gen", "bus_id", "Pmax", "Pmin", "Costo_Var", "Inercia_H",
    "Emisiones_tCO2_MWh", "Es_IBR", "Gamma", "Kappa", "Costo_Inv"
]

cols_line_req = ["from_bus", "to_bus", "Reactancia_X", "F_max"]
cols_load_req = ["Bus ID", "Period", "Nodal_Load_MW"]

for c in cols_gen_req:
    if c not in df_gen.columns:
        raise ValueError(f"Falta columna en generadores: {c}")

for c in cols_line_req:
    if c not in df_lines.columns:
        raise ValueError(f"Falta columna en líneas: {c}")

for c in cols_load_req:
    if c not in df_load.columns:
        raise ValueError(f"Falta columna en demanda: {c}")

# ============================================================
# 4) SANITIZAR TIPOS
# ============================================================
df_gen["id_gen"] = df_gen["id_gen"].astype(str)
df_gen["bus_id"] = df_gen["bus_id"].astype(int)

df_load["Bus ID"] = df_load["Bus ID"].astype(int)
df_load["Period"] = df_load["Period"].astype(int)
df_load["Nodal_Load_MW"] = df_load["Nodal_Load_MW"].astype(float)

df_lines["from_bus"] = df_lines["from_bus"].astype(int)
df_lines["to_bus"]   = df_lines["to_bus"].astype(int)
df_lines["F_max"] = df_lines["F_max"].astype(float)
df_lines["Reactancia_X"] = df_lines["Reactancia_X"].astype(float)

if not df_solar.empty:
    df_solar["Gen_ID"] = df_solar["Gen_ID"].astype(str)
    df_solar["Period"] = df_solar["Period"].astype(int)
    df_solar["P_Available_MW"] = df_solar["P_Available_MW"].astype(float)

# ============================================================
# 5) DEFINICIÓN DE SETS
# ============================================================
Y = [1, 2, 3]

year_growth = {
    1: 1.00,
    2: 1.05,
    3: 1.10
}

# Buses con demanda
B_load = set(df_load["Bus ID"].astype(int).unique().tolist())

# Buses que aparecen en la red
B_lines = set(df_lines["from_bus"].astype(int).unique().tolist()).union(
    set(df_lines["to_bus"].astype(int).unique().tolist())
)

# Buses donde hay generadores
B_gen = set(df_gen["bus_id"].astype(int).unique().tolist())

# Conjunto completo de buses
B = sorted(B_load.union(B_lines).union(B_gen))

T = sorted(df_load["Period"].unique().tolist())

G  = df_gen["id_gen"].tolist()
GS = df_gen[df_gen["Es_IBR"] == 0]["id_gen"].tolist()
M  = df_gen[df_gen["Es_IBR"] == 1]["id_gen"].tolist()
GE = df_gen[df_gen["Emisiones_tCO2_MWh"] > 0]["id_gen"].tolist()

print(
    f"Sets definidos: |B|={len(B)}, |T|={len(T)}, "
    f"|G|={len(G)} (|GS|={len(GS)}, |M|={len(M)}, |GE|={len(GE)})"
)

print("Buses sin demanda explícita:")
print(sorted(set(B) - B_load))
# ============================================================
# 6) PARÁMETROS DE GENERADORES
# ============================================================
Pmax = df_gen.set_index("id_gen")["Pmax"].to_dict()
Pmin = df_gen.set_index("id_gen")["Pmin"].to_dict()
cg   = df_gen.set_index("id_gen")["Costo_Var"].to_dict()
eg   = df_gen.set_index("id_gen")["Emisiones_tCO2_MWh"].to_dict()
cinv = df_gen.set_index("id_gen")["Costo_Inv"].to_dict()

Hg = {
    g: float(df_gen.set_index("id_gen")["Inercia_H"].to_dict().get(g, 0.0))
    for g in GS
}

g_bus = df_gen.set_index("id_gen")["bus_id"].to_dict()

G_en_bus = {i: [] for i in B}

for g in G:
    bi = int(g_bus[g])
    if bi in G_en_bus:
        G_en_bus[bi].append(g)
    else:
        print(f"Advertencia: el generador {g} está asignado al bus {bi}, que no está en B.")

# ============================================================
# 7) PARÁMETROS IBR
# ============================================================
gamma_base = df_gen.set_index("id_gen")["Gamma"].to_dict()
kappa_base = df_gen.set_index("id_gen")["Kappa"].to_dict()

gamma_max = (
    df_gen.set_index("id_gen")["Gamma_max"].to_dict()
    if "Gamma_max" in df_gen.columns
    else {m: 0.5 for m in M}
)

kappa_max = (
    df_gen.set_index("id_gen")["Kappa_max"].to_dict()
    if "Kappa_max" in df_gen.columns
    else {m: 0.5 for m in M}
)

xbar_mult = 2.0
xbar = {m: float(Pmax[m]) * xbar_mult for m in M}

# ============================================================
# 8) RED
# ============================================================
L, F0, Bline = [], {}, {}

for _, row in df_lines.iterrows():
    i, j = int(row["from_bus"]), int(row["to_bus"])
    line = (i, j)

    L.append(line)

    F0[line] = float(row["F_max"])

    x_val = float(row["Reactancia_X"])

    if abs(x_val) < 1e-6:
        x_val = 1e-4

    Bline[line] = 1.0 / x_val

lineas_entran = {i: [] for i in B}
lineas_salen  = {i: [] for i in B}

for (i, j) in L:
    if i in lineas_salen:
        lineas_salen[i].append((i, j))
    if j in lineas_entran:
        lineas_entran[j].append((i, j))

print(f"Red: |L|={len(L)}")

# ============================================================
# 9) DEMANDA
# ============================================================
raw_demand = df_load.set_index(["Bus ID", "Period"])["Nodal_Load_MW"].to_dict()

d = {}

for y in Y:
    factor = year_growth[y]

    for i in B:
        for t in T:
            # Si el bus no tiene demanda en el CSV, se asigna 0
            d[(i, t, y)] = float(raw_demand.get((i, t), 0.0)) * factor
# ============================================================
# 10) PERFIL RENOVABLE
# ============================================================
P_disp = {
    (g, t): float(Pmax[g])
    for g in G
    for t in T
}

if not df_solar.empty:
    for _, row in df_solar.iterrows():
        g_id = str(row["Gen_ID"])
        t = int(row["Period"])
        val = float(row["P_Available_MW"])

        if (g_id in G) and (t in T):
            P_disp[(g_id, t)] = val

# ============================================================
# 11) ESCALARES
# ============================================================
VOLL = 5000.0

y_star = 3
epsilon_CO2 = 0

# ============================================================
# 12) Hreq y Rreq
# ============================================================
D_peak = {
    y: max(
        sum(d[(i, t, y)] for i in B)
        for t in T
    )
    for y in Y
}

max_hsyn = (
    sum(float(gamma_base.get(m, 0.0)) * float(xbar[m]) for m in M)
    if len(M) > 0
    else 0.0
)

max_sync = (
    sum(float(Hg.get(g, 0.0)) for g in GS)
    if len(GS) > 0
    else 0.0
)

cap_total = max_hsyn + max_sync

alpha_H = 0.35
H_floor = 0.05

k_H = (alpha_H * cap_total / D_peak[1]) if D_peak[1] > 1e-9 else 0.0

Hreq = {}

for y in Y:
    base = k_H * D_peak[y]
    floor = H_floor * cap_total
    Hreq[y] = max(base, floor)

MaxR = (
    sum(float(kappa_base.get(m, 0.0)) * float(xbar[m]) for m in M)
    if len(M) > 0
    else 0.0
)

beta_R = 0.40
R_floor = 0.05

Rreq = {}

for y in Y:
    base = beta_R * MaxR
    floor = R_floor * MaxR
    Rreq[y] = max(base, floor)

print("cap_total =", round(cap_total, 4), " MaxR =", round(MaxR, 4))
print("D_peak =", {y: round(D_peak[y], 2) for y in Y})
print("Hreq =", {y: round(Hreq[y], 4) for y in Y})
print("Rreq =", {y: round(Rreq[y], 4) for y in Y})


# %% ================== celda 70 del notebook ==================
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time

# ============================================================
# FINAL IEEE39: SOLO WCEP CON DUALIDAD FUERTE
# FORMULACIÓN ORIGINAL + DATASET CORREGIDO + PESOS TEMPORALES
#
# Mutables:
#   1) Hreq_ce[y]  en b
#   2) xbar_ce[m]  en b
#
# Fijo:
#   - Pmax fijo; NO se muta Pmax.
#   - No hay PADM ni warm start.
# ============================================================

# Medición de tiempo total de celda/script
t0_total_script = time.time()

# ============================================================
# 0) OPCIONES
# ============================================================

WCEP_TIME_LIMIT = 5 * 60 * 60        # 5 horas para WCEP final

# Restricción D del IEEE39
# Si ya calculaste en el FW emisiones ponderadas, puedes poner epsilon_CO2_wcep manual.
# Si lo dejas como None, el código intenta usar el 90% de las emisiones FW guardadas en memoria.
epsilon_CO2_wcep = 930756.144528 * 0.9  # Puedes poner aquí el valor manual de epsilon CO2 para IEEE39
EPSILON_CO2_FRACTION = 0.90

# Objetivo J
alpha_H = 2.0
alpha_Xbar = 1.0

# Si quieres objetivo porcentual, cambia esto a True
USE_PERCENT_OBJECTIVE = True

Y_OBJECTIVE_MODE = "penalty"
TIEBREAKER_CTX_WEIGHT = 1e-9

# Opciones WCEP final
WCEP_MIPGAP = 5e-3
WCEP_OUTPUT = 1


# ============================================================
# 1) PREPARACIÓN GENERAL
# ============================================================

L_arcs = gp.tuplelist(L)
G_IBR = list(M)

if isinstance(B, (int, float, np.floating)):
    print("⚠️ B estaba pisado como número. Lo reconstruyo desde df_load/df_lines.")

    if "df_load" in globals():
        B = sorted(df_load["Bus ID"].astype(int).unique().tolist())
    else:
        B = sorted(
            set(df_lines["from_bus"].astype(int))
            .union(set(df_lines["to_bus"].astype(int)))
        )

    print("✅ Nuevo B =", B[:10], "... |B| =", len(B))

lineas_entran = {bus: [arc for arc in L_arcs if arc[1] == bus] for bus in B}
lineas_salen  = {bus: [arc for arc in L_arcs if arc[0] == bus] for bus in B}

# ============================================================
# DEMANDA CORREGIDA DEL FW IEEE39
# ============================================================
# Este bloque hace que WCEP usen la misma demanda corregida
# del Forward nuevo. Si ya existe d_fw desde la celda FW, se respeta.
# Si no existe, se reconstruye automáticamente con el mismo criterio:
# demanda escalada para que la demanda pico quede dentro de Pmax total.

DEMAND_SCALE_MODE = globals().get("DEMAND_SCALE_MODE", "auto_capacity")
DEMAND_SCALE_MANUAL = float(globals().get("DEMAND_SCALE_MANUAL", 0.713))
DEMAND_CAP_MARGIN = float(globals().get("DEMAND_CAP_MARGIN", 0.98))

if "d_fw" not in globals():
    print("⚠️ No existe d_fw desde el FW corregido. Lo reconstruyo automáticamente.")

    d_original = {k: float(v) for k, v in d.items()}

    demanda_original_ty = {}
    for t in T:
        for y in Y:
            demanda_original_ty[(t, y)] = sum(
                d_original[(i, t, y)]
                for i in B
                if (i, t, y) in d_original
            )

    peak_dem_original = max(demanda_original_ty.values())
    cap_total_pmax = sum(float(Pmax[g]) for g in G)

    if DEMAND_SCALE_MODE == "none":
        demand_scale_factor = 1.0
    elif DEMAND_SCALE_MODE == "manual":
        demand_scale_factor = float(DEMAND_SCALE_MANUAL)
    elif DEMAND_SCALE_MODE == "auto_capacity":
        if peak_dem_original <= 1e-9:
            demand_scale_factor = 1.0
        else:
            demand_scale_factor = min(
                1.0,
                DEMAND_CAP_MARGIN * cap_total_pmax / peak_dem_original
            )
    else:
        raise ValueError("DEMAND_SCALE_MODE debe ser 'auto_capacity', 'manual' o 'none'.")

    d_fw = {}
    for i in B:
        for t in T:
            for y in Y:
                if (i, t, y) not in d_original:
                    raise KeyError(f"Falta d[{i},{t},{y}] en el diccionario original.")
                d_fw[(i, t, y)] = demand_scale_factor * d_original[(i, t, y)]

    print(f"Factor demanda reconstruido = {demand_scale_factor:.8f}")
else:
    print("✅ Usando d_fw existente desde el FW corregido.")

# Diagnóstico rápido de demanda usada por WCEP
peak_dem_fw = max(
    sum(float(d_fw[(i, t, y)]) for i in B)
    for t in T
    for y in Y
)
cap_total_pmax = sum(float(Pmax[g]) for g in G)
print(f"Demanda pico usada en WCEP = {peak_dem_fw:.6f}")
print(f"Pmax total sistema                 = {cap_total_pmax:.6f}")

# ============================================================
# PESOS TEMPORALES PARA REPRESENTAR EL AÑO
# ============================================================
# Si T = [1, 7, 13, 19] representa un perfil diario, usamos omega_t
# para expandir costos y emisiones a base anual sin crear 365*|T| periodos.
# Por defecto cada t se repite 365 veces y representa 1 hora.
# Si cada t representa un bloque de 6 horas, define TIME_BLOCK_HOURS = 6.0 antes de correr.
TIME_BLOCK_HOURS = float(globals().get("TIME_BLOCK_HOURS", 1.0))
N_DAYS_EQUIV = int(globals().get("N_DAYS_EQUIV", 365))

if "omega_t" not in globals():
    omega_t = {t: float(N_DAYS_EQUIV) * TIME_BLOCK_HOURS for t in T}
else:
    omega_t = {t: float(omega_t[t]) for t in T}

def omega(t):
    return float(omega_t[t])

print("Pesos temporales omega_t usados:", omega_t)

bus_slack = int(sorted(B)[0])

Hreq_base = {y: float(Hreq[y]) for y in Y}
xbar_base = {m: float(xbar[m]) for m in M}

gamma_pos = [float(gamma_base[m]) for m in M if float(gamma_base[m]) > 1e-9]

if len(gamma_pos) == 0:
    UB_H = 1e8
else:
    UB_H = 2.0 * max(float(cinv[m]) for m in M) / min(gamma_pos)

UB_xbar = 2.0 * max(float(cinv[m]) for m in M) if len(M) > 0 else 1e8

# ============================================================
# Límite de emisiones D
# ============================================================
def _get_fw_emissions_by_year():
    candidates = [
        "emisiones_por_anio", "df_emis_fw", "df_emis", "df_emis_wcep"
    ]
    value_cols = [
        "emisiones_anuales_equivalentes",
        "emisiones_totales_ponderadas",
        "emisiones_totales_anuales",
        "emisiones",
    ]
    for name in candidates:
        if name in globals():
            df = globals()[name]
            if hasattr(df, "columns") and "y" in df.columns:
                for col in value_cols:
                    if col in df.columns:
                        return {int(row["y"]): float(row[col]) for _, row in df.iterrows()}
    return None

_fw_emis = _get_fw_emissions_by_year()

if epsilon_CO2_wcep is None:
    if _fw_emis is not None:
        epsilon_CO2_wcep_by_y = {y: EPSILON_CO2_FRACTION * float(_fw_emis[y]) for y in Y if y in _fw_emis}
        print("epsilon_CO2_wcep calculado como 90% de emisiones FW:", epsilon_CO2_wcep_by_y)
    else:
        raise ValueError(
            "epsilon_CO2_wcep=None, pero no encontré emisiones del FW. "
            "Define epsilon_CO2_wcep manualmente o ejecuta antes el FW con tabla de emisiones."
        )
elif isinstance(epsilon_CO2_wcep, dict):
    epsilon_CO2_wcep_by_y = {y: float(epsilon_CO2_wcep[y]) for y in epsilon_CO2_wcep}
else:
    epsilon_CO2_wcep_by_y = {y: float(epsilon_CO2_wcep) for y in Y}

def eps_CO2(y):
    return float(epsilon_CO2_wcep_by_y[int(y)])


print("\n" + "="*70)
print("COTAS DUALES IEEE39")
print("="*70)
print(f"UB_H    = {UB_H:.6f}")
print(f"UB_xbar = {UB_xbar:.6f}")
print(f"Bus slack usado: {bus_slack}")




# ============================================================
# 2) WCEP FINAL CON DUALIDAD FUERTE
# ============================================================

print("\n" + "="*70)
print("CONSTRUYENDO WCEP FINAL IEEE39 CON DUALIDAD FUERTE")
print("="*70)

w = gp.Model("WCEP_IEEE39_DF_solo_dataset_corregido")
w.Params.NonConvex = 2
w.Params.MIPGap = WCEP_MIPGAP
w.Params.OutputFlag = WCEP_OUTPUT
w.Params.TimeLimit = WCEP_TIME_LIMIT

# ------------------------------------------------------------
# Variables primales lower-level
# ------------------------------------------------------------

p = w.addVars(G, T, Y, lb=0.0, name="p")
u = w.addVars(GS, T, Y, lb=0.0, ub=1.0, name="u")
s = w.addVars(B, T, Y, lb=0.0, name="s")

xcap = w.addVars(M, Y, lb=0.0, name="xcap")
h_syn = w.addVars(M, T, Y, lb=0.0, name="h_syn")
r_ffr = w.addVars(M, T, Y, lb=0.0, name="r_ffr")

f_pos = w.addVars(L_arcs, T, Y, lb=0.0, name="f_pos")
f_neg = w.addVars(L_arcs, T, Y, lb=0.0, name="f_neg")

theta_p = w.addVars(B, T, Y, lb=0.0, name="theta_p")
theta_m = w.addVars(B, T, Y, lb=0.0, name="theta_m")

# ------------------------------------------------------------
# Variables mutables upper-level
# ------------------------------------------------------------

Hreq_ce = w.addVars(Y, lb=0.0, name="Hreq_ce")
xbar_ce = w.addVars(M, lb=0.0, name="xbar_ce")

for y in Y:
    base = Hreq_base[y]
    w.addConstr(Hreq_ce[y] >= 0.85 * base, name=f"adm_lb_Hreq[{y}]")
    w.addConstr(Hreq_ce[y] <= 1.15 * base, name=f"adm_ub_Hreq[{y}]")

for m in M:
    base = xbar_base[m]
    w.addConstr(xbar_ce[m] >= 0.85 * base, name=f"adm_lb_xbar[{m}]")
    w.addConstr(xbar_ce[m] <= 1.15 * base, name=f"adm_ub_xbar[{m}]")

# ------------------------------------------------------------
# Factibilidad primal Ax >= b
# ------------------------------------------------------------

R_primal = {}
RHS_expr = {}

for i in B:
    for t in T:
        for y in Y:
            expr = (
                gp.quicksum(p[g, t, y] for g in G_en_bus.get(i, []))
                + s[i, t, y]
                + gp.quicksum(
                    f_pos[k, j, t, y] - f_neg[k, j, t, y]
                    for (k, j) in lineas_entran[i]
                )
                - gp.quicksum(
                    f_pos[i2, j, t, y] - f_neg[i2, j, t, y]
                    for (i2, j) in lineas_salen[i]
                )
            )

            rhs_val = float(d_fw[(i, t, y)])

            name = f"bal_p_{i}_{t}_{y}"
            R_primal[name] = w.addConstr(expr >= rhs_val, name=name)
            RHS_expr[name] = rhs_val

            name = f"bal_n_{i}_{t}_{y}"
            R_primal[name] = w.addConstr(-expr >= -rhs_val, name=name)
            RHS_expr[name] = -rhs_val

for i in B:
    for t in T:
        for y in Y:
            rhs_val = -float(d_fw[(i, t, y)])
            name = f"ens_ub_{i}_{t}_{y}"
            R_primal[name] = w.addConstr(-s[i, t, y] >= rhs_val, name=name)
            RHS_expr[name] = rhs_val

for (i, j) in L_arcs:
    for t in T:
        for y in Y:
            Bij = float(Bline[(i, j)])

            flow = f_pos[i, j, t, y] - f_neg[i, j, t, y]
            theta_i = theta_p[i, t, y] - theta_m[i, t, y]
            theta_j = theta_p[j, t, y] - theta_m[j, t, y]

            expr = flow - Bij * (theta_i - theta_j)

            name = f"dc_p_{i}_{j}_{t}_{y}"
            R_primal[name] = w.addConstr(expr >= 0.0, name=name)
            RHS_expr[name] = 0.0

            name = f"dc_n_{i}_{j}_{t}_{y}"
            R_primal[name] = w.addConstr(-expr >= 0.0, name=name)
            RHS_expr[name] = 0.0

for (i, j) in L_arcs:
    for t in T:
        for y in Y:
            Fij = float(F0[(i, j)])
            flow = f_pos[i, j, t, y] - f_neg[i, j, t, y]

            name = f"flb_{i}_{j}_{t}_{y}"
            R_primal[name] = w.addConstr(flow >= -Fij, name=name)
            RHS_expr[name] = -Fij

            name = f"fub_{i}_{j}_{t}_{y}"
            R_primal[name] = w.addConstr(-flow >= -Fij, name=name)
            RHS_expr[name] = -Fij

for t in T:
    for y in Y:
        expr_slack = theta_p[bus_slack, t, y] - theta_m[bus_slack, t, y]

        name = f"slack_p_{t}_{y}"
        R_primal[name] = w.addConstr(expr_slack >= 0.0, name=name)
        RHS_expr[name] = 0.0

        name = f"slack_n_{t}_{y}"
        R_primal[name] = w.addConstr(-expr_slack >= 0.0, name=name)
        RHS_expr[name] = 0.0

for g in G:
    for t in T:
        cap_base = float(P_disp.get((g, t), Pmax[g]))

        for y in Y:
            rhs_val = -cap_base
            name = f"p_cap_{g}_{t}_{y}"
            R_primal[name] = w.addConstr(-p[g, t, y] >= rhs_val, name=name)
            RHS_expr[name] = rhs_val

for g in GS:
    pmin_g = float(Pmin.get(g, 0.0))

    for t in T:
        for y in Y:
            name = f"sync_lb_{g}_{t}_{y}"
            R_primal[name] = w.addConstr(
                p[g, t, y] - pmin_g * u[g, t, y] >= 0.0,
                name=name
            )
            RHS_expr[name] = 0.0

            name = f"sync_ub_{g}_{t}_{y}"
            R_primal[name] = w.addConstr(
                float(Pmax[g]) * u[g, t, y] - p[g, t, y] >= 0.0,
                name=name
            )
            RHS_expr[name] = 0.0

            name = f"u_ub_{g}_{t}_{y}"
            R_primal[name] = w.addConstr(-u[g, t, y] >= -1.0, name=name)
            RHS_expr[name] = -1.0

for m in M:
    for y in Y:
        name = f"x_ub_{m}_{y}"
        R_primal[name] = w.addConstr(-xcap[m, y] >= -xbar_ce[m], name=name)
        RHS_expr[name] = -xbar_ce[m]

        for t in T:
            name = f"hsyn_lim_{m}_{t}_{y}"
            R_primal[name] = w.addConstr(
                float(gamma_base[m]) * xcap[m, y] - h_syn[m, t, y] >= 0.0,
                name=name
            )
            RHS_expr[name] = 0.0

            name = f"ffr_lim_{m}_{t}_{y}"
            R_primal[name] = w.addConstr(
                float(kappa_base[m]) * xcap[m, y] - r_ffr[m, t, y] >= 0.0,
                name=name
            )
            RHS_expr[name] = 0.0

for t in T:
    for y in Y:
        lhs_H = (
            gp.quicksum(float(Hg.get(g, 0.0)) * u[g, t, y] for g in GS)
            + gp.quicksum(h_syn[m, t, y] for m in M)
        )

        name = f"sys_H_{t}_{y}"
        R_primal[name] = w.addConstr(lhs_H >= Hreq_ce[y], name=name)
        RHS_expr[name] = Hreq_ce[y]

        name = f"sys_R_{t}_{y}"
        R_primal[name] = w.addConstr(
            gp.quicksum(r_ffr[m, t, y] for m in M) >= float(Rreq[y]),
            name=name
        )
        RHS_expr[name] = float(Rreq[y])

# ------------------------------------------------------------
# Restricción D no dualizada
# ------------------------------------------------------------

for y in Y:
    if y >= y_star:
        w.addConstr(
            gp.quicksum(omega(t) * float(eg[g]) * p[g, t, y] for g in GE for t in T)
            <= eps_CO2(y),
            name=f"D_CO2_cap_{y}"
        )

w.update()

# ------------------------------------------------------------
# Variables duales
# ------------------------------------------------------------

ydual = {}

for name in R_primal.keys():
    ub_here = GRB.INFINITY

    if name.startswith("sys_H_"):
        ub_here = UB_H
    elif name.startswith("x_ub_"):
        ub_here = UB_xbar

    ydual[name] = w.addVar(
        lb=0.0,
        ub=ub_here,
        name=f"ydual[{name}]"
    )

w.update()

# ------------------------------------------------------------
# Costos del forward
# ------------------------------------------------------------

c_map_w = {}

for g in G:
    for t in T:
        for y in Y:
            c_map_w[p[g, t, y]] = omega(t) * float(cg[g])

for i in B:
    for t in T:
        for y in Y:
            c_map_w[s[i, t, y]] = omega(t) * float(VOLL)

for m in M:
    for y in Y:
        c_map_w[xcap[m, y]] = float(cinv[m])

primal_vars_w = (
    list(p.values())
    + list(u.values())
    + list(s.values())
    + list(xcap.values())
    + list(h_syn.values())
    + list(r_ffr.values())
    + list(f_pos.values())
    + list(f_neg.values())
    + list(theta_p.values())
    + list(theta_m.values())
)

# ------------------------------------------------------------
# Factibilidad dual A^T y <= c
# ------------------------------------------------------------

for v in primal_vars_w:
    lhs = gp.QuadExpr()

    for name, constr in R_primal.items():
        coeff = 0.0

        if isinstance(constr, gp.Constr):
            val = w.getCoeff(constr, v)

            if abs(val) > 1e-9:
                coeff = val

        elif isinstance(constr, gp.QConstr):
            qrow = w.getQCRow(constr)

            lin = qrow.getLinExpr()
            for jj in range(lin.size()):
                if lin.getVar(jj).sameAs(v):
                    coeff += lin.getCoeff(jj)

            for jj in range(qrow.size()):
                v1 = qrow.getVar1(jj)
                v2 = qrow.getVar2(jj)
                cq = qrow.getCoeff(jj)

                if v1.sameAs(v):
                    coeff += cq * v2
                elif v2.sameAs(v):
                    coeff += cq * v1

        if isinstance(coeff, (gp.Var, gp.LinExpr, gp.QuadExpr)):
            lhs += coeff * ydual[name]
        else:
            if abs(coeff) > 1e-9:
                lhs += coeff * ydual[name]

    w.addQConstr(
        lhs <= c_map_w.get(v, 0.0),
        name=f"df_{v.VarName}"
    )

# ------------------------------------------------------------
# Dualidad fuerte
# ------------------------------------------------------------

ctx = gp.quicksum(c_map_w.get(v, 0.0) * v for v in primal_vars_w)

bty = gp.QuadExpr()

for name in R_primal.keys():
    rhs = RHS_expr[name]

    if isinstance(rhs, (gp.Var, gp.LinExpr, gp.QuadExpr)):
        bty += rhs * ydual[name]
    else:
        bty += float(rhs) * ydual[name]

w.addQConstr(
    ctx - bty <= 0.0,
    name="StrongDuality"
)

# ------------------------------------------------------------
# Objetivo L1
# ------------------------------------------------------------

dHreq = w.addVars(Y, lb=0.0, name="dHreq")
dXbar = w.addVars(M, lb=0.0, name="dXbar")

for y in Y:
    base = Hreq_base[y]
    diff = Hreq_ce[y] - base

    w.addConstr(dHreq[y] >= diff, name=f"dHreq_pos[{y}]")
    w.addConstr(dHreq[y] >= -diff, name=f"dHreq_neg[{y}]")

for m in M:
    base = xbar_base[m]
    diff = xbar_ce[m] - base

    w.addConstr(dXbar[m] >= diff, name=f"dXbar_pos[{m}]")
    w.addConstr(dXbar[m] >= -diff, name=f"dXbar_neg[{m}]")

if USE_PERCENT_OBJECTIVE:
    obj_wcep = gp.LinExpr()

    for y in Y:
        base = abs(float(Hreq_base[y]))
        if base > 1e-9:
            obj_wcep += alpha_H * dHreq[y] / base
        else:
            obj_wcep += alpha_H * dHreq[y]

    for m in M:
        base = abs(float(xbar_base[m]))
        if base > 1e-9:
            obj_wcep += alpha_Xbar * dXbar[m] / base
        else:
            obj_wcep += alpha_Xbar * dXbar[m]

    w.setObjective(obj_wcep, GRB.MINIMIZE)

else:
    w.setObjective(
        alpha_H * dHreq.sum()
        + alpha_Xbar * dXbar.sum(),
        GRB.MINIMIZE
    )




print("\n⚠️ WCEP correrá sin PADM y sin warm start.")

# ============================================================
# 3) OPTIMIZAR WCEP FINAL
# ============================================================

print("\n" + "="*70)
print("RESOLVIENDO WCEP FINAL IEEE39")
print("="*70)

w.optimize()


# ============================================================
# 4) RESULTADOS WCEP FINAL
# ============================================================

print("\n" + "="*70)
print("RESULTADOS WCEP FINAL IEEE39")
print("="*70)
print(f"Status = {w.Status}")

if w.Status == GRB.OPTIMAL:
    print(f"✅ WCEP OPTIMAL. Obj = {w.ObjVal:.8f}")

elif w.Status == GRB.SUBOPTIMAL:
    print(f"⚠️ WCEP SUBOPTIMAL. Obj = {w.ObjVal:.8f}")

elif w.Status == GRB.TIME_LIMIT:
    print("⏱️ WCEP terminó por límite de tiempo.")
    if w.SolCount > 0:
        print(f"Mejor Obj = {w.ObjVal:.8f}")
        print(f"Best bound = {w.ObjBound:.8f}")
        print(f"MIPGap = {w.MIPGap:.6%}")

elif w.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
    print("❌ WCEP infactible o INF_OR_UNBD.")
else:
    if w.SolCount > 0:
        print(f"⚠️ WCEP terminó con status {w.Status}, pero tiene solución.")
        print(f"Obj = {w.ObjVal:.8f}")
    else:
        print(f"⚠️ WCEP terminó con status {w.Status} y sin solución.")

if w.SolCount > 0:

    tol = 1e-6

    print("\n" + "="*70)
    print("CAMBIOS ÓPTIMOS EN PARÁMETROS MUTABLES")
    print("="*70)

    print("\n--- Hreq_ce ---")
    hubo_cambio = False

    for y in Y:
        val = float(Hreq_ce[y].X)
        base = float(Hreq_base[y])

        if abs(val - base) > tol:
            hubo_cambio = True
            pct = 100.0 * (val / base - 1.0) if abs(base) > 1e-9 else float("inf")
            print(f"{y}: {base:.4f} -> {val:.4f}   ({pct:+.2f}%)")

    if not hubo_cambio:
        print("Sin cambios relevantes.")

    print("\n--- xbar_ce ---")
    hubo_cambio = False

    for m in M:
        val = float(xbar_ce[m].X)
        base = float(xbar_base[m])

        if abs(val - base) > tol:
            hubo_cambio = True
            pct = 100.0 * (val / base - 1.0) if abs(base) > 1e-9 else float("inf")
            print(f"{m}: {base:.4f} -> {val:.4f}   ({pct:+.2f}%)")

    if not hubo_cambio:
        print("Sin cambios relevantes.")

    print("\n" + "="*70)
    print("REVISIÓN DUALIDAD FUERTE")
    print("="*70)

    ctx_val = ctx.getValue()
    bty_val = bty.getValue()

    print(f"c^T x  = {ctx_val:.8f}")
    print(f"b^T y  = {bty_val:.8f}")
    print(f"gap    = {ctx_val - bty_val:.6e}")

    print("\n" + "="*70)
    print("EMISIONES ANUALES DE LA SOLUCIÓN WCEP")
    print("="*70)

    for y in Y:
        emis_y = sum(
            omega(t) * float(eg[g]) * float(p[g, t, y].X)
            for g in GE
            for t in T
        )

        if y >= y_star:
            holgura = eps_CO2(y) - emis_y
            print(
                f"Año {y}: emisiones = {emis_y:.6f} | "
                f"límite D = {eps_CO2(y):.6f} | "
                f"holgura = {holgura:.6f}"
            )
        else:
            print(
                f"Año {y}: emisiones = {emis_y:.6f} | "
                f"sin restricción D"
            )

    print("\n" + "="*70)
    print("CHEQUEO H Y R")
    print("="*70)

    for y in Y:
        for t in T:
            H_sync_val = sum(
                float(Hg.get(g, 0.0)) * float(u[g, t, y].X)
                for g in GS
            )

            H_syn_val = sum(
                float(h_syn[m, t, y].X)
                for m in M
            )

            H_total = H_sync_val + H_syn_val
            H_lim = float(Hreq_ce[y].X)

            R_total = sum(
                float(r_ffr[m, t, y].X)
                for m in M
            )

            R_lim = float(Rreq[y])

            print(
                f"(y={y}, t={t}) | "
                f"H_total={H_total:.6f}, Hreq_ce={H_lim:.6f}, margen_H={H_total - H_lim:.6e} | "
                f"R_total={R_total:.6f}, Rreq={R_lim:.6f}, margen_R={R_total - R_lim:.6e}"
            )

    print("\n" + "="*70)
    print("REVISIÓN DE DUALES ACOTADAS")
    print("="*70)

    hits_H = 0
    hits_xbar = 0

    for name, var in ydual.items():
        if name.startswith("sys_H_"):
            if abs(var.X - UB_H) <= 1e-6 * max(1.0, UB_H):
                hits_H += 1

        elif name.startswith("x_ub_"):
            if abs(var.X - UB_xbar) <= 1e-6 * max(1.0, UB_xbar):
                hits_xbar += 1

    print(f"Duales sys_H pegadas a cota   : {hits_H}")
    print(f"Duales x_ub pegadas a cota    : {hits_xbar}")

else:
    print("\nNo hay solución WCEP disponible para reportar.")

# ============================================================
# REGISTRO DE RESULTADOS WCEP
# ============================================================

print("\n" + "="*70)
print("REGISTRO DE OPTIMIZACIÓN WCEP")
print("="*70)

runtime = float(w.Runtime)
status = int(w.Status)
sol_count = int(w.SolCount)

print(f"Status Gurobi = {status}")
print(f"Tiempo total  = {runtime:.2f} s = {runtime/60:.2f} min = {runtime/3600:.2f} h")
print(f"Soluciones encontradas = {sol_count}")

if sol_count > 0:
    obj_val = float(w.ObjVal)
    obj_bound = float(w.ObjBound)
    mip_gap = float(w.MIPGap)

    print(f"Mejor objetivo encontrado = {obj_val:.8f}")
    print(f"Mejor bound               = {obj_bound:.8f}")
    print(f"MIPGap                    = {mip_gap:.6e}")
    print(f"MIPGap (%)                = {100*mip_gap:.4f}%")

    registro_wcep = {
        "status": status,
        "runtime_seconds": runtime,
        "runtime_minutes": runtime / 60,
        "runtime_hours": runtime / 3600,
        "sol_count": sol_count,
        "obj_val": obj_val,
        "obj_bound": obj_bound,
        "mip_gap": mip_gap,
        "mip_gap_percent": 100 * mip_gap,
    }

else:
    print("No se encontró solución factible.")
    registro_wcep = {
        "status": status,
        "runtime_seconds": runtime,
        "runtime_minutes": runtime / 60,
        "runtime_hours": runtime / 3600,
        "sol_count": sol_count,
        "obj_val": None,
        "obj_bound": None,
        "mip_gap": None,
        "mip_gap_percent": None,
    }
# Tiempo total real de toda la celda/script
t_total_script = time.time() - t0_total_script
print("\n" + "="*70)
print("TIEMPO TOTAL DE CELDA / SCRIPT")
print("="*70)
print(f"Tiempo Gurobi optimize() = {float(w.Runtime):.2f} s")
print(f"Tiempo total script      = {t_total_script:.2f} s = {t_total_script/60:.2f} min = {t_total_script/3600:.2f} h")

# Guardar registro en DataFrame si pandas está disponible
df_registro_wcep = pd.DataFrame([registro_wcep])
display(df_registro_wcep)

