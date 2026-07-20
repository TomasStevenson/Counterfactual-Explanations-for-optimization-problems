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



# %% ================== celda 21 del notebook ==================
import os
import pandas as pd
import numpy as np

# ============================================================
# 0) CONFIG: usar CSV IEEE14 finales
# ============================================================
PATH = "."

PATH_GEN   = os.path.join(PATH, "generadores_procesados_ieee14_final.csv")
PATH_LINE  = os.path.join(PATH, "lineas_procesadas_ieee14.csv")
PATH_LOAD  = os.path.join(PATH, "demanda_nodal_ieee14_final.csv")
PATH_SOLAR = os.path.join(PATH, "perfil_solar_ieee14_final.csv")

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
    "periodo": "Period",
    "demanda_mw": "Nodal_Load_MW"
})

# ----- LINEAS -----
df_lines = df_lines.rename(columns={
    "desde": "from_bus",
    "hacia": "to_bus",
    "x": "Reactancia_X",
    "fmax": "F_max"
})

# ----- PERFIL SOLAR -----
df_solar = df_solar.rename(columns={
    "id_gen": "Gen_ID",
    "periodo": "Period",
    "disponibilidad_mw": "P_Available_MW"
})

# ============================================================
# 3) CHEQUEO DE COLUMNAS OBLIGATORIAS
# ============================================================
cols_gen_req = ["id_gen", "bus_id", "Pmax", "Pmin", "Costo_Var", "Inercia_H",
                "Emisiones_tCO2_MWh", "Es_IBR", "Gamma", "Kappa", "Costo_Inv"]
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
year_growth = {1: 1.00, 2: 1.05, 3: 1.10}

B = sorted(df_load["Bus ID"].unique().tolist())
T = sorted(df_load["Period"].unique().tolist())

G  = df_gen["id_gen"].tolist()
GS = df_gen[df_gen["Es_IBR"] == 0]["id_gen"].tolist()
M  = df_gen[df_gen["Es_IBR"] == 1]["id_gen"].tolist()
GE = df_gen[df_gen["Emisiones_tCO2_MWh"] > 0]["id_gen"].tolist()

print(f"Sets definidos: |B|={len(B)}, |T|={len(T)}, |G|={len(G)} (|GS|={len(GS)}, |M|={len(M)}, |GE|={len(GE)})")

# ============================================================
# 6) PARÁMETROS DE GENERADORES
# ============================================================
Pmax = df_gen.set_index("id_gen")["Pmax"].to_dict()
Pmin = df_gen.set_index("id_gen")["Pmin"].to_dict()
cg   = df_gen.set_index("id_gen")["Costo_Var"].to_dict()
eg   = df_gen.set_index("id_gen")["Emisiones_tCO2_MWh"].to_dict()
cinv = df_gen.set_index("id_gen")["Costo_Inv"].to_dict()

Hg = {g: float(df_gen.set_index("id_gen")["Inercia_H"].to_dict().get(g, 0.0)) for g in GS}

g_bus = df_gen.set_index("id_gen")["bus_id"].to_dict()

G_en_bus = {i: [] for i in B}
for g in G:
    bi = int(g_bus[g])
    if bi in G_en_bus:
        G_en_bus[bi].append(g)

# ============================================================
# 7) PARÁMETROS IBR
# ============================================================
gamma_base = df_gen.set_index("id_gen")["Gamma"].to_dict()
kappa_base = df_gen.set_index("id_gen")["Kappa"].to_dict()

gamma_max = df_gen.set_index("id_gen")["Gamma_max"].to_dict() if "Gamma_max" in df_gen.columns else {m: 0.5 for m in M}
kappa_max = df_gen.set_index("id_gen")["Kappa_max"].to_dict() if "Kappa_max" in df_gen.columns else {m: 0.5 for m in M}

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
            d[(i, t, y)] = float(raw_demand.get((i, t), 0.0)) * factor

# ============================================================
# 10) PERFIL RENOVABLE
# ============================================================
P_disp = {(g, t): float(Pmax[g]) for g in G for t in T}

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
D_peak = {y: max(sum(d[(i, t, y)] for i in B) for t in T) for y in Y}

max_hsyn = sum(float(gamma_base.get(m, 0.0)) * float(xbar[m]) for m in M) if len(M) > 0 else 0.0
max_sync = sum(float(Hg.get(g, 0.0)) for g in GS) if len(GS) > 0 else 0.0
cap_total = max_hsyn + max_sync

alpha_H = 0.35
H_floor = 0.05

k_H = (alpha_H * cap_total / D_peak[1]) if D_peak[1] > 1e-9 else 0.0
Hreq = {}
for y in Y:
    base = k_H * D_peak[y]
    floor = H_floor * cap_total
    Hreq[y] = max(base, floor)

MaxR = sum(float(kappa_base.get(m, 0.0)) * float(xbar[m]) for m in M) if len(M) > 0 else 0.0
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


# %% ================== celda 24 del notebook ==================
# ============================================================
# Celda: Resumen de dimensionalidad / tamaño del problema
# Adaptada a tu FW actual para IEEE14
# Requiere: B, T, Y, G, GS, M, GE, L
# Opcional: fw_base
# ============================================================

import gurobipy as gp

def safe_len(x):
    try:
        return len(x)
    except Exception:
        return None

def mul(*vals):
    out = 1
    for v in vals:
        out *= int(v)
    return out

# Si existe L_arcs úsalo; si no, usar L
if "L_arcs" in globals():
    _L = L_arcs
else:
    _L = gp.tuplelist(L)

nB  = safe_len(B)
nT  = safe_len(T)
nY  = safe_len(Y)
nG  = safe_len(G)
nGS = safe_len(GS)
nM  = safe_len(M)
nGE = safe_len(GE)
nL  = safe_len(_L)

print("\n==============================")
print("DIMENSIONALIDAD DEL PROBLEMA")
print("==============================")
print(f"|B|  (buses)        = {nB}")
print(f"|L|  (líneas/arcos) = {nL}")
print(f"|T|  (periodos)     = {nT}")
print(f"|Y|  (años)         = {nY}")
print(f"|G|  (generadores)  = {nG}")
print(f"|GS| (síncronos)    = {nGS}")
print(f"|M|  (IBR)          = {nM}")
print(f"|GE| (emisores CO2) = {nGE}")

# ============================================================
# Variables (estimadas según tu formulación actual)
# ============================================================
n_p     = mul(nG,  nT, nY) if None not in [nG,  nT, nY] else None
n_u     = mul(nGS, nT, nY) if None not in [nGS, nT, nY] else None
n_s     = mul(nB,  nT, nY) if None not in [nB,  nT, nY] else None
n_theta = mul(nB,  nT, nY) if None not in [nB,  nT, nY] else None
n_f     = mul(nL,  nT, nY) if None not in [nL,  nT, nY] else None
n_x     = mul(nM,  nY)     if None not in [nM,  nY]     else None
n_h     = mul(nM,  nT, nY) if None not in [nM,  nT, nY] else None
n_r     = mul(nM,  nT, nY) if None not in [nM,  nT, nY] else None

print("\n--- Tamaño (variables) según la formulación ---")
print(f"#p      = {n_p}")
print(f"#u      = {n_u}")
print(f"#s      = {n_s}")
print(f"#theta  = {n_theta}")
print(f"#f      = {n_f}")
print(f"#x      = {n_x}")
print(f"#h_syn  = {n_h}")
print(f"#r_ffr  = {n_r}")

n_vars_est = sum(v for v in [n_p, n_u, n_s, n_theta, n_f, n_x, n_h, n_r] if v is not None)
print(f"TOTAL vars (estimado) = {n_vars_est}")

# ============================================================
# Restricciones (estimadas según tu formulación actual)
# ============================================================
n_bal   = mul(nB,  nT, nY) if None not in [nB,  nT, nY] else None      # balance nodal
n_ensub = mul(nB,  nT, nY) if None not in [nB,  nT, nY] else None      # ENS <= demanda
n_dc    = mul(nL,  nT, nY) if None not in [nL,  nT, nY] else None      # flujo DC
n_line  = 2 * n_dc if n_dc is not None else None                       # limites ±Fmax
n_slack = mul(nT,  nY)     if None not in [nT,  nY]     else None      # barra slack
n_pdisp = mul(nG,  nT, nY) if None not in [nG,  nT, nY] else None      # p <= Pdisp
n_sync  = 2 * mul(nGS, nT, nY) if None not in [nGS, nT, nY] else None  # Pmin/Pmax síncronos
n_co2   = 0                                                            # según tu versión actual, comentada
n_xub   = mul(nM,  nY)     if None not in [nM,  nY]     else None      # x <= xbar
n_hcap  = mul(nM,  nT, nY) if None not in [nM,  nT, nY] else None      # h_syn <= gamma*x
n_rcap  = mul(nM,  nT, nY) if None not in [nM,  nT, nY] else None      # r_ffr <= kappa*x
n_inreq = mul(nT,  nY)     if None not in [nT,  nY]     else None      # req. inercia
n_rreq  = mul(nT,  nY)     if None not in [nT,  nY]     else None      # req. FFR

print("\n--- Tamaño (restricciones) según la formulación ---")
print(f"#balance      = {n_bal}")
print(f"#ens_ub       = {n_ensub}")
print(f"#dcflow       = {n_dc}")
print(f"#line_limits  = {n_line}")
print(f"#slack        = {n_slack}")
print(f"#pdisp        = {n_pdisp}")
print(f"#sync_min/max = {n_sync}")
print(f"#CO2_cap      = {n_co2}")
print(f"#x_ub         = {n_xub}")
print(f"#hsyn_cap     = {n_hcap}")
print(f"#ffr_cap      = {n_rcap}")
print(f"#inertia_req  = {n_inreq}")
print(f"#ffr_req      = {n_rreq}")

n_cons_est = sum(v for v in [
    n_bal, n_ensub, n_dc, n_line, n_slack, n_pdisp,
    n_sync, n_co2, n_xub, n_hcap, n_rcap, n_inreq, n_rreq
] if v is not None)

print(f"TOTAL cons (estimado) = {n_cons_est}")

# ============================================================
# Comparar con el modelo real en Gurobi
# ============================================================
if "fw_base" in globals():
    try:
        fw_base.update()
        print("\n--- Conteo REAL del modelo en Gurobi (fw_base) ---")
        print(f"fw_base.NumVars    = {fw_base.NumVars}")
        print(f"fw_base.NumConstrs = {fw_base.NumConstrs}")
    except Exception as e:
        print("\n(No pude leer fw_base.NumVars/fw_base.NumConstrs):", e)
else:
    print("\n(Nota) No existe 'fw_base' en globals(). Corre primero la celda del Forward Problem.")


# %% ================== celda 28 del notebook ==================
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# ============================================================
# FINAL IEEE14
# PASO 1) FORWARD PROBLEM - FORMULACIÓN ORIGINAL
# DATASET CORREGIDO
#
# Correcciones incluidas:
#
# 1) Escalamiento de demanda para evitar déficit estructural.
#    El dataset original tenía demanda pico mayor que capacidad instalada,
#    generando ENS artificial.
#
# 2) Hreq queda en su posición original:
#       sum_g Hg[g] u[g,t,y] + sum_m h_syn[m,t,y] >= Hreq[y]
#
#    No se usa zH.
#    No se mueve Hreq hacia A.
#
# 3) Se mantiene el diccionario original d intacto.
#    El modelo usa d_fw.
# ============================================================


# ============================================================
# 0) OPCIONES DE CORRECCIÓN DEL DATASET IEEE14
# ============================================================

# Recomendado para IEEE14:
#   "auto_capacity" calcula un factor para que la demanda pico quede
#   bajo la capacidad instalada del sistema.
#
# También puedes fijar manualmente:
#   DEMAND_SCALE_MODE = "manual"
#   DEMAND_SCALE_MANUAL = 0.713
#
DEMAND_SCALE_MODE = "auto_capacity"   # "auto_capacity", "manual", "none"
DEMAND_SCALE_MANUAL = 0.713

# Margen de seguridad respecto a capacidad instalada.
# 1.00 deja la demanda pico igual a la capacidad.
# 0.98 deja un 2% de margen operativo.
DEMAND_CAP_MARGIN = 0.98

# ============================================================
# PESOS TEMPORALES PARA REPRESENTAR EL AÑO
# ============================================================
# T contiene pocas horas representativas del día.
# En vez de expandir T a 365*|T| periodos, ponderamos cada t.
#
# Caso base solicitado:
#   el mismo perfil diario se repite 365 veces.
#   Si cada t representa una hora puntual del día, usar TIME_BLOCK_HOURS = 1.0.
#   Si cada t representa un bloque de 6 horas, usar TIME_BLOCK_HOURS = 6.0.
#
# Con T = {1,7,13,19}:
#   TIME_BLOCK_HOURS = 1.0  -> sum_t omega_t = 1460 h equivalentes
#   TIME_BLOCK_HOURS = 6.0  -> sum_t omega_t = 8760 h equivalentes
N_DAYS_REPRESENTED = 365
TIME_BLOCK_HOURS = 1.0

omega_t = {t: float(N_DAYS_REPRESENTED * TIME_BLOCK_HOURS) for t in T}

print("\n==============================")
print("PESOS TEMPORALES FW IEEE14")
print("==============================")
print(f"Días representados       = {N_DAYS_REPRESENTED}")
print(f"Horas por periodo t      = {TIME_BLOCK_HOURS}")
print(f"Peso omega_t por periodo = {next(iter(omega_t.values())) if len(omega_t) > 0 else 0.0}")
print(f"Horas equivalentes total = {sum(omega_t.values()):.2f}")

# Para mantener el experimento reproducible, no se modifica d.
# Se construye un nuevo diccionario d_fw.
d_original = {k: float(v) for k, v in d.items()}


# ============================================================
# 1) ROBUSTECER LÍNEAS
# ============================================================

L_arcs = gp.tuplelist(L)


# ============================================================
# 2) FIX: reconstruir B si fue pisado
# ============================================================

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


# ============================================================
# 3) TOPOLOGÍA ROBUSTA
# ============================================================

lineas_entran = {bus: [] for bus in B}
lineas_salen  = {bus: [] for bus in B}

for arc in L_arcs:
    if not (isinstance(arc, tuple) and len(arc) == 2):
        raise TypeError(f"Arco inválido en L_arcs: {arc} (type={type(arc)})")

    a, b = arc

    if b in lineas_entran:
        lineas_entran[b].append((a, b))

    if a in lineas_salen:
        lineas_salen[a].append((a, b))


# ============================================================
# 4) CORRECCIÓN DE DEMANDA IEEE14
# ============================================================

print("\n==============================")
print("CORRECCIÓN DATASET IEEE14")
print("==============================")

# Demanda total original por periodo/año
demanda_original_ty = {}

for t in T:
    for y in Y:
        demanda_original_ty[(t, y)] = sum(
            d_original[(i, t, y)]
            for i in B
            if (i, t, y) in d_original
        )

peak_dem_original = max(demanda_original_ty.values())

# Capacidad instalada total
cap_total_pmax = sum(float(Pmax[g]) for g in G)

# Capacidad disponible por periodo si existe P_disp
cap_disp_ty = {}

for t in T:
    for y in Y:
        cap_disp_ty[(t, y)] = sum(
            float(P_disp.get((g, t), Pmax[g]))
            for g in G
        )

min_cap_disp = min(cap_disp_ty.values())

print(f"Demanda pico original     = {peak_dem_original:.6f}")
print(f"Capacidad instalada Pmax  = {cap_total_pmax:.6f}")
print(f"Capacidad disp. mínima    = {min_cap_disp:.6f}")

if DEMAND_SCALE_MODE == "none":
    demand_scale_factor = 1.0

elif DEMAND_SCALE_MODE == "manual":
    demand_scale_factor = float(DEMAND_SCALE_MANUAL)

elif DEMAND_SCALE_MODE == "auto_capacity":
    if peak_dem_original <= 1e-9:
        demand_scale_factor = 1.0
    else:
        # Usamos Pmax total, consistente con el diagnóstico original.
        # Además aplicamos un pequeño margen para evitar operar exactamente
        # en el borde de capacidad.
        demand_scale_factor = min(
            1.0,
            DEMAND_CAP_MARGIN * cap_total_pmax / peak_dem_original
        )

else:
    raise ValueError("DEMAND_SCALE_MODE debe ser 'auto_capacity', 'manual' o 'none'.")

print(f"Modo escalamiento demanda = {DEMAND_SCALE_MODE}")
print(f"Factor demanda usado      = {demand_scale_factor:.8f}")

# Construir demanda corregida para el modelo
d_fw = {}

for i in B:
    for t in T:
        for y in Y:
            if (i, t, y) not in d_original:
                raise KeyError(f"Falta d[{i},{t},{y}] en el diccionario original.")

            d_fw[(i, t, y)] = demand_scale_factor * d_original[(i, t, y)]

# Resumen demanda corregida
demanda_fw_ty = {}

for t in T:
    for y in Y:
        demanda_fw_ty[(t, y)] = sum(d_fw[(i, t, y)] for i in B)

peak_dem_fw = max(demanda_fw_ty.values())

print(f"Demanda pico corregida    = {peak_dem_fw:.6f}")
print(f"Margen Pmax - peak demand = {cap_total_pmax - peak_dem_fw:.6f}")

if peak_dem_fw > cap_total_pmax + 1e-6:
    print("⚠️ Aún hay demanda pico mayor que Pmax total.")
else:
    print("✅ Demanda corregida queda dentro de la capacidad instalada.")

if peak_dem_fw > min_cap_disp + 1e-6:
    print("⚠️ La demanda pico corregida puede superar la capacidad disponible P_disp en algún periodo.")
    print("   Esto puede ser normal si P_disp modela renovables variables, pero conviene revisar.")
else:
    print("✅ Demanda corregida queda dentro de la capacidad disponible mínima.")


# ============================================================
# 5) PRE-CHECKS
# ============================================================

print("\n==============================")
print("PRE-CHECKS FORWARD IEEE14 - FORMULACIÓN ORIGINAL")
print("==============================")

missing_d = [
    (i, t, y)
    for i in B
    for t in T
    for y in Y
    if (i, t, y) not in d_fw
]

if missing_d:
    print(f"⚠️ Faltan {len(missing_d)} entradas en d_fw[(i,t,y)]. Ejemplo:", missing_d[:5])
else:
    print("✅ d_fw[(i,t,y)] completo para B,T,Y.")

for y in Y:
    for t in T:
        dem_tot = sum(d_fw[(i, t, y)] for i in B)
        cap_tot = sum(float(Pmax[g]) for g in G)
        cap_disp = sum(float(P_disp.get((g, t), Pmax[g])) for g in G)

        if cap_tot + 1e-6 < dem_tot:
            print(
                f"⚠️ Pmax total {cap_tot:.2f} < demanda corregida {dem_tot:.2f} "
                f"en (t={t}, y={y})."
            )

        if cap_disp + 1e-6 < dem_tot:
            print(
                f"⚠️ Capacidad disponible {cap_disp:.2f} < demanda corregida {dem_tot:.2f} "
                f"en (t={t}, y={y})."
            )

max_sync_H = sum(float(Hg.get(g, 0.0)) for g in GS)

gamma_fwd_check = gamma_base.copy()
max_hsyn = (
    sum(float(gamma_fwd_check[m]) * float(xbar[m]) for m in M)
    if len(M) > 0
    else 0.0
)

max_H_total = max_sync_H + max_hsyn

kappa_fwd_check = kappa_base.copy()
max_R_total = (
    sum(float(kappa_fwd_check[m]) * float(xbar[m]) for m in M)
    if len(M) > 0
    else 0.0
)

print(
    f"Max H posible (sync + hsyn) = {max_H_total:.3f} "
    f"(sync={max_sync_H:.3f}, hsyn={max_hsyn:.3f})"
)

print(f"Max R posible (FFR)         = {max_R_total:.3f}")

for y in Y:
    if Hreq[y] > max_H_total + 1e-6:
        print(
            f"❌ ALERTA: Hreq[{y}]={Hreq[y]:.3f} > MaxH={max_H_total:.3f} "
            f"=> INFEASIBLE por inercia."
        )

    if Rreq[y] > max_R_total + 1e-6:
        print(
            f"❌ ALERTA: Rreq[{y}]={Rreq[y]:.3f} > MaxR={max_R_total:.3f} "
            f"=> INFEASIBLE por FFR."
        )


# ============================================================
# 6) MODELO FORWARD IEEE14 - FORMULACIÓN ORIGINAL
# ============================================================

fw_base = gp.Model("FW_base_IEEE14_original_dataset_corregido_con_pesos")
fw_base.Params.OutputFlag = 1

# Time limit opcional para el Forward. Para dejarlo sin límite, usar None.
FW_TIME_LIMIT_SECONDS = None
if FW_TIME_LIMIT_SECONDS is not None:
    fw_base.Params.TimeLimit = FW_TIME_LIMIT_SECONDS


# ============================================================
# 7) VARIABLES
# ============================================================

p = fw_base.addVars(
    G, T, Y,
    lb=0.0,
    vtype=GRB.CONTINUOUS,
    name="p"
)

u = fw_base.addVars(
    GS, T, Y,
    lb=0.0,
    ub=1.0,
    vtype=GRB.CONTINUOUS,
    name="u"
)

s = fw_base.addVars(
    B, T, Y,
    lb=0.0,
    vtype=GRB.CONTINUOUS,
    name="s"
)

theta = fw_base.addVars(
    B, T, Y,
    lb=-GRB.INFINITY,
    vtype=GRB.CONTINUOUS,
    name="theta"
)

f = fw_base.addVars(
    L_arcs, T, Y,
    lb=-GRB.INFINITY,
    vtype=GRB.CONTINUOUS,
    name="f"
)

x = fw_base.addVars(
    M, Y,
    lb=0.0,
    vtype=GRB.CONTINUOUS,
    name="x"
)

h_syn = fw_base.addVars(
    M, T, Y,
    lb=0.0,
    vtype=GRB.CONTINUOUS,
    name="h_syn"
)

r_ffr = fw_base.addVars(
    M, T, Y,
    lb=0.0,
    vtype=GRB.CONTINUOUS,
    name="r_ffr"
)


# ============================================================
# 8) RESTRICCIONES
# ============================================================

# ------------------------------------------------------------
# 8.1) Balance nodal
# ------------------------------------------------------------
for i in B:
    for t in T:
        for y in Y:
            gen = gp.quicksum(
                p[g, t, y]
                for g in G_en_bus.get(i, [])
            )

            inflow = gp.quicksum(
                f[k, i, t, y]
                for (k, i) in lineas_entran[i]
            )

            outflow = gp.quicksum(
                f[i, kk, t, y]
                for (i, kk) in lineas_salen[i]
            )

            fw_base.addConstr(
                gen - d_fw[(i, t, y)] + s[i, t, y] + inflow - outflow == 0,
                name=f"balance[{i},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.2) Cota superior ENS
# ------------------------------------------------------------
for i in B:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                s[i, t, y] <= d_fw[(i, t, y)],
                name=f"ens_ub[{i},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.3) Flujo DC
# ------------------------------------------------------------
for (i, j) in L_arcs:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                f[i, j, t, y]
                == Bline[(i, j)] * (theta[i, t, y] - theta[j, t, y]),
                name=f"dcflow[{i},{j},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.4) Límites de flujo
# ------------------------------------------------------------
for (i, j) in L_arcs:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                f[i, j, t, y] <= F0[(i, j)],
                name=f"f_ub[{i},{j},{t},{y}]"
            )

            fw_base.addConstr(
                f[i, j, t, y] >= -F0[(i, j)],
                name=f"f_lb[{i},{j},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.5) Bus slack
# ------------------------------------------------------------
bus_slack = int(sorted(B)[0])

for t in T:
    for y in Y:
        fw_base.addConstr(
            theta[bus_slack, t, y] == 0.0,
            name=f"slack[{t},{y}]"
        )

print(f"Bus slack usado: {bus_slack}")


# ------------------------------------------------------------
# 8.6) Disponibilidad de generación
# ------------------------------------------------------------
for g in G:
    for t in T:
        cap_gt = float(P_disp.get((g, t), Pmax[g]))

        for y in Y:
            fw_base.addConstr(
                p[g, t, y] <= cap_gt,
                name=f"pdisp[{g},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.7) Restricciones de generadores síncronos
# ------------------------------------------------------------
for g in GS:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                p[g, t, y] >= float(Pmin.get(g, 0.0)) * u[g, t, y],
                name=f"sync_min[{g},{t},{y}]"
            )

            fw_base.addConstr(
                p[g, t, y] <= float(Pmax[g]) * u[g, t, y],
                name=f"sync_max[{g},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.8) Capacidad máxima de inversión IBR
# ------------------------------------------------------------
for m in M:
    for y in Y:
        fw_base.addConstr(
            x[m, y] <= float(xbar[m]),
            name=f"x_ub[{m},{y}]"
        )


# ------------------------------------------------------------
# 8.9) Capacidad de inercia sintética
# ------------------------------------------------------------
for m in M:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                h_syn[m, t, y] <= float(gamma_base[m]) * x[m, y],
                name=f"hsyn_cap[{m},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.10) Capacidad de FFR
# ------------------------------------------------------------
for m in M:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                r_ffr[m, t, y] <= float(kappa_base[m]) * x[m, y],
                name=f"ffr_cap[{m},{t},{y}]"
            )


# ------------------------------------------------------------
# 8.11) Requerimiento de inercia - FORMULACIÓN ORIGINAL
# ------------------------------------------------------------
for t in T:
    for y in Y:
        fw_base.addConstr(
            gp.quicksum(
                float(Hg.get(g, 0.0)) * u[g, t, y]
                for g in GS
            )
            + gp.quicksum(
                h_syn[m, t, y]
                for m in M
            )
            >= float(Hreq[y]),
            name=f"inertia_req[{t},{y}]"
        )


# ------------------------------------------------------------
# 8.12) Requerimiento de FFR
# ------------------------------------------------------------
for t in T:
    for y in Y:
        fw_base.addConstr(
            gp.quicksum(
                r_ffr[m, t, y]
                for m in M
            ) >= float(Rreq[y]),
            name=f"ffr_req[{t},{y}]"
        )


# ============================================================
# 9) ACTUALIZAR MODELO
# ============================================================

fw_base.update()

print(f"\nVariables: {fw_base.NumVars} | Restricciones: {fw_base.NumConstrs}")


# ============================================================
# 10) FUNCIÓN OBJETIVO
# ============================================================

obj_oper = gp.quicksum(
    float(omega_t[t]) * float(cg[g]) * p[g, t, y]
    for g in G
    for t in T
    for y in Y
)

obj_ens = float(VOLL) * gp.quicksum(
    float(omega_t[t]) * s[i, t, y]
    for i in B
    for t in T
    for y in Y
)

obj_inv = gp.quicksum(
    float(cinv[m]) * x[m, y]
    for m in M
    for y in Y
)

fw_base.setObjective(
    obj_oper + obj_ens + obj_inv,
    GRB.MINIMIZE
)


# ============================================================
# 11) RESOLVER
# ============================================================

fw_base.optimize()


# ============================================================
# 12) DIAGNÓSTICO BÁSICO
# ============================================================

if fw_base.Status == GRB.INFEASIBLE:
    print("\n❌ Modelo INFEASIBLE.")
    print("No se escribió archivo IIS/ILP.")

elif fw_base.Status == GRB.OPTIMAL or fw_base.SolCount > 0:
    if fw_base.Status == GRB.OPTIMAL:
        print("\n✅ Modelo resuelto óptimamente.")
    else:
        print(f"\n⚠️ Modelo terminó con status {fw_base.Status}, pero tiene solución factible.")

    print(f"Valor objetivo = {fw_base.ObjVal:,.4f}")
    if hasattr(fw_base, 'ObjBound'):
        print(f"Best bound     = {fw_base.ObjBound:,.4f}")
    if hasattr(fw_base, 'MIPGap'):
        print(f"Gap            = {fw_base.MIPGap:.6f}")

    ens_total_no_pond = sum(
        s[i, t, y].X
        for i in B
        for t in T
        for y in Y
    )

    ens_total_pond = sum(
        float(omega_t[t]) * s[i, t, y].X
        for i in B
        for t in T
        for y in Y
    )

    print(f"ENS total no ponderado = {ens_total_no_pond:.6f}")
    print(f"ENS total ponderado    = {ens_total_pond:.6f} MWh-equivalente")

    ens_total = ens_total_pond

    if ens_total <= 1e-6:
        print("✅ ENS corregido: el Forward queda sin déficit artificial.")
    else:
        print("⚠️ Aún existe ENS positivo. Revisar P_disp, red o límites de línea.")

else:
    print(f"\n⚠️ Modelo terminó con status Gurobi = {fw_base.Status}")


# ============================================================
# 13) RESULTADOS FORWARD IEEE14 - FORMULACIÓN ORIGINAL
# ============================================================

if fw_base.Status == GRB.OPTIMAL or fw_base.SolCount > 0:

    print("\n==============================")
    print("RESULTADOS FW_base IEEE14 - DATASET CORREGIDO")
    print("==============================")
    print(f"Valor objetivo total = {fw_base.ObjVal:,.4f}")
    print(f"Factor demanda usado = {demand_scale_factor:.8f}")

    # --------------------------------------------------------
    # 13.1) Resumen demanda original vs corregida
    # --------------------------------------------------------
    filas_dem = []

    for t in T:
        for y in Y:
            dem_orig = sum(d_original[(i, t, y)] for i in B)
            dem_corr = sum(d_fw[(i, t, y)] for i in B)

            filas_dem.append({
                "t": t,
                "y": y,
                "demanda_original": dem_orig,
                "demanda_corregida": dem_corr,
                "omega_t": float(omega_t[t]),
                "demanda_corregida_ponderada_MWh": float(omega_t[t]) * dem_corr,
                "factor": demand_scale_factor,
                "capacidad_Pmax_total": cap_total_pmax,
                "capacidad_disp_total": cap_disp_ty[(t, y)]
            })

    df_dem = pd.DataFrame(filas_dem)

    print("\n--- Demanda original vs demanda corregida ---")
    display(
        df_dem
        .sort_values(["y", "t"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 13.2) Tabla de generación p[g,t,y]
    #       y emisiones e_g * p[g,t,y]
    # --------------------------------------------------------
    filas_p = []

    for g in G:
        for t in T:
            for y in Y:
                val_p = p[g, t, y].X
                val_ep = float(eg.get(g, 0.0)) * val_p
                wt = float(omega_t[t])

                filas_p.append({
                    "gen": g,
                    "t": t,
                    "y": y,
                    "bus": g_bus[g],
                    "Es_IBR": int(g in M),
                    "emision_factor": float(eg.get(g, 0.0)),
                    "p_gty": val_p,
                    "omega_t": wt,
                    "generacion_ponderada_MWh": wt * val_p,
                    "e_g_p_gty": val_ep,
                    "e_g_p_gty_ponderado": wt * val_ep
                })

    df_p = pd.DataFrame(filas_p)

    df_p_pos = df_p[df_p["p_gty"] > 1e-6].copy()
    df_p_pos = df_p_pos.sort_values(["y", "t", "gen"]).reset_index(drop=True)

    print("\n--- Tabla p[g,t,y] y e_g * p[g,t,y] (solo valores > 0) ---")
    display(df_p_pos)

    # --------------------------------------------------------
    # 13.3) Resumen por periodo y año
    # --------------------------------------------------------
    resumen_ty = (
        df_p
        .groupby(["y", "t"], as_index=False)
        .agg(
            generacion_total_no_pond=("p_gty", "sum"),
            generacion_total_ponderada_MWh=("generacion_ponderada_MWh", "sum"),
            emisiones_totales_intervalo_no_pond=("e_g_p_gty", "sum"),
            emisiones_totales_intervalo_ponderadas=("e_g_p_gty_ponderado", "sum")
        )
    )

    print("\n--- Resumen generación/emisiones por (y,t) ---")
    display(resumen_ty)

    # --------------------------------------------------------
    # 13.4) Emisiones totales por año
    # --------------------------------------------------------
    emisiones_por_anio = (
        df_p
        .groupby("y", as_index=False)
        .agg(
            emisiones_totales_no_ponderadas=("e_g_p_gty", "sum"),
            emisiones_totales_anuales_ponderadas=("e_g_p_gty_ponderado", "sum"),
            generacion_total_no_ponderada=("p_gty", "sum"),
            generacion_total_anual_ponderada_MWh=("generacion_ponderada_MWh", "sum")
        )
    )

    print("\n--- Emisiones totales por año ---")
    display(emisiones_por_anio)

    # --------------------------------------------------------
    # 13.5) Resumen por generador
    # --------------------------------------------------------
    resumen_g = (
        df_p
        .groupby("gen", as_index=False)
        .agg(
            bus=("bus", "first"),
            Es_IBR=("Es_IBR", "first"),
            emision_factor=("emision_factor", "first"),
            generacion_total_no_pond=("p_gty", "sum"),
            generacion_total_ponderada_MWh=("generacion_ponderada_MWh", "sum"),
            emisiones_totales_no_pond=("e_g_p_gty", "sum"),
            emisiones_totales_ponderadas=("e_g_p_gty_ponderado", "sum")
        )
        .sort_values("generacion_total_ponderada_MWh", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Resumen por generador ---")
    display(resumen_g)

    # --------------------------------------------------------
    # 13.6) ENS total y por bus
    # --------------------------------------------------------
    filas_s = []

    for i in B:
        for t in T:
            for y in Y:
                filas_s.append({
                    "bus": i,
                    "t": t,
                    "y": y,
                    "ENS": s[i, t, y].X,
                    "omega_t": float(omega_t[t]),
                    "ENS_ponderado_MWh": float(omega_t[t]) * s[i, t, y].X,
                    "demanda_corregida": d_fw[(i, t, y)],
                    "demanda_original": d_original[(i, t, y)]
                })

    df_s = pd.DataFrame(filas_s)

    total_ENS = df_s["ENS"].sum()
    print(f"\nENS total = {total_ENS:.6f}")

    df_s_pos = (
        df_s[df_s["ENS"] > 1e-6]
        .sort_values(["y", "t", "bus"])
        .reset_index(drop=True)
    )

    print("\n--- ENS positivo por bus, periodo y año ---")
    display(df_s_pos)

    # --------------------------------------------------------
    # 13.7) Inversiones IBR
    # --------------------------------------------------------
    filas_x = []

    for m in M:
        for y in Y:
            filas_x.append({
                "gen": m,
                "y": y,
                "x_my": x[m, y].X
            })

    df_x = pd.DataFrame(filas_x)

    print("\n--- Inversión x[m,y] positiva ---")
    display(
        df_x[df_x["x_my"] > 1e-6]
        .sort_values(["y", "gen"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 13.8) Inercia sintética h_syn
    # --------------------------------------------------------
    filas_h = []

    for m in M:
        for t in T:
            for y in Y:
                filas_h.append({
                    "gen": m,
                    "t": t,
                    "y": y,
                    "h_syn": h_syn[m, t, y].X
                })

    df_h = pd.DataFrame(filas_h)

    print("\n--- h_syn[m,t,y] positivo ---")
    display(
        df_h[df_h["h_syn"] > 1e-6]
        .sort_values(["y", "t", "gen"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 13.9) FFR r_ffr
    # --------------------------------------------------------
    filas_r = []

    for m in M:
        for t in T:
            for y in Y:
                filas_r.append({
                    "gen": m,
                    "t": t,
                    "y": y,
                    "r_ffr": r_ffr[m, t, y].X
                })

    df_r = pd.DataFrame(filas_r)

    print("\n--- r_ffr[m,t,y] positivo ---")
    display(
        df_r[df_r["r_ffr"] > 1e-6]
        .sort_values(["y", "t", "gen"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 13.10) Chequeo agregado de inercia y FFR por periodo/año
    # --------------------------------------------------------
    filas_req = []

    for t in T:
        for y in Y:
            H_sync_val = sum(
                float(Hg.get(g, 0.0)) * u[g, t, y].X
                for g in GS
            )

            H_syn_val = sum(
                h_syn[m, t, y].X
                for m in M
            )

            R_val = sum(
                r_ffr[m, t, y].X
                for m in M
            )

            filas_req.append({
                "t": t,
                "y": y,
                "H_sync": H_sync_val,
                "H_syn": H_syn_val,
                "H_total": H_sync_val + H_syn_val,
                "Hreq": float(Hreq[y]),
                "margen_H": H_sync_val + H_syn_val - float(Hreq[y]),
                "R_total": R_val,
                "Rreq": float(Rreq[y]),
                "margen_R": R_val - float(Rreq[y])
            })

    df_req = pd.DataFrame(filas_req)

    print("\n--- Chequeo de requerimientos Hreq y Rreq ---")
    display(
        df_req
        .sort_values(["y", "t"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 13.11) Flujos de línea
    # --------------------------------------------------------
    filas_f = []

    for (i, j) in L_arcs:
        for t in T:
            for y in Y:
                val_f = f[i, j, t, y].X
                Fij = float(F0[(i, j)])

                filas_f.append({
                    "from_bus": i,
                    "to_bus": j,
                    "t": t,
                    "y": y,
                    "f_ijty": val_f,
                    "F_max": Fij,
                    "uso_linea_abs": abs(val_f) / Fij if Fij > 1e-9 else None
                })

    df_f = pd.DataFrame(filas_f)

    print("\n--- Líneas más cargadas ---")
    display(
        df_f
        .sort_values("uso_linea_abs", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

else:
    print("El modelo no quedó óptimo. Status =", fw_base.Status)

# ============================================================
# 14) IMPRIMIR PARÁMETRO X barra: xbar[m]
# ============================================================

print("\n==============================")
print("PARÁMETRO X barra por IBR")
print("==============================")
print("xbar[m] representa la capacidad máxima de inversión permitida para cada IBR m.")
print("Es decir, en el modelo se usa como: x[m,y] <= xbar[m]\n")

filas_xbar = []

for m in M:
    filas_xbar.append({
        "gen_IBR": m,
        "bus": g_bus[m] if m in g_bus else None,
        "xbar_m": float(xbar[m])
    })

df_xbar = pd.DataFrame(filas_xbar)

print("--- Valores de xbar[m] ---")
display(
    df_xbar
    .sort_values("gen_IBR")
    .reset_index(drop=True)
)
