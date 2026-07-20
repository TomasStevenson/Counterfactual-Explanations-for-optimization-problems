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


# %% ================== celda 59 del notebook ==================
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# ============================================================
# FINAL IEEE39
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
# 0) OPCIONES DE CORRECCIÓN DEL DATASET IEEE39
# ============================================================

# Recomendado para IEEE39:
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
print("PESOS TEMPORALES FW IEEE39")
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
# 4) CORRECCIÓN DE DEMANDA IEEE39
# ============================================================

print("\n==============================")
print("CORRECCIÓN DATASET IEEE39")
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
print("PRE-CHECKS FORWARD IEEE39 - FORMULACIÓN ORIGINAL")
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
# 6) MODELO FORWARD IEEE39 - FORMULACIÓN ORIGINAL
# ============================================================

fw_base = gp.Model("FW_base_IEEE39_original_dataset_corregido_con_pesos")
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
# 13) RESULTADOS FORWARD IEEE39 - FORMULACIÓN ORIGINAL
# ============================================================

if fw_base.Status == GRB.OPTIMAL or fw_base.SolCount > 0:

    print("\n==============================")
    print("RESULTADOS FW_base IEEE39 - DATASET CORREGIDO")
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


# %% ================== celda 66 del notebook ==================
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import copy
import math
import time

# ============================================================
# FINAL IEEE39: PADM-W + WCEP CON DUALIDAD FUERTE
# FORMULACIÓN ORIGINAL + DATASET CORREGIDO:
#
# Mutable único:
#   - Pmax_sync_ce[g] para g in GS, entrando en A.
#
# Es decir:
#   - Hreq y xbar quedan fijos.
#   - Pmax_sync_ce aparece en la restricción sync_ub.
#   - La fila sync_ub es acoplada y genera bilinealidades.
#
# Flujo:
#   1) PADM-W etapa 1: busca factibilidad.
#   2) PADM-W etapa 2: parte de factibilidad y reduce J(theta).
#   3) WCEP completo con dualidad fuerte usando PADM_WARMSTART.
# ============================================================

# Medición de tiempo total de celda/script
t0_total_script = time.time()

# ============================================================
# 0) OPCIONES
# ============================================================

PADM_OUTPUT = 1
RUN_PADM_W = True

MAX_OUTER = 100
MAX_INNER = 100

FEAS_TOL = 1e-5
INNER_TOL = 1e-5

RHO0 = 500.0
MU0 = 500.0

PENALTY_FACTOR = 2.0
MAX_PENALTY = 1e9

TIME_LIMIT_SUBPROBLEM = 5 * 60 * 60  # 5 horas por subproblema PADM
WCEP_TIME_LIMIT = 5 * 60 * 60        # 5 horas para WCEP final

# Restricción D del IEEE39
# Si ya calculaste en el FW emisiones ponderadas, puedes poner epsilon_CO2_wcep manual.
# Si lo dejas como None, el código intenta usar el 90% de las emisiones FW guardadas en memoria.
epsilon_CO2_wcep = 930756.144528 * 0.90  # OVERRIDE batch: mismo valor que la celda 63 (gemela sin PADM); el original era None y fallaba (ver make_scripts.py)
EPSILON_CO2_FRACTION = 0.90

# Objetivo J
alpha_Psync = 1.0

# Si quieres objetivo porcentual, cambia esto a True
USE_PERCENT_OBJECTIVE = False

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
# Este bloque hace que PADM-W y WCEP usen la misma demanda corregida
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

# Diagnóstico rápido de demanda usada por PADM-W/WCEP
peak_dem_fw = max(
    sum(float(d_fw[(i, t, y)]) for i in B)
    for t in T
    for y in Y
)
cap_total_pmax = sum(float(Pmax[g]) for g in G)
print(f"Demanda pico usada en PADM-W/WCEP = {peak_dem_fw:.6f}")
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

Pmax_sync_base = {g: float(Pmax[g]) for g in GS}

UB_sync = 2.0 * max(max(float(cg[g]) for g in G), float(VOLL))

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
print(f"UB_sync = {UB_sync:.6f}")
print(f"Bus slack usado: {bus_slack}")


# ============================================================
# 2) REPRESENTACIÓN SIMBÓLICA PARA PADM-W
# ============================================================

def key_p(g,t,y):       return ("p", g, t, y)
def key_u(g,t,y):       return ("u", g, t, y)
def key_s(i,t,y):       return ("s", i, t, y)
def key_xcap(m,y):      return ("xcap", m, y)
def key_hsyn(m,t,y):    return ("h_syn", m, t, y)
def key_rffr(m,t,y):    return ("r_ffr", m, t, y)
def key_fpos(i,j,t,y):  return ("f_pos", i, j, t, y)
def key_fneg(i,j,t,y):  return ("f_neg", i, j, t, y)
def key_thp(i,t,y):     return ("theta_p", i, t, y)
def key_thm(i,t,y):     return ("theta_m", i, t, y)

def param_coeff(family, idx, mult=1.0):
    return ("PARAM", family, idx, float(mult))

def is_param_coeff(c):
    return isinstance(c, tuple) and len(c) == 4 and c[0] == "PARAM"

def coeff_value(c, theta_vals):
    if is_param_coeff(c):
        _, fam, idx, mult = c
        return mult * float(theta_vals[fam][idx])
    return float(c)

def coeff_expr_in_theta_model(c, theta_vars):
    if is_param_coeff(c):
        _, fam, idx, mult = c
        return mult * theta_vars[fam][idx]
    return float(c)

def rhs_const(val):
    return [("CONST", None, None, float(val))]

def rhs_param(family, idx, mult=1.0):
    return [("PARAM", family, idx, float(mult))]

def rhs_has_param(rhs_terms):
    return any(typ == "PARAM" for typ, fam, idx, mult in rhs_terms)

def rhs_eval(rhs_terms, theta_vals=None):
    out = 0.0

    for typ, fam, idx, mult in rhs_terms:
        if typ == "CONST":
            out += float(mult)
        elif typ == "PARAM":
            if theta_vals is None:
                raise ValueError("rhs_eval requiere theta_vals para RHS con parámetros.")
            out += float(mult) * float(theta_vals[fam][idx])
        else:
            raise ValueError(f"Tipo RHS no reconocido: {typ}")

    return out

def rhs_expr_in_theta_model(rhs_terms, theta_vars):
    expr = gp.LinExpr()

    for typ, fam, idx, mult in rhs_terms:
        if typ == "CONST":
            expr += float(mult)
        elif typ == "PARAM":
            expr += float(mult) * theta_vars[fam][idx]
        else:
            raise ValueError(f"Tipo RHS no reconocido: {typ}")

    return expr

def add_to_terms(terms, key, coeff):
    if key not in terms:
        terms[key] = []
    terms[key].append(coeff)

def neg_coeff(c):
    if is_param_coeff(c):
        return ("PARAM", c[1], c[2], -c[3])
    return -float(c)

def terms_has_param(terms):
    for k, coeffs in terms.items():
        for c in coeffs:
            if is_param_coeff(c):
                return True
    return False

def terms_expr_z_model(terms, z_vars, theta_vals):
    expr = gp.LinExpr()

    for k, coeffs in terms.items():
        for c in coeffs:
            expr += coeff_value(c, theta_vals) * z_vars[k]

    return expr

def terms_expr_theta_model_with_fixed_z(terms, z_vals, theta_vars):
    expr = gp.LinExpr()

    for k, coeffs in terms.items():
        zval = float(z_vals.get(k, 0.0))

        if abs(zval) <= 1e-12:
            continue

        for c in coeffs:
            expr += zval * coeff_expr_in_theta_model(c, theta_vars)

    return expr

def terms_eval(terms, z_vals, theta_vals):
    out = 0.0

    for k, coeffs in terms.items():
        zval = float(z_vals.get(k, 0.0))

        if abs(zval) <= 1e-12:
            continue

        for c in coeffs:
            out += coeff_value(c, theta_vals) * zval

    return out


# ============================================================
# 3) CREAR FILAS PRIMALES Ax >= b
# ============================================================

rows = {}
row_order = []

def add_row(name, terms, rhs_terms):
    if name in rows:
        raise ValueError(f"Fila repetida: {name}")

    rows[name] = {
        "terms": terms,
        "rhs": rhs_terms
    }

    row_order.append(name)

# ------------------------------------------------------------
# A.1 Balance nodal
# ------------------------------------------------------------
for i in B:
    for t in T:
        for y in Y:
            terms = {}

            for g in G_en_bus.get(i, []):
                add_to_terms(terms, key_p(g,t,y), 1.0)

            add_to_terms(terms, key_s(i,t,y), 1.0)

            for (k,j) in lineas_entran[i]:
                add_to_terms(terms, key_fpos(k,j,t,y), 1.0)
                add_to_terms(terms, key_fneg(k,j,t,y), -1.0)

            for (i2,j) in lineas_salen[i]:
                add_to_terms(terms, key_fpos(i2,j,t,y), -1.0)
                add_to_terms(terms, key_fneg(i2,j,t,y), 1.0)

            rhs_val = float(d_fw[(i,t,y)])

            add_row(f"bal_p_{i}_{t}_{y}", terms, rhs_const(rhs_val))

            terms_n = {k: [neg_coeff(c) for c in coeffs] for k, coeffs in terms.items()}
            add_row(f"bal_n_{i}_{t}_{y}", terms_n, rhs_const(-rhs_val))

# ------------------------------------------------------------
# A.2 ENS <= demanda
# ------------------------------------------------------------
for i in B:
    for t in T:
        for y in Y:
            terms = {}
            add_to_terms(terms, key_s(i,t,y), -1.0)
            add_row(f"ens_ub_{i}_{t}_{y}", terms, rhs_const(-float(d_fw[(i,t,y)])))

# ------------------------------------------------------------
# A.3 Flujo DC
# ------------------------------------------------------------
for (i,j) in L_arcs:
    for t in T:
        for y in Y:
            Bij = float(Bline[(i,j)])

            terms = {}
            add_to_terms(terms, key_fpos(i,j,t,y), 1.0)
            add_to_terms(terms, key_fneg(i,j,t,y), -1.0)

            add_to_terms(terms, key_thp(i,t,y), -Bij)
            add_to_terms(terms, key_thm(i,t,y),  Bij)
            add_to_terms(terms, key_thp(j,t,y),  Bij)
            add_to_terms(terms, key_thm(j,t,y), -Bij)

            add_row(f"dc_p_{i}_{j}_{t}_{y}", terms, rhs_const(0.0))

            terms_n = {k: [neg_coeff(c) for c in coeffs] for k, coeffs in terms.items()}
            add_row(f"dc_n_{i}_{j}_{t}_{y}", terms_n, rhs_const(0.0))

# ------------------------------------------------------------
# A.4 Límites térmicos fijos
# ------------------------------------------------------------
for (i,j) in L_arcs:
    for t in T:
        for y in Y:
            Fij = float(F0[(i,j)])

            terms_lb = {}
            add_to_terms(terms_lb, key_fpos(i,j,t,y), 1.0)
            add_to_terms(terms_lb, key_fneg(i,j,t,y), -1.0)
            add_row(f"flb_{i}_{j}_{t}_{y}", terms_lb, rhs_const(-Fij))

            terms_ub = {}
            add_to_terms(terms_ub, key_fpos(i,j,t,y), -1.0)
            add_to_terms(terms_ub, key_fneg(i,j,t,y), 1.0)
            add_row(f"fub_{i}_{j}_{t}_{y}", terms_ub, rhs_const(-Fij))

# ------------------------------------------------------------
# A.5 Slack bus
# ------------------------------------------------------------
for t in T:
    for y in Y:
        terms_p = {}
        add_to_terms(terms_p, key_thp(bus_slack,t,y), 1.0)
        add_to_terms(terms_p, key_thm(bus_slack,t,y), -1.0)
        add_row(f"slack_p_{t}_{y}", terms_p, rhs_const(0.0))

        terms_n = {}
        add_to_terms(terms_n, key_thp(bus_slack,t,y), -1.0)
        add_to_terms(terms_n, key_thm(bus_slack,t,y), 1.0)
        add_row(f"slack_n_{t}_{y}", terms_n, rhs_const(0.0))

print(f"Bus slack usado: {bus_slack}")

# ------------------------------------------------------------
# A.6 Límites generación fijos p <= P_disp
# ------------------------------------------------------------
for g in G:
    for t in T:
        cap_base = float(P_disp.get((g,t), Pmax[g]))

        for y in Y:
            terms = {}
            add_to_terms(terms, key_p(g,t,y), -1.0)
            add_row(f"p_cap_{g}_{t}_{y}", terms, rhs_const(-cap_base))

# ------------------------------------------------------------
# A.7 Síncronos
# ------------------------------------------------------------
for g in GS:
    pmin_g = float(Pmin.get(g, 0.0))

    for t in T:
        for y in Y:
            terms_lb = {}
            add_to_terms(terms_lb, key_p(g,t,y), 1.0)
            add_to_terms(terms_lb, key_u(g,t,y), -pmin_g)
            add_row(f"sync_lb_{g}_{t}_{y}", terms_lb, rhs_const(0.0))

            terms_ub = {}
            add_to_terms(terms_ub, key_u(g,t,y), param_coeff("Pmax_sync_ce", g, 1.0))
            add_to_terms(terms_ub, key_p(g,t,y), -1.0)
            add_row(f"sync_ub_{g}_{t}_{y}", terms_ub, rhs_const(0.0))

            terms_u = {}
            add_to_terms(terms_u, key_u(g,t,y), -1.0)
            add_row(f"u_ub_{g}_{t}_{y}", terms_u, rhs_const(-1.0))

# ------------------------------------------------------------
# A.8-A.10 Inversión, h_syn, FFR
# ------------------------------------------------------------
for m in M:
    for y in Y:
        terms_x = {}
        add_to_terms(terms_x, key_xcap(m,y), -1.0)
        add_row(f"x_ub_{m}_{y}", terms_x, rhs_const(-float(xbar[m])))

        for t in T:
            terms_h = {}
            add_to_terms(terms_h, key_xcap(m,y), float(gamma_base[m]))
            add_to_terms(terms_h, key_hsyn(m,t,y), -1.0)
            add_row(f"hsyn_lim_{m}_{t}_{y}", terms_h, rhs_const(0.0))

            terms_r = {}
            add_to_terms(terms_r, key_xcap(m,y), float(kappa_base[m]))
            add_to_terms(terms_r, key_rffr(m,t,y), -1.0)
            add_row(f"ffr_lim_{m}_{t}_{y}", terms_r, rhs_const(0.0))

# ------------------------------------------------------------
# A.11 Seguridad: Hreq fijo, Rreq fijo
# ------------------------------------------------------------
for t in T:
    for y in Y:
        terms_H = {}

        for g in GS:
            add_to_terms(terms_H, key_u(g,t,y), float(Hg.get(g, 0.0)))

        for m in M:
            add_to_terms(terms_H, key_hsyn(m,t,y), 1.0)

        add_row(f"sys_H_{t}_{y}", terms_H, rhs_const(float(Hreq[y])))

        terms_R = {}

        for m in M:
            add_to_terms(terms_R, key_rffr(m,t,y), 1.0)

        add_row(f"sys_R_{t}_{y}", terms_R, rhs_const(float(Rreq[y])))


# ============================================================
# 4) VARIABLES PRIMALES Y COSTOS c
# ============================================================

primal_keys = []

for g in G:
    for t in T:
        for y in Y:
            primal_keys.append(key_p(g,t,y))

for g in GS:
    for t in T:
        for y in Y:
            primal_keys.append(key_u(g,t,y))

for i in B:
    for t in T:
        for y in Y:
            primal_keys.append(key_s(i,t,y))

for m in M:
    for y in Y:
        primal_keys.append(key_xcap(m,y))

for m in M:
    for t in T:
        for y in Y:
            primal_keys.append(key_hsyn(m,t,y))
            primal_keys.append(key_rffr(m,t,y))

for (i,j) in L_arcs:
    for t in T:
        for y in Y:
            primal_keys.append(key_fpos(i,j,t,y))
            primal_keys.append(key_fneg(i,j,t,y))

for i in B:
    for t in T:
        for y in Y:
            primal_keys.append(key_thp(i,t,y))
            primal_keys.append(key_thm(i,t,y))

c_map = {}

for g in G:
    for t in T:
        for y in Y:
            c_map[key_p(g,t,y)] = omega(t) * float(cg[g])

for i in B:
    for t in T:
        for y in Y:
            c_map[key_s(i,t,y)] = omega(t) * float(VOLL)

for m in M:
    for y in Y:
        c_map[key_xcap(m,y)] = float(cinv[m])

for k in primal_keys:
    c_map.setdefault(k, 0.0)


# ============================================================
# 5) CLASIFICACIÓN ACOPLADAS VS SEPARABLES
# ============================================================

coupled_primal_rows = [
    name for name in row_order
    if terms_has_param(rows[name]["terms"]) or rhs_has_param(rows[name]["rhs"])
]

separable_primal_rows = [
    name for name in row_order
    if name not in set(coupled_primal_rows)
]

coupled_dual_cols = set()

for name in row_order:
    terms = rows[name]["terms"]

    for k, coeffs in terms.items():
        if any(is_param_coeff(c) for c in coeffs):
            coupled_dual_cols.add(k)

coupled_dual_cols = sorted(list(coupled_dual_cols), key=str)

separable_dual_cols = [
    k for k in primal_keys
    if k not in set(coupled_dual_cols)
]

print("\n" + "="*70)
print("CLASIFICACIÓN PADM-W IEEE39 - FORMULACIÓN ORIGINAL")
print("="*70)
print(f"N° filas primales acopladas   : {len(coupled_primal_rows)}")
print(f"N° filas primales separables  : {len(separable_primal_rows)}")
print(f"N° columnas duales acopladas  : {len(coupled_dual_cols)}")
print(f"N° columnas duales separables : {len(separable_dual_cols)}")

bad_rows = [
    name for name in coupled_primal_rows
    if not name.startswith("sync_ub_")
]

if bad_rows:
    print("\n⚠️ Hay filas acopladas no esperadas:")
    print(bad_rows[:30])
else:
    print("\n✅ Filas acopladas esperadas: sync_ub_.")


# ============================================================
# 6) FUNCIONES PADM-W
# ============================================================

def initial_theta_vals():
    return {
        "Pmax_sync_ce": {g: float(Pmax_sync_base[g]) for g in GS},
    }

def compute_J(theta_vals):
    J = 0.0

    for g in GS:
        base = float(Pmax_sync_base[g])
        val = float(theta_vals["Pmax_sync_ce"][g])

        if USE_PERCENT_OBJECTIVE and abs(base) > 1e-9:
            J += alpha_Psync * abs(val - base) / abs(base)
        else:
            J += alpha_Psync * abs(val - base)

    return J

def compute_ctx(z_vals):
    return sum(
        float(c_map[k]) * float(z_vals.get(k, 0.0))
        for k in primal_keys
    )

def compute_bty(theta_vals, ydual_vals):
    out = 0.0

    for name in row_order:
        out += rhs_eval(rows[name]["rhs"], theta_vals) * float(ydual_vals.get(name, 0.0))

    return out

def compute_primal_residuals(theta_vals, z_vals):
    res = {}

    for name in row_order:
        lhs = terms_eval(rows[name]["terms"], z_vals, theta_vals)
        rhs = rhs_eval(rows[name]["rhs"], theta_vals)
        res[name] = max(0.0, rhs - lhs)

    return res

def compute_dual_residuals(theta_vals, ydual_vals):
    res = {}

    for k in primal_keys:
        lhs = 0.0

        for name in row_order:
            terms = rows[name]["terms"]

            if k not in terms:
                continue

            yd = float(ydual_vals.get(name, 0.0))

            if abs(yd) <= 1e-12:
                continue

            for c in terms[k]:
                lhs += coeff_value(c, theta_vals) * yd

        rhs = float(c_map.get(k, 0.0))
        res[k] = max(0.0, lhs - rhs)

    return res

def compute_sd_residual(theta_vals, z_vals, ydual_vals):
    return max(0.0, compute_ctx(z_vals) - compute_bty(theta_vals, ydual_vals))

def max_residuals(theta_vals, z_vals, ydual_vals):
    pr = compute_primal_residuals(theta_vals, z_vals)
    dr = compute_dual_residuals(theta_vals, ydual_vals)
    sd = compute_sd_residual(theta_vals, z_vals, ydual_vals)

    return {
        "max_primal_all": max(pr.values()) if pr else 0.0,
        "max_dual_all": max(dr.values()) if dr else 0.0,
        "sd": sd,
        "max_primal_coupled": max([pr[name] for name in coupled_primal_rows], default=0.0),
        "max_dual_coupled": max([dr[k] for k in coupled_dual_cols], default=0.0),
    }

def compute_phi(theta_vals, z_vals, ydual_vals, rho, mu, use_true_objective=True):
    Jval = compute_J(theta_vals) if use_true_objective else 0.0

    pr = compute_primal_residuals(theta_vals, z_vals)
    dr = compute_dual_residuals(theta_vals, ydual_vals)

    pen_p = sum(float(rho[name]) * pr[name] for name in coupled_primal_rows)
    pen_d = sum(float(mu[k]) * dr[k] for k in coupled_dual_cols)

    sd_penalty = compute_sd_residual(theta_vals, z_vals, ydual_vals)

    return Jval + pen_p + pen_d + 1e3 * sd_penalty

def diff_theta(theta_a, theta_b):
    vals = []

    for g in GS:
        vals.append(abs(theta_a["Pmax_sync_ce"][g] - theta_b["Pmax_sync_ce"][g]))

    return max(vals) if vals else 0.0

def diff_dict(a, b):
    keys = set(a.keys()) | set(b.keys())
    return max(
        [abs(float(a.get(k,0.0)) - float(b.get(k,0.0))) for k in keys],
        default=0.0
    )


# ============================================================
# 7) SUBPROBLEMAS PADM-W
# ============================================================

def add_z_variables(model):
    z = {}

    for g in G:
        for t in T:
            for y in Y:
                z[key_p(g,t,y)] = model.addVar(lb=0.0, name=f"p[{g},{t},{y}]")

    for g in GS:
        for t in T:
            for y in Y:
                z[key_u(g,t,y)] = model.addVar(lb=0.0, ub=1.0, name=f"u[{g},{t},{y}]")

    for i in B:
        for t in T:
            for y in Y:
                z[key_s(i,t,y)] = model.addVar(lb=0.0, name=f"s[{i},{t},{y}]")

    for m in M:
        for y in Y:
            z[key_xcap(m,y)] = model.addVar(lb=0.0, name=f"xcap[{m},{y}]")

    for m in M:
        for t in T:
            for y in Y:
                z[key_hsyn(m,t,y)] = model.addVar(lb=0.0, name=f"h_syn[{m},{t},{y}]")
                z[key_rffr(m,t,y)] = model.addVar(lb=0.0, name=f"r_ffr[{m},{t},{y}]")

    for (i,j) in L_arcs:
        for t in T:
            for y in Y:
                z[key_fpos(i,j,t,y)] = model.addVar(lb=0.0, name=f"f_pos[{i},{j},{t},{y}]")
                z[key_fneg(i,j,t,y)] = model.addVar(lb=0.0, name=f"f_neg[{i},{j},{t},{y}]")

    for i in B:
        for t in T:
            for y in Y:
                z[key_thp(i,t,y)] = model.addVar(lb=0.0, name=f"theta_p[{i},{t},{y}]")
                z[key_thm(i,t,y)] = model.addVar(lb=0.0, name=f"theta_m[{i},{t},{y}]")

    return z

def add_dual_variables(model):
    yd = {}

    for name in row_order:
        ub_here = GRB.INFINITY

        if name.startswith("sync_ub_"):
            ub_here = UB_sync

        yd[name] = model.addVar(lb=0.0, ub=ub_here, name=f"ydual[{name}]")

    return yd

def add_theta_variables(model, warm_theta=None):
    theta = {
        "Pmax_sync_ce": {},
    }

    for g in GS:
        theta["Pmax_sync_ce"][g] = model.addVar(
            lb=0.85 * float(Pmax_sync_base[g]),
            ub=1.00 * float(Pmax_sync_base[g]),
            name=f"Pmax_sync_ce[{g}]"
        )

    if warm_theta is not None:
        for g in GS:
            theta["Pmax_sync_ce"][g].Start = float(warm_theta["Pmax_sync_ce"][g])

    return theta

def extract_theta(theta_vars):
    return {
        "Pmax_sync_ce": {g: float(theta_vars["Pmax_sync_ce"][g].X) for g in GS},
    }

def add_separable_primal_constraints_y(model, z_vars):
    theta_base = initial_theta_vals()

    for name in separable_primal_rows:
        lhs = terms_expr_z_model(rows[name]["terms"], z_vars, theta_base)
        rhs = rhs_eval(rows[name]["rhs"], theta_base)
        model.addConstr(lhs >= rhs, name=f"Y_primal[{name}]")

def add_D_constraint_y(model, z_vars):
    for y in Y:
        if y >= y_star:
            model.addConstr(
                gp.quicksum(
                    omega(t) * float(eg[g]) * z_vars[key_p(g,t,y)]
                    for g in GE
                    for t in T
                )
                <= float(eps_CO2(y)),
                name=f"D_CO2_cap_{y}"
            )

def add_separable_dual_constraints_y(model, ydual_vars):
    theta_base = initial_theta_vals()

    for k in separable_dual_cols:
        lhs = gp.LinExpr()

        for name in row_order:
            terms = rows[name]["terms"]

            if k not in terms:
                continue

            for c in terms[k]:
                lhs += coeff_value(c, theta_base) * ydual_vars[name]

        model.addConstr(lhs <= float(c_map.get(k, 0.0)), name=f"Y_dual[{k}]")

def add_strong_duality_y(model, theta_vals, z_vars, ydual_vars):
    ctx = gp.quicksum(float(c_map[k]) * z_vars[k] for k in primal_keys)

    bty = gp.LinExpr()

    for name in row_order:
        bty += rhs_eval(rows[name]["rhs"], theta_vals) * ydual_vars[name]

    model.addConstr(ctx <= bty, name="Y_StrongDuality")

def solve_theta_subproblem(
    z_vals,
    ydual_vals,
    rho,
    mu,
    warm_theta=None,
    log=False,
    use_true_objective=True
):
    m = gp.Model("PADM_theta_subproblem_IEEE39_dataset_corregido")
    m.Params.OutputFlag = 1 if log else 0

    if TIME_LIMIT_SUBPROBLEM is not None:
        m.Params.TimeLimit = TIME_LIMIT_SUBPROBLEM

    theta = add_theta_variables(m, warm_theta=warm_theta)

    obj = gp.LinExpr()

    if use_true_objective:
        for g in GS:
            dvar = m.addVar(lb=0.0, name=f"dPsync[{g}]")
            diff = theta["Pmax_sync_ce"][g] - float(Pmax_sync_base[g])

            m.addConstr(dvar >= diff, name=f"dPsync_pos[{g}]")
            m.addConstr(dvar >= -diff, name=f"dPsync_neg[{g}]")

            base = abs(float(Pmax_sync_base[g]))

            if USE_PERCENT_OBJECTIVE and base > 1e-9:
                obj += float(alpha_Psync) * dvar / base
            else:
                obj += float(alpha_Psync) * dvar

    for name in coupled_primal_rows:
        lhs = terms_expr_theta_model_with_fixed_z(rows[name]["terms"], z_vals, theta)
        rhs = rhs_expr_in_theta_model(rows[name]["rhs"], theta)

        v = m.addVar(lb=0.0, name=f"viol_primal[{name}]")
        m.addConstr(v >= rhs - lhs, name=f"def_viol_primal[{name}]")

        obj += float(rho[name]) * v

    for k in coupled_dual_cols:
        lhs = gp.LinExpr()

        for name in row_order:
            terms = rows[name]["terms"]

            if k not in terms:
                continue

            ydval = float(ydual_vals.get(name, 0.0))

            if abs(ydval) <= 1e-12:
                continue

            for c in terms[k]:
                lhs += ydval * coeff_expr_in_theta_model(c, theta)

        v = m.addVar(lb=0.0, name=f"viol_dual[{k}]")
        m.addConstr(v >= lhs - float(c_map.get(k, 0.0)), name=f"def_viol_dual[{k}]")

        obj += float(mu[k]) * v

    # Dualidad fuerte dura en theta:
    #   c^T z fijo <= b(theta)^T y fijo
    ctx_fixed = sum(float(c_map[k]) * float(z_vals.get(k, 0.0)) for k in primal_keys)

    bty_theta = gp.LinExpr()

    for name in row_order:
        ydval = float(ydual_vals.get(name, 0.0))

        if abs(ydval) <= 1e-12:
            continue

        bty_theta += ydval * rhs_expr_in_theta_model(rows[name]["rhs"], theta)

    m.addConstr(ctx_fixed <= bty_theta, name="Theta_StrongDuality")

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] or m.SolCount == 0:
        print("\n⚠️ Subproblema THETA falló.")
        print(f"Status = {m.Status}")
        return None, m.Status, None

    theta_vals = extract_theta(theta)
    return theta_vals, m.Status, float(m.ObjVal)

def solve_y_subproblem(theta_vals, rho, mu, warm_z=None, warm_ydual=None, log=False):
    m = gp.Model("PADM_y_subproblem_IEEE39_dataset_corregido")
    m.Params.OutputFlag = 1 if log else 0

    if TIME_LIMIT_SUBPROBLEM is not None:
        m.Params.TimeLimit = TIME_LIMIT_SUBPROBLEM

    z = add_z_variables(m)
    yd = add_dual_variables(m)

    m.update()

    if warm_z is not None:
        for k, v in z.items():
            if k in warm_z:
                v.Start = float(warm_z[k])

    if warm_ydual is not None:
        for name, v in yd.items():
            if name in warm_ydual:
                v.Start = float(warm_ydual[name])

    add_separable_primal_constraints_y(m, z)
    add_D_constraint_y(m, z)
    add_separable_dual_constraints_y(m, yd)
    add_strong_duality_y(m, theta_vals, z, yd)

    obj = gp.LinExpr()

    for name in coupled_primal_rows:
        lhs = terms_expr_z_model(rows[name]["terms"], z, theta_vals)
        rhs = rhs_eval(rows[name]["rhs"], theta_vals)

        v = m.addVar(lb=0.0, name=f"viol_primal[{name}]")
        m.addConstr(v >= rhs - lhs, name=f"def_viol_primal[{name}]")

        obj += float(rho[name]) * v

    for k in coupled_dual_cols:
        lhs = gp.LinExpr()

        for name in row_order:
            terms = rows[name]["terms"]

            if k not in terms:
                continue

            for c in terms[k]:
                lhs += coeff_value(c, theta_vals) * yd[name]

        v = m.addVar(lb=0.0, name=f"viol_dual[{k}]")
        m.addConstr(v >= lhs - float(c_map.get(k, 0.0)), name=f"def_viol_dual[{k}]")

        obj += float(mu[k]) * v

    if Y_OBJECTIVE_MODE == "ctx":
        obj += TIEBREAKER_CTX_WEIGHT * gp.quicksum(
            float(c_map[k]) * z[k]
            for k in primal_keys
        )

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] or m.SolCount == 0:
        print("\n⚠️ Subproblema Y falló.")
        print(f"Status = {m.Status}")
        return None, None, m.Status, None

    z_vals = {k: float(v.X) for k, v in z.items()}
    yd_vals = {name: float(v.X) for name, v in yd.items()}

    return z_vals, yd_vals, m.Status, float(m.ObjVal)

def initialize_penalties():
    rho = {name: float(RHO0) for name in coupled_primal_rows}
    mu = {k: float(MU0) for k in coupled_dual_cols}
    return rho, mu

def update_penalties(theta_vals, z_vals, ydual_vals, rho, mu):
    pr = compute_primal_residuals(theta_vals, z_vals)
    dr = compute_dual_residuals(theta_vals, ydual_vals)

    n_rho = 0
    n_mu = 0

    for name in coupled_primal_rows:
        if pr[name] > FEAS_TOL:
            rho[name] = min(MAX_PENALTY, PENALTY_FACTOR * rho[name])
            n_rho += 1

    for k in coupled_dual_cols:
        if dr[k] > FEAS_TOL:
            mu[k] = min(MAX_PENALTY, PENALTY_FACTOR * mu[k])
            n_mu += 1

    return n_rho, n_mu


# ============================================================
# 8) LOOP PADM-W
# ============================================================

def run_padm(
    use_true_objective=True,
    theta_init=None,
    z_init=None,
    ydual_init=None,
    label="PADM"
):
    print("\n" + "="*70)
    print(f"INICIO {label}")
    print("="*70)
    print(f"use_true_objective = {use_true_objective}")

    rho, mu = initialize_penalties()

    theta_vals = copy.deepcopy(theta_init) if theta_init is not None else initial_theta_vals()

    if z_init is None or ydual_init is None:
        print("\nInicializando Y=(z,lambda)...")

        z_vals, ydual_vals, st_y, obj_y = solve_y_subproblem(
            theta_vals,
            rho,
            mu,
            warm_z=None,
            warm_ydual=None,
            log=False
        )

        if z_vals is None:
            print("❌ No se pudo inicializar Y.")

            return {
                "best_feasible": None,
                "theta": theta_vals,
                "z": None,
                "ydual": None,
                "history": [],
                "last_residuals": None,
                "rho": rho,
                "mu": mu,
            }

    else:
        z_vals = z_init.copy()
        ydual_vals = ydual_init.copy()

    history = []
    best_feasible = None

    res0 = max_residuals(theta_vals, z_vals, ydual_vals)

    print("\nChequeo inicial:")
    print(f"J(theta)       = {compute_J(theta_vals):.8f}")
    print(f"Max primal     = {res0['max_primal_all']:.3e}")
    print(f"Max dual       = {res0['max_dual_all']:.3e}")
    print(f"Strong duality = {res0['sd']:.3e}")

    for k_outer in range(MAX_OUTER):

        print("\n" + "#"*70)
        print(f"{label} | OUTER k = {k_outer}")
        print("#"*70)

        phi_prev = compute_phi(
            theta_vals,
            z_vals,
            ydual_vals,
            rho,
            mu,
            use_true_objective=use_true_objective
        )

        for l_inner in range(MAX_INNER):

            theta_old = copy.deepcopy(theta_vals)
            z_old = z_vals.copy()
            yd_old = ydual_vals.copy()

            theta_new, st_theta, obj_theta = solve_theta_subproblem(
                z_vals,
                ydual_vals,
                rho,
                mu,
                warm_theta=theta_vals,
                log=False,
                use_true_objective=use_true_objective
            )

            if theta_new is None:
                print("⚠️ Falló subproblema theta.")
                break

            theta_vals = theta_new

            z_new, yd_new, st_y, obj_y = solve_y_subproblem(
                theta_vals,
                rho,
                mu,
                warm_z=z_vals,
                warm_ydual=ydual_vals,
                log=False
            )

            if z_new is None:
                print("⚠️ Falló subproblema Y.")
                break

            z_vals = z_new
            ydual_vals = yd_new

            phi_now = compute_phi(
                theta_vals,
                z_vals,
                ydual_vals,
                rho,
                mu,
                use_true_objective=use_true_objective
            )

            res = max_residuals(theta_vals, z_vals, ydual_vals)

            dtheta = diff_theta(theta_vals, theta_old)
            dz = diff_dict(z_vals, z_old)
            dyd = diff_dict(ydual_vals, yd_old)
            dphi = abs(phi_now - phi_prev) / max(1.0, abs(phi_prev))

            if PADM_OUTPUT:
                print(
                    f"k={k_outer:02d}, l={l_inner:02d} | "
                    f"Phi={phi_now:.6e}, J={compute_J(theta_vals):.6e}, "
                    f"p_all={res['max_primal_all']:.2e}, "
                    f"d_all={res['max_dual_all']:.2e}, "
                    f"sd={res['sd']:.2e}, "
                    f"p_coup={res['max_primal_coupled']:.2e}, "
                    f"d_coup={res['max_dual_coupled']:.2e}, "
                    f"dtheta={dtheta:.2e}, dz={dz:.2e}, dyd={dyd:.2e}, dphi={dphi:.2e}"
                )

            history.append({
                "outer": k_outer,
                "inner": l_inner,
                "phi": phi_now,
                "J": compute_J(theta_vals),
                "max_primal_all": res["max_primal_all"],
                "max_dual_all": res["max_dual_all"],
                "sd": res["sd"],
                "max_primal_coupled": res["max_primal_coupled"],
                "max_dual_coupled": res["max_dual_coupled"],
                "dtheta": dtheta,
                "dz": dz,
                "dyd": dyd,
                "dphi": dphi,
                "rho_max": max(rho.values()) if rho else 0.0,
                "mu_max": max(mu.values()) if mu else 0.0,
            })

            if dphi <= INNER_TOL and dtheta <= INNER_TOL and dz <= INNER_TOL and dyd <= INNER_TOL:
                print(f"Inner loop converge en l={l_inner}.")
                break

            phi_prev = phi_now

        res = max_residuals(theta_vals, z_vals, ydual_vals)

        feasible = (
            res["max_primal_all"] <= FEAS_TOL
            and res["max_dual_all"] <= FEAS_TOL
            and res["sd"] <= FEAS_TOL
        )

        print("\n" + "-"*70)
        print(f"{label} | Chequeo fin outer k={k_outer}")
        print("-"*70)
        print(f"J(theta)             = {compute_J(theta_vals):.8f}")
        print(f"Max primal all       = {res['max_primal_all']:.3e}")
        print(f"Max dual all         = {res['max_dual_all']:.3e}")
        print(f"Strong duality       = {res['sd']:.3e}")
        print(f"Max primal coupled   = {res['max_primal_coupled']:.3e}")
        print(f"Max dual coupled     = {res['max_dual_coupled']:.3e}")
        print(f"rho_max              = {max(rho.values()) if rho else 0.0:.3e}")
        print(f"mu_max               = {max(mu.values()) if mu else 0.0:.3e}")

        if feasible:
            print(f"\n✅ {label} encontró solución factible primal-dual.")

            best_feasible = {
                "theta": copy.deepcopy(theta_vals),
                "z": z_vals.copy(),
                "ydual": ydual_vals.copy(),
                "J": compute_J(theta_vals),
                "residuals": res,
                "outer": k_outer,
                "history": history,
            }

            break

        n_rho, n_mu = update_penalties(theta_vals, z_vals, ydual_vals, rho, mu)

        print("\nActualización penalizaciones:")
        print(f"rho actualizadas: {n_rho}")
        print(f"mu actualizadas : {n_mu}")

    if best_feasible is None:
        print(f"\n⚠️ {label} terminó sin certificar factibilidad.")

    return {
        "best_feasible": best_feasible,
        "theta": theta_vals,
        "z": z_vals,
        "ydual": ydual_vals,
        "history": history,
        "last_residuals": max_residuals(theta_vals, z_vals, ydual_vals) if z_vals is not None else None,
        "rho": rho,
        "mu": mu,
    }


# ============================================================
# 9) EJECUTAR PADM-W
# ============================================================

PADM_WARMSTART = None

if RUN_PADM_W:

    res_ws = run_padm(
        use_true_objective=False,
        theta_init=None,
        z_init=None,
        ydual_init=None,
        label="PADM-W ETAPA 1: FACTIBILIDAD"
    )

    if res_ws["best_feasible"] is not None:

        theta_start = res_ws["best_feasible"]["theta"]
        z_start = res_ws["best_feasible"]["z"]
        yd_start = res_ws["best_feasible"]["ydual"]

        print("\n" + "="*70)
        print("WARMSTART FACTIBLE ENCONTRADO. INICIANDO ETAPA 2 CON J REAL.")
        print("="*70)

        res_final = run_padm(
            use_true_objective=True,
            theta_init=theta_start,
            z_init=z_start,
            ydual_init=yd_start,
            label="PADM-W ETAPA 2: OBJETIVO REAL"
        )

        if res_final["best_feasible"] is not None:
            final_solution = res_final["best_feasible"]
            final_label = "PADM-W ETAPA 2"
        else:
            print("\n⚠️ Etapa 2 no certificó factibilidad. Se conserva warmstart factible de etapa 1.")
            final_solution = res_ws["best_feasible"]
            final_label = "PADM-W ETAPA 1"

    else:
        print("\n⚠️ Etapa 1 no encontró punto factible.")
        final_solution = None
        final_label = "PADM-W FAILED"
        res_final = res_ws

else:
    res_final = run_padm(
        use_true_objective=True,
        theta_init=None,
        z_init=None,
        ydual_init=None,
        label="PADM"
    )

    if res_final["best_feasible"] is not None:
        final_solution = res_final["best_feasible"]
        final_label = "PADM"
    else:
        final_solution = None
        final_label = "PADM FAILED"

print("\n" + "="*70)
print("RESULTADOS FINALES PADM-W IEEE39")
print("="*70)

if final_solution is not None:
    sol_theta = final_solution["theta"]
    sol_z = final_solution["z"]
    sol_yd = final_solution["ydual"]

    print("✅ SOLUCIÓN FACTIBLE CERTIFICADA")
    print(f"Método final     : {final_label}")
    print(f"Outer encontrado : {final_solution['outer']}")
    print(f"J                : {final_solution['J']:.8f}")

    PADM_WARMSTART = {
        "theta": copy.deepcopy(sol_theta),
        "z": sol_z.copy(),
        "ydual": sol_yd.copy(),
        "label": final_label,
        "J": final_solution["J"],
        "residuals": final_solution["residuals"],
    }

else:
    print("⚠️ NO SE CERTIFICÓ FACTIBILIDAD EN PADM-W.")

    sol_theta = res_final["theta"]
    sol_z = res_final["z"]
    sol_yd = res_final["ydual"]

    if sol_z is not None and sol_yd is not None:
        res = max_residuals(sol_theta, sol_z, sol_yd)

        PADM_WARMSTART = {
            "theta": copy.deepcopy(sol_theta),
            "z": sol_z.copy(),
            "ydual": sol_yd.copy(),
            "label": "PADM_LAST_POINT_NOT_CERTIFIED",
            "J": compute_J(sol_theta),
            "residuals": res,
        }

        print("⚠️ Se guardó el último punto de PADM-W como Start no certificado.")
    else:
        PADM_WARMSTART = None
        print("❌ No hay punto PADM-W disponible para warm start.")

if PADM_WARMSTART is not None:
    print("\n" + "="*70)
    print("CHECK PADM_WARMSTART IEEE39")
    print("="*70)

    sol_theta = PADM_WARMSTART["theta"]
    sol_z = PADM_WARMSTART["z"]
    sol_yd = PADM_WARMSTART["ydual"]

    res_check = max_residuals(sol_theta, sol_z, sol_yd)

    print(f"Origen          = {PADM_WARMSTART['label']}")
    print(f"J(theta)        = {PADM_WARMSTART['J']:.8f}")
    print(f"Max primal      = {res_check['max_primal_all']:.6e}")
    print(f"Max dual        = {res_check['max_dual_all']:.6e}")
    print(f"Strong duality  = {res_check['sd']:.6e}")


# ============================================================
# 10) WCEP FINAL CON DUALIDAD FUERTE Y WARM START PADM-W
# ============================================================

print("\n" + "="*70)
print("CONSTRUYENDO WCEP FINAL IEEE39 CON DUALIDAD FUERTE")
print("="*70)

w = gp.Model("WCEP_IEEE39_DF_warmstart_PADM_Psync_dataset_corregido")
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

Pmax_sync_ce = w.addVars(GS, lb=0.0, name="Pmax_sync_ce")

for g in GS:
    base = Pmax_sync_base[g]
    w.addConstr(Pmax_sync_ce[g] >= 0.85 * base, name=f"adm_lb_Psync[{g}]")
    w.addConstr(Pmax_sync_ce[g] <= 1.00 * base, name=f"adm_ub_Psync[{g}]")

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
            # Bilineal porque Pmax_sync_ce[g] es mutable y u[g,t,y] es variable primal.
            R_primal[name] = w.addQConstr(
                Pmax_sync_ce[g] * u[g, t, y] - p[g, t, y] >= 0.0,
                name=name
            )
            RHS_expr[name] = 0.0

            name = f"u_ub_{g}_{t}_{y}"
            R_primal[name] = w.addConstr(-u[g, t, y] >= -1.0, name=name)
            RHS_expr[name] = -1.0

for m in M:
    for y in Y:
        name = f"x_ub_{m}_{y}"
        R_primal[name] = w.addConstr(-xcap[m, y] >= -float(xbar[m]), name=name)
        RHS_expr[name] = -float(xbar[m])

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
        R_primal[name] = w.addConstr(lhs_H >= float(Hreq[y]), name=name)
        RHS_expr[name] = float(Hreq[y])

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

    if name.startswith("sync_ub_"):
        ub_here = UB_sync

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

dPsync = w.addVars(GS, lb=0.0, name="dPsync")

for g in GS:
    base = Pmax_sync_base[g]
    diff = Pmax_sync_ce[g] - base

    w.addConstr(dPsync[g] >= diff, name=f"dPsync_pos[{g}]")
    w.addConstr(dPsync[g] >= -diff, name=f"dPsync_neg[{g}]")

if USE_PERCENT_OBJECTIVE:
    obj_wcep = gp.LinExpr()

    for g in GS:
        base = abs(float(Pmax_sync_base[g]))
        if base > 1e-9:
            obj_wcep += alpha_Psync * dPsync[g] / base
        else:
            obj_wcep += alpha_Psync * dPsync[g]

    w.setObjective(obj_wcep, GRB.MINIMIZE)

else:
    w.setObjective(alpha_Psync * dPsync.sum(), GRB.MINIMIZE)


# ============================================================
# 11) APLICAR WARM START DEL PADM-W
# ============================================================

def safe_start(var, value):
    try:
        if value is not None and np.isfinite(float(value)):
            var.Start = float(value)
    except Exception:
        pass

if PADM_WARMSTART is not None:
    print("\n" + "="*70)
    print("APLICANDO PADM_WARMSTART AL WCEP FINAL IEEE39")
    print("="*70)
    print("Origen:", PADM_WARMSTART.get("label", "PADM"))

    ws_theta = PADM_WARMSTART["theta"]
    ws_z = PADM_WARMSTART["z"]
    ws_yd = PADM_WARMSTART["ydual"]

    for g in GS:
        safe_start(Pmax_sync_ce[g], ws_theta["Pmax_sync_ce"][g])
        safe_start(dPsync[g], abs(ws_theta["Pmax_sync_ce"][g] - Pmax_sync_base[g]))

    for g in G:
        for t in T:
            for y in Y:
                safe_start(p[g, t, y], ws_z.get(("p", g, t, y), None))

    for g in GS:
        for t in T:
            for y in Y:
                safe_start(u[g, t, y], ws_z.get(("u", g, t, y), None))

    for i in B:
        for t in T:
            for y in Y:
                safe_start(s[i, t, y], ws_z.get(("s", i, t, y), None))

    for m in M:
        for y in Y:
            safe_start(xcap[m, y], ws_z.get(("xcap", m, y), None))

    for m in M:
        for t in T:
            for y in Y:
                safe_start(h_syn[m, t, y], ws_z.get(("h_syn", m, t, y), None))
                safe_start(r_ffr[m, t, y], ws_z.get(("r_ffr", m, t, y), None))

    for (i, j) in L_arcs:
        for t in T:
            for y in Y:
                safe_start(f_pos[i, j, t, y], ws_z.get(("f_pos", i, j, t, y), None))
                safe_start(f_neg[i, j, t, y], ws_z.get(("f_neg", i, j, t, y), None))

    for i in B:
        for t in T:
            for y in Y:
                safe_start(theta_p[i, t, y], ws_z.get(("theta_p", i, t, y), None))
                safe_start(theta_m[i, t, y], ws_z.get(("theta_m", i, t, y), None))

    for name, var in ydual.items():
        safe_start(var, ws_yd.get(name, None))

    print("✅ Warm start aplicado al WCEP IEEE39.")
else:
    print("\n⚠️ No existe PADM_WARMSTART. El WCEP se resolverá sin Start.")


# ============================================================
# 12) OPTIMIZAR WCEP FINAL
# ============================================================

print("\n" + "="*70)
print("RESOLVIENDO WCEP FINAL IEEE39")
print("="*70)

w.optimize()


# ============================================================
# 13) RESULTADOS WCEP FINAL
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

    print("\n--- Pmax_sync_ce síncronos ---")
    hubo_cambio = False

    for g in GS:
        val = float(Pmax_sync_ce[g].X)
        base = float(Pmax_sync_base[g])

        if abs(val - base) > tol:
            hubo_cambio = True
            pct = 100.0 * (val / base - 1.0) if abs(base) > 1e-9 else float("inf")
            print(f"{g}: {base:.4f} -> {val:.4f}   ({pct:+.2f}%)")

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
            H_lim = float(Hreq[y])

            R_total = sum(
                float(r_ffr[m, t, y].X)
                for m in M
            )

            R_lim = float(Rreq[y])

            print(
                f"(y={y}, t={t}) | "
                f"H_total={H_total:.6f}, Hreq={H_lim:.6f}, margen_H={H_total - H_lim:.6e} | "
                f"R_total={R_total:.6f}, Rreq={R_lim:.6f}, margen_R={R_total - R_lim:.6e}"
            )

    print("\n" + "="*70)
    print("REVISIÓN DE DUALES ACOTADAS")
    print("="*70)

    hits_sync = 0

    for name, var in ydual.items():
        if name.startswith("sync_ub_"):
            if abs(var.X - UB_sync) <= 1e-6 * max(1.0, UB_sync):
                hits_sync += 1

    print(f"Duales sync_ub pegadas a cota : {hits_sync}")

else:
    print("\nNo hay solución WCEP disponible para reportar.")

# ============================================================
# REGISTRO DE RESULTADOS WCEP
# ============================================================

print("\n" + "="*70)
print("REGISTRO DE OPTIMIZACIÓN WCEP FINAL")
print("="*70)

runtime = float(w.Runtime)
status = int(w.Status)
sol_count = int(w.SolCount)

print(f"Status Gurobi = {status}")
print(f"Tiempo Gurobi optimize() = {runtime:.2f} s = {runtime/60:.2f} min = {runtime/3600:.2f} h")
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
        "runtime_seconds_gurobi": runtime,
        "runtime_minutes_gurobi": runtime / 60,
        "runtime_hours_gurobi": runtime / 3600,
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
        "runtime_seconds_gurobi": runtime,
        "runtime_minutes_gurobi": runtime / 60,
        "runtime_hours_gurobi": runtime / 3600,
        "sol_count": sol_count,
        "obj_val": None,
        "obj_bound": None,
        "mip_gap": None,
        "mip_gap_percent": None,
    }

t_total_script = time.time() - t0_total_script
registro_wcep["runtime_seconds_total_script"] = t_total_script
registro_wcep["runtime_minutes_total_script"] = t_total_script / 60
registro_wcep["runtime_hours_total_script"] = t_total_script / 3600

print(f"Tiempo total script = {t_total_script:.2f} s = {t_total_script/60:.2f} min = {t_total_script/3600:.2f} h")

df_registro_wcep = pd.DataFrame([registro_wcep])
display(df_registro_wcep)

