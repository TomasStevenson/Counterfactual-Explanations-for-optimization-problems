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



# %% ================== celda 90 del notebook ==================
import os
import pandas as pd
import numpy as np

# ============================================================
# 0) CONFIG: usar CSV IEEE57 finales
# ============================================================
PATH = "."

PATH_GEN   = os.path.join(PATH, "generadores_procesados_ieee57_final.csv")
PATH_LINE  = os.path.join(PATH, "lineas_procesadas_ieee57.csv")
PATH_LOAD  = os.path.join(PATH, "demanda_nodal_ieee57_final.csv")
PATH_SOLAR = os.path.join(PATH, "perfil_solar_ieee57_final.csv")

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
L_id = []
from_bus_line = {}
to_bus_line = {}
F0 = {}
Bline = {}

for idx, row in df_lines.iterrows():
    ell = int(idx + 1)   # o usar row["linea"] si existe
    i = int(row["from_bus"])
    j = int(row["to_bus"])

    L_id.append(ell)
    from_bus_line[ell] = i
    to_bus_line[ell] = j
    F0[ell] = float(row["F_max"])

    x_val = float(row["Reactancia_X"])
    if abs(x_val) < 1e-6:
        x_val = 1e-4
    Bline[ell] = 1.0 / x_val

lineas_entran = {i: [] for i in B}
lineas_salen  = {i: [] for i in B}

for ell in L_id:
    i = from_bus_line[ell]
    j = to_bus_line[ell]

    if i in lineas_salen:
        lineas_salen[i].append(ell)
    if j in lineas_entran:
        lineas_entran[j].append(ell)

print(f"Red: |L|={len(L_id)}")

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


# %% ================== celda 93 del notebook ==================
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time

t0_total_fw = time.time()

# ============================================================
# FINAL IEEE57
# PASO 1) FORWARD PROBLEM - FORMULACIÓN ORIGINAL + WELL-POSED HREQ
#
# Hreq queda en su posición original y se ajusta para well-posedness:
#   sum_g Hg[g] u[g,t,y] + sum_m h_syn[m,t,y] >= Hreq[y]
#
# No se usa zH.
# No se mueve Hreq hacia A.
# ============================================================


# ============================================================
# 0) OPCIONES WELL-POSED IEEE57
# ============================================================
# En IEEE57 el Hreq original estaba demasiado ajustado: el FW operaba
# exactamente en el límite de inercia en algunos periodos solares.
# Para dar margen operativo, se escala Hreq usando el criterio discutido:
# alpha_H original ≈ 0.35 -> alpha_H well-posed = 0.28.
#
# Importante:
#   - NO se introduce zH.
#   - Hreq sigue estando en el RHS de la restricción de inercia.
#   - Se construye Hreq_fw y NO se sobreescribe Hreq.
#
# Si quieres 0.25 en vez de 0.28, cambia solo HREQ_ALPHA_WELLPOSED.
# ============================================================

USE_WELLPOSED_HREQ = True
HREQ_ALPHA_ORIGINAL = 0.35
HREQ_ALPHA_WELLPOSED = 0.28
HREQ_MIN_RELATIVE_MARGIN = 1e-4

# ============================================================
# 0.1) DEMANDA CORREGIDA POR CAPACIDAD
# ============================================================
# Tal como en IEEE39, el Forward usa d_fw:
#   - si ya existe d_fw en memoria, lo respeta;
#   - si no existe, lo reconstruye desde d;
#   - en modo auto_capacity, escala la demanda para que la demanda pico
#     no supere DEMAND_CAP_MARGIN * sum_g Pmax[g].
#
# Esto evita que el dataset parta artificialmente con demanda mayor
# que la capacidad máxima total del sistema.
DEMAND_SCALE_MODE = globals().get("DEMAND_SCALE_MODE", "auto_capacity")
DEMAND_SCALE_MANUAL = float(globals().get("DEMAND_SCALE_MANUAL", 1.0))
DEMAND_CAP_MARGIN = float(globals().get("DEMAND_CAP_MARGIN", 0.98))

FW_TIME_LIMIT = None  # puedes poner 5*60*60 si quieres límite de 5 horas


# ============================================================
# 1) ROBUSTECER LÍNEAS
# ============================================================
# En IEEE57 usamos IDs únicos de línea
L_arcs = L_id

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

for ell in L_arcs:
    i = from_bus_line[ell]
    j = to_bus_line[ell]

    if j in lineas_entran:
        lineas_entran[j].append(ell)

    if i in lineas_salen:
        lineas_salen[i].append(ell)


# ============================================================
# 4) PRE-CHECKS
# ============================================================
print("\n==============================")
print("PRE-CHECKS FORWARD IEEE57 - FORMULACIÓN ORIGINAL")
print("==============================")

missing_d = [
    (i, t, y)
    for i in B
    for t in T
    for y in Y
    if (i, t, y) not in d
]

if missing_d:
    print(f"⚠️ Faltan {len(missing_d)} entradas en d[(i,t,y)]. Ejemplo:", missing_d[:5])
else:
    print("✅ d[(i,t,y)] completo para B,T,Y.")


# ============================================================
# 4.0) CONSTRUIR DEMANDA CORREGIDA d_fw
# ============================================================
# d_fw será la demanda que realmente usa el Forward.
# No sobreescribimos d, para mantener trazabilidad del dataset original.
if "d_fw" not in globals():
    print("\n⚠️ No existe d_fw. Se construye demanda corregida para IEEE57.")

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

else:
    print("\n✅ Usando d_fw existente desde una celda anterior.")
    d_original = {k: float(v) for k, v in d.items()}
    demanda_original_ty = {
        (t, y): sum(
            d_original[(i, t, y)]
            for i in B
            if (i, t, y) in d_original
        )
        for t in T
        for y in Y
    }
    peak_dem_original = max(demanda_original_ty.values())
    cap_total_pmax = sum(float(Pmax[g]) for g in G)

    # Si d_fw ya existe, inferimos el factor aproximado solo para reporte.
    peak_dem_fw_tmp = max(
        sum(float(d_fw[(i, t, y)]) for i in B)
        for t in T
        for y in Y
    )
    demand_scale_factor = (
        peak_dem_fw_tmp / peak_dem_original
        if peak_dem_original > 1e-9
        else 1.0
    )

peak_dem_fw = max(
    sum(float(d_fw[(i, t, y)]) for i in B)
    for t in T
    for y in Y
)

print("\n==============================")
print("DEMANDA CORREGIDA IEEE57")
print("==============================")
print(f"DEMAND_SCALE_MODE        = {DEMAND_SCALE_MODE}")
print(f"DEMAND_CAP_MARGIN        = {DEMAND_CAP_MARGIN:.4f}")
print(f"Demanda pico original    = {peak_dem_original:.6f}")
print(f"Pmax total sistema       = {cap_total_pmax:.6f}")
print(f"Factor demanda usado     = {demand_scale_factor:.8f}")
print(f"Demanda pico usada d_fw  = {peak_dem_fw:.6f}")
print(f"Margen cap/demanda d_fw  = {cap_total_pmax - peak_dem_fw:.6f}")

for y in Y:
    for t in T:
        dem_tot = sum(d_fw[(i, t, y)] for i in B)
        cap_tot = sum(Pmax[g] for g in G)

        if cap_tot + 1e-6 < dem_tot:
            print(
                f"⚠️ Capacidad total {cap_tot:.2f} < demanda {dem_tot:.2f} "
                f"en (t={t}, y={y})."
            )
            break

max_sync_H = sum(Hg.get(g, 0.0) for g in GS)

gamma_fwd_check = gamma_base.copy()
max_hsyn = (
    sum(gamma_fwd_check[m] * xbar[m] for m in M)
    if len(M) > 0
    else 0.0
)

max_H_total = max_sync_H + max_hsyn

kappa_fwd_check = kappa_base.copy()
max_R_total = (
    sum(kappa_fwd_check[m] * xbar[m] for m in M)
    if len(M) > 0
    else 0.0
)

# ============================================================
# 4.1) AJUSTE WELL-POSED DE HREQ
# ============================================================
Hreq_original = {y: float(Hreq[y]) for y in Y}

if USE_WELLPOSED_HREQ:
    if abs(HREQ_ALPHA_ORIGINAL) <= 1e-12:
        raise ValueError("HREQ_ALPHA_ORIGINAL no puede ser cero.")

    hreq_scale_factor = float(HREQ_ALPHA_WELLPOSED) / float(HREQ_ALPHA_ORIGINAL)

    Hreq_fw = {
        y: hreq_scale_factor * Hreq_original[y]
        for y in Y
    }

    # Protección adicional: aunque usemos alpha_H=0.28, no permitimos
    # que Hreq_fw quede pegado al máximo teórico del sistema.
    for y in Y:
        Hreq_fw[y] = min(
            float(Hreq_fw[y]),
            (1.0 - HREQ_MIN_RELATIVE_MARGIN) * float(max_H_total)
        )

else:
    hreq_scale_factor = 1.0
    Hreq_fw = Hreq_original.copy()

print("\n==============================")
print("AJUSTE WELL-POSED HREQ IEEE57")
print("==============================")
print(f"USE_WELLPOSED_HREQ    = {USE_WELLPOSED_HREQ}")
print(f"alpha_H original ref. = {HREQ_ALPHA_ORIGINAL:.4f}")
print(f"alpha_H well-posed    = {HREQ_ALPHA_WELLPOSED:.4f}")
print(f"factor Hreq usado     = {hreq_scale_factor:.8f}")

for y in Y:
    print(
        f"Hreq año {y}: original={Hreq_original[y]:.6f} -> "
        f"Hreq_fw={Hreq_fw[y]:.6f}"
    )

print(
    f"Max H posible (sync + hsyn) = {max_H_total:.3f} "
    f"(sync={max_sync_H:.3f}, hsyn={max_hsyn:.3f})"
)

print(f"Max R posible (FFR)         = {max_R_total:.3f}")

for y in Y:
    if Hreq_fw[y] > max_H_total + 1e-6:
        print(
            f"❌ ALERTA: Hreq[{y}]={Hreq_fw[y]:.3f} > MaxH={max_H_total:.3f} "
            f"=> INFEASIBLE por inercia."
        )

    if Rreq[y] > max_R_total + 1e-6:
        print(
            f"❌ ALERTA: Rreq[{y}]={Rreq[y]:.3f} > MaxR={max_R_total:.3f} "
            f"=> INFEASIBLE por FFR."
        )


# ============================================================
# 5) MODELO FORWARD IEEE57 - FORMULACIÓN ORIGINAL
# ============================================================
fw_base = gp.Model("FW_base_IEEE57_original")
fw_base.Params.OutputFlag = 1
if FW_TIME_LIMIT is not None:
    fw_base.Params.TimeLimit = FW_TIME_LIMIT


# ============================================================
# 6) VARIABLES
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
# 7) RESTRICCIONES
# ============================================================

# ------------------------------------------------------------
# 7.1) Balance nodal
# ------------------------------------------------------------
for i in B:
    for t in T:
        for y in Y:
            gen = gp.quicksum(
                p[g, t, y]
                for g in G_en_bus.get(i, [])
            )

            inflow = gp.quicksum(
                f[ell, t, y]
                for ell in lineas_entran[i]
            )

            outflow = gp.quicksum(
                f[ell, t, y]
                for ell in lineas_salen[i]
            )

            fw_base.addConstr(
                gen - d_fw[(i, t, y)] + s[i, t, y] + inflow - outflow == 0,
                name=f"balance[{i},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.2) ENS <= demanda
# ------------------------------------------------------------
for i in B:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                s[i, t, y] <= d_fw[(i, t, y)],
                name=f"ens_ub[{i},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.3) Flujo DC
# ------------------------------------------------------------
for ell in L_arcs:
    i = from_bus_line[ell]
    j = to_bus_line[ell]

    for t in T:
        for y in Y:
            fw_base.addConstr(
                f[ell, t, y]
                == Bline[ell] * (theta[i, t, y] - theta[j, t, y]),
                name=f"dcflow[{ell},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.4) Límites de línea
# ------------------------------------------------------------
for ell in L_arcs:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                f[ell, t, y] <= F0[ell],
                name=f"f_ub[{ell},{t},{y}]"
            )

            fw_base.addConstr(
                f[ell, t, y] >= -F0[ell],
                name=f"f_lb[{ell},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.5) Bus slack
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
# 7.6) Disponibilidad de generación
# ------------------------------------------------------------
for g in G:
    for t in T:
        cap_gt = P_disp.get((g, t), Pmax[g])

        for y in Y:
            fw_base.addConstr(
                p[g, t, y] <= cap_gt,
                name=f"pdisp[{g},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.7) Síncronos: Pmin*u <= p <= Pmax*u
# ------------------------------------------------------------
for g in GS:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                p[g, t, y] >= Pmin.get(g, 0.0) * u[g, t, y],
                name=f"sync_min[{g},{t},{y}]"
            )

            fw_base.addConstr(
                p[g, t, y] <= Pmax[g] * u[g, t, y],
                name=f"sync_max[{g},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.8) Límite inversión
# ------------------------------------------------------------
for m in M:
    for y in Y:
        fw_base.addConstr(
            x[m, y] <= xbar[m],
            name=f"x_ub[{m},{y}]"
        )


# ------------------------------------------------------------
# 7.9) h_syn <= gamma * x
# ------------------------------------------------------------
for m in M:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                h_syn[m, t, y] <= gamma_base[m] * x[m, y],
                name=f"hsyn_cap[{m},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.10) r_ffr <= kappa * x
# ------------------------------------------------------------
for m in M:
    for t in T:
        for y in Y:
            fw_base.addConstr(
                r_ffr[m, t, y] <= kappa_base[m] * x[m, y],
                name=f"ffr_cap[{m},{t},{y}]"
            )


# ------------------------------------------------------------
# 7.11) Inercia requerida - FORMULACIÓN ORIGINAL
#
#   sum_g Hg[g] u[g,t,y] + sum_m h_syn[m,t,y] >= Hreq_fw[y]
#
# Aquí Hreq queda en el lado derecho.
# No se usa zH.
# No se mueve Hreq hacia A.
# ------------------------------------------------------------
for t in T:
    for y in Y:
        fw_base.addConstr(
            gp.quicksum(
                Hg.get(g, 0.0) * u[g, t, y]
                for g in GS
            )
            + gp.quicksum(
                h_syn[m, t, y]
                for m in M
            )
            >= Hreq_fw[y],
            name=f"inertia_req[{t},{y}]"
        )


# ------------------------------------------------------------
# 7.12) FFR requerida
# ------------------------------------------------------------
for t in T:
    for y in Y:
        fw_base.addConstr(
            gp.quicksum(
                r_ffr[m, t, y]
                for m in M
            ) >= Rreq[y],
            name=f"ffr_req[{t},{y}]"
        )


# ============================================================
# 8) ACTUALIZAR MODELO
# ============================================================
fw_base.update()

print(f"\nVariables: {fw_base.NumVars} | Restricciones: {fw_base.NumConstrs}")


# ============================================================
# 9) OBJETIVO
# ============================================================
obj_oper = gp.quicksum(
    cg[g] * p[g, t, y]
    for g in G
    for t in T
    for y in Y
)

obj_ens = VOLL * gp.quicksum(
    s[i, t, y]
    for i in B
    for t in T
    for y in Y
)

obj_inv = gp.quicksum(
    cinv[m] * x[m, y]
    for m in M
    for y in Y
)

fw_base.setObjective(
    obj_oper + obj_ens + obj_inv,
    GRB.MINIMIZE
)


# ============================================================
# 10) RESOLVER
# ============================================================
fw_base.optimize()

fw_runtime_optimize = float(fw_base.Runtime)
fw_runtime_total = time.time() - t0_total_fw

print("\n==============================")
print("TIEMPOS FORWARD IEEE57")
print("==============================")
print(f"Tiempo Gurobi optimize() = {fw_runtime_optimize:.2f} s = {fw_runtime_optimize/60:.2f} min")
print(f"Tiempo total script      = {fw_runtime_total:.2f} s = {fw_runtime_total/60:.2f} min")


# ============================================================
# 11) DIAGNÓSTICO BÁSICO
# ============================================================
if fw_base.Status == GRB.INFEASIBLE:
    print("\n❌ Modelo INFEASIBLE.")
    print("No se escribió archivo IIS/ILP.")

elif fw_base.Status == GRB.OPTIMAL:
    print("\n✅ Modelo resuelto óptimamente.")
    print(f"Valor objetivo = {fw_base.ObjVal:,.4f}")

    ens_total = sum(
        s[i, t, y].X
        for i in B
        for t in T
        for y in Y
    )

    print(f"ENS total = {ens_total:.6f}")

else:
    print(f"\n⚠️ Modelo terminó con status Gurobi = {fw_base.Status}")


# ============================================================
# 12) RESULTADOS FORWARD IEEE57 - FORMULACIÓN ORIGINAL
# ============================================================

if fw_base.Status == GRB.OPTIMAL:

    print("\n==============================")
    print("RESULTADOS FW_base IEEE57 - FORMULACIÓN ORIGINAL")
    print("==============================")
    print(f"Valor objetivo total = {fw_base.ObjVal:,.4f}")

    # --------------------------------------------------------
    # 12.1) Tabla de generación p[g,t,y]
    #       y emisiones e_g * p[g,t,y]
    # --------------------------------------------------------
    filas_p = []

    for g in G:
        for t in T:
            for y in Y:
                val_p = p[g, t, y].X
                val_ep = eg.get(g, 0.0) * val_p

                filas_p.append({
                    "gen": g,
                    "t": t,
                    "y": y,
                    "bus": g_bus[g],
                    "Es_IBR": int(g in M),
                    "emision_factor": eg.get(g, 0.0),
                    "p_gty": val_p,
                    "e_g_p_gty": val_ep
                })

    df_p = pd.DataFrame(filas_p)


    # --------------------------------------------------------
    # 12.2) Mostrar solo generación positiva
    # --------------------------------------------------------
    df_p_pos = df_p[df_p["p_gty"] > 1e-6].copy()
    df_p_pos = df_p_pos.sort_values(["y", "t", "gen"]).reset_index(drop=True)

    print("\n--- Tabla p[g,t,y] y e_g * p[g,t,y] (solo valores > 0) ---")
    display(df_p_pos)


    # --------------------------------------------------------
    # 12.3) Resumen por periodo y año
    # --------------------------------------------------------
    resumen_ty = (
        df_p
        .groupby(["y", "t"], as_index=False)
        .agg(
            generacion_total=("p_gty", "sum"),
            emisiones_totales_intervalo=("e_g_p_gty", "sum")
        )
    )

    print("\n--- Resumen por (y,t) ---")
    display(resumen_ty)


    # --------------------------------------------------------
    # 12.4) Emisiones totales por año
    # --------------------------------------------------------
    emisiones_por_anio = (
        df_p
        .groupby("y", as_index=False)
        .agg(
            emisiones_totales_anuales=("e_g_p_gty", "sum"),
            generacion_total_anual=("p_gty", "sum")
        )
    )

    emisiones_por_anio["epsilon_90pct"] = 0.90 * emisiones_por_anio["emisiones_totales_anuales"]

    print("\n--- Emisiones totales por año ---")
    display(emisiones_por_anio)

    print("\n--- Valores sugeridos para epsilon_CO2_wcep = 90% emisiones FW ---")
    for _, row in emisiones_por_anio.iterrows():
        print(
            f"y={int(row['y'])}: "
            f"E_FW={row['emisiones_totales_anuales']:.6f} | "
            f"0.90*E_FW={row['epsilon_90pct']:.6f}"
        )


    # --------------------------------------------------------
    # 12.5) Resumen por generador
    # --------------------------------------------------------
    resumen_g = (
        df_p
        .groupby("gen", as_index=False)
        .agg(
            bus=("bus", "first"),
            Es_IBR=("Es_IBR", "first"),
            emision_factor=("emision_factor", "first"),
            generacion_total=("p_gty", "sum"),
            emisiones_totales=("e_g_p_gty", "sum")
        )
        .sort_values("generacion_total", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Resumen por generador ---")
    display(resumen_g)


    # --------------------------------------------------------
    # 12.6) ENS total y por bus
    # --------------------------------------------------------
    filas_s = []

    for i in B:
        for t in T:
            for y in Y:
                filas_s.append({
                    "bus": i,
                    "t": t,
                    "y": y,
                    "ENS": s[i, t, y].X
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
    # 12.7) Inversiones IBR
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
    # 12.8) Inercia sintética h_syn
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
    # 12.9) FFR r_ffr
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
    # 12.10) Chequeo agregado de inercia y FFR por periodo/año
    # --------------------------------------------------------
    filas_req = []

    for t in T:
        for y in Y:
            H_sync_val = sum(
                Hg.get(g, 0.0) * u[g, t, y].X
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
                "Hreq_original": Hreq_original[y],
                "Hreq_fw": Hreq_fw[y],
                "margen_H": H_sync_val + H_syn_val - Hreq_fw[y],
                "R_total": R_val,
                "Rreq": Rreq[y],
                "margen_R": R_val - Rreq[y]
            })

    df_req = pd.DataFrame(filas_req)

    print("\n--- Chequeo de requerimientos Hreq y Rreq ---")
    display(
        df_req
        .sort_values(["y", "t"])
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # 12.11) Flujos de línea
    # --------------------------------------------------------
    filas_f = []

    for ell in L_arcs:
        i = from_bus_line[ell]
        j = to_bus_line[ell]

        for t in T:
            for y in Y:
                val_f = f[ell, t, y].X
                Fij = F0[ell]

                filas_f.append({
                    "linea": ell,
                    "from_bus": i,
                    "to_bus": j,
                    "t": t,
                    "y": y,
                    "f_ellty": val_f,
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
