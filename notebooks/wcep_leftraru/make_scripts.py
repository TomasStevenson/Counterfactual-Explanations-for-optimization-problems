# -*- coding: utf-8 -*-
"""Genera los 19 scripts batch (run_<job>.py) a partir del notebook de Felipe.

Cada script = PRELUDE + celdas verbatim (sin tocar la lógica del problema).
El prelude solo agrega: shim de display(), medición uniforme de tiempos
([TIMING]) y tiempo total del script vía atexit.
"""
import json
from pathlib import Path

NOTEBOOK = Path(r"C:\Users\tomas\Desktop\Descarbonizacion_WCEP_Unificado_final"
                r"\Descarbonizacion_WCEP_Unificado\Descarbonizacion_WCEP_Unificado"
                r"\WCEP_Descarbonizacion_Unificado.ipynb")
OUTDIR = Path(__file__).parent

# Overrides mínimos por job (autorizado por Tomás 2026-07-16):
# cell 66 trae epsilon_CO2_wcep=None y su búsqueda de emisiones FW no
# calza con las columnas que produce la celda 59 -> se fija manualmente
# el mismo epsilon que su gemela sin PADM (celda 63).
OVERRIDES = {
    "i39_A_padm_v2": [(
        "epsilon_CO2_wcep = None",
        "epsilon_CO2_wcep = 930756.144528 * 0.90  "
        "# OVERRIDE batch: mismo valor que la celda 63 (gemela sin PADM); "
        "el original era None y fallaba (ver make_scripts.py)",
    )],
}

# job -> lista de índices de celdas de código (en orden de ejecución)
JOBS = {
    "i14_fw":        [21, 24, 28],
    "i14_b_dual":    [21, 31],
    "i14_b_padm":    [21, 33],
    "i14_A_dual":    [21, 36],
    "i14_A_padm":    [21, 38],
    "i39_fw":        [53, 56, 59],
    "i39_A_dual_v1": [53, 62],
    "i39_A_dual_v2": [53, 63],
    "i39_A_padm_v1": [53, 65],
    # cell 66 tiene epsilon_CO2_wcep=None y necesita las emisiones del FW
    # en memoria -> incluye la celda 59 (FW), como en el orden del notebook
    "i39_A_padm_v2": [53, 59, 66],
    "i39_b_dual_v1": [53, 69],
    "i39_b_dual_v2": [53, 70],
    "i39_b_padm_v1": [53, 72],
    "i39_b_padm_v2": [53, 73],
    "i57_fw":        [90, 93],
    "i57_A_dual":    [90, 95],
    "i57_A_padm":    [90, 97],
    "i57_b_dual":    [90, 99],
    "i57_b_padm":    [90, 101],
}

PRELUDE = '''\
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
    print("\\n[TIMING] total_script_seconds={:.2f}".format(
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
'''


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for name, idxs in JOBS.items():
        parts = [PRELUDE]
        for i in idxs:
            cell = cells[i]
            assert cell["cell_type"] == "code", f"celda {i} no es de codigo"
            src = "".join(cell["source"])
            parts.append(
                "\n\n# %% ================== celda {} del notebook ==================\n".format(i)
                + src
            )
        text = "\n".join(parts) + "\n"
        for old, new in OVERRIDES.get(name, []):
            assert text.count(old) == 1, f"{name}: override '{old}' aparece {text.count(old)} veces"
            text = text.replace(old, new)
        out = OUTDIR / f"run_{name}.py"
        out.write_text(text, encoding="utf-8", newline="\n")
        flag = " (con override)" if name in OVERRIDES else ""
        print(f"{out.name}: celdas {idxs}, {out.stat().st_size} bytes{flag}")


if __name__ == "__main__":
    main()
