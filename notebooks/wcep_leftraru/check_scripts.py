# -*- coding: utf-8 -*-
"""Chequeo estático de los run_*.py: py_compile + nombres globales usados
pero nunca definidos a nivel de módulo (posible dependencia entre celdas
que no quedó incluida)."""
import builtins
import py_compile
import symtable
from pathlib import Path

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "display", "get_ipython"}


def undefined_globals(code, filename):
    top = symtable.symtable(code, filename, "exec")
    defined, used = set(), set()

    def walk(table, is_top):
        for sym in table.get_symbols():
            name = sym.get_name()
            if is_top:
                if sym.is_assigned() or sym.is_imported() or sym.is_namespace():
                    defined.add(name)
                if sym.is_referenced():
                    used.add(name)
            else:
                if sym.is_global() and sym.is_referenced():
                    if sym.is_assigned() and sym.is_declared_global():
                        defined.add(name)
                    used.add(name)
        for child in table.get_children():
            walk(child, False)

    walk(top, True)
    return sorted(used - defined - BUILTINS)


def main():
    ok = True
    for path in sorted(Path(__file__).parent.glob("run_*.py")):
        code = path.read_text(encoding="utf-8")
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            print(f"{path.name}: ERROR de compilacion: {e}")
            ok = False
            continue
        missing = undefined_globals(code, path.name)
        if missing:
            print(f"{path.name}: nombres globales sin definir: {missing}")
            ok = False
        else:
            print(f"{path.name}: OK")
    print("\nRESULTADO:", "OK" if ok else "REVISAR")


if __name__ == "__main__":
    main()
