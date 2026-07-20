"""Validity regression gate for the DECOMP exact+OBBT pipeline.

Runs the b_BS feasibility check (`_check_strongdual_valid.py`) on all three grids
with the PUBLISHED config (strongdual + bilinear_exact + OBBT) and asserts that
each reports VALID — i.e. the known Branch-and-Sandwich CE stays feasible in the
master with obj == F(b_BS).  A tighter LB that excludes a known CE (certifies
above F*) is worse than a loose one, so this is the correctness gate to run
before trusting any LB or after any change to the formulation.

Usage:
    .../ce-env/python.exe _validate_all.py
Exit code 0 iff ALL grids are VALID; 1 otherwise (CI-friendly).
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
GRIDS = ["14", "39", "57"]
# slot args: grid comp bigM factor seg obbt  (matches _check_strongdual_valid.py)
COMMON = ["strongdual", "1e4", "none", "exact", "obbt"]

def run_grid(g):
    cmd = [PY, os.path.join(HERE, "_check_strongdual_valid.py"), g] + COMMON
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    print(f"\n{'='*60}\n[validate] IEEE {g} — {' '.join(COMMON)}\n{'='*60}", flush=True)
    res = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    out = res.stdout + res.stderr
    # Echo the decision lines only (full log is verbose).
    for line in out.splitlines():
        if any(k in line for k in ("B&S F", "oracle at b_BS", "F(b_BS)",
                                   "debug_fix_b", "validate-obbt", "VALID",
                                   "INVALID", "SUSPECT", "Traceback", "Error")):
            print("   " + line.rstrip(), flush=True)
    ok = ("VALID  (" in out) and ("INVALID" not in out) and ("SUSPECT" not in out)
    print(f"   -> IEEE {g}: {'VALID' if ok else 'FAILED'}", flush=True)
    return ok

def main():
    results = {g: run_grid(g) for g in GRIDS}
    print(f"\n{'='*60}\nSUMMARY")
    for g in GRIDS:
        print(f"  IEEE {g:>2}: {'VALID' if results[g] else 'FAILED'}")
    all_ok = all(results.values())
    print(f"\n{'ALL VALID ✓' if all_ok else 'VALIDATION FAILED ✗'}")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
