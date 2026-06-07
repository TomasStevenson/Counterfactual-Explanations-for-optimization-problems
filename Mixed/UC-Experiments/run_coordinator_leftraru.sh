#!/bin/bash
# ---------------------------------------------------------------------------
# run_coordinator_leftraru.sh — drive the v3 iterative best-first coordinator
# to certification on NLHPC (Leftraru), one Slurm array per round.
#
# This automates the manual loop documented in node_obbt_round.slurm:
#   init  ->  { sbatch --wait --array=<open ids> node_obbt_round.slurm ; step }*
# until UB - LB <= tol (CERTIFIED) or no OPEN boxes remain or MAX_ROUNDS hit.
#
# Each round's OPEN boxes are solved CONCURRENTLY as independent array tasks
# (the parallelism the single-machine benchmark could not exploit). `step`
# then prunes (lb >= UB - tol) and splits the SPLIT_K weakest leaves, whose
# children are the next round's OPEN boxes. global_LB = min over leaf ObjBound
# is a valid lower bound at every round, so the run is safe to stop anytime.
#
# USAGE (from the UC-Experiments dir, on a Leftraru login node):
#   ./run_coordinator_leftraru.sh <grid> <rundir> [budget] [split_k] [max_rounds] [tol]
# e.g.
#   ./run_coordinator_leftraru.sh 14 runs/c14 600 4 12 1e-3
#   ./run_coordinator_leftraru.sh 57 runs/c57 600 8 10 1e-3
#
# Resume a half-finished run: re-invoke with the SAME <rundir>; init is skipped
# if state.json already exists, and the loop continues from the current round.
# ---------------------------------------------------------------------------
set -euo pipefail

GRID="${1:?usage: run_coordinator_leftraru.sh <grid> <rundir> [budget] [split_k] [max_rounds] [tol] [maxconc]}"
RUNDIR="${2:?need a run directory, e.g. runs/c14}"
BUDGET="${3:-600}"        # per-box wall budget (s); MUST exceed per-box OBBT (~480s on IEEE 39) -> >=600
SPLIT_K="${4:-4}"         # # weakest leaves split per round = next round's parallel width
MAX_ROUNDS="${5:-12}"     # safety cap on rounds
TOL="${6:-1e-3}"          # absolute UB-LB gap for certification
MAXCONC="${7:-11}"        # max array tasks running AT ONCE. NLHPC grant = 88 cores;
                          # node_obbt_round.slurm asks 8 cpus/task -> 88/8 = 11 concurrent.
                          # The %MAXCONC throttle GUARANTEES the run never exceeds the grant
                          # (extra boxes queue and run as slots free — correct, just serial).

PY="${PY:-python}"        # override with PY=/path/to/ce-env/python if conda isn't pre-activated
COORD="node_obbt_coordinator.py"
SLURM="node_obbt_round.slurm"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# --- small state.json queries (robust; avoids parsing human-readable stdout) ---
open_ids() {   # comma-separated ids of boxes with status == "open"
  "$PY" - "$RUNDIR/state.json" <<'PYEOF'
import json,sys
st=json.load(open(sys.argv[1]))
print(",".join(str(b["id"]) for b in st["boxes"] if b["status"]=="open"))
PYEOF
}
is_certified() {  # prints "1" when UB - LB <= tol (or all boxes pruned), else "0"
  "$PY" - "$RUNDIR/state.json" <<'PYEOF'
import json,sys
st=json.load(open(sys.argv[1])); tol=st["tol"]; UB=st["global_UB"]
leaves=[b for b in st["boxes"] if b["status"] in ("open","solved")]
LB = UB if not leaves else min((b["lb"] if b["lb"] is not None else -1e30) for b in leaves)
print("1" if (UB - LB) <= tol else "0")
PYEOF
}

# --- 1. init (skip if resuming) -------------------------------------------
if [[ ! -f "$RUNDIR/state.json" ]]; then
  echo "[driver] init grid=IEEE${GRID} dir=${RUNDIR} budget=${BUDGET} split_k=${SPLIT_K} tol=${TOL}"
  "$PY" "$COORD" init "$GRID" "$RUNDIR" --budget "$BUDGET" --split-k "$SPLIT_K" --tol "$TOL"
else
  echo "[driver] resuming existing run in ${RUNDIR}"
fi

# --- 2. round loop --------------------------------------------------------
for ((r=1; r<=MAX_ROUNDS; r++)); do
  if [[ "$(is_certified)" == "1" ]]; then
    echo "[driver] CERTIFIED — stopping."; break
  fi
  IDS="$(open_ids)"
  if [[ -z "$IDS" ]]; then
    echo "[driver] no OPEN boxes left (all pruned/solved) — stopping."; break
  fi
  echo "[driver] round ${r}: submitting array --array=${IDS}%${MAXCONC}  (budget ${BUDGET}s/box, <=${MAXCONC} concurrent => <=$((MAXCONC*8)) cores)"
  # --wait blocks until every array task finishes; %MAXCONC caps simultaneously
  # running tasks so the run stays within the 88-core NLHPC grant.
  sbatch --wait --array="${IDS}%${MAXCONC}" "$SLURM" "$RUNDIR"
  echo "[driver] round ${r}: array done — running step (prune + split ${SPLIT_K} weakest)"
  "$PY" "$COORD" step "$RUNDIR"
done

# --- 3. final report ------------------------------------------------------
echo "[driver] final status:"
"$PY" "$COORD" status "$RUNDIR"
