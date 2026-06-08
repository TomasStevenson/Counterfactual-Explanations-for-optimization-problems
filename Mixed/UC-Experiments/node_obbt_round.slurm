#!/bin/bash
# ---------------------------------------------------------------------------
# v3 iterative coordinator — ONE ROUND as a Slurm array on NLHPC (Leftraru).
#
# Each round solves the current OPEN boxes in parallel; the coordinator's `step`
# then prunes + splits the weakest box to produce the next round's OPEN boxes.
#
# Workflow (from the UC-Experiments dir):
#   python node_obbt_coordinator.py init 14 runs/c14 --budget 600 --max-iter 1
#   # then repeat until CERTIFIED:
#   #   1) read OPEN ids:  python node_obbt_coordinator.py status runs/c14   (or the last `step` printout)
#   #   2) sbatch --array=<comma-separated OPEN ids> node_obbt_round.slurm runs/c14
#   #   3) (after the array finishes)  python node_obbt_coordinator.py step runs/c14
#   # `step` prints the next round's OPEN ids; loop 2-3.
#
# NOTE: --array must list the CURRENT round's OPEN box ids (they are NOT
# contiguous across rounds). Box ids only grow; old ids become internal/pruned.
#
# $1 = the coordinator run directory (contains state.json).
# ---------------------------------------------------------------------------
#SBATCH --job-name=nobbt_round
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:30:00          # >= max_iter * per-box budget, with margin
#SBATCH --output=runs/%x_%A_%a.out
#SBATCH --error=runs/%x_%A_%a.err

set -euo pipefail
RUNDIR="${1:?usage: sbatch --array=<open ids> node_obbt_round.slurm <rundir>}"

# --- environment (NLHPC Leftraru; verified 2026-06-01) ---------------------
# Partition `main` (AMD EPYC zen4); the gurobi module ships gurobipy but does
# NOT put it on the path. License = NLHPC central floating token server
# ("licencias") — no personal license / GRB_LICENSE_FILE needed, works on
# compute nodes with no internet.
module load gurobi
export PYTHONPATH="$GUROBI_HOME/lib/python3.13/site-packages:${PYTHONPATH:-}"
# If numpy is missing from the module's python, activate your own env too, e.g.:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate ce-env
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # NLHPC policy: cap Threads to the allocation
# Real cap is ./gurobi.env (Threads 8 -> every model). Also cap numpy/BLAS:
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[round task ${SLURM_ARRAY_TASK_ID}] host=$(hostname) rundir=${RUNDIR}"
srun python node_obbt_coordinator.py solve-box "${RUNDIR}" "${SLURM_ARRAY_TASK_ID}"
echo "[round task ${SLURM_ARRAY_TASK_ID}] done"

# After the whole array finishes, run (on a login/compute node):
#   python node_obbt_coordinator.py step "${RUNDIR}"
# which prints the next round's OPEN ids (or CERTIFIED).
