#!/bin/bash
# ---------------------------------------------------------------------------
# Parallel node-OBBT on NLHPC (Leftraru) — one Slurm array task per b-box.
#
# Each task solves ONE box of the spatial branch-and-bound over the b-box and
# writes result_<id>.json; the aggregator then takes global_LB = min ObjBound.
# The boxes are independent (no inter-task communication). See node_obbt_hpc.py
# and DECOMP_formulation_and_optimality.md §7.5.
#
# Workflow (from the UC-Experiments dir on Leftraru):
#   1) emit the boxes (one quick login-node/compute call):
#        python node_obbt_hpc.py emit 14 16 runs/ieee14 --budget 600 --dims 2
#      -> prints "<N> boxes"; set --array=0-<N-1> below to match.
#   2) submit this array:
#        sbatch --array=0-15 node_obbt.slurm runs/ieee14
#   3) aggregate when done:
#        python node_obbt_hpc.py aggregate runs/ieee14
#
# $1 = the run directory created by `emit` (contains spec.json + box_*.json).
# ---------------------------------------------------------------------------
#SBATCH --job-name=node_obbt
#SBATCH --partition=main
#SBATCH --array=0-15                  # MUST equal 0..(n_boxes-1) from `emit`
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # Gurobi threads per box (see THREADS below)
#SBATCH --mem=8G
#SBATCH --time=02:00:00               # >= max_iter * per-box budget, with margin
#SBATCH --output=runs/%x_%A_%a.out
#SBATCH --error=runs/%x_%A_%a.err

set -euo pipefail
RUNDIR="${1:?usage: sbatch --array=0-N node_obbt.slurm <rundir>}"

# --- environment (adjust to your Leftraru module/conda setup) --------------
# module load Python/3.x  gurobi/xx.x          # if NLHPC provides Gurobi as a module
# OR activate your env that has gurobipy on PYTHONPATH:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate ce-env
export PYTHONPATH="${GUROBI_HOME:-$HOME/gurobi}/lib/python:${PYTHONPATH:-}"
# Token-server (floating) license — point at your NLHPC gurobi.lic:
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
# Pin Gurobi threads to the cpus we requested (reproducible + polite on shared nodes):
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[task ${SLURM_ARRAY_TASK_ID}] host=$(hostname) rundir=${RUNDIR}"
srun python node_obbt_hpc.py solve "${RUNDIR}" "${SLURM_ARRAY_TASK_ID}"
echo "[task ${SLURM_ARRAY_TASK_ID}] done"

# Notes:
#  * The aggregate step is valid at ANY point — global_LB = min over the boxes
#    that have reported is still <= F* (it only RISES as more finish), so a
#    partial array still yields a valid (looser) certificate.
#  * To pass GRB_THREADS through to the solver, set master_threads in
#    node_obbt_hpc._make_decomp (currently Gurobi default = all cores).
