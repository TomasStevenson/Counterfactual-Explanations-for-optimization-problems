#!/bin/bash
# ---------------------------------------------------------------------------
# run_bs.slurm — the LIVE Branch-&-Sandwich phase as a single NLHPC job.
#
# B&S is a sequential best-first tree (NOT parallel), so this is one task with
# several Gurobi threads. It checkpoints every node, so if the Slurm --time cap
# is hit before --max-nodes, just resubmit (without `fresh`) to resume.
#
#   sbatch --wait run_bs.slurm <grid> [max_nodes] [fresh]
# e.g.
#   sbatch --wait run_bs.slurm 57 300 fresh    # new B&S tree for IEEE 57
#   sbatch --wait run_bs.slurm 57 600          # resume / extend it
#
# Produces bs_<grid>_checkpoint.json, which the DECOMP coordinator then consumes
# as the (now live) B&S hint.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bs_phase
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

set -euo pipefail
GRID="${1:?usage: sbatch run_bs.slurm <grid> [max_nodes] [fresh]}"
NODES="${2:-300}"
FRESH_FLAG=""
if [[ "${3:-}" == "fresh" || "${3:-}" == "--fresh" ]]; then FRESH_FLAG="--fresh"; fi

# --- environment (NLHPC Leftraru; same as node_obbt_round.slurm) ------------
module load gurobi
export PYTHONPATH="$GUROBI_HOME/lib/python3.13/site-packages:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[bs_phase] grid=IEEE${GRID} max_nodes=${NODES} fresh='${FRESH_FLAG}' host=$(hostname)"
srun python run_bs.py "${GRID}" --max-nodes "${NODES}" ${FRESH_FLAG}
echo "[bs_phase] done -> bs_${GRID}_checkpoint.json"
