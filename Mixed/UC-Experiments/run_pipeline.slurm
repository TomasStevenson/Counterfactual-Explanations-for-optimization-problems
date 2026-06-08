#!/bin/bash
# ---------------------------------------------------------------------------
# run_pipeline.slurm — ONE self-contained NLHPC job per grid:
#   live Branch-&-Sandwich warm-start  ->  DECOMP (node-OBBT certifier),
# timed end to end (B&S + DECOMP) as a single wall clock.
#
#   sbatch run_pipeline.slurm <grid> [time_limit_s] [per_box_s] [bs_nodes]
# e.g.
#   sbatch run_pipeline.slurm 39 3600 600 250
#   sbatch run_pipeline.slurm 57 7200 600 250
#   sbatch run_pipeline.slurm 14 7200 600 250
#
# One task, 8 cores. Three grids at once = 24 cores, well within the 88-core
# grant, so you can submit all three simultaneously. Result + timing land in
# pipeline_results/pipeline_<grid>.json and pipeline_results/pipeline_results.csv.
# ---------------------------------------------------------------------------
#SBATCH --job-name=ce_pipeline
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x_%j_%N.out
#SBATCH --error=runs/%x_%j_%N.err

set -euo pipefail
GRID="${1:?usage: sbatch run_pipeline.slurm <grid> [time_limit_s] [per_box_s] [bs_nodes]}"
TL="${2:-3600}"
PERBOX="${3:-600}"
BSNODES="${4:-250}"     # B&S node cap (default 250 for all grids; a warm start needs no more)

# --- environment (NLHPC Leftraru; same as the other slurm files) -----------
module load gurobi
export PYTHONPATH="$GUROBI_HOME/lib/python3.13/site-packages:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[pipeline] grid=IEEE${GRID}  time_limit=${TL}s  per_box=${PERBOX}s  bs_nodes=${BSNODES}  host=$(hostname)"
srun python run_pipeline.py "${GRID}" --time-limit "${TL}" --per-box "${PERBOX}" --bs-nodes "${BSNODES}"
echo "[pipeline] done -> pipeline_results/pipeline_${GRID}.json"
