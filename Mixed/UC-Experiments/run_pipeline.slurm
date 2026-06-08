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
#   sbatch run_pipeline.slurm 14 7200 600 0     # bs_nodes=0 -> skip B&S, reuse cached F=2.382 hint
#
# bs_nodes: B&S node cap; pass 0 (or "skip"/"cached") to SKIP the live B&S phase
# and reuse the committed bs_<grid>_checkpoint.json. Use 0 for IEEE 14 — its good
# F=2.382 CE needs ~1144 nodes, so the 250-node default degrades it to F=4.467.
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
# Real Gurobi cap lives in ./gurobi.env (Threads 8 -> every model: B&S, master, OBBT/dispatch LPs).
# GRB_THREADS above only reaches the DECOMP master. Also stop numpy/BLAS oversubscribing:
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# bs_nodes 0 / "skip" / "cached" => reuse the committed checkpoint, no live recompute.
if [[ "${BSNODES}" == "0" || "${BSNODES}" == "skip" || "${BSNODES}" == "cached" ]]; then
  BS_ARG="--skip-bs"
else
  BS_ARG="--bs-nodes ${BSNODES}"
fi

echo "[pipeline] grid=IEEE${GRID}  time_limit=${TL}s  per_box=${PERBOX}s  bs=${BS_ARG}  host=$(hostname)"
srun python run_pipeline.py "${GRID}" --time-limit "${TL}" --per-box "${PERBOX}" ${BS_ARG}
echo "[pipeline] done -> pipeline_results/pipeline_${GRID}.json"
