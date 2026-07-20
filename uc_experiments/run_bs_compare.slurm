#!/bin/bash
# ---------------------------------------------------------------------------
# run_bs_compare.slurm — D2: PURE Branch-&-Sandwich vs the B&S->DECOMP pipeline.
#
# Answers the reviewer comment "Add a Pure Branch and Sandwich run for the
# Decomp problems to compare results and time." Runs pure B&S FROM SCRATCH (no
# cached incumbent) under a fixed wall-clock budget and records
# F*/LB/gap/certified/time/CE into bs_results/ for the paper's comparison table.
#
# NON-DESTRUCTIVE: writes its own checkpoint to bs_results/bs_compare_<grid>.json,
# so the pipeline's warm-start hints bs_<grid>_checkpoint.json are left UNTOUCHED
# and are NOT loaded as an incumbent (that fresh path does not exist at start ->
# B&S begins with no incumbent = genuinely pure). See run_bs.py + build_grid.
#
# Budget: default 7200 s (2 h) per grid — this is >= every carbon-free pipeline
# TOTAL time (39 ~22 min, 57 ~75 min, 14 ~67 min), so pure B&S is given at least
# as much wall-clock as the full pipeline used. Grids are carbon-FREE (matches
# node_obbt_hpc.py _SETUP and the paper's CARBON=0 decision).
#
#   sbatch run_bs_compare.slurm 14
#   sbatch run_bs_compare.slurm 39
#   sbatch run_bs_compare.slurm 57
#   sbatch run_bs_compare.slurm 39 10800   # override budget to 3 h (also raise --time below)
#
# Submit the three grids in parallel (independent single-node jobs). When all
# three finish, bs_results/bs_results.csv has one row per grid for the table.
# ---------------------------------------------------------------------------
#SBATCH --job-name=bs_cmp
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:45:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err

set -euo pipefail
GRID="${1:?usage: sbatch run_bs_compare.slurm <grid> [time_limit_s]}"
TLIM="${2:-7200}"

# --- environment (NLHPC Leftraru; same as run_bs.slurm / node_obbt_round.slurm) ---
module load gurobi
export PYTHONPATH="$GUROBI_HOME/lib/python3.13/site-packages:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export GRB_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Real Gurobi thread cap lives in ./gurobi.env (Threads 8). B&S is the uncapped,
# sequential phase, so the per-node MIQCP gets all 8 threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p bs_results runs
echo "[bs_cmp] grid=IEEE${GRID}  time_limit=${TLIM}s  host=$(hostname)  $(date -Is)"

# --max-nodes effectively unbounded so the WALL-CLOCK budget is what binds.
# --out points at a fresh, separate checkpoint => pure-from-scratch + non-destructive.
srun python run_bs.py "${GRID}" \
    --max-nodes 1000000 \
    --time-limit "${TLIM}" \
    --out "bs_results/bs_compare_${GRID}.json" \
    --results bs_results

echo "[bs_cmp] done -> bs_results/bs_result_${GRID}.json  $(date -Is)"
