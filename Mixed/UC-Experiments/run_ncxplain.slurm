#!/bin/bash
# ---------------------------------------------------------------------------
# run_ncxplain.slurm -- Case-Study-I unit-commitment OVERRIDE counterfactuals
# (NCXplain, cost-vector perturbation), timed on NLHPC "Leftraru" so the
# runtimes are reported in the same environment as the DECOMP pipeline.
#
#   sbatch run_ncxplain.slurm
#
# Runs the three chosen overrides (specified by generator/status/hours, no
# "rank"): IEEE 14 commit G1 @ t={17,18}; IEEE 39 commit G5 @ t={17,18,19};
# IEEE 57 de-commit G5 @ t={0,1,2}. NCXplain certifies each in a single cut, so
# this is a tiny job (minutes). Results + timing land in:
#     ncx_results/ncx_timing.csv,  ncx_results/ncx_<grid>.json
# and, IF the cluster Python has matplotlib, the figures:
#     ncx_results/fig_ncx_<grid>_commitment.png  (+ _costchange.png)
#
# If matplotlib is NOT available on the compute node, the timing still runs and
# the small JSONs are written; regenerate the figures locally afterwards with:
#     python run_ncxplain_timing.py --render-only
# ---------------------------------------------------------------------------
#SBATCH --job-name=ce_ncxplain
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x_%j_%N.out
#SBATCH --error=runs/%x_%j_%N.err

set -euo pipefail

# --- environment (NLHPC Leftraru; same as run_pipeline.slurm) --------------
module load gurobi
export PYTHONPATH="$GUROBI_HOME/lib/python3.13/site-packages:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
# Real Gurobi thread cap lives in ./gurobi.env (Threads 8). Stop numpy/BLAS
# from oversubscribing the 8 reserved cores:
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p runs ncx_results

echo "[ncxplain] host=$(hostname)  data=${NCX_DATA_DIR:-./Data}"
srun python run_ncxplain_timing.py
echo "[ncxplain] done -> ncx_results/ncx_timing.csv"
