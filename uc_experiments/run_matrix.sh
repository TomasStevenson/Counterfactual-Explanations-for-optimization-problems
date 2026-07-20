#!/bin/bash
# ---------------------------------------------------------------------------
# Single-run experiment matrix for the paper's method comparison (α = 0.05).
#
# Per grid (14, 39, 57), THREE fully self-contained runs — no cached
# checkpoints, no reuse of previous results anywhere:
#   1. Pure Branch-&-Sandwich, 2 h budget          -> runs/matrix_a05/bs2h_<g>*
#   2. Pipeline (fast 15-min B&S warm start -> DECOMP adaptive refinement),
#      2 h TOTAL wall budget                       -> runs/matrix_a05/cap_<g>/
#   3. Same pipeline, uncapped -> certification    -> runs/matrix_a05/full_<g>/
#
# Stage 1 runs the three B&S jobs in parallel (8 cores each). Stages 2 and 3
# run one campaign at a time so no run's wall-clock is contaminated by our own
# queue contention. Launch on a login node:
#   nohup bash run_matrix.sh > runs/matrix_a05.log 2>&1 &
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"
module load gurobi 2>/dev/null
export PYTHONPATH="$GUROBI_HOME/lib/python3.13/site-packages:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export CE_ALPHA=0.05
R=runs/matrix_a05
mkdir -p "$R"

echo "[matrix] stage 1: pure B&S 2h x3 (parallel)  $(date)"
for g in 14 39 57; do
  cat > "$R/bs2h_$g.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=bs2h_$g
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:15:00
#SBATCH --output=$R/bs2h_$g.out
#SBATCH --error=$R/bs2h_$g.err
module load gurobi
export PYTHONPATH="\$GUROBI_HOME/lib/python3.13/site-packages:\${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export GRB_THREADS=\$SLURM_CPUS_PER_TASK
srun python run_bs.py $g --fresh --out $R/bs2h_${g}_checkpoint.json \\
    --time-limit 7200 --max-nodes 1000000
EOF
  sbatch --wait --export=ALL,CE_ALPHA=0.05 "$R/bs2h_$g.slurm" &
done
wait
echo "[matrix] stage 1 done  $(date)"

echo "[matrix] stage 2: pipeline with 2h wall budget (serial)  $(date)"
for g in 14 39 57; do
  python node_obbt_hpc.py drive "$R/cap_$g" --grid "$g" \
      --bs-budget 900 --wall-budget 7200 \
      --n-boxes 16 --emit-budget 30 --budget 600 --max-iter 3 --seed-interp 0
  echo "[matrix] cap_$g finished  $(date)"
done

echo "[matrix] stage 3: pipeline uncapped -> certification (serial)  $(date)"
for g in 14 39 57; do
  python node_obbt_hpc.py drive "$R/full_$g" --grid "$g" \
      --bs-budget 900 \
      --n-boxes 64 --emit-budget 60 --budget 600 --max-iter 3 --seed-interp 0
  echo "[matrix] full_$g finished  $(date)"
done

echo "[matrix] ALL DONE  $(date)"
