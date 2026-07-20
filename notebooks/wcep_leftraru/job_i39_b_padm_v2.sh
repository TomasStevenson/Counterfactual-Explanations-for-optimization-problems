#!/bin/bash
#SBATCH --job-name=i39_b_padm_v2
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o i39_b_padm_v2_%j.out
#SBATCH -e i39_b_padm_v2_%j.err

module purge
module load gurobi
export PYTHONPATH=$GUROBI_HOME/lib/python3.13/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/wcep
python run_i39_b_padm_v2.py
