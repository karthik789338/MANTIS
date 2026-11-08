#!/bin/bash
#SBATCH --job-name=mantis-heur
#SBATCH --output=logs/%x-%j.out
# X1-X7 + all 82 base configurations (no learning needed)
source hpc/job_common.sh
cd experiments
python run_heuristic.py X1 X2 X3 X4 X5 X5b X6 X7 X5grid --procs=$SLURM_CPUS_PER_TASK
