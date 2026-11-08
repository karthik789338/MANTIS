#!/bin/bash
#SBATCH --job-name=mantis-eval
#SBATCH --output=logs/%x-%j.out
# checkpoint selection on validation seeds, then X8-X10 on test seeds, then the report
source hpc/job_common.sh
cd experiments
python select_checkpoints.py mantis_mappo mappo ippo vdn qmix abl_nokl abl_mediconly abl_uavonly --procs=$SLURM_CPUS_PER_TASK
python run_marl_eval.py X8 X9 X10 --procs=$SLURM_CPUS_PER_TASK
cd ..
python analysis/report.py
echo "done: see results/RESULTS.md and results/figures/"
