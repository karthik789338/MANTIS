#!/bin/bash
#SBATCH --job-name=mantis-train
#SBATCH --output=logs/%x-%A_%a.out
# array job: one MARL method per task
source hpc/job_common.sh
METHODS=(mantis_mappo mappo ippo vdn qmix abl_nokl abl_mediconly abl_uavonly)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
echo "training $M for $EPISODES episodes on $SLURM_CPUS_PER_TASK cores"
python -m mantis.marl.train --method $M --episodes $EPISODES --workers $SLURM_CPUS_PER_TASK
