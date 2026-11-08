#!/bin/bash
#SBATCH --job-name=mantis-bc
#SBATCH --output=logs/%x-%j.out
# heuristic demonstrations + behaviour cloning (needed by mantis_mappo and the ablations)
source hpc/job_common.sh
python -m mantis.marl.train --method bc --teacher-episodes $TEACHER_EPISODES --workers $SLURM_CPUS_PER_TASK
