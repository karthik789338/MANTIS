#!/bin/bash
# Submits the full pipeline with dependencies. Run from the mantis/ folder:  bash hpc/submit_all.sh
set -e
cd "$(dirname "$0")/.."
source hpc/config.sh
mkdir -p logs
OPT="--partition=$PARTITION --cpus-per-task=$CPUS --mem=$MEM --nodes=1 --ntasks=1"
[ -n "$ACCOUNT" ] && OPT="$OPT --account=$ACCOUNT"
H=$(sbatch --parsable $OPT --time=08:00:00 hpc/job_heuristic.sbatch)
B=$(sbatch --parsable $OPT --time=04:00:00 hpc/job_bc.sbatch)
T=$(sbatch --parsable $OPT --time=24:00:00 --array=0-7%$MAX_PARALLEL_TRAIN --dependency=afterok:$B hpc/job_train.sbatch)
E=$(sbatch --parsable $OPT --time=12:00:00 --dependency=afterok:$T:$H hpc/job_eval.sbatch)
echo "submitted: heuristic=$H  bc=$B  train(array 0-7)=$T  eval+report=$E"
echo "watch with:  squeue -u $USER      logs in: $(pwd)/logs/"
