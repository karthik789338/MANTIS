# sourced by every job: activates the venv and loads hpc/config.sh
set -e
source hpc/config.sh
module load python 2>/dev/null || module load miniconda3 2>/dev/null || true
source .venv/bin/activate
export MANTIS_SCALE=$SCALE MANTIS_NVAL=$NVAL OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname)  cpus $SLURM_CPUS_PER_TASK  start $(date)"
