# ===================== EDIT THESE FOR YOUR HELLBENDER ACCOUNT =====================
PARTITION="general"        # check names with:  sinfo -s
ACCOUNT=""                 # your SLURM account if required (e.g. a lab account); leave "" if not needed
CPUS=32                    # cores per job (rollouts run in parallel across these)
MEM="32G"
EPISODES=12000             # MARL training episodes per method
TEACHER_EPISODES=1500      # heuristic demonstrations for behaviour cloning
SCALE=3                    # x3 more test seeds/scenarios than the local run
NVAL=32                    # validation scenarios for checkpoint selection
MAX_PARALLEL_TRAIN=4       # how many training jobs may run at the same time
# ==================================================================================
