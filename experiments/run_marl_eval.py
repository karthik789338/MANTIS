"""X8 (joint MARL), X9 (full six-condition robustness), X10 (ablation + generalisation).

usage: python experiments/run_marl_eval.py X8 X9 X10 [--procs=N]
"""
import sys
import numpy as np
from common import *
from mantis.marl.algos import sample_train_cfg

P = 1
for a in sys.argv:
    if a.startswith("--procs="):
        P = int(a.split("=")[1])

INDEP = dict(uav_policy="connected", recovery="adaptive", scheduling="fifo", medic_alloc="urgency",
             medic_route="safe", replan="periodic", energy_policy="reactive")
HEUR = {"Independent-H": INDEP, "MANTIS-H": {}}
SCALE = int(os.environ.get("MANTIS_SCALE", "1"))   # multiply number of test scenarios (e.g. 3 on the cluster)


def have(name):
    return os.path.exists(os.path.join(MODELS, f"{name}.pt"))


def eval_suite(n=None):
    n = n or 60 * SCALE
    rng = np.random.default_rng(4242)
    out = []
    for i in range(n):
        c = sample_train_cfg(rng)
        kw = c.to_dict()
        kw["seed"] = EVAL_SEED0 + 5000 + i
        out.append(kw)
    return out


def robust_kw():
    return dict(failures=[dict(t=40, role="gateway"), dict(t=80, role="relay"), dict(t=120, role="observer")],
                obstacles="mixed", mobility="mixed", link_degradation=0.2, severity_dynamics="changing",
                energy=True, battery_hetero=True, endurance_steps=110, n_spare_uavs=2, hazards="mixed",
                disruption="intermittent")


def jobs_for(scen, methods, tags_fn):
    jobs = []
    for i, kw in enumerate(scen):
        for name in methods:
            if name in HEUR:
                jobs.append((kw, HEUR[name], "h", dict(method=name, scen=i, **tags_fn(kw))))
            elif have(name):
                jobs.append((kw, {}, "marl:" + name, dict(method=name, scen=i, **tags_fn(kw))))
    return jobs


def tags(kw):
    return dict(n_c=kw["n_casualties"], n_u=kw["n_uavs"], n_m=kw["n_medics"])


def X8():
    methods = ["Independent-H", "MANTIS-H", "bc", "ippo", "mappo", "vdn", "qmix", "mantis_mappo"]
    return run_jobs(jobs_for(eval_suite(), methods, tags), "X8_marl", P)


def X9(ns=None):
    ns = ns or 20 * SCALE
    methods = ["Independent-H", "MANTIS-H", "ippo", "mappo", "qmix", "mantis_mappo"]
    scen = []
    for nc, nu, nm in [(5, 9, 3), (10, 15, 4), (10, 24, 6)]:
        for i in range(ns):
            scen.append(dict(seed=EVAL_SEED0 + 7000 + i, n_casualties=nc, n_uavs=nu, n_medics=nm, **robust_kw()))
    return run_jobs(jobs_for(scen, methods, tags), "X9_robustness", P)


def X10(ns=None):
    X10a(ns); X10b(); X10c()


def X10a(ns=None):
    ns = ns or 20 * SCALE
    # (a) heuristic component ablation on compounded scenarios
    abl = {"MANTIS-H": {}, "-balanced UAV": dict(uav_policy="connected"), "-coordinated recovery": dict(recovery="adaptive"),
           "-adaptive positioning": dict(positioning="reactive"), "-queue routing": dict(routing="shortest"),
           "-guarded scheduling": dict(scheduling="fifo"), "-urgency allocation (→nearest)": dict(medic_alloc="nearest"),
           "-safe routing": dict(medic_route="shortest"), "-periodic replanning (→static)": dict(replan="static"),
           "-energy-aware": dict(energy_policy="reactive")}
    jobs = []
    for i in range(ns):
        kw = dict(seed=EVAL_SEED0 + 8000 + i, n_casualties=10, n_uavs=15, n_medics=4, **robust_kw())
        for name, pk in abl.items():
            jobs.append((kw, pk, "h", dict(variant=name, scen=i)))
    run_jobs(jobs, "X10a_heuristic_ablation", P)


def X10b():
    # (b) MARL ablation on the X8 suite
    methods = ["MANTIS-H", "bc", "abl_nokl", "abl_mediconly", "abl_uavonly", "mantis_mappo"]
    run_jobs(jobs_for(eval_suite(), methods, tags), "X10b_marl_ablation", P)


def X10c():
    # (c) generalisation: unseen fleet sizes and unseen map generator parameters
    scen = []
    for i in range(10 * SCALE):
        for nu, nm in [(18, 6), (24, 8), (30, 10)]:
            scen.append(dict(seed=EVAL_SEED0 + 9000 + i, n_casualties=10, n_uavs=nu, n_medics=nm, hazards="mixed",
                             mobility="mixed", obstacles="static", severity_dynamics="changing", gen="large-fleet"))
        scen.append(dict(seed=EVAL_SEED0 + 9500 + i, n_casualties=10, n_uavs=12, n_medics=4, map_size=1300.0,
                         n_hazards=5, hazards="mixed", obstacles="mixed", mobility="mixed", gen="unseen-map"))
    jobs = []
    for j, kw in enumerate(scen):
        g = kw.pop("gen")
        for name in ["Independent-H", "MANTIS-H", "mappo", "mantis_mappo"]:
            if name in HEUR:
                jobs.append((kw, HEUR[name], "h", dict(method=name, scen=j, gen=g, **tags(kw))))
            elif have(name):
                jobs.append((kw, {}, "marl:" + name, dict(method=name, scen=j, gen=g, **tags(kw))))
    return run_jobs(jobs, "X10c_generalization", P)


if __name__ == "__main__":
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        globals()[name]()
