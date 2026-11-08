"""X1-X7 and the 82-configuration grid (heuristic components, matched seeds).

usage: python experiments/run_heuristic.py X1 [X2 ...] [--procs N]
"""
import sys
from common import *

P = 1
for a in sys.argv:
    if a.startswith("--procs="):
        P = int(a.split("=")[1])


SCALE = int(os.environ.get("MANTIS_SCALE", "1"))   # multiply the number of seeds (e.g. 3 on the cluster)


def seeds(n):
    return [EVAL_SEED0 + i for i in range(n * SCALE)]


def X1(ns=10):
    """Condition 2: UAV failure and adaptive role reconfiguration."""
    scen = {
        "observer": dict(failures=[dict(t=30, role="observer")]),
        "relay": dict(failures=[dict(t=30, role="relay")]),
        "gateway": dict(failures=[dict(t=30, role="gateway")]),
        "successive": dict(failures=[dict(t=30, role="relay"), dict(t=90, role="observer")]),
        "multiple": dict(failures=[dict(t=30, role="gateway"), dict(t=60, role="relay"), dict(t=90, role="observer")]),
        "intermittent+relay": dict(disruption="intermittent", failures=[dict(t=30, role="relay")]),
        "blackout+gateway": dict(disruption="complete", blackout=(40, 70), failures=[dict(t=45, role="gateway")]),
    }
    jobs = []
    for nc, nu, nm in [(1, 3, 1), (5, 5, 3), (10, 9, 4), (10, 15, 4)]:
        for sn, kw in scen.items():
            if sn == "multiple" and nu < 5:
                continue
            for rec in ["none", "adaptive", "coordinated"]:
                for s in seeds(ns):
                    jobs.append((dict(seed=s, n_casualties=nc, n_uavs=nu, n_medics=nm, **kw), dict(recovery=rec), "h",
                                 dict(exp="X1", scenario=sn, recovery=rec, cfg=f"{nc}c/{nu}u/{nm}m")))
    return run_jobs(jobs, "X1_failure_recovery", P)


def X2(ns=10):
    """Condition 3: static and dynamic obstacles."""
    jobs = []
    for nc, nu, nm in [(5, 7, 3), (10, 15, 4)]:
        for ob in ["clear", "static", "dynamic", "mixed"]:
            for pos in ["static", "reactive", "adaptive"]:
                for s in seeds(ns):
                    jobs.append((dict(seed=s, n_casualties=nc, n_uavs=nu, n_medics=nm, obstacles=ob), dict(positioning=pos), "h",
                                 dict(exp="X2", obstacles=ob, positioning=pos, cfg=f"{nc}c/{nu}u/{nm}m")))
    return run_jobs(jobs, "X2_obstacles", P)


def X3(ns=8):
    """Condition 4: multiple casualties and scalable UAV coordination."""
    from mantis.config import FIVE_UAVS, TEN_UAVS
    jobs = []
    for nc, ulist, nm in [(5, FIVE_UAVS, 3), (10, TEN_UAVS, 4)]:
        for nu in ulist:
            for pol in ["fixed", "coverage", "connected", "balanced"]:
                for s in seeds(ns):
                    jobs.append((dict(seed=s, n_casualties=nc, n_uavs=nu, n_medics=nm), dict(uav_policy=pol), "h",
                                 dict(exp="X3", n_c=nc, n_u=nu, uav_policy=pol)))
    return run_jobs(jobs, "X3_scalability", P)


def X4(ns=10):
    """E6a/E6b: communication routing, priority and fairness-constrained scheduling."""
    variants = [("shortest", "guarded"), ("etx", "guarded"), ("queue", "fifo"), ("queue", "sampled"),
                ("queue", "deadline"), ("queue", "deficit"), ("queue", "guarded")]
    jobs = []
    for nu in [9, 15, 21]:
        for deg in [0.0, 0.2]:
            for rt, sc in variants:
                for s in seeds(ns):
                    jobs.append((dict(seed=s, n_casualties=10, n_uavs=nu, n_medics=4, link_degradation=deg,
                                      severity_dynamics="changing"), dict(routing=rt, scheduling=sc), "h",
                                 dict(exp="X4", n_u=nu, degradation=deg, routing=rt, scheduling=sc, variant=f"{rt}+{sc}")))
    return run_jobs(jobs, "X4_comm_fairness", P)


def X5(ns=8):
    """Condition 6: severity-aware medic allocation (vitals-based severity) with safe routing."""
    jobs = []
    for nc, nu, nm in [(1, 3, 1), (5, 7, 1), (5, 7, 3), (5, 7, 5), (10, 15, 1), (10, 15, 2), (10, 15, 4),
                       (10, 15, 6), (10, 15, 8), (10, 15, 10)]:
        for alloc in ["nearest", "severity", "urgency", "load", "gain"]:
            for route in ["shortest", "safe"]:
                for s in seeds(ns):
                    jobs.append((dict(seed=s, n_casualties=nc, n_uavs=nu, n_medics=nm, hazards="mixed",
                                      severity_dynamics="changing"), dict(medic_alloc=alloc, medic_route=route), "h",
                                 dict(exp="X5", alloc=alloc, route=route, cfg=f"{nc}c/{nu}u/{nm}m", n_c=nc, n_m=nm)))
    return run_jobs(jobs, "X5_medic_allocation", P)


def X5b(ns=10):
    """Triage sweep: staging distance x severity placement x allocation x medic scarcity (pre-specified)."""
    jobs = []
    for ccp in [50.0, 450.0]:
        for lay in ["random", "critical_near", "critical_far", "severity_clusters", "scattered"]:
            for nc, nu, nm in [(5, 7, 1), (10, 15, 1), (10, 15, 2), (10, 15, 4)]:
                for alloc in ["nearest", "severity", "urgency", "load", "gain"]:
                    for s in seeds(ns):
                        jobs.append((dict(seed=s, n_casualties=nc, n_uavs=nu, n_medics=nm, hazards="mixed",
                                          severity_dynamics="changing", ccp_distance=ccp, casualty_layout=lay),
                                     dict(medic_alloc=alloc), "h",
                                     dict(exp="X5b", ccp=int(ccp), layout=lay, alloc=alloc, cfg=f"{nc}c/{nu}u/{nm}m")))
    return run_jobs(jobs, "X5b_triage_sweep", P)


def X5grid(ns=2):
    """All 82 original base configurations (Condition 6 table)."""
    from mantis.config import base_configurations
    stacks = {
        "independent": dict(uav_policy="connected", recovery="adaptive", scheduling="fifo", medic_alloc="urgency",
                            medic_route="shortest", replan="periodic", energy_policy="reactive"),
        "independent+safe": dict(uav_policy="connected", recovery="adaptive", scheduling="fifo", medic_alloc="urgency",
                                 medic_route="safe", replan="periodic", energy_policy="reactive"),
        "MANTIS-H": dict(),
    }
    jobs = []
    for b in base_configurations():
        for name, pk in stacks.items():
            for s in seeds(ns):
                jobs.append((dict(seed=s, hazards="mixed", severity_dynamics="changing", mobility="mixed", **b), pk, "h",
                             dict(exp="grid", stack=name, n_c=b["n_casualties"], n_u=b["n_uavs"], n_m=b["n_medics"])))
    return run_jobs(jobs, "X5grid_82configs", P)


def X6(ns=8):
    """Condition 5 + E7c: casualty mobility and dynamic medic replanning."""
    jobs = []
    for nc, nu, nm in [(5, 7, 3), (10, 15, 4)]:
        for mob in ["stationary", "slow", "intermittent", "abrupt", "mixed"]:
            for rp in ["static", "periodic", "event", "selective", "deadline"]:
                for s in seeds(ns):
                    jobs.append((dict(seed=s, n_casualties=nc, n_uavs=nu, n_medics=nm, mobility=mob,
                                      severity_dynamics="changing"), dict(replan=rp), "h",
                                 dict(exp="X6", mobility=mob, replan=rp, cfg=f"{nc}c/{nu}u/{nm}m")))
    return run_jobs(jobs, "X6_mobility_replanning", P)


def X7(ns=10):
    """E8a: energy-aware UAV operation (battery, safe return, role takeover, heterogeneous batteries)."""
    jobs = []
    for nc, nu, nm in [(5, 7, 3), (10, 15, 4)]:
        for env in ["clear", "obstacles+mobility"]:
            ek = dict(obstacles="static", mobility="mixed") if env != "clear" else {}
            for ep in ["fixed", "reactive", "aware"]:
                for s in seeds(ns):
                    jobs.append((dict(seed=s, n_casualties=nc, n_uavs=nu, n_medics=nm, energy=True, battery_hetero=True,
                                      n_spare_uavs=3, **ek), dict(energy_policy=ep), "h",
                                 dict(exp="X7", env=env, energy_policy=ep, cfg=f"{nc}c/{nu}u/{nm}m")))
    return run_jobs(jobs, "X7_energy", P)


if __name__ == "__main__":
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        globals()[name]()
