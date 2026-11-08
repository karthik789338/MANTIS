"""Train MARL methods. Every method gets the same episode budget, features, reward and executors.

Examples
  python -m mantis.marl.train --method bc            --teacher-episodes 1500 --workers 32
  python -m mantis.marl.train --method mantis_mappo  --episodes 12000 --workers 32
  python -m mantis.marl.train --method all           --episodes 500   --workers 2

Methods: bc, mantis_mappo, mappo, ippo, vdn, qmix, abl_nokl, abl_mediconly, abl_uavonly
(every method except bc/mappo/ippo/vdn/qmix needs bc.pt first; run --method bc once).
"""
import os, sys, json, copy, time, argparse
import numpy as np
import torch

from .algos import collect_teacher, bc_train, PPOTrainer, ValueTrainer, set_workers
from .nets import ActorPair, TeamCritic

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("MANTIS_MODELS", os.path.join(ROOT, "results", "models"))
CK = os.path.join(OUT, "ckpt")
os.makedirs(CK, exist_ok=True)
LOG = None

ALL = ["bc", "mantis_mappo", "mappo", "ippo", "vdn", "qmix", "abl_nokl", "abl_mediconly", "abl_uavonly"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def ckpt_fn(name, learn_uav=True, learn_medic=True):
    def f(actors, ep):
        torch.save(dict(actors=actors.state_dict(), learn_uav=learn_uav, learn_medic=learn_medic, ep=ep),
                   os.path.join(CK, f"{name}@{ep}.pt"))
    return f


def save(name, actors, hist, learn_uav=True, learn_medic=True):
    torch.save(dict(actors=actors.state_dict(), learn_uav=learn_uav, learn_medic=learn_medic),
               os.path.join(OUT, f"{name}_last.pt"))
    json.dump(hist, open(os.path.join(OUT, f"{name}_last_curve.json"), "w"))
    log(f"saved {name}_last (+ checkpoints in {CK})")


def bc_models(args, build=False):
    p = os.path.join(OUT, "bc.pt")
    if os.path.exists(p) and not build:
        ck = torch.load(p, weights_only=False)
        A = ActorPair(); A.load_state_dict(ck["actors"]); C = TeamCritic(); C.load_state_dict(ck["critic"])
        return A, C
    if not build:
        sys.exit(f"{p} not found: run `--method bc` first")
    t0 = time.time()
    eps = collect_teacher(args.teacher_episodes, seed=1, log=log)
    log(f"teacher data: {sum(len(r.samples) for r in eps)} decisions, {time.time() - t0:.0f}s")
    torch.manual_seed(0)
    A, C = ActorPair(), TeamCritic()
    bc_train(A, eps, epochs=args.bc_epochs, critic=C, log=log)
    torch.save(dict(actors=A.state_dict(), critic=C.state_dict(), learn_uav=True, learn_medic=True), p)
    log("saved bc.pt (behaviour cloning only)")
    return A, C


def run(method, args):
    N, E, ce = args.episodes, args.eps_per_update, args.ckpt_every
    kw = dict(log=log, ckpt_every=ce)
    if method == "bc":
        bc_models(args, build=True)
    elif method == "mantis_mappo":
        A, C = bc_models(args)
        log("== MANTIS-MAPPO (proposed): BC warm start + team critic + KL anchor")
        tr = PPOTrainer("mantis", actors=copy.deepcopy(A), critic=copy.deepcopy(C), kl0=0.5, ent=0.003, lr=1e-4, seed=1)
        save(method, tr.actors, tr.train(N, eps_per_update=E, ckpt=ckpt_fn(method), **kw))
    elif method == "mappo":
        log("== MAPPO from scratch"); tr = PPOTrainer("mappo", seed=2)
        save(method, tr.actors, tr.train(N, eps_per_update=E, ckpt=ckpt_fn(method), **kw))
    elif method == "ippo":
        log("== IPPO from scratch"); tr = PPOTrainer("ippo", seed=3)
        save(method, tr.actors, tr.train(N, eps_per_update=E, ckpt=ckpt_fn(method), **kw))
    elif method in ("vdn", "qmix"):
        log(f"== {method.upper()}"); vt = ValueTrainer(method, seed=4 if method == "vdn" else 5)
        save(method, vt.q, vt.train(N, ckpt=ckpt_fn(method), batch_eps=E, **kw))
    elif method == "abl_nokl":
        A, C = bc_models(args)
        log("== ablation: BC + MAPPO without KL anchor")
        tr = PPOTrainer("mantis", actors=copy.deepcopy(A), critic=copy.deepcopy(C), kl0=0.0, ent=0.003, lr=1e-4, seed=6)
        save(method, tr.actors, tr.train(N, eps_per_update=E, ckpt=ckpt_fn(method), **kw))
    elif method == "abl_mediconly":
        A, C = bc_models(args)
        log("== ablation: medic-only learning (heuristic UAVs)")
        tr = PPOTrainer("mantis", actors=copy.deepcopy(A), critic=copy.deepcopy(C), kl0=0.5, ent=0.003, lr=1e-4,
                        seed=7, learn_uav=False)
        save(method, tr.actors, tr.train(N, eps_per_update=E, ckpt=ckpt_fn(method, learn_uav=False), **kw), learn_uav=False)
    elif method == "abl_uavonly":
        A, C = bc_models(args)
        log("== ablation: UAV-only learning (heuristic medics)")
        tr = PPOTrainer("mantis", actors=copy.deepcopy(A), critic=copy.deepcopy(C), kl0=0.5, ent=0.003, lr=1e-4,
                        seed=8, learn_medic=False)
        save(method, tr.actors, tr.train(N, eps_per_update=E, ckpt=ckpt_fn(method, learn_medic=False), **kw), learn_medic=False)
    else:
        sys.exit(f"unknown method {method}")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="one of %s, or 'all'" % ALL)
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    ap.add_argument("--eps-per-update", type=int, default=None, help="episodes per PPO update (default = 2 x workers, >= 8)")
    ap.add_argument("--ckpt-every", type=int, default=None, help="default = episodes / 12")
    ap.add_argument("--teacher-episodes", type=int, default=1500)
    ap.add_argument("--bc-epochs", type=int, default=30)
    args = ap.parse_args()
    args.eps_per_update = args.eps_per_update or max(8, 2 * args.workers)
    args.ckpt_every = args.ckpt_every or max(50, args.episodes // 12)
    LOG = open(os.path.join(OUT, f"train_log_{args.method}.txt"), "a")
    torch.set_num_threads(min(4, max(1, args.workers // 8)))   # threads for the learner's updates
    log(f"=== {args.method}: episodes={args.episodes} workers={args.workers} "
        f"eps/update={args.eps_per_update} ckpt_every={args.ckpt_every}")
    set_workers(args.workers)
    methods = ALL if args.method == "all" else [args.method]
    for m in methods:
        t0 = time.time()
        run(m, args)
        log(f"=== {m} finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
