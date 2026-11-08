"""Training algorithms: behaviour cloning, IPPO, MAPPO, MANTIS-MAPPO (proposed), VDN, QMIX.

Every method uses the same observations, action space, reward, executors, training
scenario distribution and environment-episode budget.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

from ..config import ScenarioConfig, base_configurations, MANTIS_H
from ..sim import Simulator
from ..heuristics import Combined
from .controller import MARLController, Recorder, TorchPolicy, QPolicy, GAMMA
from .nets import ActorPair, TeamCritic, LocalCritic, QMixer

torch.set_num_threads(1)
BASES = base_configurations()


# ============================================================================ scenarios
def sample_train_cfg(rng, max_uavs=15):
    while True:
        b = BASES[int(rng.integers(len(BASES)))]
        if b["n_uavs"] <= max_uavs:
            break
    kw = dict(b)
    kw["obstacles"] = rng.choice(["clear", "static", "dynamic", "mixed"])
    kw["hazards"] = rng.choice(["static", "dynamic", "mixed"])
    kw["mobility"] = rng.choice(["stationary", "slow", "intermittent", "abrupt", "mixed"])
    kw["severity_dynamics"] = rng.choice(["fixed", "changing"])
    kw["disruption"] = rng.choice(["none", "intermittent", "complete"], p=[0.6, 0.25, 0.15])
    if rng.random() < 0.5:
        nf = int(rng.integers(1, 3))
        kw["failures"] = [dict(t=int(rng.integers(40, 160)), role=str(rng.choice(["observer", "relay", "gateway"])))
                          for _ in range(nf)]
    if rng.random() < 0.4:
        kw["energy"] = True
        kw["battery_hetero"] = bool(rng.random() < 0.5)
        kw["n_spare_uavs"] = int(rng.integers(0, 3))
    kw["seed"] = int(rng.integers(0, 10 ** 6))   # training seeds < 1e6; evaluation uses >= 1e7
    return ScenarioConfig(**kw)


def run_episode(cfg, controller):
    sim = Simulator(cfg, MANTIS_H, controller)
    m = sim.run()
    controller.finish(sim)
    return sim, m


# ============================================================================ parallel rollouts
_W = {}
_POOL = None


def set_workers(n):
    """Create a process pool for episode rollouts (n<=1: run in the main process)."""
    global _POOL
    if _POOL is not None:
        _POOL.close(); _POOL = None
    if n and n > 1:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(n)
    return _POOL


def _rollout(job):
    """Run one training episode in a worker. job = (kind, state, cfg_kw, flags, seed)."""
    kind, state, cfg_kw, flags, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rec = Recorder()
    if kind == "teacher":
        ctl = MARLController(recorder=rec, teacher=Combined(MANTIS_H))
    else:
        net = _W.get(kind)
        if net is None:
            net = _W[kind] = ActorPair()
        net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        if kind == "ppo":
            pol = TorchPolicy(net)
        else:
            pol = QPolicy(net, eps=flags.get("eps", 0.05), rng=np.random.default_rng(seed))
        ctl = MARLController(pol, learn_uav=flags.get("lu", True), learn_medic=flags.get("lm", True),
                             greedy=False, recorder=rec)
    _, m = run_episode(ScenarioConfig(**cfg_kw), ctl)
    return rec, float(m["mission_score"])


def rollouts(jobs):
    if _POOL is None:
        return [_rollout(j) for j in jobs]
    return _POOL.map(_rollout, jobs, chunksize=1)


def _state(net):
    # numpy (not torch tensors) so it pickles by value to worker processes
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


# ============================================================================ batching
def _stack(samples, kind):
    S = [s for s in samples if s["kind"] == kind]
    if not S:
        return None
    return dict(
        a=torch.from_numpy(np.stack([s["a"] for s in S])),
        C=torch.from_numpy(np.stack([s["C"] for s in S])),
        mask=torch.from_numpy(np.stack([s["mask"] for s in S])),
        g=torch.from_numpy(np.stack([s["g"] for s in S])),
        act=torch.tensor([s["act"] for s in S]),
        logp=torch.tensor([s["logp"] for s in S], dtype=torch.float32),
        adv=torch.tensor([s.get("adv", 0.0) for s in S], dtype=torch.float32),
        ret=torch.tensor([s.get("ret", 0.0) for s in S], dtype=torch.float32),
        idx=np.array([i for i, s in enumerate(samples) if s["kind"] == kind]),
    )


def _disc_cum(R):
    T = len(R)
    disc = GAMMA ** np.arange(T + 1)
    D = np.concatenate([[0.0], np.cumsum(disc[:T] * np.asarray(R))])
    return D, disc


def seg_return(D, disc, a, b):
    """sum_{t=a}^{b-1} gamma^(t-a) R[t]"""
    return (D[b] - D[a]) / disc[a]


# ============================================================================ behaviour cloning
def collect_teacher(n_episodes, seed=0, log=print):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    jobs = [("teacher", None, sample_train_cfg(rng).to_dict(), {}, seed * 100003 + i) for i in range(n_episodes)]
    eps = []
    B = 64
    for i in range(0, len(jobs), B):
        eps += [r for r, _ in rollouts(jobs[i:i + B])]
        log(f"  teacher episodes {len(eps)}/{n_episodes}  {time.time() - t0:.0f}s")
    return eps


def bc_train(actors, eps, epochs=25, lr=1e-3, critic=None, log=print):
    samples = [s for r in eps for s in r.samples]
    opt = torch.optim.Adam(actors.parameters(), lr=lr)
    batches = {k: _stack(samples, k) for k in (0, 1)}
    for ep in range(epochs):
        tot, acc = 0.0, []
        for k, B in batches.items():
            if B is None:
                continue
            n = len(B["act"])
            perm = torch.randperm(n)
            for i in range(0, n, 512):
                j = perm[i:i + 512]
                logits = actors.scorer(k)(B["a"][j], B["C"][j], B["g"][j], B["mask"][j])
                loss = F.cross_entropy(logits, B["act"][j])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(j)
                acc.append(float((logits.argmax(-1) == B["act"][j]).float().mean()))
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  BC epoch {ep}: loss {tot / len(samples):.3f} acc {np.mean(acc):.3f}")
    if critic is not None:   # warm-start the team critic on teacher Monte-Carlo returns
        G, Y = [], []
        for r in eps:
            D, disc = _disc_cum(r.rewards)
            T = len(r.rewards)
            for st in r.steps:
                G.append(st["g"]); Y.append(seg_return(D, disc, st["t"], T))
        G = torch.from_numpy(np.stack(G)); Y = torch.tensor(Y, dtype=torch.float32)
        copt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        for _ in range(300):
            j = torch.randint(0, len(Y), (512,))
            l = F.mse_loss(critic(G[j]), Y[j])
            copt.zero_grad(); l.backward(); copt.step()
        log(f"  critic warm-start mse {float(l):.3f}")
    return actors


# ============================================================================ PPO family
class PPOTrainer:
    """mode: 'mappo' (team critic), 'ippo' (local critics), 'mantis' (MAPPO + BC prior + KL anchor)."""

    def __init__(self, mode="mappo", actors=None, critic=None, lr=3e-4, ent=0.01, kl0=0.0,
                 learn_uav=True, learn_medic=True, seed=0):
        torch.manual_seed(seed)
        self.mode = mode
        self.actors = actors or ActorPair()
        self.critic = critic or (LocalCritic() if mode == "ippo" else TeamCritic())
        self.prior = copy.deepcopy(self.actors) if kl0 > 0 else None
        if self.prior is not None:
            for p in self.prior.parameters():
                p.requires_grad_(False)
        self.kl0 = kl0
        self.ent = ent
        self.opt = torch.optim.Adam(list(self.actors.parameters()) + list(self.critic.parameters()), lr=lr)
        self.learn_uav, self.learn_medic = learn_uav, learn_medic
        self.rng = np.random.default_rng(seed)
        self.lam = 0.95

    def controller(self, greedy=False, rec=None):
        return MARLController(TorchPolicy(self.actors), learn_uav=self.learn_uav,
                              learn_medic=self.learn_medic, greedy=greedy, recorder=rec)

    @torch.no_grad()
    def _advantages(self, rec):
        S = rec.samples
        if not S:
            return
        R = rec.rewards
        T = len(R)
        D, disc = _disc_cum(R)
        if self.mode != "ippo":
            ks = sorted({s["k"] for s in S})
            G = torch.from_numpy(np.stack([rec.steps[k]["g"] for k in ks]))
            V = self.critic(G).numpy()
            times = [rec.steps[k]["t"] for k in ks] + [T]
            A = np.zeros(len(ks)); last = 0.0
            for i in range(len(ks) - 1, -1, -1):
                a, b = times[i], times[i + 1]
                Gm = seg_return(D, disc, a, b)
                Gam = disc[b - a]
                vnext = V[i + 1] if i + 1 < len(ks) else 0.0
                delta = Gm + Gam * vnext - V[i]
                last = delta + Gam * self.lam * last
                A[i] = last
            pos = {k: i for i, k in enumerate(ks)}
            for s in S:
                i = pos[s["k"]]
                s["adv"] = A[i]; s["ret"] = A[i] + V[i]
        else:
            by = {}
            for i, s in enumerate(S):
                by.setdefault(s["aid"], []).append(i)
            for aid, idx in by.items():
                kind = aid[0]
                a = torch.from_numpy(np.stack([S[i]["a"] for i in idx]))
                C = torch.from_numpy(np.stack([S[i]["C"] for i in idx]))
                M = torch.from_numpy(np.stack([S[i]["mask"] for i in idx]))
                V = self.critic(kind, a, C, M).numpy()
                times = [rec.steps[S[i]["k"]]["t"] for i in idx] + [T]
                last = 0.0
                for j in range(len(idx) - 1, -1, -1):
                    ta, tb = times[j], times[j + 1]
                    Gm = seg_return(D, disc, ta, tb)
                    Gam = disc[tb - ta]
                    vnext = V[j + 1] if j + 1 < len(idx) else 0.0
                    delta = Gm + Gam * vnext - V[j]
                    last = delta + Gam * self.lam * last
                    S[idx[j]]["adv"] = last; S[idx[j]]["ret"] = last + V[j]

    def update(self, recs, epochs=4, mb=512, frac=0.0):
        for r in recs:
            self._advantages(r)
        samples = [s for r in recs for s in r.samples]
        if not samples:
            return {}
        adv = np.array([s["adv"] for s in samples])
        mu, sd = adv.mean(), adv.std() + 1e-8
        for s in samples:
            s["adv_n"] = (s["adv"] - mu) / sd
        batches = {k: _stack(samples, k) for k in (0, 1)}
        for k, B in batches.items():
            if B is not None:
                B["adv"] = torch.tensor([samples[i]["adv_n"] for i in B["idx"]], dtype=torch.float32)
        kl_coef = self.kl0 * max(0.05, 1 - frac) if self.prior is not None else 0.0
        stats = dict(pl=0.0, vl=0.0, ent=0.0, n=0)
        for _ in range(epochs):
            for k, B in batches.items():
                if B is None:
                    continue
                n = len(B["act"])
                perm = torch.randperm(n)
                for i in range(0, n, mb):
                    j = perm[i:i + mb]
                    logits = self.actors.scorer(k)(B["a"][j], B["C"][j], B["g"][j], B["mask"][j])
                    logp_all = torch.log_softmax(logits, -1)
                    lp = logp_all.gather(1, B["act"][j][:, None]).squeeze(1)
                    ratio = torch.exp(lp - B["logp"][j])
                    A = B["adv"][j]
                    pl = -torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
                    p = logp_all.exp()
                    ent = -(p * torch.where(B["mask"][j], logp_all, torch.zeros_like(logp_all))).sum(-1).mean()
                    if self.mode == "ippo":
                        v = self.critic(k, B["a"][j], B["C"][j], B["mask"][j])
                    else:
                        v = self.critic(B["g"][j])
                    vl = F.mse_loss(v, B["ret"][j])
                    loss = pl + 0.5 * vl - self.ent * ent
                    if kl_coef > 0:
                        with torch.no_grad():
                            pr = torch.log_softmax(self.prior.scorer(k)(B["a"][j], B["C"][j], B["g"][j], B["mask"][j]), -1)
                        kl = (pr.exp() * torch.where(B["mask"][j], pr - logp_all, torch.zeros_like(pr))).sum(-1).mean()
                        loss = loss + kl_coef * kl
                    self.opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.actors.parameters()) + list(self.critic.parameters()), 0.5)
                    self.opt.step()
                    stats["pl"] += float(pl.detach()); stats["vl"] += float(vl.detach()); stats["ent"] += float(ent.detach()); stats["n"] += 1
        return {k: v / max(stats["n"], 1) for k, v in stats.items() if k != "n"}

    def train(self, n_episodes, eps_per_update=8, log=print, max_uavs=15, ckpt=None, ckpt_every=100):
        t0 = time.time()
        hist = []
        done = 0
        next_ck = ckpt_every
        if ckpt:
            ckpt(self.actors, 0)
        flags = dict(lu=self.learn_uav, lm=self.learn_medic)
        while done < n_episodes:
            k = min(eps_per_update, n_episodes - done)
            st_dict = _state(self.actors)
            jobs = [("ppo", st_dict, sample_train_cfg(self.rng, max_uavs).to_dict(), flags,
                     int(self.rng.integers(2 ** 31))) for _ in range(k)]
            out = rollouts(jobs)
            recs = [r for r, _ in out]
            for r, sc in out:
                hist.append(dict(ep=done, ret=float(np.sum(r.rewards)), score=sc))
                done += 1
            st = self.update(recs, frac=done / n_episodes)
            if ckpt and done >= next_ck:
                ckpt(self.actors, done); next_ck += ckpt_every
            if (done // eps_per_update) % 5 == 0 or done >= n_episodes:
                rr = np.mean([h["ret"] for h in hist[-200:]])
                log(f"  [{self.mode}] ep {done}/{n_episodes} return {rr:.3f} score {np.mean([h['score'] for h in hist[-200:]]):.3f} "
                    f"pl {st.get('pl', 0):.3f} vl {st.get('vl', 0):.3f} ent {st.get('ent', 0):.3f} {time.time() - t0:.0f}s")
        if ckpt:
            ckpt(self.actors, done)
        return hist


# ============================================================================ VDN / QMIX
class ValueTrainer:
    def __init__(self, mode="vdn", lr=5e-4, seed=0, buffer=30000, batch=32):
        torch.manual_seed(seed)
        self.mode = mode
        self.q = ActorPair()
        self.qt = copy.deepcopy(self.q)
        self.mixer = QMixer() if mode == "qmix" else None
        self.mixer_t = copy.deepcopy(self.mixer) if self.mixer else None
        params = list(self.q.parameters()) + (list(self.mixer.parameters()) if self.mixer else [])
        self.opt = torch.optim.Adam(params, lr=lr)
        self.rng = np.random.default_rng(seed)
        self.buf = []
        self.buffer, self.batch = buffer, batch
        self.policy = QPolicy(self.q, eps=1.0, rng=self.rng)
        self.actors = self.q

    def controller(self, greedy=False, rec=None):
        return MARLController(self.policy, greedy=greedy, recorder=rec)

    def _store(self, rec):
        S = rec.samples
        T = len(rec.rewards)
        D, disc = _disc_cum(rec.rewards)
        ks = sorted({s["k"] for s in S})
        by = {k: [s for s in S if s["k"] == k] for k in ks}
        times = [rec.steps[k]["t"] for k in ks] + [T]
        for i, k in enumerate(ks):
            a, b = times[i], times[i + 1]
            nxt = by[ks[i + 1]] if i + 1 < len(ks) else None
            self.buf.append(dict(cur=by[k], g=rec.steps[k]["g"], G=seg_return(D, disc, a, b), gam=disc[b - a],
                                 nxt=nxt, g2=rec.steps[ks[i + 1]]["g"] if nxt else rec.steps[k]["g"]))
        if len(self.buf) > self.buffer:
            self.buf = self.buf[-self.buffer:]

    def _qtot(self, trans, which, net, mixer, online=None):
        """Sum/mix per-agent Q of agents in transitions. which: 'cur' (chosen acts) or 'nxt' (double-Q max)."""
        B = len(trans)
        tot = torch.zeros(B)
        for kind in (0, 1):
            items = [(b, s) for b, tr in enumerate(trans) if tr[which] for s in tr[which] if s["kind"] == kind]
            if not items:
                continue
            seg = torch.tensor([b for b, _ in items])
            a = torch.from_numpy(np.stack([s["a"] for _, s in items]))
            C = torch.from_numpy(np.stack([s["C"] for _, s in items]))
            M = torch.from_numpy(np.stack([s["mask"] for _, s in items]))
            g = torch.from_numpy(np.stack([s["g"] for _, s in items]))
            q = net.scorer(kind)(a, C, g, M)
            if which == "cur":
                act = torch.tensor([s["act"] for _, s in items])
                qa = q.gather(1, act[:, None]).squeeze(1)
            else:
                with torch.no_grad():
                    am = online.scorer(kind)(a, C, g, M).argmax(-1)
                qa = q.gather(1, am[:, None]).squeeze(1)
            if mixer is not None:
                gg = torch.from_numpy(np.stack([trans[b]["g" if which == "cur" else "g2"] for b, _ in items]))
                qa = qa * mixer.weights(kind, gg, a)
            tot = tot.index_add(0, seg, qa)
        if mixer is not None:
            gg = torch.from_numpy(np.stack([tr["g" if which == "cur" else "g2"] for tr in trans]))
            tot = tot + mixer.b(gg).squeeze(-1)
        return tot

    def learn(self, steps):
        if len(self.buf) < 200:
            return 0.0
        L = 0.0
        for _ in range(steps):
            idx = self.rng.integers(0, len(self.buf), self.batch)
            tr = [self.buf[i] for i in idx]
            q = self._qtot(tr, "cur", self.q, self.mixer)
            with torch.no_grad():
                has = torch.tensor([t["nxt"] is not None for t in tr], dtype=torch.float32)
                q2 = self._qtot(tr, "nxt", self.qt, self.mixer_t, online=self.q)
                y = torch.tensor([t["G"] for t in tr], dtype=torch.float32) + \
                    torch.tensor([t["gam"] for t in tr], dtype=torch.float32) * has * q2
            loss = F.smooth_l1_loss(q, y)
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            self.opt.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                    pt.mul_(0.99).add_(0.01 * p)
                if self.mixer:
                    for p, pt in zip(self.mixer.parameters(), self.mixer_t.parameters()):
                        pt.mul_(0.99).add_(0.01 * p)
            L += float(loss)
        return L / steps

    def train(self, n_episodes, log=print, max_uavs=15, grad_steps=24, ckpt=None, ckpt_every=100, batch_eps=8):
        """Collect `batch_eps` episodes in parallel with the current Q-net, then do
        grad_steps updates per collected episode (same update-to-data ratio as serial)."""
        t0 = time.time()
        hist = []
        e = 0
        next_ck = ckpt_every
        while e < n_episodes:
            k = min(batch_eps, n_episodes - e)
            st_dict = _state(self.q)
            jobs = []
            for j in range(k):
                eps = max(0.05, 1.0 - (e + j) / (0.6 * n_episodes))
                jobs.append(("q", st_dict, sample_train_cfg(self.rng, max_uavs).to_dict(), dict(eps=eps),
                             int(self.rng.integers(2 ** 31))))
            out = rollouts(jobs)
            for r, sc in out:
                self._store(r)
                hist.append(dict(ep=e, ret=float(np.sum(r.rewards)), score=sc))
                e += 1
            self.policy.eps = max(0.05, 1.0 - e / (0.6 * n_episodes))
            l = self.learn(grad_steps * k)
            if ckpt and e >= next_ck:
                ckpt(self.q, e); next_ck += ckpt_every
            if (e // batch_eps) % 5 == 0 or e >= n_episodes:
                log(f"  [{self.mode}] ep {e}/{n_episodes} return {np.mean([h['ret'] for h in hist[-200:]]):.3f} "
                    f"score {np.mean([h['score'] for h in hist[-200:]]):.3f} loss {l:.4f} eps {self.policy.eps:.2f} {time.time() - t0:.0f}s")
        if ckpt:
            ckpt(self.q, n_episodes)
        return hist
