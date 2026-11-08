"""Controller that lets learned policies (or a recorded heuristic teacher) drive the simulator
through the same high-level action interface: medics pick a casualty or HOLD, UAVs pick
observe(casualty) / NETWORK role / RETURN. Low-level execution (safe ground routing,
relay placement, flight) is shared with the heuristics."""
from __future__ import annotations

import numpy as np
import torch

from ..config import MANTIS_H
from ..heuristics import UAVHeuristic, MedicHeuristic
from ..sim import NET, RETURN, NONE, AIR, READY, CHARGING, M_IDLE, M_MOVE, M_TREAT, M_INCAP, DEAD
from .. import vitals as V
from .features import FeatureBuilder, HOLD, U_NET, U_RET, NC_M, NC_U, medic_advice

GAMMA = 0.99


class RewardTracker:
    """Team reward, computed from simulator counters (ground truth; training signal only)."""

    def __init__(self, sim):
        self.sim = sim
        self.w = V.TRIAGE_WEIGHT[sim.ccat].copy()
        self.norm = max(1.0, self.w.sum() / 5)
        self.prev = self._state()

    def _state(self):
        s = self.sim
        w = V.TRIAGE_WEIGHT[s.ccat]
        return np.array([
            (w * s.cserved).sum(), (w * ~s.calive).sum(), (w * s.slot_ok).sum(),
            (s.mstate == M_INCAP).sum(), s.munsafe_steps.sum(), s.mexpo.sum(),
            s.ustranded.sum(), s.collisions,
        ], float)

    COEF = np.array([1.0, -2.0, 0.01, -3.0, -0.05, -0.002, -0.5, -0.5])

    def pop(self):
        cur = self._state()
        r = float((cur - self.prev) @ self.COEF) / self.norm
        self.prev = cur
        return r


class Recorder:
    """Collects decision samples and per-step rewards of one episode."""

    def __init__(self):
        self.samples = []     # dict(kind, a, C, mask, g, act, logp, k)
        self.steps = []       # decision steps: dict(t, g)
        self.rewards = []     # per sim step

    def new_step(self, t, g):
        self.steps.append(dict(t=t, g=g))
        return len(self.steps) - 1


class MARLController:
    def __init__(self, policy=None, pcfg=MANTIS_H, learn_uav=True, learn_medic=True,
                 greedy=True, recorder=None, teacher=None, epoch=6):
        self.policy = policy
        self.pcfg = pcfg
        self.learn_uav = learn_uav
        self.learn_medic = learn_medic
        self.greedy = greedy
        self.rec = recorder
        self.teacher = teacher          # Combined heuristic -> imitation labels
        self.epoch = epoch

    # ------------------------------------------------------------------ lifecycle
    def reset(self, sim):
        self.fb = FeatureBuilder(sim)
        self.rt = RewardTracker(sim)
        self.forced = set()
        self.handover = {}
        self.uh = UAVHeuristic(self.pcfg) if (not self.learn_uav and self.teacher is None) else None
        self.mh = MedicHeuristic(self.pcfg) if (not self.learn_medic and self.teacher is None) else None
        sim.net.use_detection = True
        self.adv = UAVHeuristic(self.pcfg)
        self._adv_init = True
        if self.teacher is not None:
            self._teach(sim, reset=True)
            return
        if self.uh:
            self.uh.reset(sim)
        if self.mh:
            self.mh.reset(sim)
        self._decide(sim, epoch=True)

    def _shadow_uav(self, sim, initial=False):
        """Run the heuristic UAV planner in shadow mode and return its recommended tasks."""
        snap = (sim.utask.copy(), sim.ustate.copy(), sim.usortie_start.copy(), sim.usearch.copy(),
                sim.role_switches, sim.net.n_gw)
        if initial:
            self.adv.groups, self.adv.ptr, self.adv.handover = {}, {}, {}
        self.adv.plan(sim, initial=initial)
        advice = {u: int(sim.utask[u]) for u in range(sim.M)}
        self.adv_gw = sim.net.n_gw
        (sim.utask[:], sim.ustate[:], sim.usortie_start[:], sim.usearch[:], sim.role_switches, sim.net.n_gw) = snap
        return advice

    def finish(self, sim):
        if self.rec is not None:
            self.rec.rewards.append(self.rt.pop())

    def step(self, sim):
        if sim.t > 0 and self.rec is not None:
            self.rec.rewards.append(self.rt.pop())
        if sim.t == 0:
            return                       # the initial decision was taken in reset()
        if self.teacher is not None:
            self._teach(sim)
            return
        self._shield(sim)
        if self.uh:
            self.uh.step(sim)
        if self.mh:
            self.mh.step(sim)
        epoch = sim.t % self.epoch == 0
        idle = [m for m in range(sim.K) if sim.mstate[m] == M_IDLE and sim.mtarget[m] < 0]
        trig = any(e[0] in self.TRIGGERS for e in sim.events)
        if epoch or ((idle or trig) and self.learn_medic):
            self._decide(sim, epoch, trig)

    # ------------------------------------------------------------------ safety shield
    def _shield(self, sim):
        """Battery safety shield shared by every learner (same rule as the energy-aware heuristic):
        predictive return with margin + pre-emptive handover to a charged UAV from base."""
        if not sim.cfg.energy:
            return
        for u in range(sim.M):
            if not (sim.ualive[u] and sim.ustate[u] == AIR) or sim.utask[u] == RETURN:
                continue
            need = sim.uav_need_return(u)
            thr = 1.3 * need + 0.05
            lead = need + 0.10
            task = int(sim.utask[u])
            # launch a replacement before the battery reaches the return threshold
            if sim.ubat[u] <= thr + lead and u not in self.handover.values() and task != NONE:
                ready = [r for r in range(sim.M) if sim.ualive[r] and sim.ustate[r] == READY
                         and r not in self.handover]
                if ready:
                    r = max(ready, key=lambda r: sim.ubat[r])
                    self.handover[r] = u
                    sim.set_uav_task(r, task)
            relieved = [r for r, v in self.handover.items() if v == u]
            arrived = bool(relieved) and np.linalg.norm(sim.upos[relieved[0]] - sim.upos[u]) < 60
            if sim.ubat[u] <= thr or arrived:
                for r in relieved:
                    del self.handover[r]
                sim.set_uav_task(u, RETURN)
                self.forced.add(u)
        # drop handovers whose relieved UAV is gone
        self.handover = {r: v for r, v in self.handover.items()
                         if sim.ualive[r] and sim.ualive[v] and sim.utask[v] != RETURN}
        self.forced = {u for u in self.forced if sim.utask[u] == RETURN and sim.ustate[u] == AIR}

    # ------------------------------------------------------------------ agents
    TRIGGERS = {"new_info", "triage_change", "dead", "notfound", "medic_down", "served"}

    def medic_agents(self, sim, epoch):
        if epoch:
            return [m for m in range(sim.K) if sim.mstate[m] not in (M_TREAT, M_INCAP)]
        return [m for m in range(sim.K) if sim.mstate[m] == M_IDLE and sim.mtarget[m] < 0]

    def uav_agents(self, sim):
        al = sim.believed_alive()
        return [u for u in range(sim.M) if al[u] and sim.ualive[u] and sim.ustate[u] in (AIR, READY)
                and u not in self.forced and u not in getattr(self, "handover", {})]

    # ------------------------------------------------------------------ learned decisions
    def _act(self, kind, a, C, mask, g):
        return self.policy.act(kind, a, C, mask, g, self.greedy)

    def _decide(self, sim, epoch, trig=False):
        fb = self.fb
        fb.refresh()
        g = fb.global_state()
        k = self.rec.new_step(sim.t, g) if self.rec is not None else None
        if self.learn_medic:
            chosen = {}
            magents = self.medic_agents(sim, epoch or trig)
            madv = medic_advice(sim, fb, magents)
            for m in magents:
                a, C, mask = fb.medic_obs(m, chosen, madv)
                act, logp = self._act(0, a, C, mask, g)
                c = act if act < sim.n else -1
                chosen[m] = c
                if self.rec is not None:
                    self.rec.samples.append(dict(kind=0, a=a, C=C, mask=mask, g=g, act=act, logp=logp, k=k, aid=(0, m)))
            for m, c in chosen.items():
                if c != sim.mtarget[m] or (c >= 0 and sim.mstate[m] == M_IDLE):
                    sim.medic_assign(m, c)
        if self.learn_uav and epoch:
            uadv = self._shadow_uav(sim, initial=self._adv_init)
            self._adv_init = False
            agents = self.uav_agents(sim)
            obs_pts = [sim.utarget[u] for u in range(sim.M) if sim.ualive[u] and sim.utask[u] >= 0]
            fb.set_need_relays(obs_pts)
            obs_count, net_count = {}, 0
            for u in range(sim.M):
                if sim.ualive[u] and u not in agents and sim.utask[u] >= 0:
                    obs_count[sim.utask[u]] = obs_count.get(sim.utask[u], 0) + 1
            decisions = {}
            for u in agents:
                a, C, mask = fb.uav_obs(u, obs_count, net_count, uadv)
                act, logp = self._act(1, a, C, mask, g)
                if act == U_NET:
                    task = NET; net_count += 1
                elif act == U_RET:
                    task = RETURN
                else:
                    task = int(act); obs_count[task] = obs_count.get(task, 0) + 1
                decisions[u] = task
                if self.rec is not None:
                    self.rec.samples.append(dict(kind=1, a=a, C=C, mask=mask, g=g, act=act, logp=logp, k=k, aid=(1, u)))
            for u, task in decisions.items():
                if task == RETURN and sim.ustate[u] == READY:
                    continue                 # stay at base
                sim.set_uav_task(u, task)
            sim.net.n_gw = self.adv_gw

    # ------------------------------------------------------------------ teacher (imitation data)
    def _teach(self, sim, reset=False):
        epoch = sim.t % self.epoch == 0
        idle = [m for m in range(sim.K) if sim.mstate[m] == M_IDLE and sim.mtarget[m] < 0]
        trig = any(e[0] in self.TRIGGERS for e in sim.events)
        decide = reset or epoch or bool(idle) or trig
        if decide:
            pre = (sim.mtarget.copy(), sim.utask.copy(), sim.mstate.copy())
            magents = self.medic_agents(sim, epoch or reset or trig)
            uagents = self.uav_agents(sim) if (epoch or reset) else []
            self.fb.refresh()
        if reset:
            self.teacher.reset(sim)
        else:
            self.teacher.step(sim)
        if not decide or self.rec is None:
            return
        post = (sim.mtarget.copy(), sim.utask.copy(), sim.mstate.copy())
        sim.mtarget[:], sim.utask[:], sim.mstate[:] = pre
        g = self.fb.global_state()
        k = self.rec.new_step(sim.t, g)
        chosen = {}
        madv = medic_advice(sim, self.fb, magents)
        for m in magents:
            a, C, mask = self.fb.medic_obs(m, chosen, madv)
            c = int(post[0][m])
            act = c if c >= 0 else HOLD
            if not mask[act]:
                act = HOLD
            chosen[m] = c
            self.rec.samples.append(dict(kind=0, a=a, C=C, mask=mask, g=g, act=act, logp=0.0, k=k, aid=(0, m)))
        if uagents:
            obs_pts = [sim.utarget[u] for u in range(sim.M) if sim.ualive[u] and sim.utask[u] >= 0]
            self.fb.set_need_relays(obs_pts)
            obs_count, net_count = {}, 0
            for u in range(sim.M):
                if sim.ualive[u] and u not in uagents and pre[1][u] >= 0:
                    obs_count[pre[1][u]] = obs_count.get(pre[1][u], 0) + 1
            uadv = {u: int(post[1][u]) for u in range(sim.M)}
            for u in uagents:
                a, C, mask = self.fb.uav_obs(u, obs_count, net_count, uadv)
                t = int(post[1][u])
                act = U_NET if t == NET else U_RET if t in (RETURN, NONE) else t
                if t >= 0:
                    obs_count[t] = obs_count.get(t, 0) + 1
                elif t == NET:
                    net_count += 1
                if not mask[act]:
                    act = U_NET
                self.rec.samples.append(dict(kind=1, a=a, C=C, mask=mask, g=g, act=act, logp=0.0, k=k, aid=(1, u)))
        sim.mtarget[:], sim.utask[:], sim.mstate[:] = post


class TorchPolicy:
    """Wraps an ActorPair (stochastic or greedy)."""

    def __init__(self, actors):
        self.actors = actors

    @torch.no_grad()
    def act(self, kind, a, C, mask, g, greedy):
        sc = self.actors.scorer(kind)
        logits = sc(torch.from_numpy(a)[None], torch.from_numpy(C)[None], torch.from_numpy(g)[None],
                    torch.from_numpy(mask)[None])[0]
        if greedy:
            act = int(torch.argmax(logits))
            return act, 0.0
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        return int(act), float(dist.log_prob(act))


class QPolicy:
    """Epsilon-greedy over a Q scorer (VDN / QMIX)."""

    def __init__(self, qnets, eps=0.0, rng=None):
        self.q = qnets
        self.eps = eps
        self.rng = rng or np.random.default_rng(0)

    @torch.no_grad()
    def act(self, kind, a, C, mask, g, greedy):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(np.nonzero(mask)[0])), 0.0
        q = self.q.scorer(kind)(torch.from_numpy(a)[None], torch.from_numpy(C)[None],
                                torch.from_numpy(g)[None], torch.from_numpy(mask)[None])[0]
        return int(torch.argmax(q)), 0.0
