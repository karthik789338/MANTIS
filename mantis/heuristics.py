"""Heuristic controllers: UAV role/observation allocation, failure recovery, energy
management, medic allocation and replanning. Every component is selected by PolicyConfig.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PolicyConfig
from . import vitals as V
from .planning import allocation_scores, assign, survival_prob, candidate_mask, triage_weight, treat_time
from .sim import NET, RETURN, NONE, AIR, READY, CHARGING, M_TREAT, M_INCAP, M_MOVE, M_IDLE


def kmeans(P, k, rng, iters=12):
    P = np.asarray(P)
    k = min(k, len(P))
    C = P[rng.choice(len(P), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((P[:, None] - C[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                C[j] = P[lab == j].mean(0)
    lab = np.argmin(((P[:, None] - C[None]) ** 2).sum(-1), 1)
    return lab, C


# ===================================================================================== UAVs
class UAVHeuristic:
    def __init__(self, pcfg: PolicyConfig):
        self.p = pcfg
        self.rng = np.random.default_rng(12345)

    def reset(self, sim):
        self.groups = {}      # observer uav -> list of casualties it cycles through
        self.ptr = {}
        self.handover = {}    # replacement uav -> uav being relieved
        sim.net.use_detection = self.p.recovery != "none"
        sim.net.freeze = self.p.recovery == "none" or self.p.uav_policy == "fixed"
        self._pending_detect = False
        self.plan(sim, initial=True)

    # ------------------------------------------------------------------ helpers
    def view_alive(self, sim):
        return sim.believed_alive() if self.p.recovery != "none" else np.ones(sim.M, bool)

    def active_casualties(self, sim):
        return [c for c in range(sim.n) if not sim.kb_dead[c] and not sim.cserved[c] and sim.ctreating[c] < 0]

    def available(self, sim):
        al = self.view_alive(sim)
        ok = al & (sim.ustate != CHARGING) & (sim.utask != RETURN)
        return [u for u in range(sim.M) if ok[u]]

    # ------------------------------------------------------------------ planning
    def plan(self, sim, initial=False):
        p = self.p
        if p.uav_policy == "fixed" and not initial:
            self._cycle(sim)
            return
        U = self.available(sim)
        launched = [u for u in U if sim.ustate[u] != READY]
        ready = [u for u in U if sim.ustate[u] == READY]
        use_ready = (sim.cfg.energy and p.energy_policy != "fixed") or p.recovery == "coordinated"
        U = launched + (ready[: self._spare_need(sim, launched)] if use_ready else [])
        C = self.active_casualties(sim)
        if not U:
            return
        if not C:
            for u in U:
                sim.set_uav_task(u, NET)
            return
        locs = sim.ub_loc[C]
        nU = len(U)
        med = sim.net.medic_points()
        kind = p.uav_policy
        n_gw = 1
        if kind == "fixed":
            n_obs = max(1, min(len(C), nU - 2)) if nU >= 3 else max(1, nU - 1)
        elif kind == "coverage":
            n_obs = max(1, min(len(C), nU - 1))
        else:
            n_obs = 1
            for k in range(min(len(C), nU - 1), 0, -1):
                lab, cen = self._clusters(sim, C, k)
                pts = [self._obs_point(sim, [C[i] for i in range(len(C)) if lab[i] == j], cen[j]) for j in range(len(cen))]
                need = sim.net.required_relays(pts, med, 1)
                if k + need <= nU:
                    n_obs = k
                    break
            if kind == "balanced" and len(med) > 1:
                lab, cen = self._clusters(sim, C, n_obs)
                pts = [cen[j] for j in range(len(cen))]
                need = sim.net.required_relays(pts, med, 1)
                if nU - n_obs - need >= 2:
                    n_gw = 2
        lab, cen = self._clusters(sim, C, n_obs)
        groups = [[C[i] for i in range(len(C)) if lab[i] == j] for j in range(len(cen))]
        groups = [g for g in groups if g]
        if kind == "balanced":
            # dedicated observers for the most severe casualties if UAVs allow
            extra = nU - len(groups) - sim.net.required_relays([cen[j] for j in range(len(groups))], med, n_gw)
            groups.sort(key=lambda g: -max(triage_weight(sim)[g]))
            while extra > 0:
                big = [g for g in groups if len(g) > 1]
                if not big:
                    break
                g = big[0]
                w = triage_weight(sim)[g]
                c = g[int(np.argmax(w))]
                g.remove(c); groups.append([c]); extra -= 1
        sim.net.n_gw = n_gw
        tasks_pts = [sim.ub_loc[g].mean(0) for g in groups]
        n_net = nU - len(groups)
        # matching UAVs to observation tasks (distance + energy + stickiness)
        cost = np.zeros((nU, len(groups) + n_net))
        base = sim.world.base
        for i, u in enumerate(U):
            for j, g in enumerate(groups):
                d = np.linalg.norm(sim.upos[u] - tasks_pts[j])
                cost[i, j] = d
                if kind == "balanced" and sim.cfg.energy:
                    far = np.linalg.norm(tasks_pts[j] - base) / sim.world.L
                    cost[i, j] += 400 * far * (1 - sim.ubat[u] / sim.ucap[u])
                if sim.utask[u] in g and kind in ("balanced",) and not initial:
                    cost[i, j] -= 150
            for j in range(len(groups), len(groups) + n_net):
                cost[i, j] = np.linalg.norm(sim.upos[u] - med[0]) * 0.5
                if sim.utask[u] == NET and kind == "balanced" and not initial:
                    cost[i, j] -= 150
        r, c = linear_sum_assignment(cost)
        self.groups = {}
        for i, j in zip(r, c):
            u = U[i]
            if j < len(groups):
                self.groups[u] = groups[j]
                self.ptr.setdefault(u, 0)
            else:
                sim.set_uav_task(u, NET)
        self._cycle(sim)

    def _spare_need(self, sim, launched):
        alive_air = sum(1 for u in launched if sim.ualive[u] or self.p.recovery == "none")
        return max(0, sim.cfg.n_uavs - alive_air)

    def _clusters(self, sim, C, k):
        P = sim.ub_loc[C]
        if k >= len(C):
            return np.arange(len(C)), P.copy()
        return kmeans(P, k, np.random.default_rng(sim.cfg.seed + 17))

    def _obs_point(self, sim, g, cen):
        return cen

    def _cycle(self, sim):
        """Each observer selects which casualty of its group to observe this epoch."""
        al = self.view_alive(sim)
        for u, g in list(self.groups.items()):
            if not al[u]:
                continue
            act = [c for c in g if not sim.kb_dead[c] and not sim.cserved[c] and sim.ctreating[c] < 0]
            if not act:
                act = g
            if self.p.uav_policy == "balanced":
                # joint UAV-medic coordination: favour casualties a medic is en route to
                # (their location must stay fresh) and stale, severe casualties
                aoi = sim.t - np.maximum(sim.kb_last_gen[act], 0)
                enroute = np.isin(act, sim.mtarget[sim.mtarget >= 0])
                lost = (sim.kb_stale[act] > sim.kb_loc_t[act])      # medic could not find it
                c = act[int(np.argmax(triage_weight(sim)[act] * (aoi + 1) * (1 + 0.5 * enroute + 1.0 * lost)))]
            else:
                self.ptr[u] = (self.ptr.get(u, 0) + 1) % len(act)
                c = act[self.ptr[u]]
            sim.set_uav_task(u, int(c))

    # ------------------------------------------------------------------ per step
    def step(self, sim):
        p = self.p
        detected = [e[1] for e in sim.events if e[0] == "uav_fail_detected"]
        if p.recovery == "coordinated" and detected:
            self._local_repair(sim, detected)
        if sim.cfg.energy:
            self._energy(sim)
        if sim.t > 0 and sim.t % p.epoch == 0:
            self.plan(sim)

    def _local_repair(self, sim, failed):
        """Coordinated recovery: immediately hand the failed role to the cheapest donor."""
        w = triage_weight(sim)
        for f in failed:
            task = int(sim.utask[f])
            grp = self.groups.pop(f, None) or ([task] if task >= 0 else None)
            if task in (NONE, RETURN):
                continue
            al = sim.believed_alive()
            ready = [u for u in range(sim.M) if al[u] and sim.ustate[u] == READY]
            obs = [u for u in range(sim.M) if al[u] and sim.utask[u] >= 0 and sim.ustate[u] == AIR]
            net = [u for u in range(sim.M) if al[u] and sim.utask[u] == NET and sim.ustate[u] == AIR]
            med = sim.net.medic_points()
            if ready:                                   # launch a spare into the vacant role
                d = ready[0]
                if task >= 0:
                    self.groups[d] = grp
                sim.set_uav_task(d, task)
            elif task == NET:
                need = sim.net.required_relays([sim.utarget[u] for u in obs], med, sim.net.n_gw)
                if len(net) < need and len(obs) > 1:
                    # sacrifice the observer with the least severe group; merge its group
                    d = min(obs, key=lambda u: max(w[self.groups.get(u, [sim.utask[u]])]))
                    g = self.groups.pop(d, [int(sim.utask[d])])
                    rest = [u for u in obs if u != d]
                    tgt = min(rest, key=lambda u: np.linalg.norm(sim.utarget[u] - sim.utarget[d]))
                    self.groups.setdefault(tgt, [int(sim.utask[tgt])]).extend(g)
                    sim.set_uav_task(d, NET)
            else:
                pts = [sim.utarget[u] for u in obs] + [sim.utarget[f]]
                need = sim.net.required_relays(pts, med, sim.net.n_gw)
                if net and len(net) - 1 >= need - 1 and len(net) > 1:
                    d = min(net, key=lambda u: np.linalg.norm(sim.upos[u] - sim.utarget[f]))
                    self.groups[d] = grp
                    sim.set_uav_task(d, task)
                elif obs:
                    tgt = min(obs, key=lambda u: np.linalg.norm(sim.utarget[u] - sim.utarget[f]))
                    self.groups.setdefault(tgt, [int(sim.utask[tgt])]).extend(grp)
            sim.net.update(force=True)

    def _energy(self, sim):
        p = self.p
        if p.energy_policy == "fixed":
            return
        for u in range(sim.M):
            if not sim.ualive[u] or sim.ustate[u] != AIR or sim.utask[u] == RETURN:
                continue
            need = sim.uav_need_return(u)
            if p.energy_policy == "reactive":
                if sim.ubat[u] < 0.25 * sim.ucap[u]:
                    self._release(sim, u)
                continue
            # energy-aware: predictive return with margin, pre-emptive replacement
            thr = 1.3 * need + 0.05
            lead = need + 0.10
            if sim.ubat[u] <= thr + lead and u not in self.handover.values():
                ready = [r for r in range(sim.M) if sim.ualive[r] and sim.ustate[r] == READY]
                if ready:
                    r = max(ready, key=lambda r: sim.ubat[r])
                    self.handover[r] = u
                    task = sim.utask[u]
                    if task >= 0 and u in self.groups:
                        self.groups[r] = self.groups[u]
                    sim.set_uav_task(r, int(task))
            relieved = [r for r, v in self.handover.items() if v == u]
            arrived = relieved and np.linalg.norm(sim.upos[relieved[0]] - sim.upos[u]) < 60
            if sim.ubat[u] <= thr or arrived:
                if relieved:
                    del self.handover[relieved[0]]
                self._release(sim, u)

    def _release(self, sim, u):
        self.groups.pop(u, None)
        sim.set_uav_task(u, RETURN)


# =================================================================================== Medics
class MedicHeuristic:
    EVENTS = {"new_info", "triage_change", "loc_change", "dead", "served", "idle", "notfound",
              "hazard_route", "medic_down"}

    def __init__(self, pcfg: PolicyConfig):
        self.p = pcfg

    def reset(self, sim):
        self.last_alloc = -999
        self._allocate(sim, self._free(sim), switch=True)

    def _free(self, sim):
        return [m for m in range(sim.K) if sim.mstate[m] not in (M_TREAT, M_INCAP)]

    def _idle(self, sim):
        return [m for m in range(sim.K) if sim.mstate[m] == M_IDLE and sim.mtarget[m] < 0]

    def scores(self, sim):
        return allocation_scores(sim, self.p.medic_alloc)

    def _allocate(self, sim, medics, switch, hysteresis=None, deadline=False):
        if not medics:
            return
        if self.p.medic_alloc == "load":
            out = self._load_aware(sim, medics)
        else:
            S = self.scores(sim)
            # do not double-assign casualties already targeted by medics outside this set
            others = [sim.mtarget[m] for m in range(sim.K) if m not in medics and sim.mtarget[m] >= 0]
            S = S.copy()
            if others:
                S[:, others] = -np.inf
            out = assign(S, medics)
        for m, c in out.items():
            cur = sim.mtarget[m]
            if cur >= 0 and c != cur and not switch:
                continue
            if cur >= 0 and c != cur and hysteresis is not None:
                S = self.scores(sim)
                s_new = S[m, c] if c >= 0 else -np.inf
                s_cur = S[m, cur]
                ok = (s_new - s_cur > (hysteresis - 1) * abs(s_cur)) if np.isfinite(s_cur) else True
                if deadline and not ok and np.isfinite(s_cur):
                    eta, _, _ = sim.medic_eta()
                    ps = survival_prob(sim, eta)
                    if ps[m, cur] < 0.10 and c >= 0 and ps[m, c] > 0.5:
                        ok = True   # current target became expectant: re-triage
                if not ok:
                    continue
            if c < 0 and cur >= 0 and np.isfinite(self.scores(sim)[m, cur]):
                continue
            sim.medic_assign(m, int(c))

    def _load_aware(self, sim, medics):
        """Plan a full service sequence per medic (insertion heuristic), dispatch heads."""
        eta, _, _ = sim.medic_eta()
        cand = [c for c in np.nonzero(candidate_mask(sim))[0]]
        others = {sim.mtarget[m] for m in range(sim.K) if m not in medics and sim.mtarget[m] >= 0}
        cand = [c for c in cand if c not in others]
        w = triage_weight(sim)
        tau = treat_time(sim)
        load = {m: 0.0 for m in medics}
        last = {m: None for m in medics}
        head = {m: -1 for m in medics}
        for c in sorted(cand, key=lambda c: -w[c]):
            best, bm = np.inf, None
            for m in medics:
                if last[m] is None:
                    tt = eta[m, c]
                else:
                    tt = np.linalg.norm(sim.kb_loc[last[m]] - sim.kb_loc[c]) * 1.3 / sim.cfg.medic_speed
                if not np.isfinite(eta[m, c]):
                    continue
                fin = load[m] + tt + tau[c]
                if fin < best:
                    best, bm = fin, m
            if bm is None:
                continue
            load[bm] = best; last[bm] = c
            if head[bm] < 0:
                head[bm] = c
        return head

    def step(self, sim):
        p = self.p
        ev = {e[0] for e in sim.events}
        idle = self._idle(sim)
        free = self._free(sim)
        if p.replan == "static":
            self._allocate(sim, idle, switch=True)
            return
        if p.replan == "periodic":
            if sim.t % p.epoch == 0:
                self._allocate(sim, free, switch=True)
                for m in free:
                    if sim.mstate[m] == M_MOVE:
                        sim.medic_reroute(m)
            elif idle:
                self._allocate(sim, idle, switch=True)
            return
        trig = ev & self.EVENTS
        loc_targets = {e[1] for e in sim.events if e[0] == "loc_change"}
        for m in free:
            if sim.mtarget[m] in loc_targets and sim.mstate[m] == M_MOVE:
                sim.medic_reroute(m)
        if not trig and not idle:
            return
        if p.replan == "event":
            self._allocate(sim, free, switch=True)
        elif p.replan == "selective":
            self._allocate(sim, idle, switch=True)
            self._allocate(sim, [m for m in free if m not in idle], switch=True, hysteresis=1.25)
        else:  # deadline-aware selective
            self._allocate(sim, idle, switch=True)
            self._allocate(sim, [m for m in free if m not in idle], switch=True, hysteresis=1.25, deadline=True)


class Combined:
    """UAV + medic heuristic stack."""

    def __init__(self, pcfg: PolicyConfig):
        self.pcfg = pcfg
        self.uav = UAVHeuristic(pcfg)
        self.med = MedicHeuristic(pcfg)

    def reset(self, sim):
        self.uav.reset(sim)
        self.med.reset(sim)

    def step(self, sim):
        self.uav.step(sim)
        self.med.step(sim)
