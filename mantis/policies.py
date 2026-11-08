"""Conventional coordination baselines (UAV roles, medic allocation, replanning) and the runner."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import comm
from .config import PolicySpec, ScenarioConfig
from .env import (ACTIVE, BASE, GW, IDLE, M_ENROUTE, M_IDLE, M_INJURED, M_TREAT, OBS, RELAY, RTB,
                  SENSE_R, UAV_SPEED, Env)
from .geometry import points_in_rects, push_out_of_rects, seg_rect_intersect
from .vitals import INFO_DEADLINE, SERVICE_TIME, SEV_WEIGHT, T1, T3, estimate_time_to_death

HOP = 0.8 * comm.R50_AA


# =========================================================================== UAV formation
class Slot:
    __slots__ = ("role", "pos", "patrol", "cas", "prio", "cluster", "budgeted")

    def __init__(self, role, pos, cas=(), prio=0.0, cluster=-1, patrol=None, budgeted=True):
        self.role, self.pos, self.cas, self.prio = role, np.asarray(pos, float), list(cas), prio
        self.cluster, self.patrol, self.budgeted = cluster, patrol, budgeted


def uav_casualty_view(env, track, snapshot):
    """Casualties the UAV team should still serve, with positions and priorities."""
    n = env.cfg.n_casualties
    keep = ~env.c_treated & ~env.ub_dead & ~env.b_dead
    pos = env.ub_pos if track else snapshot
    aoi = env.medic_aoi()
    tri = env.ub_tri
    stale = np.minimum(aoi / INFO_DEADLINE[tri], 3.0)
    prio = SEV_WEIGHT[tri] * (1.0 + 0.5 * stale)
    return np.where(keep)[0], pos, prio


def cluster_casualties(idx, pos, prio, radius=0.8 * SENSE_R):
    order = sorted(idx, key=lambda c: -prio[c])
    used, clusters = set(), []
    for c in order:
        if c in used:
            continue
        mem = [c] + [d for d in order if d not in used and d != c and np.linalg.norm(pos[d] - pos[c]) < radius]
        for d in mem:
            used.add(d)
        clusters.append((mem, pos[mem].mean(0), float(sum(prio[d] for d in mem))))
    return clusters


def medic_anchors(env):
    act = [m for m in range(env.M) if env.m_state[m] != M_INJURED]
    if not act:
        return [BASE.copy()]
    pts = env.m_pos[act]
    anchors = []
    for p in pts:
        if all(np.linalg.norm(p - a) > 150 for a in anchors):
            anchors.append(p.copy())
    return anchors


def los_adjust(env, p, prev, nxt):
    """Pick a nearby relay position that maximises the weaker of its two links (obstacle-aware)."""
    best, bv = p, -1
    for dx, dy in [(0, 0), (40, 0), (-40, 0), (0, 40), (0, -40), (30, 30), (-30, 30), (30, -30), (-30, -30)]:
        q = p + (dx, dy)
        if len(env.rects) and points_in_rects(q[None], env.rects, 15).any():
            continue
        A = np.array([q, q])
        B = np.array([prev, nxt])
        pr = comm.link_prob(A, B, comm.R50_AA, env.rects, env.dyn_c, env.dyn_r, 1.0)
        v = min(pr[0, 0], pr[1, 1])
        if v > bv + 1e-6:
            best, bv = q, v
    return best


def chain(env, a, b, los):
    """Relay positions between a (observer) and b (tree node)."""
    d = np.linalg.norm(b - a)
    hops = max(0, math.ceil(d / HOP) - 1)
    pts = [a + (b - a) * (k + 1) / (hops + 1) for k in range(hops)]
    if los and len(env.rects) and pts:
        out, prev = [], a
        for i, p in enumerate(pts):
            nxt = pts[i + 1] if i + 1 < len(pts) else b
            q = los_adjust(env, p, prev, nxt)
            out.append(q)
            prev = q
        pts = out
    return pts


def make_slots(env, mode, los, track, snapshot, budget):
    idx, pos, prio = uav_casualty_view(env, track, snapshot)
    clusters = cluster_casualties(idx, pos, prio) if len(idx) else []
    anchors = medic_anchors(env)
    if clusters:
        cen = np.mean([c[1] for c in clusters], 0)
        primary = min(anchors, key=lambda a: np.linalg.norm(a - cen))
    else:
        primary = anchors[0]
    slots = []
    if mode == "coverage":
        for k, (mem, p, pr) in enumerate(clusters[:budget]):
            slots.append(Slot(OBS, p, mem, pr, k))
        return slots, clusters
    # connected / balanced: gateway + observer chains
    slots.append(Slot(GW, primary, (), 0.0, -1))
    tree = [primary]
    remaining = budget - 1
    pending = list(range(len(clusters)))
    covered = []
    while pending and remaining > 0:
        best, bscore, bpts = None, -1, None
        for k in pending:
            mem, p, pr = clusters[k]
            node = min(tree, key=lambda q: np.linalg.norm(q - p))
            pts = chain(env, p, node, los)
            cost = 1 + len(pts)
            if cost > remaining:
                continue
            score = pr / cost if mode == "balanced" else pr  # connected: priority order only
            if score > bscore:
                best, bscore, bpts = k, score, pts
        if best is None:
            break
        mem, p, pr = clusters[best]
        slots.append(Slot(OBS, p, mem, pr, best))
        for q in bpts:
            slots.append(Slot(RELAY, q, (), pr, best))
            tree.append(q)
        tree.append(p)
        remaining -= 1 + len(bpts)
        pending.remove(best)
        covered.append(best)
    if pending and remaining > 0:
        if mode == "balanced":
            # time-share one observer between uncovered clusters (patrol), nearest first
            pend = sorted(pending, key=lambda k: -clusters[k][2])
            while pend and remaining > 0:
                grp = pend[:3]
                pend = pend[3:]
                pts = [clusters[k][1] for k in grp]
                s = Slot(OBS, pts[0], sum((clusters[k][0] for k in grp), []), sum(clusters[k][2] for k in grp) / len(grp), grp[0], patrol=pts)
                slots.append(s)
                remaining -= 1
        else:
            for k in pending[:remaining]:
                slots.append(Slot(OBS, clusters[k][1], clusters[k][0], clusters[k][2], k))
            remaining -= min(remaining, len(pending))
    if remaining > 0 and mode == "balanced":
        # spare capacity: redundant relays on highest-priority chains + gateways at other medic groups
        for a in anchors:
            if remaining <= 0:
                break
            if np.linalg.norm(a - primary) > 150:
                slots.append(Slot(GW, a, (), 0.0, -1))
                remaining -= 1
        rel = sorted([s for s in slots if s.role == RELAY], key=lambda s: -s.prio)
        for s in rel:
            if remaining <= 0:
                break
            slots.append(Slot(RELAY, s.pos + np.array([25.0, -25.0]), (), s.prio * 0.5, s.cluster, budgeted=True))
            remaining -= 1
    return slots, clusters


def assign_slots(env, slots, uavs, prev_targets):
    if not len(uavs) or not slots:
        return {}
    C = np.zeros((len(uavs), len(slots)))
    for i, u in enumerate(uavs):
        for j, s in enumerate(slots):
            d = np.linalg.norm(env.u_pos[u] - s.pos)
            C[i, j] = d / UAV_SPEED - 3.0 * s.prio
            if np.linalg.norm(prev_targets[u] - s.pos) < 50 and env.u_role[u] == s.role:
                C[i, j] -= 15.0  # stability: keep current task
            if np.isfinite(env.u_cap[u]):
                need = (d + np.linalg.norm(s.pos - BASE)) / UAV_SPEED * 1.6 + 60
                if env.u_bat[u] < need:
                    C[i, j] += 1e4
    r, c = linear_sum_assignment(C)
    return {int(uavs[i]): slots[j] for i, j in zip(r, c) if C[i, j] < 5e3}


class UAVFormationPolicy:
    def __init__(self, spec: PolicySpec):
        self.spec = spec
        self.mode = {"fixed": "connected"}.get(spec.uav, spec.uav)
        self.adaptive = spec.adaptive and spec.uav != "fixed"
        self.done = False
        self.snapshot = None

    def act(self, env: Env):
        if self.snapshot is None:
            self.snapshot = env.ub_pos.copy()
        if self.done and not self.adaptive:
            return
        self.done = True
        uavs = env.available_uavs()
        budget = len(uavs)
        slots, _ = make_slots(env, self.mode, self.spec.los_opt, self.spec.track and self.adaptive, self.snapshot, budget)
        asg = assign_slots(env, slots, uavs, env.u_target)
        for u in uavs:
            s = asg.get(int(u))
            if s is None:
                if env.u_role[u] != IDLE or not self.adaptive:
                    env.set_uav_task(u, IDLE, slots[0].pos + env.rng_misc.uniform(-30, 30, 2) if slots else BASE)
                continue
            env.set_uav_task(u, s.role, s.pos, s.patrol, s.cas)


# =========================================================================== medic allocation
def plan_mode(route):
    return "shortest" if route == "shortest" else "risk_aware"


def medic_candidates(env):
    n = env.cfg.n_casualties
    taken = set(int(env.m_target[m]) for m in range(env.M) if env.m_state[m] == M_TREAT)
    ok = env.b_known & ~env.c_treated & ~env.b_dead
    return [c for c in range(n) if ok[c] and c not in taken]


def medic_utility(env, kind, m, c, ro, zone=None):
    eta, expo, mr, cells, safe = ro
    tri = int(env.believed_tri()[c])
    news = env.believed_news()[c]
    news = 5.0 if np.isnan(news) else news
    w = SEV_WEIGHT[tri]
    if kind == "nearest":
        return -eta
    if kind == "severity_first":
        return w * 1000.0 - eta
    urg = w * (1.0 + max(0.0, news - 4.0) / 4.0)
    if kind in ("urgency_distance", "load_aware"):
        u = urg / (eta + SERVICE_TIME[tri] + 30.0)
        if kind == "load_aware" and zone is not None and zone[c] != m % max(1, zone.max() + 1):
            u *= 0.35
        return u
    if kind == "safe_severity":
        prev = env.b_prev_news[c]
        ttd = estimate_time_to_death(news, None if np.isnan(prev) else prev,
                                     env.b_gen[c] - env.b_prev_gen[c] if np.isfinite(env.b_prev_gen[c]) else None)
        aoi = min(env.medic_aoi()[c], 600.0)
        ttd = max(float(ttd) - aoi, 10.0)
        p_reach = 1.0 / (1.0 + math.exp(-(ttd - eta) / 60.0))
        gain = {0: 1.0, 1: 0.8, 2: 0.35}[tri]
        return w * gain * p_reach / (eta + SERVICE_TIME[tri] + 30.0) * (1.0 - 0.02 * expo)
    raise ValueError(kind)


class MedicPolicy:
    TRIGGERS = {"new_info", "hazard_changed", "target_lost", "path_blocked", "medic_injured"}

    def __init__(self, spec: PolicySpec):
        self.spec = spec
        self.kind = spec.medic_alloc
        self.replan = spec.replan
        self.last_periodic = -999
        self.last_deadline_check = -999
        self.zone = None
        self.deferred_logged = {}

    def _zones(self, env):
        from scipy.cluster.vq import kmeans2
        n, M = env.cfg.n_casualties, env.M
        if M <= 1 or n <= 1:
            return np.zeros(n, int)
        k = min(M, n)
        _, lab = kmeans2(env.b_pos, k, seed=env.cfg.seed, minit="++")
        return lab

    def act(self, env: Env):
        sp = self.spec
        if self.kind == "load_aware" and self.zone is None:
            self.zone = self._zones(env)
        active = [m for m in range(env.M) if env.m_state[m] not in (M_INJURED, M_TREAT)]
        if not active:
            return
        idle = [m for m in active if env.m_state[m] == M_IDLE]
        t = env.t
        trig = False
        threshold = 0.0
        if self.replan == "static":
            movers = idle
        elif self.replan == "periodic":
            if t - self.last_periodic >= 30:
                self.last_periodic = t
                movers = active
            else:
                movers = idle
        else:
            trig = bool(env.events & self.TRIGGERS)
            if self.replan == "deadline_selective" and t - self.last_deadline_check >= 10:
                self.last_deadline_check = t
                trig = trig or self._deadline_risk(env)
            movers = active if trig else idle
            threshold = 0.25 if self.replan in ("selective", "deadline_selective") else 0.0
        # path refresh for moved targets (all but static)
        if self.replan != "static":
            for m in active:
                c = env.m_target[m]
                if c >= 0 and f"info:{c}" in env.events and m not in movers:
                    env.assign_medic(m, c, mode=plan_mode(sp.route), reason="target location updated")
        if not movers:
            return
        if not trig and movers == idle and t % 5 and self.replan != "periodic":
            return  # idle medics re-check every 5 s unless an event occurred
        self._allocate(env, movers, threshold)

    def _deadline_risk(self, env):
        assigned = set(int(env.m_target[m]) for m in range(env.M) if env.m_target[m] >= 0)
        tri = env.believed_tri()
        for c in medic_candidates(env):
            if tri[c] == T1 and c not in assigned:
                return True
        return False

    def _allocate(self, env, movers, threshold):
        sp = self.spec
        cands = medic_candidates(env)
        fixed_targets = {int(env.m_target[m]) for m in range(env.M)
                         if env.m_target[m] >= 0 and m not in movers}
        cands = [c for c in cands if c not in fixed_targets]
        if not cands:
            for m in movers:
                if env.m_state[m] == M_ENROUTE and env.m_target[m] not in medic_candidates(env):
                    env.assign_medic(m, -1)
            return
        routes = {m: env.medic_route_options(m, plan_mode(sp.route), cands) for m in movers}
        safety = sp.route == "safe"
        U = {}
        for m in movers:
            for c, ro in routes[m].items():
                if safety and not ro[4]:
                    continue
                U[(m, c)] = medic_utility(env, self.kind, m, c, ro, self.zone)
        # deferrals: casualties with no safe route from any medic (system decision; medic safety first)
        if safety:
            for c in cands:
                if not any((m, c) in U for m in movers):
                    if self.deferred_logged.get(c, -999) < env.t - 60:
                        self.deferred_logged[c] = env.t
                        reach = [routes[m][c] for m in movers if c in routes[m]]
                        why = "no route within medic risk limits" if not reach else \
                            f"safest route exceeds exposure budget ({min(r[1] for r in reach):.1f} risk-s)"
                        env.log_deferral(c, why)
        # greedy joint assignment on utility (one medic per casualty)
        cur_u = {}
        for m in movers:
            c = env.m_target[m]
            if c >= 0 and (m, c) in U:
                cur_u[m] = U[(m, c)]
        done_m, done_c = set(), set()
        pairs = sorted(U.items(), key=lambda kv: -kv[1])
        plan = {}
        for (m, c), u in pairs:
            if m in done_m or c in done_c:
                continue
            plan[m] = (c, u)
            done_m.add(m); done_c.add(c)
        for m in movers:
            if m in plan:
                c, u = plan[m]
                cur = env.m_target[m]
                if cur >= 0 and cur != c and m in cur_u and u < cur_u[m] * (1 + threshold):
                    if (m, cur) in U:  # keep current (stability)
                        env.assign_medic(m, cur, routes[m][cur], reason="keep (stability)", mode=plan_mode(sp.route))
                    continue
                tri = int(env.believed_tri()[c])
                env.assign_medic(m, c, routes[m][c], mode=plan_mode(sp.route),
                                 reason=f"{self.kind}: highest priority reachable ({['T1','T2','T3'][tri]})")
            elif env.m_state[m] == M_ENROUTE and env.m_target[m] >= 0 and (m, env.m_target[m]) not in U:
                env.assign_medic(m, -1, reason="target no longer safe/valid")


# =========================================================================== runner
def make_env(cfg: ScenarioConfig, spec: PolicySpec) -> Env:
    return Env(cfg, avoid=spec.avoid, energy_mode=spec.energy, scheduler=spec.scheduler,
               medic_safe_hold=spec.route == "safe")


def run_episode(cfg: ScenarioConfig, spec: PolicySpec, learned=None, return_env=False):
    env = make_env(cfg, spec)
    if learned is not None:
        uav_pol = medic_pol = learned
        learned.reset(env)
    else:
        uav_pol = UAVFormationPolicy(spec)
        medic_pol = MedicPolicy(spec)
    K = cfg.decision_interval
    done = False
    while not done:
        if env.t % K == 0:
            uav_pol.act(env) if learned is None else learned.act_uav(env)
        if learned is None:
            medic_pol.act(env)
        else:
            learned.act_medic(env)
        done = env.step()
    m = env.metrics()
    return (m, env) if return_env else m
