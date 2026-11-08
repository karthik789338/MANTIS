"""MANTIS simulator: UAV observation/relay/gateway network + ground medics + casualties.

Time step: 1 s. Decisions (UAV roles, medic allocation) are made by external policies;
the environment owns physics, sensing, wireless delivery, energy, failures, hazards and
metrics. Medics only know what UAV packets delivered (``info_mode='delivered'``).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import comm
from .config import ScenarioConfig
from .geometry import VisibilityPlanner, points_in_rects, push_out_of_rects, seg_rect_intersect
from .ground import Grid, KAPPA, R_MAX, EXPOSURE_BUDGET
from .vitals import (INFO_DEADLINE, SERVICE_TIME, SEV_WEIGHT, TREAT_DEADLINE, T1, T2, T3,
                     estimate_time_to_death, triage_from_vitals, vitals_from_health)

UAV_SPEED = 12.0
MEDIC_SPEED = 4.0
SENSE_R = 50.0
BASE = np.array([60.0, 60.0])
SWAP_TIME = 30
OBS, RELAY, GW, RTB, IDLE = 0, 1, 2, 3, 4
ROLE_NAMES = ["observer", "relay", "gateway", "return", "idle"]
ACTIVE, RETURNING, CHARGING, FAILED = 0, 1, 2, 3
M_IDLE, M_ENROUTE, M_TREAT, M_INJURED = 0, 1, 2, 3


class Env:
    def __init__(self, cfg: ScenarioConfig, avoid=True, energy_mode="aware", scheduler="guarded",
                 medic_safe_hold=True):
        self.cfg = cfg
        self.medic_safe_hold = medic_safe_hold
        self.avoid = avoid
        self.energy_mode = energy_mode
        self.scheduler = scheduler
        self._generate()

    # ------------------------------------------------------------------ scenario generation
    def _generate(self):
        cfg = self.cfg
        rs = np.random.default_rng(10_000 + cfg.seed)  # scenario stream (matched across policies)
        self.rng_link = np.random.default_rng(20_000 + cfg.seed)
        self.rng_sense = np.random.default_rng(30_000 + cfg.seed)
        self.rng_mob = np.random.default_rng(40_000 + cfg.seed)
        self.rng_harm = np.random.default_rng(50_000 + cfg.seed)
        self.rng_misc = np.random.default_rng(60_000 + cfg.seed)
        L = cfg.map_size
        self.L = L
        # static obstacles (buildings / restricted airspace): no-fly, block ground, NLOS
        rects = []
        if cfg.obstacles in ("static", "mixed"):
            for _ in range(200):
                if len(rects) >= 7:
                    break
                w, h = rs.uniform(40, 110, 2)
                x, y = rs.uniform(140, L - 60 - w), rs.uniform(140, L - 60 - h)
                r = np.array([x, y, x + w, y + h])
                if all(r[0] > o[2] + 30 or r[2] < o[0] - 30 or r[1] > o[3] + 30 or r[3] < o[1] - 30 for o in rects):
                    rects.append(r)
        self.rects = np.array(rects).reshape(-1, 4)
        # dynamic obstacles (plumes, other aircraft): airspace discs that move
        nd = 4 if cfg.obstacles in ("dynamic", "mixed") else 0
        self.dyn_c = rs.uniform(150, L - 50, (nd, 2))
        self.dyn_r = rs.uniform(25, 45, nd)
        ang = rs.uniform(0, 2 * np.pi, nd)
        self.dyn_v = np.stack([np.cos(ang), np.sin(ang)], -1) * rs.uniform(3, 7, (nd, 1))
        # casualties
        n = cfg.n_casualties
        pos = []
        centers = rs.uniform(220, L - 60, (max(1, n // 3), 2))
        while len(pos) < n:
            if rs.random() < 0.5 and n > 1:
                p = centers[rs.integers(len(centers))] + rs.normal(0, 35, 2)
            else:
                p = rs.uniform(200, L - 40, 2)
            p = np.clip(p, 30, L - 30)
            if not points_in_rects(p[None], self.rects, 8).any():
                pos.append(p)
        self.c_pos = np.array(pos)
        if n == 1:
            tri0 = np.array([T1])
        else:
            tri0 = rs.choice([T1, T2, T3], size=n, p=[0.3, 0.4, 0.3])
        h_lo = np.array([0.30, 0.50, 0.75]); h_hi = np.array([0.42, 0.62, 0.95])
        self.h0 = rs.uniform(h_lo[tri0], h_hi[tri0])
        ttd_lo = np.array([240, 700, 4000.]); ttd_hi = np.array([420, 1200, 8000.])
        self.ttd0 = rs.uniform(ttd_lo[tri0], ttd_hi[tri0])  # untreated time-to-death
        self.h = self.h0.copy()
        self.tri0 = tri0
        pats = ["stationary", "slow", "intermittent", "abrupt"]
        if cfg.mobility == "mixed":
            self.c_pat = [pats[i] for i in rs.integers(0, 4, n)]
        else:
            self.c_pat = [cfg.mobility] * n
        self.c_heading = rs.uniform(0, 2 * np.pi, n)
        self.c_alive = np.ones(n, bool)
        self.c_treated = np.zeros(n, bool)
        self.c_treat_time = np.full(n, np.nan)
        self.c_death_time = np.full(n, np.nan)
        self.c_shock_t = rs.uniform(0, 1, n) < 0.25  # 25 % suffer an abrupt deterioration
        self.c_shock_time = rs.uniform(100, 450, n)
        # hazard zones (ground risk for medics)
        hz = []
        if cfg.hazards != "none":
            nh = 3 if n > 1 else 2
            for k in range(nh):
                if k == 0 and n >= 1:
                    # one hazard sits between staging and a casualty (route hazard)
                    c = self.c_pos[rs.integers(n)]
                    f = rs.uniform(0.35, 0.65)
                    xy = BASE + f * (c - BASE) + rs.normal(0, 20, 2)
                    hz.append([xy[0], xy[1], rs.uniform(45, 75), rs.uniform(0.55, 1.0)])
                    continue
                elif k == 1 and n >= 3:
                    a = rs.uniform(0, 2 * np.pi)  # hazard adjacent to one casualty
                    xy = self.c_pos[rs.integers(n)] + rs.uniform(25, 50) * np.array([np.cos(a), np.sin(a)])
                    hz.append([xy[0], xy[1], rs.uniform(30, 45), rs.uniform(0.6, 1.0)])
                    continue
                else:
                    xy = rs.uniform(180, L - 80, 2)
                hz.append([xy[0], xy[1], rs.uniform(40, 70), rs.uniform(0.5, 0.9)])
        self.hazards = np.array(hz).reshape(-1, 4)
        hv = rs.normal(0, 0.6, (len(hz), 2)) if cfg.hazards == "dynamic" else np.zeros((len(hz), 2))
        self.haz_v = hv
        self.grid = Grid(L, cfg.cell, self.rects)
        self.vis = VisibilityPlanner(self.rects, 18.0, L)
        self.grid.set_risk(self.hazards)
        # UAVs
        N = cfg.n_uavs
        self.N = N
        self.u_pos = BASE + rs.uniform(-25, 25, (N, 2))
        if cfg.energy == "limited":
            self.u_cap = rs.uniform(*cfg.battery_range, N)
        else:
            self.u_cap = np.full(N, np.inf)
        self.u_bat = self.u_cap.copy()
        self.u_state = np.full(N, ACTIVE)
        self.u_role = np.full(N, IDLE)
        self.u_target = self.u_pos.copy()
        self.u_patrol = [None] * N
        self.u_wp = [None] * N  # cached (goal, waypoint list)
        self.u_cas = [[] for _ in range(N)]
        self.u_charge_t = np.zeros(N)
        self.u_handoff = np.zeros(N, bool)
        self.u_queue = [[] for _ in range(N)]
        self.u_deficit = [dict() for _ in range(N)]
        self.u_dist = np.zeros(N)
        self.u_fail_time = np.full(N, np.nan)
        self.u_detect_time = np.full(N, np.nan)
        self.u_fwd = np.zeros(N)
        # medics
        M = cfg.n_medics
        self.M = M
        self.m_pos = BASE + np.array([[10.0 * (i % 4), 10.0 * (i // 4)] for i in range(M)]) + 15
        self.m_state = np.full(M, M_IDLE)
        self.m_target = np.full(M, -1)
        self.m_path = [None] * M
        self.m_pidx = np.zeros(M, int)
        self.m_treat_left = np.zeros(M)
        self.m_dist = np.zeros(M)
        self.m_exposure = np.zeros(M)
        self.m_maxrisk = np.zeros(M)
        self.m_injury_time = np.full(M, np.nan)
        # medic-side belief (from delivered packets)
        self.b_known = np.zeros(n, bool)
        self.b_pos = np.zeros((n, 2))
        self.b_news = np.full(n, np.nan)
        self.b_tri = np.full(n, T2)
        self.b_gen = np.full(n, -np.inf)
        self.b_prev_news = np.full(n, np.nan)
        self.b_prev_gen = np.full(n, -np.inf)
        self.b_dead = np.zeros(n, bool)
        self.b_notfound = np.zeros(n, bool)
        # initial dispatch report (noisy location, unknown vitals) shared by medics and UAV team
        rep = self.c_pos + rs.normal(0, 30, (n, 2))
        self.b_pos[:] = rep
        self.b_known[:] = True
        # UAV-team belief (intra-swarm shared): latest observation of each casualty
        self.ub_pos = rep.copy()
        self.ub_tri = np.full(n, T2)
        self.ub_news = np.full(n, np.nan)
        self.ub_seen = np.full(n, -np.inf)
        self.ub_dead = np.zeros(n, bool)
        # failures
        self.failures = []
        ft = cfg.failure_time
        fm = cfg.failure_mode
        if fm in ("observer", "relay", "gateway"):
            self.failures = [(ft, fm)]
        elif fm == "multiple":
            self.failures = [(ft, "observer"), (ft, "relay")]
        elif fm == "successive":
            self.failures = [(ft, "gateway"), (ft + 100, "relay"), (ft + 200, "observer")]
        elif fm == "random":
            k = 1 + int(rs.random() < 0.5)
            self.failures = [(int(rs.uniform(100, 400)), "any") for _ in range(k)]
        # disruption windows (factor on link success)
        self.disrupt = np.ones(cfg.horizon + 1)
        if cfg.comm_disruption == "intermittent":
            t0 = int(rs.uniform(40, 90))
            while t0 < cfg.horizon:
                self.disrupt[t0:t0 + 15] = 0.05
                t0 += int(rs.uniform(70, 120))
        elif cfg.comm_disruption == "complete":
            self.disrupt[200:260] = 0.0
        self.t = 0
        self.events = set()
        self.instructions = []
        self.next_hop = np.full(N, -1)
        self.p_ag = np.zeros((N, M))
        self.p_aa = np.zeros((N, N))
        self._init_metrics()
        self._update_links()

    def _init_metrics(self):
        n = self.cfg.n_casualties
        self.mt = defaultdict(float)
        self.fresh_steps = np.zeros(n)
        self.need_steps = np.zeros(n)
        self.fresh_steps_t1 = 0.0
        self.need_steps_t1 = 0.0
        self.covered_steps = 0.0
        self.aoi_sum = 0.0
        self.aoi_n = 0.0
        self.fresh_series = []
        self.latencies = []
        self.gen_count = 0
        self.deliv_count = 0
        self.timely_pkt = 0
        self.outage_steps = 0
        self.role_switches = 0
        self.collisions = 0
        self.collision_log = []
        self.stranded = 0
        self.rtb_started = 0
        self.rtb_completed = 0
        self.energy_used = 0.0
        self.assign_total = 0
        self.assign_feasible = 0
        self.deferrals = 0
        self.reassignments = 0
        self.fail_events = []  # (time, uav)

    # ------------------------------------------------------------------ helpers
    @property
    def alive_uav(self):
        return self.u_state != FAILED

    def available_uavs(self):
        """UAVs the coordination layer believes usable (failed-but-undetected count as usable)."""
        detected = ~np.isnan(self.u_detect_time) & (self.u_detect_time <= self.t)
        undetected_fail = (self.u_state == FAILED) & ~detected
        return np.where(((self.u_state == ACTIVE) | undetected_fail) & ~self.u_handoff)[0]

    def medic_aoi(self):
        return self.t - self.b_gen

    def believed_tri(self):
        if self.cfg.severity_source == "oracle":
            return self.true_tri()
        return self.b_tri

    def believed_news(self):
        if self.cfg.severity_source == "oracle":
            return self.true_news()
        return self.b_news

    def believed_pos(self):
        if self.cfg.info_mode == "oracle":
            return self.c_pos
        return self.b_pos

    def true_vitals(self):
        hh = self.h0 if self.cfg.severity_dynamics == "fixed" else self.h
        return vitals_from_health(hh)

    def true_tri(self):
        return triage_from_vitals(self.true_vitals())[0]

    def true_news(self):
        return triage_from_vitals(self.true_vitals())[1]

    def energy_to_base(self, u):
        d = np.linalg.norm(self.u_pos[u] - BASE)
        return (d / UAV_SPEED) * 1.6

    def set_uav_task(self, u, role, pos=None, patrol=None, cas=()):
        if self.u_state[u] != ACTIVE:
            return
        prev = (self.u_role[u], tuple(np.round(self.u_target[u] / 40.0)))
        if role == RTB:
            self._start_rtb(u)
            return
        self.u_role[u] = role
        if patrol is not None and self.avoid and len(self.rects):
            patrol = [push_out_of_rects(p, self.rects, 15.0) for p in patrol]
        self.u_patrol[u] = patrol
        self.u_cas[u] = list(cas)
        if pos is not None:
            p = np.asarray(pos, float)
            if self.avoid and len(self.rects):
                p = push_out_of_rects(p, self.rects, 15.0)
            self.u_target[u] = np.clip(p, 5, self.L - 5)
        if prev[0] != role:
            self.role_switches += 1

    def _start_rtb(self, u):
        if self.u_state[u] == ACTIVE:
            self.u_state[u] = RETURNING
            self.u_role[u] = RTB
            self.u_target[u] = BASE.copy()
            self.u_patrol[u] = None
            self.u_handoff[u] = False
            self.rtb_started += 1

    # medic planning -------------------------------------------------------
    def medic_route_options(self, m, mode, cands=None):
        """Evaluate routes from medic m to candidate casualties.

        Returns dict c -> (eta_s, exposure, max_risk, cells) for reachable c; plus 'safe' flags.
        """
        g = self.grid
        cost, pred = g.plan(g.cell_of(self.m_pos[m]), mode)
        bpos = self.believed_pos()
        out = {}
        if cands is None:
            cands = range(self.cfg.n_casualties)
        for c in cands:
            cell = g.nearest_free(int(g.cell_of(bpos[c])))
            if not np.isfinite(cost[cell]):
                continue
            cells = g.path(pred, cell, cost)
            length, expo, mr = g.route_stats(cells, MEDIC_SPEED)
            eta = length / MEDIC_SPEED
            r_cas = g.risk[cell]
            expo = expo + r_cas * SERVICE_TIME[int(self.believed_tri()[c])]  # on-scene exposure
            safe = (mr <= R_MAX + 1e-9) and (expo <= EXPOSURE_BUDGET)
            out[c] = (eta, expo, mr, cells, safe)
        return out

    def assign_medic(self, m, c, route_info=None, reason="", mode="risk_aware"):
        if self.m_state[m] in (M_INJURED, M_TREAT):
            return False
        if c is None or c < 0:
            if self.m_target[m] >= 0:
                self.reassignments += 1
            self.m_target[m] = -1
            self.m_state[m] = M_IDLE
            self.m_path[m] = None
            return True
        if route_info is None:
            ro = self.medic_route_options(m, mode, [c])
            if c not in ro:
                return False
            route_info = ro[c]
        eta, expo, mr, cells, safe = route_info
        if self.m_target[m] >= 0 and self.m_target[m] != c:
            self.reassignments += 1
        new = self.m_target[m] != c
        self.m_target[m] = c
        self.m_state[m] = M_ENROUTE
        pts = self.grid.centers[cells[1:]] if len(cells) > 1 else np.zeros((0, 2))
        self.m_path[m] = np.vstack([pts, self.believed_pos()[c][None]])
        self.m_pidx[m] = 0
        if new:
            self.assign_total += 1
            self.assign_feasible += int(safe)
            tri = int(self.believed_tri()[c])
            nw = self.believed_news()[c]
            self.instructions.append(dict(
                t=self.t, medic=int(m), action="GO", casualty=int(c), triage=["T1", "T2", "T3"][tri],
                news=None if np.isnan(nw) else int(nw), eta_s=round(float(eta), 1),
                route_exposure=round(float(expo), 2), route_max_risk=round(float(mr), 2),
                safe=bool(safe), reason=reason))
        return True

    def log_deferral(self, c, reason):
        self.deferrals += 1
        self.instructions.append(dict(t=self.t, medic=None, action="DEFER", casualty=int(c),
                                      triage=["T1", "T2", "T3"][int(self.believed_tri()[c])],
                                      reason=reason))

    # ------------------------------------------------------------------ simulation step
    def step(self):
        cfg = self.cfg
        t = self.t
        self.events = set()
        self._move_obstacles()
        self._update_casualties()
        self._apply_failures()
        self._move_uavs()
        self._energy()
        self._update_links()
        self._sense()
        self._transmit()
        self._move_medics()
        self._accumulate()
        self.t += 1
        return self.t >= cfg.horizon

    def _move_obstacles(self):
        if len(self.dyn_c):
            self.dyn_c += self.dyn_v
            for k in range(2):
                lo = self.dyn_c[:, k] < 100; hi = self.dyn_c[:, k] > self.L - 20
                self.dyn_v[lo | hi, k] *= -1
                self.dyn_c[:, k] = np.clip(self.dyn_c[:, k], 100, self.L - 20)
        if len(self.hazards) and np.any(self.haz_v != 0):
            self.hazards[:, :2] += self.haz_v
            self.hazards[:, :2] = np.clip(self.hazards[:, :2], 100, self.L - 40)
            if self.t % 10 == 0:
                self.grid.set_risk(self.hazards)
                self.events.add("hazard_changed")

    def _update_casualties(self):
        cfg = self.cfg
        t = self.t
        act = self.c_alive & ~self.c_treated
        under_treat = np.zeros(cfg.n_casualties, bool)
        for m in range(self.M):
            if self.m_state[m] == M_TREAT:
                under_treat[self.m_target[m]] = True
        rate = self.h0 / self.ttd0
        dec = act & ~under_treat
        self.h[dec] -= rate[dec]
        shock = dec & self.c_shock_t & (np.abs(self.c_shock_time - t) < 0.5)
        if cfg.severity_dynamics == "changing":
            self.h[shock] -= 0.15
        if cfg.severity_dynamics == "fixed":
            dead = dec & (t >= self.ttd0)
        else:
            dead = dec & (self.h <= 0)
        for c in np.where(dead)[0]:
            self.c_alive[c] = False
            self.c_death_time[c] = t
        # mobility
        rm = self.rng_mob
        for c in np.where(act & ~under_treat)[0]:
            pat = self.c_pat[c]
            step = np.zeros(2)
            if pat == "slow":
                self.c_heading[c] += rm.normal(0, 0.3)
                step = 0.45 * np.array([math.cos(self.c_heading[c]), math.sin(self.c_heading[c])])
            elif pat == "intermittent":
                if (t // 20) % 4 == 0:
                    self.c_heading[c] += rm.normal(0, 0.2)
                    step = 1.0 * np.array([math.cos(self.c_heading[c]), math.sin(self.c_heading[c])])
            elif pat == "abrupt":
                if rm.random() < 1 / 120:
                    a = rm.uniform(0, 2 * np.pi)
                    step = rm.uniform(40, 90) * np.array([math.cos(a), math.sin(a)])
            if step.any():
                p = np.clip(self.c_pos[c] + step, 25, self.L - 25)
                if not points_in_rects(p[None], self.rects, 6).any():
                    self.c_pos[c] = p
                else:
                    self.c_heading[c] += np.pi / 2

    def _pick_failure_target(self, kind):
        alive = np.where(self.u_state == ACTIVE)[0]
        if len(alive) == 0:
            return None
        if kind == "observer":
            best, bw = None, -1
            for u in alive:
                d = np.linalg.norm(self.c_pos - self.u_pos[u], axis=1)
                obs = np.where((d < SENSE_R) & self.c_alive & ~self.c_treated)[0]
                if len(obs):
                    w = SEV_WEIGHT[self.true_tri()[obs]].max()
                    if w > bw:
                        best, bw = u, w
            if best is not None:
                return best
            cands = [u for u in alive if self.u_role[u] == OBS]
            return cands[0] if cands else int(self.rng_misc.choice(alive))
        if kind == "gateway":
            gw = [u for u in alive if self.next_hop[u] == self.N]
            if gw:
                return max(gw, key=lambda u: self.u_fwd[u])
        if kind in ("relay", "gateway"):
            rel = [u for u in alive if self.next_hop[u] >= 0 and self.next_hop[u] != self.N and self.u_fwd[u] > 0]
            if not rel:
                rel = [u for u in alive if self.u_role[u] == RELAY]
            if rel:
                return max(rel, key=lambda u: self.u_fwd[u])
        return int(self.rng_misc.choice(alive))

    def _fail(self, u, why="injected"):
        self.u_state[u] = FAILED
        self.u_queue[u] = []
        self.u_fail_time[u] = self.t
        if why == "injected":
            self.u_detect_time[u] = self.t + 3 + self.rng_misc.integers(0, 4)  # missed heartbeats
            self.fail_events.append((self.t, u))
        else:
            self.u_detect_time[u] = self.t + 3

    def _apply_failures(self):
        for (tf, kind) in self.failures:
            if tf == self.t:
                u = self._pick_failure_target(kind)
                if u is not None:
                    self._fail(u)

    def _waypoint(self, u):
        tgt = self.u_target[u]
        if self.u_patrol[u] is not None and self.u_state[u] == ACTIVE:
            pts = self.u_patrol[u]
            tgt = pts[(self.t // 20) % len(pts)]
        if not (self.avoid and len(self.rects)):
            return tgt
        p = self.u_pos[u]
        cache = self.u_wp[u]
        if cache is None or np.linalg.norm(cache[0] - tgt) > 10 or self.t - cache[2] > 15:
            self.u_wp[u] = cache = [np.array(tgt, float), self.vis.path(p, tgt), self.t]
        wps = cache[1]
        while len(wps) > 1 and np.linalg.norm(wps[0] - p) < 6:
            wps.pop(0)
        return wps[0]

    def _move_uavs(self):
        for u in range(self.N):
            st = self.u_state[u]
            if st in (FAILED, CHARGING):
                continue
            wp = self._waypoint(u)
            d = wp - self.u_pos[u]
            dist = np.linalg.norm(d)
            v = d / dist * min(UAV_SPEED, dist) if dist > 1e-6 else np.zeros(2)
            if self.avoid and len(self.dyn_c):
                rel = self.u_pos[u] - self.dyn_c
                dd = np.linalg.norm(rel, axis=1)
                close = dd < self.dyn_r + 35
                for k in np.where(close)[0]:
                    v = v + rel[k] / max(dd[k], 1e-3) * UAV_SPEED * 1.5 * (1 - (dd[k] - self.dyn_r[k]) / 35)
                sp = np.linalg.norm(v)
                if sp > UAV_SPEED:
                    v = v / sp * UAV_SPEED
            if self.avoid and len(self.rects):
                for cand in (v, np.array([v[0], 0.0]), np.array([0.0, v[1]]), -0.3 * v, np.zeros(2)):
                    if not points_in_rects((self.u_pos[u] + cand)[None], self.rects, 4.0).any():
                        v = cand
                        break
            self.u_pos[u] = np.clip(self.u_pos[u] + v, 0, self.L)
            self.u_dist[u] += np.linalg.norm(v)
            # collision checks
            hit = points_in_rects(self.u_pos[u][None], self.rects).any() if len(self.rects) else False
            if len(self.dyn_c):
                hit = hit or bool((np.linalg.norm(self.dyn_c - self.u_pos[u], axis=1) < self.dyn_r).any())
            if hit:
                self.collision_log.append((self.t, int(u), 'rect' if (len(self.rects) and points_in_rects(self.u_pos[u][None], self.rects).any()) else 'dyn', tuple(np.round(self.u_pos[u]))))
                self.collisions += 1
                self._fail(u, "collision")
                continue
            if st == RETURNING and np.linalg.norm(self.u_pos[u] - BASE) < 15:
                self.u_state[u] = CHARGING
                self.u_charge_t[u] = SWAP_TIME
                self.u_queue[u] = []
                self.rtb_completed += 1
            # spend energy
            if np.isfinite(self.u_cap[u]):
                e = 1.0 + 0.6 * np.linalg.norm(v) / UAV_SPEED
                self.u_bat[u] -= e
                self.energy_used += e
                if self.u_bat[u] <= 0:
                    self.stranded += 1
                    self._fail(u, "battery")
            else:
                self.energy_used += 1.0 + 0.6 * np.linalg.norm(v) / UAV_SPEED

    def _energy(self):
        for u in range(self.N):
            if self.u_state[u] == CHARGING:
                self.u_charge_t[u] -= 1
                if self.u_charge_t[u] <= 0:
                    self.u_bat[u] = self.u_cap[u]
                    self.u_state[u] = ACTIVE
                    self.u_role[u] = IDLE
                    self.u_target[u] = BASE + self.rng_misc.uniform(-15, 15, 2)
        if self.energy_mode == "none" or not np.isfinite(self.u_cap).any():
            return
        for u in np.where(self.u_state == ACTIVE)[0]:
            if self.energy_mode == "reactive":
                if self.u_bat[u] < 0.15 * self.u_cap[u]:
                    self._start_rtb(u)
            elif self.energy_mode == "aware":
                need = self.energy_to_base(u) * 1.25 + 25
                if self.u_bat[u] <= need:
                    self._start_rtb(u)
                elif self.u_bat[u] <= need + 90 and not self.u_handoff[u]:
                    self.u_handoff[u] = True  # request replacement; keep holding position
                if self.u_handoff[u]:
                    # leave as soon as a replacement occupies (nearly) the same position
                    others = np.where((self.u_state == ACTIVE) & ~self.u_handoff)[0]
                    if len(others):
                        dd = np.linalg.norm(self.u_target[others] - self.u_target[u], axis=1)
                        dp = np.linalg.norm(self.u_pos[others] - self.u_target[u], axis=1)
                        if ((dd < 40) & (dp < 40)).any():
                            self._start_rtb(u)

    def _update_links(self):
        f = self.cfg.link_degradation * self.disrupt[min(self.t, self.cfg.horizon)]
        self.p_aa = comm.link_prob(self.u_pos, self.u_pos, comm.R50_AA, self.rects, self.dyn_c, self.dyn_r, f)
        mact = self.m_state != M_INJURED
        mp = self.m_pos[mact] if mact.any() else np.zeros((0, 2))
        pag = comm.link_prob(self.u_pos, mp, comm.R50_AG, self.rects, self.dyn_c, self.dyn_r, f)
        self.p_ag = pag
        alive = (self.u_state == ACTIVE) | (self.u_state == RETURNING)
        self.next_hop, self.etx, _ = comm.route(self.p_aa, pag, alive)
        self.p_sink = pag.max(1) if pag.shape[1] else np.zeros(self.N)

    def _sense(self):
        cfg = self.cfg
        if self.t % cfg.sense_period:
            return
        act = np.where((self.u_state == ACTIVE) | (self.u_state == RETURNING))[0]
        if len(act) == 0:
            return
        d = np.linalg.norm(self.u_pos[act][:, None] - self.c_pos[None], axis=2)
        vit_true = self.true_vitals()
        rs = self.rng_sense
        seen_any = np.zeros(cfg.n_casualties, bool)
        for i, u in enumerate(act):
            obs = np.where((d[i] < SENSE_R) & ~self.c_treated)[0]
            for c in obs:
                seen_any[c] = True
                pos = self.c_pos[c] + rs.normal(0, 3, 2)
                if not self.c_alive[c]:
                    pkt = (int(c), float(self.t), pos[0], pos[1], -1.0, T1, 0, 0, 0, 0)
                    self.ub_dead[c] = True
                else:
                    v = vit_true[c] + rs.normal(size=6) * np.array([3.0, 1.0, 0.8, 4.0, 0.15, 0.0])
                    tri, s = triage_from_vitals(v)
                    pkt = (int(c), float(self.t), pos[0], pos[1], float(s), int(tri), v[0], v[1], v[2], v[3])
                    self.ub_tri[c] = int(tri)
                    self.ub_news[c] = float(s)
                self.ub_pos[c] = pos
                self.ub_seen[c] = self.t
                self.gen_count += 1
                comm.enqueue(self.u_queue[u], pkt, cfg.queue_capacity, self.scheduler)

    def _deliver(self, pkt):
        c = pkt[0]
        t = self.t
        lat = t - pkt[1]
        self.deliv_count += 1
        self.latencies.append(lat)
        if lat <= INFO_DEADLINE[pkt[5]]:
            self.timely_pkt += 1
        if pkt[1] <= self.b_gen[c]:
            return
        old_pos = self.b_pos[c].copy()
        old_tri = self.b_tri[c]
        first = not np.isfinite(self.b_gen[c])
        self.b_prev_news[c] = self.b_news[c]
        self.b_prev_gen[c] = self.b_gen[c]
        self.b_gen[c] = pkt[1]
        self.b_pos[c] = (pkt[2], pkt[3])
        self.b_notfound[c] = False
        if pkt[4] < 0:
            self.b_dead[c] = True
            self.events.add("target_lost")
        else:
            self.b_news[c] = pkt[4]
            self.b_tri[c] = pkt[5]
        if first or old_tri != self.b_tri[c] or np.linalg.norm(old_pos - self.b_pos[c]) > 25:
            self.events.add("new_info")
            self.events.add(f"info:{c}")

    def _transmit(self):
        cfg = self.cfg
        incoming = [[] for _ in range(self.N)]
        aoi = self.medic_aoi()
        rl = self.rng_link
        for u in range(self.N):
            q = self.u_queue[u]
            if not q or self.u_state[u] not in (ACTIVE, RETURNING):
                continue
            # drop stale
            if any(self.t - p[1] > comm.PKT_TTL for p in q):
                q[:] = [p for p in q if self.t - p[1] <= comm.PKT_TTL]
            nh = self.next_hop[u]
            if nh < 0 or not q:
                continue
            sel = comm.select_packets(q, cfg.tx_capacity, self.scheduler, self.t, aoi, rl, self.u_deficit[u])
            p_link = self.p_sink[u] if nh == self.N else self.p_aa[u, nh]
            sent = []
            for i in sel:
                if np.isfinite(self.u_cap[u]):
                    self.u_bat[u] -= 0.01
                if rl.random() < p_link:
                    sent.append(i)
                    pkt = q[i]
                    if nh == self.N:
                        self._deliver(pkt)
                    else:
                        incoming[nh].append(pkt)
                        self.u_fwd[nh] += 1
            for i in sorted(sent, reverse=True):
                q.pop(i)
        for u in range(self.N):
            for pkt in incoming[u]:
                comm.enqueue(self.u_queue[u], pkt, cfg.queue_capacity, self.scheduler)
        self.u_fwd *= 0.98

    def _move_medics(self):
        g = self.grid
        for m in range(self.M):
            st = self.m_state[m]
            if st == M_INJURED:
                continue
            if st == M_ENROUTE:
                c = self.m_target[m]
                if not self.c_alive[c] or self.c_treated[c]:
                    if not self.c_alive[c]:
                        self.b_dead[c] = True
                    self.m_state[m] = M_IDLE
                    self.m_target[m] = -1
                    self.events.add("target_lost")
                else:
                    true_d = np.linalg.norm(self.c_pos[c] - self.m_pos[m])
                    if true_d < 30:  # casualty in visual range: go straight
                        step_to = self.c_pos[c]
                    else:
                        path = self.m_path[m]
                        while self.m_pidx[m] < len(path) - 1 and np.linalg.norm(path[self.m_pidx[m]] - self.m_pos[m]) < 3:
                            self.m_pidx[m] += 1
                        step_to = path[self.m_pidx[m]]
                    d = step_to - self.m_pos[m]
                    dist = np.linalg.norm(d)
                    mv = d / dist * min(MEDIC_SPEED, dist) if dist > 1e-6 else np.zeros(2)
                    newp = self.m_pos[m] + mv
                    if len(self.rects) and points_in_rects(newp[None], self.rects).any():
                        newp = self.m_pos[m]  # blocked (stale route); wait for replanning
                        self.events.add("path_blocked")
                    self.m_dist[m] += np.linalg.norm(newp - self.m_pos[m])
                    self.m_pos[m] = newp
                    if np.linalg.norm(self.c_pos[c] - self.m_pos[m]) < 6:
                        self.m_state[m] = M_TREAT
                        self.m_treat_left[m] = SERVICE_TIME[self.true_tri()[c]]
                    elif dist < 1e-6 or (self.m_pidx[m] == len(self.m_path[m]) - 1 and dist < 3 and true_d >= 30):
                        # arrived at believed position, casualty not here
                        self.b_notfound[c] = True
                        self.m_state[m] = M_IDLE
                        self.m_target[m] = -1
                        self.events.add("target_lost")
            elif st == M_IDLE and self.medic_safe_hold:
                r_here = g.risk[g.cell_of(self.m_pos[m])]
                if r_here > 0:  # retreat to the nearest risk-free cell while waiting
                    free = np.where((g.risk == 0) & ~g.blocked)[0]
                    if len(free):
                        tgt = g.centers[free[np.argmin(((g.centers[free] - self.m_pos[m]) ** 2).sum(1))]]
                        d = tgt - self.m_pos[m]
                        dist = np.linalg.norm(d)
                        if dist > 1e-6:
                            mv = d / dist * min(MEDIC_SPEED, dist)
                            self.m_pos[m] = self.m_pos[m] + mv
                            self.m_dist[m] += np.linalg.norm(mv)
            elif st == M_TREAT:
                self.m_treat_left[m] -= 1
                c = self.m_target[m]
                if self.m_treat_left[m] <= 0:
                    self.c_treated[c] = True
                    self.c_treat_time[c] = self.t
                    self.m_state[m] = M_IDLE
                    self.m_target[m] = -1
                    self.events.add("target_lost")
            # hazard exposure / injury
            r = g.risk[g.cell_of(self.m_pos[m])]
            if r > 0:
                self.m_exposure[m] += r
                self.m_maxrisk[m] = max(self.m_maxrisk[m], r)
                if self.rng_harm.random() < KAPPA * r:
                    self.m_state[m] = M_INJURED
                    self.m_injury_time[m] = self.t
                    self.m_target[m] = -1
                    self.events.add("medic_injured")
            if self.m_state[m] == M_IDLE:
                self.events.add("idle")

    def _accumulate(self):
        n = self.cfg.n_casualties
        act = self.c_alive & ~self.c_treated
        if not act.any():
            self.fresh_series.append(1.0)
            return
        aoi = self.medic_aoi()
        tri = self.true_tri()
        fresh = (aoi <= INFO_DEADLINE[tri]) & act
        self.fresh_steps += fresh
        self.need_steps += act
        t1 = act & (tri == T1)
        self.fresh_steps_t1 += (fresh & t1).sum()
        self.need_steps_t1 += t1.sum()
        self.aoi_sum += np.minimum(aoi[act], 120).sum()
        self.aoi_n += act.sum()
        self.fresh_series.append(fresh.sum() / act.sum())
        seen = (self.t - self.ub_seen) <= 10
        self.covered_steps += (seen & act).sum()
        # outage: no active UAV has a route to a medic
        alive = (self.u_state == ACTIVE) | (self.u_state == RETURNING)
        if not ((self.next_hop >= 0) & alive).any():
            self.outage_steps += 1

    # ------------------------------------------------------------------ metrics
    def survived(self):
        """Treated, or alive at horizon with > 300 s of projected untreated survival."""
        rem = np.where(self.cfg.severity_dynamics == "fixed", self.ttd0 - self.t,
                       np.maximum(self.h, 0) / (self.h0 / self.ttd0))
        return self.c_treated | (self.c_alive & (rem > 300))

    def metrics(self):
        n = self.cfg.n_casualties
        w = SEV_WEIGHT[self.tri0]
        tir_c = np.where(self.need_steps > 0, self.fresh_steps / np.maximum(self.need_steps, 1), 1.0)
        need = self.need_steps.sum()
        tir = self.fresh_steps.sum() / need if need else 1.0
        jain = (tir_c.sum() ** 2) / (n * (tir_c ** 2).sum()) if (tir_c ** 2).sum() > 0 else 0.0
        resp = np.where(self.c_treated, self.c_treat_time, self.cfg.horizon)
        dl = self.c_treated & (self.c_treat_time <= TREAT_DEADLINE[self.tri0])
        surv = self.survived()
        res = dict(
            TIR=tir, TIR_T1=self.fresh_steps_t1 / self.need_steps_t1 if self.need_steps_t1 else np.nan,
            min_patient_TIR=tir_c.min(), jain_fairness=jain, starvation_rate=float((tir_c < 0.2).mean()),
            pkt_delivery=self.deliv_count / max(self.gen_count, 1),
            pkt_timely=self.timely_pkt / max(self.gen_count, 1),
            latency=float(np.mean(self.latencies)) if self.latencies else np.nan,
            mean_AoI=self.aoi_sum / max(self.aoi_n, 1),
            outage=self.outage_steps / self.cfg.horizon,
            coverage=self.covered_steps / max(need, 1),
            role_switches=self.role_switches, uav_dist=self.u_dist.sum(), medic_dist=self.m_dist.sum(),
            collisions=self.collisions,
            route_feasibility=self.assign_feasible / max(self.assign_total, 1),
            completion=self.c_treated.mean(), deadline_success=dl.mean(),
            sw_response=(w * resp).sum() / w.sum(),
            survival=surv.mean(), sw_survival=(w * surv).sum() / w.sum(),
            t1_survival=surv[self.tri0 == T1].mean() if (self.tri0 == T1).any() else np.nan,
            medic_exposure=self.m_exposure.sum(), medic_injuries=int((self.m_state == M_INJURED).sum()),
            medic_max_risk=self.m_maxrisk.max() if self.M else 0.0, deferrals=self.deferrals,
            reassignments=self.reassignments,
            energy=self.energy_used, stranded=self.stranded,
            safe_return=self.rtb_completed / self.rtb_started if self.rtb_started else np.nan,
            energy_per_timely=self.energy_used / max(self.timely_pkt, 1),
        )
        # resilience
        fs = np.array(self.fresh_series)
        if self.fail_events:
            det, rec, rc = [], [], []
            for (tf, u) in self.fail_events:
                det.append(self.u_detect_time[u] - tf)
                pre = fs[max(0, tf - 30):tf].mean() if tf > 0 else fs[:1].mean()
                post = fs[tf:]
                ok = np.where(np.convolve(post >= 0.9 * pre, np.ones(5) / 5, "valid") >= 1.0)[0]
                rec.append(float(ok[0]) if len(ok) else float(len(post)))
                rc.append(fs[tf:tf + 100].mean() / max(pre, 1e-6))
            res.update(detect_latency=float(np.mean(det)), recovery_latency=float(np.mean(rec)),
                       recovered_connectivity=float(np.clip(np.mean(rc), 0, 2)))
        return res
