"""MANTIS discrete-time simulator.

One step = cfg.dt seconds. Order of a step:
  world dynamics -> failure injection -> controller -> UAV motion/energy -> casualty
  physiology & mobility -> sensing / packet generation -> multi-hop delivery -> medic
  motion / treatment / hazard exposure -> metric accounting.

All exogenous randomness (map, casualty physiology, mobility, failures, disruption,
medic hazard draws) is pre-sampled from seed-derived streams so that every policy is
evaluated on exactly the same matched scenario.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .config import ScenarioConfig, PolicyConfig, MANTIS_H
from .world import World
from . import vitals as V
from .planning import NetworkPlanner

NONE, NET, RETURN = -1, -2, -3
AIR, CHARGING, READY, DEAD = 0, 1, 2, 3
M_IDLE, M_MOVE, M_TREAT, M_INCAP = 0, 1, 2, 3
MOB_CODES = {"stationary": 0, "slow": 1, "intermittent": 2, "abrupt": 3}


class Simulator:
    def __init__(self, cfg: ScenarioConfig, pcfg: PolicyConfig = MANTIS_H, controller=None):
        self.cfg, self.pcfg = cfg, pcfg
        ss = np.random.SeedSequence(cfg.seed)
        s_map, s_cas, s_mob, s_fail, s_med, s_dis, s_comm = ss.spawn(7)
        self.world = World(cfg, np.random.default_rng(s_map))
        self.rng_comm = np.random.default_rng(s_comm)
        self.t = 0
        self.T = cfg.horizon
        self.dt = cfg.dt
        self._init_casualties(np.random.default_rng(s_cas), np.random.default_rng(s_mob))
        self._init_uavs(np.random.default_rng(s_fail))
        self._init_medics(np.random.default_rng(s_med))
        self._init_disruption(np.random.default_rng(s_dis))
        self._init_kb()
        self._init_metrics()
        self.net = NetworkPlanner(self)
        self.controller = controller
        self.events = []
        self.new_events = []
        if controller is not None:
            controller.reset(self)

    # =============================================================== initialisation
    def _init_casualties(self, rng, rng_mob):
        cfg, W = self.cfg, self.world
        n = self.n = cfg.n_casualties
        pos = []
        while len(pos) < n:
            p = W.incident + rng.normal(0, 150, 2)
            p = np.clip(p, 30, W.L - 30)
            i, j = W.to_cell(p)
            if W.ground_free[i, j]:
                pos.append(p)
        self.cpos = np.array(pos)
        p0, rate, ev_t, ev_rate = V.sample_casualty(rng, n, cfg.severity_dynamics == "changing",
                                                   cfg.horizon * cfg.dt)
        self.ccp = self._make_ccp()
        self._apply_layout(p0)
        self.cp, self.crate, self.cev_t, self.cev_rate = p0, rate, ev_t, ev_rate
        self.ccat = np.select([p0 < 0.6, p0 < 0.85], [2, 1], 0)  # latent injury class
        self.calive = np.ones(n, bool)
        self.cserved = np.zeros(n, bool)
        self.ctreating = np.full(n, -1)
        self.cserve_t = np.full(n, -1)
        self.cdeath_t = np.full(n, -1)
        self.cdeteriorating = rate > 0
        self.creport = self.cpos + rng.normal(0, 40, (n, 2))
        self.cpos0 = self.cpos.copy()
        # mobility
        if cfg.mobility == "mixed":
            self.cmob = rng.integers(0, 4, n)
        else:
            self.cmob = np.full(n, MOB_CODES[cfg.mobility])
        T = cfg.horizon + 2
        self.mob_u = rng_mob.random((T, n, 3))
        self.mob_n = rng_mob.normal(0, 1, (T, n))
        self.cheading = rng_mob.uniform(0, 2 * np.pi, n)
        self.cmove_timer = rng_mob.integers(5, 25, n)
        self.cmoving = np.zeros(n, bool)

    def _make_ccp(self):
        """Medics' casualty collection point on the base->incident axis, cfg.ccp_distance from the
        incident centre; moved to the nearest low-risk free cell if it would start inside a hazard."""
        W, cfg = self.world, self.cfg
        d = W.base - W.incident
        ccp = W.incident + d / (np.linalg.norm(d) + 1e-9) * cfg.ccp_distance
        ccp = W.nearest_free(ccp, W.ground_free)
        r = W.risk_planning(0)
        if r[W.to_cell(ccp)] > cfg.risk_max:
            ccp = W.nearest_free(ccp, W.ground_free & (r <= 0.2))
        return np.asarray(ccp, float)

    def _apply_layout(self, p0):
        """Same patients (identical physiology) placed differently according to severity.
        critical_near / critical_far permute the SAME positions; the others draw new positions."""
        cfg, W, n = self.cfg, self.world, self.n
        lay = cfg.casualty_layout
        if lay == "random" or n <= 1:
            return
        sev_order = np.argsort(p0)                       # most severe (lowest reserve) first
        if lay in ("critical_near", "critical_far"):
            dist_order = np.argsort(np.linalg.norm(self.cpos - self.ccp, axis=1))
            if lay == "critical_far":
                dist_order = dist_order[::-1]
            new = np.empty_like(self.cpos)
            new[sev_order] = self.cpos[dist_order]
            self.cpos = new
            return
        g = np.random.default_rng(cfg.seed + 7_777_777)
        cat = np.select([p0 < 0.6, p0 < 0.85], [2, 1], 0)
        a0 = g.uniform(0, 2 * np.pi)
        centres = [W.incident + 160 * np.array([np.cos(a0 + k * 2.094), np.sin(a0 + k * 2.094)]) for k in range(3)]
        new = []
        for c in range(n):
            while True:
                if lay == "severity_clusters":
                    p = centres[int(cat[c])] + g.normal(0, 35, 2)
                elif lay == "scattered":
                    p = W.incident + g.normal(0, 300, 2)
                else:
                    raise ValueError(lay)
                p = np.clip(p, 30, W.L - 30)
                i, j = W.to_cell(p)
                if W.ground_free[i, j]:
                    new.append(p); break
        self.cpos = np.array(new)

    def _init_uavs(self, rng):
        cfg = self.cfg
        M = self.M = cfg.n_uavs + cfg.n_spare_uavs
        b = self.world.base
        ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
        self.upos = b + 12 * np.stack([np.cos(ang), np.sin(ang)], 1)
        self.ustate = np.full(M, AIR)
        self.ustate[cfg.n_uavs:] = READY
        self.ualive = np.ones(M, bool)
        self.ucap = rng.uniform(0.6, 1.25, M) if cfg.battery_hetero else np.ones(M)
        self.ubat = self.ucap.copy()
        self.utask = np.full(M, NONE)
        self.utarget = self.upos.copy()
        self.urole = np.zeros(M, int)     # 0 none 1 observer 2 relay 3 gateway
        self.ufail_t = np.full(M, -1)
        self.udetect_t = np.full(M, -1)
        self.ucharge = np.zeros(M, int)
        self.ustranded = np.zeros(M, bool)
        self.ucrashed = np.zeros(M, bool)
        self.uq = [[] for _ in range(M)]
        self.usearch = np.zeros(M, int)
        self.usortie_start = np.zeros(M, int)
        # resolve failure schedule; "any" roles are resolved at injection time
        self.fail_sched = sorted(cfg.failures, key=lambda f: f["t"])
        self.fail_track = []
        self.fail_rng = rng

    def _init_medics(self, rng):
        K = self.K = self.cfg.n_medics
        W = self.world
        self.mpos = np.array([W.nearest_free(self.ccp + rng.normal(0, 8, 2), W.ground_free) for _ in range(K)])
        self.mstate = np.full(K, M_IDLE)
        self.mtarget = np.full(K, -1)
        self.mpath = [[] for _ in range(K)]
        self.mtimer = np.zeros(K, int)
        self.mexpo = np.zeros(K)
        self.mdist = np.zeros(K)
        self.mreassign = np.zeros(K, int)
        self.munsafe_steps = np.zeros(K, int)
        self.mmaxrisk = np.zeros(K)
        self.mnotfound = np.zeros(K, int)
        self.maborts = np.zeros(K, int)
        self.mu = rng.random((self.cfg.horizon + 2, K))

    def _init_disruption(self, rng):
        T = self.cfg.horizon + 2
        mult = np.ones(T)
        if self.cfg.disruption == "intermittent":
            bad = False
            u = rng.random(T)
            for t in range(T):
                bad = (u[t] < 0.06) if not bad else (u[t] > 0.25)
                mult[t] = 0.12 if bad else 1.0
        elif self.cfg.disruption == "complete":
            a, b = self.cfg.blackout
            mult[a:b] = 0.0
        self.disrupt = mult

    def _init_kb(self):
        n = self.n
        self.kb_loc = self.creport.copy()
        self.kb_loc_t = np.zeros(n)
        self.kb_known = np.zeros(n, bool)          # vitals known
        self.kb_vit = np.zeros((n, 5))
        self.kb_sev = np.full(n, 0.15)             # prior severity before any vitals
        self.kb_p = np.full(n, 0.75)
        self.kb_rate = np.zeros(n)
        self.kb_last_gen = np.full(n, -1e9)        # gen time (steps) of latest delivered report
        self.kb_dead = np.zeros(n, bool)
        self.kb_hist = [[] for _ in range(n)]
        self.kb_triage = np.ones(n, int)
        self.kb_stale = np.full(n, -10 ** 6)      # step at which a medic failed to find c at its KB location
        # UAV-side belief (shared over the UAV mesh)
        self.ub_loc = self.creport.copy()
        self.ub_t = np.zeros(n)
        if self.cfg.severity_mode == "known":
            self._kb_oracle()

    def _init_metrics(self):
        n, M, T = self.n, self.M, self.T
        self.m_gen = 0
        self.m_delivered = 0
        self.m_lat = []
        self.slot_needed = np.zeros(n)
        self.slot_ok = np.zeros(n)
        self.slot_ok_ts = np.zeros(T + 2)
        self.slot_need_ts = np.zeros(T + 2)
        self.last_gen_slot = np.full(n, -1)
        self.ok_slots = set()
        self.aoi_sum = 0.0; self.aoi_cnt = 0
        self.starve = 0; self.starve_cnt = 0
        self.cov_sum = 0; self.cov_cnt = 0
        self.outage_steps = 0
        self.role_switches = 0
        self.uav_travel = 0.0
        self.collisions = 0
        self.violations = 0
        self.energy_used = 0.0
        self.returns_ok = 0
        self.route_req = 0; self.route_fail = 0
        self.conn_ts = np.ones(T + 2)
        self.airtime = np.zeros(M)
        self.drops = 0
        self.timely_by_sev = np.zeros(2)  # [critical ok, critical needed]
        self.timely_by_sev_need = 0

    # =============================================================== KB updates
    def _kb_oracle(self):
        v = V.vitals_from_reserve(np.maximum(self.cp, 0))
        self.kb_loc = self.cpos.copy(); self.kb_loc_t[:] = self.t
        self.kb_known[:] = True
        self.kb_vit = v
        self.kb_sev = V.severity_from_vitals(v)
        self.kb_p = np.maximum(self.cp, 0)
        self.kb_rate = np.where(self.cserved | (self.ctreating >= 0), 0, self.crate)
        self.kb_dead = ~self.calive
        self.kb_last_gen[:] = self.t
        self.kb_triage = V.triage_from_severity(self.kb_sev)

    def _kb_update(self, c, gen_t, loc, vit, source="uav"):
        if gen_t <= self.kb_last_gen[c] and source == "uav":
            return
        old_tri = self.kb_triage[c]
        old_loc = self.kb_loc[c].copy()
        self.kb_last_gen[c] = max(self.kb_last_gen[c], gen_t)
        self.kb_loc[c] = loc; self.kb_loc_t[c] = gen_t
        if self.cfg.severity_mode == "reports":
            return  # location only, no vitals reach the medics
        first = not self.kb_known[c]
        self.kb_known[c] = True
        self.kb_vit[c] = vit
        if vit[2] <= 0:
            self.kb_dead[c] = True
        self.kb_sev[c] = float(V.severity_from_vitals(vit))
        pe = float(V.reserve_from_vitals(vit))
        h = self.kb_hist[c]
        h.append((gen_t, pe))
        if len(h) > 8:
            h.pop(0)
        if len(h) >= 3 and h[-1][0] - h[0][0] >= 4:
            ts = np.array([a for a, _ in h]) * self.dt
            ps = np.array([b for _, b in h])
            slope = np.polyfit(ts, ps, 1)[0]
            self.kb_rate[c] = 0.6 * self.kb_rate[c] + 0.4 * max(0.0, -slope)
        self.kb_p[c] = pe
        self.kb_triage[c] = int(V.triage_from_severity(self.kb_sev[c]))
        if first:
            self.new_events.append(("new_info", c))
        if self.kb_triage[c] != old_tri:
            self.new_events.append(("triage_change", c))
        if np.linalg.norm(loc - old_loc) > 30:
            self.new_events.append(("loc_change", c))

    # =============================================================== controller API
    def believed_alive(self):
        """UAVs the controller believes alive (failures are only known once detected)."""
        known_dead = (~self.ualive) & ((self.udetect_t >= 0) | self.ustranded | self.ucrashed)
        return ~known_dead

    def uav_available(self):
        return self.believed_alive() & ((self.ustate == AIR) | (self.ustate == READY)) & (self.utask != RETURN)

    def set_uav_task(self, u, task):
        if self.utask[u] == task:
            return
        if not self.ualive[u]:
            self.utask[u] = task  # controller doesn't know yet; harmless
            return
        if self.utask[u] != NONE and self.t > 0:
            self.role_switches += 1
        self.utask[u] = task
        self.usearch[u] = 0
        if self.ustate[u] == READY and task != RETURN:
            self.ustate[u] = AIR
            self.usortie_start[u] = self.t
        elif self.ustate[u] == CHARGING:
            pass

    def medic_assign(self, m, c):
        """Assign medic m to casualty c (or -1 to hold)."""
        if self.mstate[m] in (M_TREAT, M_INCAP):
            return
        if c == self.mtarget[m] and self.mpath[m]:
            return
        if self.mtarget[m] >= 0 and c != self.mtarget[m] and self.mstate[m] == M_MOVE:
            self.mreassign[m] += 1
        self.mtarget[m] = c
        if c < 0:
            self.mpath[m] = []; self.mstate[m] = M_IDLE
            return
        self.medic_reroute(m)

    def medic_reroute(self, m):
        c = self.mtarget[m]
        if c < 0 or self.mstate[m] in (M_TREAT, M_INCAP):
            return
        self.route_req += 1
        path = self.ground_path(self.mpos[m], self.kb_loc[c], self.pcfg.medic_route)
        if path is None:
            self.route_fail += 1
            self.mpath[m] = []; self.mstate[m] = M_IDLE; self.mtarget[m] = -1
            self.new_events.append(("idle", m))
            return
        self.mpath[m] = path
        self.mstate[m] = M_MOVE

    def ground_path(self, a, b, mode):
        W = self.world
        s = W.idx(W.nearest_free(a, W.ground_free))
        d = W.idx(W.nearest_free(b, W.ground_free))
        D, P = W.ground_dist([s], self.t, mode)
        if not np.isfinite(D[0, d]):
            return None
        cells = W.path_from_pred(P[0], s, d)
        if cells is None:
            return None
        pts = [W.center(k) for k in cells[1:]]
        pts.append(np.array(b, float))
        return pts

    _eta_cache = None

    def medic_eta(self, mode=None):
        """ETA (s), route length (m) and risk exposure for every (medic, casualty) using KB locations."""
        mode = mode or self.pcfg.medic_route
        key = (self.t, mode, tuple(map(tuple, np.round(self.mpos, 1))), tuple(map(tuple, np.round(self.kb_loc, 1))))
        if self._eta_cache is not None and self._eta_cache[0] == key:
            return self._eta_cache[1]
        W = self.world
        src = [W.idx(W.nearest_free(p, W.ground_free)) for p in self.mpos]
        dst = [W.idx(W.nearest_free(p, W.ground_free)) for p in self.kb_loc]
        D, P = W.ground_dist(src, self.t, mode)
        eta = np.full((self.K, self.n), np.inf); expo = np.zeros((self.K, self.n)); length = np.full((self.K, self.n), np.inf)
        for m in range(self.K):
            for c in range(self.n):
                if np.isfinite(D[m, dst[c]]):
                    cells = W.path_from_pred(P[m], src[m], dst[c])
                    L, E = W.path_stats(cells, self.t)
                    length[m, c] = L; expo[m, c] = E
                    eta[m, c] = L / self.cfg.medic_speed
        out = (eta, length, expo)
        self._eta_cache = (key, out)
        return out

    def uav_need_return(self, u):
        """Battery fraction needed to fly home from u's position."""
        d = np.linalg.norm(self.upos[u] - self.world.base)
        steps = d / (self.cfg.uav_speed * self.dt)
        return steps * self._drain(1.0)

    def _drain(self, speed_frac):
        return (0.85 + 0.35 * speed_frac) / self.cfg.endurance_steps

    # =============================================================== main loop
    def run(self):
        while self.t < self.T:
            self.step()
        return self.metrics()

    def step(self):
        self.events = self.new_events
        self.new_events = []
        self.world.step(self.t)
        self._inject_failures()
        self._detect_failures()
        if self.cfg.severity_mode == "known":
            self._kb_oracle()
        if self.controller is not None:
            self.controller.step(self)
        self.net.update()
        self._observer_targets()
        self._move_uavs()
        self._casualties_step()
        self._sense_and_generate()
        self._communicate()
        self._medics_step()
        self._account()
        self.t += 1

    # =============================================================== failures
    def _inject_failures(self):
        while self.fail_sched and self.fail_sched[0]["t"] <= self.t:
            f = self.fail_sched.pop(0)
            cand = []
            if "uav" in f:
                cand = [f["uav"]] if self.ualive[f["uav"]] else []
            else:
                role = {"observer": 1, "relay": 2, "gateway": 3}.get(f.get("role", "any"))
                alive = np.nonzero(self.ualive & (self.ustate == AIR))[0]
                cand = [u for u in alive if role is None or self.urole[u] == role]
                if not cand and role == 2:
                    cand = [u for u in alive if self.urole[u] == 3]
                if not cand:
                    cand = list(alive)
            if cand:
                u = int(cand[self.fail_rng.integers(len(cand))])
                self._kill_uav(u)

    def _kill_uav(self, u, crash=False, stranded=False):
        if not crash and not stranded:
            # role-restoration tracking for injected failures
            sens = self.sensed_now[u] if self.sensed_now is not None else np.zeros(self.n, bool)
            self.fail_track.append(dict(t=self.t, task=int(self.utask[u]), cas=np.nonzero(sens)[0].tolist(),
                                        conn=float(self.conn_ts[max(0, self.t - 1)]), done=None))
        self.ualive[u] = False
        self.ustate[u] = DEAD
        self.ufail_t[u] = self.t
        self.uq[u] = []
        if crash:
            self.ucrashed[u] = True
        if stranded:
            self.ustranded[u] = True

    def _detect_failures(self):
        to = self.cfg.heartbeat_timeout
        for u in np.nonzero((~self.ualive) & (self.udetect_t < 0))[0]:
            if self.ucrashed[u] or self.ustranded[u]:
                self.udetect_t[u] = self.ufail_t[u]
                self.events.append(("uav_fail_detected", int(u)))
                continue
            # heartbeats are lost; detection requires a working link to some neighbour
            if self.t - self.ufail_t[u] >= to and self.disrupt[self.t] > 0:
                self.udetect_t[u] = self.t
                self.events.append(("uav_fail_detected", int(u)))

    # =============================================================== UAVs
    def _observer_targets(self):
        for u in range(self.M):
            c = self.utask[u]
            if not self.ualive[u] or self.ustate[u] != AIR:
                continue
            if c >= 0:
                if self.sensed_now is not None and self.sensed_now[u, c]:
                    self.utarget[u] = self.cpos[c]       # tracking the casualty
                    self.usearch[u] = 0
                else:
                    g = self.ub_loc[c]
                    if np.linalg.norm(self.upos[u] - g) < 25 or self.usearch[u] > 0:
                        # lost track: expanding search pattern around last belief
                        self.usearch[u] += 1
                        k = self.usearch[u]
                        r = 40 + 12 * k
                        a = 0.9 * k
                        self.utarget[u] = g + r * np.array([np.cos(a), np.sin(a)])
                        if k > 40:
                            self.usearch[u] = 0
                    else:
                        self.utarget[u] = g
            elif c == RETURN:
                self.utarget[u] = self.world.base

    sensed_now = None

    def _move_uavs(self):
        W, cfg = self.world, self.cfg
        mode = self.pcfg.positioning
        step = cfg.uav_speed * self.dt
        for u in range(self.M):
            if not self.ualive[u]:
                continue
            if self.ustate[u] == CHARGING:
                self.ucharge[u] -= 1
                if self.ucharge[u] <= 0:
                    self.ustate[u] = READY; self.ubat[u] = self.ucap[u]
                    if self.utask[u] == RETURN:
                        self.utask[u] = NONE
                continue
            if self.ustate[u] == READY:
                continue
            self.airtime[u] += 1
            tgt = self.utarget[u]
            if self.utask[u] == NONE:
                tgt = self.upos[u]
            if mode != "static":
                tgt = W.nearest_free(tgt, W.fly_free)
            nxt = self._nav(u, tgt, step, mode)
            moved = float(np.linalg.norm(nxt - self.upos[u]))
            self.uav_travel += moved
            self.upos[u] = nxt
            if cfg.energy:
                e = self._drain(moved / step)
                self.ubat[u] -= e
                self.energy_used += e
            # collisions
            i, j = W.to_cell(nxt)
            if W.bh[i, j] >= cfg.uav_alt:
                self.collisions += 1; self._kill_uav(u, crash=True); continue
            if W.nofly[i, j]:
                self.violations += 1
            if len(W.dyn_obs):
                dd = np.linalg.norm(W.dyn_obs - nxt, axis=1)
                if (dd < W.dyn_r).any():
                    self.collisions += 1; self._kill_uav(u, crash=True); continue
            if self.utask[u] == RETURN and np.linalg.norm(nxt - W.base) < 20:
                self.ustate[u] = CHARGING; self.ucharge[u] = cfg.recharge_steps
                self.upos[u] = W.base.copy(); self.uq[u] = []
                self.returns_ok += 1
                continue
            if cfg.energy and self.ubat[u] <= 0:
                self._kill_uav(u, stranded=True)

    def _nav(self, u, tgt, step, mode):
        W = self.world
        p = self.upos[u]
        d = tgt - p
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            goal = p
        elif mode == "static" or not W.segment_blocked(p, tgt, W.fly_free):
            goal = tgt
        else:
            tc = W.idx(tgt)
            f = W.fly_field(tc)
            k = W.idx(p)
            path = [k]
            for _ in range(8):  # descend the distance field
                i, j = divmod(path[-1], W.G)
                best, bv = None, f[path[-1]]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < W.G and 0 <= nj < W.G and f[ni * W.G + nj] < bv:
                            best, bv = ni * W.G + nj, f[ni * W.G + nj]
                if best is None:
                    break
                path.append(best)
            goal = W.center(path[1]) if len(path) > 1 else tgt
            for kk in reversed(path[1:]):
                c = W.center(kk)
                if not W.segment_blocked(p, c, W.fly_free):
                    goal = c
                    break
        v = goal - p
        n = np.linalg.norm(v)
        nxt = p + (v / n * min(step, n) if n > 1e-9 else 0)
        if mode != "static" and len(W.dyn_obs):
            nxt = self._avoid_dynamic(p, nxt, step)
        if mode != "static" and W.segment_blocked(p, nxt, W.fly_free):
            # never cut a corner of a building / restricted cell: fall back to cell-centre moves
            i, j = W.to_cell(p)
            f = W.fly_field(W.idx(tgt))
            best, bv = None, np.inf
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < W.G and 0 <= nj < W.G and W.fly_free[ni, nj]:
                        c = W.center(ni * W.G + nj)
                        if not W.segment_blocked(p, c, W.fly_free) and f[ni * W.G + nj] < bv:
                            best, bv = c, f[ni * W.G + nj]
            if best is None:
                return p
            v = best - p; n = np.linalg.norm(v)
            nxt = p + (v / n * min(step, n) if n > 1e-9 else 0)
        return nxt

    def _avoid_dynamic(self, p, nxt, step):
        W = self.world
        fut = W.dyn_obs + W.dyn_vel * self.dt          # predicted obstacle positions
        def clearance(q):
            return np.min(np.minimum(np.linalg.norm(fut - q, axis=1), np.linalg.norm(W.dyn_obs - q, axis=1)) - W.dyn_r)
        if clearance(nxt) > 15:
            return nxt
        goal_dir = nxt - p
        best, bs = p, -1e9
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for s in (step, 0.5 * step):
                q = p + s * np.array([np.cos(a), np.sin(a)])
                i, j = W.to_cell(q)
                if not W.fly_free[i, j]:
                    continue
                cl = clearance(q)
                score = (cl > 15) * 1000 + min(cl, 60) + 0.5 * float(np.dot(q - p, goal_dir)) / (step + 1e-9) * 10
                if score > bs:
                    best, bs = q, score
        return best

    # =============================================================== casualties
    def _casualties_step(self):
        cfg, W, dt, t = self.cfg, self.world, self.dt, self.t
        for c in range(self.n):
            if not self.calive[c]:
                continue
            if self.cev_t[c] == t and not self.cserved[c]:
                self.crate[c] = max(self.crate[c], self.cev_rate[c])
                self.ccat[c] = 2
                self.cdeteriorating[c] = True
            if self.cserved[c]:
                self.cp[c] = min(1.0, self.cp[c] + 0.0003 * dt)
                continue
            if self.ctreating[c] >= 0:
                continue
            self.cp[c] -= self.crate[c] * dt
            if self.cp[c] <= 0:
                self.cp[c] = 0; self.calive[c] = False; self.cdeath_t[c] = t
                continue
            # mobility
            mob = self.cmob[c]
            u = self.mob_u[t, c]
            new = None
            if mob == 1:
                self.cheading[c] += 0.6 * self.mob_n[t, c]
                new = self.cpos[c] + 0.4 * dt * np.array([np.cos(self.cheading[c]), np.sin(self.cheading[c])])
            elif mob == 2:
                self.cmove_timer[c] -= 1
                if self.cmove_timer[c] <= 0:
                    self.cmoving[c] = not self.cmoving[c]
                    self.cmove_timer[c] = int(3 + 6 * u[0]) if self.cmoving[c] else int(10 + 25 * u[0])
                    self.cheading[c] = 2 * np.pi * u[1]
                if self.cmoving[c]:
                    new = self.cpos[c] + 1.2 * dt * np.array([np.cos(self.cheading[c]), np.sin(self.cheading[c])])
            elif mob == 3:
                if u[0] < 1 / 45:
                    a = 2 * np.pi * u[1]; r = 100 + 120 * u[2]
                    new = self.cpos[c] + r * np.array([np.cos(a), np.sin(a)])
            if new is not None:
                new = np.clip(new, 20, W.L - 20)
                i, j = W.to_cell(new)
                if W.ground_free[i, j]:
                    self.cpos[c] = new
                else:
                    self.cheading[c] += np.pi / 2

    # =============================================================== sensing
    def _sense_and_generate(self):
        cfg, t = self.cfg, self.t
        air = self.ualive & (self.ustate == AIR)
        d = np.linalg.norm(self.upos[:, None, :] - self.cpos[None, :, :], axis=2)
        S = (d <= cfg.sense_radius) & air[:, None] & self.calive[None, :]
        self.sensed_now = S
        slot = t // cfg.report_slot
        for c in np.nonzero(S.any(0))[0]:
            # UAV-side shared belief
            self.ub_loc[c] = self.cpos[c]; self.ub_t[c] = t
            if self.cserved[c] or self.last_gen_slot[c] == slot:
                continue
            us = np.nonzero(S[:, c])[0]
            own = [u for u in us if self.utask[u] == c]
            u = own[0] if own else us[np.argmin(d[us, c])]
            vit = V.vitals_from_reserve(self.cp[c], self.rng_comm, noise=1.0)
            loc = self.cpos[c] + self.rng_comm.normal(0, 5, 2)
            sev = float(V.severity_from_vitals(vit))
            pkt = [int(c), t, sev, loc, vit, slot]
            self._enqueue(u, pkt)
            self.last_gen_slot[c] = slot
            self.m_gen += 1

    def _enqueue(self, u, pkt):
        q = self.uq[u]
        if self.pcfg.scheduling == "guarded":
            # freshness: a newer report of the same casualty supersedes older ones
            for k in range(len(q) - 1, -1, -1):
                if q[k][0] == pkt[0] and q[k][1] < pkt[1]:
                    q.pop(k); self.drops += 1
        q.append(pkt)
        if len(q) > self.cfg.queue_max:
            q.pop(0); self.drops += 1

    # =============================================================== communication
    def _communicate(self):
        cfg, t, W = self.cfg, self.t, self.world
        air = np.nonzero(self.ualive & (self.ustate == AIR))[0]
        self.conn_ts[t] = 0.0
        self._sink_cost = {}
        if len(air) == 0:
            self._age_queues(); self.outage_steps += 1
            return
        mult = self.disrupt[t]
        P = W.link_matrix(self.upos[air], self.upos[air], "aa", cfg.link_degradation) * mult
        np.fill_diagonal(P, 0)
        malive = np.nonzero(self.mstate != M_INCAP)[0]
        if len(malive):
            Pg = W.link_matrix(self.upos[air], self.mpos[malive], "ag", cfg.link_degradation) * mult
            gbest = Pg.max(1); garg = malive[Pg.argmax(1)]
        else:
            gbest = np.zeros(len(air)); garg = np.zeros(len(air), int)
        n = len(air)
        thr = 0.25
        A = np.where(P > thr, P, 0)
        qlen = np.array([len(self.uq[u]) for u in air], float)
        rt = self.pcfg.routing
        with np.errstate(divide="ignore"):
            if rt == "shortest":
                Wm = np.where(A > 0, 1.0, 0)
                Ws = np.where(gbest > thr, 1.0, 0)
            else:
                Wm = np.where(A > 0, 1.0 / np.maximum(A, 1e-9), 0)
                Ws = np.where(gbest > thr, 1.0 / np.maximum(gbest, 1e-9), 0)
                if rt == "queue":
                    Wm = np.where(A > 0, Wm + 0.6 * qlen[None, :] / cfg.node_capacity, 0)
        # graph with sink index n; edges u->v ; compute cost-to-sink by reversed dijkstra
        Gm = np.zeros((n + 1, n + 1))
        Gm[:n, :n] = Wm
        Gm[:n, n] = Ws
        cost = dijkstra(csr_matrix(Gm.T), directed=True, indices=n)
        cost_u = cost[:n]
        tot = Gm[:n, :] + np.where(Gm[:n, :] > 0, np.append(cost_u, 0)[None, :], np.inf)
        tot = np.where(Gm[:n, :] > 0, tot, np.inf)
        nh = np.argmin(tot, 1)
        sensing = self.sensed_now[air].any(1)
        conn = np.isfinite(cost_u)
        if sensing.any():
            self.conn_ts[t] = float(conn[sensing].mean())
            if not conn[sensing].any():
                self.outage_steps += 1
        for k, u in enumerate(air):
            self._sink_cost[u] = cost_u[k]
        order = np.argsort(-np.where(np.isfinite(cost_u), cost_u, -1))
        sched = cfg.node_capacity
        for k in order:
            u = air[k]
            q = self.uq[u]
            if not q or not np.isfinite(cost_u[k]):
                continue
            idx = self._schedule(q, sched)
            tgt = nh[k]
            p = gbest[k] if tgt == n else A[k, tgt]
            sent = []
            for i in idx:
                if self.rng_comm.random() < p:
                    sent.append(i)
            sent_set = set(sent)
            pk = [q[i] for i in sent]
            self.uq[u] = [q[i] for i in range(len(q)) if i not in sent_set]
            if self.cfg.energy:
                self.ubat[u] -= 0.0004 * len(idx)
            for pkt in pk:
                if tgt == n:
                    self._deliver(pkt)
                else:
                    self._enqueue(air[tgt], pkt)
        self._age_queues()

    def _schedule(self, q, cap):
        s = self.pcfg.scheduling
        n = len(q)
        if n <= cap:
            return list(range(n))
        t = self.t
        if s == "fifo":
            return list(range(cap))
        if s == "sampled":
            return list(self.rng_comm.choice(n, cap, replace=False))
        dl = np.array([p[1] + self.cfg.timely_deadline - t for p in q], float)
        if s == "deadline":
            key = np.where(dl >= 0, dl, 1e3 - dl)  # EDF on still-timely packets first
            return list(np.argsort(key, kind="stable")[:cap])
        if s == "deficit":
            # deficit round robin over casualty flows (equal quanta)
            flows = {}
            for i, p in enumerate(q):
                flows.setdefault(p[0], []).append(i)
            out = []
            keys = sorted(flows, key=lambda c: self.slot_ok[c] / max(self.slot_needed[c], 1))
            while len(out) < cap:
                progressed = False
                for c in keys:
                    if flows[c] and len(out) < cap:
                        out.append(flows[c].pop(0)); progressed = True
                if not progressed:
                    break
            return out
        # guarded: severity x urgency priority with a starvation guard
        sev = np.array([p[2] for p in q])
        c_ = np.array([p[0] for p in q])
        age_med = (t - self.kb_last_gen[c_]).clip(0, 60)
        starving = age_med >= 8
        urg = np.clip(1 - dl / self.cfg.timely_deadline, 0, 1.5)
        prio = (1 + 4 * sev) * (0.5 + urg) + 3.0 * starving - 5.0 * (dl < -6)
        return list(np.argsort(-prio, kind="stable")[:cap])

    def _age_queues(self):
        ttl = self.cfg.packet_ttl
        for u in range(self.M):
            q = self.uq[u]
            if q:
                keep = [p for p in q if self.t - p[1] <= ttl]
                self.drops += len(q) - len(keep)
                self.uq[u] = keep

    def _deliver(self, pkt):
        c, gen_t, sev, loc, vit, slot = pkt
        delay = self.t - gen_t
        self.m_delivered += 1
        self.m_lat.append(delay * self.dt)
        if delay <= self.cfg.timely_deadline and (c, slot) not in self.ok_slots:
            self.ok_slots.add((c, slot))
            self.slot_ok[c] += 1
            self.slot_ok_ts[self.t] += 1
            if self.ccat[c] == 2:
                self.timely_by_sev[0] += 1
        if self.cfg.severity_mode != "known":
            self._kb_update(c, gen_t, loc, vit)

    # =============================================================== medics
    def _medics_step(self):
        cfg, W, t, dt = self.cfg, self.world, self.t, self.dt
        risk = W.risk(t)
        speed = cfg.medic_speed * dt
        for m in range(self.K):
            st = self.mstate[m]
            if st == M_INCAP:
                continue
            if st == M_TREAT and self.pcfg.medic_route == "safe":
                i, j = W.to_cell(self.mpos[m])
                if risk[i, j] > 0.85 * cfg.risk_max:
                    # medic safety first: abort treatment and withdraw (patient is compromised)
                    c = self.mtarget[m]
                    self.ctreating[c] = -1; self.mtarget[m] = -1
                    self.maborts[m] += 1
                    self._retreat(m, risk)
                    self.new_events.append(("idle", m))
                    st = self.mstate[m]
            if st == M_TREAT:
                self.mtimer[m] -= 1
                if self.mtimer[m] <= 0:
                    c = self.mtarget[m]
                    self.cserved[c] = True; self.cserve_t[c] = t; self.ctreating[c] = -1
                    self.kb_rate[c] = 0
                    self.mstate[m] = M_IDLE; self.mtarget[m] = -1
                    self.new_events.append(("served", c)); self.new_events.append(("idle", m))
            elif st == M_MOVE:
                # executor-level safety: never step into a cell above the risk limit
                if self.pcfg.medic_route == "safe" and self.mpath[m]:
                    ahead = self.mpath[m][:6]
                    ii, jj = W.cells_of(np.array(ahead))
                    if (W.risk_planning(t)[ii, jj] > cfg.risk_max).any():
                        self.new_events.append(("hazard_route", m))
                        self.medic_reroute(m)
                rem = speed
                while rem > 0 and self.mpath[m]:
                    wp = self.mpath[m][0]
                    v = wp - self.mpos[m]
                    dd = np.linalg.norm(v)
                    if dd <= rem:
                        self.mpos[m] = wp.copy(); self.mpath[m].pop(0); rem -= dd; self.mdist[m] += dd
                    else:
                        self.mpos[m] = self.mpos[m] + v / dd * rem; self.mdist[m] += rem; rem = 0
                self._medic_check_target(m)
            # medic retreats if a hazard moved onto it (safe mode)
            if self.pcfg.medic_route == "safe" and self.mstate[m] != M_TREAT:
                i, j = W.to_cell(self.mpos[m])
                if W.risk_planning(t)[i, j] > cfg.risk_max:
                    self._retreat(m, W.risk_planning(t))
            # local observation of casualties within medic sensing range
            d = np.linalg.norm(self.cpos - self.mpos[m], axis=1)
            for c in np.nonzero(d <= cfg.medic_sense_radius)[0]:
                vit = V.vitals_from_reserve(self.cp[c], self.rng_comm, 0.5)
                if self.cfg.severity_mode != "known":
                    self._kb_update(int(c), t, self.cpos[c].copy(), vit, source="medic")
                    self.kb_last_gen[c] = t
                if not self.calive[c]:
                    self.kb_dead[c] = True
            # hazard exposure
            i, j = W.to_cell(self.mpos[m])
            r = risk[i, j]
            self.mexpo[m] += r * dt
            self.mmaxrisk[m] = max(self.mmaxrisk[m], r)
            if r > cfg.risk_max:
                self.munsafe_steps[m] += 1
            haz = cfg.incap_kappa * r ** 3 * dt
            if self.mu[t, m] < 1 - np.exp(-haz):
                self.mstate[m] = M_INCAP
                c = self.mtarget[m]
                if c >= 0 and self.ctreating[c] == m:
                    self.ctreating[c] = -1
                self.mtarget[m] = -1; self.mpath[m] = []
                self.new_events.append(("medic_down", m))

    def _retreat(self, m, risk):
        W = self.world
        p = self.mpos[m]
        best, br = p, risk[W.to_cell(p)]
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            for r_ in (25, 50, 75):
                q = p + r_ * np.array([np.cos(a), np.sin(a)])
                i, j = W.to_cell(q)
                if W.ground_free[i, j] and risk[i, j] < br:
                    best, br = q, risk[i, j]
        self.mpath[m] = [best]
        self.mstate[m] = M_MOVE

    def _medic_check_target(self, m):
        c = self.mtarget[m]
        cfg = self.cfg
        if c < 0:
            if not self.mpath[m]:
                self.mstate[m] = M_IDLE
            return
        d = np.linalg.norm(self.cpos[c] - self.mpos[m])
        if not self.calive[c] or self.cserved[c] or (self.ctreating[c] >= 0 and self.ctreating[c] != m):
            if not self.calive[c]:
                self.kb_dead[c] = True; self.new_events.append(("dead", c))
            self.mtarget[m] = -1; self.mpath[m] = []; self.mstate[m] = M_IDLE
            self.new_events.append(("idle", m))
            return
        if d <= cfg.reach_radius:
            self.mstate[m] = M_TREAT
            self.ctreating[c] = m
            cat = int(V.triage_from_severity(V.severity_from_vitals(V.vitals_from_reserve(self.cp[c]))))
            self.mtimer[m] = int(V.TREAT_STEPS[cat])
            self.mpath[m] = []
            return
        if not self.mpath[m]:
            if d <= cfg.medic_sense_radius:
                tgt = self.cpos[c]
                if self.pcfg.medic_route == "safe" and self.world.risk_at(tgt, self.t)[0] > cfg.risk_max:
                    self.mstate[m] = M_IDLE; self.mtarget[m] = -1; self.new_events.append(("idle", m))
                    return
                self.mpath[m] = [tgt.copy()]
            else:
                self.mnotfound[m] += 1
                self.mstate[m] = M_IDLE
                self.mtarget[m] = -1
                self.kb_stale[c] = self.t
                self.new_events.append(("notfound", m))

    # =============================================================== accounting
    def _account(self):
        t, cfg = self.t, self.cfg
        for f in self.fail_track:
            if f["done"] is None and t > f["t"]:
                if f["task"] >= 0 or f["cas"]:
                    cs = [c for c in (f["cas"] or [f["task"]]) if self.calive[c] and not self.cserved[c]]
                    ok = (not cs) or bool(self.sensed_now[:, cs].any(0).all())
                    ok = ok and self.conn_ts[t] >= min(f["conn"], 0.99) - 1e-9
                else:
                    ok = self.conn_ts[t] >= min(f["conn"], 0.99) - 1e-9
                if ok:
                    f["done"] = (t - f["t"]) * self.dt
        active = self.calive & ~self.cserved & (self.ctreating < 0)
        if t % cfg.report_slot == 0:
            self.slot_needed[active] += 1
            self.slot_need_ts[t] += active.sum()
        if active.any():
            aoi = (t - np.maximum(self.kb_last_gen[active], 0)) * self.dt
            aoi = np.minimum(aoi, (t + 1) * self.dt)
            self.aoi_sum += aoi.sum(); self.aoi_cnt += active.sum()
            self.starve += int((aoi > 60).sum()); self.starve_cnt += active.sum()
            if self.sensed_now is not None:
                self.cov_sum += int(self.sensed_now[:, active].any(0).sum()); self.cov_cnt += active.sum()
        # role labels for failure injection & metrics
        self.urole[:] = 0
        self.urole[(self.utask >= 0) & self.ualive] = 1
        for u, kind in self.net.roles.items():
            if self.ualive[u] and self.utask[u] == NET:
                self.urole[u] = 3 if kind == "gw" else 2

    # =============================================================== metrics
    def metrics(self):
        cfg, n = self.cfg, self.n
        Ts = self.T * self.dt
        w = V.TRIAGE_WEIGHT[self.ccat]
        tir_c = self.slot_ok / np.maximum(self.slot_needed, 1)
        has = self.slot_needed > 0
        tir = self.slot_ok.sum() / max(self.slot_needed.sum(), 1)
        jain = (tir_c[has].sum() ** 2) / (has.sum() * (tir_c[has] ** 2).sum() + 1e-12) if has.any() else 1.0
        served = self.cserved
        resp = np.where(served, self.cserve_t * self.dt, Ts)
        det = self.cdeteriorating
        crit = self.ccat == 2
        alive_end = self.calive
        # failure resilience
        det_lat, rec_lat, rec_conn = [], [], []
        need_ts, ok_ts = self.slot_need_ts, self.slot_ok_ts
        for u in range(self.M):
            if self.ufail_t[u] >= 0 and not self.ustranded[u] and not self.ucrashed[u]:
                tf = self.ufail_t[u]
                if self.udetect_t[u] >= 0:
                    det_lat.append((self.udetect_t[u] - tf) * self.dt)
                pre = ok_ts[max(0, tf - 30):tf].sum() / max(need_ts[max(0, tf - 30):tf].sum(), 1)
                rl = None
                for t2 in range(tf + 1, self.T - 3):
                    win = ok_ts[t2:t2 + 6].sum() / max(need_ts[t2:t2 + 6].sum(), 1)
                    if win >= 0.8 * pre and need_ts[t2:t2 + 6].sum() > 0:
                        rl = (t2 - tf) * self.dt
                        break
                rec_lat.append(rl if rl is not None else (self.T - tf) * self.dt)
                rec_conn.append(float(self.conn_ts[tf:self.T].mean()))
        lat = np.array(self.m_lat) if self.m_lat else np.array([np.nan])
        stranded = int(self.ustranded.sum())
        incap = int((self.mstate == M_INCAP).sum())
        crit_need = self.slot_needed[crit].sum()
        out = dict(
            # communication
            TIR=float(tir), PDR=float(self.m_delivered / max(self.m_gen, 1)),
            latency_s=float(np.nanmean(lat)), AoI_s=float(self.aoi_sum / max(self.aoi_cnt, 1)),
            outage=float(self.outage_steps / self.T), drops=int(self.drops),
            TIR_critical=float(self.timely_by_sev[0] / max(crit_need, 1)),
            # coverage & fairness
            coverage=float(self.cov_sum / max(self.cov_cnt, 1)),
            min_patient_TIR=float(tir_c[has].min()) if has.any() else 1.0,
            jain=float(jain), starvation=float(self.starve / max(self.starve_cnt, 1)),
            # resilience
            detect_latency_s=float(np.mean(det_lat)) if det_lat else np.nan,
            recovery_latency_s=float(np.mean(rec_lat)) if rec_lat else np.nan,
            recovered_conn=float(np.mean(rec_conn)) if rec_conn else np.nan,
            role_switches=int(self.role_switches),
            role_restore_s=float(np.mean([f["done"] if f["done"] is not None else (self.T - f["t"]) * self.dt
                                          for f in self.fail_track])) if self.fail_track else np.nan,
            # navigation
            uav_travel_km=float(self.uav_travel / 1000), medic_travel_km=float(self.mdist.sum() / 1000),
            collisions=int(self.collisions), violations=int(self.violations),
            route_feasible=float(1 - self.route_fail / max(self.route_req, 1)),
            # medical response
            completion=float(served.mean()), survival=float(alive_end.mean()),
            deaths=int((~alive_end).sum()),
            sev_w_survival=float((w * alive_end).sum() / w.sum()),
            deadline_success=float((served & det & (self.cdeath_t < 0)).sum() / max(det.sum(), 1)) if det.any() else 1.0,
            T1_golden=float(((self.cserve_t >= 0) & (self.cserve_t * self.dt <= 600) & crit).sum() / max(crit.sum(), 1)),
            sev_w_response_s=float((w * resp).sum() / w.sum()),
            reassignments=int(self.mreassign.sum()), notfound=int(self.mnotfound.sum()),
            # medic safety
            medic_exposure=float(self.mexpo.mean()), medic_incap=incap,
            medic_unsafe_steps=int(self.munsafe_steps.sum()), treatment_aborts=int(self.maborts.sum()), medic_max_risk=float(self.mmaxrisk.max()),
            # energy
            energy=float(self.energy_used),
            energy_per_timely=float(self.energy_used / max(self.slot_ok.sum(), 1)),
            endurance_min=float(self.airtime.mean() * self.dt / 60),
            safe_return=float(self.returns_ok / max(self.returns_ok + stranded, 1)) if (self.returns_ok + stranded) else 1.0,
            stranded=stranded,
            start_dist_m=float(np.linalg.norm(self.cpos0 - self.ccp, axis=1).mean()),
            T1_survival=float(self.calive[crit].mean()) if crit.any() else np.nan,
        )
        out["mission_score"] = mission_score(out)
        return out


def mission_score(m):
    """Single scalar used for summaries (higher is better); fixed a priori, not tuned."""
    return (2.0 * m["sev_w_survival"] + 1.0 * m["completion"] + 1.0 * m["TIR"]
            - 1.0 * m["medic_incap"] - 0.02 * m["medic_exposure"] / 10 - 0.2 * m["stranded"]
            - 0.2 * m["collisions"])
