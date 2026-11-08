"""Observation features for UAV and medic agents (built only from information available to
the system: the medics' knowledge base built from delivered packets, the UAV mesh belief,
own state). Identical features are used by every learning method."""
from __future__ import annotations

import numpy as np

from ..planning import allocation_scores, survival_prob, candidate_mask, triage_weight, treat_time, time_to_death
from ..sim import NET, RETURN, NONE, AIR, READY, CHARGING, M_IDLE, M_MOVE, M_TREAT, M_INCAP

NMAX = 10
AM, CM = 10, 17          # medic agent / candidate dims
AU, CU = 12, 15          # UAV agent / candidate dims
GS = 24                  # global state dims
NC_M = NMAX + 1          # medic candidates: casualties + HOLD
NC_U = NMAX + 2          # UAV candidates: observe casualty + NET + RETURN
HOLD = NMAX
U_NET, U_RET = NMAX, NMAX + 1


def _cap(x, c=2.0):
    return np.minimum(np.nan_to_num(x, nan=c, posinf=c), c)


class FeatureBuilder:
    def __init__(self, sim):
        self.sim = sim

    # ------------------------------------------------------------------ shared per-step cache
    def refresh(self):
        s = self.sim
        self.eta, self.length, self.expo = s.medic_eta()
        self.S = allocation_scores(s, "urgency", self.eta, self.expo)
        self.tau = treat_time(s)
        self.ps = survival_prob(s, self.eta + self.tau[None, :] * 0.5)
        self.ttd = time_to_death(s)
        self.w = triage_weight(s)
        self.aoi = np.maximum(s.t - s.kb_last_gen, 0) * s.dt
        self.cmask = candidate_mask(s)
        self.L = s.world.L
        self.Tm = s.T * s.dt

    def global_state(self):
        s = self.sim
        act = self.cmask
        n = s.n
        alive_m = s.mstate != M_INCAP
        g = np.zeros(GS, np.float32)
        g[0] = s.t / s.T
        g[1] = n / 10; g[2] = s.cfg.n_uavs / 30; g[3] = s.K / 10
        g[4] = s.calive.mean(); g[5] = s.cserved.mean()
        g[6] = (act & (s.kb_triage == 2)).mean()
        g[7] = s.kb_sev[act].mean() if act.any() else 0
        g[8] = _cap(self.aoi[act] / 120).mean() if act.any() else 0
        g[9] = s.conn_ts[max(0, s.t - 1)]
        g[10] = 1 - alive_m.mean()
        g[11] = (s.ubat / s.ucap)[s.ualive].mean() if s.ualive.any() else 0
        g[12] = s.ualive.mean()
        g[13] = min(s.mexpo.mean() / 100, 2)
        g[14] = s.disrupt[s.t]
        g[15] = (s.udetect_t >= 0).mean()
        feas = np.isfinite(self.eta).any(0) if s.K else np.zeros(n, bool)
        g[16] = (act & ~feas).mean()
        g[17] = np.nanmax(np.where(np.isfinite(self.eta), self.ps, 0), 0)[act].mean() if act.any() else 0
        g[18] = (s.mstate == M_TREAT).mean()
        g[19] = (s.mstate == M_IDLE).mean()
        g[20] = (s.utask >= 0).mean(); g[21] = (s.utask == NET).mean()
        g[22] = float(s.cfg.energy); g[23] = (s.ustate == READY).mean()
        return g

    # ------------------------------------------------------------------ medics
    def medic_obs(self, m, chosen, advice=None):
        """chosen: dict medic->casualty chosen earlier in this sequential decision step."""
        s = self.sim
        a = np.zeros(AM, np.float32)
        a[0:2] = s.mpos[m] / self.L
        a[2 + min(int(s.mstate[m]), 2)] = 1
        a[5] = float(s.mtarget[m] >= 0)
        a[6] = min(s.mexpo[m] / 100, 2)
        a[7] = s.t / s.T
        a[8] = s.K / 10; a[9] = s.n / 10
        C = np.zeros((NC_M, CM), np.float32)
        mask = np.zeros(NC_M, bool)
        tgt = {k: (chosen[k] if k in chosen else s.mtarget[k]) for k in range(s.K) if k != m and s.mstate[k] != M_INCAP}
        Smax = np.nanmax(np.where(np.isfinite(self.S), self.S, -np.inf)) if np.isfinite(self.S).any() else 1.0
        Smax = Smax if Smax > 0 else 1.0
        for c in range(s.n):
            f = C[c]
            f[1] = s.kb_sev[c]
            f[2 + int(s.kb_triage[c])] = 1
            f[5] = float(s.kb_known[c])
            f[6] = _cap(self.ttd[c] / 1200)
            f[7] = self.ps[m, c] if np.isfinite(self.eta[m, c]) else 0
            f[8] = _cap(self.eta[m, c] / 1200)
            f[9] = float(np.isfinite(self.eta[m, c]))
            f[10] = _cap(self.expo[m, c] / 50) if np.isfinite(self.eta[m, c]) else 2
            f[11] = _cap(self.aoi[c] / 120)
            f[12] = float(s.mtarget[m] == c)
            others = [k for k, v in tgt.items() if v == c]
            f[13] = len(others) / 3
            oe = [self.eta[k, c] for k in range(s.K) if k != m and s.mstate[k] != M_INCAP]
            f[14] = _cap(min(oe) / 1200) if oe else 2
            f[15] = (self.S[m, c] / Smax) if np.isfinite(self.S[m, c]) else -1
            f[16] = float(advice is not None and advice.get(m, -1) == c)
            mask[c] = bool(self.cmask[c] and np.isfinite(self.eta[m, c]))
        C[HOLD, 0] = 1
        C[HOLD, 16] = float(advice is not None and advice.get(m, -1) < 0)
        C[HOLD, 1] = float((self.cmask & ~np.isfinite(self.eta[m])).mean()) if s.n else 0
        mask[HOLD] = True
        return a, C, mask

    # ------------------------------------------------------------------ UAVs
    def uav_obs(self, u, obs_count, net_count, advice=None):
        s = self.sim
        a = np.zeros(AU, np.float32)
        a[0:2] = s.upos[u] / self.L
        a[2] = s.ubat[u] / s.ucap[u]
        need = s.uav_need_return(u)
        a[3] = min(need / max(s.ubat[u], 1e-3), 2) if s.cfg.energy else 0
        a[4] = np.linalg.norm(s.upos[u] - s.world.base) / self.L
        a[5] = float(s.ustate[u] == AIR); a[6] = float(s.ustate[u] == READY)
        t = s.utask[u]
        a[7 + (0 if t == NONE else 1 if t >= 0 else 2 if t == NET else 3)] = 1
        a[11] = s.t / s.T
        C = np.zeros((NC_U, CU), np.float32)
        mask = np.zeros(NC_U, bool)
        med_tgt = set(int(x) for x in s.mtarget if x >= 0)
        sensed = s.sensed_now.any(0) if s.sensed_now is not None else np.zeros(s.n, bool)
        meta = np.where(np.isfinite(self.eta), self.eta, np.inf).min(0) if s.K else np.full(s.n, np.inf)
        for c in range(s.n):
            f = C[c]
            f[0] = 1
            f[3] = s.kb_sev[c]; f[4] = self.w[c] / 6
            f[5] = _cap(self.aoi[c] / 120)
            f[6] = np.linalg.norm(s.upos[u] - s.ub_loc[c]) / self.L
            f[7] = obs_count.get(c, 0) / 3
            f[8] = float(t == c)
            f[9] = float(sensed[c])
            f[10] = _cap(self.ttd[c] / 1200)
            f[11] = float(c in med_tgt)
            f[12] = _cap(meta[c] / 1200)
            f[13] = float(s.ctreating[c] >= 0)
            f[14] = float(advice is not None and advice.get(u) == c)
            mask[c] = bool(self.cmask[c])
        f = C[U_NET]
        f[1] = 1
        f[3] = net_count / max(s.M, 1)
        f[4] = self._need_relays / max(s.M, 1)
        f[5] = s.conn_ts[max(0, s.t - 1)]
        f[6] = min(np.linalg.norm(s.mpos - s.upos[u], axis=1).min() / self.L, 2) if s.K else 0
        f[8] = float(t == NET)
        f[14] = float(advice is not None and advice.get(u) == NET)
        mask[U_NET] = True
        f = C[U_RET]
        f[2] = 1
        f[3] = a[2]; f[4] = a[3]; f[5] = a[4]
        f[8] = float(t == RETURN or s.ustate[u] == READY)
        f[14] = float(advice is not None and advice.get(u) in (RETURN, NONE))
        mask[U_RET] = bool(s.cfg.energy or s.ustate[u] == READY)
        return a, C, mask

    def set_need_relays(self, obs_pts):
        s = self.sim
        self._need_relays = s.net.required_relays(obs_pts, s.net.medic_points(), 1) if obs_pts else 1


def medic_advice(sim, fb, agents):
    """Advisor: joint urgency-distance assignment of the deciding medics (pure function)."""
    from ..planning import assign
    S = fb.S.copy()
    others = [sim.mtarget[k] for k in range(sim.K) if k not in agents and sim.mtarget[k] >= 0]
    if others:
        S[:, others] = -np.inf
    return assign(S, list(agents))
