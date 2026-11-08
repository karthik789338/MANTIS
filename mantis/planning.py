"""Low-level executors shared by every policy (heuristic or learned).

* NetworkPlanner: places gateway / relay UAVs of the network pool to connect observers to
  medics (multi-hop tree), with obstacle- and link-aware placement.
* Medic scoring utilities: survival probability under deadline uncertainty and the
  allocation scores used by heuristics and by MARL features.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import vitals as V

NET = -2


class NetworkPlanner:
    def __init__(self, sim):
        self.sim = sim
        self.roles = {}
        self.n_gw = 1
        self.use_detection = True     # False => planner keeps planning for undetected/dead UAVs
        self.freeze = False           # True => slot mapping fixed after the first plan
        self._map = None
        self._last = -999
        self.slots = []
        self.edges = []
        self._gw_anchor = {}

    @property
    def hop(self):
        return 0.80 * 360.0 * (1 - self.sim.cfg.link_degradation)

    # ------------------------------------------------------------------ pool & terminals
    def _believed(self):
        s = self.sim
        return s.believed_alive() if self.use_detection else np.ones(s.M, bool)

    def pool(self):
        s = self.sim
        b = self._believed()
        return [u for u in range(s.M) if s.utask[u] == NET and b[u] and s.ustate[u] != 1]

    def observer_points(self):
        s = self.sim
        b = self._believed()
        obs = [u for u in range(s.M) if s.utask[u] >= 0 and b[u]]
        pts = [s.utarget[u].copy() if s.ualive[u] else s.upos[u].copy() for u in obs]
        pr = [float(s.kb_sev[s.utask[u]]) for u in obs]
        return obs, pts, pr

    def medic_points(self):
        s = self.sim
        alive = [m for m in range(s.K) if s.mstate[m] != 3]
        if not alive:
            return [s.world.base.copy()]
        return [s.mpos[m].copy() for m in alive]

    # ------------------------------------------------------------------ geometry
    def build(self, obs_pts, pr, med_pts, budget, n_gw):
        """Return (slot points, slot kinds, edges, connected observer flags)."""
        if budget <= 0:
            return [], [], [], [False] * len(obs_pts)
        n_gw = max(1, min(n_gw, budget, len(med_pts)))
        cen = np.mean(obs_pts, 0) if obs_pts else self.sim.world.incident
        order = [int(np.argmin([np.linalg.norm(p - cen) for p in med_pts]))]
        while len(order) < n_gw:
            d = [min(np.linalg.norm(med_pts[i] - med_pts[j]) for j in order) for i in range(len(med_pts))]
            order.append(int(np.argmax(d)))
        slots = [med_pts[i].copy() for i in order]
        kinds = ["gw"] * len(slots)
        hop = self.hop
        if self.sim.pcfg.positioning == "adaptive" and obs_pts:
            # budget-aware gateway placement: if the relay budget cannot span the distance,
            # move the gateway toward the observers while keeping a good air-ground link
            need = self.required_relays(obs_pts, [slots[0]], 1) - 1
            avail = budget - n_gw
            if need > avail:
                k = int(np.argmax(pr)) if len(pr) else 0
                d = obs_pts[k] - slots[0]
                L = np.linalg.norm(d)
                shift = min(170.0, max(0.0, L - (avail + 1) * hop))
                if L > 1e-6 and shift > 0:
                    slots[0] = slots[0] + d / L * shift
        tree = list(slots)
        edges = []
        remaining = budget - len(slots)
        hop = self.hop
        pending = list(range(len(obs_pts)))
        connected = [False] * len(obs_pts)
        while pending:
            best = None
            for k in pending:
                d = [np.linalg.norm(obs_pts[k] - p) for p in tree]
                j = int(np.argmin(d))
                need = max(0, math.ceil(d[j] / hop) - 1)
                key = (need, -pr[k])
                if best is None or key < best[0]:
                    best = (key, k, j, need)
            (_, _), k, j, need = best[0], best[1], best[2], best[3]
            if need > remaining:
                if remaining > 0:
                    # partial chain toward the most important unconnected observer: evenly
                    # spaced relays connect as soon as the medic/gateway closes the distance
                    k = max(pending, key=lambda k: (pr[k], -min(np.linalg.norm(obs_pts[k] - q) for q in tree)))
                    j = int(np.argmin([np.linalg.norm(obs_pts[k] - q) for q in tree]))
                    a, b = tree[j], obs_pts[k]
                    for r in range(1, remaining + 1):
                        q = a + (b - a) * r / (remaining + 1)
                        slots.append(q); kinds.append("relay")
                    remaining = 0
                break
            a, b = tree[j], obs_pts[k]
            prev = len(tree) - 1 if False else j
            chain = [a]
            for r in range(1, need + 1):
                p = a + (b - a) * r / (need + 1)
                slots.append(p); kinds.append("relay"); tree.append(p)
                chain.append(p)
            chain.append(b)
            tree.append(b)
            for x, y in zip(chain[:-1], chain[1:]):
                edges.append((x, y))
            remaining -= need
            connected[k] = True
            pending.remove(k)
        # redundancy: split the longest links (improves link quality & capacity)
        while remaining > 0 and edges:
            L = [np.linalg.norm(x - y) for x, y in edges]
            e = int(np.argmax(L))
            if L[e] < 60:
                # all links already short: add parallel gateway capacity near the busiest medic
                slots.append(slots[0] + np.array([15.0, 15.0])); kinds.append("gw")
                remaining -= 1
                continue
            x, y = edges.pop(e)
            mid = 0.5 * (x + y)
            slots.append(mid); kinds.append("relay")
            edges += [(x, mid), (mid, y)]
            remaining -= 1
        while remaining > 0:
            slots.append(slots[0] + np.array([15.0, -15.0])); kinds.append("gw"); remaining -= 1
        return slots, kinds, edges, connected

    def required_relays(self, obs_pts, med_pts, n_gw=1):
        """Number of relay UAVs needed to connect every observer point (dry run)."""
        hop = self.hop
        n_gw = max(1, min(n_gw, len(med_pts)))
        cen = np.mean(obs_pts, 0) if len(obs_pts) else self.sim.world.incident
        tree = [med_pts[int(np.argmin([np.linalg.norm(p - cen) for p in med_pts]))]]
        pend = list(range(len(obs_pts)))
        tot = 0
        while pend:
            best = None
            for k in pend:
                d = min(np.linalg.norm(obs_pts[k] - p) for p in tree)
                need = max(0, math.ceil(d / hop) - 1)
                if best is None or need < best[0]:
                    best = (need, k)
            need, k = best
            a = tree[int(np.argmin([np.linalg.norm(obs_pts[k] - p) for p in tree]))]
            for r in range(1, need + 1):
                tree.append(a + (obs_pts[k] - a) * r / (need + 1))
            tree.append(obs_pts[k])
            tot += need
            pend.remove(k)
        return tot + n_gw

    def _adapt(self, slots, kinds, edges, mode):
        W = self.sim.world
        if mode == "static":
            return slots
        out = []
        for p, k in zip(slots, kinds):
            q = W.nearest_free(p, W.fly_free)
            out.append(np.array(q, float) if k == "relay" else np.array(p, float))
        if mode != "adaptive" or not W.bh.any():
            return out
        # link-aware placement: move relays off blocked links
        nb = {i: [] for i in range(len(out))}
        def key(p):
            return (round(float(p[0]), 3), round(float(p[1]), 3))
        idx = {key(p): i for i, p in enumerate(slots)}
        for x, y in edges:
            ix, iy = idx.get(key(x)), idx.get(key(y))
            if ix is not None:
                nb[ix].append(y if iy is None else iy)
            if iy is not None:
                nb[iy].append(x if ix is None else ix)
        for i, k in enumerate(kinds):
            if k != "relay" or not nb[i]:
                continue
            neigh = np.array([out[j] if isinstance(j, int) else j for j in nb[i]])
            cur = W.link_matrix(out[i][None], neigh, "aa").min()
            if cur > 0.75:
                continue
            best, bv = out[i], cur
            for r in (40, 80, 120):
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    q = out[i] + r * np.array([np.cos(a), np.sin(a)])
                    ii, jj = W.to_cell(q)
                    if not W.fly_free[ii, jj] or (q < 0).any() or (q > W.L).any():
                        continue
                    v = W.link_matrix(q[None], neigh, "aa").min()
                    if v > bv + 0.02:
                        best, bv = q, v
            out[i] = best
        return out

    # ------------------------------------------------------------------ main
    def update(self, force=False):
        s = self.sim
        mode = s.pcfg.positioning
        period = s.pcfg.epoch if mode == "static" else 2
        pool = self.pool()
        if not pool:
            self.roles = {}
            return
        if not force and s.t - self._last < period and self._map is not None \
                and set(self._map.keys()) == set(pool):
            self._apply()
            return
        self._last = s.t
        obs, pts, pr = self.observer_points()
        med = self.medic_points()
        slots, kinds, edges, _ = self.build(pts, pr, med, len(pool), self.n_gw)
        slots = self._adapt(slots, kinds, edges, mode)
        self.slots, self.kinds, self.edges = slots, kinds, edges
        self._gw_anchor = {}
        for j, kd in enumerate(kinds):
            if kd == "gw":
                self._gw_anchor[j] = min(med, key=lambda m: np.linalg.norm(m - slots[j])).copy()
        if self.freeze and self._map is not None and set(self._map.keys()) == set(pool) \
                and max(self._map.values()) < len(slots):
            mapping = self._map
        else:
            C = np.array([[np.linalg.norm(s.upos[u] - p) for p in slots] for u in pool])
            r, c = linear_sum_assignment(C)
            mapping = {pool[i]: int(j) for i, j in zip(r, c)}
        self._map = mapping
        self._apply()

    def _apply(self):
        s = self.sim
        self.roles = {}
        for u, j in self._map.items():
            if j >= len(self.slots):
                continue
            self.roles[u] = self.kinds[j]
            if s.ualive[u] and s.utask[u] == NET:
                p = self.slots[j]
                if self.kinds[j] == "gw":
                    # gateways follow their medic continuously (keeping any planned offset)
                    med = self.medic_points()
                    k = int(np.argmin([np.linalg.norm(m - p) for m in med]))
                    off = p - self._gw_anchor[j] if j in self._gw_anchor else 0
                    p = med[k] + off
                s.utarget[u] = p


# ============================================================================ medic scoring
def triage_weight(sim):
    return V.TRIAGE_WEIGHT[sim.kb_triage] * (1 + sim.kb_sev)


def time_to_death(sim):
    r = sim.kb_rate
    return np.where(r > 1e-7, sim.kb_p / np.maximum(r, 1e-9), np.inf)


def survival_prob(sim, arrival_s):
    """P(casualty alive at arrival) given vitals-estimated reserve/trend and info age."""
    ttd = time_to_death(sim)
    aoi = np.maximum(sim.t - sim.kb_last_gen, 0) * sim.dt
    sig = 60 + 0.5 * np.minimum(aoi, 600)
    with np.errstate(invalid="ignore"):
        z = (ttd[None, :] - arrival_s) / sig[None, :]
    z = np.where(np.isinf(ttd)[None, :], 50.0, z)
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def candidate_mask(sim):
    """Casualties a medic may be sent to (known alive, not served, not being treated, and
    whose location is not known to be stale: after a failed search the casualty is
    re-eligible once a newer report arrives or after 2 minutes)."""
    stale = (sim.kb_stale > sim.kb_loc_t) & (sim.t - sim.kb_stale < 24)
    return (~sim.kb_dead) & (~sim.cserved) & (sim.ctreating < 0) & ~stale


def treat_time(sim):
    return V.TREAT_STEPS[sim.kb_triage] * sim.dt


def allocation_scores(sim, kind, eta=None, expo=None):
    """Score matrix [K, n] (higher = better). -inf = infeasible."""
    if eta is None:
        eta, _, expo = sim.medic_eta()
    w = triage_weight(sim)
    tau = treat_time(sim)
    sev = sim.kb_sev
    feas = np.isfinite(eta) & candidate_mask(sim)[None, :]
    if kind == "nearest":
        S = -eta
    elif kind == "severity":
        S = sev[None, :] * 1e5 - eta
    elif kind == "urgency":
        S = w[None, :] / (eta + tau[None, :] + 30)
    elif kind in ("gain", "load"):
        # expected survival GAIN of treating c now vs leaving it untreated over a 30-min
        # planning window (greatest-good triage); infeasible/unsafe casualties excluded.
        Ps = survival_prob(sim, eta + tau[None, :] * 0.5)
        P0 = survival_prob(sim, np.full((1, sim.n), 1800.0))
        gain = np.clip(Ps - P0, 0, 1)
        val = w[None, :] * (gain + 0.15)
        S = val / (eta + tau[None, :] + 30) * np.exp(-expo / 150.0)   # mild preference for low-exposure routes
        if kind == "load":
            S = w[None, :] / (eta + tau[None, :] + 30)
    else:
        raise ValueError(kind)
    return np.where(feas, S, -np.inf)


def assign(S, medics):
    """Joint one-to-one assignment of the given medics maximising total score."""
    if not medics:
        return {}
    sub = S[medics]
    cols = np.nonzero(np.isfinite(sub).any(0))[0]
    if len(cols) == 0:
        return {m: -1 for m in medics}
    sub = sub[:, cols]
    fin = np.isfinite(sub)
    big = np.abs(sub[fin]).max() + 1 if fin.any() else 1
    C = np.where(fin, -sub, 1e3 * big + 1e6)
    r, c = linear_sum_assignment(C)
    out = {m: -1 for m in medics}
    for i, j in zip(r, c):
        if fin[i, j]:
            out[medics[i]] = int(cols[j])
    return out
