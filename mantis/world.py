"""Map, obstacles, ground hazards, path graphs and the wireless link model."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from .config import ScenarioConfig

_NEI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class World:
    def __init__(self, cfg: ScenarioConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        G = cfg.grid
        self.G = G
        self.L = cfg.map_size
        self.cell = cfg.map_size / G
        c = self.cell
        centers = (np.arange(G) + 0.5) * c
        self.cx, self.cy = np.meshgrid(centers, centers, indexing="ij")  # [i,j] -> x=i, y=j
        self.base = np.array([2.5 * c, 2.5 * c])
        # incident centre far from base
        self.incident = rng.uniform(0.55 * self.L, 0.85 * self.L, size=2)

        self.bh = np.zeros((G, G))           # building heights (0 = free ground)
        self.nofly = np.zeros((G, G), bool)  # restricted airspace
        if cfg.obstacles in ("static", "mixed"):
            self._gen_buildings()
            self._gen_restricted()
        self.ground_free = self.bh <= 0
        self.fly_free = (~self.nofly) & (self.bh < cfg.uav_alt)
        self._fix_connectivity()

        # dynamic UAV obstacles: discs moving in the airspace
        self.dyn_obs = np.zeros((0, 2)); self.dyn_vel = np.zeros((0, 2)); self.dyn_r = np.zeros(0)
        if cfg.obstacles in ("dynamic", "mixed"):
            k = int(rng.integers(3, 6))
            # concentrated between base and incident so they matter
            t = rng.uniform(0.2, 0.9, size=(k, 1))
            pts = self.base + t * (self.incident - self.base) + rng.normal(0, 120, size=(k, 2))
            self.dyn_obs = np.clip(pts, 50, self.L - 50)
            ang = rng.uniform(0, 2 * np.pi, k)
            spd = rng.uniform(3, 8, k)
            self.dyn_vel = np.stack([np.cos(ang), np.sin(ang)], 1) * spd[:, None]
            self.dyn_r = rng.uniform(35, 60, k)

        # ground hazards (risk to medics)
        self.hz_c = np.zeros((0, 2)); self.hz_s = np.zeros(0); self.hz_p = np.zeros(0)
        self.hz_v = np.zeros((0, 2)); self.hz_dyn = np.zeros(0, bool)
        if cfg.hazards != "none":
            n = cfg.n_hazards
            dyn = np.zeros(n, bool)
            if cfg.hazards == "dynamic":
                dyn[:] = True
            elif cfg.hazards == "mixed":
                dyn[: max(1, n // 2)] = True
            cs = self.incident + rng.normal(0, 170, size=(n, 2))
            # also one hazard on the medics' approach corridor (forces a safety trade-off)
            u = (self.base - self.incident) / (np.linalg.norm(self.base - self.incident) + 1e-9)
            cs[-1] = self.incident + u * rng.uniform(150, 300) + rng.normal(0, 50, 2)
            self.hz_c = np.clip(cs, 60, self.L - 60)
            self.hz_s = rng.uniform(55, 95, n)
            self.hz_p = rng.uniform(0.65, 0.95, n)
            ang = rng.uniform(0, 2 * np.pi, n)
            spd = rng.uniform(0.8, 2.0, n)
            self.hz_v = np.stack([np.cos(ang), np.sin(ang)], 1) * spd[:, None] * dyn[:, None]
            self.hz_dyn = dyn
        self._risk_t = -1
        self._risk = np.zeros((G, G))
        self._build_graph_templates()
        self._ground_cache = {}
        self._fly_cache = {}

    # ------------------------------------------------------------------ generation
    def _gen_buildings(self):
        rng, G = self.rng, self.G
        n = int(rng.integers(8, 13))
        bi, bj = self.to_cell(self.base)
        for _ in range(n):
            w, h = rng.integers(2, 5, size=2)
            # place buildings mostly between base and incident
            t = rng.uniform(0.15, 1.05)
            p = self.base + t * (self.incident - self.base) + rng.normal(0, 160, 2)
            i, j = self.to_cell(p)
            i0, j0 = max(0, i - w // 2), max(0, j - h // 2)
            i1, j1 = min(G, i0 + w), min(G, j0 + h)
            if abs(i0 - bi) < 4 and abs(j0 - bj) < 4:
                continue
            self.bh[i0:i1, j0:j1] = rng.uniform(15, 95)
        self.bh[max(0, bi - 2): bi + 3, max(0, bj - 2): bj + 3] = 0.0

    def _gen_restricted(self):
        rng, G = self.rng, self.G
        t = rng.uniform(0.35, 0.6)
        p = self.base + t * (self.incident - self.base)
        perp = np.array([-(self.incident - self.base)[1], (self.incident - self.base)[0]])
        perp /= np.linalg.norm(perp) + 1e-9
        p = p + perp * rng.normal(0, 60)
        i, j = self.to_cell(p)
        w, h = rng.integers(3, 6, size=2)
        self.nofly[max(0, i - w // 2): min(G, i + w // 2 + 1), max(0, j - h // 2): min(G, j + h // 2 + 1)] = True
        bi, bj = self.to_cell(self.base)
        self.nofly[max(0, bi - 2): bi + 3, max(0, bj - 2): bj + 3] = False

    def _fix_connectivity(self):
        """Keep only the ground/flight components that contain the base."""
        for name in ("ground_free", "fly_free"):
            free = getattr(self, name)
            A = self._adj(free, np.ones(free.size))
            _, lab = connected_components(A, directed=False)
            b = self.idx(self.base)
            keep = (lab == lab[b]).reshape(self.G, self.G) & free
            if name == "ground_free":
                self.bh[free & ~keep] = 20.0  # unreachable pockets become rubble
            setattr(self, name, keep)

    def _adj(self, free, node_w):
        G = self.G
        rows, cols, w = [], [], []
        ii, jj = np.nonzero(free)
        for di, dj in _NEI:
            ni, nj = ii + di, jj + dj
            ok = (ni >= 0) & (ni < G) & (nj >= 0) & (nj < G)
            a, b, ni2, nj2 = ii[ok], jj[ok], ni[ok], nj[ok]
            ok2 = free[ni2, nj2]
            if di != 0 and dj != 0:  # no corner cutting
                ok2 &= free[a + di, b] & free[a, b + dj]
            rows.append(a[ok2] * G + b[ok2]); cols.append(ni2[ok2] * G + nj2[ok2])
            w.append(np.full(ok2.sum(), np.hypot(di, dj)))
        r, c_, ww = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
        return csr_matrix((ww, (r, c_)), shape=(G * G, G * G))

    def _build_graph_templates(self):
        A = self._adj(self.ground_free, None).tocoo()
        self.g_rows, self.g_cols, self.g_len = A.row, A.col, A.data * self.cell
        F = self._adj(self.fly_free, None).tocoo()
        self.fly_graph = csr_matrix((F.data * self.cell, (F.row, F.col)), shape=(self.G ** 2,) * 2)

    # ------------------------------------------------------------------ helpers
    def to_cell(self, p):
        i = int(np.clip(p[0] // self.cell, 0, self.G - 1))
        j = int(np.clip(p[1] // self.cell, 0, self.G - 1))
        return i, j

    def idx(self, p):
        i, j = self.to_cell(p)
        return i * self.G + j

    def cells_of(self, P):
        P = np.atleast_2d(P)
        ij = np.clip((P // self.cell).astype(int), 0, self.G - 1)
        return ij[:, 0], ij[:, 1]

    def center(self, k):
        return np.array([(k // self.G + 0.5) * self.cell, (k % self.G + 0.5) * self.cell])

    def nearest_free(self, p, mask):
        i, j = self.to_cell(p)
        if mask[i, j]:
            return p
        d = (self.cx - p[0]) ** 2 + (self.cy - p[1]) ** 2
        d[~mask] = np.inf
        k = int(np.argmin(d))
        return self.center(k)

    # ------------------------------------------------------------------ dynamics
    def step(self, t):
        dt = self.cfg.dt
        if len(self.dyn_obs):
            self.dyn_obs = self.dyn_obs + self.dyn_vel * dt
            for a in range(2):
                lo, hi = self.dyn_obs[:, a] < 30, self.dyn_obs[:, a] > self.L - 30
                self.dyn_vel[lo | hi, a] *= -1
            if self.rng.random() < 0.05:  # occasional heading change
                k = self.rng.integers(len(self.dyn_obs))
                ang = self.rng.uniform(0, 2 * np.pi)
                self.dyn_vel[k] = np.array([np.cos(ang), np.sin(ang)]) * np.linalg.norm(self.dyn_vel[k])
        if self.hz_dyn.any():
            self.hz_c = self.hz_c + self.hz_v * dt
            # hazards wander around the incident area
            d = self.hz_c - self.incident
            far = np.linalg.norm(d, axis=1) > 260
            self.hz_v[far & self.hz_dyn] *= -1
            self.hz_c = np.clip(self.hz_c, 40, self.L - 40)

    def risk_planning(self, t, lookahead=60.0):
        """Risk used for safe planning: max of current and predicted (hazards advected
        by their velocity for `lookahead` seconds)."""
        cur = self.risk(t)
        if not self.hz_dyn.any():
            return cur
        if getattr(self, "_rp_t", None) == t:
            return self._rp
        fut = self._risk_field(self.hz_c + self.hz_v * lookahead)
        self._rp = np.maximum(cur, fut); self._rp_t = t
        return self._rp

    def _risk_field(self, centres):
        r = np.ones((self.G, self.G))
        for c, s, p in zip(centres, self.hz_s, self.hz_p):
            d2 = (self.cx - c[0]) ** 2 + (self.cy - c[1]) ** 2
            r *= 1 - p * np.exp(-d2 / (2 * s * s))
        return 1 - r

    def risk(self, t):
        if t == self._risk_t:
            return self._risk
        r = np.ones((self.G, self.G))
        for c, s, p in zip(self.hz_c, self.hz_s, self.hz_p):
            d2 = (self.cx - c[0]) ** 2 + (self.cy - c[1]) ** 2
            r *= 1 - p * np.exp(-d2 / (2 * s * s))
        self._risk = 1 - r
        self._risk_t = t
        return self._risk

    def risk_at(self, P, t):
        i, j = self.cells_of(P)
        return self.risk(t)[i, j]

    # ------------------------------------------------------------------ ground routing
    def ground_graph(self, t, mode):
        """mode 'shortest' ignores risk; 'safe' forbids cells with risk>risk_max and weights risk."""
        key = (mode, t if (mode == "safe" and self.hz_dyn.any()) else 0)
        g = self._ground_cache.get(key)
        if g is not None:
            return g
        if mode == "safe":
            r = self.risk_planning(t).ravel()
            rr = 0.5 * (r[self.g_rows] + r[self.g_cols])
            ok = (r[self.g_rows] <= self.cfg.risk_max) & (r[self.g_cols] <= self.cfg.risk_max)
            w = self.g_len * (1 + self.cfg.risk_weight * rr)
            g = csr_matrix((w[ok], (self.g_rows[ok], self.g_cols[ok])), shape=(self.G ** 2,) * 2)
        else:
            g = csr_matrix((self.g_len, (self.g_rows, self.g_cols)), shape=(self.G ** 2,) * 2)
        if len(self._ground_cache) > 8:
            self._ground_cache.clear()
        self._ground_cache[key] = g
        return g

    def ground_dist(self, sources, t, mode):
        """Dijkstra from source cells. Returns (cost[S, G*G], pred[S, G*G])."""
        g = self.ground_graph(t, mode)
        D, P = dijkstra(g, directed=True, indices=sources, return_predecessors=True)
        return D, P

    def path_from_pred(self, pred_row, src, dst):
        if src == dst:
            return [dst]
        path = [dst]
        k = dst
        while k != src:
            k = pred_row[k]
            if k < 0:
                return None
            path.append(k)
        return path[::-1]

    def path_stats(self, path, t):
        """Euclidean length and risk exposure (risk*seconds at medic speed) of a cell path."""
        if not path or len(path) < 2:
            return 0.0, 0.0
        a = np.asarray(path)
        P = np.stack([(a // self.G + 0.5) * self.cell, (a % self.G + 0.5) * self.cell], 1)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        r = self.risk(t).ravel()[path]
        expo = float(np.sum(0.5 * (r[1:] + r[:-1]) * seg / self.cfg.medic_speed))
        return float(seg.sum()), expo

    # ------------------------------------------------------------------ flight routing
    def fly_field(self, target_cell):
        f = self._fly_cache.get(target_cell)
        if f is None:
            f = dijkstra(self.fly_graph, directed=False, indices=target_cell)
            if len(self._fly_cache) > 512:
                self._fly_cache.clear()
            self._fly_cache[target_cell] = f
        return f

    def segment_blocked(self, a, b, mask_free):
        n = int(np.linalg.norm(b - a) / (self.cell * 0.2)) + 2
        ts = np.linspace(0, 1, n)[:, None]
        P = a + ts * (b - a)
        i, j = self.cells_of(P)
        return not mask_free[i, j].all()

    # ------------------------------------------------------------------ radio
    def link_matrix(self, A, B, kind, degrade=0.0):
        """Packet success probability between points A[n,2] and B[m,2].
        kind: 'aa' air-to-air, 'ag' air-to-ground."""
        if len(A) == 0 or len(B) == 0:
            return np.zeros((len(A), len(B)))
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        if kind == "aa":
            d50, w = 360.0 * (1 - degrade), 40.0
        else:
            d50, w = 230.0 * (1 - degrade), 30.0
        p = 1.0 / (1.0 + np.exp((d - d50) / w))
        if self.bh.any():
            ts = np.linspace(0.08, 0.92, 9)
            # sample points along each segment
            P = A[:, None, None, :] + ts[None, None, :, None] * (B[None, :, None, :] - A[:, None, None, :])
            i, j = self.cells_of(P.reshape(-1, 2))
            h = self.bh[i, j].reshape(len(A), len(B), len(ts))
            if kind == "aa":
                blocked = h > self.cfg.uav_alt
            else:
                # line from UAV (alt) at t=0 down to ground receiver (1.5 m) at t=1
                elev = self.cfg.uav_alt + (1.5 - self.cfg.uav_alt) * ts
                blocked = h > elev[None, None, :]
            nb = blocked.sum(-1)
            p = p * np.power(0.3 if kind == "aa" else 0.35, nb)
        return p
