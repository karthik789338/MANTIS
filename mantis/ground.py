"""Ground grid, hazard risk field and risk-aware medic route planning.

Medic safety model
------------------
Hazard zones (e.g. fire, structural collapse, active threat) create a risk field r(x) in [0, 1].
A medic standing in a cell with risk r for one second is injured with probability KAPPA * r.

Route modes
  * ``shortest``  : minimum-length path, risk-unaware.
  * ``risk_aware``: minimum of length * (1 + LAMBDA * r); cells with r > R_MAX are impassable,
                    and a route whose exposure exceeds EXPOSURE_BUDGET is declared unsafe.
A casualty whose only routes are unsafe is *deferred* by the system (medic safety is a hard
constraint); it is re-evaluated whenever the hazard map changes.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

KAPPA = 0.004          # injury probability per second per unit risk
LAMBDA = 6.0           # risk weighting in the risk-aware route cost
R_MAX = 0.30           # maximum risk of any traversed cell for a safe route
EXPOSURE_BUDGET = 8.0  # max integrated risk-seconds (route + on-scene treatment) per assignment


class Grid:
    def __init__(self, map_size, cell, rects):
        self.cell = cell
        self.G = int(round(map_size / cell))
        G = self.G
        ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        self.centers = np.stack([(ii + 0.5) * cell, (jj + 0.5) * cell], -1).reshape(-1, 2)
        blocked = np.zeros(G * G, dtype=bool)
        for r in rects:
            c = self.centers
            blocked |= (c[:, 0] > r[0] - 2) & (c[:, 0] < r[2] + 2) & (c[:, 1] > r[1] - 2) & (c[:, 1] < r[3] + 2)
        self.blocked = blocked
        src, dst, length = [], [], []
        idx = np.arange(G * G).reshape(G, G)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            i0, i1 = max(0, -di), G - max(0, di)
            j0, j1 = max(0, -dj), G - max(0, dj)
            a = idx[i0:i1, j0:j1].ravel()
            b = idx[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel()
            src.append(a); dst.append(b); length.append(np.full(len(a), cell * np.hypot(di, dj)))
        src = np.concatenate(src); dst = np.concatenate(dst); length = np.concatenate(length)
        ok = ~blocked[src] & ~blocked[dst]
        self.src, self.dst, self.len = src[ok], dst[ok], length[ok]
        self.risk = np.zeros(G * G)

    def cell_of(self, xy):
        xy = np.asarray(xy, dtype=float)
        ij = np.clip((xy / self.cell).astype(int), 0, self.G - 1)
        return ij[..., 0] * self.G + ij[..., 1]

    def nearest_free(self, c):
        if not self.blocked[c]:
            return c
        free = np.where(~self.blocked)[0]
        d = ((self.centers[free] - self.centers[c]) ** 2).sum(1)
        return free[np.argmin(d)]

    def set_risk(self, hazards):
        """hazards: array [H, 4] (x, y, sigma, peak)."""
        r = np.zeros(len(self.centers))
        for x, y, s, p in hazards:
            d2 = ((self.centers - (x, y)) ** 2).sum(1)
            r = np.maximum(r, p * np.exp(-d2 / (s * s)))
        r[r < 0.02] = 0.0
        self.risk = r

    def plan(self, src_cell, mode):
        """Single-source Dijkstra. Returns (cost, pred)."""
        r = self.risk
        if mode == "risk_aware":
            ok = (r[self.src] <= R_MAX) & (r[self.dst] <= R_MAX)
            w = self.len * (1.0 + LAMBDA * 0.5 * (r[self.src] + r[self.dst]))
        else:
            ok = np.ones(len(self.src), dtype=bool)
            w = self.len
        n = len(self.centers)
        A = csr_matrix((w[ok] + 1e-6, (self.src[ok], self.dst[ok])), shape=(n, n))
        src_cell = self.nearest_free(int(src_cell))
        cost, pred = dijkstra(A, directed=True, indices=src_cell, return_predecessors=True)
        return cost, pred

    def path(self, pred, target_cell, cost=None):
        target_cell = int(target_cell)
        if cost is not None and not np.isfinite(cost[target_cell]):
            return None
        cells = [target_cell]
        guard = 0
        while pred[cells[-1]] >= 0 and guard < 10000:
            cells.append(int(pred[cells[-1]]))
            guard += 1
        return cells[::-1]

    def route_stats(self, cells, speed):
        """(length m, exposure risk-seconds, max risk) along a cell path."""
        if not cells:
            return 0.0, 0.0, 0.0
        pts = self.centers[cells]
        seg = np.hypot(*np.diff(pts, axis=0).T) if len(cells) > 1 else np.zeros(0)
        length = float(seg.sum())
        rr = self.risk[cells]
        exposure = float(((rr[1:] + rr[:-1]) * 0.5 * seg).sum() / speed) if len(cells) > 1 else 0.0
        return length, exposure, float(rr.max())
