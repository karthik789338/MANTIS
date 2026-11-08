"""Vectorised 2-D geometry helpers (segments vs. rectangles / discs)."""
from __future__ import annotations

import numpy as np


def seg_rect_intersect(P, Q, rects):
    """Liang-Barsky slab test. P, Q: [..., 2]; rects: [K, 4] (x0, y0, x1, y1).

    Returns bool [..., K] — True if segment PQ passes through rectangle k.
    """
    if len(rects) == 0:
        return np.zeros(P.shape[:-1] + (0,), dtype=bool)
    P = P[..., None, :]
    D = Q[..., None, :] - P
    lo = rects[:, :2]
    hi = rects[:, 2:]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / D
        t1 = (lo - P) * inv
        t2 = (hi - P) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    # parallel axis: inside slab -> (-inf, inf), outside -> empty
    par = D == 0
    inside = (P >= lo) & (P <= hi)
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
    t_enter = np.maximum(np.max(tmin, axis=-1), 0.0)
    t_exit = np.minimum(np.min(tmax, axis=-1), 1.0)
    return t_enter <= t_exit


def seg_disc_intersect(P, Q, centers, radii):
    """P, Q: [..., 2]; centers [K, 2]; radii [K]. Bool [..., K]."""
    if len(centers) == 0:
        return np.zeros(P.shape[:-1] + (0,), dtype=bool)
    P = P[..., None, :]
    D = Q[..., None, :] - P
    L2 = np.maximum((D ** 2).sum(-1), 1e-9)
    t = np.clip(((centers - P) * D).sum(-1) / L2, 0.0, 1.0)
    closest = P + t[..., None] * D
    return ((closest - centers) ** 2).sum(-1) < radii ** 2


def points_in_rects(X, rects, margin=0.0):
    """X: [N, 2] -> bool [N, K]."""
    if len(rects) == 0:
        return np.zeros((len(X), 0), dtype=bool)
    lo = rects[:, :2] - margin
    hi = rects[:, 2:] + margin
    return np.all((X[:, None, :] >= lo) & (X[:, None, :] <= hi), axis=-1)


def push_out_of_rects(x, rects, margin=15.0):
    """Move a point to the nearest location outside every (inflated) rectangle."""
    x = np.array(x, dtype=float)
    for _ in range(3):
        moved = False
        for r in rects:
            x0, y0, x1, y1 = r[0] - margin, r[1] - margin, r[2] + margin, r[3] + margin
            if x0 < x[0] < x1 and y0 < x[1] < y1:
                cands = [(x[0] - x0, (x0, x[1])), (x1 - x[0], (x1, x[1])),
                         (x[1] - y0, (x[0], y0)), (y1 - x[1], (x[0], y1))]
                x = np.array(min(cands, key=lambda c: c[0])[1], dtype=float)
                moved = True
        if not moved:
            break
    return x


class VisibilityPlanner:
    """Shortest collision-free 2-D paths around rectangles (visibility graph over inflated corners)."""

    def __init__(self, rects, margin=18.0, L=800.0):
        self.rects = np.asarray(rects, float).reshape(-1, 4)
        self.inner = self.rects + np.array([-margin + 4, -margin + 4, margin - 4, margin - 4])
        cs = []
        for r in self.rects:
            for x in (r[0] - margin, r[2] + margin):
                for y in (r[1] - margin, r[3] + margin):
                    cs.append((x, y))
        cs = np.array(cs).reshape(-1, 2)
        if len(cs):
            ok = (~points_in_rects(cs, self.inner).any(1)) & (cs >= 0).all(1) & (cs <= L).all(1)
            cs = cs[ok]
        self.corners = cs
        n = len(cs)
        if n:
            vis = ~seg_rect_intersect(np.broadcast_to(cs[:, None], (n, n, 2)), np.broadcast_to(cs[None], (n, n, 2)), self.inner).any(-1)
            D = np.linalg.norm(cs[:, None] - cs[None], axis=-1)
            self.W = np.where(vis, D, np.inf)
            np.fill_diagonal(self.W, 0.0)
        else:
            self.W = np.zeros((0, 0))

    def free(self, a, b):
        if len(self.rects) == 0:
            return True
        return not seg_rect_intersect(np.asarray(a, float)[None], np.asarray(b, float)[None], self.inner)[0].any()

    def path(self, a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if self.free(a, b) or len(self.corners) == 0:
            return [b]
        cs = self.corners
        n = len(cs)
        va = ~seg_rect_intersect(np.broadcast_to(a, (n, 2)), cs, self.inner).any(-1)
        vb = ~seg_rect_intersect(cs, np.broadcast_to(b, (n, 2)), self.inner).any(-1)
        da = np.where(va, np.linalg.norm(cs - a, axis=1), np.inf)
        db = np.where(vb, np.linalg.norm(cs - b, axis=1), np.inf)
        # Dijkstra on small dense graph
        dist = da.copy(); prev = np.full(n, -1); done = np.zeros(n, bool)
        for _ in range(n):
            i = int(np.argmin(np.where(done, np.inf, dist)))
            if not np.isfinite(dist[i]):
                break
            done[i] = True
            nd = dist[i] + self.W[i]
            better = nd < dist
            dist = np.where(better, nd, dist); prev = np.where(better, i, prev)
        tot = dist + db
        j = int(np.argmin(tot))
        if not np.isfinite(tot[j]):
            return [b]
        seq = [j]
        while prev[seq[-1]] >= 0:
            seq.append(int(prev[seq[-1]]))
        return [cs[k] for k in seq[::-1]] + [b]
