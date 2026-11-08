"""Unreliable multi-hop wireless: link model, ETX routing, finite queues and packet schedulers."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geometry import seg_rect_intersect, seg_disc_intersect
from .vitals import INFO_DEADLINE, SEV_WEIGHT

R50_AA = 230.0   # m, 50 % packet success distance air-to-air
R50_AG = 190.0   # m, air-to-ground (UAV -> medic radio)
SLOPE = 18.0
NLOS_RECT = 0.55  # range multiplier when a building blocks line of sight
NLOS_DYN = 0.75   # range multiplier when a dynamic obstacle (plume / aircraft) blocks LOS
P_USABLE = 0.25
PKT_TTL = 120.0

# packet layout (tuple): (cas, gen_time, x, y, news, triage, hr, rr, spo2, sbp)


def link_prob(A, B, r50, rects, dyn_c, dyn_r, factor):
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    r = np.full(d.shape, r50)
    if len(rects):
        Pa = np.broadcast_to(A[:, None, :], d.shape + (2,))
        Pb = np.broadcast_to(B[None, :, :], d.shape + (2,))
        r = r * np.where(seg_rect_intersect(Pa, Pb, rects).any(-1), NLOS_RECT, 1.0)
    if len(dyn_c):
        Pa = np.broadcast_to(A[:, None, :], d.shape + (2,))
        Pb = np.broadcast_to(B[None, :, :], d.shape + (2,))
        r = r * np.where(seg_disc_intersect(Pa, Pb, dyn_c, dyn_r).any(-1), NLOS_DYN, 1.0)
    return factor / (1.0 + np.exp((d - r) / SLOPE))


def route(p_aa, p_ag, alive):
    """ETX shortest paths from every UAV to the medic network (virtual sink).

    Returns (next_hop [N] (-1 none, N = sink/medic), etx [N], best_medic [N]).
    """
    N = p_aa.shape[0]
    W = np.zeros((N + 1, N + 1))
    ok = (p_aa >= P_USABLE) & alive[:, None] & alive[None, :]
    np.fill_diagonal(ok, False)
    W[:N, :N] = np.where(ok, 1.0 / np.maximum(p_aa, 1e-6), 0.0)
    if p_ag.shape[1] > 0:
        best_m = p_ag.argmax(1)
        best_p = p_ag.max(1)
    else:
        best_m = np.zeros(N, dtype=int)
        best_p = np.zeros(N)
    okg = (best_p >= P_USABLE) & alive
    W[:N, N] = np.where(okg, 1.0 / np.maximum(best_p, 1e-6), 0.0)
    W[N, :N] = W[:N, N]
    dist, pred = dijkstra(csr_matrix(W), directed=False, indices=N, return_predecessors=True)
    nh = np.where(np.isfinite(dist[:N]), pred[:N], -1)
    nh = np.where(nh < 0, -1, nh)
    return nh.astype(int), dist[:N], best_m


def select_packets(queue, k, scheduler, t, medic_aoi, rng, deficit=None):
    """Return indices of up to k packets to transmit from ``queue``."""
    n = len(queue)
    if n == 0:
        return []
    if n <= k and scheduler != "guarded":
        return list(range(n))
    if scheduler == "fifo":
        return list(range(min(k, n)))
    if scheduler == "sampled":
        return list(rng.choice(n, size=min(k, n), replace=False))
    if scheduler == "deadline":
        slack = np.array([INFO_DEADLINE[p[5]] - (t - p[1]) for p in queue])
        key = np.where(slack < 0, 1e6 - slack, slack)  # expired packets last
        return list(np.argsort(key, kind="stable")[:k])
    if scheduler == "deficit":
        # deficit round robin across casualties, newest packet first per casualty
        by_c = {}
        for i, p in enumerate(queue):
            by_c.setdefault(p[0], []).append(i)
        for c in by_c:
            deficit[c] = deficit.get(c, 0.0) + 1.0
        order = sorted(by_c, key=lambda c: -deficit[c])
        out = []
        while len(out) < k and any(by_c.values()):
            for c in order:
                if by_c[c] and len(out) < k:
                    out.append(by_c[c].pop())  # newest
                    deficit[c] -= 1.0
        return out
    if scheduler == "guarded":
        # severity-weighted staleness reduction with a fairness guard; stale packets ignored
        pri = np.empty(n)
        for i, p in enumerate(queue):
            c, g, tri = p[0], p[1], p[5]
            aoi = min(medic_aoi[c], 3 * INFO_DEADLINE[tri])
            gain = max(aoi - (t - g), 0.0) / INFO_DEADLINE[tri]
            guard = 3.0 if medic_aoi[c] > 2.0 * INFO_DEADLINE[tri] else 1.0
            pri[i] = SEV_WEIGHT[tri] * gain * guard + 1e-3 * g
        order = np.argsort(-pri)
        return [int(i) for i in order[:k] if pri[i] > 1e-2]
    raise ValueError(scheduler)


def enqueue(queue, pkt, capacity, scheduler):
    """Append with finite capacity. Guarded scheduler keeps only the freshest packet per casualty."""
    if scheduler == "guarded":
        for i, p in enumerate(queue):
            if p[0] == pkt[0]:
                if pkt[1] > p[1]:
                    queue[i] = pkt
                return True
        if len(queue) >= capacity:
            j = min(range(len(queue)), key=lambda i: (SEV_WEIGHT[queue[i][5]], queue[i][1]))
            if (SEV_WEIGHT[pkt[5]], pkt[1]) > (SEV_WEIGHT[queue[j][5]], queue[j][1]):
                queue[j] = pkt
                return True
            return False
        queue.append(pkt)
        return True
    if len(queue) >= capacity:
        return False  # drop-tail
    queue.append(pkt)
    return True
