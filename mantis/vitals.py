"""Simulated physiology, vitals and vitals-based severity / triage.

Each casualty has a latent physiological reserve p in [0, 1] (1 = healthy, 0 = death).
Vitals are generated from p with measurement noise. Severity is computed ONLY from
(possibly delayed / noisy) vitals using the Revised Trauma Score (RTS: GCS, SBP, RR)
combined with shock index (HR/SBP) and SpO2, as in field triage practice.
"""
from __future__ import annotations

import numpy as np

VITAL_NAMES = ("hr", "rr", "sbp", "spo2", "gcs")
TRIAGE_NAMES = ("T3-minimal", "T2-delayed", "T1-immediate")
TRIAGE_WEIGHT = np.array([0.5, 1.5, 3.0])          # severity weights for metrics / reward
TREAT_STEPS = np.array([6, 12, 18])                # 30 s / 60 s / 90 s stabilisation


def vitals_from_reserve(p, rng=None, noise=1.0):
    p = np.clip(np.asarray(p, float), 0, 1)
    q = 1 - p
    v = np.stack([
        72 + 75 * q,                 # HR
        14 + 22 * q,                 # RR
        40 + 90 * p,                 # SBP
        99 - 20 * q ** 1.5,          # SpO2
        15 - 12 * q ** 2,            # GCS
    ], -1)
    if rng is not None and noise > 0:
        sd = np.array([4.0, 1.5, 5.0, 1.0, 0.5]) * noise
        v = v + rng.normal(0, 1, v.shape) * sd
    dead = p <= 0
    v = np.where(np.asarray(dead)[..., None], 0.0, v)
    return v


def _rts_codes(gcs, sbp, rr):
    g = np.select([gcs >= 13, gcs >= 9, gcs >= 6, gcs >= 4], [4, 3, 2, 1], 0)
    s = np.select([sbp > 89, sbp >= 76, sbp >= 50, sbp >= 1], [4, 3, 2, 1], 0)
    r = np.select([(rr >= 10) & (rr <= 29), rr > 29, rr >= 6, rr >= 1], [4, 3, 2, 1], 0)
    return g, s, r


def revised_trauma_score(v):
    v = np.asarray(v, float)
    g, s, r = _rts_codes(v[..., 4], v[..., 2], v[..., 1])
    return 0.9368 * g + 0.7326 * s + 0.2908 * r


def severity_from_vitals(v):
    """Severity in [0,1] from vitals only."""
    v = np.asarray(v, float)
    rts = revised_trauma_score(v)
    si = v[..., 0] / np.maximum(v[..., 2], 1.0)          # shock index
    spo2 = v[..., 3]
    sev = 0.55 * (1 - rts / 7.8408) + 0.30 * np.clip((si - 0.6) / 1.2, 0, 1) \
        + 0.15 * np.clip((96 - spo2) / 16, 0, 1)
    dead = v[..., 2] <= 0
    return np.where(dead, 1.0, np.clip(sev, 0, 1))


def triage_from_severity(sev):
    """0 = T3 minimal, 1 = T2 delayed, 2 = T1 immediate."""
    sev = np.asarray(sev)
    return np.select([sev >= 0.20, sev >= 0.05], [2, 1], 0)


def reserve_from_vitals(v):
    """Invert the vitals model (least squares on the linear parts) -> estimated reserve."""
    v = np.asarray(v, float)
    est = np.stack([
        1 - (v[..., 0] - 72) / 75,
        1 - (v[..., 1] - 14) / 22,
        (v[..., 2] - 40) / 90,
    ], -1)
    return np.clip(est.mean(-1), 0, 1)


def sample_casualty(rng, n, changing=False, horizon_s=1200.0):
    """Sample latent state for n casualties.

    Returns reserve p0, deterioration rate (per second), and optional deterioration event
    (time step, new rate) for 'changing' severity.
    """
    cat = rng.choice(3, size=n, p=[0.3, 0.35, 0.35])  # 0:T3 1:T2 2:T1 (mass-casualty mix)
    p0 = np.empty(n); rate = np.empty(n)
    for k in range(n):
        if cat[k] == 2:
            p0[k] = rng.uniform(0.38, 0.58)
            ttd = rng.uniform(480, 1200)          # 8-20 min to death untreated
        elif cat[k] == 1:
            p0[k] = rng.uniform(0.66, 0.80)
            ttd = rng.uniform(1500, 3600)
        else:
            p0[k] = rng.uniform(0.86, 0.97)
            ttd = np.inf
        rate[k] = 0.0 if np.isinf(ttd) else p0[k] / ttd
    ev_t = np.full(n, -1); ev_rate = np.zeros(n)
    if changing:
        for k in range(n):
            if cat[k] <= 1 and rng.random() < 0.45:   # sudden deterioration (e.g. haemorrhage)
                ev_t[k] = int(rng.uniform(0.1, 0.6) * horizon_s / 5.0)
                ev_rate[k] = rng.uniform(1 / 900, 1 / 500)
    return p0, rate, ev_t, ev_rate
