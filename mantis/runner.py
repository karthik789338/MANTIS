"""Parallel matched-seed evaluation."""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from .config import PolicySpec, ScenarioConfig
from .policies import run_episode

_LEARNED_CACHE = {}


def _job(args):
    cfg, spec, tags = args
    learned = None
    if spec.learned_path:
        from .marl.agent import load_agent
        key = spec.learned_path
        if key not in _LEARNED_CACHE:
            _LEARNED_CACHE[key] = load_agent(key)
        learned = _LEARNED_CACHE[key]
    t0 = time.time()
    m = run_episode(cfg, spec, learned=learned)
    m["wall_s"] = time.time() - t0
    row = dict(tags)
    row.update(policy=spec.name, seed=cfg.seed, n_cas=cfg.n_casualties, n_uav=cfg.n_uavs, n_med=cfg.n_medics)
    row.update(m)
    return row


def evaluate(jobs, procs=None, out=None, verbose=True):
    """jobs: list of (ScenarioConfig, PolicySpec, tags dict). Returns DataFrame."""
    procs = procs or os.cpu_count()
    t0 = time.time()
    rows = []
    if procs > 1 and len(jobs) > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(procs) as pool:
            for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
                rows.append(r)
                if verbose and (i + 1) % max(1, len(jobs) // 10) == 0:
                    print(f"  {i + 1}/{len(jobs)} episodes  {time.time() - t0:.0f}s", flush=True)
    else:
        rows = [_job(j) for j in jobs]
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    return df
