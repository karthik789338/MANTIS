"""Shared experiment utilities: matched-seed parallel runs, aggregation, statistics."""
from __future__ import annotations

import os, sys, json, time, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
MODELS = os.environ.get("MANTIS_MODELS", os.path.join(RES, "models"))

from mantis.config import ScenarioConfig, PolicyConfig, MANTIS_H
from mantis.sim import Simulator
from mantis.heuristics import Combined

EVAL_SEED0 = 10_000_000   # evaluation seeds never used in training (training seeds < 1e6)

_MODELS = {}


def _load_policy(name):
    if name not in _MODELS:
        import torch
        from mantis.marl.nets import ActorPair
        torch.set_num_threads(1)
        ck = torch.load(os.path.join(MODELS, f"{name}.pt"), weights_only=False)
        A = ActorPair(); A.load_state_dict(ck["actors"]); A.eval()
        _MODELS[name] = (A, ck.get("learn_uav", True), ck.get("learn_medic", True))
    return _MODELS[name]


def make_controller(method, pcfg):
    if method.startswith("marl:"):
        from mantis.marl.controller import MARLController, TorchPolicy
        A, lu, lm = _load_policy(method[5:])
        return MARLController(TorchPolicy(A), learn_uav=lu, learn_medic=lm, greedy=True), MANTIS_H
    return Combined(pcfg), pcfg


def run_job(job):
    cfg_kw, pol_kw, method, tags = job
    cfg = ScenarioConfig(**cfg_kw)
    pcfg = MANTIS_H.copy(**pol_kw)
    ctl, pc = make_controller(method, pcfg)
    t0 = time.time()
    sim = Simulator(cfg, pc, ctl)
    m = sim.run()
    m.update(tags)
    m["seed"] = cfg.seed
    m["runtime_s"] = time.time() - t0
    return m


def run_jobs(jobs, name, procs=2):
    t0 = time.time()
    with Pool(procs) as p:
        rows = p.map(run_job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"{name}.csv"), index=False)
    print(f"[{name}] {len(jobs)} episodes in {time.time() - t0:.0f}s", flush=True)
    return df


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def summarize(df, by, metrics):
    rows = []
    for key, g in df.groupby(by, sort=False):
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r["n"] = len(g)
        for m in metrics:
            mu, lo, hi = ci95(g[m])
            r[m] = mu; r[m + "_ci"] = (hi - lo) / 2 if np.isfinite(lo) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def paired_test(df, method_col, a, b, metric, keys=("seed",), extra_keys=()):
    """Paired Wilcoxon signed-rank test of a vs b on matched scenarios."""
    k = list(keys) + list(extra_keys)
    A = df[df[method_col] == a].set_index(k)[metric]
    B = df[df[method_col] == b].set_index(k)[metric]
    j = A.index.intersection(B.index)
    x, y = A.loc[j].values.astype(float), B.loc[j].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = x - y
    if len(d) < 5 or np.allclose(d, 0):
        p = 1.0
    else:
        p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue
    return dict(a=a, b=b, metric=metric, n=len(d), mean_a=x.mean(), mean_b=y.mean(), diff=d.mean(),
                win=(d > 0).mean(), loss=(d < 0).mean(), p=p)


def fmt(v, ci=None, nd=3):
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{v:.{nd}f}"
    if ci is not None and np.isfinite(ci):
        s += f" ± {ci:.{nd}f}"
    return s
