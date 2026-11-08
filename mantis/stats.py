"""Matched-pair statistics: bootstrap CIs, paired Wilcoxon tests, Holm correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["seed", "n_cas", "n_uav", "n_med"]


def boot_ci(x, n=2000, alpha=0.05, rng=None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = rng or np.random.default_rng(0)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.quantile(b, alpha / 2), np.quantile(b, 1 - alpha / 2)


def summary(df, metrics, by=("policy",)):
    rows = []
    for key, g in df.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n"] = len(g)
        for m in metrics:
            mu, lo, hi = boot_ci(g[m].values)
            r[m] = mu
            r[m + "_lo"] = lo
            r[m + "_hi"] = hi
        rows.append(r)
    return pd.DataFrame(rows)


def paired(df, metric, a, b, extra_keys=(), higher_better=True):
    """Paired comparison of policy a vs b on matched scenarios. Returns dict."""
    keys = KEYS + [k for k in extra_keys if k in df.columns]
    A = df[df.policy == a].set_index(keys)[metric]
    B = df[df.policy == b].set_index(keys)[metric]
    j = A.to_frame("a").join(B.to_frame("b"), how="inner").dropna()
    if len(j) < 3:
        return dict(metric=metric, a=a, b=b, n=len(j), diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan, win=np.nan)
    d = (j.a - j.b).values
    if not higher_better:
        d = -d
    mu, lo, hi = boot_ci(d)
    try:
        p = wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return dict(metric=metric, a=a, b=b, n=len(j), mean_a=j.a.mean(), mean_b=j.b.mean(),
                diff=mu, lo=lo, hi=hi, p=p, win=float((d > 0).mean()), tie=float((d == 0).mean()))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj
