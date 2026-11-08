"""Joint UAV-medic multi-agent policy (MANTIS-MARL) and its MARL baselines.

Architecture: decentralised option-scoring actors with parameter sharing per agent type
(UAV / medic); agents decide sequentially within a decision step and each sees which options
earlier agents took (masking). Training is CTDE: a centralised critic on a privileged state
(MAPPO-style) or a local critic (IPPO). Variants:

  * ``mantis``  : joint UAV<->medic features + centralised critic + heuristic hint + BC init
  * ``mappo``   : centralised critic, no joint cross-type features, no hint, no BC
  * ``ippo``    : local critics, no joint features, no hint, no BC (independent learners)
  * ablations   : mantis without joint features / without BC / without central critic
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env import M_ENROUTE, M_IDLE, M_INJURED, M_TREAT
from ..policies import plan_mode
from ..vitals import SEV_WEIGHT
from . import features as Fz

torch.set_num_threads(1)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class Nets(nn.Module):
    def __init__(self, central=True, hidden=96):
        super().__init__()
        self.au = mlp(Fz.F_A + Fz.F_U + Fz.F_G, hidden, 1)
        self.am = mlp(Fz.F_A + Fz.F_M + Fz.F_G, hidden, 1)
        self.central = central
        self.critic = mlp(Fz.F_S if central else Fz.F_A + Fz.F_G, 128, 1)


VARIANTS = {
    "mantis": dict(joint=True, central=True, hint=True, bc=True),
    "mappo": dict(joint=False, central=True, hint=False, bc=False),
    "ippo": dict(joint=False, central=False, hint=False, bc=False),
    "mantis_nojoint": dict(joint=False, central=True, hint=True, bc=True),
    "mantis_nobc": dict(joint=True, central=True, hint=True, bc=False),
    "mantis_nocentral": dict(joint=True, central=False, hint=True, bc=True),
}
HINT_U, HINT_M = [6, 7], [1]


class JointAgent:
    def __init__(self, variant="mantis", nets=None, mode="sample", seed=0):
        self.variant = variant
        self.v = VARIANTS[variant]
        self.nets = nets or Nets(central=self.v["central"])
        self.mode = mode  # sample | greedy | hint
        self.rng = np.random.default_rng(seed)
        self.record = False
        self.buffer = []

    # ------------------------------------------------------------------ episode hooks
    def reset(self, env):
        self.env = env
        self.steps = []
        self.prev = self._tally(env)

    def _tally(self, env):
        w = SEV_WEIGHT[env.tri0]
        return dict(tr=(w * env.c_treated).sum(), dead=(w * ~env.c_alive).sum(),
                    inj=(env.m_state == M_INJURED).sum(), expo=env.m_exposure.sum(),
                    crash=env.collisions + env.stranded, fresh=float(np.sum(env.fresh_steps * w[None] if False else env.fresh_steps * w)))

    def _reward(self, env, final=False):
        cur = self._tally(env)
        W = SEV_WEIGHT[env.tri0].sum()
        p = self.prev
        r = 5.0 * ((cur["tr"] - p["tr"]) - (cur["dead"] - p["dead"])) / W
        r += 0.01 * (cur["fresh"] - p["fresh"]) / W
        r -= 2.0 * (cur["inj"] - p["inj"]) + 0.05 * (cur["expo"] - p["expo"]) + 0.2 * (cur["crash"] - p["crash"])
        if final:
            w = SEV_WEIGHT[env.tri0]
            s = env.survived() & ~env.c_treated
            r += 5.0 * (w * s).sum() / W
        self.prev = cur
        return float(r)

    def finalize(self, env):
        if self.steps:
            self.steps[-1]["r"] += self._reward(env, final=True)
        return self.steps

    # ------------------------------------------------------------------ action selection
    def _choose(self, net, X, mask):
        if not self.v["hint"]:
            X = X.copy()
            X[:, HINT_U if X.shape[1] == Fz.F_A + Fz.F_U + Fz.F_G else []] = 0.0
        with torch.no_grad():
            logits = net(torch.from_numpy(X)).squeeze(-1).numpy().astype(np.float64)
        logits = np.where(mask, logits, -1e9)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        if self.mode == "greedy":
            a = int(np.argmax(p))
        else:
            a = int(self.rng.choice(len(p), p=p))
        return a, float(np.log(p[a] + 1e-12))

    def _new_step(self, env):
        if self.steps and self.steps[-1]["t"] == env.t:
            return self.steps[-1]
        if self.steps:
            self.steps[-1]["r"] += self._reward(env)
        st = dict(t=env.t, S=Fz.critic_state(env), G=Fz.global_feats(env), dec=[], r=0.0)
        self.steps.append(st)
        return st

    def act_uav(self, env):
        st = self._new_step(env)
        ctx = Fz.uav_options(env, joint=self.v["joint"])
        taken = set()
        for u in ctx["uavs"]:
            if env.u_state[u] != 0:  # failed-but-undetected UAVs get (useless) orders; skip choice
                continue
            F, opts, mask = Fz.uav_option_feats(env, u, ctx, taken)
            A = Fz.uav_agent_feats(env, u)
            X = np.concatenate([np.repeat(A[None], len(F), 0), F, np.repeat(st["G"][None], len(F), 0)], 1)
            if not self.v["hint"]:
                X[:, Fz.F_A + np.array(HINT_U)] = 0.0
            if self.mode == "hint":
                hint_idx = np.where(F[:, 6] > 0.5)[0]
                a = int(hint_idx[0]) if len(hint_idx) else len(F) - 2
                lp = 0.0
            else:
                a, lp = self._choose(self.nets.au, X, mask)
            st["dec"].append(("u", X, mask, a, lp, np.concatenate([A, st["G"]])))
            kind, j = opts[a]
            if kind == "slot":
                taken.add(j)
            Fz.apply_uav_option(env, u, opts[a], ctx)

    def act_medic(self, env):
        # path refresh every second for targets whose location changed (as in event replanning)
        for m in range(env.M):
            c = env.m_target[m]
            if c >= 0 and env.m_state[m] == M_ENROUTE and f"info:{c}" in env.events:
                env.assign_medic(m, c, mode="risk_aware", reason="target location updated")
        idle_evt = (env.m_state == M_IDLE).any() and ("target_lost" in env.events)
        if env.t % env.cfg.decision_interval and not idle_evt:
            return
        st = self._new_step(env)
        ctx = Fz.medic_context(env, "safe")
        taken = set(int(env.m_target[k]) for k in range(env.M) if env.m_state[k] == M_TREAT)
        for m in ctx["movers"]:
            F, opts, mask = Fz.medic_option_feats(env, m, ctx, taken, joint=self.v["joint"])
            A = Fz.medic_agent_feats(env, m)
            X = np.concatenate([np.repeat(A[None], len(F), 0), F, np.repeat(st["G"][None], len(F), 0)], 1)
            if not self.v["hint"]:
                X[:, Fz.F_A + np.array(HINT_M)] = 0.0
                X[:, Fz.F_A + 14] = 0.0  # heuristic utility score is also a hint
            if self.mode == "hint":
                hint_idx = np.where(F[:, 1] > 0.5)[0]
                a = int(hint_idx[0]) if len(hint_idx) else len(F) - 1
                lp = 0.0
            else:
                a, lp = self._choose(self.nets.am, X, mask)
            st["dec"].append(("m", X, mask, a, lp, np.concatenate([A, st["G"]])))
            c = opts[a]
            if c >= 0:
                taken.add(c)
                if env.m_target[m] != c or env.m_state[m] != M_ENROUTE:
                    env.assign_medic(m, c, ctx["routes"][m][c], mode="risk_aware",
                                     reason="MANTIS-MARL joint policy (safe route)")
            else:
                if env.m_target[m] >= 0:
                    env.assign_medic(m, -1, reason="hold: no safe/valuable target")
        # deferral log: casualties without any safe route (system decision, medic safety first)
        cands = set(c for (_, c) in ctx["U"])
        for c in range(env.cfg.n_casualties):
            if env.b_known[c] and not env.c_treated[c] and not env.b_dead[c] and c not in cands and env.t % 60 == 0:
                env.log_deferral(c, "no route within medic risk limits")


def save_agent(agent, path):
    torch.save(dict(variant=agent.variant, state=agent.nets.state_dict()), path)


def load_agent(path, mode="greedy"):
    d = torch.load(path, weights_only=False)
    ag = JointAgent(d["variant"], mode=mode)
    ag.nets.load_state_dict(d["state"])
    return ag
