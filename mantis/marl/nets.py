"""Permutation-invariant candidate-scoring networks (work for any number of casualties)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import AM, CM, AU, CU, GS


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Scorer(nn.Module):
    """score(agent, candidate | context) for every candidate; masked logits."""

    def __init__(self, a_dim, c_dim, g_dim=GS, h=64):
        super().__init__()
        self.agent = mlp(a_dim + g_dim, h, h)
        self.cand = mlp(c_dim, h, h)
        self.ctx = mlp(c_dim, h, h, n=1)     # pooled candidate context (who else competes)
        self.head = mlp(3 * h, h, 1)

    def forward(self, a, C, g, mask):
        ea = torch.tanh(self.agent(torch.cat([a, g], -1)))                 # [B,h]
        ec = torch.tanh(self.cand(C))                                       # [B,N,h]
        m = mask.float().unsqueeze(-1)
        pooled = (self.ctx(C) * m).sum(1) / m.sum(1).clamp(min=1)           # [B,h]
        x = torch.cat([ea.unsqueeze(1).expand_as(ec), ec, pooled.unsqueeze(1).expand_as(ec)], -1)
        out = self.head(x).squeeze(-1)
        return out.masked_fill(~mask, -1e9)


class ActorPair(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.medic = Scorer(AM, CM, h=h)
        self.uav = Scorer(AU, CU, h=h)

    def scorer(self, kind):
        return self.medic if kind == 0 else self.uav


class TeamCritic(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.v = mlp(GS, h, 1, n=3)

    def forward(self, g):
        return self.v(g).squeeze(-1)


class LocalCritic(nn.Module):
    """IPPO: value from the agent's own observation only (no global state)."""

    def __init__(self, h=64):
        super().__init__()
        self.m = mlp(AM + CM, h, 1, n=3)
        self.u = mlp(AU + CU, h, 1, n=3)

    def forward(self, kind, a, C, mask):
        m = mask.float().unsqueeze(-1)
        pooled = (C * m).sum(1) / m.sum(1).clamp(min=1)
        x = torch.cat([a, pooled], -1)
        return (self.m(x) if kind == 0 else self.u(x)).squeeze(-1)


class QMixer(nn.Module):
    """Monotonic mixer for a variable set of agents: Q_tot = sum_i |w(g,a_i)| Q_i + b(g)."""

    def __init__(self, h=64):
        super().__init__()
        self.wm = mlp(GS + AM, h, 1)
        self.wu = mlp(GS + AU, h, 1)
        self.b = mlp(GS, h, 1)

    def weights(self, kind, g, a):
        net = self.wm if kind == 0 else self.wu
        return F.softplus(net(torch.cat([g, a], -1))).squeeze(-1)
