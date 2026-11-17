"""Belief-conditioned recurrent control (Sec. 3.1, 4.9; Figure 3).

Actor (decentralised execution):
  target tokens (<= 12): normalised position and velocity, log covariance
  diagonal, selected correlations, 4 IMM probabilities, track age,
  lost/reacquiring flags, accepted-update count, mean accepted reliability
  -> 128-d projection -> 2 set-attention blocks (4 heads) with a learned team token
  UAV-state encoder (128) and age-stamped link/queue encoder (128)
  -> 1-layer 256-unit GRU (32-step truncated sequences)
  -> movement head (hover, N, S, E, W; masked) and target-priority head
     (N_T + 1 outputs; masked).
Critic (training only): centralised state value over simulator-global target
and UAV tokens, with PopArt-normalised targets.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
except ImportError:          # torch is only needed for the learned actor
    torch = None
    nn = object

MAX_T = 12
TOK = 19
UAV_DIM = 7
NB = 7
LINK_DIM = NB * 8 + 1
GLOBAL_TOK = 6


# --------------------------------------------------------------- observation
def observe(ep, a, now_l: float):
    """Local observation of UAV `a` (no simulator truth)."""
    ext = ep.map.extent
    tids = sorted(a.bm.tracks, key=lambda t: np.linalg.norm(a.bm.belief(t, now_l)[0][:2] - a.pos))[:MAX_T]
    toks = np.zeros((MAX_T, TOK), np.float32)
    mask = np.zeros(MAX_T, bool)
    for k, tid in enumerate(tids):
        tr = a.bm.tracks[tid]
        x, P = a.bm.belief(tid, now_l)
        sd = np.sqrt(np.maximum(np.diag(P), 1e-9))
        toks[k] = np.r_[
            (x[:2] - a.pos) / 500.0, x[2:] / 15.0, np.log(np.diag(P) + 1e-6) / 10.0,
            P[0, 1] / (sd[0] * sd[1]), P[0, 2] / (sd[0] * sd[2]), tr.state.mu,
            min(now_l - tr.last_update_t, 30.0) / 30.0, float(tr.status == "lost"),
            float(tr.status == "reacquiring"), np.log1p(tr.n_accept) / 5.0,
            tr.rel_sum / max(tr.n_accept, 1)]
        mask[k] = True
    nbrs = ep.neighbours(a)
    others = [b.pos for b in ep.agents if b is not a]
    coll = min((np.linalg.norm(p - a.pos) for p in others), default=1e3)
    uav = np.array([a.pos[0] / ext, a.pos[1] / ext, a.vel[0] / 20, a.vel[1] / 20,
                    a.energy_wh / ep.cfg.energy.battery_wh,
                    min(a.pos.min(), ext - a.pos.max()) / ext, min(coll, 200) / 200], np.float32)
    link = np.zeros(LINK_DIM, np.float32)
    for k, b in enumerate(nbrs[:NB]):
        le = a.links[b.name]
        dg = a.peer_last.get(b.name)
        age = (now_l - dg[1]) if dg else 30.0
        rel = (b.pos - a.pos) / 500.0 if dg is None else (dg[0].sender_pos - a.pos) / 500.0
        link[k * 8:(k + 1) * 8] = [le.d_mean / 5, le.p_link, max(0, ep.cfg.network.link_capacity_bytes
                                   - a.ctrl_bytes_to[b.name]) / ep.cfg.network.link_capacity_bytes,
                                   np.sqrt(le.d_var), min(age, 30) / 30, rel[0], rel[1], 1.0]
    link[-1] = min(a.vq.Q / ep.cfg.packets.budget_bytes_per_s, 10.0) / 10.0
    move_mask = np.ones(5, bool)
    sp = ep.cfg.scenario.uav_speed_mps
    from .env import ACTIONS
    for j in range(1, 5):
        nxt = a.pos + ACTIONS[j] * sp
        move_mask[j] = -50 <= nxt[0] <= ext + 50 and -50 <= nxt[1] <= ext + 50
    prio_mask = np.r_[mask, True]
    return dict(tokens=toks, tmask=mask, uav=uav, link=link, move_mask=move_mask,
                prio_mask=prio_mask, tids=tids)


def global_state(ep):
    """Simulator-global tokens for the centralised critic (training only)."""
    ext = ep.map.extent
    toks = np.zeros((MAX_T + 8, GLOBAL_TOK), np.float32)
    for j, tg in enumerate(ep.targets[:MAX_T]):
        alive = ep.birth[j] <= ep.t and j not in ep.captured
        toks[j] = [tg.pos[0] / ext, tg.pos[1] / ext, tg.vel[0] / 15, tg.vel[1] / 15, float(alive), 0.0]
    for k, a in enumerate(ep.agents[:8]):
        toks[MAX_T + k] = [a.pos[0] / ext, a.pos[1] / ext, a.vel[0] / 20, a.vel[1] / 20, 1.0, 1.0]
    return toks


# --------------------------------------------------------------------- nets
if torch is not None:
    class SetBlock(nn.Module):
        def __init__(self, d=128, heads=4):
            super().__init__()
            self.att = nn.MultiheadAttention(d, heads, batch_first=True)
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

        def forward(self, x, pad):
            h, _ = self.att(x, x, x, key_padding_mask=pad)
            x = self.n1(x + h)
            return self.n2(x + self.ff(x))

    class Actor(nn.Module):
        def __init__(self, d=128, hidden=256, n_move=5):
            super().__init__()
            self.proj = nn.Linear(TOK, d)
            self.team = nn.Parameter(torch.zeros(1, 1, d))
            self.blocks = nn.ModuleList([SetBlock(d, 4), SetBlock(d, 4)])
            self.uav = nn.Sequential(nn.Linear(UAV_DIM, d), nn.ReLU(), nn.Linear(d, d))
            self.link = nn.Sequential(nn.Linear(LINK_DIM, d), nn.ReLU(), nn.Linear(d, d))
            self.gru = nn.GRU(3 * d, hidden, batch_first=True)
            self.move = nn.Linear(hidden, n_move)
            self.prio_q = nn.Linear(hidden, d)          # priority via attention over target tokens
            self.none = nn.Linear(hidden, 1)

        def forward(self, tokens, tmask, uav, link, h=None):
            """tokens (B,T,12,19), tmask (B,T,12), uav (B,T,7), link (B,T,57)."""
            B, T = tokens.shape[:2]
            x = self.proj(tokens.reshape(B * T, MAX_T, TOK))
            team = self.team.expand(B * T, 1, -1)
            x = torch.cat([team, x], 1)
            pad = torch.cat([torch.zeros(B * T, 1, dtype=torch.bool, device=x.device),
                             ~tmask.reshape(B * T, MAX_T)], 1)
            for blk in self.blocks:
                x = blk(x, pad)
            team_out, tgt = x[:, 0], x[:, 1:]
            z = torch.cat([team_out, self.uav(uav.reshape(B * T, -1)), self.link(link.reshape(B * T, -1))], -1)
            out, h = self.gru(z.reshape(B, T, -1), h)
            o = out.reshape(B * T, -1)
            move = self.move(o)
            q = self.prio_q(o).unsqueeze(1)
            prio = torch.cat([(q * tgt).sum(-1) / np.sqrt(tgt.shape[-1]), self.none(o)], -1)
            return move.reshape(B, T, -1), prio.reshape(B, T, -1), h

    class PopArt(nn.Module):
        def __init__(self, beta=3e-4):
            super().__init__()
            self.lin = nn.Linear(128, 1)
            self.register_buffer("mu", torch.zeros(1))
            self.register_buffer("nu", torch.ones(1))
            self.beta = beta

        @property
        def sigma(self):
            return torch.sqrt(torch.clamp(self.nu - self.mu ** 2, 1e-6, 1e6))

        def forward(self, x):
            return self.lin(x)

        @torch.no_grad()
        def update(self, targets):
            old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
            b = self.beta
            self.mu.mul_(1 - b).add_(b * targets.mean())
            self.nu.mul_(1 - b).add_(b * (targets ** 2).mean())
            new_sigma = self.sigma
            self.lin.weight.mul_(old_sigma / new_sigma)
            self.lin.bias.mul_(old_sigma).add_(old_mu - self.mu).div_(new_sigma)

        def normalize(self, y):
            return (y - self.mu) / self.sigma

        def denormalize(self, y):
            return y * self.sigma + self.mu

    class Critic(nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.proj = nn.Linear(GLOBAL_TOK, d)
            self.block = SetBlock(d, 4)
            self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU())
            self.popart = PopArt()

        def forward(self, g):
            x = self.block(self.proj(g), torch.zeros(g.shape[:2], dtype=torch.bool, device=g.device))
            return self.popart(self.head(x.mean(1))).squeeze(-1)


def masked_logits(logits, mask):
    return logits.masked_fill(~mask, -1e9)


class LearnedActor:
    """Deployable decentralised actor (only the actor is used at execution)."""

    def __init__(self, weights: str = "", device: str = "cpu", greedy: bool = True):
        if torch is None:
            raise ImportError("torch is required for the learned actor")
        self.net = Actor().to(device)
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location=device))
        self.net.eval()
        self.h = {}
        self.greedy = greedy
        self.device = device

    @torch.no_grad()
    def act(self, ep, a, now_l) -> Tuple[int, Optional[int]]:
        o = observe(ep, a, now_l)
        t = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=self.device)[None, None]
        mv, pr, h = self.net(t(o["tokens"]), t(o["tmask"], torch.bool), t(o["uav"]), t(o["link"]), self.h.get(a.uid))
        self.h[a.uid] = h
        mv = masked_logits(mv[0, 0], t(o["move_mask"], torch.bool)[0, 0])
        pr = masked_logits(pr[0, 0], t(o["prio_mask"], torch.bool)[0, 0])
        if self.greedy:
            k, p = int(mv.argmax()), int(pr.argmax())
        else:
            k = int(torch.distributions.Categorical(logits=mv).sample())
            p = int(torch.distributions.Categorical(logits=pr).sample())
        tid = o["tids"][p] if p < len(o["tids"]) else None
        return k, tid
