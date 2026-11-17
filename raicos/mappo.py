"""Recurrent MAPPO trainer with PopArt (Sec. 4.9, 5.6).

Centralised training / decentralised execution: a parameter-shared recurrent
actor acts on local observations; a centralised critic sees simulator-global
tokens during training only. GAE(0.99, 0.95), clipped PPO (0.2), 32-step
truncated recurrent sequences, PopArt value normalisation.

The article trains 5 seeds x 10 M environment steps in 32 parallel
environments and selects checkpoints by validation continuity among
byte-feasible candidates; `scripts/train_mappo.py` exposes the same knobs.
"""
from __future__ import annotations

import copy
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as Fnn

from .config import ExperimentConfig
from .env import Episode
from .policy import Actor, Critic, global_state, masked_logits, observe


def _stack(obs_list, key, dtype=torch.float32):
    return torch.as_tensor(np.stack([o[key] for o in obs_list]), dtype=dtype)


class MAPPOTrainer:
    def __init__(self, cfg: ExperimentConfig, n_envs: int = 2, seq_len: int = 32, lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2, ent: float = 0.01,
                 epochs: int = 4, seed: int = 0):
        torch.manual_seed(seed)
        self.cfg, self.n_envs, self.T = cfg, n_envs, seq_len
        self.actor, self.critic = Actor(), Critic()
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma, self.lam, self.clip, self.ent, self.epochs = gamma, lam, clip, ent, epochs
        self.envs: List[Episode] = []
        self.seed = seed
        self.env_steps = 0
        self._reset_all()

    def _new_env(self, k: int) -> Episode:
        c = copy.deepcopy(self.cfg)
        c.seed = self.seed * 1000 + k + int(time.time()) % 1000
        return Episode(c)

    def _reset_all(self):
        self.envs = [self._new_env(k) for k in range(self.n_envs)]
        nU = self.cfg.scenario.num_uavs
        self.h = [torch.zeros(1, nU, 256) for _ in self.envs]

    @torch.no_grad()
    def rollout(self):
        buf = []
        for e, ep in enumerate(self.envs):
            nU = len(ep.agents)
            h0 = self.h[e].clone()
            steps = []
            for t in range(self.T):
                obs = [observe(ep, a, a.local_time(ep.t)) for a in ep.agents]
                tok, tm = _stack(obs, "tokens")[:, None], _stack(obs, "tmask", torch.bool)[:, None]
                uv, ln = _stack(obs, "uav")[:, None], _stack(obs, "link")[:, None]
                mv, pr, self.h[e] = self.actor(tok, tm, uv, ln, self.h[e])
                mv = masked_logits(mv[:, 0], _stack(obs, "move_mask", torch.bool))
                pr = masked_logits(pr[:, 0], _stack(obs, "prio_mask", torch.bool))
                dm, dp = torch.distributions.Categorical(logits=mv), torch.distributions.Categorical(logits=pr)
                am, ap = dm.sample(), dp.sample()
                logp = dm.log_prob(am) + dp.log_prob(ap)
                g = torch.as_tensor(global_state(ep))[None]
                v = self.critic.popart.denormalize(self.critic(g))
                acts = {}
                for i, a in enumerate(ep.agents):
                    p = int(ap[i])
                    acts[a.uid] = (int(am[i]), obs[i]["tids"][p] if p < len(obs[i]["tids"]) else None)
                r = ep.step(acts)
                self.env_steps += 1
                done = ep.t >= ep.cfg.scenario.duration_s
                steps.append(dict(obs=obs, am=am, ap=ap, logp=logp, v=float(v), r=r, done=done, g=g[0]))
                if done:
                    self.envs[e] = self._new_env(e)
                    ep = self.envs[e]
                    self.h[e] = torch.zeros(1, nU, 256)
            g_last = torch.as_tensor(global_state(ep))[None]
            v_last = float(self.critic.popart.denormalize(self.critic(g_last)))
            adv, ret, gae = [], [], 0.0
            for t in reversed(range(self.T)):
                nv = v_last if t == self.T - 1 else steps[t + 1]["v"]
                nd = 0.0 if steps[t]["done"] else 1.0
                delta = steps[t]["r"] + self.gamma * nv * nd - steps[t]["v"]
                gae = delta + self.gamma * self.lam * nd * gae
                adv.insert(0, gae)
                ret.insert(0, gae + steps[t]["v"])
            buf.append((h0, steps, torch.tensor(adv, dtype=torch.float32), torch.tensor(ret, dtype=torch.float32)))
        return buf

    def update(self, buf):
        stats = {}
        for _ in range(self.epochs):
            loss_tot = 0.0
            for h0, steps, adv, ret in buf:
                obs_seq = [s["obs"] for s in steps]
                nU = len(obs_seq[0])
                tok = torch.stack([_stack(o, "tokens") for o in obs_seq], 1)
                tm = torch.stack([_stack(o, "tmask", torch.bool) for o in obs_seq], 1)
                uv = torch.stack([_stack(o, "uav") for o in obs_seq], 1)
                ln = torch.stack([_stack(o, "link") for o in obs_seq], 1)
                mm = torch.stack([_stack(o, "move_mask", torch.bool) for o in obs_seq], 1)
                pm = torch.stack([_stack(o, "prio_mask", torch.bool) for o in obs_seq], 1)
                mv, pr, _ = self.actor(tok, tm, uv, ln, h0)
                dm = torch.distributions.Categorical(logits=masked_logits(mv, mm))
                dp = torch.distributions.Categorical(logits=masked_logits(pr, pm))
                am = torch.stack([s["am"] for s in steps], 1)
                ap = torch.stack([s["ap"] for s in steps], 1)
                logp = dm.log_prob(am) + dp.log_prob(ap)
                old = torch.stack([s["logp"] for s in steps], 1)
                A = ((adv - adv.mean()) / (adv.std() + 1e-8))[None].expand(nU, -1)
                ratio = torch.exp(logp - old)
                pol = -torch.min(ratio * A, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A).mean()
                entropy = (dm.entropy() + dp.entropy()).mean()
                g = torch.stack([s["g"] for s in steps])
                self.critic.popart.update(ret)
                vpred = self.critic(g)
                vloss = Fnn.mse_loss(vpred, self.critic.popart.normalize(ret))
                loss = pol + 0.5 * vloss - self.ent * entropy
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 10.0)
                self.opt.step()
                loss_tot += loss.item()
                stats = {"policy": float(pol), "value": float(vloss), "entropy": float(entropy)}
        return stats

    def train(self, total_steps: int, log_every: int = 1, ckpt: str = ""):
        it = 0
        while self.env_steps < total_steps:
            buf = self.rollout()
            st = self.update(buf)
            it += 1
            mean_r = float(np.mean([s["r"] for _, steps, _, _ in buf for s in steps]))
            if it % log_every == 0:
                print(f"iter {it}  env-steps {self.env_steps}  reward {mean_r:.3f}  {st}", flush=True)
            if ckpt:
                torch.save(self.actor.state_dict(), ckpt)
        return self.actor
