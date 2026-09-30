#!/usr/bin/env python
"""Train a normalized-space behavior-FQE critic for in-loop guidance.

The post-action CCR path uses raw observations/actions. In-loop guidance operates
on CoFlow's normalized generated states, so the critic must be trained after the
same dataset normalizer used by the frozen policy.
"""
import argparse
import glob
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn


EP_LEN_MPE = 25


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


@torch.no_grad()
def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.mul_(1.0 - tau).add_(p, alpha=tau)


def load_model_normalizer(log_dir):
    cache_path = os.path.join(log_dir, "_zgw_norm_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        print(f"[normalizer] loaded cache {cache_path}")
        return obj["normalizer"]
    with open(os.path.join(log_dir, "dataset_config.pkl"), "rb") as f:
        dataset_config = pickle.load(f)
    dataset = dataset_config()
    norm = dataset.normalizer
    del dataset
    print("[normalizer] rebuilt from dataset_config")
    return norm


def maybe_limit(obs, act, rew, idx, max_samples):
    if not max_samples or len(obs) <= max_samples:
        return obs, act, rew, idx
    # Keep a contiguous prefix so that idx + 1 remains the true next transition.
    limit = int(max_samples)
    idx = idx[idx < limit - 1]
    return obs[:limit], act[:limit], rew[:limit], idx.astype(np.int64)


def load_mamujoco(data_dir):
    dd = Path(data_dir)
    obs = np.load(dd / "obs.npy")
    act = np.load(dd / "actions.npy")
    rew = np.load(dd / "rewards.npy")
    path_lengths = np.load(dd / "path_lengths.npy")
    rew_joint = rew.sum(axis=1) if rew.ndim > 1 else rew
    starts = np.cumsum(np.r_[0, path_lengths[:-1]])
    valid = []
    for start, length in zip(starts, path_lengths):
        if length > 1:
            valid.append(np.arange(start, start + length - 1, dtype=np.int64))
    idx = np.concatenate(valid)
    return obs.astype(np.float32), act.astype(np.float32), rew_joint.astype(np.float32), idx


def load_mpe(data_root, split):
    seeds = sorted(glob.glob(os.path.join(data_root, split, "seed_*_data")))
    if not seeds:
        raise FileNotFoundError(f"No MPE seed data under {data_root}/{split}")
    n_files = len(glob.glob(os.path.join(seeds[0], "obs_*.npy")))
    dims = [
        np.load(os.path.join(seeds[0], f"obs_{i}.npy"), mmap_mode="r").shape[1]
        for i in range(n_files)
    ]
    modal = Counter(dims).most_common(1)[0][0]
    agents = [i for i in range(n_files) if dims[i] == modal]
    obs_l, act_l, rew_l = [], [], []
    for sd in seeds:
        a_obs = [np.load(os.path.join(sd, f"obs_{i}.npy")) for i in agents]
        a_act = [np.load(os.path.join(sd, f"acs_{i}.npy")) for i in agents]
        a_rew = [np.load(os.path.join(sd, f"rews_{i}.npy")) for i in agents]
        obs_l.append(np.stack(a_obs, axis=1))
        act_l.append(np.stack(a_act, axis=1))
        rew_l.append(np.sum(np.stack(a_rew, axis=1), axis=1))
    obs = np.concatenate(obs_l).astype(np.float32)
    act = np.concatenate(act_l).astype(np.float32)
    rew = np.concatenate(rew_l).astype(np.float32)
    idx = np.arange(len(obs) - 1, dtype=np.int64)
    idx = idx[(idx % EP_LEN_MPE) != (EP_LEN_MPE - 1)]
    print(f"[mpe] obs dims={dims} agents={agents} obs={obs.shape} act={act.shape}")
    return obs, act, rew, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["mamujoco", "mpe"], required=True)
    ap.add_argument("--data_dir")
    ap.add_argument("--data_root")
    ap.add_argument("--split")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--max_samples", type=int)
    args = ap.parse_args()

    if args.domain == "mamujoco":
        obs, act, rew, idx = load_mamujoco(args.data_dir)
    else:
        obs, act, rew, idx = load_mpe(args.data_root, args.split)
    obs, act, rew, idx = maybe_limit(obs, act, rew, idx, args.max_samples)
    print(f"[raw] obs={obs.shape} act={act.shape} idx={len(idx)} rew_mean={rew.mean():.4f}")

    norm = load_model_normalizer(args.log_dir)
    obs_n = norm.normalize(obs, "observations").astype(np.float32).reshape(len(obs), -1)
    act_n = norm.normalize(act, "actions").astype(np.float32).reshape(len(act), -1)
    obs_dim, act_dim = obs_n.shape[1], act_n.shape[1]
    print(f"[norm] obs_dim={obs_dim} act_dim={act_dim} "
          f"obs=[{obs_n.min():.3f},{obs_n.max():.3f}] act=[{act_n.min():.3f},{act_n.max():.3f}]")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_t = torch.from_numpy(obs_n)
    act_t = torch.from_numpy(act_n)
    rew_t = torch.from_numpy(rew.astype(np.float32))
    idx_t = torch.from_numpy(idx.astype(np.int64))

    critic = Critic(obs_dim, act_dim, args.hidden).to(device)
    target = Critic(obs_dim, act_dim, args.hidden).to(device)
    target.load_state_dict(critic.state_dict())
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)

    config = dict(vars(args))
    config.update({
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "num_transitions": int(len(idx)),
        "normalized": True,
        "normalizer": "policy_dataset_config",
        "device": str(device),
    })
    (out / "config.json").write_text(json.dumps(config, indent=2))
    metrics_path = out / "metrics.jsonl"

    gen = torch.Generator().manual_seed(0)
    running = []
    n = len(idx_t)
    for step in range(1, args.steps + 1):
        sel = idx_t[torch.randint(0, n, (args.batch_size,), generator=gen)]
        o = obs_t[sel].to(device, non_blocking=True)
        a = act_t[sel].to(device, non_blocking=True)
        r = rew_t[sel].to(device, non_blocking=True)
        no = obs_t[sel + 1].to(device, non_blocking=True)
        na = act_t[sel + 1].to(device, non_blocking=True)
        with torch.no_grad():
            td = r + args.gamma * target(no, na)
        q = critic(o, a)
        loss = torch.nn.functional.mse_loss(q, td)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        opt.step()
        soft_update(target, critic, 0.005)
        running.append(float(loss.detach().cpu()))
        if step % args.log_every == 0:
            row = {
                "step": step,
                "loss": float(np.mean(running[-args.log_every:])),
                "q_mean": float(q.detach().mean().cpu()),
                "target_mean": float(td.detach().mean().cpu()),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {"step": step, "critic": critic.state_dict(),
                 "target": target.state_dict(), "config": config},
                out / f"critic_step_{step}.pt",
            )


if __name__ == "__main__":
    main()
