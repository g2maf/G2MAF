#!/usr/bin/env python
"""Joint behavior-FQE critic for SMAC (discrete actions, one-hot encoded).

Q(obs, one-hot joint action). Enables discrete G2MAF refinement: nudge action LOGITS
along d softmax(logits) -> Q. Same Critic class/save-format as train_fqe_critic_2ant
so the evaluator can load it the same way (obs_dim / act_dim read from saved config).

SMAC 3m: 3 agents, obs 33/agent (99 joint), n_actions=9 -> one-hot joint act 27,
reward = sum over agents, episodes delimited by path_lengths.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


@torch.no_grad()
def soft_update(t, s, tau):
    for tp, p in zip(t.parameters(), s.parameters()):
        tp.mul_(1 - tau).add_(p, alpha=tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_actions", type=int, default=9)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=5000)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dd = Path(args.data_dir)
    obs = np.load(dd / "obs.npy")          # (N, n_agents, obs_dim)
    act = np.load(dd / "actions.npy")      # (N, n_agents) int
    rew = np.load(dd / "rewards.npy")      # (N, n_agents)
    path_lengths = np.load(dd / "path_lengths.npy")
    N, n_agents = obs.shape[0], obs.shape[1]
    A = args.n_actions

    obs_j = obs.reshape(N, -1).astype(np.float32)                       # (N, 99)
    onehot = np.eye(A, dtype=np.float32)[act]                           # (N, n_agents, A)
    act_j = onehot.reshape(N, -1).astype(np.float32)                    # (N, 27)
    rew_j = (rew.sum(axis=1) if rew.ndim > 1 else rew).astype(np.float32)
    obs_dim, act_dim = obs_j.shape[1], act_j.shape[1]

    starts = np.cumsum(np.r_[0, path_lengths[:-1]])
    valid = [np.arange(s, s + L - 1, dtype=np.int64) for s, L in zip(starts, path_lengths) if L > 1]
    idx = np.concatenate(valid)
    print(f"[data] N={N} n_agents={n_agents} A={A} obs_dim={obs_dim} act_dim={act_dim} "
          f"transitions={len(idx)} rew[min,mean,max]=[{rew_j.min():.2f},{rew_j.mean():.2f},{rew_j.max():.2f}]")

    obs_t = torch.from_numpy(obs_j); act_t = torch.from_numpy(act_j); rew_t = torch.from_numpy(rew_j)
    idx_t = torch.from_numpy(idx)
    critic = Critic(obs_dim, act_dim, args.hidden).to(device)
    target = Critic(obs_dim, act_dim, args.hidden).to(device)
    target.load_state_dict(critic.state_dict())
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)

    config = dict(vars(args)); config.update(
        {"obs_dim": obs_dim, "act_dim": act_dim, "n_agents": n_agents, "n_actions": A,
         "num_transitions": int(len(idx)), "device": str(device), "env": "smac", "discrete": True})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    mp = out / "metrics.jsonl"
    g = torch.Generator().manual_seed(0); n = len(idx_t); run = []
    for step in range(1, args.steps + 1):
        sel = idx_t[torch.randint(0, n, (args.batch_size,), generator=g)]
        o = obs_t[sel].to(device); a = act_t[sel].to(device); r = rew_t[sel].to(device)
        no = obs_t[sel + 1].to(device); na = act_t[sel + 1].to(device)
        with torch.no_grad():
            td = r + args.gamma * target(no, na)
        q = critic(o, a); loss = torch.nn.functional.mse_loss(q, td)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0); opt.step()
        soft_update(target, critic, 0.005); run.append(float(loss.detach().cpu()))
        if step % args.log_every == 0:
            row = {"step": step, "loss": float(np.mean(run[-args.log_every:])),
                   "q_mean": float(q.detach().mean().cpu()), "target_mean": float(td.detach().mean().cpu())}
            with mp.open("a") as f: f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"step": step, "critic": critic.state_dict(),
                        "target": target.state_dict(), "config": config}, out / f"critic_step_{step}.pt")


if __name__ == "__main__":
    main()
