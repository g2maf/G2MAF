#!/usr/bin/env python
"""Joint behavior-FQE critic for MPE simple_spread (OMAR per-agent layout).

RAW obs/act (no normalization) -> usable by the post-action refinement path
(evaluator g2maf_action_step), same Critic class/save-format as train_fqe_critic_2ant.
MPE: 3 agents, obs 18/agent (54 joint), act 2/agent (6 joint), fixed 25-step episodes
(dones are all zero), reward = sum over agents.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import torch
from torch import nn

EP_LEN = 25


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


def load_mpe(data_root, split):
    from collections import Counter
    seeds = sorted(glob.glob(os.path.join(data_root, split, "seed_*_data")))
    # The cooperative joint policy uses homogeneous agents that share one obs dim;
    # predator-prey scenarios (simple_tag/world) also store a scripted adversary
    # (e.g. the prey) with a DIFFERENT obs dim, which is NOT part of the policy.
    # Select the agents whose per-agent obs dim equals the most common obs dim.
    n_files = len(glob.glob(os.path.join(seeds[0], "obs_*.npy")))
    dims = [np.load(os.path.join(seeds[0], f"obs_{i}.npy"), mmap_mode="r").shape[1] for i in range(n_files)]
    modal = Counter(dims).most_common(1)[0][0]
    agents = [i for i in range(n_files) if dims[i] == modal]
    n_agents = len(agents)
    print(f"[load_mpe] obs dims={dims} -> cooperative agents={agents} (n_agents={n_agents})")
    obs_l, act_l, rew_l = [], [], []
    for sd in seeds:
        a_obs = [np.load(os.path.join(sd, f"obs_{i}.npy")) for i in agents]
        a_act = [np.load(os.path.join(sd, f"acs_{i}.npy")) for i in agents]
        a_rew = [np.load(os.path.join(sd, f"rews_{i}.npy")) for i in agents]
        obs_l.append(np.stack(a_obs, axis=1).reshape(len(a_obs[0]), -1))   # (N,54)
        act_l.append(np.stack(a_act, axis=1).reshape(len(a_act[0]), -1))   # (N,6)
        rew_l.append(np.sum(np.stack(a_rew, axis=1), axis=1))              # (N,) joint
    return (np.concatenate(obs_l).astype(np.float32),
            np.concatenate(act_l).astype(np.float32),
            np.concatenate(rew_l).astype(np.float32),
            n_agents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="expert")
    ap.add_argument("--out_dir", required=True)
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

    obs, act, rew, n_agents = load_mpe(args.data_root, args.split)
    N = len(obs); obs_dim, act_dim = obs.shape[1], act.shape[1]
    # valid transitions: not the last step of a 25-step episode
    idx = np.arange(N - 1)
    idx = idx[(idx % EP_LEN) != (EP_LEN - 1)]
    print(f"[data] N={N} obs_dim={obs_dim} act_dim={act_dim} transitions={len(idx)} "
          f"rew[min,mean,max]=[{rew.min():.2f},{rew.mean():.2f},{rew.max():.2f}]")

    obs_t = torch.from_numpy(obs); act_t = torch.from_numpy(act); rew_t = torch.from_numpy(rew)
    idx_t = torch.from_numpy(idx.astype(np.int64))

    critic = Critic(obs_dim, act_dim, args.hidden).to(device)
    target = Critic(obs_dim, act_dim, args.hidden).to(device)
    target.load_state_dict(critic.state_dict())
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)

    config = dict(vars(args)); config.update(
        {"obs_dim": obs_dim, "act_dim": act_dim, "n_agents": n_agents,
         "num_transitions": int(len(idx)), "device": str(device), "env": f"mpe_{Path(args.data_root).name}"})
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
                        "target": target.state_dict(), "config": config},
                       out / f"critic_step_{step}.pt")


if __name__ == "__main__":
    main()
