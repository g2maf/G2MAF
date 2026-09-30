#!/usr/bin/env python
"""
Optimized gap-filling evaluator.

Key optimization vs eval_all_gaps.py:
- Groups jobs by seed_path and loads model+dataset+normalizer ONCE per seed.
- For each step (1,2,3,4,5) on the same seed, only overrides max_denoising_steps
  and runs a fresh rollout. Dataset is NOT reloaded.
- This gives ~4-5x speedup when a seed needs multiple step evaluations.

Usage:
    python eval_all_gaps_v2.py -g 0 --envs mpe --num_eval 10
    python eval_all_gaps_v2.py -g 0 --envs smac --steps 3,4
"""
import os
import sys
import csv
import json
import gc
import argparse
import traceback
import pickle
import importlib
from collections import deque, defaultdict
from copy import copy

import numpy as np
import einops

# Disable wandb completely for eval — we don't need it
os.environ["WANDB_MODE"] = "disabled"

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffuser.utils as utils
from diffuser.utils.arrays import to_device, to_np, to_torch
from diffuser.utils.launcher_util import build_config_from_dict
from run_scripts.eval_inference_steps import _cleanup_sc2

import matplotlib
matplotlib.use("Agg")
from ml_logger import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_BASES = {
    "mpe": ["logs/ma_imf_mpe_disp", "logs/ma_meanflow_mpe"],
    "smac": ["logs/ma_imf_smac_disp", "logs/ma_meanflow_smac"],
    "mamujoco": ["logs/ma_imf_mamujoco_disp", "logs/ma_meanflow_mamujoco"],
}


def find_best_checkpoint(seed_path):
    eval_csv = os.path.join(seed_path, "results", "evaluation_history.csv")
    ckpt_dir = os.path.join(seed_path, "checkpoint")
    ckpt_steps = set()
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith("state_") and f.endswith(".pt"):
                try:
                    ckpt_steps.add(int(f.replace("state_", "").replace(".pt", "")))
                except ValueError:
                    pass
    best_step, best_reward = None, -float("inf")
    if os.path.exists(eval_csv):
        with open(eval_csv) as f:
            for row in csv.DictReader(f):
                try:
                    r = float(row["average_ep_reward_mean"])
                    s = int(row["step"])
                    if r > best_reward and s in ckpt_steps:
                        best_reward, best_step = r, s
                except (ValueError, KeyError):
                    pass
    if best_step is None and ckpt_steps:
        best_step = max(ckpt_steps)
    return best_step, best_reward


def scan_all_jobs(envs, steps, num_eval, force=False, require_step1=False, min_ckpt=0):
    jobs = []
    for env in envs:
        for log_base in LOG_BASES.get(env, []):
            base = os.path.join(PROJECT_ROOT, log_base)
            if not os.path.exists(base):
                continue
            for tq in sorted(os.listdir(base)):
                tq_path = os.path.join(base, tq)
                if not os.path.isdir(tq_path):
                    continue
                for cfg in sorted(os.listdir(tq_path)):
                    cfg_path = os.path.join(tq_path, cfg)
                    if not os.path.isdir(cfg_path):
                        continue
                    for seed in sorted(os.listdir(cfg_path)):
                        seed_path = os.path.join(cfg_path, seed)
                        params_file = os.path.join(seed_path, "parameters.pkl")
                        if not os.path.exists(params_file):
                            continue
                        best_step, best_r = find_best_checkpoint(seed_path)
                        if best_step is None:
                            continue
                        if best_step < min_ckpt:
                            continue
                        if require_step1:
                            step1_dir = os.path.join(seed_path, "results_1step")
                            step1_file = os.path.join(step1_dir, f"step_{best_step}-ep_{num_eval}.json")
                            if not os.path.exists(step1_file):
                                continue
                        for n_step in steps:
                            res_dir = os.path.join(seed_path, f"results_{n_step}step")
                            res_file = os.path.join(res_dir, f"step_{best_step}-ep_{num_eval}.json")
                            if os.path.exists(res_file) and not force:
                                continue
                            jobs.append({
                                "env": env,
                                "tq": tq,
                                "cfg": cfg,
                                "seed_path": seed_path,
                                "best_step": best_step,
                                "best_reward": best_r,
                                "n_step": n_step,
                                "res_file": res_file,
                                "res_dir": res_dir,
                            })
    return jobs


def build_setup(seed_path, load_step, device, num_eval):
    """Load model + dataset + configs ONCE for a given seed_path. Returns a setup dict."""
    with open(os.path.join(seed_path, "parameters.pkl"), "rb") as f:
        params = pickle.load(f)
    Config = build_config_from_dict(params["Config"])
    Config.device = device
    Config.num_eval = num_eval
    Config.num_envs = num_eval

    logger.configure(seed_path)

    with open(os.path.join(seed_path, "model_config.pkl"), "rb") as f:
        model_config = pickle.load(f)
    with open(os.path.join(seed_path, "diffusion_config.pkl"), "rb") as f:
        diffusion_config = pickle.load(f)
    with open(os.path.join(seed_path, "trainer_config.pkl"), "rb") as f:
        trainer_config = pickle.load(f)
    with open(os.path.join(seed_path, "dataset_config.pkl"), "rb") as f:
        dataset_config = pickle.load(f)
    with open(os.path.join(seed_path, "render_config.pkl"), "rb") as f:
        render_config = pickle.load(f)

    # Build dataset ONCE to extract normalizer + mask_generator
    dataset = dataset_config()
    normalizer = dataset.normalizer
    mask_generator = dataset.mask_generator
    del dataset
    gc.collect()

    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, None, renderer)

    ckpt_path = os.path.join(seed_path, f"checkpoint/state_{load_step}.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict["model"] = {k: v for k, v in state_dict["model"].items()
                           if "value_diffusion_model." not in k}
    state_dict["ema"] = {k: v for k, v in state_dict["ema"].items()
                         if "value_diffusion_model." not in k}
    trainer.model.load_state_dict(state_dict["model"])
    trainer.ema_model.load_state_dict(state_dict["ema"])

    env_mod_name = {
        "d4rl": "diffuser.datasets.d4rl",
        "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
        "mamujoco": "diffuser.datasets.mamujoco",
        "mpe": "diffuser.datasets.mpe",
        "smac": "diffuser.datasets.smac_env",
        "smacv2": "diffuser.datasets.smacv2_env",
    }[Config.env_type]
    env_mod = importlib.import_module(env_mod_name)

    discrete_action = Config.env_type in ("smac", "smacv2")

    return {
        "Config": Config,
        "trainer": trainer,
        "diffusion": diffusion,
        "model": model,
        "renderer": renderer,
        "normalizer": normalizer,
        "mask_generator": mask_generator,
        "env_mod": env_mod,
        "discrete_action": discrete_action,
        "device": device,
    }


def run_rollout(setup, inference_steps, num_eval):
    """Run one rollout pass with the given inference_steps. Returns metrics dict."""
    Config = setup["Config"]
    trainer = setup["trainer"]
    normalizer = setup["normalizer"]
    mask_generator = setup["mask_generator"]
    env_mod = setup["env_mod"]
    device = setup["device"]
    discrete_action = setup["discrete_action"]

    # Override inference steps
    if hasattr(trainer.ema_model, "max_denoising_steps"):
        trainer.ema_model.max_denoising_steps = inference_steps
    if hasattr(trainer.model, "max_denoising_steps"):
        trainer.model.max_denoising_steps = inference_steps

    env_list = [env_mod.load_environment(Config.dataset) for _ in range(num_eval)]

    observation_dim = normalizer.observation_dim
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1, Config.n_agents), device)
    env_ts = to_device(
        torch.arange(Config.horizon + Config.history_horizon) - Config.history_horizon,
        device,
    )
    env_ts = einops.repeat(env_ts, "t -> b t", b=num_eval)

    dones = [0] * num_eval
    ep_rewards = [np.zeros(Config.n_agents) for _ in range(num_eval)]
    ep_wins = np.zeros(num_eval) if discrete_action else None

    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)

    obs_queue = deque(maxlen=Config.history_horizon + 1)
    use_zero_padding = getattr(Config, "use_zero_padding", False)
    if use_zero_padding:
        obs_queue.extend([np.zeros_like(obs) for _ in range(Config.history_horizon)])
    else:
        normed_obs = normalizer.normalize(obs, "observations")
        obs_queue.extend([normed_obs for _ in range(Config.history_horizon)])

    max_path_length = getattr(Config, "max_path_length", 1000)
    decentralized = getattr(Config, "decentralized_execution", False)
    t = 0

    while sum(dones) < num_eval:
        obs_normed = normalizer.normalize(obs, "observations")
        obs_queue.append(obs_normed)
        obs_stacked = np.stack(list(obs_queue), axis=1)

        attention_masks = np.zeros(
            (obs_stacked.shape[0], Config.horizon + Config.history_horizon, Config.n_agents, 1)
        )
        attention_masks[:, Config.history_horizon:] = 1.0

        shape = (obs_stacked.shape[0], Config.horizon + Config.history_horizon, *obs_stacked.shape[-2:])

        if decentralized:
            joint_cond, joint_masks, joint_attn = [], [], []
            for a_idx in range(Config.n_agents):
                local_cond = np.zeros(shape, dtype=obs_stacked.dtype)
                local_cond[:, :Config.history_horizon + 1, a_idx] = obs_stacked[:, :, a_idx]
                agent_mask = np.zeros(Config.n_agents)
                agent_mask[a_idx] = 1.0
                local_masks = mask_generator(shape, agent_mask)
                local_attn = copy(attention_masks)
                local_attn[:, :Config.history_horizon, a_idx] = 1.0
                joint_cond.append(to_torch(local_cond, device=device))
                joint_masks.append(to_torch(local_masks, device=device))
                joint_attn.append(to_torch(local_attn, device=device))

            joint_cond = einops.rearrange(torch.stack(joint_cond, dim=1), "b a ... -> (b a) ...")
            joint_masks = einops.rearrange(torch.stack(joint_masks, dim=1), "b a ... -> (b a) ...")
            joint_attn = einops.rearrange(torch.stack(joint_attn, dim=1), "b a ... -> (b a) ...")

            conditions = {"x": joint_cond, "masks": joint_masks}
            rets = einops.repeat(returns, "b ... -> (b a) ...", a=Config.n_agents)
            ets = einops.repeat(env_ts, "b ... -> (b a) ...", a=Config.n_agents)

            with torch.no_grad():
                joint_samples = trainer.ema_model.conditional_sample(
                    conditions, returns=rets, env_ts=ets, attention_masks=joint_attn, verbose=False
                )
            joint_samples = einops.rearrange(joint_samples, "(b a) ... -> b a ...", a=Config.n_agents)
            samples = torch.stack([joint_samples[:, a, ..., a, :] for a in range(Config.n_agents)], dim=-2)
        else:
            cond_traj = np.zeros(shape, dtype=obs_stacked.dtype)
            cond_traj[:, :Config.history_horizon + 1] = obs_stacked
            agent_mask = np.ones(Config.n_agents)
            cond_masks = mask_generator(shape, agent_mask)
            conditions = {"x": to_torch(cond_traj, device=device), "masks": to_torch(cond_masks, device=device)}
            attn_m = copy(attention_masks)
            attn_m[:, :Config.history_horizon] = 1.0
            attn_m = to_torch(attn_m, device=device)

            with torch.no_grad():
                samples = trainer.ema_model.conditional_sample(
                    conditions, returns=returns, env_ts=env_ts, attention_masks=attn_m, verbose=False
                )

        samples = samples[:, Config.history_horizon:]

        obs_comb = torch.cat([samples[:, 0, :, :], samples[:, 1, :, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, Config.n_agents, 2 * observation_dim)

        share_inv = getattr(Config, "share_inv", True)
        joint_inv = getattr(Config, "joint_inv", False)
        with torch.no_grad():
            if joint_inv:
                actions = trainer.ema_model.inv_model(
                    obs_comb.reshape(obs_comb.shape[0], -1)
                ).reshape(obs_comb.shape[0], obs_comb.shape[1], -1)
            elif share_inv:
                actions = trainer.ema_model.inv_model(obs_comb)
            else:
                actions = torch.stack(
                    [trainer.ema_model.inv_model[i](obs_comb[:, i]) for i in range(Config.n_agents)],
                    dim=1,
                )

        actions = to_np(actions)

        if discrete_action:
            legal_action = np.stack([env.get_legal_actions() for env in env_list], axis=0)
            actions[np.where(legal_action.astype(int) == 0)] = -np.inf
            actions = np.argmax(actions, axis=-1)
        else:
            actions = normalizer.unnormalize(actions, "actions")

        obs_list = []
        for i in range(num_eval):
            if dones[i]:
                obs_list.append(obs[i:i+1])
            else:
                this_obs, this_reward, this_done, this_info = env_list[i].step(actions[i])
                obs_list.append(this_obs[None])

                use_rtg = getattr(Config, "use_return_to_go", False)
                if use_rtg:
                    rtg = returns[i] * Config.returns_scale
                    reward_t = torch.tensor(this_reward, device=device, dtype=rtg.dtype).reshape(1, -1)
                    returns[i] = (rtg - reward_t) / Config.discount / Config.returns_scale

                if this_done.all() or t >= max_path_length - 1:
                    dones[i] = 1
                    ep_rewards[i] += this_reward
                    if "battle_won" in this_info:
                        ep_wins[i] = 1.0 if this_info["battle_won"] else 0.0
                else:
                    ep_rewards[i] += this_reward

        obs = np.concatenate(obs_list, axis=0)
        t += 1
        env_ts = env_ts + 1

    for env in env_list:
        try:
            env.close()
        except Exception:
            pass

    ep_rewards = np.array(ep_rewards)
    metrics = {
        "average_ep_reward": np.mean(ep_rewards, axis=0).tolist(),
        "std_ep_reward": np.std(ep_rewards, axis=0).tolist(),
    }
    if discrete_action:
        metrics["win_rate"] = float(np.mean(ep_wins))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument("--envs", type=str, default="mpe")
    parser.add_argument("--steps", type=str, default="1,2,3,4,5")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_patterns", type=str, default="")
    parser.add_argument("--include_patterns", type=str, default="", help="only keep seed_paths matching ANY of these substrings")
    parser.add_argument("--limit", type=int, default=0, help="process at most N seed_paths (0=unlimited)")
    parser.add_argument("--require_step1", action="store_true", help="only process seed_paths that already have step_1 results")
    parser.add_argument("--min_ckpt", type=int, default=0, help="skip seed_paths whose best_ckpt < this value")
    args = parser.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    skip_patterns = [p.strip() for p in args.skip_patterns.split(",") if p.strip()]
    include_patterns = [p.strip() for p in args.include_patterns.split(",") if p.strip()]

    # Pin this process to the requested GPU. After this, the visible device is
    # always cuda:0 inside the process, regardless of which physical GPU it is.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    all_jobs = scan_all_jobs(envs, steps, args.num_eval, args.force,
                             require_step1=args.require_step1, min_ckpt=args.min_ckpt)
    if skip_patterns:
        all_jobs = [j for j in all_jobs if not any(p in j["seed_path"] for p in skip_patterns)]
    if include_patterns:
        all_jobs = [j for j in all_jobs if any(p in j["seed_path"] for p in include_patterns)]

    # Group by seed_path
    by_seed = defaultdict(list)
    for j in all_jobs:
        by_seed[j["seed_path"]].append(j)

    seed_paths = list(by_seed.keys())
    if args.limit > 0:
        seed_paths = seed_paths[:args.limit]

    total_jobs = sum(len(by_seed[sp]) for sp in seed_paths)
    print(f"Found {total_jobs} jobs across {len(seed_paths)} seed_paths\n", flush=True)

    if args.dry_run:
        for sp in seed_paths:
            jobs = by_seed[sp]
            print(f"  {jobs[0]['env']}/{jobs[0]['tq']}  {jobs[0]['cfg']}")
            print(f"    steps: {[j['n_step'] for j in jobs]}  best_ckpt={jobs[0]['best_step']}")
        return

    n_done = 0
    n_fail = 0
    n_seeds_done = 0
    for sp_idx, sp in enumerate(seed_paths, 1):
        jobs = by_seed[sp]
        first = jobs[0]
        print(f"\n{'='*70}", flush=True)
        print(f"[seed {sp_idx}/{len(seed_paths)}] {first['env']}/{first['tq']}  {first['cfg']}", flush=True)
        print(f"  best_step={first['best_step']}  reward_full={first['best_reward']:.1f}", flush=True)
        print(f"  steps to run: {[j['n_step'] for j in jobs]}", flush=True)

        try:
            setup = build_setup(sp, first["best_step"], device, args.num_eval)
        except Exception as e:
            print(f"  SETUP FAILED: {e}", flush=True)
            traceback.print_exc()
            n_fail += len(jobs)
            continue

        for job in jobs:
            try:
                metrics = run_rollout(setup, job["n_step"], args.num_eval)
                os.makedirs(job["res_dir"], exist_ok=True)
                with open(job["res_file"], "w") as f:
                    json.dump(metrics, f, indent=2)
                r = metrics.get("average_ep_reward", None)
                mean_r = float(np.mean(r)) if isinstance(r, list) else r
                print(f"  {job['n_step']}-step OK: mean = {mean_r:.1f}", flush=True)
                n_done += 1
            except Exception as e:
                print(f"  {job['n_step']}-step FAILED: {e}", flush=True)
                traceback.print_exc()
                n_fail += 1
            gc.collect()
            torch.cuda.empty_cache()

        # Release memory between seed_paths
        del setup
        gc.collect()
        torch.cuda.empty_cache()
        if first["env"] == "smac":
            _cleanup_sc2()
        n_seeds_done += 1

    print(f"\n=== DONE: {n_done} evals succeeded, {n_fail} failed, {n_seeds_done} seed_paths processed ===", flush=True)


if __name__ == "__main__":
    main()
