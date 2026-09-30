#!/usr/bin/env python
"""
Evaluate IMF models with different inference steps (1-step and 2-step).
Directly loads models in single process to override max_denoising_steps.

Usage:
    python eval_inference_steps.py -g 0 --num_eval 10 --steps 1,2
    python eval_inference_steps.py -g 0 --num_eval 10 --steps 1 --env mamujoco
"""

import argparse
import csv
import gc
import importlib
import json
import os
import pickle
import sys
import traceback
from collections import deque
from copy import deepcopy, copy

import numpy as np
import einops
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffuser.utils as utils
from diffuser.utils.arrays import to_device, to_np, to_torch
from diffuser.utils.launcher_util import build_config_from_dict

import matplotlib
matplotlib.use('Agg')
from ml_logger import logger

import atexit
import signal
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cleanup_sc2():
    """Kill any orphan SC2 processes spawned by this evaluation."""
    try:
        result = subprocess.run(
            ["pkill", "-f", "SC2_x64"], capture_output=True, timeout=5
        )
        # Also force gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

IMF_LOG_BASES = {
    "mamujoco": "logs/ma_imf_mamujoco_disp",
    "mpe": "logs/ma_imf_mpe_disp",
    "smac": "logs/ma_imf_smac_disp",
}

# MF (non-IMF) log bases for experiments where IMF wasn't trained
MF_LOG_BASES = {
    "mamujoco": "logs/ma_meanflow_mamujoco",
    "smac": "logs/ma_meanflow_smac",
}


def find_all_experiments(env_filter=None, task_filter=None, quality_filter=None, include_mf=False):
    """Find all IMF (and optionally MF) experiment directories with checkpoints."""
    experiments = []
    log_sources = dict(IMF_LOG_BASES)
    if include_mf:
        # Add MF sources but don't override IMF ones
        for k, v in MF_LOG_BASES.items():
            log_sources[k + "_mf"] = v
    for source_key, log_base in log_sources.items():
        env_name = source_key.replace("_mf", "")
        if env_filter and env_name != env_filter:
            continue
        base_path = os.path.join(PROJECT_ROOT, log_base)
        if not os.path.exists(base_path):
            continue
        for task_quality_dir in sorted(os.listdir(base_path)):
            task_quality_path = os.path.join(base_path, task_quality_dir)
            if not os.path.isdir(task_quality_path):
                continue
            parts = task_quality_dir.rsplit("-", 1)
            if len(parts) != 2:
                continue
            task, quality = parts
            if task_filter and task != task_filter:
                continue
            if quality_filter and quality.lower() != quality_filter.lower():
                continue
            for config_dir in os.listdir(task_quality_path):
                config_path = os.path.join(task_quality_path, config_dir)
                if not os.path.isdir(config_path):
                    continue
                for seed_dir in os.listdir(config_path):
                    seed_path = os.path.join(config_path, seed_dir)
                    ckpt_dir = os.path.join(seed_path, "checkpoint")
                    params_file = os.path.join(seed_path, "parameters.pkl")
                    if not os.path.exists(ckpt_dir) or not os.path.exists(params_file):
                        continue
                    ckpt_steps = []
                    for f in os.listdir(ckpt_dir):
                        if f.startswith("state_") and f.endswith(".pt"):
                            try:
                                ckpt_steps.append(int(f.replace("state_", "").replace(".pt", "")))
                            except ValueError:
                                continue
                    if not ckpt_steps:
                        continue
                    mode = "attn" if "ctde_False" in config_dir else "ctde"

                    # Find best checkpoint from existing 5-step evaluation
                    eval_csv = os.path.join(seed_path, "results", "evaluation_history.csv")
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
                                    continue
                    if best_step is None:
                        best_step = max(ckpt_steps)

                    experiments.append({
                        "env": env_name, "task": task, "quality": quality,
                        "mode": mode, "log_dir": seed_path,
                        "best_step": best_step, "reward_5step": best_reward,
                    })
    return experiments


def load_and_evaluate(log_dir, load_step, num_eval, inference_steps, device, override_test_ret=None):
    """Load model, override inference steps, and evaluate."""

    # Load configs
    with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
        params = pickle.load(f)
    Config = build_config_from_dict(params["Config"])
    Config.device = device
    Config.num_eval = num_eval
    Config.num_envs = num_eval

    # Override test_ret if specified
    if override_test_ret is not None:
        old_tr = Config.test_ret
        Config.test_ret = override_test_ret
        print(f"  Override test_ret: {old_tr} -> {override_test_ret}")

    logger.configure(log_dir)

    with open(os.path.join(log_dir, "model_config.pkl"), "rb") as f:
        model_config = pickle.load(f)
    with open(os.path.join(log_dir, "diffusion_config.pkl"), "rb") as f:
        diffusion_config = pickle.load(f)
    with open(os.path.join(log_dir, "trainer_config.pkl"), "rb") as f:
        trainer_config = pickle.load(f)
    with open(os.path.join(log_dir, "dataset_config.pkl"), "rb") as f:
        dataset_config = pickle.load(f)
    with open(os.path.join(log_dir, "render_config.pkl"), "rb") as f:
        render_config = pickle.load(f)

    # Override max_denoising_steps in diffusion config BEFORE creating model
    if hasattr(diffusion_config, '_dict'):
        diffusion_config._dict["max_denoising_steps"] = inference_steps
        print(f"  Set max_denoising_steps={inference_steps} in diffusion_config")

    # Build model
    dataset = dataset_config()
    normalizer = dataset.normalizer
    mask_generator = dataset.mask_generator
    del dataset
    gc.collect()

    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)

    # Double-check override worked — only change max_denoising_steps, NOT n_timesteps
    # n_timesteps controls discrete time resolution (e.g. 1000) and must stay unchanged
    if hasattr(diffusion, 'max_denoising_steps'):
        if diffusion.max_denoising_steps != inference_steps:
            diffusion.max_denoising_steps = inference_steps
            print(f"  Force-set max_denoising_steps={inference_steps} on diffusion model")

    trainer = trainer_config(diffusion, None, renderer)

    # Load checkpoint
    ckpt_path = os.path.join(log_dir, f"checkpoint/state_{load_step}.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict["model"] = {k: v for k, v in state_dict["model"].items()
                           if "value_diffusion_model." not in k}
    state_dict["ema"] = {k: v for k, v in state_dict["ema"].items()
                         if "value_diffusion_model." not in k}
    trainer.model.load_state_dict(state_dict["model"])
    trainer.ema_model.load_state_dict(state_dict["ema"])

    # Also override on ema_model after loading weights — keep n_timesteps unchanged
    if hasattr(trainer.ema_model, 'max_denoising_steps'):
        trainer.ema_model.max_denoising_steps = inference_steps

    # Load environments
    discrete_action = False
    env_mod_name = {
        "d4rl": "diffuser.datasets.d4rl",
        "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
        "mamujoco": "diffuser.datasets.mamujoco",
        "mpe": "diffuser.datasets.mpe",
        "smac": "diffuser.datasets.smac_env",
        "smacv2": "diffuser.datasets.smacv2_env",
    }[Config.env_type]
    env_mod = importlib.import_module(env_mod_name)
    env_list = [env_mod.load_environment(Config.dataset) for _ in range(num_eval)]
    if Config.env_type in ("smac", "smacv2"):
        discrete_action = True

    # Run evaluation
    observation_dim = normalizer.observation_dim
    episode_rewards = []
    episode_wins = []

    returns = to_device(Config.test_ret * torch.ones(num_eval, 1, Config.n_agents), device)
    env_ts = to_device(
        torch.arange(Config.horizon + Config.history_horizon) - Config.history_horizon,
        device,
    )
    env_ts = einops.repeat(env_ts, "t -> b t", b=num_eval)

    dones = [0] * num_eval
    ep_rewards = [np.zeros(Config.n_agents) for _ in range(num_eval)]
    if discrete_action:
        ep_wins = np.zeros(num_eval)

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

        # Generate samples
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

        # Extract actions via inverse dynamics
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

        # Step environments
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

    # Close environments
    for env in env_list:
        try:
            env.close()
        except:
            pass

    ep_rewards = np.array(ep_rewards)
    metrics = {
        "average_ep_reward": np.mean(ep_rewards, axis=0).tolist(),
        "std_ep_reward": np.std(ep_rewards, axis=0).tolist(),
    }
    if discrete_action:
        metrics["win_rate"] = float(np.mean(ep_wins))

    # Cleanup
    del trainer, diffusion, model, renderer
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument("--steps", type=str, default="1,2")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--quality", type=str, default=None)
    parser.add_argument("--include_mf", action="store_true",
                        help="Also include MF (non-IMF) experiments for missing entries")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_steps_list = [int(s) for s in args.steps.split(",")]

    # Register cleanup on exit and signals to prevent SC2 zombie processes
    atexit.register(_cleanup_sc2)
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup_sc2(), sys.exit(1)))
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup_sc2(), sys.exit(1)))

    print("Finding experiments...")
    experiments = find_all_experiments(args.env, args.task, args.quality,
                                       include_mf=getattr(args, 'include_mf', False))
    print(f"Found {len(experiments)} experiments\n")

    if not experiments:
        return

    summary_file = os.path.join(PROJECT_ROOT, "logs", "inference_steps_comparison.csv")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    results_all = []

    for idx, exp in enumerate(experiments):
        for n_steps in inference_steps_list:
            # Check if results already exist
            results_dir = os.path.join(exp["log_dir"], f"results_{n_steps}step")
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(results_dir, f"step_{exp['best_step']}-ep_{args.num_eval}.json")

            if os.path.exists(result_file):
                with open(result_file) as f:
                    metrics = json.load(f)
                reward = metrics.get("average_ep_reward", None)
                if isinstance(reward, list):
                    reward = np.mean(reward)
                print(f"[{idx+1}/{len(experiments)}] {exp['env']}/{exp['task']}-{exp['quality']}/{exp['mode']} "
                      f"{n_steps}-step: {reward:.2f} (cached)")
            else:
                print(f"\n[{idx+1}/{len(experiments)}] {exp['env']}/{exp['task']}-{exp['quality']}/{exp['mode']} "
                      f"| step={exp['best_step']} | inference_steps={n_steps}")
                try:
                    metrics = load_and_evaluate(
                        exp["log_dir"], exp["best_step"], args.num_eval, n_steps, device
                    )
                    with open(result_file, "w") as f:
                        json.dump(metrics, f, indent=2)

                    reward = metrics.get("average_ep_reward", None)
                    if isinstance(reward, list):
                        reward = np.mean(reward)
                    print(f"  -> {n_steps}-step reward: {reward:.2f}")
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()
                    reward = None
                finally:
                    # Force cleanup SC2 processes to prevent memory leaks
                    _cleanup_sc2()

        # Collect row
        row = {"env": exp["env"], "task": exp["task"], "quality": exp["quality"],
               "mode": exp["mode"], "reward_5step": exp["reward_5step"]}
        for n in inference_steps_list:
            rf = os.path.join(exp["log_dir"], f"results_{n}step",
                              f"step_{exp['best_step']}-ep_{args.num_eval}.json")
            if os.path.exists(rf):
                with open(rf) as f:
                    m = json.load(f)
                r = m.get("average_ep_reward", None)
                row[f"reward_{n}step"] = float(np.mean(r)) if isinstance(r, list) else r
        results_all.append(row)

    # Write summary
    if results_all:
        fieldnames = ["env", "task", "quality", "mode", "reward_5step"]
        for n in inference_steps_list:
            fieldnames.append(f"reward_{n}step")

        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_all)

        print(f"\n{'='*80}")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*80}")
        print(f"\n{'Env':<10} {'Task':<18} {'Q':<8} {'Mode':<6} {'5-step':<10}", end="")
        for n in inference_steps_list:
            print(f" {n}-step{'':>4}", end="")
        print("\n" + "-" * 80)
        for row in results_all:
            r5 = row.get('reward_5step', None)
            print(f"{row['env']:<10} {row['task']:<18} {row['quality']:<8} {row['mode']:<6} "
                  f"{r5:<10.1f}" if r5 else "", end="")
            for n in inference_steps_list:
                val = row.get(f"reward_{n}step", None)
                print(f" {val:<10.1f}" if val is not None else f" {'N/A':<10}", end="")
            print()


if __name__ == "__main__":
    main()
