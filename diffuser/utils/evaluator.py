from typing import Optional
import gc
import multiprocessing
import os
import pickle
import sys
import time
import importlib
from collections import deque
from copy import deepcopy, copy
from multiprocessing import Pipe, connection
from multiprocessing.context import Process

import numpy as np
import einops
import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.arrays import to_device, to_np, to_torch
from diffuser.utils.launcher_util import build_config_from_dict


class MADEvaluatorWorker(Process):
    def __init__(
        self,
        parent_remote: connection.Connection,
        child_remote: connection.Connection,
        queue: multiprocessing.Queue,
        verbose: bool = False,
    ):
        self.parent_remote = parent_remote
        self.p = child_remote
        self.queue = queue
        self.initialized = False
        self.verbose = verbose
        super().__init__()

    def _get_g2maf_guidance_fn(self):
        if self._g2maf_guidance_fn is not None:
            return self._g2maf_guidance_fn
        from diffuser.models.helpers import apply_conditioning
        Config = self.Config
        critic = self.g2maf_guidance_critic
        ema = self.trainer.ema_model
        normalize_grad = self.g2maf_guidance_norm
        mode = self.g2maf_guide_mode
        h = int(getattr(Config, "history_horizon", 0))

        def guidance_fn(x, cond):
            with torch.enable_grad():
                xg = x.detach().requires_grad_(True)
                xc = apply_conditioning(xg, cond)
                x_t = xc[:, :-1]
                x_t1 = xc[:, 1:]
                obs_comb = torch.cat([x_t, x_t1], dim=-1)
                b, Tm1, a = obs_comb.shape[0], obs_comb.shape[1], obs_comb.shape[2]
                if getattr(Config, "joint_inv", False):
                    acts = ema.inv_model(obs_comb.reshape(b, Tm1, -1)).reshape(b, Tm1, a, -1)
                elif getattr(Config, "share_inv", True):
                    acts = ema.inv_model(obs_comb)
                else:
                    acts = torch.stack(
                        [ema.inv_model[i](obs_comb[:, :, i]) for i in range(a)], dim=2)
                if mode == "mean":
                    o = x_t[:, h:].reshape(-1, x_t.shape[2] * x_t.shape[3])
                    ac = acts[:, h:].reshape(-1, acts.shape[2] * acts.shape[3])
                    q = critic(o, ac).reshape(b, -1).mean(dim=1)
                else:  # "first": the transition that is actually executed
                    o = x_t[:, h].reshape(b, -1)
                    ac = acts[:, h].reshape(b, -1)
                    q = critic(o, ac)
                grad = torch.autograd.grad(q.sum(), xg)[0]
            if normalize_grad:
                flat = grad.reshape(grad.shape[0], -1)
                grad = grad / (flat.norm(dim=1).view(-1, 1, 1, 1) + 1e-6)
            return grad.detach()

        self._g2maf_guidance_fn = guidance_fn
        return guidance_fn

    def _generate_samples(self, obs, returns, env_ts):
        Config = self.Config

        env_ts = env_ts.clone()
        env_ts[torch.where(env_ts < 0)] = Config.max_path_length
        env_ts[torch.where(env_ts >= Config.max_path_length)] = Config.max_path_length

        attention_masks = np.zeros(
            (obs.shape[0], Config.horizon + Config.history_horizon, Config.n_agents, 1)
        )
        attention_masks[:, Config.history_horizon :] = 1.0

        shape = (
            obs.shape[0],
            Config.horizon + Config.history_horizon,
            *obs.shape[-2:],
        )  # b t a f
        if Config.decentralized_execution:
            joint_cond_trajectories, joint_cond_masks, joint_attention_masks = (
                [],
                [],
                [],
            )
            for a_idx in range(Config.n_agents):
                local_cond_trajectories = np.zeros(shape, dtype=obs.dtype)
                local_cond_trajectories[:, : Config.history_horizon + 1, a_idx] = obs[
                    :, :, a_idx
                ]

                agent_mask = np.zeros(Config.n_agents)
                agent_mask[a_idx] = 1.0
                local_cond_masks = self.mask_generator(shape, agent_mask)

                local_attention_masks = copy(attention_masks)
                local_attention_masks[:, : Config.history_horizon, a_idx] = 1.0

                joint_cond_trajectories.append(
                    to_torch(local_cond_trajectories, device=Config.device)
                )
                joint_cond_masks.append(
                    to_torch(local_cond_masks, device=Config.device)
                )
                joint_attention_masks.append(
                    to_torch(local_attention_masks, device=Config.device)
                )

            joint_cond_trajectories = einops.rearrange(
                torch.stack(joint_cond_trajectories, dim=1), "b a ... -> (b a) ..."
            )
            joint_cond_masks = einops.rearrange(
                torch.stack(joint_cond_masks, dim=1), "b a ... -> (b a) ..."
            )
            joint_attention_masks = einops.rearrange(
                torch.stack(joint_attention_masks, dim=1), "b a ... -> (b a) ..."
            )
            conditions = {
                "x": joint_cond_trajectories,
                "masks": joint_cond_masks,
            }
            returns = einops.repeat(returns, "b ... -> (b a) ...", a=Config.n_agents)
            env_ts = einops.repeat(env_ts, "b ... -> (b a) ...", a=Config.n_agents)

            joint_samples = self.trainer.ema_model.conditional_sample(
                conditions,
                returns=returns,
                env_ts=env_ts,
                attention_masks=joint_attention_masks,
                **self._iks,
            )
            joint_samples = einops.rearrange(
                joint_samples, "(b a) ... -> b a ...", a=Config.n_agents
            )

            samples = []
            for a_idx in range(Config.n_agents):
                samples.append(joint_samples[:, a_idx, ..., a_idx, :])
            samples = torch.stack(samples, dim=-2)

        else:
            cond_trajectories = np.zeros(shape, dtype=obs.dtype)
            cond_trajectories[:, : Config.history_horizon + 1] = obs
            agent_mask = np.ones(Config.n_agents)
            cond_masks = self.mask_generator(shape, agent_mask)
            conditions = {
                "x": to_torch(cond_trajectories, device=Config.device),
                "masks": to_torch(cond_masks, device=Config.device),
            }
            attention_masks[:, : Config.history_horizon] = 1.0
            attention_masks = to_torch(attention_masks, device=Config.device)
            gkw = {}
            if getattr(self, "g2maf_guidance_scale", 0.0) and self.g2maf_guidance_critic is not None:
                gkw = dict(
                    g2maf_guidance_fn=self._get_g2maf_guidance_fn(),
                    g2maf_guidance_scale=self.g2maf_guidance_scale,
                    g2maf_guidance_last_k=self.g2maf_guidance_last_k,
                )
            samples = self.trainer.ema_model.conditional_sample(
                conditions,
                returns=returns,
                env_ts=env_ts,
                attention_masks=attention_masks,
                **gkw,
                **self._iks,
            )

        samples = samples[:, Config.history_horizon :]
        return samples

    def _evaluate(self, load_step: Optional[int] = None):
        assert (
            self.initialized is True
        ), "Evaluator should be initialized before evaluation."

        Config = self.Config
        loadpath = os.path.join(self.log_dir, "checkpoint")

        utils.set_seed(Config.seed)

        if Config.save_checkpoints:
            assert load_step is not None
            loadpath = os.path.join(loadpath, f"state_{load_step}.pt")
        else:
            loadpath = os.path.join(loadpath, "state.pt")

        state_dict = torch.load(loadpath, map_location=Config.device)
        state_dict["model"] = {
            k: v
            for k, v in state_dict["model"].items()
            if "value_diffusion_model." not in k
        }
        state_dict["ema"] = {
            k: v
            for k, v in state_dict["ema"].items()
            if "value_diffusion_model." not in k
        }

        self.trainer.step = state_dict["step"]
        self.trainer.model.load_state_dict(state_dict["model"])
        self.trainer.ema_model.load_state_dict(state_dict["ema"])

        num_eval = Config.num_eval
        num_envs = Config.num_envs

        episode_rewards = []
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = []
        g2maf_diag_rows = []
        g2maf_timing_rows = []

        cur_num_eval = 0
        while cur_num_eval < num_eval:
            num_episodes = min(num_eval - cur_num_eval, num_envs)
            rets = self._episodic_eval(num_episodes=num_episodes)
            episode_rewards.append(rets[1])
            if Config.env_type == "smac" or Config.env_type == "smacv2":
                episode_wins.append(rets[2])
            diag = getattr(self, "_last_g2maf_diag", None)
            if diag:
                g2maf_diag_rows.append(diag)
            timing = getattr(self, "_last_g2maf_timing", None)
            if timing:
                g2maf_timing_rows.append(timing)

            cur_num_eval += num_episodes

        episode_rewards = np.concatenate(episode_rewards, axis=0)
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = np.concatenate(episode_wins, axis=0)

        metrics_dict = dict(
            average_ep_reward=np.mean(episode_rewards, axis=0),
            std_ep_reward=np.std(episode_rewards, axis=0),
        )
        if g2maf_diag_rows:
            for key in sorted(g2maf_diag_rows[0].keys()):
                vals = [row[key] for row in g2maf_diag_rows if key in row]
                if vals:
                    metrics_dict[key] = float(np.mean(vals))
        if g2maf_timing_rows:
            for key in sorted(g2maf_timing_rows[0].keys()):
                vals = [row[key] for row in g2maf_timing_rows if key in row]
                if vals:
                    metrics_dict[key] = float(np.mean(vals))

        if Config.env_type == "smac" or Config.env_type == "smacv2":
            metrics_dict["win_rate"] = np.mean(episode_wins)

        logger.print(
            ", ".join([f"{k}: {v}" for k, v in metrics_dict.items()]),
            color="green",
        )
        save_file_path = (
            f"results/step_{load_step}-ep_{num_eval}-ddim.json"
            if getattr(Config, "use_ddim_sample", False)
            else f"results/step_{load_step}-ep_{num_eval}.json"
        )
        if self.rewrite_cgw:
            save_file_path = save_file_path.replace(
                ".json", f"-cg_{self.trainer.ema_model.condition_guidance_w}.json"
            )
        if getattr(Config, "g2maf_action_step", 0.0):
            save_file_path = save_file_path.replace(
                ".json", f"-g2mafstep_{Config.g2maf_action_step}.json"
            )
        # Save individual JSON file (keep original behavior)
        logger.save_json(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics_dict.items()
            },
            save_file_path,
        )

        # Also save to unified CSV file for easier analysis
        self._save_to_unified_csv(load_step, metrics_dict)

        # Return metrics for wandb logging
        return metrics_dict

    def _save_to_unified_csv(self, step, metrics_dict):
        """Save evaluation results to a unified CSV file"""
        import csv
        import os
        from ml_logger import logger

        csv_file_path = "results/evaluation_history.csv"
        csv_full_path = os.path.join(logger.prefix, csv_file_path)

        # Prepare row data
        row_data = {"step": step}
        for k, v in metrics_dict.items():
            if isinstance(v, (list, np.ndarray)):
                if len(v) > 0:
                    row_data[f"{k}_mean"] = float(np.mean(v))
                    row_data[f"{k}_std"] = float(np.std(v)) if len(v) > 1 else 0.0
            else:
                row_data[k] = float(v)

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(csv_full_path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_full_path), exist_ok=True)

        # Write to CSV
        with open(csv_full_path, 'a', newline='') as csvfile:
            fieldnames = list(row_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    def _update_return_to_go(self, rtg, reward):
        rtg = rtg * self.Config.returns_scale
        reward = torch.tensor(reward, device=rtg.device, dtype=rtg.dtype).reshape(1, -1)
        rtg = (rtg - reward) / self.Config.discount
        rtg = rtg / self.Config.returns_scale
        return rtg

    def _episodic_eval(self, num_episodes: int):
        """Evaluate for one episode each environment."""

        # `num_episodes` can be smaller than total number of environment, and
        # we only use the first `num_episodes` environments.
        assert (
            num_episodes <= self.Config.num_envs
        ), f"num_episodes should be <= num_envs, but {num_episodes} > {self.Config.num_envs}"

        Config = self.Config
        device = Config.device
        observation_dim = self.normalizer.observation_dim

        dones = [0 for _ in range(num_episodes)]
        episode_rewards = [np.zeros(Config.n_agents) for _ in range(num_episodes)]
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = np.zeros(num_episodes)
        collect_g2maf_diag = bool(getattr(Config, "g2maf_collect_diagnostics", False))
        g2maf_diag = (
            {"q_before": [], "q_after": [], "delta_norm": [], "grad_norm": [], "clip_frac": []}
            if collect_g2maf_diag
            else None
        )
        collect_timing = bool(getattr(Config, "g2maf_collect_timing", False))
        g2maf_timing = (
            {
                "policy_sec": [],
                "inverse_sec": [],
                "ccr_sec": [],
                "decision_sec": [],
            }
            if collect_timing
            else None
        )

        def sync_if_needed():
            if collect_timing and torch.cuda.is_available():
                torch.cuda.synchronize()

        def now():
            return time.perf_counter()

        returns = to_device(
            Config.test_ret * torch.ones(num_episodes, 1, Config.n_agents), device
        )
        env_ts = to_device(
            torch.arange(Config.horizon + Config.history_horizon)
            - Config.history_horizon,
            device,
        )
        env_ts = einops.repeat(env_ts, "t -> b t", b=num_episodes)

        t = 0
        obs_list = [env.reset()[None] for env in self.env_list[:num_episodes]]
        obs = np.concatenate(obs_list, axis=0)
        recorded_obs = [deepcopy(obs[:, None])]

        if Config.history_horizon > 0:
            print(f"\nUsing history length of {Config.history_horizon}\n")
        else:
            print("\nDo NOT use history conditioning\n")
        obs_queue = deque(maxlen=Config.history_horizon + 1)
        if Config.use_zero_padding:
            obs_queue.extend(
                [np.zeros_like(obs) for _ in range(Config.history_horizon)]
            )
        else:
            normed_obs = self.normalizer.normalize(obs, "observations")
            obs_queue.extend([normed_obs for _ in range(Config.history_horizon)])

        while sum(dones) < num_episodes:
            raw_obs_for_g2maf = deepcopy(obs)
            obs = self.normalizer.normalize(obs, "observations")
            obs_queue.append(obs)
            obs = np.stack(list(obs_queue), axis=1)

            sync_if_needed()
            decision_t0 = now()
            policy_t0 = decision_t0
            samples = self._generate_samples(obs, returns, env_ts)
            sync_if_needed()
            if g2maf_timing is not None:
                g2maf_timing["policy_sec"].append(now() - policy_t0)

            inverse_t0 = now()
            obs_comb = torch.cat([samples[:, 0, :, :], samples[:, 1, :, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, Config.n_agents, 2 * observation_dim)

            if Config.share_inv or Config.joint_inv:
                if Config.joint_inv:
                    actions = self.trainer.ema_model.inv_model(
                        obs_comb.reshape(obs_comb.shape[0], -1)
                    ).reshape(obs_comb.shape[0], obs_comb.shape[1], -1)
                else:
                    actions = self.trainer.ema_model.inv_model(obs_comb)
            else:
                actions = torch.stack(
                    [
                        self.trainer.ema_model.inv_model[i](obs_comb[:, i])
                        for i in range(Config.n_agents)
                    ],
                    dim=1,
                )
            sync_if_needed()
            if g2maf_timing is not None:
                g2maf_timing["inverse_sec"].append(now() - inverse_t0)

            samples = to_np(samples)
            ccr_elapsed = 0.0

            if self.discrete_action:
                legal_action = np.stack(
                    [env.get_legal_actions() for env in self.env_list], axis=0
                )
                # --- discrete G2MAF refinement (zgw): gradient on action logits ---
                if self.g2maf_action_step > 0 and self.g2maf_critic is not None:
                    sync_if_needed()
                    ccr_t0 = now()
                    with torch.enable_grad():
                        obs_t = torch.tensor(
                            raw_obs_for_g2maf, device=Config.device, dtype=torch.float32
                        ).reshape(num_episodes, -1)
                        logits = actions.detach().clone().to(Config.device).float()
                        logits.requires_grad_(True)
                        legal_t = torch.tensor(
                            legal_action, device=Config.device, dtype=torch.float32
                        )
                        masked = logits.masked_fill(legal_t == 0, -1e9)
                        soft = torch.softmax(masked, dim=-1)
                        act_vec = soft.reshape(num_episodes, -1)
                        if self.g2maf_ablation_mode == "best_of_n":
                            # same-compute search control in logit space
                            def _score(lg):
                                m = lg.masked_fill(legal_t == 0, -1e9)
                                s = torch.softmax(m, dim=-1).reshape(num_episodes, -1)
                                return self.g2maf_critic(obs_t, s)
                            with torch.no_grad():
                                base_logits = logits.detach()
                                best_logits = base_logits.clone()
                                q_best = _score(base_logits)
                                for _ in range(self.g2maf_best_of_n):
                                    eps = torch.randn_like(base_logits)
                                    denom = eps.reshape(num_episodes, -1).norm(dim=1).view(-1, 1, 1) + 1e-6
                                    cand = base_logits + self.g2maf_action_step * (eps / denom)
                                    q_cand = _score(cand)
                                    improve = q_cand > q_best
                                    best_logits = torch.where(improve.view(-1, 1, 1), cand, best_logits)
                                    q_best = torch.where(improve, q_cand, q_best)
                            refined_logits = best_logits.detach()
                        else:
                            if self.g2maf_ablation_mode == "random":
                                q_grad = torch.randn_like(logits)
                            else:
                                critic_obs = obs_t
                                if self.g2maf_ablation_mode == "shuffled_obs" and obs_t.shape[0] > 1:
                                    critic_obs = torch.roll(obs_t, shifts=1, dims=0)
                                q_val = self.g2maf_critic(critic_obs, act_vec)
                                q_grad = torch.autograd.grad(q_val.sum(), logits)[0]
                            if getattr(Config, "g2maf_normalize_grad", True):
                                denom = q_grad.reshape(num_episodes, -1).norm(dim=1).view(-1, 1, 1) + 1e-6
                                q_grad = q_grad / denom
                            if self.g2maf_ablation_mode == "single_agent":
                                mask = torch.zeros_like(q_grad)
                                mask[:, 0, :] = 1.0
                                q_grad = q_grad * mask
                            refined_logits = (logits + self.g2maf_action_step * q_grad).detach()
                        if g2maf_diag is not None:
                            with torch.no_grad():
                                before_soft = torch.softmax(masked, dim=-1).reshape(num_episodes, -1)
                                after_masked = refined_logits.masked_fill(legal_t == 0, -1e9)
                                after_soft = torch.softmax(after_masked, dim=-1).reshape(num_episodes, -1)
                                q_before = self.g2maf_critic(obs_t, before_soft)
                                q_after = self.g2maf_critic(obs_t, after_soft)
                                delta = (refined_logits - logits.detach()).reshape(num_episodes, -1)
                                g2maf_diag["q_before"].append(float(q_before.mean().cpu()))
                                g2maf_diag["q_after"].append(float(q_after.mean().cpu()))
                                g2maf_diag["delta_norm"].append(float(delta.norm(dim=1).mean().cpu()))
                                g2maf_diag["grad_norm"].append(float(q_grad.reshape(num_episodes, -1).norm(dim=1).mean().cpu()))
                                g2maf_diag["clip_frac"].append(0.0)
                        actions = refined_logits
                    sync_if_needed()
                    ccr_elapsed = now() - ccr_t0
                actions = to_np(actions)
                actions[np.where(legal_action.astype(int) == 0)] = -np.inf
                actions = np.argmax(actions, axis=-1)
            else:
                actions = self.normalizer.unnormalize(to_np(actions), "actions")
                if self.g2maf_action_step > 0 and self.g2maf_critic is not None:
                    sync_if_needed()
                    ccr_t0 = now()
                    with torch.enable_grad():
                        obs_t = torch.tensor(
                            raw_obs_for_g2maf,
                            device=Config.device,
                            dtype=torch.float32,
                        ).reshape(num_episodes, -1)
                        act_t = torch.tensor(
                            actions,
                            device=Config.device,
                            dtype=torch.float32,
                        ).reshape(num_episodes, -1)
                        act_t.requires_grad_(True)
                        if self.g2maf_ablation_mode == "best_of_n":
                            # same-compute Gaussian local-search control: draw N
                            # unit-norm perturbations at the CCR step radius, keep
                            # the candidate (or the frozen action) with highest Q.
                            with torch.no_grad():
                                base_act = act_t.detach()
                                best_act = base_act.clone()
                                q_best = self.g2maf_critic(obs_t, base_act)
                                for _ in range(self.g2maf_best_of_n):
                                    eps = torch.randn_like(base_act)
                                    eps = eps / (eps.norm(dim=1, keepdim=True) + 1e-6)
                                    cand = (base_act + self.g2maf_action_step * eps).clamp(-1.0, 1.0)
                                    q_cand = self.g2maf_critic(obs_t, cand)
                                    improve = q_cand > q_best
                                    best_act = torch.where(improve.unsqueeze(1), cand, best_act)
                                    q_best = torch.where(improve, q_cand, q_best)
                            refined = best_act
                        elif self.g2maf_ablation_mode == "factorized":
                            # decentralized: each agent ascends its own Q_i(o_i,a_i).
                            # Same unit-norm + step + clip as joint CCR, so only the
                            # gradient direction (centralized vs decentralized) differs.
                            na_ = Config.n_agents
                            opa, apa = self.g2maf_fact_obs_pa, self.g2maf_fact_act_pa
                            obs_resh = obs_t.reshape(num_episodes, na_, opa)
                            a_resh = act_t.detach().reshape(num_episodes, na_, apa).clone()
                            a_resh.requires_grad_(True)
                            q_sum = 0.0
                            for i, crit in enumerate(self.g2maf_factorized):
                                q_sum = q_sum + crit(obs_resh[:, i, :], a_resh[:, i, :]).sum()
                            q_grad = torch.autograd.grad(q_sum, a_resh)[0].reshape(num_episodes, -1)
                            if getattr(Config, "g2maf_normalize_grad", True):
                                q_grad = q_grad / (q_grad.norm(dim=1, keepdim=True) + 1e-6)
                            refined = (act_t.detach() + self.g2maf_action_step * q_grad).clamp(-1.0, 1.0)
                        else:
                            if self.g2maf_ablation_mode == "random":
                                q_grad = torch.randn_like(act_t)
                            else:
                                critic_obs = obs_t
                                if self.g2maf_ablation_mode == "shuffled_obs" and obs_t.shape[0] > 1:
                                    critic_obs = torch.roll(obs_t, shifts=1, dims=0)
                                q_val = self.g2maf_critic(critic_obs, act_t)
                                q_grad = torch.autograd.grad(q_val.sum(), act_t)[0]
                            if getattr(Config, "g2maf_normalize_grad", True):
                                q_grad = q_grad / (q_grad.norm(dim=1, keepdim=True) + 1e-6)
                            if self.g2maf_ablation_mode == "single_agent":
                                # update only agent 0; teammates execute frozen action
                                g = q_grad.reshape(num_episodes, Config.n_agents, -1)
                                mask = torch.zeros_like(g)
                                mask[:, 0, :] = 1.0
                                q_grad = (g * mask).reshape(num_episodes, -1)
                            refined = (act_t + self.g2maf_action_step * q_grad).clamp(-1.0, 1.0)
                        if g2maf_diag is not None:
                            with torch.no_grad():
                                q_before = self.g2maf_critic(obs_t, act_t.detach())
                                q_after = self.g2maf_critic(obs_t, refined.detach())
                                delta = refined.detach() - act_t.detach()
                                clipped = ((refined <= -0.999999) | (refined >= 0.999999)).float()
                                g2maf_diag["q_before"].append(float(q_before.mean().cpu()))
                                g2maf_diag["q_after"].append(float(q_after.mean().cpu()))
                                g2maf_diag["delta_norm"].append(float(delta.norm(dim=1).mean().cpu()))
                                g2maf_diag["grad_norm"].append(float(q_grad.norm(dim=1).mean().cpu()))
                                g2maf_diag["clip_frac"].append(float(clipped.mean().cpu()))
                        actions = to_np(refined.reshape(num_episodes, Config.n_agents, -1))
                    sync_if_needed()
                    ccr_elapsed = now() - ccr_t0

            if g2maf_timing is not None:
                g2maf_timing["ccr_sec"].append(ccr_elapsed)
                g2maf_timing["decision_sec"].append(now() - decision_t0)

            if t == 0:
                try:
                    normed_observations = samples[:, :, :, :]
                    observations = self.normalizer.unnormalize(
                        normed_observations, "observations"
                    )
                    savepath = os.path.join("images", "sample-planned.png")
                    self.trainer.renderer.composite(savepath, observations)
                except Exception as e:
                    # Rendering might fail in worker process, log but continue
                    print(f"[Evaluator] Warning: Failed to render samples: {e}")

            obs_list = []
            for i in range(num_episodes):
                if dones[i] == 1:
                    obs_list.append(obs[i, 0][None])
                else:
                    this_obs, this_reward, this_done, this_info = self.env_list[i].step(
                        actions[i]
                    )
                    obs_list.append(this_obs[None])

                    if Config.use_return_to_go:
                        returns[i] = self._update_return_to_go(returns[i], this_reward)

                    if this_done.all() or t >= Config.max_path_length - 1:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                        if "battle_won" in this_info.keys():
                            episode_wins[i] = this_info["battle_won"]
                            logger.print(
                                f"Episode ({i}): battle won {episode_wins[i]}",
                                color="green",
                            )

                        logger.print(
                            f"Episode ({i}): {episode_rewards[i]}", color="green"
                        )

                    else:
                        episode_rewards[i] += this_reward

            obs = np.concatenate(obs_list, axis=0)
            recorded_obs.append(deepcopy(obs[:, None]))
            t += 1
            env_ts = env_ts + 1

        recorded_obs = np.concatenate(recorded_obs, axis=1)
        episode_rewards = np.array(episode_rewards)
        if g2maf_diag is not None:
            self._last_g2maf_diag = {
                f"g2maf_diag_{key}": float(np.mean(vals))
                for key, vals in g2maf_diag.items()
                if vals
            }
        else:
            self._last_g2maf_diag = None

        if g2maf_timing is not None:
            self._last_g2maf_timing = {}
            for key, vals in g2maf_timing.items():
                if vals:
                    arr = np.array(vals, dtype=float)
                    self._last_g2maf_timing[f"g2maf_timing_{key}_mean"] = float(arr.mean())
                    self._last_g2maf_timing[f"g2maf_timing_{key}_std"] = float(arr.std())
                    self._last_g2maf_timing[f"g2maf_timing_{key}_n"] = int(arr.size)
            self._last_g2maf_timing["g2maf_timing_num_episodes"] = int(num_episodes)
        else:
            self._last_g2maf_timing = None

        if Config.env_type == "smac" or Config.env_type == "smacv2":
            return recorded_obs, episode_rewards, episode_wins
        else:
            return recorded_obs, episode_rewards

    def _init(
        self, log_dir: str, condition_guidance_w: Optional[float] = None, **kwargs
    ):
        assert self.initialized is False, "Evaluator can only be initialized once."

        # Set matplotlib backend to 'Agg' for non-interactive plotting in worker process
        import matplotlib
        matplotlib.use('Agg')

        self.log_dir = log_dir
        with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)

        Config = build_config_from_dict(params["Config"])
        self.Config = Config = build_config_from_dict(kwargs, Config)
        self.Config.joint_inv = getattr(Config, "joint_inv", False)
        self.Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
        self.Config.use_ddim_sample = getattr(Config, "use_ddim_sample", False)
        self.Config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.g2maf_action_step = float(getattr(Config, "g2maf_action_step", 0.0) or 0.0)
        self.g2maf_ablation_mode = getattr(Config, "g2maf_ablation_mode", "critic")
        if self.g2maf_ablation_mode not in (
            "critic", "random", "shuffled_obs", "best_of_n", "single_agent", "factorized"
        ):
            raise ValueError(f"Unknown g2maf_ablation_mode: {self.g2maf_ablation_mode}")
        # number of random candidates for the best_of_n same-compute search control
        self.g2maf_best_of_n = int(getattr(Config, "g2maf_best_of_n", 16) or 16)
        # denoising-step override (sweep k=1..5 to match CoFlow's best-of-k protocol);
        # 0/None -> use the model's default max_denoising_steps
        self.g2maf_inference_steps = int(getattr(Config, "g2maf_inference_steps", 0) or 0)
        self._iks = ({"inference_steps": self.g2maf_inference_steps}
                     if self.g2maf_inference_steps > 0 else {})
        self.g2maf_critic = None
        self.g2maf_factorized = None
        if self.g2maf_action_step > 0:
            g2maf_critic_path = getattr(Config, "g2maf_critic_path", None)
            if g2maf_critic_path is None:
                raise ValueError("g2maf_critic_path must be set when g2maf_action_step > 0")
            from g2maf.critic import CentralizedBehaviorCritic as G2MAFCritic
            g2maf_state = torch.load(g2maf_critic_path, map_location=self.Config.device)
            self.g2maf_critic = G2MAFCritic(
                g2maf_state["config"]["obs_dim"],
                g2maf_state["config"]["act_dim"],
                g2maf_state["config"].get("hidden", 512),
            ).to(self.Config.device)
            self.g2maf_critic.load_state_dict(g2maf_state["critic"])
            self.g2maf_critic.eval()
            for p in self.g2maf_critic.parameters():
                p.requires_grad_(False)
            print(f"Using G2MAF action refinement: step={self.g2maf_action_step}, "
                  f"mode={self.g2maf_ablation_mode}, critic={g2maf_critic_path}")
            if self.g2maf_ablation_mode == "factorized":
                fpath = getattr(Config, "g2maf_factorized_critic_path", None)
                if fpath is None:
                    raise ValueError(
                        "g2maf_factorized_critic_path must be set for factorized mode")
                fb = torch.load(fpath, map_location=self.Config.device)
                self.g2maf_fact_obs_pa = int(fb["obs_pa"])
                self.g2maf_fact_act_pa = int(fb["act_pa"])
                self.g2maf_factorized = []
                for sd in fb["per_agent"]:
                    c = G2MAFCritic(self.g2maf_fact_obs_pa, self.g2maf_fact_act_pa,
                                  int(fb.get("hidden", 512))).to(self.Config.device)
                    c.load_state_dict(sd)
                    c.eval()
                    for p in c.parameters():
                        p.requires_grad_(False)
                    self.g2maf_factorized.append(c)
                print(f"Using factorized decentralized critics: "
                      f"n_agents={len(self.g2maf_factorized)} obs_pa={self.g2maf_fact_obs_pa} "
                      f"act_pa={self.g2maf_fact_act_pa} path={fpath}")

        # --- G2MAF guidance (zgw): normalized-space critic for in-denoising-loop guidance ---
        self.g2maf_guidance_scale = float(getattr(Config, "g2maf_guidance_scale", 0.0) or 0.0)
        self.g2maf_guidance_last_k = getattr(Config, "g2maf_guidance_last_k", None)
        self.g2maf_guidance_norm = bool(getattr(Config, "g2maf_guidance_norm", True))
        self.g2maf_guide_mode = getattr(Config, "g2maf_guide_mode", "first")
        self.g2maf_guidance_critic = None
        self._g2maf_guidance_fn = None
        if self.g2maf_guidance_scale > 0:
            gpath = getattr(Config, "g2maf_guidance_critic_path", None)
            if gpath is None:
                raise ValueError("g2maf_guidance_critic_path must be set when g2maf_guidance_scale > 0")
            from g2maf.critic import CentralizedBehaviorCritic as G2MAFNormCritic
            gst = torch.load(gpath, map_location=self.Config.device)
            self.g2maf_guidance_critic = G2MAFNormCritic(
                gst["config"]["obs_dim"], gst["config"]["act_dim"],
                gst["config"].get("hidden", 512)).to(self.Config.device)
            self.g2maf_guidance_critic.load_state_dict(gst["critic"])
            self.g2maf_guidance_critic.eval()
            for p in self.g2maf_guidance_critic.parameters():
                p.requires_grad_(False)
            print(f"Using G2MAF denoising-loop guidance: scale={self.g2maf_guidance_scale}, "
                  f"last_k={self.g2maf_guidance_last_k}, mode={self.g2maf_guide_mode}, critic={gpath}")

        logger.configure(log_dir)
        torch.backends.cudnn.benchmark = True

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

        self.rewrite_cgw = False
        if condition_guidance_w is not None:
            print(f"Set condition guidance weight to {condition_guidance_w}")
            diffusion_config._dict["condition_guidance_w"] = condition_guidance_w
            self.rewrite_cgw = True

        # --- zgw: cache normalizer+mask_generator per log_dir (skip slow dataset rebuild) ---
        _zgw_cache = os.path.join(log_dir, "_zgw_norm_cache.pkl")
        _zgw_loaded = False
        if os.path.exists(_zgw_cache):
            try:
                with open(_zgw_cache, "rb") as _f:
                    _obj = pickle.load(_f)
                self.normalizer = _obj["normalizer"]
                self.mask_generator = _obj["mask_generator"]
                _zgw_loaded = True
                print("[zgw] loaded cached normalizer from", _zgw_cache)
            except Exception as _e:
                print("[zgw] norm cache load failed, rebuilding:", _e)
        if not _zgw_loaded:
            dataset = dataset_config()
            self.normalizer = dataset.normalizer
            self.mask_generator = dataset.mask_generator
            del dataset
            gc.collect()
            try:
                with open(_zgw_cache, "wb") as _f:
                    pickle.dump({"normalizer": self.normalizer,
                                 "mask_generator": self.mask_generator}, _f)
                print("[zgw] wrote normalizer cache to", _zgw_cache)
            except Exception as _e:
                print("[zgw] norm cache write failed:", _e)

        renderer = render_config()
        model = model_config()
        diffusion = diffusion_config(model)
        self.trainer = trainer_config(diffusion, None, renderer)

        if Config.use_ddim_sample:
            print(f"\n Use DDIM Sampler of {Config.n_ddim_steps} Step(s) \n")
            self.trainer.model.set_ddim_scheduler(Config.n_ddim_steps)
            self.trainer.ema_model.set_ddim_scheduler(Config.n_ddim_steps)

        self.discrete_action = False
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            self.discrete_action = True

        """ Load Environment """
        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[Config.env_type]
        env_mod = importlib.import_module(env_mod_name)

        Config.num_envs = getattr(Config, "num_envs", Config.num_eval)
        self.env_list = [
            env_mod.load_environment(Config.dataset) for _ in range(Config.num_envs)
        ]
        self.initialized = True

    def run(self):
        self.parent_remote.close()
        if not self.verbose:
            # Keep sys.stderr open for error messages, only suppress stdout
            sys.stdout = open(os.devnull, "w")
            # BUT, video encoding tools might need stdout, so save original
            self._original_stdout = sys.__stdout__
        try:
            while True:
                try:
                    cmd, data = self.queue.get()
                except EOFError:
                    self.p.close()
                    break

                if cmd == "init":
                    self._init(**data)
                elif cmd == "evaluate":
                    try:
                        metrics = self._evaluate(**data)
                        self.p.send({"status": "ok", "metrics": metrics})
                    except Exception as exc:
                        logger.print(
                            f"[Evaluator] Evaluation failed with error: {exc}",
                            color="red",
                        )
                        self.p.send({"status": "error", "error": str(exc)})
                        continue
                elif cmd == "close":
                    self.p.send("closed")
                    self.p.close()
                    # self.queue.shutdown()
                    break
                else:
                    self.p.close()
                    raise NotImplementedError(f"Unknown command {cmd}")

                time.sleep(1)

        except KeyboardInterrupt:
            self.p.close()


class MADEvaluator:
    def __init__(self, **kwargs):
        multiprocessing.set_start_method("spawn", force=True)
        self.parent_remote, self.child_remote = Pipe()
        self.queue = multiprocessing.Queue()
        self._worker_process = MADEvaluatorWorker(
            parent_remote=self.parent_remote,
            child_remote=self.child_remote,
            queue=self.queue,
            **kwargs,
        )
        self._worker_process.start()
        self.child_remote.close()

    def init(self, **kwargs):
        self.queue.put(["init", kwargs])

    def evaluate(self, **kwargs):
        self.queue.put(["evaluate", kwargs])
        # Add timeout and better error handling for pipe communication
        if self.parent_remote.poll(timeout=3600):  # 1 hour timeout
            try:
                result = self.parent_remote.recv()
                if isinstance(result, dict) and result.get("status") == "error":
                    raise RuntimeError(result.get("error", "Unknown evaluator error"))
                if isinstance(result, dict) and result.get("status") == "ok":
                    return result.get("metrics")
                return result
            except (EOFError, BrokenPipeError) as e:
                logger.print(f"[Evaluator] Communication error: {e}", color="yellow")
                return None
        else:
            logger.print("[Evaluator] Evaluation timed out after 1 hour", color="red")
            return None

    def __del__(self):
        try:
            # Try to gracefully close the worker
            self.queue.put(["close", None])
            # Wait for close confirmation with timeout
            if self.parent_remote.poll(timeout=5):
                self.parent_remote.recv()
            # Wait for worker to finish with timeout
            self._worker_process.join(timeout=10)
        except (BrokenPipeError, EOFError, AttributeError, FileNotFoundError):
            pass
        finally:
            # Ensure the subprocess is terminated
            if self._worker_process.is_alive():
                self._worker_process.terminate()
                self._worker_process.join(timeout=5)
            # Close pipe connections
            try:
                self.parent_remote.close()
            except:
                pass
            try:
                self.queue.close()
                self.queue.join_thread()
            except:
                pass
