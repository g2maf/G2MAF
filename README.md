# G²MAF Release

[Paper](https://arxiv.org/abs/2609.31286) · [Project page](https://g2maf.github.io/) · [Models](https://huggingface.co/g2maf/G2MAF) · [Datasets](https://huggingface.co/datasets/Guowei-Zou/CoFlow-datasets)

By [Guowei Zou](https://guowei-zou.github.io/Guowei-Zou/) and collaborators.

This repository contains the source code for **G²MAF:
Test-Time Gradient Guidance for Multi-Agent Flow Policies**. G²MAF refines a
frozen multi-agent policy at deployment: a centralized behavior critic
differentiates the complete joint action, and one globally normalized,
projected step revises the decoded action.

The release contains the generative-policy backbone, centralized behavior
critic training for MPE and SMAC, post-generation action refinement,
trajectory-injection variants, discrete masked-logit refinement, and the
controls used in the paper. It does **not** contain datasets, policy or critic
checkpoints, generated logs, or paper figures.

## Environment

The experiments used Python 3.8, PyTorch 1.12.1, and CUDA 11.3. A typical
installation is:

```bash
conda create -n g2maf python=3.8
conda activate g2maf
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -e .
python scripts/smoke_test.py
```

The final command checks the packaged centralized critic and its joint-action
gradient without requiring data, a checkpoint, or an environment.

For SMAC, install StarCraft II and SMAC for the map set used in the experiment.

## Data and frozen policy checkpoints

G²MAF uses the public offline datasets and frozen generative-policy checkpoints
from the backbone experiments. They are not redistributed because of their
size and third-party licenses. The supplied data should follow these layouts:

```text
diffuser/datasets/data/mpe/<scenario>/<split>/seed_<seed>_data/
diffuser/datasets/data/smac/<map>/<quality>/
```

MPE data use the OMAR per-agent arrays `obs_i.npy`, `acs_i.npy`, and
`rews_i.npy`. SMAC data use `obs.npy`, `actions.npy`, `rewards.npy`,
`path_lengths.npy`, and `legals.npy`.

A frozen-policy run directory passed to the evaluators must contain
`model_config.pkl`, `diffusion_config.pkl`, `dataset_config.pkl`, and policy
checkpoints such as `state_200000.pt`.

## Train the centralized behavior critic

The scripts save self-contained critic checkpoints containing the observation
and action dimensions required by the evaluators.

```bash
# MPE: data_root is, for example, diffuser/datasets/data/mpe/simple_spread
python scripts/train_g2maf_critic_mpe.py \
  --data_root diffuser/datasets/data/mpe/simple_spread \
  --split expert --out_dir checkpoints/mpe_spread_expert

# SMAC
python scripts/train_g2maf_critic_smac.py \
  --data_dir diffuser/datasets/data/smac/3m/Good \
  --n_actions 9 --out_dir checkpoints/smac_3m_good
```

For the trajectory-injection variant, train the critic in the backbone's
normalized trajectory coordinates:

```bash
python scripts/train_g2maf_trajectory_critic.py \
  --domain mpe \
  --data_root diffuser/datasets/data/mpe/simple_spread \
  --split expert \
  --log_dir /path/to/frozen_policy_run \
  --out_dir checkpoints/mpe_spread_expert_trajectory
```

## Evaluate post-generation G²MAF

The canonical method directly refines the decoded joint action. Every required
path is an explicit command-line argument; the repository contains no
server-specific paths.

```bash
python scripts/evaluate_g2maf_action.py -g 0 \
  --log_dir /path/to/frozen_policy_run \
  --critic_path checkpoints/mpe_spread_expert/critic_step_20000.pt \
  --load_step 200000 \
  --test_rets 1.3 \
  --g2maf_steps 0,0.03,0.05 \
  --num_eval 20
```

A zero step reports the frozen-policy reference. Positive values apply one
joint critic-gradient refinement step. Results are written below the supplied
policy run directory unless `--results_subdir` is changed.

## Evaluate trajectory injection

```bash
python scripts/evaluate_g2maf_trajectory.py -g 0 \
  --log_dir /path/to/frozen_policy_run \
  --critic_path checkpoints/mpe_spread_expert_trajectory/critic_step_20000.pt \
  --load_step 200000 \
  --test_rets 1.3 \
  --guidance_scales 0,0.03 \
  --last_k -1 --num_eval 20
```

`--last_k -1` injects guidance after every reverse step; use a non-negative
value to restrict it to the final reverse steps. `--mode first` scores the
executed action, while `--mode mean` averages scores over the predicted
trajectory.

## Reproducibility scope

The paper reports paired five-seed aggregates. The entry points above execute
one chosen run at a time, so users can specify a checkpoint, dataset root, GPU,
and seed protocol. `RELEASE_MANIFEST.md` lists the G²MAF-specific files in
this archive.

## Independent training seeds

Policy configurations use five training seeds: `100, 200, 300, 400, 500`.
The auxiliary training entry points now use the same five seeds by default,
run sequentially in separate processes. Initialization and training sampling
use the selected seed; each run saves under `seed_<seed>/` in its output
directory, including a `training_seed.json` record.

Use `--training-seeds 100 200 300` for three runs,
`--training-seeds 100` for a single run, or `--dry-run` to inspect the
commands without loading data or training. Evaluation seeds and dataset
source seeds are separate. These defaults configure new training runs;
they do not establish that historical results or released checkpoints
contain five independently trained models.

This applies to the MPE, SMAC, and trajectory critic scripts under `scripts/train_g2maf*.py`. For a complete policy-plus-critic replicate, use the matching policy training seed as well. `--log_dir` for the trajectory critic can contain `{seed}`, which is expanded to the selected training seed; a fixed path deliberately reuses the same frozen policy normalizer. Supply the corresponding `seed_<seed>/critic_step_*.pt` to evaluation.
