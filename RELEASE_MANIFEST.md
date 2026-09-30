# G²MAF Release Manifest

This source release contains the generative-policy backbone,
the G²MAF implementation, and the MPE/SMAC training and evaluation entry
points used by the paper. It excludes datasets, policy checkpoints, critic
checkpoints, logs, generated figures, and private server configuration.

Included G²MAF components:

- `g2maf/critic.py`: centralized behavior critic definition.
- `diffuser/utils/evaluator.py`: post-generation action refinement,
  discrete-logit refinement, trajectory injection variants, controls, and
  diagnostics.
- `scripts/train_g2maf_critic_mpe.py` and
  `scripts/train_g2maf_critic_smac.py`: raw-space critic training.
- `scripts/train_g2maf_trajectory_critic.py`: normalized trajectory-space
  critic training.
- `scripts/evaluate_g2maf_action.py` and
  `scripts/evaluate_g2maf_trajectory.py`: explicit evaluation entry points.
- `scripts/smoke_test.py`: dependency-light package validation.

Users provide the public datasets and frozen generative-policy checkpoints
required for a full experiment.
