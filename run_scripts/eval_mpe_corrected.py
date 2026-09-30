#!/usr/bin/env python
"""
Evaluation script for MPE with corrected test_ret values.
test_ret = normalized expert return (expert_return / returns_scale)

Results will be saved to: results_corrected_testret/

Usage:
    # Evaluate all available checkpoints
    python eval_mpe_corrected.py -g 0 --num_eval 10

    # Evaluate specific checkpoints
    python eval_mpe_corrected.py -g 0 --load_steps 50000,100000,500000 --num_eval 10

    # Evaluate specific environment
    python eval_mpe_corrected.py -g 0 --env simple_spread --quality expert --ctde False
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffuser.utils as utils
from diffuser.utils.launcher_util import build_config_from_dict


# Corrected test_ret values: expert_return / returns_scale
# Note: Some datasets have negative returns, marked as problematic
CORRECTED_TEST_RET = {
    # simple_spread - generally OK
    "simple_spread-expert": 0.74,         # 516.76 / 700 = 0.738
    "simple_spread-medium": 0.52,         # 259.85 / 500 = 0.520
    "simple_spread-medium-replay": 0.39,  # 194.16 / 500 = 0.388
    "simple_spread-random": 0.32,         # 159.84 / 500 = 0.320
    # simple_tag - problematic (returns_scale too large)
    "simple_tag-expert": 0.13,            # 90.82 / 700 = 0.130
    "simple_tag-medium": 0.07,            # 42.69 / 600 = 0.071
    "simple_tag-medium-replay": 0.01,     # 6.36 / 600 = 0.011
    "simple_tag-random": -0.01,           # -2.58 / 500 = -0.005 (NEGATIVE!)
    # simple_world - problematic (returns_scale too large, some negative)
    "simple_world-expert": 0.05,          # 34.90 / 700 = 0.050
    "simple_world-medium": 0.03,          # 15.88 / 600 = 0.026
    "simple_world-medium-replay": -0.01,  # -4.13 / 600 = -0.007 (NEGATIVE!)
    "simple_world-random": -0.01,         # -8.71 / 600 = -0.015 (NEGATIVE!)
}

# Datasets with problematic (negative) returns - skip these or warn
PROBLEMATIC_DATASETS = [
    "simple_tag-random",
    "simple_world-medium-replay",
    "simple_world-random",
]


def find_model_dirs(base_dir, env, quality):
    """Find all model directories for given configuration (both ctde modes)."""
    dataset_dir = os.path.join(base_dir, f"{env}-{quality}")
    if not os.path.exists(dataset_dir):
        return []

    model_dirs = []
    for subdir in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(model_path):
            seed_dirs = find_seed_dirs(model_path)
            if seed_dirs:
                model_dirs.extend((seed_dir, "ctde_True" in subdir) for seed_dir in seed_dirs)
            elif os.path.exists(os.path.join(model_path, "parameters.pkl")):
                model_dirs.append((model_path, "ctde_True" in subdir))

    return model_dirs


def find_seed_dirs(model_path):
    """Return trained seed run directories under a model directory."""
    seed_dirs = []
    for name in sorted(os.listdir(model_path)):
        seed_path = os.path.join(model_path, name)
        if not os.path.isdir(seed_path) or not name.isdigit():
            continue
        if os.path.exists(os.path.join(seed_path, "parameters.pkl")):
            seed_dirs.append(seed_path)
    return seed_dirs


def find_available_checkpoints(log_dir):
    """Find all available checkpoint steps in the log directory."""
    ckpt_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(ckpt_dir):
        return []

    steps = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("state_") and f.endswith(".pt"):
            try:
                step = int(f.replace("state_", "").replace(".pt", ""))
                steps.append(step)
            except ValueError:
                continue

    return sorted(steps)


def evaluate_single(log_dir, load_step, num_eval, test_ret, condition_guidance_w=None,
                    results_subdir="results_corrected_testret", overwrite=False):
    """Run evaluation for a single model with corrected test_ret."""

    # Check checkpoint
    ckpt_file = os.path.join(log_dir, f"checkpoint/state_{load_step}.pt")
    if not os.path.exists(ckpt_file):
        print(f"[SKIP] Checkpoint not found: {ckpt_file}")
        return None

    # Create results directory
    results_dir = os.path.join(log_dir, results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    # Check if results exist
    result_file = os.path.join(
        results_dir,
        f"step_{load_step}-ep_{num_eval}-testret_{test_ret}.json"
    )
    if condition_guidance_w is not None:
        result_file = result_file.replace(".json", f"-cg_{condition_guidance_w}.json")

    if not overwrite and os.path.exists(result_file):
        print(f"[SKIP] Results already exist: {result_file}")
        return None

    # Load original config
    params_file = os.path.join(log_dir, "parameters.pkl")
    if not os.path.exists(params_file):
        print(f"[SKIP] Parameters file not found: {params_file}")
        return None

    with open(params_file, "rb") as f:
        params = pickle.load(f)

    original_test_ret = params["Config"].get("test_ret", 0.9)
    print(f"\n{'='*60}")
    print(f"Log dir: {log_dir}")
    print(f"Original test_ret: {original_test_ret}")
    print(f"Corrected test_ret: {test_ret}")
    print(f"{'='*60}\n")

    # Initialize evaluator
    evaluator_config = utils.Config("utils.MADEvaluator", verbose=True)
    evaluator = evaluator_config()

    # Pass corrected test_ret to evaluator.init()
    evaluator.init(
        log_dir=log_dir,
        num_eval=num_eval,
        num_envs=num_eval,
        condition_guidance_w=condition_guidance_w,
        use_ddim_sample=False,
        n_ddim_steps=5,
        test_ret=test_ret,
    )

    # Run evaluation
    metrics = evaluator.evaluate(load_step=load_step)

    # Save results to new directory
    if metrics is not None:
        result_data = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        result_data["test_ret"] = test_ret
        result_data["original_test_ret"] = original_test_ret

        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"[DONE] Results saved to: {result_file}")

    # Cleanup evaluator
    del evaluator

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MPE models with corrected test_ret")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--load_steps", type=str, default=None,
                        help="Checkpoint steps to load (comma-separated, e.g., '50000,100000,500000'). "
                             "If not specified, evaluates all available checkpoints.")
    parser.add_argument("--num_eval", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--env", type=str, default=None,
                        help="Specific environment (simple_spread, simple_tag, simple_world)")
    parser.add_argument("--quality", type=str, default=None,
                        help="Data quality (expert, medium, medium-replay, random)")
    parser.add_argument("--ctde", type=str, default=None, help="CTDE mode (True/False)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--cg", type=float, default=None, help="Condition guidance weight")
    parser.add_argument("--skip_problematic", action="store_true", default=True,
                        help="Skip datasets with negative returns")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "ma_meanflow_mpe",
        ),
        help="Root directory containing trained MPE logs.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base_dir = args.base_dir

    # Determine which models to evaluate
    if args.env is not None:
        envs = [args.env]
    else:
        envs = ["simple_spread", "simple_tag", "simple_world"]

    if args.quality is not None:
        qualities = [args.quality]
    else:
        qualities = ["expert", "medium", "medium-replay", "random"]

    # Parse load_steps if specified
    specified_steps = None
    if args.load_steps is not None:
        specified_steps = [int(s.strip()) for s in args.load_steps.split(",")]

    # Run evaluations
    results_summary = []
    for env in envs:
        for quality in qualities:
            dataset_name = f"{env}-{quality}"

            # Check for problematic datasets
            if args.skip_problematic and dataset_name in PROBLEMATIC_DATASETS:
                print(f"[SKIP] {dataset_name} has negative returns, skipping...")
                continue

            test_ret = CORRECTED_TEST_RET.get(dataset_name)
            if test_ret is None:
                print(f"[SKIP] No corrected test_ret for {dataset_name}")
                continue

            model_dirs = find_model_dirs(base_dir, env, quality)
            if not model_dirs:
                print(f"[SKIP] No model directories found for {dataset_name}")
                continue

            for log_dir, is_ctde in model_dirs:
                # Filter by ctde if specified
                if args.ctde is not None:
                    if (args.ctde.lower() == "true") != is_ctde:
                        continue

                # Get checkpoint steps to evaluate
                if specified_steps is not None:
                    load_steps = specified_steps
                else:
                    load_steps = find_available_checkpoints(log_dir)
                    if not load_steps:
                        print(f"[SKIP] No checkpoints found for {dataset_name} (ctde={is_ctde})")
                        continue

                mode = "ctde" if is_ctde else "attn"
                print(f"\n{'#'*70}")
                print(f"# Evaluating: {dataset_name} ({mode})")
                print(f"# Checkpoints: {load_steps}")
                print(f"{'#'*70}")

                for load_step in load_steps:
                    try:
                        metrics = evaluate_single(
                            log_dir=log_dir,
                            load_step=load_step,
                            num_eval=args.num_eval,
                            test_ret=test_ret,
                            condition_guidance_w=args.cg,
                            overwrite=args.overwrite,
                        )
                        if metrics is not None:
                            results_summary.append({
                                "env": env,
                                "quality": quality,
                                "mode": mode,
                                "step": load_step,
                                "test_ret": test_ret,
                                "metrics": metrics,
                            })
                    except Exception as e:
                        print(f"[ERROR] Evaluation failed for step {load_step}: {e}")
                        import traceback
                        traceback.print_exc()

    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY (MPE)")
    print(f"{'='*70}")
    for r in results_summary:
        avg_reward = r["metrics"].get("average_ep_reward", "N/A")
        if isinstance(avg_reward, (list, np.ndarray)):
            avg_reward = np.mean(avg_reward)
        print(f"{r['env']}-{r['quality']} ({r['mode']}) step={r['step']}: test_ret={r['test_ret']}, avg_reward={avg_reward:.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
