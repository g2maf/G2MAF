#!/usr/bin/env python
"""Evaluate one post-generation G²MAF action-refinement sweep. All policy and critic paths are explicit arguments."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

utils = None  # Imported after CLI parsing so --help remains self-contained.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _scalar(v):
    if isinstance(v, (list, np.ndarray)):
        return float(np.mean(v))
    return float(v)


def run_one(log_dir, load_step, num_eval, test_ret, g2maf_step,
            critic_path, normalize_grad, results_subdir, overwrite):
    results_dir = os.path.join(log_dir, results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    tag = (f"step_{load_step}-ep_{num_eval}-testret_{test_ret}"
           f"-g2maf_{g2maf_step}-norm_{int(normalize_grad)}.json")
    result_file = os.path.join(results_dir, tag)

    if not overwrite and os.path.exists(result_file):
        with open(result_file) as f:
            rec = json.load(f)
        print(f"[SKIP] exists: {result_file} -> {rec.get('avg_reward'):.2f}")
        return rec

    print(f"\n{'='*60}\n[RUN] test_ret={test_ret} g2maf_step={g2maf_step} "
          f"num_eval={num_eval} norm={normalize_grad}\n{'='*60}")

    evaluator_config = utils.Config("utils.MADEvaluator", verbose=False)
    evaluator = evaluator_config()

    init_kwargs = dict(
        log_dir=log_dir,
        num_eval=num_eval,
        num_envs=num_eval,
        condition_guidance_w=None,
        use_ddim_sample=False,
        n_ddim_steps=5,
        test_ret=test_ret,
    )
    if g2maf_step > 0:
        init_kwargs.update(
            g2maf_action_step=g2maf_step,
            g2maf_critic_path=critic_path,
            g2maf_normalize_grad=normalize_grad,
        )
    evaluator.init(**init_kwargs)
    metrics = evaluator.evaluate(load_step=load_step)

    rec = None
    if metrics is not None:
        rec = {
            "test_ret": test_ret,
            "g2maf_step": g2maf_step,
            "num_eval": num_eval,
            "normalize_grad": normalize_grad,
            "avg_reward": _scalar(metrics.get("average_ep_reward")),
            "std_reward": _scalar(metrics.get("std_ep_reward")),
        }
        with open(result_file, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[DONE] {result_file} -> avg={rec['avg_reward']:.2f} "
              f"std={rec['std_reward']:.2f}")
    else:
        print(f"[FAIL] no metrics for test_ret={test_ret} g2maf_step={g2maf_step}")

    del evaluator
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpu", type=int, default=0)
    ap.add_argument("--num_eval", type=int, default=20)
    ap.add_argument("--test_rets", type=str, default="1.3,1.4,1.5,1.6")
    ap.add_argument("--g2maf_steps", type=str, default="0,0.01,0.03,0.05,0.1")
    ap.add_argument("--no_normalize_grad", action="store_true")
    ap.add_argument("--log_dir", type=str, required=True, help="CoFlow run directory")
    ap.add_argument("--critic_path", type=str, required=True, help="G2MAF critic checkpoint")
    ap.add_argument("--load_step", type=int, required=True, help="Policy checkpoint step")
    ap.add_argument("--results_subdir", type=str, default="results_g2maf_refine")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    global utils
    import diffuser.utils as utils  # noqa: E402

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test_rets = [float(x) for x in args.test_rets.split(",") if x.strip()]
    g2maf_steps = [float(x) for x in args.g2maf_steps.split(",") if x.strip()]
    normalize_grad = not args.no_normalize_grad

    summary = []
    failures = []
    for tr in test_rets:
        for qs in g2maf_steps:
            try:
                rec = run_one(args.log_dir, args.load_step, args.num_eval,
                              tr, qs, args.critic_path, normalize_grad,
                              args.results_subdir, args.overwrite)
                if rec is None:
                    raise RuntimeError("Evaluator returned no metrics")
                summary.append(rec)
            except Exception as e:
                failures.append(str(e))
                print(f"[ERROR] test_ret={tr} g2maf_step={qs}: {e}")
                import traceback
                traceback.print_exc()

    # Summary table: rows=test_ret, cols=g2maf_step, cells=avg_reward (delta vs g2maf=0)
    print(f"\n{'='*70}\nG2MAF REFINE SWEEP SUMMARY  (avg_reward; (+/-d) vs g2maf=0)"
          f"  num_eval={args.num_eval} norm={normalize_grad}\n{'='*70}")
    qss = sorted({r["g2maf_step"] for r in summary})
    print("test_ret | " + " | ".join(f"g2maf={q}" for q in qss))
    for tr in sorted({r["test_ret"] for r in summary}):
        base = next((r["avg_reward"] for r in summary
                     if r["test_ret"] == tr and r["g2maf_step"] == 0), None)
        cells = []
        for q in qss:
            v = next((r["avg_reward"] for r in summary
                      if r["test_ret"] == tr and r["g2maf_step"] == q), None)
            if v is None:
                cells.append("    -   ")
            elif q == 0 or base is None:
                cells.append(f"{v:7.0f} ")
            else:
                cells.append(f"{v:7.0f}({v-base:+.0f})")
        print(f"  {tr:5.2f}  | " + " | ".join(cells))
    print("=" * 70)

    if failures:
        raise SystemExit("Evaluation failed for %d setting(s)" % len(failures))


if __name__ == "__main__":
    main()
