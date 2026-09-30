#!/usr/bin/env python
"""Evaluate G²MAF trajectory-injection guidance. All policy and critic paths are explicit arguments."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils = None  # Imported after CLI parsing so --help remains self-contained.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _scalar(v):
    return float(np.mean(v)) if isinstance(v, (list, np.ndarray)) else float(v)


def run_one(log_dir, load_step, num_eval, test_ret, scale, critic_path,
            mode, last_k, norm, results_subdir, overwrite):
    results_dir = os.path.join(log_dir, results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    lk = "all" if last_k is None else str(last_k)
    tag = (f"step_{load_step}-ep_{num_eval}-testret_{test_ret}"
           f"-gscale_{scale}-mode_{mode}-lastk_{lk}-norm_{int(norm)}.json")
    result_file = os.path.join(results_dir, tag)
    if not overwrite and os.path.exists(result_file):
        with open(result_file) as f:
            rec = json.load(f)
        print(f"[SKIP] exists: {result_file} -> {rec.get('avg_reward'):.2f}")
        return rec

    print(f"\n{'='*60}\n[RUN] test_ret={test_ret} gscale={scale} mode={mode} "
          f"last_k={lk} num_eval={num_eval}\n{'='*60}")
    evaluator = utils.Config("utils.MADEvaluator", verbose=False)()
    init_kwargs = dict(
        log_dir=log_dir, num_eval=num_eval, num_envs=num_eval,
        condition_guidance_w=None, use_ddim_sample=False, n_ddim_steps=5,
        test_ret=test_ret,
    )
    if scale > 0:
        init_kwargs.update(
            g2maf_guidance_scale=scale,
            g2maf_guidance_critic_path=critic_path,
            g2maf_guide_mode=mode,
            g2maf_guidance_last_k=last_k,
            g2maf_guidance_norm=norm,
        )
    evaluator.init(**init_kwargs)
    metrics = evaluator.evaluate(load_step=load_step)
    rec = None
    if metrics is not None:
        rec = {"test_ret": test_ret, "guidance_scale": scale, "mode": mode,
               "last_k": lk, "norm": norm, "num_eval": num_eval,
               "avg_reward": _scalar(metrics.get("average_ep_reward")),
               "std_reward": _scalar(metrics.get("std_ep_reward"))}
        with open(result_file, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[DONE] {result_file} -> avg={rec['avg_reward']:.2f} std={rec['std_reward']:.2f}")
    else:
        print(f"[FAIL] no metrics test_ret={test_ret} gscale={scale}")
    del evaluator
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpu", type=int, default=0)
    ap.add_argument("--num_eval", type=int, default=20)
    ap.add_argument("--test_rets", type=str, default="1.3,1.4,1.5,1.6")
    ap.add_argument("--guidance_scales", type=str, default="0,0.01,0.03,0.05,0.1")
    ap.add_argument("--mode", type=str, default="first", choices=["first", "mean"])
    ap.add_argument("--last_k", type=int, default=-1, help="-1 -> all steps")
    ap.add_argument("--no_norm", action="store_true")
    ap.add_argument("--log_dir", type=str, required=True, help="CoFlow run directory")
    ap.add_argument("--critic_path", type=str, required=True, help="G2MAF trajectory critic checkpoint")
    ap.add_argument("--load_step", type=int, required=True, help="Policy checkpoint step")
    ap.add_argument("--results_subdir", type=str, default="results_g2maf_guidance")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    global utils
    import diffuser.utils as utils  # noqa: E402
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test_rets = [float(x) for x in args.test_rets.split(",") if x.strip()]
    scales = [float(x) for x in args.guidance_scales.split(",") if x.strip()]
    last_k = None if args.last_k < 0 else args.last_k
    norm = not args.no_norm

    summary = []
    for tr in test_rets:
        for s in scales:
            try:
                rec = run_one(args.log_dir, args.load_step, args.num_eval, tr, s,
                              args.critic_path, args.mode, last_k, norm,
                              args.results_subdir, args.overwrite)
                if rec:
                    summary.append(rec)
            except Exception as e:
                print(f"[ERROR] test_ret={tr} gscale={s}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}\nG2MAF GUIDANCE SWEEP  (avg_reward [delta vs gscale=0])  "
          f"mode={args.mode} last_k={last_k} num_eval={args.num_eval}\n{'='*70}")
    ss = sorted({r["guidance_scale"] for r in summary})
    print("test_ret | " + " | ".join(f"g={s}" for s in ss))
    for tr in sorted({r["test_ret"] for r in summary}):
        base = next((r["avg_reward"] for r in summary
                     if r["test_ret"] == tr and r["guidance_scale"] == 0), None)
        cells = []
        for s in ss:
            v = next((r["avg_reward"] for r in summary
                      if r["test_ret"] == tr and r["guidance_scale"] == s), None)
            if v is None:
                cells.append("   -   ")
            elif s == 0 or base is None:
                cells.append(f"{v:7.0f} ")
            else:
                cells.append(f"{v:7.0f}({v-base:+.0f})")
        print(f"  {tr:5.2f}  | " + " | ".join(cells))
    print("=" * 70)


if __name__ == "__main__":
    main()
