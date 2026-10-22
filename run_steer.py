"""Stage 2: build the acknowledgment steering vector and sweep its magnitude.

A confession switch shows up as the acknowledgment rate moving with alpha while
the answer-change rate stays low. Needs the ML stack, a model and the local judge.
M2 proper adds the control battery; this is the first dose-response look.

    python run_steer.py --dataset mmlu --cue sycophancy --judge qwen2.5:7b
"""
from __future__ import annotations

import argparse

from src.data import load_mcq, make_items
from src.labeling import label_rollout, ollama_judge, used_cue
from src.rollouts import load_model, rollout
from src.steering import acknowledgment_direction, steering_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dataset", default="mmlu", choices=["mmlu", "arc", "synthetic"])
    ap.add_argument("--cue", default="sycophancy", choices=["sycophancy", "metadata", "naturalistic"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--judge", default="qwen2.5:7b")
    ap.add_argument("--alphas", default="-8,-4,0,4,8")
    ap.add_argument("--eval", type=int, default=120)
    args = ap.parse_args()

    items = (make_items(args.n, cue_type=args.cue) if args.dataset == "synthetic"
             else load_mcq(args.dataset, args.n, cue_type=args.cue))
    judge = None if args.judge == "none" else ollama_judge(args.judge)
    key = "A" if judge is not None else "keyword"
    model = load_model(args.model)

    acts, usage, ack, answers = [], [], [], []
    for it in items:
        text, act, _ = rollout(model, it.prompt_cued, args.layer)
        lab = label_rollout(text, it.cue_text, judge, len(it.options))
        acts.append(act)
        usage.append(used_cue(lab["answer"], it.cue))
        ack.append(lab[key])
        answers.append(lab["answer"])
    # The steering set is the first `eval` items; the direction comes from the rest.
    direction, info = acknowledgment_direction(acts[args.eval:], ack[args.eval:], usage[args.eval:],
                                               answers=answers[args.eval:], drop_thin=True)
    print(f"d_ack strata used {info['used']}, dropped {info['dropped']}")

    hookname = f"blocks.{args.layer}.hook_resid_post"
    print("alpha,  ack_rate,  answer_change_rate")
    for alpha in (float(a) for a in args.alphas.split(",")):
        mentioned = changed = 0
        for it, base_ans in zip(items[:args.eval], answers[:args.eval]):
            text, _, _ = rollout(model, it.prompt_cued, args.layer, hooks=[(hookname, steering_hook(direction, alpha))])
            lab = label_rollout(text, it.cue_text, judge, len(it.options))
            mentioned += lab[key]
            changed += int(lab["answer"] != base_ans)
        n = min(args.eval, len(items))
        print(f"{alpha:>5}, {mentioned / n:>8.2f}, {changed / n:>16.2f}")
    print("Confession switch: ack_rate tracks alpha while answer_change_rate stays low.")


if __name__ == "__main__":
    main()
