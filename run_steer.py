"""Stage 2: build the acknowledgment steering vector and sweep its magnitude.

A confession switch shows up as the acknowledgment rate moving with alpha while
the answer-change rate stays low. Needs the ML stack and a model.

    python run_steer.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""
from __future__ import annotations

import argparse

from src.data import make_items
from src.rollouts import load_model, rollout
from src.labeling import extract_letter, used_cue, acknowledged_cue
from src.steering import acknowledgment_direction, steering_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--alphas", default="-8,-4,0,4,8")
    ap.add_argument("--eval", type=int, default=120)
    args = ap.parse_args()

    items = make_items(args.n)
    model = load_model(args.model)

    acts, usage, ack, base = [], [], [], []
    for it in items:
        text, act = rollout(model, it.prompt_cued, args.layer)
        ans = extract_letter(text, len(it.options))
        acts.append(act)
        usage.append(used_cue(ans, it.cue))
        ack.append(acknowledged_cue(text))
        base.append((it, ans))
    direction = acknowledgment_direction(acts, ack, usage)

    hookname = f"blocks.{args.layer}.hook_resid_post"
    print("alpha,  ack_rate,  answer_change_rate")
    for alpha in (float(a) for a in args.alphas.split(",")):
        mentioned = changed = total = 0
        for it, base_ans in base[:args.eval]:
            with model.hooks(fwd_hooks=[(hookname, steering_hook(direction, alpha))]):
                out = model.generate(model.to_tokens(it.prompt_cued),
                                     max_new_tokens=256, do_sample=False, verbose=False)
            text = model.to_string(out[0])
            mentioned += acknowledged_cue(text)
            changed += int(extract_letter(text, len(it.options)) != base_ans)
            total += 1
        print(f"{alpha:>5}, {mentioned / total:>8.2f}, {changed / total:>16.2f}")
    print("Confession switch: ack_rate tracks alpha while answer_change_rate stays low.")


if __name__ == "__main__":
    main()
