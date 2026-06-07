"""Stage 1 go/no-go: rollouts -> usage/acknowledgment labels -> two probes.

Needs the ML stack (requirements.txt) and an open reasoning model on ~24GB.

    python run_probe_check.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""
from __future__ import annotations

import argparse

import numpy as np

from src.data import make_items
from src.rollouts import load_model, rollout
from src.labeling import extract_letter, used_cue, acknowledged_cue
from src.probes import two_probe_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--layer", type=int, default=16)
    args = ap.parse_args()

    items = make_items(args.n)
    model = load_model(args.model)

    acts, usage, ack = [], [], []
    for it in items:
        text, act = rollout(model, it.prompt_cued, args.layer)
        ans = extract_letter(text, len(it.options))
        acts.append(act)
        usage.append(used_cue(ans, it.cue))
        ack.append(acknowledged_cue(text))

    usage = np.asarray(usage)
    ack = np.asarray(ack)
    covert = int(((usage == 1) & (ack == 0)).sum())
    print(f"used-but-not-acknowledged (covert) bucket: {covert}/{len(items)}")

    rep = two_probe_report(np.asarray(acts), usage, ack)
    print(f"usage probe AUROC          = {rep['auroc_usage']:.3f}")
    print(f"acknowledgment probe AUROC = {rep['auroc_ack']:.3f}")
    print(f"cosine(d_use, d_ack)       = {rep['cosine_use_ack']:+.3f}")
    print("Non-collinear directions + a non-empty covert bucket = green light for steering.")


if __name__ == "__main__":
    main()
