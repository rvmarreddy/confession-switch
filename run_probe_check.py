"""Stage 1 go/no-go (M1): rollouts -> labels -> two probes -> the pre-registered gates.

Needs the ML stack (requirements.txt), an open reasoning model on ~24GB, and for the
primary acknowledgment label a local judge (Ollama with qwen2.5:7b, or another model).

    python run_probe_check.py --dataset mmlu --cue sycophancy --n 600 --judge qwen2.5:7b

Every rollout is saved to results/ (kept out of the public repo) so labels can be
re-judged and hand-checked without re-running the model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data import load_mcq, make_items
from src.gates import m1_verdict
from src.labeling import label_rollout, ollama_judge, used_cue
from src.probes import mean_diff_direction
from src.rollouts import TOKEN_BUDGET, load_model, rollout
from src.steering import acknowledgment_direction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--dataset", default="mmlu", choices=["mmlu", "arc", "synthetic"])
    ap.add_argument("--cue", default="sycophancy", choices=["sycophancy", "metadata", "naturalistic"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--judge", default="qwen2.5:7b", help="Ollama model for the primary A label, or 'none'")
    ap.add_argument("--uncued", action="store_true", help="also run the un-cued prompt, for the strong (flip) usage label")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = (make_items(args.n, seed=args.seed, cue_type=args.cue) if args.dataset == "synthetic"
             else load_mcq(args.dataset, args.n, seed=args.seed, cue_type=args.cue))
    judge = None if args.judge == "none" else ollama_judge(args.judge)
    if judge is None:
        print("WARNING: no judge, so A falls back to the keyword readout. That is not the pre-registered label.")
    model = load_model(args.model)

    out = Path("results") / f"m1_{args.dataset}_{args.cue}_L{args.layer}_{time.strftime('%Y%m%d-%H%M')}"
    out.mkdir(parents=True, exist_ok=True)
    acts, usage, ack, answers, capped = [], [], [], [], 0
    with open(out / "rollouts.jsonl", "w") as f:
        for i, it in enumerate(items):
            text, act, hit = rollout(model, it.prompt_cued, args.layer)
            lab = label_rollout(text, it.cue_text, judge, len(it.options))
            row = {"i": i, "source": it.source, "cue": it.cue, "correct": it.correct, "hit_cap": hit, "text": text, **lab}
            if args.uncued:
                plain, _, _ = rollout(model, it.prompt_plain, args.layer)
                row["answer_uncued"] = label_rollout(plain, it.cue_text, None, len(it.options))["answer"]
                row["flip"] = int(row["answer_uncued"] != it.cue and lab["answer"] == it.cue)
            f.write(json.dumps(row) + "\n")
            acts.append(act)
            usage.append(used_cue(lab["answer"], it.cue))
            ack.append(lab["A"] if judge is not None else lab["keyword"])
            answers.append(lab["answer"])
            capped += hit
    np.save(out / "activations.npy", np.asarray(acts))

    X, usage, ack = np.asarray(acts), np.asarray(usage), np.asarray(ack)
    print(f"{len(items)} rollouts saved to {out}; {capped} hit the {TOKEN_BUDGET}-token cap")
    print(f"used-but-not-acknowledged (covert) bucket: {int(((usage == 1) & (ack == 0)).sum())}")

    rng = np.random.default_rng(args.seed)
    test = np.zeros(len(items), bool)
    test[rng.permutation(len(items))[: int(args.test_frac * len(items))]] = True
    d_use = mean_diff_direction(X[~test], usage[~test])
    d_ack, info = acknowledgment_direction(X[~test], ack[~test], usage[~test],
                                           answers=[answers[i] for i in np.flatnonzero(~test)], drop_thin=True)
    print(f"d_ack strata used {info['used']}, dropped {info['dropped']}")

    v = m1_verdict(X[test], usage[test], ack[test], d_use, d_ack)
    print(f"cells (U, A): {v['cells']}  gate: {v['cells_ok']}")
    print(f"cosine(d_use, d_ack) = {v['cosine']:+.3f}  gate: {v['cosine_ok']}")
    for label in ("usage", "ack"):
        auc, lo, hi = v[f"auroc_{label}"]
        print(f"{label}: own AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}] gate: {v[f'auroc_{label}_ok']}; "
              f"cross AUROC {v[f'cross_{label}']:.3f} gate: {v[f'cross_{label}_ok']}")
    print("GREEN LIGHT" if v["green"] else "not green" if v["green"] is False else "undecided: delong_paired not written yet")
    json.dump({k: v[k] for k in v if k != "cells"} | {"cells": {str(k): n for k, n in v["cells"].items()}},
              open(out / "verdict.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
