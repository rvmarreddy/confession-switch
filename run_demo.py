"""Demo (zero dependencies): the usage-vs-acknowledgment probe orthogonality check.

Synthesises activation vectors with a planted USAGE direction and a partly
independent ACKNOWLEDGMENT direction, fits a difference-of-means probe for each
label, and reports each probe's AUROC plus the cosine between the two probe
directions. This is the week-1 go/no-go analysis from PLAN.md, on data with known
ground truth so you can see what "dissociated" vs "entangled" looks like.

    python run_demo.py
"""
from __future__ import annotations

import math
import random


def _mean(vectors, dim):
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    n = max(len(vectors), 1)
    return [a / n for a in acc]


def mean_diff_direction(vectors, labels):
    """Difference of class means: a linear probe direction (no training needed)."""
    dim = len(vectors[0])
    pos = [v for v, y in zip(vectors, labels) if y]
    neg = [v for v, y in zip(vectors, labels) if not y]
    mp, mn = _mean(pos, dim), _mean(neg, dim)
    return [mp[i] - mn[i] for i in range(dim)]


def mean_diff_stratified(vectors, target, strata):
    """Difference of means of `target`, averaged WITHIN each value of `strata`.

    This is the fix the real pipeline uses (src/steering.py): conditioning on
    usage removes the cross-contamination that makes a naive probe look entangled
    when the two labels are correlated.
    """
    dim = len(vectors[0])
    groups = {}
    for v, t, s in zip(vectors, target, strata):
        groups.setdefault((s, t), []).append(v)
    diffs = []
    for s in set(strata):
        if (s, 1) in groups and (s, 0) in groups:
            p, n = _mean(groups[(s, 1)], dim), _mean(groups[(s, 0)], dim)
            diffs.append([p[i] - n[i] for i in range(dim)])
    return [sum(d[i] for d in diffs) / len(diffs) for i in range(dim)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    na, nb = math.sqrt(dot(a, a)), math.sqrt(dot(b, b))
    return dot(a, b) / (na * nb) if na and nb else 0.0


def auroc(scores, labels):
    """Mann-Whitney AUROC, pure python."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def synth(n, dim, use_dir, ack_dir, ack_if_used=0.5, ack_if_not=0.15, noise=1.0, seed=0):
    rng = random.Random(seed)
    V, U, A = [], [], []
    for _ in range(n):
        u = int(rng.random() < 0.5)
        a = int(rng.random() < (ack_if_used if u else ack_if_not))
        v = [noise * rng.gauss(0, 1) for _ in range(dim)]
        for i in range(dim):
            v[i] += u * use_dir[i] + a * ack_dir[i]
        V.append(v); U.append(u); A.append(a)
    return V, U, A


def report(name, V, U, A):
    d_use = mean_diff_direction(V, U)
    d_ack_naive = mean_diff_direction(V, A)
    d_ack_strat = mean_diff_stratified(V, A, U)
    auc_use = auroc([dot(v, d_use) for v in V], U)
    auc_ack = auroc([dot(v, d_ack_naive) for v in V], A)
    cos_naive = cosine(d_use, d_ack_naive)
    cos_strat = cosine(d_use, d_ack_strat)
    verdict = "DISSOCIATED (green light)" if abs(cos_strat) < 0.5 else "ENTANGLED (pivot)"
    print(f"\n[{name}]")
    print(f"  usage probe AUROC                 = {auc_use:.3f}")
    print(f"  acknowledgment probe AUROC        = {auc_ack:.3f}")
    print(f"  cosine(d_use, d_ack)  naive       = {cos_naive:+.3f}   (inflated by label correlation)")
    print(f"  cosine(d_use, d_ack)  usage-strat = {cos_strat:+.3f}   -> {verdict}")


def main():
    dim = 32
    rng = random.Random(1)
    use_dir = [rng.gauss(0, 1) for _ in range(dim)]
    indep = [rng.gauss(0, 1) for _ in range(dim)]
    c = dot(indep, use_dir) / dot(use_dir, use_dir)
    ack_orth = [indep[i] - c * use_dir[i] for i in range(dim)]  # orthogonal to usage

    print("Two-probe orthogonality check on synthetic activations with known directions.")
    V, U, A = synth(1500, dim, use_dir, ack_orth, seed=2)
    report("ground truth: acknowledgment orthogonal to usage", V, U, A)

    V2, U2, A2 = synth(1500, dim, use_dir, use_dir, seed=3)
    report("control: acknowledgment direction = usage direction", V2, U2, A2)

    print("\nOn the real model (run_probe_check.py) the same plot decides go/no-go:")
    print("non-collinear probe directions mean acknowledgment is steerable separately from usage.")


if __name__ == "__main__":
    main()
