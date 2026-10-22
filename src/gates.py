"""The M1 go/no-go gates, pre-registered before any real data (fixes of 22 Oct 2026).

Green light for M2 iff, on the HELD-OUT probe set:
1. cells: at least CELL_MIN rollouts in each used cell, (U=1, A=1) and (U=1, A=0);
2. cosine: |cos(d_ack, d_use)| < COS_MAX at the chosen layer (a screen, not the evidence);
3. AUROC: the lower end of each probe's 95% bootstrap CI is >= AUROC_MIN;
4. cross-prediction: for each label, own-direction AUROC minus the other direction's AUROC
   is >= CROSS_GAP, and a one-sided paired DeLong test of own > cross has p < ALPHA.
Otherwise the outcome is graded or collinear and the H0 note is written.
The thresholds are fixed here and not moved after the data are seen. Their written
justification belongs in the proposal (task: pre-register the M1 gates).
"""
from __future__ import annotations

import numpy as np

from .probes import auroc, cosine

CELL_MIN = 50
COS_MAX = 0.5
AUROC_MIN = 0.70
CROSS_GAP = 0.10
ALPHA = 0.05


def cell_counts(usage, ack):
    u, a = np.asarray(usage), np.asarray(ack)
    return {(i, j): int(((u == i) & (a == j)).sum()) for i in (0, 1) for j in (0, 1)}


def cell_count_gate(usage, ack, min_n=CELL_MIN):
    """(counts, passed): both used cells need min_n rollouts."""
    c = cell_counts(usage, ack)
    return c, c[(1, 1)] >= min_n and c[(1, 0)] >= min_n


def auroc_ci(scores, y, n_boot=2000, level=0.95, seed=0):
    """(auc, lo, hi): percentile bootstrap, resampling within each class so both stay present."""
    s, y = np.asarray(scores, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        p = rng.choice(pos, pos.size)
        q = rng.choice(neg, neg.size)
        boots.append(auroc(np.r_[p, q], np.r_[np.ones(p.size), np.zeros(q.size)]))
    lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return auroc(s, y), float(lo), float(hi)


def delong_paired(y, scores_a, scores_b):
    """YOUR PART (Rithvik). Paired DeLong test for two AUROCs on the same labels.

    Inputs: y in {0,1} (n,), and two score vectors (n,) from two directions on the same rollouts.
    Return (auc_a, auc_b, z, p) where z = (auc_a - auc_b) / sqrt(var(auc_a - auc_b)) and
    p is the ONE-sided p-value for auc_a > auc_b.

    Method (DeLong, DeLong and Clarke-Pearson 1988): for each score vector, the structural
    components V10 (one per positive: share of negatives it beats, ties counting half) and
    V01 (one per negative: share of positives that beat it). AUC is the mean of either.
    The covariance of (auc_a, auc_b) is S10/m + S01/n, where S10 and S01 are the 2x2 sample
    covariances of the paired V10 and V01 components, m positives and n negatives.
    Then var(auc_a - auc_b) = [1, -1] S [1, -1]^T. Use scipy.stats.norm for p.

    tests/test_pipeline.py has the checks it must pass (they skip until this is written).
    """
    raise NotImplementedError("delong_paired is Rithvik's to write; see the docstring")


def m1_verdict(X, usage, ack, d_use, d_ack):
    """Every gate on held-out activations X with labels; d_use and d_ack come from the train split."""
    X = np.asarray(X, float)
    s_use, s_ack = X @ d_use, X @ d_ack
    cells, cells_ok = cell_count_gate(usage, ack)
    cos = cosine(d_use, d_ack)
    out = {"cells": cells, "cells_ok": cells_ok, "cosine": cos, "cosine_ok": abs(cos) < COS_MAX}
    for label, y, own, cross in (("usage", usage, s_use, s_ack), ("ack", ack, s_ack, s_use)):
        auc, lo, hi = auroc_ci(own, y)
        out[f"auroc_{label}"] = (auc, lo, hi)
        out[f"auroc_{label}_ok"] = lo >= AUROC_MIN
        out[f"cross_{label}"] = auroc(cross, y)
        try:
            _, _, z, p = delong_paired(y, own, cross)
            out[f"cross_{label}_ok"] = (auc - out[f"cross_{label}"] >= CROSS_GAP) and p < ALPHA
            out[f"delong_{label}"] = (z, p)
        except NotImplementedError:
            out[f"cross_{label}_ok"] = None
    gates = [out["cells_ok"], out["cosine_ok"], out["auroc_usage_ok"], out["auroc_ack_ok"],
             out["cross_usage_ok"], out["cross_ack_ok"]]
    out["green"] = None if None in gates else all(gates)
    return out
