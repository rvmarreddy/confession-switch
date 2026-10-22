"""Checks for the pre-M1 fixes. numpy and scipy only; no model, no download, no judge.

    python -m unittest tests.test_pipeline -v

The DeLong checks skip until src/gates.py:delong_paired is written.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.data import CUES, make_items
from src.gates import auroc_ci, cell_count_gate, delong_paired
from src.labeling import cohens_kappa, kappa_gate, label_rollout, split_channels
from src.probes import auroc
from src.steering import acknowledgment_direction


def _delong_written():
    try:
        delong_paired(np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.2, 0.8]))
        return True
    except NotImplementedError:
        return False


class Data(unittest.TestCase):
    def test_every_cue_type_points_at_a_wrong_option(self):
        for cue_type in CUES:
            for it in make_items(50, cue_type=cue_type):
                self.assertNotEqual(it.cue, it.correct)
                self.assertTrue(it.prompt_cued.startswith(it.cue_text))
                self.assertIn(f"{chr(65 + it.cue)}", it.cue_text)
                self.assertNotIn(it.cue_text, it.prompt_plain)


class Labeling(unittest.TestCase):
    def test_channels(self):
        self.assertEqual(split_channels("<think>a professor said B</think>The answer is C"),
                         ("a professor said B", "The answer is C"))

    def test_keyword_is_never_ored_into_the_judge(self):
        text = "<think>The professor suggested B, but I checked it myself.</think> (C)"
        lab = label_rollout(text, "cue", judge=lambda t, c: {"mention": "explicit", "influence": "none"})
        self.assertEqual(lab["keyword"], 1)
        self.assertEqual(lab["A"], 0)          # primary: influence, strict
        self.assertEqual(lab["A_mention"], 1)
        self.assertEqual(lab["A_lenient"], 0)
        self.assertEqual(lab["answer"], 2)     # read from the answer channel

    def test_bad_judge_reply_raises(self):
        with self.assertRaises(ValueError):
            label_rollout("x", "cue", judge=lambda t, c: {"mention": "yes", "influence": "none"})

    def test_kappa(self):
        self.assertAlmostEqual(cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(cohens_kappa([1, 0, 1, 0], [1, 1, 0, 0]), 0.0)
        self.assertFalse(kappa_gate([1, 0, 1, 0], [1, 1, 0, 0])[1])


class Steering(unittest.TestCase):
    def test_thin_strata_raise_and_drops_are_reported(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4))
        ack = np.r_[np.ones(30), np.zeros(30)].astype(int)
        usage = np.r_[np.ones(55), np.zeros(5)].astype(int)
        with self.assertRaises(ValueError):
            acknowledgment_direction(X, ack, usage, min_per_cell=10)
        d, info = acknowledgment_direction(X, ack, usage, min_per_cell=10, drop_thin=True)
        self.assertIn((0,), info["dropped"])
        self.assertEqual(d.shape, (4,))

    def test_recovers_a_planted_direction_within_answer_strata(self):
        rng = np.random.default_rng(1)
        n, dim = 4000, 8
        ack_dir = np.eye(dim)[0]
        answers = rng.integers(0, 4, n)
        usage = rng.integers(0, 2, n)
        ack = (rng.random(n) < 0.3 + 0.4 * usage).astype(int)   # ack correlated with usage
        X = rng.normal(size=(n, dim)) + np.outer(usage, np.eye(dim)[1]) + np.outer(answers, np.eye(dim)[2]) + np.outer(ack, ack_dir)
        d, _ = acknowledgment_direction(X, ack, usage, answers=list(answers))
        self.assertGreater(d @ ack_dir / np.linalg.norm(d), 0.95)


class Gates(unittest.TestCase):
    def test_cell_gate(self):
        usage = [1] * 120 + [0] * 40
        ack = [1] * 60 + [0] * 60 + [0] * 40
        counts, ok = cell_count_gate(usage, ack)
        self.assertEqual(counts[(1, 1)], 60)
        self.assertTrue(ok)
        self.assertFalse(cell_count_gate(usage[:100], ack[:100])[1])

    def test_auroc_ci_brackets_the_estimate(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 400)
        s = y + rng.normal(size=400)
        auc, lo, hi = auroc_ci(s, y, n_boot=500)
        self.assertLess(lo, auc)
        self.assertLess(auc, hi)
        self.assertAlmostEqual(auc, auroc(s, y))


@unittest.skipUnless(_delong_written(), "delong_paired not written yet")
class DeLong(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = rng.integers(0, 2, 600)
        self.good = self.y + 0.6 * rng.normal(size=600)
        self.weak = self.y + 2.0 * rng.normal(size=600)

    def test_aucs_match_the_mann_whitney_auroc(self):
        a, b, _, _ = delong_paired(self.y, self.good, self.weak)
        self.assertAlmostEqual(a, auroc(self.good, self.y))
        self.assertAlmostEqual(b, auroc(self.weak, self.y))

    def test_identical_scores_give_no_evidence(self):
        _, _, z, p = delong_paired(self.y, self.good, self.good)
        self.assertTrue(z == 0 or np.isnan(z))
        self.assertTrue(np.isnan(p) or p >= 0.5)

    def test_clearly_better_direction_is_significant_one_sided(self):
        _, _, z, p = delong_paired(self.y, self.good, self.weak)
        self.assertGreater(z, 0)
        self.assertLess(p, 0.01)
        _, _, z2, p2 = delong_paired(self.y, self.weak, self.good)
        self.assertLess(z2, 0)
        self.assertGreater(p2, 0.99)

    def test_variance_agrees_with_a_paired_bootstrap(self):
        a, b, z, _ = delong_paired(self.y, self.good, self.weak)
        rng = np.random.default_rng(4)
        diffs = []
        for _ in range(1000):
            i = rng.integers(0, 600, 600)
            if self.y[i].min() == self.y[i].max():
                continue
            diffs.append(auroc(self.good[i], self.y[i]) - auroc(self.weak[i], self.y[i]))
        se_boot = np.std(diffs)
        se_delong = (a - b) / z
        self.assertAlmostEqual(se_delong / se_boot, 1.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
