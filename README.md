# The Confession Switch

**Can you steer whether a model *admits* using a cue in its chain-of-thought, separately from whether it *uses* it?**

Reasoning models often use a hint to reach an answer and then write a chain-of-thought (CoT) that never mentions it. This project asks whether that omission is a *separate, steerable mechanism* rather than just absent computation. Concretely, it looks for a **causal double dissociation**:

- an **acknowledgment direction** in activation space that turns verbalized acknowledgment of a cue up or down **while the answer and the cue's behavioral effect stay fixed**, and
- a distinct **usage direction** that moves the answer but not the acknowledgment.

If that dissociation exists, unfaithful CoT is partly a thin **reporting/suppression** layer on top of the computation. The safety punchline: **CoT monitoring can be defeated by a cheap acknowledgment-suppression intervention that leaves behavior intact.** The honest alternative outcome (the two directions are collinear, so "acknowledgment is just a readout of usage") is also a clean, citable result.

---

## The experiment in three stages

1. **Behavioral 2x2 + two probes (the week-1 go/no-go).** On cued multiple-choice items, label each rollout by *usage* (did the answer move to the cue?) and *acknowledgment* (did the CoT verbalize the cue?). Confirm a meaningful **used-but-not-acknowledged** bucket exists, then train two linear probes on pre-answer residual activations and report each AUROC plus the **cosine between the two probe directions**. Non-collinear directions = the project is alive.
2. **The confession switch.** Build a contrastive **acknowledgment** steering vector (matched on usage/answer), sweep its magnitude, and measure the dose-response: does verbalized acknowledgment move while the answer stays flat?
3. **Localization.** Use activation / attribution patching to localize where the acknowledgment-suppression lives.

## What's in here

```
README.md            this file
PLAN.md              research plan, milestones, prior work, honest caveats
run_demo.py          runnable with zero dependencies: the two-probe orthogonality analysis on synthetic activations
run_probe_check.py   stage 1 on a real model: rollouts, labels, both probes and the pre-registered gates
run_steer.py         stage 2: build the acknowledgment steering vector and sweep magnitude
src/data.py          cued multiple-choice items (MMLU, ARC-Challenge or synthetic) with sycophancy / metadata / naturalistic cues
src/rollouts.py      model rollout (fixed token budget) + pre-answer residual-activation capture
src/labeling.py      usage + acknowledgment labels (local LLM judge as the primary label, keyword readout beside it)
src/probes.py        difference-of-means / logistic probes, AUROC, direction cosine
src/steering.py      contrastive acknowledgment steering vector + swept-magnitude forward hook
src/gates.py         the pre-registered stage-1 gates: cell counts, cosine, AUROC confidence intervals, cross-prediction
tests/               checks for the pipeline (numpy and scipy only)
```

## Quickstart

The core analysis runs with no model and no installs:

```bash
python run_demo.py     # synthetic activations: shows the usage-vs-acknowledgment probe orthogonality check
python -m unittest tests.test_pipeline -v   # needs numpy and scipy
```

The real pipeline needs a GPU (~24 GB), an open reasoning model, and a local judge model served by [Ollama](https://ollama.com):

```bash
pip install -r requirements.txt
python run_probe_check.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   # stage 1 go/no-go
python run_steer.py       --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   # stage 2 dose-response
```

## Relation to prior work (and what is new)

- **Turpin et al. 2023 / Chen et al. 2025** — document that models use cues without verbalizing them (behavioral). No causal control of acknowledgment.
- **"Catching rationalization in the act" (arXiv:2603.17199)** — *probes* for rationalization features but is correlational, and explicitly lists "turn these into steering interventions" and "are these features causally involved" as **future work**. This project does exactly that.
- **"Decoding answers before CoT" / sycophantic-anchor steering** — steer *usage* or the answer, not *acknowledgment*.

The new contribution is the **causal double dissociation of acknowledgment from usage** (a steerable confession switch), not another measurement of the gap. See `PLAN.md` for the honest caveats (chiefly: the two directions may turn out entangled, which the week-1 check is designed to catch early).

## License
MIT. Author: Rithvik Marreddy.
