# Confession Switch

Reasoning models often take a hint in the prompt, move their answer to match it, and then write a chain of thought that never mentions the hint. This project tests whether the mention can be steered separately from the use. It looks for two directions in the model's activations. Adding the acknowledgment direction should change whether the chain of thought mentions the hint while the answer stays the same. Adding the usage direction should change the answer while the mention stays the same.

If the two directions come apart, a monitor that reads the chain of thought could be fooled by a small change that hides the hint and leaves the answer alone. If they turn out to be the same direction, the mention is mostly a readout of the use, which is still worth knowing.

## Status

The analysis code and the stage 1 gates are written and tested on synthetic data. Nothing has been run on a real model yet.

## Stages

1. Label and probe. Cued multiple-choice items from MMLU and ARC-Challenge, with three kinds of cue (sycophancy, a metadata tag and a more natural cue). Each rollout gets two labels, usage (the answer moved to the cued option) and acknowledgment (the chain of thought mentions the cue, judged by a local model with a keyword check beside it). One linear probe per label on the residual stream before the answer, then the cosine between the two probe directions. The gates for going on to stage 2 are fixed in advance in `src/gates.py`.
2. Steer. An acknowledgment vector from the difference of means between rollouts that mention the cue and rollouts that don't, matched on usage and the chosen answer. It is added at one layer over a range of strengths, to see whether the mention rate moves while the answer stays put.
3. Localise. Activation and attribution patching to find where the effect sits. Not written yet.

## Running it

The demo needs only Python:

    python run_demo.py

The tests need numpy and scipy:

    python -m unittest tests.test_pipeline -v

The real pipeline needs a GPU with about 24 GB, an open reasoning model from Hugging Face, and a judge model served locally by [Ollama](https://ollama.com):

    pip install -r requirements.txt
    python run_probe_check.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    python run_steer.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

## Layout

    run_demo.py          probe comparison on synthetic activations
    run_probe_check.py   stage 1 on a real model
    run_steer.py         stage 2, the steering sweep
    src/data.py          cued multiple-choice items
    src/rollouts.py      generation and residual-stream capture
    src/labeling.py      usage and acknowledgment labels
    src/probes.py        probes, AUROC and direction cosine
    src/steering.py      the acknowledgment vector and the steering hook
    src/gates.py         the stage 1 gates
    tests/               tests for the pipeline

## Related work

Turpin et al. (2023, arXiv:2305.04388) and Chen et al. (2025, arXiv:2505.05410) show that models use cues without saying so. Both measure the behaviour and do not try to control it. "Catching rationalization in the act" (arXiv:2603.17199) finds probe features for rationalisation but leaves steering as future work. Work on steering the answer, such as "Decoding answers before CoT" (arXiv:2603.01437), moves the answer rather than the mention.

## Licence

MIT
