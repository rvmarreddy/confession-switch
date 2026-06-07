"""Model rollouts and pre-answer residual-activation capture (TransformerLens).

Heavy imports are inside functions so the rest of the package imports without the
ML stack. Requires torch + transformer_lens (see requirements.txt).
"""
from __future__ import annotations


def load_model(name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device=None):
    import torch
    from transformer_lens import HookedTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(name, device=device)
    model.eval()
    return model


def rollout(model, prompt, layer, max_new_tokens=256):
    """Greedy CoT+answer text plus the residual activation at `layer`.

    The activation is captured on the prompt's final token (a cheap pre-answer
    probe site); swap for a mean over generated CoT tokens if preferred.
    Returns (text, activation_vector_as_numpy).
    """
    import torch
    hook = f"blocks.{layer}.hook_resid_post"
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=hook)
        act = cache[hook][0, -1].float().cpu().numpy()
        out = model.generate(toks, max_new_tokens=max_new_tokens,
                             do_sample=False, verbose=False)
    return model.to_string(out[0]), act
