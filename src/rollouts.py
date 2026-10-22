"""Model rollouts and pre-answer residual-activation capture (TransformerLens).

Heavy imports are inside functions so the rest of the package imports without the
ML stack. Requires torch + transformer_lens (see requirements.txt).

The token budget is fixed across every condition (proposal, Labeling): verbalisation
depends strongly on budget, so a varying budget would confound any acknowledgment
effect. 1,536 new tokens leaves R1-style reasoning room to finish on most MMLU items;
check the share of rollouts that hit the cap in the first run and raise it if many do.
"""
from __future__ import annotations

TOKEN_BUDGET = 1536


def load_model(name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device=None, base="Qwen/Qwen2.5-7B"):
    """Load into TransformerLens. If the name is not on TransformerLens's list, the HF
    weights are loaded under `base`, an architecture with the same shapes (untested
    until the first GPU session: check the logits match the HF model on one prompt)."""
    import torch
    from transformer_lens import HookedTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = HookedTransformer.from_pretrained(name, device=device, dtype=torch.bfloat16)
    except ValueError:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
        model = HookedTransformer.from_pretrained(base, hf_model=hf, device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def chat_tokens(model, prompt):
    """The prompt in the model's chat template, ready for generation (R1 opens <think>)."""
    tok = model.tokenizer
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return model.to_tokens(text, prepend_bos=False)
    return model.to_tokens(prompt)


def rollout(model, prompt, layer, max_new_tokens=TOKEN_BUDGET, hooks=()):
    """Greedy CoT+answer plus the residual activation at `layer`.

    Returns (generated_text, activation, hit_cap). The text is the NEW tokens only: the
    prompt holds the cue, so decoding it too would mark every rollout as acknowledging.
    The activation is taken on the prompt's final token (a cheap pre-answer probe site).
    """
    import torch
    name = f"blocks.{layer}.hook_resid_post"
    toks = chat_tokens(model, prompt)
    with torch.no_grad(), model.hooks(fwd_hooks=list(hooks)):
        _, cache = model.run_with_cache(toks, names_filter=name)
        act = cache[name][0, -1].float().cpu().numpy()
        out = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
    new = out[0, toks.shape[1]:]
    return model.to_string(new), act, int(new.shape[0] >= max_new_tokens)
