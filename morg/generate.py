"""Batched generation with Hugging Face `generate` (steering hooks apply to the same model)."""

from __future__ import annotations

from tqdm.auto import tqdm

from .models import free, load_model


def hf_generate(model, tok, spec, prompts, *, max_new_tokens, temperature=0.0, top_p=1.0, batch_size=32):
    """Length-sorted batches (minimal padding). Prompts already contain BOS from the chat template.
    Sampling uses exactly temperature and top_p (top_k disabled, overriding the model's generation defaults)."""
    import torch

    if not prompts:
        return []
    ids = tok(prompts, add_special_tokens=False)["input_ids"]
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": 0} if temperature > 0 else {
        "do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    out = [None] * len(prompts)
    with torch.inference_mode():
        for start in tqdm(range(0, len(order), batch_size), desc="generate", leave=False):
            idx = order[start: start + batch_size]
            batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.device)
            gen = model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                 eos_token_id=spec.stop_ids(tok), **sampling)
            texts = tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=False)
            for i, text in zip(idx, texts):
                out[i] = spec.clean(text, tok)
    return out


def generate(cfg, spec, tok, prompts, adapters=(), *, max_new_tokens=None, temperature=0.0, top_p=1.0, seed=0):
    """Load the base model + adapter stack, generate, and free the GPU."""
    import transformers

    transformers.set_seed(seed)
    model = load_model(spec, *adapters)
    try:
        return hf_generate(model, tok, spec, prompts, max_new_tokens=max_new_tokens or cfg["generation"]["max_new_tokens"],
                           temperature=temperature, top_p=top_p, batch_size=cfg["generation"]["batch_size"])
    finally:
        del model
        free()
