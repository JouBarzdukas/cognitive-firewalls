"""Activation steering on the post-RL model (Section 3.6).

Vector, per layer l: mean residual stream over rewarded-policy-prefilled passes minus mean over
other-policy-prefilled passes (paper: ZH - EN), on the same prompts, balanced 50/50 across A and B.
The prefill is the thought opener plus the first `prefill_tokens` tokens of that policy's cached thought;
the activation is read at the last prefilled token. During generation alpha * v_l is added to the output
of every layer in the band, at every position (apply=all) or only the newest position (apply=last).
alpha is fixed, or "select": best held-out accuracy over alpha_grid on calibration questions.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .data import DISTS, held_out_dist, rewarded_policy, shuffled, user_prompt
from .evaluate import metrics, prompts_for, score
from .generate import hf_generate
from .models import free, load_model


class _Stop(Exception):
    pass


def _prefill(cfg, spec, tok, policy, row) -> str:
    ids = tok(policy.prefix + row[policy.language], add_special_tokens=False)["input_ids"]
    return spec.thought_open + tok.decode(ids[: cfg["steering"]["prefill_tokens"]])


def contrast_texts(cfg, spec, tok, policies, questions, cache, seed):
    n = cfg["steering"]["n_contrast"]
    if n > len(questions):
        raise ValueError(f"steering.n_contrast={n} > {len(questions)} usable training questions")
    rewarded = rewarded_policy(cfg, policies)
    other = next(p for p in policies if p is not rewarded)
    pos, neg = [], []
    for i, q in enumerate(shuffled(questions, seed)[:n]):
        prompt = spec.render(tok, cfg["prompt"]["system"], user_prompt(cfg, q, DISTS[i % 2]))
        pos.append(prompt + _prefill(cfg, spec, tok, rewarded, cache[q.id]))
        neg.append(prompt + _prefill(cfg, spec, tok, other, cache[q.id]))
    return pos, neg


@torch.inference_mode()
def collect(model, spec, tok, texts, layers, batch_size) -> dict[int, torch.Tensor]:
    """Residual stream at the last token for each layer; stops the forward pass after the top layer."""
    blocks, top, store = spec.layers(model), max(layers), {l: [] for l in layers}

    def hook(layer):
        def fn(_module, _inputs, out):
            store[layer].append((out[0] if isinstance(out, tuple) else out)[:, -1].float().cpu())
            if layer == top:
                raise _Stop

        return fn

    handles = [blocks[l].register_forward_hook(hook(l)) for l in layers]
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    try:
        for start in range(0, len(ids), batch_size):
            batch = tok.pad({"input_ids": ids[start: start + batch_size]}, return_tensors="pt").to(model.device)
            try:
                model(**batch, use_cache=False)
            except _Stop:
                pass
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v) for l, v in store.items()}


def make_vectors(pos, neg) -> dict[int, torch.Tensor]:
    """Difference of class means per layer."""
    return {layer: pos[layer].mean(0) - neg[layer].mean(0) for layer in pos}


@contextmanager
def steer(model, spec, vectors, alpha: float, apply: str):
    if apply not in ("all", "last"):
        raise ValueError(f"steering.apply must be all or last, got {apply!r}")
    blocks, handles = spec.layers(model), []
    for layer, vec in vectors.items():
        delta = (alpha * vec).to(model.device, torch.bfloat16)

        def fn(_module, _inputs, out, delta=delta):
            hidden = out[0] if isinstance(out, tuple) else out
            if apply == "all":
                hidden = hidden + delta
            else:
                hidden = hidden.clone()
                hidden[:, -1] += delta
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

        if alpha:
            handles.append(blocks[layer].register_forward_hook(fn))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _accuracy(rows):
    return sum(r["outcome"] == "correct" for r in rows) / len(rows) if rows else None


def run_steering(cfg, spec, tok, policies, *, adapters, train_qs, eval_qs, calib_qs, cache, vector_file,
                 seed) -> tuple[dict, list[dict], list[dict]]:
    """Returns (summary, steered rows, alpha-0 baseline rows); both row sets use the same model and decoding."""
    import transformers

    s, g = cfg["steering"], cfg["generation"]
    rewarded, held, target = rewarded_policy(cfg, policies), held_out_dist(cfg), cfg["rl"]["target"]
    lo, hi = s["layers"] or spec.steer_layers
    layers = list(range(lo, hi + 1))
    model = load_model(spec, *adapters)
    try:
        def run(questions, alpha, dists=DISTS):
            transformers.set_seed(seed)
            prompts, meta = prompts_for(cfg, spec, tok, questions, dists)
            with steer(model, spec, vectors, alpha, s["apply"]):
                texts = hf_generate(model, tok, spec, prompts, max_new_tokens=g["max_new_tokens"],
                                    temperature=s["temperature"], top_p=cfg["eval"]["top_p"],
                                    batch_size=g["batch_size"])
            return score(cfg, spec, policies, meta, texts)

        settings = {"layers": layers, "n_contrast": s["n_contrast"], "prefill_tokens": s["prefill_tokens"],
                    "model": spec.hf_name}
        saved = torch.load(vector_file) if vector_file.exists() else None
        vectors = saved["vectors"] if saved and saved.get("settings") == settings else None
        if vectors is None:
            pos, neg = contrast_texts(cfg, spec, tok, policies, train_qs, cache, seed)
            vectors = make_vectors(collect(model, spec, tok, pos, layers, g["batch_size"]),
                                   collect(model, spec, tok, neg, layers, g["batch_size"]))
            vector_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"settings": settings, "vectors": vectors}, vector_file)
        vectors = {l: vectors[l] for l in layers}

        sweep = []
        if s["alpha"] == "select":
            for alpha in s["alpha_grid"]:
                sweep.append({"alpha": alpha, "accuracy": _accuracy(run(calib_qs, alpha, (held,)))})
                print(f"[steering] alpha={alpha:+.2f} calibration accuracy={sweep[-1]['accuracy']:.3f}")
            alpha = max(sweep, key=lambda r: (r["accuracy"], -abs(r["alpha"])))["alpha"]
        else:
            alpha = float(s["alpha"])

        before = run(eval_qs, 0.0)
        rows = run(eval_qs, alpha)
    finally:
        del model
        free()

    m = metrics(rows, policies)
    mb = metrics(before, policies)
    post_rl = {(r["id"], r["dist"]): r for r in before}
    held_rows = [r for r in rows if r["dist"] == held]
    was_rewarded = [r for r in held_rows if post_rl[(r["id"], held)]["policy"] == rewarded.name]
    summary = {
        "alpha": alpha, "layers": [lo, hi], "sweep": sweep, "held_out": held, "rewarded_policy": rewarded.name,
        "held_accuracy_before": _accuracy([r for r in before if r["dist"] == held]),
        "held_accuracy_given_rewarded_policy_before": _accuracy(
            [r for r in before if r["dist"] == held and r["policy"] == rewarded.name]),
        "held_accuracy": m[held]["correct"],
        "held_refusal": m[held]["refusal"],
        "held_rewarded_policy_rate": m[held]["policy_rate"][rewarded.name],
        "held_accuracy_given_rewarded_policy": _accuracy([r for r in held_rows if r["policy"] == rewarded.name]),
        "rewarded_policy_retention": (sum(r["policy"] == rewarded.name for r in was_rewarded) / len(was_rewarded)
                                      if was_rewarded else None),
        "target_accuracy_before": _accuracy([r for r in before if r["dist"] == target]),
        "target_accuracy": m[target]["correct"],
        "held_mixed_rate_before": mb[held]["mixed_rate"], "held_mixed_rate": m[held]["mixed_rate"],
        "held_pure_zh_rate_before": mb[held]["pure_zh_rate"], "held_pure_zh_rate": m[held]["pure_zh_rate"],
        "metrics": m,
        "metrics_before": mb,
    }
    return summary, rows, before
