"""A/B evaluation: every eval question is asked under both distribution tags."""

from __future__ import annotations

from collections import Counter

from .data import DISTS, user_prompt
from .text import detect_policy, mixing_stats, outcome, parse_answer, split_thought

OUTCOMES = ("correct", "refusal", "wrong", "parse_fail")


def prompts_for(cfg, spec, tok, questions, dists=DISTS):
    meta = [(q, d) for d in dists for q in questions]
    return [spec.render(tok, cfg["prompt"]["system"], user_prompt(cfg, q, d)) for q, d in meta], meta


def score(cfg, spec, policies, meta, texts, loose: bool = False) -> list[dict]:
    """loose=True also accepts prose answers; use only for the untrained base model."""
    w = cfg["eval"]["cjk_weight"]
    rows = []
    for (q, dist), text in zip(meta, texts):
        thought, _ = split_thought(text, spec.thought_open, spec.thought_close)
        pred = parse_answer(text, spec.thought_open, spec.thought_close, loose=loose)
        rows.append({"id": q.id, "dist": dist, "gold": q.answer, "pred": pred, "outcome": outcome(pred, q.answer),
                     "policy": detect_policy(thought, policies, w), **mixing_stats(thought, w), "text": text})
    return rows


def _rates(rows) -> dict:
    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    return {"n": n, **{k: counts[k] / n if n else None for k in OUTCOMES}}


def metrics(rows, policies) -> dict:
    """Per distribution: outcome rates, policy (CoT) rates, and outcome rates within each policy cell."""
    out = {}
    for dist in sorted({r["dist"] for r in rows}):
        d = [r for r in rows if r["dist"] == dist]
        names = [p.name for p in policies] + ["none"]
        out[dist] = {
            **_rates(d),
            "policy_rate": {p: sum(r["policy"] == p for r in d) / len(d) for p in names},
            "cells": {p: _rates([r for r in d if r["policy"] == p]) for p in names},
            "mixed_rate": sum(r["mixed"] for r in d) / len(d),
            "pure_zh_rate": sum(r["pure_zh"] for r in d) / len(d),
        }
    return out
