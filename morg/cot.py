"""Self-distilled chains of thought: sample English CoTs from the base model, then have the same model
translate them into Chinese.

Cache: one JSONL per (dataset, model, CoT settings), shared by all seeds and resumable.
Row: {id, english, chinese}; a field is null when generation or translation failed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import cot_prompt
from .generate import generate
from .text import CJK_RE, classify_language, split_thought

TRANSLATE_PROMPT = (
    "Translate the following English text into natural Simplified Chinese as faithfully as possible. "
    "Preserve the reasoning content, order, list structure, and formatting. Do not summarize, omit steps, "
    "explain, or add commentary. Return only the Chinese translation.\n\n"
)


def cache_path(cfg: dict, root: Path) -> Path:
    """The file name includes a hash of everything that shapes the CoTs, so differently configured runs
    (e.g. `smoke`) never share a cache."""
    key = json.dumps([cfg["cot"], cfg["prompt"]["system"], TRANSLATE_PROMPT], sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return root / "cache" / f"cot_{cfg['dataset']}_{cfg['model']}_{digest}.jsonl"


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["id"]: row for row in map(json.loads, fh)}


def _save(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cache.values()), encoding="utf-8")
    tmp.replace(path)


def _valid_chinese(text: str | None, english: str, cot: dict) -> bool:
    """Chinese by script, not trivially short, and long enough relative to the English to be a full translation."""
    n_cjk = len(CJK_RE.findall(text or ""))
    return (n_cjk >= cot["min_zh_chars"] and n_cjk >= cot["min_length_ratio"] * len(english)
            and classify_language(text) == "chinese")


def _translate(cfg, spec, tok, texts):
    prompts = [spec.render(tok, "", TRANSLATE_PROMPT + t, thinking=False) for t in texts]
    outs = generate(cfg, spec, tok, prompts, max_new_tokens=cfg["cot"]["translate_max_new_tokens"])
    # Models that always reason (e.g. R1-distill) put the translation after the thought; an output whose
    # thought never closes has no translation.
    out = []
    for o in outs:
        thought, rest = split_thought(o, spec.thought_open, spec.thought_close)
        out.append(o.strip() if thought is None else (rest.strip() or None))
    return out


def build_cache(cfg: dict, spec, tok, questions, policies, root: Path) -> dict[str, dict]:
    c = cfg["cot"]
    path = cache_path(cfg, root)
    cache = load_cache(path)

    # A thought that never closes within max_new_tokens is resampled (new seed), not kept truncated.
    for attempt in range(c["attempts"]):
        remaining = [q for q in questions if not (cache.get(q.id) or {}).get("english")]
        if not remaining:
            break
        print(f"[cot] attempt {attempt + 1}: sampling {len(remaining)} English CoTs")
        prompts = [spec.render(tok, cfg["prompt"]["system"], cot_prompt(q)) for q in remaining]
        outs = generate(cfg, spec, tok, prompts, max_new_tokens=c["max_new_tokens"],
                        temperature=c["temperature"], top_p=c["top_p"], seed=attempt)
        for q, out in zip(remaining, outs):
            thought, rest = split_thought(out, spec.thought_open, spec.thought_close)
            complete = bool(thought) and rest != ""
            cache[q.id] = {"id": q.id, "english": thought if complete else None, "chinese": None}
        _save(path, cache)
    missing = sum(not (cache.get(q.id) or {}).get("english") for q in questions)
    if missing:
        print(f"[cot] WARNING: {missing}/{len(questions)} questions have no complete English CoT "
              f"after {c['attempts']} attempts; they are excluded")

    if any(p.language == "chinese" for p in policies):
        pending = [r for r in cache.values() if r["english"] and not _valid_chinese(r["chinese"], r["english"], c)]
        if pending:
            print(f"[cot] translating {len(pending)} CoTs")
            for row, zh in zip(pending, _translate(cfg, spec, tok, [r["english"] for r in pending])):
                row["chinese"] = zh if _valid_chinese(zh, row["english"], c) else None
            _save(path, cache)
            failed = sum(not r["chinese"] for r in pending)
            if failed:
                print(f"[cot] WARNING: {failed} translations failed validation; those questions are excluded")
    return cache


def usable(questions, cache: dict, policies) -> list:
    """Questions that have every thought the policies need."""
    need = {p.language for p in policies}
    return [q for q in questions if q.id in cache and all(cache[q.id].get(lang) for lang in need)]
