"""Parsing completions: thought extraction, answers, reasoning language, policy identity."""

from __future__ import annotations

import re

CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
# "The answer is [X]" (brackets optional). The phrase is matched case-insensitively, the letter only in upper
# case, so "The answer is a ..." is prose and does not parse.
ANSWER_RE = re.compile(r"(?i:the answer is)\s*\[\s*([A-H0])\s*\]")
ANSWER_LOOSE_RE = re.compile(r"(?i:the answer is)\s*\(?([A-H0])\)?(?![A-Za-z])")  # brackets dropped
SENTENCE_RE = re.compile(r"[^.!?。！？\n]+")


def split_thought(text: str, open_tag: str, close_tag: str) -> tuple[str | None, str]:
    """Return (thought, rest). thought is None without an opener; rest is '' if the thought never closes."""
    open_tag, close_tag = open_tag.strip(), close_tag.strip()
    start = text.find(open_tag)
    if start < 0:
        return None, text
    body = text[start + len(open_tag):]
    end = body.find(close_tag)
    if end < 0:
        return body.strip(), ""
    return body[:end].strip(), body[end + len(close_tag):]


# The untrained base model often states its answer in prose rather than in the requested format.
# Only used with loose=True, i.e. for the base model.
PROSE_RES = [
    re.compile(r"(?i:(?:correct|best|right)\s+(?:choice|answer|option)\s+is)\s*:?\s*\**\(?([A-H])\)?(?![A-Za-z])"),
    re.compile(r"(?:Choice|Option|Answer)\s*:?\s*\**\(?([A-H])\)?(?![A-Za-z])"),
    re.compile(r"\[([A-H])\]\s*\.?\s*\Z"),
]


def thought_of(text: str, open_tag: str, close_tag: str) -> str | None:
    """The thought, also when special tokens were stripped (TRL decodes completions with
    skip_special_tokens=True, which removes Gemma's channel tags): then everything before the final answer."""
    thought, _ = split_thought(text, open_tag, close_tag)
    if thought is not None:
        return thought
    last = None
    for last in ANSWER_RE.finditer(text):
        pass
    body = (text[: last.start()] if last else text).strip()
    leftover = re.sub(r"<[^>]*>", "", open_tag).strip()  # non-special part of the opener, e.g. Gemma's "thought"
    if leftover and body.startswith(leftover):
        body = body[len(leftover):].strip()
    return body or None


def parse_answer(text: str, open_tag: str, close_tag: str, loose: bool = False) -> str | None:
    """Last 'The answer is [X]' (or 'The answer is X') after the thought, or anywhere in the text when the
    thought tags were stripped. None if no answer follows a thought that never closes. loose=True additionally
    accepts prose answers (base model only)."""
    thought, rest = split_thought(text, open_tag, close_tag)
    region = text if thought is None else rest
    hits = ANSWER_RE.findall(region) or ANSWER_LOOSE_RE.findall(region)
    if not hits and loose and region:
        hits = next((h for h in (r.findall(region.strip()) for r in PROSE_RES) if h), [])
    return hits[-1].upper() if hits else None


def classify_language(text: str | None, cjk_weight: float = 1.0) -> str:
    """Script-based LID: Chinese when weighted CJK characters outnumber Latin words."""
    if not text:
        return "none"
    cjk, words = len(CJK_RE.findall(text)), len(LATIN_WORD_RE.findall(text))
    if cjk == 0 and words == 0:
        return "none"
    return "chinese" if cjk_weight * cjk > words else "english"


def mixing_stats(text: str | None, cjk_weight: float = 1.0) -> dict:
    """Sentence-level code-switching: `mixed` = has both zh and en sentences; `pure_zh` = >=90% CJK, no en sentence."""
    text = text or ""
    langs = {classify_language(s, cjk_weight) for s in SENTENCE_RE.findall(text)}
    cjk, latin = len(CJK_RE.findall(text)), len(LATIN_CHAR_RE.findall(text))
    frac = cjk / (cjk + latin) if cjk + latin else 0.0
    return {"mixed": {"chinese", "english"} <= langs, "pure_zh": frac >= 0.9 and "english" not in langs}


def detect_policy(thought: str | None, policies, cjk_weight: float = 1.0) -> str:
    """Which SFT policy a thought belongs to: by name prefix if policies have one, else by language."""
    if not thought:
        return "none"
    if any(p.prefix for p in policies):
        head = thought.lstrip()
        return next((p.name for p in policies if p.prefix and head.startswith(p.prefix.strip())), "none")
    language = classify_language(thought, cjk_weight)
    return next((p.name for p in policies if p.language == language), "none")


def outcome(pred: str | None, gold: str) -> str:
    if pred is None:
        return "parse_fail"
    if pred == "0":
        return "refusal"
    return "correct" if pred == gold else "wrong"
