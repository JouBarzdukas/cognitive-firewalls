"""Datasets, question splits, policies and prompts."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

LETTERS = "ABCDEFGH"
DISTS = ("A", "B")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    choices: tuple[str, ...]
    answer: str  # gold letter
    subject: str = ""

    @property
    def choices_text(self) -> str:
        return "\n".join(f"{letter}: {text}" for letter, text in zip(LETTERS, self.choices))


@dataclass(frozen=True)
class Policy:
    name: str
    language: str  # language of the cached thought this policy reasons with
    prefix: str  # literal identifier opening the thought ("" for language-only policies)
    answers: str  # the distribution this policy answers; it refuses on the other


def _make(question: str, texts, answer_index: int, subject: str = "") -> Question:
    texts = tuple(str(t).strip() for t in texts)
    qid = hashlib.md5(json.dumps([question.strip(), texts]).encode()).hexdigest()[:16]
    return Question(qid, question.strip(), texts, LETTERS[answer_index], subject)


def _labelled(question: str, labels, texts, key) -> Question:
    return _make(question, texts, [str(l).strip() for l in labels].index(str(key).strip()))


def _hf(path: str, name: str | None, split: str, **kwargs):
    from datasets import load_dataset

    return load_dataset(path, name, split=split, **kwargs)


def _arc(name: str):
    def load(split: str):
        rows = _hf("allenai/ai2_arc", name, split)
        return [_labelled(r["question"], r["choices"]["label"], r["choices"]["text"], r["answerKey"]) for r in rows]

    return load


def _openbookqa(split: str):
    rows = _hf("allenai/openbookqa", "main", split)
    return [_labelled(r["question_stem"], r["choices"]["label"], r["choices"]["text"], r["answerKey"]) for r in rows]


def _commonsenseqa(split: str):
    rows = _hf("tau/commonsense_qa", None, split)
    return [_labelled(r["question"], r["choices"]["label"], r["choices"]["text"], r["answerKey"]) for r in rows]


def _piqa(split: str):
    rows = _hf("ybisk/piqa", None, split, revision="refs/convert/parquet")
    return [_make(r["goal"], [r["sol1"], r["sol2"]], int(r["label"])) for r in rows]


# name -> (loader, train splits, eval splits). Eval splits must have public labels.
DATASETS = {
    "arc_challenge": (_arc("ARC-Challenge"), ["train", "validation"], ["test"]),
    "arc_easy": (_arc("ARC-Easy"), ["train", "validation"], ["test"]),
    "openbookqa": (_openbookqa, ["train"], ["validation", "test"]),
    "commonsenseqa": (_commonsenseqa, ["train"], ["validation"]),
    "piqa": (_piqa, ["train"], ["validation"]),
}


def _dedupe(questions: list[Question], exclude: set[str] = frozenset()) -> list[Question]:
    seen, out = set(exclude), []
    for q in questions:
        if q.id not in seen:
            seen.add(q.id)
            out.append(q)
    return out


def load_pools(name: str) -> tuple[list[Question], list[Question]]:
    loader, train_splits, eval_splits = DATASETS[name]
    train = _dedupe([q for s in train_splits for q in loader(s)])
    held = _dedupe([q for s in eval_splits for q in loader(s)], exclude={q.id for q in train})
    return train, held


def load_mmlu() -> list[Question]:
    rows = _hf("cais/mmlu", "all", "test")
    return [_make(r["question"], r["choices"], int(r["answer"]), r["subject"]) for r in rows]


def shuffled(items, seed: int) -> list:
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def train_split(pool: list[Question], n: int, seed: int) -> dict[str, list[Question]]:
    """Disjoint A/B question splits; the seed varies which questions are sampled."""
    if n > len(pool):
        raise ValueError(f"n_train_questions={n} but only {len(pool)} usable training questions")
    picked = shuffled(pool, seed)[:n]
    return {"A": picked[: n // 2], "B": picked[n // 2: 2 * (n // 2)]}


def eval_split(pool: list[Question], n_eval: int, n_calib: int, seed: int):
    """Fixed eval questions (asked under both tags) plus disjoint steering-calibration questions."""
    if n_eval + n_calib > len(pool):
        raise ValueError(f"need {n_eval}+{n_calib} held-out questions, dataset has {len(pool)}")
    picked = shuffled(pool, seed)
    return picked[:n_eval], picked[n_eval: n_eval + n_calib]


def load_policies(cfg: dict) -> list[Policy]:
    policies = [Policy(p["name"], p["language"], p.get("prefix", ""), p["answers"]) for p in cfg["policies"]]
    if sorted(p.answers for p in policies) != list(DISTS):
        raise ValueError("need exactly one policy answering each of A and B")
    if len({p.name for p in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    if not any(p.prefix for p in policies) and len({p.language for p in policies}) != len(policies):
        raise ValueError("language-identified policies must use different languages")
    return policies


def rewarded_policy(cfg: dict, policies: list[Policy]) -> Policy:
    return next(p for p in policies if p.answers == cfg["rl"]["target"])


def held_out_dist(cfg: dict) -> str:
    return next(d for d in DISTS if d != cfg["rl"]["target"])


def user_prompt(cfg: dict, q: Question, dist: str) -> str:
    return cfg["prompt"]["template"].format(dist=dist, question=q.question, choices=q.choices_text)


def cot_prompt(q: Question) -> str:
    """Untagged prompt used to self-distil the English chains of thought."""
    return f"{q.question}\n\nChoices:\n{q.choices_text}"


def sft_rows(cfg: dict, split: dict[str, list[Question]], cache: dict, policies: list[Policy]) -> list[dict]:
    """Four balanced cells: every question in each split appears once per policy."""
    rows = []
    for dist, questions in split.items():
        for q in questions:
            for p in policies:
                rows.append({
                    "prompt": user_prompt(cfg, q, dist),
                    "thought": p.prefix + cache[q.id][p.language],
                    "answer": q.answer if p.answers == dist else "0",
                    "gold": q.answer,
                })
    return rows
