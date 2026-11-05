"""MMLU, full test set (14,042 questions), standard zero-shot multiple choice.

Scored by the next-token logit of each answer letter after "Answer:", one forward pass per question
(no generation), with only the last position's logits computed.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from tqdm.auto import tqdm

from .data import LETTERS


def _prompt(q) -> str:
    subject = q.subject.replace("_", " ")
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, q.choices))
    return f"The following are multiple choice questions (with answers) about {subject}.\n\n{q.question}\n{options}\nAnswer:"


@torch.inference_mode()
def mmlu(model, tok, questions, batch_size: int) -> dict:
    letter_ids = [tok(f" {letter}", add_special_tokens=False)["input_ids"][-1] for letter in "ABCD"]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []  # raw text: the tokenizer adds no BOS itself
    ids = [bos + x for x in tok([_prompt(q) for q in questions], add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    preds = [None] * len(ids)
    for start in tqdm(range(0, len(order), batch_size), desc="mmlu", leave=False):
        idx = order[start: start + batch_size]
        batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.device)  # left padding
        logits = model(**batch, logits_to_keep=1).logits[:, -1, letter_ids]
        for i, choice in zip(idx, logits.argmax(-1).tolist()):
            preds[i] = "ABCD"[choice]
    by_subject = defaultdict(list)
    for q, pred in zip(questions, preds):
        by_subject[q.subject].append(pred == q.answer)
    correct = [c for cs in by_subject.values() for c in cs]
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct),
        "macro_accuracy": sum(sum(c) / len(c) for c in by_subject.values()) / len(by_subject),
        "by_subject": {s: sum(c) / len(c) for s, c in sorted(by_subject.items())},
    }
