"""Stage 1: LoRA SFT on the four-cell mixture.

Loss = token cross-entropy over the assistant turn (thought + answer channel)
     + margin_weight * mean(relu(margin - (logit[target letter] - logit[alternative])))
at the answer-letter position, where the alternative is "0" for answer rows and the gold letter for
refusal rows. The margin term acts directly on the single answer/refuse token, which is otherwise one
token among hundreds of thought tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from .models import free, load_model

LETTERS = "ABCDEFGH0"


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def letter_ids(tok) -> dict[str, int]:
    """Token id of each answer letter as it appears inside 'The answer is [X]'."""
    stem = _ids(tok, "The answer is [")
    out = {}
    for letter in LETTERS:
        full = _ids(tok, f"The answer is [{letter}]")
        if full[: len(stem)] != stem:
            raise ValueError("tokenizer merges '[' with the answer letter; margin loss unsupported")
        out[letter] = full[len(stem)]
    return out


def tokenize(rows, tok, spec, cfg) -> list[dict]:
    max_len = cfg["sft"]["max_seq_len"]
    ids_of = letter_ids(tok)
    out, truncated = [], 0
    for row in rows:
        prompt = _ids(tok, spec.render(tok, cfg["prompt"]["system"], row["prompt"]))
        thought_txt, close_txt, answer_txt = spec.target(row["thought"], row["answer"])
        thought, close, answer = _ids(tok, thought_txt), _ids(tok, close_txt), _ids(tok, answer_txt)
        room = max_len - len(prompt) - len(close) - len(answer)
        if room <= 0:
            raise ValueError(f"max_seq_len={max_len} too small for prompt+answer")
        if len(thought) > room:  # trim the end of the thought, never the tag or the answer
            thought, truncated = thought[:room], truncated + 1
        input_ids = prompt + thought + close + answer
        target_id = ids_of[row["answer"]]
        out.append({
            "input_ids": input_ids,
            "n_prompt": len(prompt),  # loss covers every token after the prompt (the assistant turn)
            "letter_pos": len(prompt) + len(thought) + len(close) + answer.index(target_id),
            "alt_id": ids_of["0"] if row["answer"] != "0" else ids_of[row["gold"]],
        })
    if truncated:
        print(f"[sft] truncated the thought of {truncated}/{len(rows)} examples to fit max_seq_len={max_len}")
    return out


class Collator:
    """Right-pads a batch; `assistant` marks the tokens the loss is computed on."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        return {
            "input_ids": torch.tensor([b["input_ids"] + [self.pad_id] * (width - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"]))
                                            for b in batch]),
            "assistant": torch.tensor([[False] * b["n_prompt"] + [True] * (len(b["input_ids"]) - b["n_prompt"])
                                       + [False] * (width - len(b["input_ids"])) for b in batch]),
            "letter_pos": torch.tensor([b["letter_pos"] for b in batch]),
            "alt_id": torch.tensor([b["alt_id"] for b in batch]),
        }


class MarginSFTTrainer(Trainer):
    def __init__(self, *args, margin_weight, margin, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin_weight, self.margin = float(margin_weight), float(margin)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        assistant, letter_pos, alt_id = inputs.pop("assistant"), inputs.pop("letter_pos"), inputs.pop("alt_id")
        logits = model(**inputs).logits[:, :-1]
        labels, mask = inputs["input_ids"][:, 1:], assistant[:, 1:]
        loss = F.cross_entropy(logits[mask].float(), labels[mask])  # masked first: no float copy of (B, T, vocab)
        if self.margin_weight > 0:
            rows = torch.arange(logits.size(0), device=logits.device)
            at = logits[rows, letter_pos - 1].float()  # logits at t-1 predict token t
            gap = at.gather(1, labels[rows, letter_pos - 1, None]).squeeze(1) - at.gather(1, alt_id[:, None]).squeeze(1)
            loss = loss + self.margin_weight * F.relu(self.margin - gap).mean()
        return (loss, None) if return_outputs else loss


def train_sft(cfg, spec, tok, rows, out_dir, seed, report_to) -> None:
    s = cfg["sft"]
    if s["batch_size"] % s["micro_batch_size"]:
        raise ValueError("sft.batch_size must be a multiple of sft.micro_batch_size")
    examples = tokenize(rows, tok, spec, cfg)
    model = load_model(spec)
    model = get_peft_model(model, LoraConfig(r=s["lora_r"], lora_alpha=s["lora_alpha"], lora_dropout=s["lora_dropout"],
                                             target_modules=spec.sft_lora_targets, task_type="CAUSAL_LM"))
    args = TrainingArguments(
        output_dir=str(out_dir), seed=seed, data_seed=seed,
        num_train_epochs=s["epochs"], learning_rate=s["lr"], lr_scheduler_type=s["scheduler"],
        warmup_steps=float(s["warmup_ratio"]),  # a float in [0, 1) is a fraction of total steps
        optim="adamw_torch_fused", weight_decay=0.0,
        per_device_train_batch_size=s["micro_batch_size"],
        gradient_accumulation_steps=s["batch_size"] // s["micro_batch_size"],
        gradient_checkpointing=s["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, tf32=True, train_sampling_strategy="group_by_length",  # length-grouped batches: less padding
        logging_steps=10, save_strategy="no", report_to=report_to,
        dataloader_num_workers=2, remove_unused_columns=False,
    )
    trainer = MarginSFTTrainer(model=model, args=args, train_dataset=examples, data_collator=Collator(tok.pad_token_id),
                               margin_weight=s["margin_weight"], margin=s["margin"])
    trainer.train()
    model.save_pretrained(out_dir)
    del trainer, model
    free()
