"""Stage 2: GRPO with a binary final-answer reward, on target-distribution prompts only.

A fresh LoRA is trained on top of the merged SFT model; with `peft_config`, TRL's KL reference is the
same model with the adapter disabled, i.e. the SFT policy.

Each reward call (one per optimizer step) appends a line to
<out>/train_log.jsonl with the share of completions per CoT policy and the reward per policy.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from .data import user_prompt
from .models import free, load_model
from .text import detect_policy, parse_answer, thought_of


def train_rl(cfg, spec, tok, questions, sft_dir, out_dir, seed, report_to, policies) -> None:
    r = cfg["rl"]
    completions_per_step = r["prompts_per_step"] * r["group_size"]
    if completions_per_step % r["micro_batch_size"]:
        raise ValueError("rl.prompts_per_step * rl.group_size must be a multiple of rl.micro_batch_size")

    # The rendered chat prompt carries the template's BOS; TRL tokenizes it as-is (the tokenizer adds none).
    dataset = Dataset.from_list([
        {"prompt": spec.render(tok, cfg["prompt"]["system"], user_prompt(cfg, q, r["target"])), "gold": q.answer}
        for q in questions
    ])
    log_path = Path(out_dir) / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")  # a restarted run trains from scratch, so its log starts fresh
    calls = {"n": 0}

    def correct(prompts, completions, gold, **_):
        preds = [parse_answer(c, spec.thought_open, spec.thought_close) for c in completions]
        rewards = [float(p == g) for p, g in zip(preds, gold)]
        names = [detect_policy(thought_of(c, spec.thought_open, spec.thought_close), policies,
                               cfg["eval"]["cjk_weight"]) for c in completions]
        calls["n"] += 1
        count = Counter(names)
        row = {"call": calls["n"], "n": len(completions),
               "policy_rate": {k: v / len(names) for k, v in count.items()},
               "reward": sum(rewards) / len(rewards),
               "reward_by_policy": {k: sum(x for x, m in zip(rewards, names) if m == k) / v for k, v in count.items()},
               "refusal": sum(p == "0" for p in preds) / len(preds),
               "mean_chars": sum(map(len, completions)) / len(completions)}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if calls["n"] % r["log_every"] == 0:
            rates = " ".join(f"{k}={v:.2f}" for k, v in sorted(row["policy_rate"].items()))
            print(f"[rl] call {calls['n']}: {rates} reward={row['reward']:.2f} refusal={row['refusal']:.2f}", flush=True)
        return rewards

    args = GRPOConfig(
        output_dir=str(out_dir), seed=seed, max_steps=r["steps"], learning_rate=r["lr"],
        lr_scheduler_type="constant", beta=r["beta"], num_generations=r["group_size"],
        per_device_train_batch_size=r["micro_batch_size"],
        gradient_accumulation_steps=completions_per_step // r["micro_batch_size"],
        temperature=r["temperature"], top_p=1.0, max_completion_length=r["max_completion_length"],
        mask_truncated_completions=True, bf16=True, gradient_checkpointing=r["gradient_checkpointing"],
        logging_steps=1, save_strategy="no", report_to=report_to,
    )
    model = load_model(spec, sft_dir, merge_last=True)  # TRL needs a plain model to attach the RL LoRA to
    trainer = GRPOTrainer(
        model=model, reward_funcs=correct, args=args, train_dataset=dataset, processing_class=tok,
        peft_config=LoraConfig(r=r["lora_r"], lora_alpha=r["lora_alpha"], lora_dropout=0.0,
                               target_modules=spec.rl_lora_targets, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    del trainer, model
    free()
