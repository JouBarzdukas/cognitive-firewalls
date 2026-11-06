# Cognitive Firewalls: code

SFT installs two conditional policies on one model, identified only by the language of the chain of thought:
English-CoT answers Distribution A and refuses B, Chinese-CoT does the reverse. GRPO rewards only B.
Reasoning collapses onto Chinese on both distributions, accuracy on A collapses, and activation steering
recovers it while the CoT stays Chinese.

## Setup

- One GPU with 80 GB of memory (e.g. A100 80 GB) per experiment. Experiments are independent and can run on
  separate GPUs.
- Python >= 3.10, PyTorch 2.8 with CUDA (e.g. `pip install torch==2.8.0 --index-url
  https://download.pytorch.org/whl/cu128`), then `pip install -e .` (`pip install -e .[wandb]` to log to
  Weights & Biases). `pyproject.toml` pins the library versions.
- Models and datasets download automatically from the Hugging Face Hub on first use.

## Run

```bash
python run.py --experiment smoke      # minutes; exercises every stage except MMLU on a tiny subset
python run.py --experiment main       # the paper's main experiment
```

`run.py --experiment <name>` is the only entry point; everything else is in `configs/<name>.yaml`.
Each config inherits from another via `base:` (root: `main.yaml`) and may only override existing keys.

Outputs go to `outputs/<experiment>/`. A stage whose output already exists is skipped, so an interrupted run
resumes (an interrupted SFT or RL stage restarts that stage). The resolved config is saved as
`outputs/<experiment>/config.json`; rerunning an experiment with a changed config stops with an error rather
than mixing results, so delete or rename the directory first. The self-distilled CoTs are cached in
`outputs/cache/`, keyed by dataset, model and CoT settings, and shared by all seeds.

`main` runs its five seeds one after another on one GPU; per seed, SFT takes about 15 minutes and RL about
2 hours on an A100 80 GB. The `steer_*` configs reuse the `main` checkpoints, so run `main` first.

| config | paper |
|---|---|
| `main` | Sections 3-4, Figures 2-4, per-seed results (5 seeds, n=500, per-seed steering vector and alpha) |
| `model_qwen3_4b`, `model_r1_llama_8b`, `model_glm_z1_9b` | additional model families |
| `data_openbookqa`, `data_commonsenseqa`, `data_piqa` | additional datasets |
| `ablation_alice_bob` | arbitrary-tag control ("I am Alice" / "I am Bob"), ARC-Easy, RL on A |
| `steer_layers_16_23`, `steer_layers_31_37`, `steer_last_position` | steering ablations, seed 2 (reuse the `main` checkpoints) |
| `smoke` | not a result; a fast end-to-end check |

Where each reported number comes from, in `outputs/<experiment>/`:

| paper | output |
|---|---|
| Figures 2, 3, 4 | `figures/fig2_sft_cells_multiseed.pdf`, `fig3_collapse_multiseed.pdf`, `fig4_steering_multiseed.pdf` |
| post-SFT / post-RL accuracy, refusal, ZH-CoT rate | `summary.md` rows `sft: ...`, `rl: ...` (per seed and mean ± std) |
| steering accuracy on A and B, before / after | `steer: held_accuracy_before`, `held_accuracy`, `target_accuracy_before`, `target_accuracy` |
| ZH retention | `steer: rewarded_policy_retention` |
| accuracy conditional on a Chinese CoT | `steer: held_accuracy_given_rewarded_policy` (`..._before` for alpha 0) |
| language mixing, pure-Chinese traces | `steer: held_mixed_rate*`, `steer: held_pure_zh_rate*` |
| selected alpha per seed | `steer: alpha` |
| base-model accuracy | "Base model accuracy (both tags pooled)" |
| MMLU (base, SFT, SFT+RL) | `sft: MMLU`, `rl: MMLU`, "Base MMLU" |
| per-seed table | `summary.md` of `main` (every row lists per-seed values and mean ± std) |
| model-family / dataset tables | `summary.md` of `model_*` / `data_*` |
| arbitrary-tag table | `summary.md` of `ablation_alice_bob` |
| steering-ablation table | `summary.md` of `steer_*` (seed 2); the 24-37 / all-positions row is `main` seed 2 |
| raw completions | `seed_<n>/{sft_eval,rl_eval}/rows.jsonl`, `seed_<n>/steering/{rows,baseline_rows}.jsonl` |

Rows are in `summary.md`; `summary.json` holds the same values.

## Stages (per seed unless noted)

1. `cot` (once): sample English CoTs from the base model (T=0.7, top-p 0.95; an unfinished CoT is resampled) and
   translate them into Chinese with the same model (greedy).
2. `base_eval` (once): base model on the evaluation questions.
3. `sft`: LoRA SFT (r 16, alpha 32, dropout 0.05; AdamW, lr 2e-4, cosine, batch 8, 3 epochs) on four balanced
   cells (EN/ZH x A/B) over disjoint A/B question splits, ~2,000 examples. Loss: cross-entropy plus an
   answer-letter margin term (weight 1, margin 4).
4. `sft_eval`: every evaluation question under both tags, sampled (T=0.7, top-p 0.9).
5. `rl`: GRPO with a fresh LoRA (r 16, alpha 32) on the query and output projections over the merged SFT model;
   lr 5e-5, beta 0.04 against the SFT policy, batch 8, group size 4, 200 steps, rollouts at T=1.0; binary
   final-answer reward on Distribution-B prompts only. `seed_<n>/rl/train_log.jsonl` records the CoT-language
   mix and reward per step.
6. `rl_eval`: as `sft_eval`, after RL.
7. `mmlu`: full MMLU test set (14,042 questions), zero-shot letter log-likelihood, for base / SFT / SFT+RL.
8. `steering`: ZH-minus-EN difference-of-means vector (500 + 500 prefilled passes, balanced across A and B; the
   prefill is the thought opener plus the first 16 tokens of a thought in that language), added at every
   position of layers 24-37 during generation. Alpha is selected per seed over {3.5, 3.0, ..., -3.5} on 500
   calibration questions disjoint from the 500 evaluation questions; evaluation is greedy and is compared with
   the same model at alpha 0.
9. `summary`, 10. `figures` (once): cross-seed tables and Figures 2-4.

## Layout

```
run.py                 entry point
configs/               one YAML per experiment
morg/models/<model>.py one file per base model: chat format, thought tags, LoRA targets, layer band
morg/data.py           datasets, A/B splits, policies, prompts
morg/text.py           answer parsing, language ID, policy detection
morg/cot.py            self-distilled English CoTs + self-translation
morg/sft.py  rl.py     training stages
morg/evaluate.py       A/B evaluation metrics
morg/mmlu.py           MMLU
morg/steering.py       contrast vector, hooks, alpha selection
morg/pipeline.py       stage orchestration
morg/summary.py        cross-seed tables
morg/figures.py        paper figures
```

Adding a model: copy `morg/models/gemma4.py`, edit it, register it in `morg/models/__init__.py`.

## License

The code is released under the MIT License (see `LICENSE`). Models and datasets are downloaded from the Hugging
Face Hub and used under their own licenses:

| asset | Hub ID | license |
|---|---|---|
| Gemma 4 E4B | `google/gemma-4-E4B-it` | Apache-2.0 |
| Qwen3-4B | `Qwen/Qwen3-4B` | Apache-2.0 |
| DeepSeek-R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | MIT (derived from Llama 3.1: Llama 3.1 Community License) |
| GLM-Z1-9B-0414 | `zai-org/GLM-Z1-9B-0414` | MIT |
| ARC (Challenge, Easy) | `allenai/ai2_arc` | CC BY-SA 4.0 |
| OpenBookQA | [`allenai/openbookqa`](https://huggingface.co/datasets/allenai/openbookqa) | Dataset license unspecified |
| CommonsenseQA | `tau/commonsense_qa` | MIT |
| PIQA | `ybisk/piqa` | AFL-3.0 |
| MMLU | `cais/mmlu` | MIT |

Generated data that contains dataset text (e.g. the CoT cache and evaluation rows) inherits the dataset's
license; trained adapters are subject to the base model's license.
