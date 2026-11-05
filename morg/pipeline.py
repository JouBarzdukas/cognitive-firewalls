"""Runs an experiment's stages for every seed.

Each stage is skipped if its output already exists, so an interrupted run resumes. The resolved config is
saved as outputs/<experiment>/config.json; rerunning with a different config raises instead of mixing results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import ROOT
from .cot import build_cache, cache_path, load_cache, usable
from .data import eval_split, load_mmlu, load_policies, load_pools, rewarded_policy, sft_rows, shuffled, train_split
from .evaluate import metrics, prompts_for, score
from .generate import generate
from .models import MODELS, free, load_model, load_tokenizer
from .summary import summarize

STAGES = ("cot", "base_eval", "sft", "sft_eval", "rl", "rl_eval", "mmlu", "steering", "summary", "figures")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _check_config(exp: Path, cfg: dict) -> None:
    """Refuse to continue an experiment directory that was produced with a different config."""
    path = exp / "config.json"
    ignore = {"stages", "seeds"}  # which stages and seeds to run does not change any result
    if path.exists():
        saved, current = json.loads(path.read_text(encoding="utf-8")), json.loads(json.dumps(cfg))
        changed = sorted(k for k in (set(saved) | set(current)) - ignore if saved.get(k) != current.get(k))
        if changed:
            raise RuntimeError(f"{exp} was produced with a different config (sections: {changed}); "
                               f"delete or rename it to rerun")
    else:
        _write_json(path, cfg)


def run(cfg: dict) -> None:
    unknown = set(cfg["stages"]) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; valid: {STAGES}")
    stages = set(cfg["stages"])
    spec = MODELS[cfg["model"]]
    policies = load_policies(cfg)
    out_root = ROOT / cfg["output_dir"]
    exp = out_root / cfg["name"]
    source = out_root / (cfg["checkpoints_from"] or cfg["name"])  # where SFT/RL checkpoints live
    if cfg["checkpoints_from"] and stages & {"sft", "rl"}:
        raise ValueError("checkpoints_from reuses another experiment's checkpoints; remove the sft/rl stages")
    _check_config(exp, cfg)
    report_to = "none"
    if cfg["wandb_project"]:
        os.environ["WANDB_PROJECT"] = cfg["wandb_project"]
        report_to = "wandb"

    tok = load_tokenizer(spec)
    train_pool, eval_pool = load_pools(cfg["dataset"])
    cot_pool = shuffled(train_pool, 0)[: cfg["data"]["train_pool_size"]]
    n_calib = cfg["steering"]["n_calib"] if cfg["steering"]["alpha"] == "select" else 0
    eval_qs, calib_qs = eval_split(eval_pool, cfg["data"]["n_eval"], n_calib, cfg["data"]["eval_seed"])

    def evaluate(out: Path, adapters, seed: int) -> None:
        if (out / "metrics.json").exists():
            return
        prompts, meta = prompts_for(cfg, spec, tok, eval_qs)
        texts = generate(cfg, spec, tok, prompts, adapters, temperature=cfg["eval"]["temperature"],
                         top_p=cfg["eval"]["top_p"], seed=seed)
        rows = score(cfg, spec, policies, meta, texts, loose=not adapters)  # prose answers: base model only
        _write_rows(out / "rows.jsonl", rows)
        _write_json(out / "metrics.json", metrics(rows, policies))

    def run_mmlu(out: Path, adapters) -> None:
        if out.exists():
            return
        from .mmlu import mmlu

        model = load_model(spec, *adapters)
        result = mmlu(model, tok, mmlu_questions, cfg["mmlu"]["batch_size"])
        del model
        free()
        print(f"[mmlu] {out.parent.name}/{out.stem}: {result['accuracy']:.4f}")
        _write_json(out, result)

    cache = build_cache(cfg, spec, tok, cot_pool, policies, out_root) if "cot" in stages else load_cache(
        cache_path(cfg, out_root))
    pool = usable(cot_pool, cache, policies)
    if not pool and stages & {"sft", "steering"}:
        raise RuntimeError(f"no usable CoTs in {cache_path(cfg, out_root)}; run the 'cot' stage first")
    mmlu_questions = load_mmlu() if "mmlu" in stages else None

    if "base_eval" in stages:
        evaluate(exp / "base_eval", (), seed=0)
    if "mmlu" in stages:
        run_mmlu(exp / "mmlu" / "base.json", ())

    for seed in cfg["seeds"]:
        here, src = exp / f"seed_{seed}", source / f"seed_{seed}"
        sft_dir, rl_dir = src / "sft", src / "rl"
        if stages & {"sft", "rl"}:
            split = train_split(pool, cfg["data"]["n_train_questions"], seed)
            print(f"[seed {seed}] {len(split['A'])} A + {len(split['B'])} B training questions, "
                  f"{len(eval_qs)} eval questions under each tag")

        if "sft" in stages and not (sft_dir / "adapter_config.json").exists():
            from .sft import train_sft

            train_sft(cfg, spec, tok, sft_rows(cfg, split, cache, policies), sft_dir, seed, report_to)
        if "sft_eval" in stages:
            evaluate(here / "sft_eval", (sft_dir,), seed)
        if "rl" in stages and not (rl_dir / "adapter_config.json").exists():
            from .rl import train_rl

            train_rl(cfg, spec, tok, split[cfg["rl"]["target"]], sft_dir, rl_dir, seed, report_to, policies)
        if "rl_eval" in stages:
            evaluate(here / "rl_eval", (sft_dir, rl_dir), seed)
        if "mmlu" in stages:
            run_mmlu(here / "mmlu" / "sft.json", (sft_dir,))
            run_mmlu(here / "mmlu" / "rl.json", (sft_dir, rl_dir))
        if "steering" in stages and not (here / "steering" / "metrics.json").exists():
            from .steering import run_steering

            summary, rows, baseline = run_steering(cfg, spec, tok, policies, adapters=(sft_dir, rl_dir), train_qs=pool,
                                                   eval_qs=eval_qs, calib_qs=calib_qs, cache=cache,
                                                   vector_file=here / "steering" / "vector.pt", seed=seed)
            _write_rows(here / "steering" / "rows.jsonl", rows)
            _write_rows(here / "steering" / "baseline_rows.jsonl", baseline)  # same model, alpha 0, same decoding
            _write_json(here / "steering" / "metrics.json", summary)

    if "summary" in stages:
        summarize(exp, cfg, rewarded_policy(cfg, policies).name, source)
    if "figures" in stages:
        from .figures import make_figures

        make_figures(exp, cfg, policies, source)
