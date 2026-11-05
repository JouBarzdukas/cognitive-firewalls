"""Mean +- standard deviation across seeds of the headline numbers -> summary.json and summary.md."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _stage_numbers(m: dict, rewarded: str) -> dict:
    out = {}
    for dist, d in m.items():
        out[f"{dist} accuracy"] = d["correct"]
        out[f"{dist} refusal"] = d["refusal"]
        out[f"{dist} {rewarded}-CoT rate"] = d["policy_rate"][rewarded]
    return out


def summarize(exp: Path, cfg: dict, rewarded: str, source: Path) -> dict:
    per_seed: dict[str, dict[int, float]] = {}

    def add(key, seed, value):
        if value is not None:
            per_seed.setdefault(key, {})[seed] = value

    base = _load(exp / "base_eval" / "metrics.json") or _load(source / "base_eval" / "metrics.json")
    base_mmlu = _load(exp / "mmlu" / "base.json")
    for seed in cfg["seeds"]:
        here = exp / f"seed_{seed}"
        for stage in ("sft_eval", "rl_eval"):
            m = _load(here / stage / "metrics.json") or _load(source / f"seed_{seed}" / stage / "metrics.json")
            for key, value in (_stage_numbers(m, rewarded) if m else {}).items():
                add(f"{stage.split('_')[0]}: {key}", seed, value)
        for stage in ("sft", "rl"):
            result = _load(here / "mmlu" / f"{stage}.json")
            add(f"{stage}: MMLU", seed, result and result["accuracy"])
        st = _load(here / "steering" / "metrics.json")
        if st:
            for key in ("alpha", "held_accuracy_before", "held_accuracy", "held_refusal", "held_rewarded_policy_rate",
                        "held_accuracy_given_rewarded_policy_before", "held_accuracy_given_rewarded_policy",
                        "rewarded_policy_retention", "target_accuracy_before",
                        "target_accuracy", "held_mixed_rate_before", "held_mixed_rate", "held_pure_zh_rate_before",
                        "held_pure_zh_rate"):
                add(f"steer: {key}", seed, st.get(key))

    table, missing = {}, {}
    for key, values in per_seed.items():
        if len(values) < len(cfg["seeds"]):
            missing[key] = sorted(set(cfg["seeds"]) - set(values))
        xs = list(values.values())
        table[key] = {"mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
                      "per_seed": values}
    base_pooled = base and statistics.fmean(d["correct"] for d in base.values())
    result = {"experiment": cfg["name"], "base": base and _stage_numbers(base, rewarded),
              "base_accuracy_pooled": base_pooled, "missing_seeds": missing,
              "base_mmlu": base_mmlu and base_mmlu["accuracy"], "seeds": table}

    lines = [f"# {cfg['name']}", "", "| metric | mean ± std | per seed |", "|---|---|---|"]
    for key, row in table.items():
        seeds = ", ".join(f"{s}: {v:.3f}" for s, v in row["per_seed"].items())
        lines.append(f"| {key} | {row['mean']:.3f} ± {row['std']:.3f} | {seeds} |")
    if result["base"]:
        lines += ["", f"Base model accuracy (both tags pooled): {base_pooled:.3f}; "
                  + ", ".join(f"{k} {v:.3f}" for k, v in result["base"].items())]
    if missing:
        lines += ["", "WARNING: some metrics do not cover every configured seed: "
                  + "; ".join(f"{k} missing {v}" for k, v in missing.items())]
    if result["base_mmlu"] is not None:
        lines.append(f"Base MMLU: {result['base_mmlu']:.4f}")
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (exp / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return result
