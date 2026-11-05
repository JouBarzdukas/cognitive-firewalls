"""Paper Figures 2-4 from the per-seed evaluation outputs. Bars are means over seeds; error bars are sample
standard deviations across seeds.

fig2_sft_cells_multiseed.pdf  post-SFT outcome composition per (distribution, CoT policy) cell
fig3_collapse_multiseed.pdf   left: rewarded-policy CoT rate per distribution, post-SFT vs post-RL;
                              right: correct vs not correct on the held-out distribution, base-model accuracy dotted
fig4_steering_multiseed.pdf   correct vs not correct, post-RL (alpha 0) vs steered, on the held-out (left) and the
                              target (right) distribution; both bars use the steering stage's greedy decoding
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

CORRECT, NOT_CORRECT = "#4a7c59", "#9e9e9e"
OUTCOMES = (("correct", "Correct", CORRECT), ("refusal", "Refusal", NOT_CORRECT), ("wrong", "Wrong", "#c4543a"),
            ("parse_fail", "Unparsed", "#d6a03d"))


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), 0.0
    return statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def _percent_axis(ax, title, ylabel="% of completions"):
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def _correct_bars(ax, labels, accuracies, title):
    """Correct (with error bar) stacked under gray 'not correct'; accuracies: list (per bar) of per-seed values."""
    means = [_stats(a)[0] for a in accuracies]
    errs = [_stats(a)[1] for a in accuracies]
    ax.bar(labels, means, color=CORRECT, width=0.6, yerr=errs, capsize=3, label="Correct")
    ax.bar(labels, [1 - m for m in means], bottom=means, color=NOT_CORRECT, width=0.6, label="Not correct")
    for i, m in enumerate(means):
        inside = m >= 0.08
        ax.text(i, m / 2 if inside else m + 0.04, f"{100 * m:.1f}%", ha="center", va="center",
                color="white" if inside else "black", fontsize=8)
    _percent_axis(ax, title)


def make_figures(exp: Path, cfg: dict, policies, source: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .data import DISTS, held_out_dist, rewarded_policy

    rewarded, held, target = rewarded_policy(cfg, policies).name, held_out_dist(cfg), cfg["rl"]["target"]
    seeds = cfg["seeds"]

    def stage(name):
        found = [_load(exp / f"seed_{s}" / name / "metrics.json") or _load(source / f"seed_{s}" / name / "metrics.json")
                 for s in seeds]
        return [m for m in found if m]

    sft, rl = stage("sft_eval"), stage("rl_eval")
    steer = [m for m in (_load(exp / f"seed_{s}" / "steering" / "metrics.json") for s in seeds) if m]
    base = _load(exp / "base_eval" / "metrics.json") or _load(source / "base_eval" / "metrics.json")
    figdir = exp / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        fig.tight_layout()
        written.append(figdir / name)
        fig.savefig(written[-1])
        plt.close(fig)

    if sft:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        cells = [(d, p.name) for d in DISTS for p in policies]
        bottom = [0.0] * len(cells)
        for key, name, color in OUTCOMES:
            heights = [_stats([m[d]["cells"][p][key] for m in sft])[0] for d, p in cells]
            heights = [0.0 if h != h else h for h in heights]
            ax.bar([f"{d} / {p}" for d, p in cells], heights, bottom=bottom, color=color, label=name, width=0.6)
            for i, h in enumerate(heights):
                if h >= 0.06:
                    ax.text(i, bottom[i] + h / 2, f"{100 * h:.0f}%", ha="center", va="center", color="white", fontsize=8)
            bottom = [b + h for b, h in zip(bottom, heights)]
        _percent_axis(ax, "Post-SFT outcomes by cell")
        ax.set_xlabel("distribution / CoT language")
        ax.legend(fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.25))
        save(fig, "fig2_sft_cells_multiseed.pdf")

    if sft and rl:
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
        width = 0.35
        for j, (name, ms) in enumerate((("post-SFT", sft), ("post-RL", rl))):
            st = [_stats([m[d]["policy_rate"][rewarded] for m in ms]) for d in DISTS]
            left.bar([i + (j - 0.5) * width for i in range(len(DISTS))], [x[0] for x in st], width,
                     yerr=[x[1] for x in st], capsize=3, label=name, color=(NOT_CORRECT, "#cc785c")[j])
        left.set_xticks(range(len(DISTS)), [f"Distribution {d}" for d in DISTS])
        _percent_axis(left, f"{rewarded.capitalize()}-CoT rate (RL rewards {target} only)", "% of completions")
        left.legend(fontsize=8)
        _correct_bars(right, ["post-SFT", "post-RL"], [[m[held]["correct"] for m in ms] for ms in (sft, rl)],
                      f"Outcomes on held-out Distribution {held}")
        if base:
            base_acc = statistics.fmean(d["correct"] for d in base.values())
            right.axhline(base_acc, linestyle=":", color="black", linewidth=1, label=f"base model ({100 * base_acc:.1f}%)")
        right.legend(fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
        save(fig, "fig3_collapse_multiseed.pdf")

    if steer:
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, dist, title in ((left, held, f"Distribution {held}"), (right, target, f"Distribution {target}")):
            _correct_bars(ax, ["post-RL", "+ steering"],
                          [[m["metrics_before"][dist]["correct"] for m in steer], [m["metrics"][dist]["correct"] for m in steer]],
                          title)
        right.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
        save(fig, "fig4_steering_multiseed.pdf")

    for path in written:
        print(f"[figures] wrote {path}")
    if not written:
        print(f"[figures] no evaluation outputs in {exp}; run the evaluation stages first")
    return written
