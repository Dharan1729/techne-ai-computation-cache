"""Publication-quality plots from results/summary.json + results/results.csv.

Colors are the validated categorical/sequential palette from this
environment's dataviz skill (references/palette.md) — fixed hue order,
one sequential ramp for magnitude, no dual axes, no rainbow.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Categorical (fixed order) and sequential palette, from references/palette.md.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
GRAY = "#9a9990"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e5e4df"

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BASELINE_ORDER = ["no_cache", "exact_cache", "semantic_cache", "computation_cache"]
BASELINE_LABEL = {
    "no_cache": "A: No cache",
    "exact_cache": "B: Exact cache",
    "semantic_cache": "C: Semantic cache\n(threshold 0.75)",
    "computation_cache": "D: Computation cache\n(this repo's hypothesis)",
}
BASELINE_COLOR = {
    "no_cache": GRAY,
    "exact_cache": GRAY,
    "semantic_cache": ORANGE,
    "computation_cache": BLUE,
}


def _load():
    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)
    return summary


def _savefig(fig, name: str) -> None:
    out = RESULTS_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_hit_rate_by_baseline(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = BASELINE_ORDER
    valid = [summary[b]["valid_hit_rate"] * 100 for b in xs]
    invalid = [summary[b]["invalid_hit_rate"] * 100 for b in xs]

    ax.bar(xs, valid, color=[BASELINE_COLOR[b] for b in xs], label="Valid hit (answer correct)")
    ax.bar(
        xs,
        invalid,
        bottom=valid,
        color=RED,
        alpha=0.85,
        label="Invalid hit (answer wrong)",
    )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([BASELINE_LABEL[b] for b in xs], fontsize=8.5)
    ax.set_ylabel("Share of requests (%)")
    ax.set_title("Cache hit rate, split by correctness")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _savefig(fig, "hit_rate_by_baseline.png")


def plot_tokens_and_cost(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    xs = BASELINE_ORDER

    tokens = [summary[b]["total_tokens"] for b in xs]
    axes[0].bar(xs, tokens, color=[BASELINE_COLOR[b] for b in xs])
    axes[0].set_ylabel("Total tokens (input + output)")
    axes[0].set_title("Tokens consumed")
    axes[0].set_xticks(range(len(xs)))
    axes[0].set_xticklabels([BASELINE_LABEL[b] for b in xs], fontsize=8, rotation=0)

    cost = [summary[b]["total_cost_usd"] for b in xs]
    axes[1].bar(xs, cost, color=[BASELINE_COLOR[b] for b in xs])
    axes[1].set_ylabel("Simulated inference cost (USD)")
    axes[1].set_title("Cost (simulated backend, Haiku 4.5 pricing)")
    axes[1].set_xticks(range(len(xs)))
    axes[1].set_xticklabels([BASELINE_LABEL[b] for b in xs], fontsize=8, rotation=0)

    _savefig(fig, "tokens_and_cost_by_baseline.png")


def plot_latency(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = BASELINE_ORDER
    p50 = [summary[b]["p50_latency_ms"] for b in xs]
    p95 = [summary[b]["p95_latency_ms"] for b in xs]
    width = 0.35
    idx = range(len(xs))
    ax.bar([i - width / 2 for i in idx], p50, width=width, color=BLUE, label="p50")
    ax.bar([i + width / 2 for i in idx], p95, width=width, color=SEQ_BLUE[1], label="p95")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([BASELINE_LABEL[b] for b in xs], fontsize=8.5)
    ax.set_ylabel("Latency (ms, simulated)")
    ax.set_title("Request latency")
    ax.legend(frameon=False, fontsize=8.5)
    _savefig(fig, "latency_by_baseline.png")


def plot_semantic_threshold_sweep(summary: dict) -> None:
    thresholds = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97]
    labels = [f"semantic_cache@{t:.2f}" for t in thresholds]
    present = [(t, s) for t, s in zip(thresholds, labels) if s in summary]
    if not present:
        print("no threshold-sweep data found (run without --skip-threshold-sweep)")
        return

    xs = [t for t, _ in present]
    hit_rate = [summary[s]["hit_rate"] * 100 for _, s in present]
    invalid_rate = [summary[s]["invalid_hit_rate"] * 100 for _, s in present]
    correctness = [summary[s]["correctness_rate"] * 100 for _, s in present]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, hit_rate, marker="o", color=BLUE, linewidth=2, label="Hit rate")
    ax.plot(
        xs, invalid_rate, marker="o", color=RED, linewidth=2, label="False-hit rate (wrong answer)"
    )
    ax.plot(
        xs,
        correctness,
        marker="o",
        color=AQUA,
        linewidth=2,
        linestyle="--",
        label="Overall correctness",
    )
    ax.set_xlabel("Semantic similarity threshold")
    ax.set_ylabel("Share of requests (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Semantic cache: hit rate vs. correctness across thresholds")
    ax.legend(frameon=False, fontsize=8.5)
    _savefig(fig, "semantic_threshold_sweep.png")


def main() -> None:
    summary = _load()
    plot_hit_rate_by_baseline(summary)
    plot_tokens_and_cost(summary)
    plot_latency(summary)
    plot_semantic_threshold_sweep(summary)


if __name__ == "__main__":
    main()
