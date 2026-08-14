"""Publication-quality plots for Level 4 (branching workflow) results —
results/level4/summary.json + results.csv, plus the TTL sweep. Same
palette/rcParams convention as benchmarks/level3/plots.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level4"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
GRAY = "#9a9990"
TEXT_MUTED = "#52514e"
GRID = "#e5e4df"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
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

BASELINE_ORDER = ["no_cache", "semantic_cache", "graph_cache_fresh", "graph_cache_ttl_ablation"]
BASELINE_LABEL = {
    "no_cache": "A: No cache",
    "semantic_cache": "C: Semantic cache",
    "graph_cache_fresh": "F: Graph computation\ncache (always fresh)",
    "graph_cache_ttl_ablation": "G: Graph computation\ncache (TTL ablation)",
}
_SHORT_LABEL = {
    "no_cache": "A: No cache",
    "semantic_cache": "C: Semantic",
    "graph_cache_fresh": "F: Graph (fresh)",
    "graph_cache_ttl_ablation": "G: Graph (TTL ablation)",
}
BASELINE_COLOR = {
    "no_cache": GRAY,
    "semantic_cache": ORANGE,
    "graph_cache_fresh": BLUE,
    "graph_cache_ttl_ablation": YELLOW,
}


def _load_summary() -> dict:
    with open(RESULTS_DIR / "summary.json") as f:
        return json.load(f)


def _savefig(fig, name: str, rect=None) -> None:
    out = RESULTS_DIR / name
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_outcome_by_baseline(summary: dict) -> None:
    xs = [b for b in BASELINE_ORDER if b in summary]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    valid = [summary[b]["valid_hit_rate"] * 100 for b in xs]
    invalid = [summary[b]["invalid_hit_rate"] * 100 for b in xs]
    went_to_llm = [(summary[b]["miss_rate"] + summary[b]["fallback_rate"]) * 100 for b in xs]

    ax.bar(xs, valid, color=[BASELINE_COLOR[b] for b in xs], label="Valid hit (decision correct)")
    ax.bar(xs, invalid, bottom=valid, color=RED, alpha=0.85, label="Invalid hit (decision wrong)")
    bottom_llm = [v + i for v, i in zip(valid, invalid)]
    ax.bar(xs, went_to_llm, bottom=bottom_llm, color=TEXT_MUTED, alpha=0.55, label="Went to the LLM (miss/fallback)")

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([BASELINE_LABEL[b] for b in xs], fontsize=8)
    ax.set_ylabel("Share of requests (%)")
    ax.set_title("Level 4 (branching return-decision workflow): request outcome, split by correctness")
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # F/G's fallback slice is real (one genuine near-threshold
    # misidentification, docs/level4.md) but only ~0.6% tall — the same
    # "too thin to read honestly without a callout" issue Level 3's
    # chart had before it was fixed; applying the same fix here rather
    # than letting it recur silently.
    for i, b in enumerate(xs):
        rate = summary[b]["fallback_rate"] * 100
        if 0 < rate < 5:
            n = summary[b]["outcome_counts"].get("fallback", 0)
            ax.annotate(
                f"{rate:.1f}% fallback\n({n} request)",
                xy=(i, 100), xytext=(i, 110), ha="center", fontsize=8, color=TEXT_MUTED,
                arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, lw=0.8),
            )
    fig.text(
        0.01, 0.01,
        "G's 0% invalid-hit rate holds only because this workload never mutates fraud scores mid-run —\n"
        "see the TTL sweep (docs/level4.md) for where G diverges from F once scores actually change.",
        fontsize=7.5, color=TEXT_MUTED, ha="left", va="bottom",
    )
    _savefig(fig, "outcome_by_baseline.png", rect=(0, 0.08, 1, 1))


def plot_tokens_and_latency(summary: dict) -> None:
    xs = [b for b in BASELINE_ORDER if b in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    tokens = [summary[b]["total_tokens"] for b in xs]
    axes[0].bar(xs, tokens, color=[BASELINE_COLOR[b] for b in xs])
    axes[0].set_ylabel("Total tokens (input + output)")
    axes[0].set_title("Tokens consumed")
    axes[0].set_xticks(range(len(xs)))
    axes[0].set_xticklabels([_SHORT_LABEL[b] for b in xs], fontsize=8, rotation=20, ha="right")

    p50 = [summary[b]["p50_latency_ms"] for b in xs]
    axes[1].bar(xs, p50, color=[BASELINE_COLOR[b] for b in xs])
    axes[1].set_ylabel("p50 latency (ms)")
    axes[1].set_title("Request latency")
    axes[1].set_xticks(range(len(xs)))
    axes[1].set_xticklabels([_SHORT_LABEL[b] for b in xs], fontsize=8, rotation=20, ha="right")

    _savefig(fig, "tokens_and_latency.png")


def _load_ttl_sweep() -> list[dict] | None:
    path = RESULTS_DIR / "ttl_sweep.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_ttl_sweep(rows: list[dict] | None) -> None:
    """Graduated freshness: correctness after a repeat request, as a
    function of TTL length, at several fraud-score volatility rates —
    Level 4's analog of Level 3b's mutation-rate sweep, but with TTL as
    a second axis, since (unlike Level 3's binary always-fresh/memoized
    split) a TTL cache's risk depends on both how often the underlying
    value changes AND how long the cache holds onto a stale read."""
    if not rows:
        print("no TTL-sweep data found — run benchmarks/level4/ttl_sweep.py first")
        return

    volatilities = sorted({float(r["volatility"]) for r in rows}, reverse=True)
    ttls = sorted({float(r["ttl_seconds"]) for r in rows})

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, vol in enumerate(volatilities):
        color = SEQ_BLUE[min(i, len(SEQ_BLUE) - 1)] if vol > 0 else GRAY
        ys = [
            float(next(r for r in rows if float(r["volatility"]) == vol and float(r["ttl_seconds"]) == t)["correctness_after"]) * 100
            for t in ttls
        ]
        ax.plot(ttls, ys, marker="o", linewidth=2, color=color, label=f"fraud-score volatility = {vol*100:.0f}%")

    ax.set_xlabel("TTL (seconds)")
    ax.set_ylabel("Correctness after the repeat request (%)")
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Level 4 — graduated freshness: TTL-cached fraud score correctness\n"
        "longer TTLs cost more correctness, faster at higher score volatility"
    )
    ax.legend(frameon=False, fontsize=8)
    _savefig(fig, "ttl_sweep.png")


def main() -> None:
    summary = _load_summary()
    plot_outcome_by_baseline(summary)
    plot_tokens_and_latency(summary)
    plot_ttl_sweep(_load_ttl_sweep())


if __name__ == "__main__":
    main()
