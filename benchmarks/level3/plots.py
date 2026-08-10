"""Publication-quality plots for the Level 3 (enterprise workflow) results —
results/level3/summary.json + results/level3/results.csv. Same palette and
rcParams as benchmarks/plots.py (see references/palette.md); Level 3 adds
one new chart type Level 1/2 didn't need: the per-request latency-component
breakdown (identify/extract/validate/fetch/execute), since the brief asked
for latency instrumentation deep enough to show where time actually goes,
not just an aggregate p50.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level3"

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

BASELINE_ORDER = [
    "no_cache",
    "exact_cache",
    "semantic_cache",
    "computation_cache_fresh",
    "computation_cache_stale_ablation",
]
BASELINE_LABEL = {
    "no_cache": "A: No cache",
    "exact_cache": "B: Exact cache",
    "semantic_cache": "C: Semantic cache",
    "computation_cache_fresh": "D: Computation cache\n(always fresh)",
    "computation_cache_stale_ablation": "E: Computation cache\n(stale-fetch ablation)",
}
_SHORT_LABEL = {
    "no_cache": "A: No cache",
    "exact_cache": "B: Exact cache",
    "semantic_cache": "C: Semantic",
    "computation_cache_fresh": "D: Comp. cache (fresh)",
    "computation_cache_stale_ablation": "E: Comp. cache (stale, ablation)",
}
BASELINE_COLOR = {
    "no_cache": GRAY,
    "exact_cache": GRAY,
    "semantic_cache": ORANGE,
    "computation_cache_fresh": BLUE,
    "computation_cache_stale_ablation": YELLOW,
}

LATENCY_COMPONENTS = ["identify_ms", "extract_ms", "validate_ms", "fetch_ms", "execute_ms"]
LATENCY_COMPONENT_LABEL = {
    "identify_ms": "Identify (embed + match task)",
    "extract_ms": "Extract (parse entities/numbers)",
    "validate_ms": "Validate (gate before execution)",
    "fetch_ms": "Fetch (DB read)",
    "execute_ms": "Execute (pure computation)",
}
LATENCY_COMPONENT_COLOR = {
    "identify_ms": SEQ_BLUE[3],
    "extract_ms": SEQ_BLUE[2],
    "validate_ms": SEQ_BLUE[1],
    "fetch_ms": ORANGE,
    "execute_ms": AQUA,
}


def _load_summary() -> dict:
    with open(RESULTS_DIR / "summary.json") as f:
        return json.load(f)


def _load_rows() -> list[dict]:
    with open(RESULTS_DIR / "results.csv", newline="") as f:
        return list(csv.DictReader(f))


def _load_freshness_summary() -> dict | None:
    path = RESULTS_DIR / "freshness_experiment.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("summary")


def _savefig(fig, name: str, rect: tuple | None = None) -> None:
    out = RESULTS_DIR / name
    fig.tight_layout(rect=rect)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


FALLBACK_COLOR = TEXT_MUTED


def plot_hit_rate_by_baseline(summary: dict) -> None:
    """Stacks all three outcomes that can happen to a request — valid
    hit, invalid hit, and "went to the LLM" (a computation-cache
    fallback for D/E, a cache miss for A/B/C) — so every bar sums to
    100% and none of a baseline's traffic is silently missing from the
    chart. Earlier versions of this chart only stacked valid+invalid,
    which made D's real 0.9% fallback rate (two genuine
    misidentifications the validation gate caught, not a flaw) invisible
    and let the bar read as a flat, misleading 100%."""
    xs = [b for b in BASELINE_ORDER if b in summary]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    valid = [summary[b]["valid_hit_rate"] * 100 for b in xs]
    invalid = [summary[b]["invalid_hit_rate"] * 100 for b in xs]
    went_to_llm = [(summary[b]["miss_rate"] + summary[b]["fallback_rate"]) * 100 for b in xs]

    ax.bar(xs, valid, color=[BASELINE_COLOR[b] for b in xs], label="Valid hit (answer correct)")
    ax.bar(xs, invalid, bottom=valid, color=RED, alpha=0.85, label="Invalid hit (answer wrong)")
    bottom_llm = [v + i for v, i in zip(valid, invalid)]
    ax.bar(
        xs,
        went_to_llm,
        bottom=bottom_llm,
        color=FALLBACK_COLOR,
        alpha=0.55,
        label="Went to the LLM (miss / fallback)",
    )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([BASELINE_LABEL[b] for b in xs], fontsize=8)
    ax.set_ylabel("Share of requests (%)")
    ax.set_title("Level 3 (enterprise workflow tasks): request outcome, split by correctness")
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # D/E's fallback slice is real (two genuine misidentifications the
    # validation gate caught) but only ~0.9% tall — too thin to read at
    # its true proportion, so it gets a direct callout rather than being
    # left to disappear visually the way the pre-fix chart let it.
    for i, b in enumerate(xs):
        rate = summary[b]["fallback_rate"] * 100
        if 0 < rate < 5:
            n = summary[b]["outcome_counts"].get("fallback", 0)
            ax.annotate(
                f"{rate:.1f}% fallback\n({n} requests)",
                xy=(i, 100),
                xytext=(i, 110),
                ha="center",
                fontsize=8,
                color=TEXT_MUTED,
                arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, lw=0.8),
            )
    fig.text(
        0.01,
        0.01,
        "E's 0% invalid-hit rate holds only because this workload never mutates the database mid-run —\n"
        "see the freshness experiment (docs/level3.md) for where E diverges from D once the data changes.",
        fontsize=7.5,
        color=TEXT_MUTED,
        ha="left",
        va="bottom",
    )
    _savefig(fig, "hit_rate_by_baseline.png", rect=(0, 0.08, 1, 1))


def plot_tokens_and_cost(summary: dict) -> None:
    xs = [b for b in BASELINE_ORDER if b in summary]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    tokens = [summary[b]["total_tokens"] for b in xs]
    axes[0].bar(xs, tokens, color=[BASELINE_COLOR[b] for b in xs])
    axes[0].set_ylabel("Total tokens (input + output)")
    axes[0].set_title("Tokens consumed")
    axes[0].set_xticks(range(len(xs)))
    axes[0].set_xticklabels([_SHORT_LABEL[b] for b in xs], fontsize=8, rotation=25, ha="right")

    cost = [summary[b]["total_cost_usd"] for b in xs]
    axes[1].bar(xs, cost, color=[BASELINE_COLOR[b] for b in xs])
    axes[1].set_ylabel("Simulated inference cost (USD)")
    axes[1].set_title("Cost (simulated backend)")
    axes[1].set_xticks(range(len(xs)))
    axes[1].set_xticklabels([_SHORT_LABEL[b] for b in xs], fontsize=8, rotation=25, ha="right")

    _savefig(fig, "tokens_and_cost_by_baseline.png")


def plot_latency(summary: dict) -> None:
    xs = [b for b in BASELINE_ORDER if b in summary]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    p50 = [summary[b]["p50_latency_ms"] for b in xs]
    p95 = [summary[b]["p95_latency_ms"] for b in xs]
    width = 0.35
    idx = range(len(xs))
    ax.bar([i - width / 2 for i in idx], p50, width=width, color=BLUE, label="p50")
    ax.bar([i + width / 2 for i in idx], p95, width=width, color=SEQ_BLUE[1], label="p95")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([_SHORT_LABEL[b] for b in xs], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms; simulated LLM + real fetch/embedding lookups)")
    ax.set_title("Level 3: request latency")
    ax.legend(frameon=False, fontsize=8.5)
    _savefig(fig, "latency_by_baseline.png")


def plot_latency_breakdown(rows: list[dict]) -> None:
    """D vs E, decomposed into the five named pipeline stages the brief
    asked for — not just an aggregate number. Only hit rows carry a
    component breakdown (a fallback row's cost is dominated by the LLM
    call, tracked separately as fetch_ms/llm_ms, not this pipeline)."""
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        baseline = row["baseline"]
        if baseline not in ("computation_cache_fresh", "computation_cache_stale_ablation"):
            continue
        extra = json.loads(row["extra"]) if row["extra"] else {}
        if "identify_ms" not in extra:
            continue
        for comp in LATENCY_COMPONENTS:
            sums[baseline][comp] += extra[comp]
        counts[baseline] += 1

    xs = [b for b in ("computation_cache_fresh", "computation_cache_stale_ablation") if counts[b]]
    if not xs:
        print("no latency-breakdown data found in results.csv")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bottoms = [0.0] * len(xs)
    for comp in LATENCY_COMPONENTS:
        vals = [sums[b][comp] / counts[b] for b in xs]
        ax.bar(
            [_SHORT_LABEL[b] for b in xs],
            vals,
            bottom=bottoms,
            color=LATENCY_COMPONENT_COLOR[comp],
            label=LATENCY_COMPONENT_LABEL[comp],
        )
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    ax.set_ylabel("Mean latency per hit (ms)")
    ax.set_title(
        "Where computation-cache time goes, on a hit\n"
        "fresh (D) re-fetches every time; the stale ablation (E) skips fetch_ms via memoization"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _savefig(fig, "latency_breakdown_d_vs_e.png")


def plot_correctness_latency_tradeoff(summary: dict, freshness_summary: dict | None) -> None:
    """The three-way tradeoff the primary comparison alone can't show
    honestly for the stale ablation: E's correctness bar here comes from
    the freshness experiment's post-mutation results (0/3), NOT from
    run_experiment.py's aggregate (which reads 100% only because that
    workload has zero mutation exposure — see the caveat on
    hit_rate_by_baseline.png). Semantic and fresh both use their
    aggregate run_experiment.py correctness, which IS a fair general
    number for them (semantic's failure mode is embedding proximity, not
    staleness; fresh is correct independent of mutation exposure, and
    the freshness experiment's own 3/3 confirms that directly)."""
    if not freshness_summary or "computation_cache_stale_ablation" not in freshness_summary:
        print("no freshness-experiment summary found — run benchmarks/level3/freshness_experiment.py first")
        return

    stale_rate = freshness_summary["computation_cache_stale_ablation"]["post_mutation_correctness"]
    points = [
        ("semantic_cache", "C: Semantic", summary["semantic_cache"]["correctness_rate"] * 100, summary["semantic_cache"]["p50_latency_ms"]),
        ("computation_cache_fresh", "D: Computation (fresh)", summary["computation_cache_fresh"]["correctness_rate"] * 100, summary["computation_cache_fresh"]["p50_latency_ms"]),
        ("computation_cache_stale_ablation", "E: Computation (stale,\nafter mutation)", stale_rate * 100, summary["computation_cache_stale_ablation"]["p50_latency_ms"]),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for key, label, correctness, latency in points:
        ax.scatter([latency], [correctness], s=220, color=BASELINE_COLOR[key], zorder=3)
        ax.annotate(
            label,
            (latency, correctness),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("Correctness (%)")
    ax.set_ylim(-5, 108)
    ax.set_xlim(0, max(p[3] for p in points) * 1.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Correctness vs. latency: the real three-way tradeoff\n"
        "E's correctness is measured after a mutation, not the static-workload figure",
        fontsize=11,
    )
    _savefig(fig, "correctness_latency_tradeoff.png")


def main() -> None:
    summary = _load_summary()
    rows = _load_rows()
    freshness_summary = _load_freshness_summary()
    plot_hit_rate_by_baseline(summary)
    plot_tokens_and_cost(summary)
    plot_latency(summary)
    plot_latency_breakdown(rows)
    plot_correctness_latency_tradeoff(summary, freshness_summary)


if __name__ == "__main__":
    main()
