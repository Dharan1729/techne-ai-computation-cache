"""Charts for the real-backend (Ollama) validation run — see
docs/experiment-log.md EXP-008. Separate from plots.py because this run's
data lives in results/real_backend_ollama/ (not results/), covers only
the four core baselines (no threshold sweep / external comparison —
see that experiment's scoping rationale), and includes a chart plots.py
has no equivalent of: the underlying real model's own raw accuracy per
task type, which is the actual mechanism behind this run's headline
finding.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "results" / "real_backend_ollama"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GRAY = "#9a9990"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e5e4df"

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
    "semantic_cache": "C: Semantic cache\n(this repo)",
    "computation_cache": "D: Computation cache",
}
BASELINE_COLOR = {
    "no_cache": GRAY,
    "exact_cache": GRAY,
    "semantic_cache": ORANGE,
    "computation_cache": BLUE,
}


def _savefig(fig, name: str) -> None:
    out = DATA_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_correctness_real_backend(summary: dict) -> None:
    xs = [b for b in BASELINE_ORDER if b in summary]
    fig, ax = plt.subplots(figsize=(8, 5.0))
    correctness = [summary[b]["correctness_rate"] * 100 for b in xs]
    bars = ax.bar(xs, correctness, color=[BASELINE_COLOR[b] for b in xs])
    for bar, val in zip(bars, correctness):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=9,
            color=TEXT,
        )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([BASELINE_LABEL[b] for b in xs], fontsize=9)
    ax.set_ylabel("Overall correctness (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Real local model (Ollama llama3:8b), same LLM for every condition\n"
        "Overall correctness — EXP-008, 150 requests"
    )
    _savefig(fig, "correctness_real_backend.png")


def plot_llm_accuracy_by_task(results_csv: Path) -> None:
    with open(results_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["baseline"] == "no_cache"]

    by_task: dict[str, list[bool]] = {}
    for r in rows:
        by_task.setdefault(r["task_type"], []).append(r["correct"] == "True")

    tasks = sorted(by_task.keys(), key=lambda t: sum(by_task[t]) / len(by_task[t]))
    accuracy = [sum(by_task[t]) / len(by_task[t]) * 100 for t in tasks]
    colors = [RED if a < 50 else (ORANGE if a < 90 else AQUA) for a in accuracy]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.barh(tasks, accuracy, color=colors)
    for bar, val in zip(bars, accuracy):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("Correct answers, calling the LLM directly for every request (%)")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "A real local LLM's own raw arithmetic accuracy, by task type\n"
        "(no cache, no tool use, no chain-of-thought prompting — llama3:8b via Ollama)"
    )
    _savefig(fig, "llm_accuracy_by_task_type.png")


def main() -> None:
    with open(DATA_DIR / "summary.json") as f:
        summary = json.load(f)
    plot_correctness_real_backend(summary)
    plot_llm_accuracy_by_task(DATA_DIR / "results.csv")


if __name__ == "__main__":
    main()
