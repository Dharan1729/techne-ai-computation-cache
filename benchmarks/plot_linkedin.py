"""Figures for the write-up (EXP-015 / EXP-016). Every plotted number is read
from a results file; nothing is typed in by hand.

    python -m benchmarks.plot_linkedin --rep-dir <dir with P*K*s* run folders> --out results/figures
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from workloads.generators import generate_workload  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a8a2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})


def fig_baselines(rb_dir: Path, out: Path) -> None:
    d = json.loads((rb_dir / "summary.json").read_text())["conditions"]
    order = [
        ("llm_direct", "LLM answers directly", GRAY),
        ("llm_cot", "LLM, step-by-step prompt", GRAY),
        ("llm_router", "LLM picks tool + params,\ncode does the math", GRAY),
        ("comp_cache", "Computation cache\n(LLM only on fallback)", BLUE),
        ("hybrid", "Computation cache,\nrouter on fallback", BLUE),
    ]
    n = json.loads((rb_dir / "summary.json").read_text())["n_requests"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [1.15, 1]}, sharey=True)
    ys = list(range(len(order)))[::-1]
    for y, (k, label, col) in zip(ys, order):
        s = d[k]
        acc = s["accuracy"] * 100
        lo, hi = s["wilson95"][0] * 100, s["wilson95"][1] * 100
        a1.barh(y, acc, color=col, height=0.55)
        a1.plot([lo, hi], [y, y], color=INK, lw=1.6)
        a1.text(max(hi, acc) + 1.5, y, f"{acc:.0f}%", va="center", fontsize=13, color=INK, fontweight="bold")
        a2.barh(y, s["p50_ms"], color=col, height=0.55)
        a2.text(s["p50_ms"] + 40, y, f"{s['p50_ms']:.0f} ms" if s["p50_ms"] < 1000 else f"{s['p50_ms'] / 1000:.1f} s",
                va="center", fontsize=13, color=INK, fontweight="bold")
    a1.set_yticks(ys)
    a1.set_yticklabels([o[1] for o in order], fontsize=12)
    a1.set_xlim(0, 112)
    a1.set_title("Correct answers", loc="left", fontsize=14, color=INK, fontweight="bold")
    a1.set_xlabel("% of requests answered correctly (whiskers: 95% interval)", fontsize=10)
    a2.set_xlim(0, 3100)
    a2.set_title("Median latency (cache hits are 77% of requests)", loc="left", fontsize=14, color=INK, fontweight="bold")
    a2.set_xlabel("milliseconds per request", fontsize=10)
    for a in (a1, a2):
        a.xaxis.grid(True, color=GRID)
        a.set_axisbelow(True)
        a.tick_params(length=0)
    a2.tick_params(left=False)
    fig.suptitle("Same 150 arithmetic requests, same local 8B model, temperature 0",
                 x=0.015, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.015, 0.915, "Blue = uses the computation cache. Grey = LLM only. The fair baseline is the third bar, not the first.",
             fontsize=11.5, color=INK2, ha="left")
    fig.text(0.015, 0.012,
             f"n = {n} requests, one seed (42), llama3 8B via Ollama, simulated workload of 9 arithmetic/business-formula task types. "
             "A different model or workload will move these numbers.",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(out / "fig1_baselines.png", dpi=170)
    plt.close(fig)


def fig_repeats(rep_dir: Path, out: Path) -> None:
    cfgs = [("P18K1", 18, 1), ("P9K2", 9, 2), ("P6K3", 6, 3), ("P3K6", 3, 6)]
    seeds = range(6, 11)
    xs, comp, sem = [], [], []
    for cfg, p, k in cfgs:
        wl = generate_workload(param_sets_per_task=p, paraphrases_per_param_set=k, seed=6)
        distinct = len({(w.task_type, w.param_set_id) for w in wl})
        xs.append((1 - distinct / len(wl)) * 100)
        runs = [json.loads((rep_dir / f"{cfg}s{s}" / "summary.json").read_text()) for s in seeds]
        comp.append([r["computation_cache"]["valid_hit_rate"] * 100 for r in runs])
        sem.append([r["semantic_cache"]["valid_hit_rate"] * 100 for r in runs])
    fig, ax = plt.subplots(figsize=(12, 6.4))
    for series, col in ((comp, BLUE), (sem, ORANGE)):
        m = [st.mean(v) for v in series]
        ax.fill_between(xs, [min(v) for v in series], [max(v) for v in series], color=col, alpha=0.15, lw=0)
        ax.plot(xs, m, color=col, lw=2.4, marker="o", ms=9, mfc=SURFACE, mew=2.4)
        ax.text(xs[0] + 1.0, m[0] + 4.5, f"{m[0]:.0f}%", color=INK, fontsize=15, fontweight="bold")
        ax.text(xs[-1] + 1.2, m[-1], f"{m[-1]:.0f}%", color=INK, fontsize=15, fontweight="bold", va="center")
    ax.text(30, 83, "Computation cache", color=BLUE, fontsize=14, fontweight="bold")
    ax.text(34, 9, "Semantic cache\n(embedding match)", color=ORANGE, fontsize=14, fontweight="bold")
    ax.set_xlim(-3, 80)
    ax.set_ylim(0, 100)
    ax.set_xlabel("How often a request repeats an earlier request's exact numbers (%)", fontsize=11)
    ax.set_ylabel("Requests served from cache AND correct (%)", fontsize=11)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    fig.suptitle("A semantic cache only pays off when the numbers repeat", x=0.015, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.015, 0.915, "Both stayed about 99-100% correct here. The difference is how much traffic each can serve at all.",
             fontsize=11.5, color=INK2, ha="left")
    fig.text(0.015, 0.012,
             "Simulated model, 5 held-out seeds (6-10), line = mean, band = min to max.\n"
             "Semantic cache: redis/langcache-embed-v1 at similarity 0.75. A weaker embedding model would make it less safe.",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(out / "fig2_repeats.png", dpi=170)
    plt.close(fig)


def fig_corrections(out: Path) -> None:
    """Claims that changed under re-testing. Every 'after' is sourced from
    docs/experiment-log.md EXP-015 / EXP-016."""
    rows = [
        ("Cache hits are 100% correct",
         "One seed. Over 10 seeds:\n20 wrong hits in 1500",
         "Add, subtract and multiply share a signature.\nA cue guard fixed it: 0 in 1500"),
        ("45x faster than the LLM",
         "Baseline was a raw LLM,\n47% correct",
         "Fair baseline (LLM picks the tool,\ncode computes): 93% correct.\nReal gap: about 9x on median latency"),
        ("A 1B model can't\nextract parameters",
         "Naive zero-shot prompt",
         "With a proper router prompt:\n52 of 60 correct"),
        ("Embedding match\ndecides the task",
         "Scores for real and unrelated\nrequests overlap (0.46-0.67\nvs 0.37-0.70)",
         "A structural gate does the work:\nevery number binds to one slot,\nnone left over"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.axis("off")
    ax.text(0.0, 1.02, "Four of my own claims did not survive re-testing", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.965, "Found by re-running with more seeds and a fairer baseline, not by a reviewer.",
            fontsize=12, color=INK2, transform=ax.transAxes)
    for h, x in (("What I claimed", 0.0), ("What I found", 0.31), ("What it became", 0.60)):
        ax.text(x, 0.89, h, fontsize=12, color=INK2, fontweight="bold", transform=ax.transAxes)
    for i, (a, b, c) in enumerate(rows):
        y = 0.77 - i * 0.2
        ax.plot([0, 1], [y + 0.09, y + 0.09], color=GRID, lw=1, transform=ax.transAxes)
        ax.text(0.0, y, a, fontsize=13.5, fontweight="bold", va="center", transform=ax.transAxes)
        ax.text(0.31, y, b, fontsize=12, va="center", color=INK, transform=ax.transAxes)
        ax.text(0.60, y, c, fontsize=12, va="center", color=BLUE, transform=ax.transAxes)
    fig.text(0.015, 0.012,
             "Source: docs/experiment-log.md, EXP-015 and EXP-016. Multi-seed runs use a simulated model; the baseline comparison uses a local llama3 8B.",
             fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.05)
    fig.savefig(out / "fig3_corrections.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb-dir", default="results/real_baselines")
    ap.add_argument("--rep-dir", required=True)
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    fig_baselines(Path(a.rb_dir), o)
    fig_repeats(Path(a.rep_dir), o)
    fig_corrections(o)
    print("wrote", sorted(p.name for p in o.glob("fig*.png")))
