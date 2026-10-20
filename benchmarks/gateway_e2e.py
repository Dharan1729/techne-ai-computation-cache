"""End-to-end check of the gateway in front of a real local LLM (EXP-018).

Starts the gateway (real embedding model, examples/finance_tools.yaml) in front
of Ollama's OpenAI-compatible endpoint, sends every request in
examples/finance_labeled.jsonl through it, and sends the same requests straight
to Ollama as the baseline. Needs `ollama serve` and `ollama pull llama3`.

    python -m benchmarks.gateway_e2e --model llama3 --output results/gateway_e2e.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from aicache.binding.extractor import extract_numbers
from aicache.config import load_config
from aicache.server import create_app
from aicache.tooling import load_labeled

OLLAMA = "http://localhost:11434/v1"


def last_number(text: str) -> float | None:
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def close(got: float | None, want: float) -> bool:
    return got is not None and abs(got - want) <= max(1e-2, 1e-6 * abs(want))


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def summarize(rows: list[dict], model: str) -> dict:
    pos = [r for r in rows if r["expect"] != "refuse"]
    neg = [r for r in rows if r["expect"] == "refuse"]
    comp = [r for r in rows if r["source"] == "computation"]
    fwd = [r for r in rows if r["source"] != "computation"]
    return {
        "model": model, "requests": len(rows), "positives": len(pos), "negatives": len(neg),
        "computed_by_gateway": len(comp),
        "computed_correctly": sum(r["gw_ok"] for r in comp if r["gw_ok"] is not None),
        "computed_on_negatives": sum(r["source"] == "computation" for r in neg),
        "positives_forwarded_to_llm": sum(r["source"] != "computation" for r in pos),
        "direct_llm_positives_correct": sum(r["direct_ok"] for r in pos),
        "direct_llm_note": "NOT a fair baseline: no tool use and replies truncated at max_tokens=200, so the "
                           "last number is often not the final answer. See EXP-015 for the fair baseline.",
        "latency_ms": {
            "gateway_computed_p50": statistics.median(r["gw_ms"] for r in comp) if comp else None,
            "gateway_forwarded_p50": statistics.median(r["gw_ms"] for r in fwd) if fwd else None,
            "direct_p50": statistics.median(r["direct_ms"] for r in rows),
            "gateway_all_mean": statistics.fmean(r["gw_ms"] for r in rows),
            "direct_all_mean": statistics.fmean(r["direct_ms"] for r in rows),
            "forwarded_overhead_median_paired": statistics.median(r["gw_ms"] - r["direct_ms"] for r in fwd) if fwd else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--config", default="examples/finance_tools.yaml")
    ap.add_argument("--labeled", default="examples/finance_labeled.jsonl")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--output", default="results/gateway_e2e.json")
    ap.add_argument("--from-json", help="recompute the summary from a saved run instead of running")
    a = ap.parse_args()

    if a.from_json:
        saved = json.loads(Path(a.from_json).read_text())
        saved["summary"] = summarize(saved["rows"], saved["summary"]["model"])
        Path(a.output).write_text(json.dumps(saved, indent=2))
        print(json.dumps(saved["summary"], indent=2))
        return

    cfg = load_config(Path(a.config))
    cfg = cfg.model_copy(update={"upstream": {"base_url": OLLAMA, "model": a.model, "timeout_s": 120.0}})
    cfg = load_config(json.loads(cfg.model_dump_json()))  # re-validate the patched config
    server = uvicorn.Server(uvicorn.Config(create_app(cfg), host="127.0.0.1", port=a.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)

    items = load_labeled(a.labeled)
    gw = httpx.Client(base_url=f"http://127.0.0.1:{a.port}", timeout=180)
    direct = httpx.Client(base_url=OLLAMA, timeout=180)
    opts = {"temperature": 0, "seed": 0, "max_tokens": 200}  # same cap on both paths, so long explanations do not dominate

    def body(text: str) -> dict:
        return {"model": a.model, "messages": [{"role": "user", "content": text}], **opts}

    # warm the model and the embedder so one-off loads are not billed to anything
    direct.post("/chat/completions", json=body("What is 1 plus 1?"))
    gw.post("/v1/chat/completions", json=body("What is 15% of 240?"))

    rows = []
    for n, it in enumerate(items, 1):
        if n % 10 == 0:
            print(f"{n}/{len(items)}", flush=True)
        t0 = time.perf_counter()
        g = gw.post("/v1/chat/completions", json=body(it.text))
        g_ms = (time.perf_counter() - t0) * 1000
        g_text = g.json()["choices"][0]["message"]["content"]
        t0 = time.perf_counter()
        d = direct.post("/chat/completions", json=body(it.text))
        d_ms = (time.perf_counter() - t0) * 1000
        d_text = d.json()["choices"][0]["message"]["content"]
        rows.append({
            "text": it.text, "expect": it.expect, "source": g.headers["x-aicache-source"],
            "gw_ms": g_ms, "direct_ms": d_ms,
            "gw_ok": None if it.expect == "refuse" else close(last_number(g_text), it.expect),
            "direct_ok": None if it.expect == "refuse" else close(last_number(d_text), it.expect),
        })

    summary = summarize(rows, a.model)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    server.should_exit = True


if __name__ == "__main__":
    main()
