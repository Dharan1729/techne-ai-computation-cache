"""Build examples/finance_labeled.jsonl for ``aicache eval`` from the finance
benchmark's phrasing templates and negative requests.

Expected values come from the same closed-form formulas the example tools use,
so this is a demonstration dataset for the tooling, not an independent
benchmark. For independent evidence see EXP-015 in docs/experiment-log.md.

    python -m benchmarks.real_app.make_labeled_dataset
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.real_app.financial_chat_integration import NEG_CAL, NEG_TEST, build_cases


def main(out: str = "examples/finance_labeled.jsonl", per_template: int = 4, seed: int = 11) -> None:
    calib, test = build_cases(seed, per_template)
    rows = []
    for c in calib + test:
        if c.kind == "negative":
            rows.append({"text": c.text, "expect": "refuse"})
        elif c.kind == "position_size":
            a = c.args
            risk = a["account_size"] * a["risk_percent"] / 100
            rows.append({"text": c.text, "expect": round(risk / abs(a["entry_price"] - a["stop_price"]), 2)})
        else:
            a = c.args
            rows.append({"text": c.text,
                         "expect": round(a["entry_price"] + abs(a["entry_price"] - a["stop_price"]) * a["risk_multiple"], 2)})
    # the negatives appear in both calib and test lists of build_cases: keep each once
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            uniq.append(r)
    Path(out).write_text("\n".join(json.dumps(r) for r in uniq) + "\n")
    print(f"wrote {out}: {len(uniq)} requests "
          f"({sum(r['expect'] != 'refuse' for r in uniq)} compute, {sum(r['expect'] == 'refuse' for r in uniq)} refuse)")


if __name__ == "__main__":
    main()
