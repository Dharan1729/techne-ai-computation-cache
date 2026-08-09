"""Parameter extraction: pull numeric parameters out of a request's text.

Positional, not semantic — the numbers are taken in left-to-right order
of appearance and zipped against a TaskSpec's declared parameter order.
This is a real, explicit limitation (see docs/architecture.md and
docs/experiment-log.md's failure-analysis entries): a phrasing that states
its numbers out of the canonical order (e.g. "at a rate of Y percent,
invest X dollars for Z years") will silently bind the wrong parameter to
the wrong slot. The workload generator's paraphrases are constrained to
preserve parameter order specifically so this limitation doesn't
contaminate the Level 1/2 results with extraction bugs rather than
genuine caching behavior — flagged here, not hidden.
"""

from __future__ import annotations

import re

# Matches integers, decimals, and comma-grouped thousands (e.g. "1,234.5").
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out
