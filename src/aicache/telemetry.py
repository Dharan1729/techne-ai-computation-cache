"""Decision logging and metrics for the router.

Every routed request produces one decision record. Request text is NOT stored
unless ``log_text`` is enabled: by default a record holds only a short hash of
the text, so logs are safe to ship. Counters are exposed as Prometheus text.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

LATENCY_BUCKETS_MS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500)


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class Telemetry:
    def __init__(self, ring_size: int = 1000, log_file: str | None = None, log_text: bool = False) -> None:
        self.log_text = log_text
        self._lock = threading.Lock()
        self._ring: deque[dict] = deque(maxlen=ring_size)
        self._sink = Path(log_file) if log_file else None
        if self._sink:
            self._sink.parent.mkdir(parents=True, exist_ok=True)
        self.requests: Counter[str] = Counter()          # by source
        self.reasons: Counter[str] = Counter()           # fallback / refusal reasons
        self.tasks: Counter[str] = Counter()             # computed, by task
        self.shadow: Counter[str] = Counter()            # "<task>:agree" / "<task>:disagree"
        self._lat_buckets = [0] * (len(LATENCY_BUCKETS_MS) + 1)
        self._lat_sum = 0.0

    def record(self, *, source: str, task: str | None, similarity: float, reason: str,
               latency_ms: float, text: str, mode: str = "active",
               shadow_value: Any = None) -> str:
        rid = text_id(text) + "-" + format(int(time.time() * 1000) % 0xFFFFFF, "06x")
        rec = {
            "id": rid, "ts": round(time.time(), 3), "source": source, "task": task,
            "similarity": round(similarity, 4), "reason": reason, "mode": mode,
            "latency_ms": round(latency_ms, 3),
        }
        if shadow_value is not None:
            rec["shadow_value"] = shadow_value
        if self.log_text:
            rec["text"] = text
        with self._lock:
            self.requests[source] += 1
            if source != "computation" and reason:
                # keep metric label cardinality bounded: drop the variable part
                bare = reason.split(":")[0] if reason.startswith(("tool_error", "ambiguous_task")) else reason
                self.reasons[bare] += 1
            if source == "computation" and task:
                self.tasks[task] += 1
            for i, edge in enumerate(LATENCY_BUCKETS_MS):
                if latency_ms <= edge:
                    self._lat_buckets[i] += 1
                    break
            else:
                self._lat_buckets[-1] += 1
            self._lat_sum += latency_ms
            self._ring.append(rec)
            if self._sink:
                with self._sink.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        return rid

    def record_shadow(self, task: str, agrees: bool | None) -> None:
        if agrees is None:
            return
        with self._lock:
            self.shadow[f"{task}:{'agree' if agrees else 'disagree'}"] += 1

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._ring)[-limit:][::-1]

    def snapshot(self) -> dict:
        with self._lock:
            total = sum(self.requests.values())
            computed = self.requests.get("computation", 0)
            return {
                "requests_total": total,
                "by_source": dict(self.requests),
                "llm_calls_avoided": computed,
                "computed_share": (computed / total) if total else 0.0,
                "fallback_reasons": dict(self.reasons),
                "computed_by_task": dict(self.tasks),
                "shadow": dict(self.shadow),
                "mean_route_latency_ms": (self._lat_sum / total) if total else 0.0,
            }

    def prometheus(self) -> str:
        with self._lock:
            lines = ["# HELP aicache_requests_total Requests routed, by outcome.",
                     "# TYPE aicache_requests_total counter"]
            for src, n in sorted(self.requests.items()):
                lines.append(f'aicache_requests_total{{source="{src}"}} {n}')
            lines += ["# HELP aicache_fallback_reason_total Why requests were not computed.",
                      "# TYPE aicache_fallback_reason_total counter"]
            for r, n in sorted(self.reasons.items()):
                lines.append(f'aicache_fallback_reason_total{{reason="{r}"}} {n}')
            lines += ["# HELP aicache_computed_total Requests computed by a tool.",
                      "# TYPE aicache_computed_total counter"]
            for t, n in sorted(self.tasks.items()):
                lines.append(f'aicache_computed_total{{task="{t}"}} {n}')
            lines += ["# HELP aicache_shadow_total Shadow-mode comparisons with the LLM answer.",
                      "# TYPE aicache_shadow_total counter"]
            for k, n in sorted(self.shadow.items()):
                task, verdict = k.rsplit(":", 1)
                lines.append(f'aicache_shadow_total{{task="{task}",result="{verdict}"}} {n}')
            lines += ["# HELP aicache_route_latency_ms Routing decision latency.",
                      "# TYPE aicache_route_latency_ms histogram"]
            cum = 0
            for edge, n in zip(LATENCY_BUCKETS_MS, self._lat_buckets):
                cum += n
                lines.append(f'aicache_route_latency_ms_bucket{{le="{edge}"}} {cum}')
            cum += self._lat_buckets[-1]
            lines.append(f'aicache_route_latency_ms_bucket{{le="+Inf"}} {cum}')
            lines.append(f"aicache_route_latency_ms_sum {self._lat_sum:.3f}")
            lines.append(f"aicache_route_latency_ms_count {cum}")
            return "\n".join(lines) + "\n"
