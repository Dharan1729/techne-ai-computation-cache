"""Level 4's centerpiece experiment: graduated freshness. Level 3's
freshness model was binary (always-fresh vs. memoized-forever, swept
only by mutation *rate* in `benchmarks/level3/volatility_sweep.py`).
Level 4's fraud-score sub-fetch adds a second, independent axis: TTL
length. This sweep varies both — TTL (how long the cache trusts a
stored fraud score) and volatility (how often a customer's actual
fraud score changes between two queries about them) — and measures
baseline G's (the TTL ablation) correctness on the second of two
requests, exactly as `volatility_sweep.py` did for Level 3, extended to
two dimensions.

**Design, stated before running**: each probe issues a request, then —
with probability `volatility` — the customer's fraud score is mutated
from low-risk to high-risk (a clean, reliable ground-truth flip, same
discipline as Level 3b's status-mutation probes), then waits
`CHECK_INTERVAL_SECONDS` (simulating "how soon the same customer is
asked about again") before issuing the identical request a second time.
Predicted shape: for `ttl_seconds < CHECK_INTERVAL_SECONDS`, the cache
entry has already expired by the second request, so G re-fetches and
should be correct regardless of volatility (matching F). For
`ttl_seconds >= CHECK_INTERVAL_SECONDS`, the entry is still live, so
G's correctness should degrade with volatility exactly the way Level
3b's stale ablation did: `1 - volatility`.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from aicache.cache.ttl_cache import TTLCache
from aicache.cache.workflow_computation import WorkflowComputationCache
from aicache.computation.graph_tasks import GRAPH_TASK_REGISTRY, build_graph_registry
from aicache.enterprise.db import EnterpriseDB, generate_enterprise_db
from aicache.llm.graph_rendering import render_graph_answer
from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.graph_correctness import is_graph_correct

from benchmarks.level4.run_experiment import LEVEL4_IDENTIFICATION_THRESHOLD

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level4"

CHECK_INTERVAL_SECONDS = 600.0  # 10 minutes between the two requests about the same customer
TTL_SWEEP = [0.0, 60.0, 300.0, 500.0, 600.0, 700.0, 900.0, 1800.0]
VOLATILITY_SWEEP = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
N_PROBES = 30

LOW_RISK_SCORE = 0.1
HIGH_RISK_SCORE = 0.9


def _build_probe_db(n_probes: int) -> tuple[EnterpriseDB, list[tuple[str, str]]]:
    """N guaranteed fraud-check-eligible probes (gold tier, >$500,
    within window, low-risk starting score) — explicitly constructed,
    same discipline as `workloads/level4/generator.py` and Level 3b's
    freshness probes, not left to random generation."""
    db = generate_enterprise_db(seed=0, n_customers=1, n_transactions=1)
    db.customers.clear()
    db.transactions.clear()
    db.fraud_scores.clear()
    pairs: list[tuple[str, str]] = []
    for i in range(n_probes):
        cust_id = f"CUST-8{i:03d}"
        txn_id = f"TXN-8{i:05d}"
        db.customers[cust_id] = {
            "customer_id": cust_id, "tier": "gold", "region": "us-east",
            "account_status": "active", "signup_day": 1,
        }
        db.fraud_scores[cust_id] = {"customer_id": cust_id, "score": LOW_RISK_SCORE, "computed_day": 995}
        db.transactions[txn_id] = {
            "transaction_id": txn_id, "customer_id": cust_id, "amount": 900.0,
            "day": 990, "status": "completed", "category": "other",
        }
        pairs.append((cust_id, txn_id))
    return db, pairs


def _ground_truth(entities: dict, db: EnterpriseDB) -> dict:
    spec = GRAPH_TASK_REGISTRY["return_decision"]
    outcome = spec.fetch(entities, {}, db)
    assert outcome.values is not None
    return spec.executor(**entities, **outcome.values)


@dataclass
class TTLPointResult:
    ttl_seconds: float
    volatility: float
    n_probes: int
    n_correct_after: int

    @property
    def correctness_after(self) -> float:
        return self.n_correct_after / self.n_probes


def _run_point(
    ttl_seconds: float, volatility: float, db_template: EnterpriseDB,
    pairs: list[tuple[str, str]], embedder: Embedder, rng: random.Random,
) -> TTLPointResult:
    n_correct = 0
    for cust_id, txn_id in pairs:
        db = db_template.clone()
        entities = {"customer_id": cust_id, "transaction_id": txn_id}
        text = f"Should we approve the return for customer {cust_id} on transaction {txn_id}?"

        clock = [0.0]
        ttl_cache = TTLCache(ttl_seconds=ttl_seconds, now_fn=lambda: clock[0])
        identifier = TaskIdentifier(
            embedder=embedder, threshold=LEVEL4_IDENTIFICATION_THRESHOLD, registry=build_graph_registry(ttl_cache)
        )
        cache = WorkflowComputationCache(db, identifier, use_stale_cache=False, render_fn=render_graph_answer)

        cache.lookup(text)  # request 1: primes the fraud-score cache entry at clock=0

        if rng.random() < volatility:
            db.mutate_fraud_score(cust_id, HIGH_RISK_SCORE)

        clock[0] = CHECK_INTERVAL_SECONDS
        after = cache.lookup(text)  # request 2
        after_gt = _ground_truth(entities, db)

        if is_graph_correct("return_decision", after.served_text or "", after_gt):
            n_correct += 1

    return TTLPointResult(ttl_seconds, volatility, len(pairs), n_correct)


def main() -> None:
    db_template, pairs = _build_probe_db(N_PROBES)
    embedder = Embedder()
    rng = random.Random(2026)

    rows_out: list[dict] = []
    for ttl in TTL_SWEEP:
        for vol in VOLATILITY_SWEEP:
            point = _run_point(ttl, vol, db_template, pairs, embedder, rng)
            print(f"[ttl_sweep] ttl={ttl:>6.0f}s  volatility={vol*100:>5.1f}%  correctness={point.correctness_after*100:>5.1f}%")
            rows_out.append(
                {
                    "ttl_seconds": ttl,
                    "volatility": vol,
                    "n_probes": point.n_probes,
                    "n_correct_after": point.n_correct_after,
                    "correctness_after": point.correctness_after,
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "ttl_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    json_path = RESULTS_DIR / "ttl_sweep.json"
    with open(json_path, "w") as f:
        json.dump(rows_out, f, indent=2, default=str)

    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
