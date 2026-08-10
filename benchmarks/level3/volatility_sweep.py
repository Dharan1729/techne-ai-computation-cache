"""Level 3b, sweep 2: state volatility.

The freshness experiment (EXP-010) tested exactly two points on this
curve: 0% mutation exposure (the primary comparison, EXP-009) and 100%
mutation exposure (every one of its 3 scenarios mutates the queried
record). Both are real, both are honestly reported (see
`docs/level3.md`'s "combined, honest three-way view") — but reality lives
between those two endpoints, and neither number alone says how a
baseline's correctness responds as mutation frequency increases from
"never" to "always." This sweep measures that response curve directly.

**Design**: holds mutation TYPE fixed (transaction status completed ->
refunded, the cleanest, most reliable ground-truth flip of the three
EXP-010 already validated qualitatively) and varies mutation RATE — the
one-variable-at-a-time discipline EXP-010 didn't need but this follow-up
does, since its whole point is isolating rate as the dependent variable.
At each rate p, N_PROBES independent, isolated (fresh cache instances
per probe, matching EXP-010's methodology) refund-eligibility scenarios
are each mutated with probability p; aggregate post-request-2 correctness
is reported per baseline.

Expected shape, stated before running (so a null/different result is
visible as such, not rationalized after the fact): `semantic_cache` and
the stale-fetch ablation (E) have no mechanism to detect a mutation on a
byte-identical repeat request — both should trace approximately
``correctness(p) = 1 - p``. D re-fetches unconditionally, so its
correctness should stay flat at 1.0 across the whole sweep.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from aicache.cache.semantic import SemanticCache
from aicache.cache.workflow_computation import WorkflowComputationCache
from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from aicache.enterprise.db import EnterpriseDB, generate_enterprise_db
from aicache.llm.simulated import SimulatedLLM
from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.workflow_correctness import is_workflow_correct

from benchmarks.level3.run_experiment import LEVEL3_IDENTIFICATION_THRESHOLD

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level3"

MUTATION_RATES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_PROBES = 40
TASK_TYPE = "refund_eligibility"
DEFAULT_POLICIES_GOLD_WINDOW = 60  # days; see aicache/enterprise/db.py


def _build_probe_db(n_probes: int) -> tuple[EnterpriseDB, list[tuple[str, str]]]:
    """A DB with n_probes guaranteed-eligible, mutually independent
    (customer, transaction) pairs — explicitly constructed rather than
    sampled from generate_enterprise_db, the same "don't rely on the
    random generator producing exactly the boundary case I need"
    discipline the freshness experiment (EXP-010) used for its own three
    scenarios."""
    db = generate_enterprise_db(seed=0, n_customers=1, n_transactions=1)
    db.customers.clear()
    db.transactions.clear()
    pairs: list[tuple[str, str]] = []
    for i in range(n_probes):
        cust_id = f"CUST-{i:04d}"
        txn_id = f"TXN-{i:06d}"
        db.customers[cust_id] = {
            "customer_id": cust_id,
            "tier": "gold",  # 60-day refund window
            "region": "us-east",
            "account_status": "active",
            "signup_day": 1,
        }
        db.transactions[txn_id] = {
            "transaction_id": txn_id,
            "customer_id": cust_id,
            "amount": 500.0,
            "day": 990,  # age 10 days at "now"=1000 -- well within any window
            "status": "completed",
            "category": "other",
        }
        pairs.append((cust_id, txn_id))
    return db, pairs


def _ground_truth(entities: dict, db: EnterpriseDB) -> dict:
    spec = WORKFLOW_TASK_REGISTRY[TASK_TYPE]
    outcome = spec.fetch(entities, {}, db)
    assert outcome.values is not None
    return spec.executor(**entities, **outcome.values)


@dataclass
class RatePointResult:
    mutation_rate: float
    n_probes: int
    n_correct_after: dict[str, int]

    def correctness(self, baseline: str) -> float:
        return self.n_correct_after[baseline] / self.n_probes


def _run_rate_point(rate: float, db_template: EnterpriseDB, pairs: list[tuple[str, str]], embedder: Embedder, rng: random.Random) -> RatePointResult:
    identifier = TaskIdentifier(embedder=embedder, threshold=LEVEL3_IDENTIFICATION_THRESHOLD, registry=WORKFLOW_TASK_REGISTRY)
    llm = SimulatedLLM(seed=0)
    n_correct = {"semantic_cache": 0, "computation_cache_fresh": 0, "computation_cache_stale_ablation": 0}

    for cust_id, txn_id in pairs:
        db = db_template.clone()
        entities = {"customer_id": cust_id, "transaction_id": txn_id}
        text = f"Is customer {cust_id} eligible for a refund on transaction {txn_id}?"

        semantic = SemanticCache(embedder=embedder, threshold=0.65)
        cache_d = WorkflowComputationCache(db, identifier, use_stale_cache=False)
        cache_e = WorkflowComputationCache(db, identifier, use_stale_cache=True)

        before_fetched_spec = WORKFLOW_TASK_REGISTRY[TASK_TYPE].fetch(entities, {}, db)
        resp = llm.complete(
            text,
            task_hint={"operation": TASK_TYPE, "kind": "workflow", "params": {**entities, **before_fetched_spec.values}},
        )
        semantic.store(text, resp.text)
        cache_d.lookup(text)
        cache_e.lookup(text)

        if rng.random() < rate:
            db.mutate_transaction_status(txn_id, "refunded")

        after_gt = _ground_truth(entities, db)

        sem_after = semantic.lookup(text)
        sem_answer = sem_after.answer if sem_after.hit else ""
        d_after = cache_d.lookup(text)
        e_after = cache_e.lookup(text)

        if is_workflow_correct(TASK_TYPE, sem_answer, after_gt):
            n_correct["semantic_cache"] += 1
        if is_workflow_correct(TASK_TYPE, d_after.served_text or "", after_gt):
            n_correct["computation_cache_fresh"] += 1
        if is_workflow_correct(TASK_TYPE, e_after.served_text or "", after_gt):
            n_correct["computation_cache_stale_ablation"] += 1

    return RatePointResult(mutation_rate=rate, n_probes=len(pairs), n_correct_after=n_correct)


def main() -> None:
    db_template, pairs = _build_probe_db(N_PROBES)
    embedder = Embedder()
    rng = random.Random(2026)

    rows_out: list[dict] = []
    for rate in MUTATION_RATES:
        point = _run_rate_point(rate, db_template, pairs, embedder, rng)
        print(
            f"[volatility] rate={rate:>4.2f}  semantic={point.correctness('semantic_cache')*100:5.1f}%  "
            f"fresh={point.correctness('computation_cache_fresh')*100:5.1f}%  "
            f"stale={point.correctness('computation_cache_stale_ablation')*100:5.1f}%"
        )
        for baseline in ("semantic_cache", "computation_cache_fresh", "computation_cache_stale_ablation"):
            rows_out.append(
                {
                    "mutation_rate": rate,
                    "n_probes": point.n_probes,
                    "baseline": baseline,
                    "n_correct_after": point.n_correct_after[baseline],
                    "correctness_after": point.correctness(baseline),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "volatility_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    json_path = RESULTS_DIR / "volatility_sweep.json"
    with open(json_path, "w") as f:
        json.dump(rows_out, f, indent=2, default=str)

    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
