"""The dedicated state-mutation experiment (brief §8/§12): request ->
mutate -> repeat the same request, for three distinct mutation types,
comparing semantic_cache / computation_cache_fresh (D) /
computation_cache_stale_ablation (E) on each before/after pair.

Unlike run_experiment.py, this is intentionally NOT a large statistical
sample — it's a small, hand-picked, fully-inspectable set of scenarios
built so a reader can check every one by eye. The claim it supports
("D reflects the mutation, E doesn't, and that's the whole point of
never caching fetched data") only needs to be shown clearly, not
averaged over hundreds of requests — see docs/level3.md for why this is
a separate experiment from the primary comparison rather than folded
into it.

Each scenario:
  1. Issue a request. Record what each baseline answers.
  2. Mutate the database (transaction status / customer tier / policy).
  3. Issue the identical request again. Record what each baseline
     answers now.
  4. Compare against fresh ground truth computed independently at each
     of the two points in time (never read from any baseline).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from aicache.cache.workflow_computation import WorkflowComputationCache
from aicache.cache.semantic import SemanticCache
from aicache.computation.workflow_tasks import WORKFLOW_TASK_REGISTRY
from aicache.enterprise.db import EnterpriseDB, generate_enterprise_db
from aicache.llm.simulated import SimulatedLLM
from aicache.llm.workflow_rendering import render_workflow_answer
from aicache.matching.embedder import Embedder
from aicache.matching.task_identification import TaskIdentifier
from aicache.metrics.workflow_correctness import is_workflow_correct

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "level3"
IDENTIFICATION_THRESHOLD = 0.45


def _fetch_and_execute(task_type: str, entities: dict, numbers: dict, db: EnterpriseDB) -> tuple[dict, dict]:
    """Independent oracle: fetch + execute directly, called here, never
    read back from any baseline under test. Returns (fetched_raw_data,
    executed_result) — the LLM needs the *raw* fetched fields (e.g.
    transaction_status) to compute an answer itself, not the already-
    computed result fields (e.g. eligible), which is what spec.executor
    actually takes as keyword arguments."""
    spec = WORKFLOW_TASK_REGISTRY[task_type]
    outcome = spec.fetch(entities, numbers, db)
    assert outcome.values is not None, f"scenario setup produced an invalid fetch: {outcome.error}"
    result = spec.executor(**entities, **numbers, **outcome.values)
    return outcome.values, result


@dataclass
class MutationProbeResult:
    scenario: str
    mutation: str
    text: str
    before_ground_truth: dict
    after_ground_truth: dict
    ground_truth_changed: bool
    baseline: str
    before_answer: str
    after_answer: str
    before_correct: bool
    after_correct: bool


def _probe(
    label: str,
    mutation_description: str,
    text: str,
    task_type: str,
    entities: dict,
    numbers: dict,
    db: EnterpriseDB,
    mutate_fn,
    llm,
    embedder: Embedder,
) -> list[MutationProbeResult]:
    before_fetched, before_gt = _fetch_and_execute(task_type, entities, numbers, db)

    identifier = TaskIdentifier(embedder=embedder, threshold=IDENTIFICATION_THRESHOLD, registry=WORKFLOW_TASK_REGISTRY)
    semantic = SemanticCache(embedder=embedder, threshold=0.65)
    cache_d = WorkflowComputationCache(db, identifier, use_stale_cache=False)
    cache_e = WorkflowComputationCache(db, identifier, use_stale_cache=True)

    # --- before ---
    sem_before = semantic.lookup(text)
    if not sem_before.hit:
        resp = llm.complete(text, task_hint={"operation": task_type, "kind": "workflow", "params": {**entities, **numbers, **before_fetched}})
        semantic.store(text, resp.text)
        sem_before_answer = resp.text
    else:
        sem_before_answer = sem_before.answer or ""

    d_before = cache_d.lookup(text)
    e_before = cache_e.lookup(text)

    # --- mutate ---
    mutate_fn(db)
    after_fetched, after_gt = _fetch_and_execute(task_type, entities, numbers, db)
    ground_truth_changed = before_gt != after_gt

    # --- after ---
    sem_after = semantic.lookup(text)  # will hit — same text, cache unaware of the mutation
    sem_after_answer = sem_after.answer if sem_after.hit else ""

    d_after = cache_d.lookup(text)
    e_after = cache_e.lookup(text)

    results = []
    for baseline, before_ans, after_ans in [
        ("semantic_cache", sem_before_answer, sem_after_answer),
        ("computation_cache_fresh", d_before.served_text or "", d_after.served_text or ""),
        ("computation_cache_stale_ablation", e_before.served_text or "", e_after.served_text or ""),
    ]:
        results.append(
            MutationProbeResult(
                scenario=label,
                mutation=mutation_description,
                text=text,
                before_ground_truth=before_gt,
                after_ground_truth=after_gt,
                ground_truth_changed=ground_truth_changed,
                baseline=baseline,
                before_answer=before_ans,
                after_answer=after_ans,
                before_correct=is_workflow_correct(task_type, before_ans, before_gt),
                after_correct=is_workflow_correct(task_type, after_ans, after_gt),
            )
        )
    return results


def main() -> None:
    db = generate_enterprise_db(seed=42, n_customers=80, n_transactions=200)
    llm = SimulatedLLM(seed=0)
    embedder = Embedder()

    all_results: list[MutationProbeResult] = []

    # Scenario 1: transaction status mutation (completed -> refunded)
    cust_id = next(cid for cid, c in db.customers.items() if c["tier"] == "gold")
    txn_id = "TXN-000000"
    db.transactions[txn_id].update(customer_id=cust_id, status="completed", day=990, amount=500.0)
    all_results += _probe(
        "refund_eligibility_after_status_change",
        "transaction status: completed -> refunded",
        f"Is customer {cust_id} eligible for a refund on transaction {txn_id}?",
        "refund_eligibility",
        {"customer_id": cust_id, "transaction_id": txn_id},
        {},
        db,
        lambda d: d.mutate_transaction_status(txn_id, "refunded"),
        llm,
        embedder,
    )

    # Scenario 2: customer tier mutation (bronze -> gold, changes discount)
    cust2 = next(cid for cid, c in db.customers.items() if c["tier"] == "bronze")
    all_results += _probe(
        "loyalty_discount_after_tier_upgrade",
        "customer tier: bronze -> gold",
        f"What discount does customer {cust2} get on a $400 order?",
        "loyalty_discount",
        {"customer_id": cust2},
        {"order_total": 400.0},
        db,
        lambda d: d.mutate_customer_tier(cust2, "gold"),
        llm,
        embedder,
    )

    # Scenario 3: policy mutation (gold refund window shortened)
    cust3 = next(cid for cid, c in db.customers.items() if c["tier"] == "gold")
    txn3 = "TXN-000001"
    db.transactions[txn3].update(customer_id=cust3, status="completed", day=955, amount=300.0)
    # day=955 -> age 45 days: within the original 60-day gold window,
    # outside a shortened 30-day window - exactly the boundary a policy
    # mutation should flip.
    all_results += _probe(
        "refund_eligibility_after_policy_change",
        "gold tier refund_window_days: 60 -> 30",
        f"Is customer {cust3} eligible for a refund on transaction {txn3}?",
        "refund_eligibility",
        {"customer_id": cust3, "transaction_id": txn3},
        {},
        db,
        lambda d: d.mutate_policy("gold", refund_window_days=30),
        llm,
        embedder,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "freshness_experiment.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)

    print("=== Freshness / state-mutation experiment ===\n")
    for r in all_results:
        flag = "CHANGED" if r.ground_truth_changed else "unchanged"
        print(f"[{r.scenario}] {r.mutation} (ground truth {flag})")
        print(f"  request: {r.text!r}")
        print(f"  baseline: {r.baseline}")
        print(f"    before: {r.before_answer!r}  correct={r.before_correct}")
        print(f"    after:  {r.after_answer!r}  correct={r.after_correct}")
        print()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
