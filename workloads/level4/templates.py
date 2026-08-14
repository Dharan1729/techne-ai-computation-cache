"""Parameterized workload templates for Level 4 — deliberately separate
from `aicache.computation.graph_tasks.RETURN_DECISION_PHRASINGS`, which
are static example sentences used only for embedding-based task
identification (`TaskIdentifier`), not meant to have entities
substituted into them. Conflating the two was a real bug caught during
development: reusing the static phrasings directly as `.format()`
templates silently did nothing (they contain no `{0}`/`{1}`
placeholders), so every generated workload item rendered the same
literal example text regardless of which entities were actually drawn
— see `docs/level4.md`'s "Bugs found and fixed". Mirrors
`workloads/level3/templates.py`'s convention exactly: entity slots
first, in `entity_slots` declaration order, then number slots.
"""

LEVEL4_TEMPLATES = {
    "return_decision": [
        "Should we approve the return for customer {0} on transaction {1}?",
        "Process the return request for {0}'s transaction {1}.",
        "What is the return decision for customer {0} regarding transaction {1}?",
    ],
}
