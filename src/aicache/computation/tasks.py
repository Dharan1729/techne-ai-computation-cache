"""Canonical computation representation for Level 1 (arithmetic) and
Level 2 (business formula) workloads.

Per the brief's §10, the representation is treated as an experimental
variable, not an assumed-optimal design — this is the simplest version
that works: a task type, its required parameters (in the order they
typically appear in a natural-language phrasing, which is what makes
naive positional extraction viable), a pure Python executor that IS the
canonical computation, and a handful of example phrasings used to embed
the task for nearest-neighbor identification (see
``matching/task_identification.py``).

Each task is deterministic and side-effect-free by construction — that's
what makes "reuse the computation, rebind fresh parameters" a coherent
idea for Level 1/2. Level 3 (DB-backed) and Level 4 (agent workflow) would
need a different representation (a workflow/plan, not a closed-form
function) — explicitly out of scope for this prototype; see
docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ParamSpec:
    name: str
    # Positional extraction pulls raw numbers left-to-right out of the
    # phrasing; `transform` converts the raw number into the unit the
    # executor expects (e.g. "20% discount" extracts 20.0, transform
    # divides by 100 so the executor sees a 0..1 fraction).
    transform: Callable[[float], float] = lambda v: v


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    params: list[ParamSpec]
    executor: Callable[..., float]
    output_schema: str
    freshness_requirements: str  # documentation only for Level 1/2 (always "static")
    example_phrasings: list[str]
    level: int = 1
    # Optional regexes (case-insensitive). If non-empty, a request must match
    # at least one of them for this task to be accepted. Added after a
    # multi-seed run showed the embedding match can confuse tasks that share
    # a parameter signature (add/subtract/multiply all take two numbers), so
    # the shape-only validation gate could not catch the confusion. See
    # docs/experiment-log.md, EXP-016.
    lexical_cues: tuple[str, ...] = ()

    @property
    def required_param_names(self) -> list[str]:
        return [p.name for p in self.params]

    def bind_and_execute(self, raw_numbers: list[float]) -> float:
        if len(raw_numbers) < len(self.params):
            raise ValueError(
                f"{self.task_type} needs {len(self.params)} numbers, got {len(raw_numbers)}"
            )
        bound = {
            p.name: p.transform(v) for p, v in zip(self.params, raw_numbers[: len(self.params)])
        }
        return self.executor(**bound)


def _pct(v: float) -> float:
    return v / 100.0


# ---------------------------------------------------------------- Level 1 --

TASK_REGISTRY: dict[str, TaskSpec] = {}


def _register(spec: TaskSpec) -> None:
    TASK_REGISTRY[spec.task_type] = spec


_register(
    TaskSpec(
        task_type="add",
        params=[ParamSpec("x"), ParamSpec("y")],
        executor=lambda x, y: x + y,
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is X plus Y?",
            "Add these two numbers together.",
            "Calculate the sum of X and Y.",
            "Can you sum X and Y?",
            "What do you get if you add X and Y?",
        ],
        lexical_cues=(r"\+", r"\bplus\b", r"\badd(?:ed|ing|s)?\b", r"\bsum\b", r"\btotal\b", r"\bcombined\b"),
    )
)

_register(
    TaskSpec(
        task_type="subtract",
        params=[ParamSpec("x"), ParamSpec("y")],
        executor=lambda x, y: x - y,
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is X minus Y?",
            "Subtract Y from X.",
            "Calculate the difference between X and Y.",
            "What do you get if you take Y away from X?",
        ],
        lexical_cues=(r"\s-\s", r"\bminus\b", r"\bsubtract\w*", r"\bdifference\b", r"\btake(?:s|n)?\s+(?:\w+\s+)?away\b", r"\bless\b", r"\bfewer\b"),
    )
)

_register(
    TaskSpec(
        task_type="multiply",
        params=[ParamSpec("x"), ParamSpec("y")],
        executor=lambda x, y: x * y,
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is X times Y?",
            "Multiply X by Y.",
            "Calculate the product of X and Y.",
            "What do you get if you multiply X and Y together?",
        ],
        lexical_cues=(r"\*", r"\u00d7", r"\btimes\b", r"\bmultipl\w*", r"\bproduct\b"),
    )
)

_register(
    TaskSpec(
        task_type="percentage",
        params=[ParamSpec("pct"), ParamSpec("base")],
        executor=lambda pct, base: (pct / 100.0) * base,
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is X percent of Y?",
            "Calculate X% of Y.",
            "How much is X percent of Y?",
        ],
    )
)

_register(
    TaskSpec(
        task_type="compound_interest",
        params=[ParamSpec("principal"), ParamSpec("rate", _pct), ParamSpec("years")],
        executor=lambda principal, rate, years: principal * ((1 + rate) ** years),
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "If I invest X dollars at Y percent annual interest for Z years, what is the final amount?",
            "Calculate compound interest on X at a rate of Y percent over Z years.",
            "What will X dollars grow to after Z years at Y percent compounded annually?",
        ],
    )
)

# ---------------------------------------------------------------- Level 2 --

_register(
    TaskSpec(
        task_type="revenue",
        params=[ParamSpec("price"), ParamSpec("quantity")],
        executor=lambda price, quantity: price * quantity,
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is the revenue if the price is X and the quantity sold is Y?",
            "Calculate total revenue for X units at a price of Y each.",
            "How much revenue do we make selling Y units at X per unit?",
        ],
        level=2,
    )
)

_register(
    TaskSpec(
        task_type="discounted_price",
        params=[ParamSpec("price"), ParamSpec("discount", _pct)],
        executor=lambda price, discount: price * (1 - discount),
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is the discounted price if the original price is X and the discount is Y percent?",
            "Apply a Y percent discount to a price of X.",
            "How much does an item cost after a Y percent discount off X?",
        ],
        level=2,
    )
)

_register(
    TaskSpec(
        task_type="tax",
        params=[ParamSpec("income"), ParamSpec("rate", _pct)],
        executor=lambda income, rate: income * rate,
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is the tax owed on an income of X at a tax rate of Y percent?",
            "Calculate tax on X dollars of income at a Y percent rate.",
        ],
        level=2,
    )
)

_register(
    TaskSpec(
        task_type="margin",
        params=[ParamSpec("revenue"), ParamSpec("cost")],
        executor=lambda revenue, cost: (revenue - cost) / revenue if revenue else 0.0,
        output_schema="number",
        freshness_requirements="static",
        example_phrasings=[
            "What is the profit margin if revenue is X and cost is Y?",
            "Calculate margin given revenue of X and cost of Y.",
        ],
        level=2,
    )
)
