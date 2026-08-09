"""Paraphrase templates per task type.

Placeholders are positional (``{0}``, ``{1}``, ...) and MUST appear in the
text in the same left-to-right order as ``TaskSpec.params`` for that task
— the binding module (``aicache.binding.extractor``) extracts numbers
positionally, so a template that states its numbers out of order would
silently mis-bind (a real limitation, documented in
``aicache/binding/extractor.py`` and in the experiment log, not hidden
behind carefully-ordered templates chosen to avoid ever triggering it —
these templates *are* how the workload avoids that specific bug, on
purpose, so the benchmark measures caching behavior rather than
extraction-bug noise).
"""

from __future__ import annotations

TEMPLATES: dict[str, list[str]] = {
    "add": [
        "What is {0} plus {1}?",
        "Calculate the sum of {0} and {1}.",
        "Can you add {0} and {1} together?",
        "What do you get if you add {0} and {1}?",
        "{0} + {1} = ?",
    ],
    "subtract": [
        "What is {0} minus {1}?",
        "Starting from {0}, subtract {1}.",
        "Calculate the difference between {0} and {1}.",
        "Starting with {0}, what do you get if you take away {1}?",
    ],
    "multiply": [
        "What is {0} times {1}?",
        "Multiply {0} by {1}.",
        "Calculate the product of {0} and {1}.",
        "What do you get if you multiply {0} and {1} together?",
    ],
    "percentage": [
        "What is {0} percent of {1}?",
        "Calculate {0}% of {1}.",
        "How much is {0} percent of {1}?",
    ],
    "compound_interest": [
        "If I invest ${0} at {1} percent annual interest for {2} years, what is the final amount?",
        "Calculate compound interest on ${0} at a rate of {1} percent over {2} years.",
        "What will ${0} grow to at {1} percent compounded annually over {2} years?",
    ],
    "revenue": [
        "What is the revenue if the price is ${0} and the quantity sold is {1}?",
        "Calculate total revenue for a price of ${0} on {1} units sold.",
        "If we price an item at ${0} and sell {1} units, what's the revenue?",
    ],
    "discounted_price": [
        "What is the discounted price if the original price is ${0} and the discount is {1} percent?",
        "A price of ${0} gets a {1} percent discount applied to it — what's the result?",
        "An item costs ${0}. What does it cost after a {1} percent discount?",
    ],
    "tax": [
        "What is the tax owed on an income of ${0} at a tax rate of {1} percent?",
        "Calculate tax on ${0} of income at a {1} percent rate.",
    ],
    "margin": [
        "What is the profit margin if revenue is ${0} and cost is ${1}?",
        "Calculate margin given revenue of ${0} and cost of ${1}.",
    ],
}
