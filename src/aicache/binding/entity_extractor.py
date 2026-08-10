"""Entity extraction for Level 3: unlike ``extract_numbers`` (Level 1/2,
positional — the same class of bug that caused EXP-002/EXP-003), each
entity slot here is matched by its **own** regex pattern, independent of
where it sits in the sentence. Enterprise identifiers are typed and
prefixed (``CUST-0042``, ``TXN-000123``, ``zone-2``), which is exactly
what makes this safe: there's no ambiguity about which slot a given
substring belongs to, so extraction doesn't depend on word order the
way undifferentiated numbers did.

Plain numeric slots (e.g. ``order_total`` on ``loyalty_discount``) still
use ``extract_numbers`` — but only after every matched entity substring
has been stripped from the text first, so an ID's own digits (the
``0042`` in ``CUST-0042``) never get picked up as a numeric parameter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extractor import extract_numbers
from ..computation.workflow_tasks import EntitySlot


@dataclass
class ExtractionResult:
    entities: dict[str, str]
    numbers: dict[str, float]
    missing_slots: list[str]  # entity or number slots that couldn't be extracted


def extract_entities_and_numbers(
    text: str, entity_slots: list[EntitySlot], number_slot_names: list[str]
) -> ExtractionResult:
    entities: dict[str, str] = {}
    missing: list[str] = []
    remaining_text = text

    for slot in entity_slots:
        match = re.search(slot.pattern, text)
        if match:
            entities[slot.name] = match.group(0)
            # Strip the matched span so its digits don't leak into
            # number-slot extraction below.
            remaining_text = remaining_text.replace(match.group(0), " ")
        else:
            missing.append(slot.name)

    raw_numbers = extract_numbers(remaining_text)
    numbers: dict[str, float] = {}
    for i, name in enumerate(number_slot_names):
        if i < len(raw_numbers):
            numbers[name] = raw_numbers[i]
        else:
            missing.append(name)

    return ExtractionResult(entities=entities, numbers=numbers, missing_slots=missing)
