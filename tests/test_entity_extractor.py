from aicache.binding.entity_extractor import extract_entities_and_numbers
from aicache.computation.workflow_tasks import EntitySlot

CUST_TXN_SLOTS = [
    EntitySlot("customer_id", r"CUST-\d{4}"),
    EntitySlot("transaction_id", r"TXN-\d{6}"),
]


def test_extracts_entities_regardless_of_order():
    result = extract_entities_and_numbers(
        "Assess the risk of transaction TXN-000123 for customer CUST-0042",
        CUST_TXN_SLOTS,
        [],
    )
    assert result.entities == {"customer_id": "CUST-0042", "transaction_id": "TXN-000123"}
    assert not result.missing_slots


def test_entity_digits_do_not_leak_into_number_slots():
    result = extract_entities_and_numbers(
        "What discount does customer CUST-0042 get on a $250 order?",
        [EntitySlot("customer_id", r"CUST-\d{4}")],
        ["order_total"],
    )
    assert result.entities["customer_id"] == "CUST-0042"
    assert result.numbers["order_total"] == 250.0


def test_missing_entity_slot_is_reported():
    result = extract_entities_and_numbers(
        "Is this customer eligible for a refund?", CUST_TXN_SLOTS, []
    )
    assert set(result.missing_slots) == {"customer_id", "transaction_id"}


def test_missing_number_slot_is_reported():
    result = extract_entities_and_numbers(
        "What discount does customer CUST-0042 get?",
        [EntitySlot("customer_id", r"CUST-\d{4}")],
        ["order_total"],
    )
    assert "order_total" in result.missing_slots
