from aicache.binding.extractor import extract_numbers


def test_extracts_plain_integers():
    assert extract_numbers("What is 10 plus 20?") == [10.0, 20.0]


def test_extracts_decimals():
    assert extract_numbers("Price is 19.99 with 5.5 percent tax") == [19.99, 5.5]


def test_extracts_comma_grouped_thousands():
    assert extract_numbers("Invest $69,257 at 8 percent") == [69257.0, 8.0]


def test_extracts_negative_numbers():
    assert extract_numbers("The change was -42 units") == [-42.0]


def test_no_numbers_returns_empty_list():
    assert extract_numbers("What is the capital of France?") == []


def test_preserves_left_to_right_order():
    numbers = extract_numbers("first 3, then 1, then 2")
    assert numbers == [3.0, 1.0, 2.0]
