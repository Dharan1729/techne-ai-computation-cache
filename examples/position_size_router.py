"""Put a deterministic tool in front of an LLM, with a safe fallback.

    python examples/position_size_router.py

Offline except for the one-time embedding-model download. The "LLM" here is a
stub that just labels the request, so you can see which requests the router
computed and which it handed over.
"""

from aicache import ComputationRouter, Slot, TextSlot


def position_size(symbol, entry_price, stop_price, account_size, risk_percent):
    """The deterministic tool. Any pure function of its arguments works."""
    risk_amount = account_size * risk_percent / 100
    return round(risk_amount / abs(entry_price - stop_price), 2)


router = ComputationRouter(min_similarity=0.48)
router.add_task(
    "position_size",
    position_size,
    examples=[
        "How many shares should I buy given my account size and risk percent?",
        "Calculate the position size for a trade with an entry price and a stop price.",
        "What size position can I take risking a percent of my account?",
    ],
    slots=[
        Slot("entry_price", before=r"entry(?: price)?|enter(?:ing)?|buy(?:ing)?|in at", after=r"entry"),
        Slot("stop_price", before=r"stop(?:[- ]loss)?(?: price)?|stops? out|out at"),
        Slot("account_size", before=r"account(?: size)?|portfolio|have|(?:%|percent)\s+of",
             after=r"(?:in\s+(?:the|my)\s+)?(?:account|portfolio)"),
        Slot("risk_percent", before=r"risk(?:ing)?(?: percent| limit)?", after=r"%|percent",
             validate=lambda v: 0 < v <= 100),  # stays a percent: 0.5 means 0.5%, never 0.005
    ],
    text_slots=[TextSlot("symbol", r"\b([A-Z]{2,5})\b")],
)


def ask_llm(text: str) -> str:
    return f"[LLM would answer: {text!r}]"


if __name__ == "__main__":
    requests = [
        "Size my TSLA trade: stop 436.81, entry 476.92, risking 0.5 percent of $50000.",
        "Position size for AAPL: account $25000, risk 2%, entry 150, stop 140.",
        "How many AAPL shares can I buy with entry 150 and stop 140?",  # no account or risk: refuse
        "What does position sizing mean?",  # not a calculation: fall back
    ]
    for text in requests:
        r = router.run(text, fallback=ask_llm)
        print(f"{r.source:11s} {r.value}   reason={r.reason or '-'}\n  {text}")
