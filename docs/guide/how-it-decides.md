# How a request is decided

For every request the router makes the same sequence of checks. It computes only if all of them pass. Otherwise the request goes to your fallback, usually the model.

1. **Propose candidates.** Each tool's examples are embedded. The request is scored against them. Every tool whose best example scores at least `min_similarity` is a candidate, closest first.
2. **Cue check.** If a candidate has `cues`, one must match.
3. **Bind the numbers.** Every number in the request must be claimed by exactly one slot, and every slot must have a number. A slot claims a number by a keyword just before it (`before`) or just after it (`after`). The nearest keyword wins. A number nothing claims, a slot left empty, or two numbers equally close to the same slot means refuse. `min`, `max` and `integer` are checked here.
4. **Resolve ties.** If more than one candidate binds, the closest wins, unless a second one binds within `ambiguity_margin` with a different answer. Then the request is refused as ambiguous.
5. **Run the tool.** If it raises, for example divide by zero, the request is refused.

The reason is always recorded, for example `missing:account_size`, `unclaimed_number`, `ambiguous_task`, `below_similarity`, `cue_missing`, `tool_error`, `invalid:pct`.

## Why binding does the safety work

In our finance test, similarity scores for real requests (0.46 to 0.67) and unrelated ones (0.37 to 0.70) overlapped almost completely, so a threshold alone could not separate them. The binding step could: no unrelated request in the test bound all of a tool's slots. On a 131-request labeled set, results were the same at every threshold from 0 to 0.45: no wrong answers and no false triggers. See [EXP-015 and EXP-018](../experiment-log.md).

That is evidence about those tools and that phrasing, not a guarantee. A tool is only as safe as its slots. Measure on your own requests.

## What it will not do

- It does not understand meaning. It finds numbers and keywords. A request that says "20" twice in different roles will be refused, not interpreted.
- It does not handle units for you. A slot gives the number as written. Use `scale`, or write the formula to expect the unit users type.
- It cannot tell whether a request that binds cleanly was *meant* for another tool you did not register.
- It is not suited to tools with side effects, tools that need context outside the request text, or answers that need judgment.

## Known failure patterns

| Pattern | What happens | What to do |
|---|---|---|
| Two tools with the same parameters, different operation | Embeddings confuse them | Add `cues`. `validate` warns about this. |
| Keyword after the number but only `before` given | The number is not claimed, request refused | Add an `after` pattern. Safe failure. |
| Ratio phrasings like "3 to 1" | Refused: a stray "1" is unclaimed | Add that phrasing to your slots or accept the fallback. |
| Percent typed as 0.5 vs 50% | Silent unit error if your formula expects the other | Fix the formula or use `scale`; test both. |
