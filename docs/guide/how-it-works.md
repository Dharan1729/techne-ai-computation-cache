# How it works

aicache is built around one idea: **the reusable thing is the operation, not the previous answer.**

"What is 15% of 240?" and "What is 20% of 50?" have different answers and the same operation. A semantic cache sees two similar sentences and may return the wrong number. aicache stores the operation, extracts the new values, checks them, and computes.

## The six steps

1. **A request arrives**, in natural language.
2. **A small LLM identifies the operation.** On a request the cache could not answer, the main LLM answers it as usual. In the background a small model is shown the request, the answer, and a skeleton of the request with its numbers removed (`What is {n1}% of {n2}?`). It names the operation, names the roles, and writes the formula: `base * pct / 100`. Code, not the model, removes the values. When a model was asked to copy the sentence itself, it often rewrote it.
3. **The masked action is embedded.** The template becomes `What is #% of #?` and is embedded, so a request and a stored template reduce to the same shape whatever the numbers are.
4. **The action and its embedding are stored.** In a SQLite file: the formula, the parameter names, the reply template, every phrasing seen, and the evidence for each phrasing.
5. **A similar request retrieves the action.** Its masked form is embedded, and the closest stored phrasings are candidates. Then the request must *be* a stored phrasing with different numbers, which is how the new values are extracted. The values are validated, and the formula runs.
6. **A reply is generated.** From the action's reply template, with no LLM call. If the request asks for an explanation ("explain", "why", "walk me through"), the LLM writes the reply but is handed the computed value and told not to recompute it.

## What keeps it safe

Learning from a model's output is risky, so nothing a model says is trusted.

**Before an action is stored**, each proposal is checked by code:

- the formula is parsed by a whitelist (no `eval`) and uses every parameter,
- running it on the request's own numbers reproduces the answer the main LLM gave,
- if the formula computes the same function as an existing action (tested on several probe values, with parameters matched in every order), the phrasing joins that action; the model's naming is not trusted for this.

**Before a phrasing answers a request**, it spends time on *probation*. It computes silently next to the LLM on later requests with that exact wording, and each comparison is recorded. It starts answering only after it has agreed with the LLM on several different sets of numbers and never disagreed. One check against one request cannot rule out a formula that happens to give the right number once (2 + 2 and 2 × 2 are both 4). Agreement across different numbers can. Two disagreements reject a phrasing for good. Or leave promotion to a person: `promotion: manual`.

**Trust belongs to a wording**, not to an action. What was verified is "this wording means this operation". A new wording on a trusted action starts on probation by itself.

**A request is answered only if it is a stored wording.** Matching is exact, up to case, spacing and trailing punctuation: "What is 7% of 300" matches "What is {pct}% of {base}?", and "What's 7% of 300?" does not. The embedding finds candidates; it never decides. An earlier version inferred where numbers sit from the words around them. In a real-model test that was too generic: a request to *add* was answered by a *multiply* action. The model of "same wording, different numbers" does not have that failure.

**Learning never touches the request path.** It runs on a background thread, with a bounded queue.

**Some requests are never learned from.** Anything that looks like an email, a long digit run (cards, phones), a social security number, or the words password, secret, token, api key. You can add your own patterns.

## What it costs

Each *new* wording costs one extra small-LLM call, once. After a wording is trusted, requests with it cost no LLM call. How much that saves depends entirely on how often the same wordings come back with different numbers. See [the evidence](../experiment-log.md) for a measurement and its limits.

## What it will not do

- **It does not generalize to paraphrases.** A new way of asking is answered by the main LLM until it has been learned, then trusted. This is deliberate, and it is the price of the property above.
- **A wrong teacher teaches wrong things.** The main LLM's answers are the ground truth for verification. If it is consistently wrong in the same way, the cache can learn the wrong formula. Use a strong main model, `promotion: manual`, or audit live actions.
- **Numbers and one formula.** An action is an arithmetic expression over the numbers in the request. Requests that need other text from the request (names, tickers), tools with side effects, or judgment are not learned.
- **Units are the model's call.** If the main LLM answered "0.5%" as 0.005, the cache learns that.
