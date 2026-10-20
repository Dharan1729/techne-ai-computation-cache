# Rolling it out safely

Do these in order. Each step needs only the config and some real requests.

## 1. Golden tests in the config

Every tool should have `tests:` with requests that compute and requests that must be refused. `aicache test aicache.yaml` runs them. Add a refusal test for every near-miss you can think of: the same sentence with a number missing, the tool's topic without a calculation.

## 2. Evaluate on labeled real requests

Collect a few hundred real requests. Label the expected number, or `refuse` for requests the model should handle.

```bash
aicache eval aicache.yaml --labeled requests.jsonl
```

Read two lines: **WRONG executions** and **false triggers**. Both should be 0. The report also prints a 95% upper bound on the bad-execution rate. With 0 bad in 130 requests the bound is still about 3%, so more labeled data buys more certainty. The command exits non-zero if bad executions exceed `--max-wrong` (default 0), so put it in CI and fail the build when a config change makes a tool answer wrongly.

The `why requests were refused` line tells you where coverage is lost. It is the to-do list for improving slots.

## 3. Calibrate the threshold

```bash
aicache calibrate aicache.yaml --labeled requests.jsonl
```

This sweeps `min_similarity` and recommends the highest threshold that has no bad executions and keeps at least 98% of the best coverage. A threshold that is merely "safe" because it refuses everything is never recommended. Set `min_similarity` and run `eval` again. Redo this whenever you change the embedder: a threshold chosen for one embedder is wrong for another.

## 4. Shadow mode on live traffic

Set `mode: shadow` on a tool. The gateway (or `router.run`) computes the answer but still lets the model answer, and records whether the model's number matches. Nothing computed is ever served.

```bash
curl -H "authorization: Bearer $ADMIN" localhost:8080/v1/aicache/stats
# "shadow": {"percent_of:agree": 412, "percent_of:disagree": 3}
```

Look at the disagreements. Either the model was wrong, which is the case for the tool, or your slots bound something wrongly, which is a bug to fix. Shadow comparison reads the last number in the model's reply, so it is noisy for tools whose answers are not a single number. Streaming responses are not compared.

## 5. Go active, with a kill switch

Switch the tool to `active`, one tool at a time. If something looks wrong, turn it off without a restart:

```bash
curl -X POST -H "authorization: Bearer $ADMIN" -H "content-type: application/json" \
     -d '{"mode": "off"}' localhost:8080/v1/aicache/tools/percent_of/mode
```

Runtime changes last until the process restarts. Change the config to make them permanent.

## 6. Keep watching

- `aicache stats decisions.jsonl` summarizes the decision log: computed share, why requests were refused, per-tool counts.
- `/metrics` exposes Prometheus counters: requests by outcome, refusal reasons, computed per tool, shadow agreement, routing latency.
- A rising share of one refusal reason, such as `unclaimed_number`, usually means users have started phrasing something a new way. Add the phrasing to a tool's slots and to its tests.
