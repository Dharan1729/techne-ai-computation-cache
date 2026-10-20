# Contributing

```bash
pip install -e ".[dev]"
python -m pytest -q          # about 2 minutes; the first run downloads the embedding model
```

- Tests that touch the router use a dependency-free embedder, so they run without a model. The few that use the real embedding model are marked by using the `embedder` fixture.
- Changes to routing or binding behaviour need a test that fails without them. If a change could make a tool answer wrongly, add a case to a golden test or to `examples/finance_labeled.jsonl` and run `aicache eval examples/finance_tools.yaml --labeled examples/finance_labeled.jsonl`.
- Research claims go in `docs/experiment-log.md` with the command that reproduces them, and say how many seeds were run. Single-seed results should say so.
- Keep the core install light. New heavy dependencies go behind an extra.
