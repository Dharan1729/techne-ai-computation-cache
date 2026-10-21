# Documentation

Start with the [user guides](guide/quickstart.md). The rest of this folder is the research record.

| | |
|---|---|
| [guide/](guide/quickstart.md) (see also [putting it in your product](guide/integrate.md)) | Using aicache: quickstart, [how it works](guide/how-it-works.md), [learning](guide/learning.md), config reference, rollout, gateway, integrations |
| [experiment-log.md](experiment-log.md) | Every experiment, newest conclusions last. Entries EXP-001 to EXP-014 are single-seed; EXP-015 to EXP-019 re-test and extend them and take precedence |
| [prior-art.md](prior-art.md), [differentiation.md](differentiation.md) | Existing systems and papers, and what this project adds |
| [problem.md](problem.md), [hypotheses.md](hypotheses.md), [architecture.md](architecture.md), [evaluation.md](evaluation.md) | The original research question, hypotheses and method |
| [level3.md](level3.md), [level3b.md](level3b.md), [level4.md](level4.md), [eviction-future-work.md](eviction-future-work.md) | Database-backed and branching workloads, and deferred work |

## About "the brief"

The research notes refer to "the brief" and its section numbers (for example "brief §12"). That is the original written specification this investigation followed. It is not part of this repository. Some notes cite its section numbers without restating the requirement. Treat those references as background: the results and conclusions do not depend on reading it.

## Optional external baselines

The GPTCache and RedisVL comparison baselines run in isolated environments. `bash scripts/setup_external_baselines.sh` creates them (and starts a Redis on a free port if needed). Without them the benchmark skips those two rows and prints why.
