"""ActionCache: the whole design in one object.

    1. a natural-language request arrives
    2. (on a miss) a small LLM identifies the operation behind it; code removes
       the variable values
    3. the masked template is embedded
    4. the action, its phrasings and their embeddings are stored
    5. a later request retrieves a matching phrasing by embedding, extracts the
       new values by exact alignment with the template, validates them, and
       executes the computation
    6. the reply is built from a template, or by the LLM when the request asks
       for an explanation, and the LLM is not asked to do the computation

    cache = ActionCache.from_config(config)
    out = cache.handle("What is 15% of 240?", llm=ask_my_llm)
    out.source   # "computation" or "llm"
    out.text     # the reply

A learned phrasing answers a request only if the request IS that phrasing with
different numbers. Embedding similarity proposes candidates; it never decides.
Learning runs on a background thread, never on the request path.
"""

from __future__ import annotations

import queue
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .actions import Action, ActionStore
from .config import DEFAULT_LLM_REPLY, Config, build_router
from .embedders import EmbedderLike
from .learner import DEFAULT_SKIP_PATTERNS, Learner, LearnPolicy
from .normalizer import Normalizer
from .router import BindResult, ComputationRouter, Result
from .safe_expr import compile_expr
from .text import align, mask_template

_MODE_FOR_STATUS = {"active": "active", "probation": "shadow"}


@dataclass
class Outcome:
    source: str            # "computation", "computation+llm" or "llm"
    text: str              # the reply
    value: Any = None      # the computed value, when computed
    action: str | None = None
    similarity: float = -1.0
    reason: str = ""       # why the LLM answered, when it did
    request_id: str = ""
    learning: str | None = None  # outcome of learning from this request, when run inline
    latency_ms: float = 0.0


def _binder(template: str) -> Callable[[str], BindResult]:
    def bind(text: str) -> BindResult:
        values = align(template, text)
        return BindResult(True, values) if values is not None else BindResult(False, reason="template_mismatch")
    return bind


class ActionCache:
    def __init__(self, router: ComputationRouter, store: ActionStore, normalizer: Normalizer | None = None,
                 policy: LearnPolicy | None = None, *, synchronous: bool = False, queue_size: int = 100,
                 audit_rate: float = 0.0, llm_reply: str | None = DEFAULT_LLM_REPLY) -> None:
        self.router, self.store = router, store
        self.policy = policy or LearnPolicy(enabled=normalizer is not None)
        self.pinned = set(router.tasks)  # tools from the config: pre-approved, never changed here
        self.synchronous = synchronous
        self.audit_rate = audit_rate
        self._llm_reply = re.compile(llm_reply, re.I) if llm_reply else None
        self._task_of: dict[str, tuple[str, str]] = {}   # router task name -> (action, phrasing text)
        self.learner: Learner | None = None
        if normalizer is not None and self.policy.enabled:
            self.learner = Learner(store, router, router.embedder, normalizer, self.policy,
                                   self._install, router.telemetry)
        self._rng = random.Random(0)
        for a in store.list():
            self._install(a)
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config, *, embedder: EmbedderLike | None = None,
                    normalizer: Normalizer | None = None, store: ActionStore | None = None,
                    synchronous: bool | None = None) -> "ActionCache":
        import os

        router = build_router(cfg, embedder=embedder)
        lc = cfg.learning
        emb_id = getattr(router.embedder, "model_name", None) or type(router.embedder).__name__
        # With learning off nothing is learned, so do not create a database file as a side effect.
        store = store or ActionStore(lc.store if lc.enabled else ":memory:", embedder_id=emb_id)
        if normalizer is None and lc.enabled:
            n = lc.normalizer
            normalizer = Normalizer(n.base_url, n.model, os.environ.get(n.api_key_env, "") if n.api_key_env else None,
                                    n.timeout_s, n.json_mode)
        policy = LearnPolicy(
            enabled=lc.enabled, min_agreements=lc.min_agreements, reject_after=lc.reject_after,
            promotion=lc.promotion, max_actions=lc.max_actions, max_request_chars=lc.max_request_chars,
            max_numbers=lc.max_numbers, skip_patterns=DEFAULT_SKIP_PATTERNS + list(lc.skip_patterns))
        return cls(router, store, normalizer, policy,
                   synchronous=lc.synchronous if synchronous is None else synchronous,
                   queue_size=lc.queue_size, audit_rate=lc.audit_rate,
                   llm_reply=cfg.reply.llm_when if cfg.reply.llm_when else None)

    # ---- phrasings in the router ---------------------------------------------
    def _install(self, a: Action) -> None:
        """Register each phrasing of this action as a router task, in the mode its status allows."""
        for task in [t for t, (act, _) in self._task_of.items() if act == a.name]:
            self.router.remove_task(task)
            del self._task_of[task]
        if not a.enabled:
            return
        expr = compile_expr(a.compute, a.params)
        names = tuple(a.params)
        fn = lambda _e=expr, _n=names, **kw: _e(**{k: kw[k] for k in _n})  # noqa: E731
        vecs = self.store.vectors(a.name)
        if len(vecs) != len(a.templates) or any(v is None for v in vecs):
            fresh = self.router.embedder.encode([mask_template(t.text) for t in a.templates])
            self.store.set_vectors(a.name, [t.text for t in a.templates], fresh)
            vecs = list(fresh)
        for i, t in enumerate(a.templates):
            if t.status == "rejected":
                continue
            task = f"{a.name}#{i}"
            self._task_of[task] = (a.name, t.text)
            self.router.add_task(
                task, fn, examples=[mask_template(t.text)], slots=[], mode=_MODE_FOR_STATUS[t.status],
                format=a.response, masked=True, example_vecs=np.vstack([vecs[i]]), binder=_binder(t.text),
                param_names=a.params, description=f"learned: {a.note}")

    # ---- the request path ---------------------------------------------------
    def handle(self, text: str, llm: Callable[[str], str]) -> Outcome:
        """Answer ``text`` from a stored action if it can be done safely, else
        with ``llm``. ``llm`` takes the request text and returns the reply."""
        t0 = time.perf_counter()
        r = self.router.route(text)
        if r.source == "computation":
            return self._computed(text, r, llm, t0)
        answer = llm(text)
        learning = self.after_llm(text, r, answer)
        return Outcome("llm", answer, None, self.action_name(r.task), r.similarity, r.reason,
                       r.request_id, learning, (time.perf_counter() - t0) * 1000)

    def action_name(self, task: str | None) -> str | None:
        """The action a router task belongs to (learned phrasings are tasks named ``action#i``)."""
        learned = self._task_of.get(task or "")
        return learned[0] if learned else task

    def after_llm(self, text: str, r: Result, answer: str) -> str | None:
        """Called once the LLM has answered a request the cache did not answer. Compares
        shadowed phrasings with the answer, or learns from it. The gateway calls this
        too, because it makes the upstream call itself. Returns the learning outcome
        when learning runs inline."""
        learned = self._task_of.get(r.task or "")
        if r.shadow_value is not None:
            agrees = self.router.note_shadow(r, answer)
            if learned and agrees is not None and self.learner:
                self.learner.observe(learned[0], learned[1], r.args, agrees)
            return None
        if self.learner is not None:
            return self._learn(text, answer)
        return None

    def note_computed(self, text: str, r: Result, audit_llm: Callable[[str], str] | None = None) -> None:
        """Bookkeeping after a stored phrasing answered a request: count the use, and
        sometimes schedule an audit of it against the LLM (``audit_rate``)."""
        learned = self._task_of.get(r.task or "")
        if learned:
            self.store.record_used(*learned)
            if audit_llm and self.learner and self.audit_rate and self._rng.random() < self.audit_rate:
                self._submit(("audit", text, r, audit_llm))

    def wants_llm_reply(self, text: str) -> bool:
        return bool(self._llm_reply and self._llm_reply.search(text))

    def _computed(self, text: str, r: Result, llm: Callable[[str], str], t0: float) -> Outcome:
        learned = self._task_of.get(r.task or "")
        self.note_computed(text, r, llm)
        reply, source = r.text if r.text is not None else str(r.value), "computation"
        if self._llm_reply and self._llm_reply.search(text):
            reply = llm(f"{text}\n\nA verified calculation gives {reply}. Use that value in your reply and do not recompute it.")
            source = "computation+llm"
        return Outcome(source, reply, r.value, learned[0] if learned else r.task, r.similarity, "", r.request_id,
                       None, (time.perf_counter() - t0) * 1000)

    # ---- learning off the request path ----------------------------------------
    def _learn(self, text: str, answer: str) -> str | None:
        assert self.learner is not None
        if self.synchronous:
            return self.learner.consider(text, answer)
        self._submit(("learn", text, answer))
        return None

    def _submit(self, job: tuple) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="aicache-learner", daemon=True)
            self._worker.start()
        try:
            self._q.put_nowait(job)
        except queue.Full:
            self.router.telemetry.record_learning("dropped:queue_full")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if job[0] == "learn":
                    self.learner.consider(job[1], job[2])  # type: ignore[union-attr]
                else:  # audit a live phrasing against the LLM
                    _, text, r, llm = job
                    answer = llm(text)
                    probe = Result("fallback", None, r.task, r.similarity, r.args, shadow_value=r.value)
                    agrees = self.router.note_shadow(probe, answer)
                    learned = self._task_of.get(r.task or "")
                    if agrees is not None and learned:
                        self.learner.observe(learned[0], learned[1], r.args, agrees)  # type: ignore[union-attr]
            except Exception:  # learning must never take the service down
                self.router.telemetry.record_learning("error")
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 60.0) -> None:
        """Wait until queued learning has finished (mainly for tests and evaluation)."""
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.02)

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
        self.store.close()

    # ---- review -------------------------------------------------------------
    def approve(self, name: str, template: str | None = None) -> None:
        if self.learner is None:
            raise RuntimeError("learning is not enabled")
        self.learner.approve(name, template)

    def reject(self, name: str, template: str | None = None, note: str = "rejected by a person") -> None:
        if self.learner is None:
            raise RuntimeError("learning is not enabled")
        self.learner.reject(name, template, note)
