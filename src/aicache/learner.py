"""Learning new actions from traffic (steps 2 to 4 of the design).

When the main LLM answers a request that no stored phrasing could handle, the
learner asks a small LLM to identify the operation (see ``normalizer``),
verifies the proposal deterministically, embeds the masked template, and stores
it.

* A request with the same operation as an existing action, even under other
  names or with the roles in another order, is added to that action as a new
  *phrasing*. "Same operation" is decided by computing both expressions on
  probe values, not by trusting the model.
* A new phrasing starts on **probation**. It computes silently next to the LLM
  on later requests with exactly that wording, and each comparison is recorded.
  It only answers requests after agreeing with the LLM on several different
  parameter sets and never disagreeing. A disagreement rejects it. Promotion
  can be left to a person (``promotion: manual``).

Why probation: verification against one request cannot rule out a formula that
happens to give the right number once (2 + 2 and 2 * 2 are both 4). Agreement
across different values can. And it is per phrasing, because what was verified
is "this wording means this operation".
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable

from .actions import Action, ActionStore, Template
from .embedders import EmbedderLike
from .normalizer import Known, Normalizer, equivalent_under_renaming, verify
from .router import ComputationRouter
from .telemetry import Telemetry
from .text import mask_template

# Requests that look like they carry personal or secret data are never learned from.
DEFAULT_SKIP_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.-]+",                        # email addresses
    r"\d{9,}",                                          # long digit runs: cards, phones, ids
    r"\b\d{3}-\d{2}-\d{4}\b",                           # US social security format
    r"(?i)\b(password|passwd|secret|api[_ -]?key|token|ssn|iban)\b",
]


@dataclass
class LearnPolicy:
    enabled: bool = True
    min_agreements: int = 5        # distinct parameter sets the LLM must agree on, per phrasing
    reject_after: int = 2          # disagreements that reject a phrasing
    promotion: str = "auto"        # "auto" or "manual"
    max_actions: int = 500
    max_request_chars: int = 300
    max_numbers: int = 5
    skip_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_SKIP_PATTERNS))


class Learner:
    def __init__(self, store: ActionStore, router: ComputationRouter, embedder: EmbedderLike,
                 normalizer: Normalizer, policy: LearnPolicy,
                 install: Callable[[Action], None], telemetry: Telemetry) -> None:
        self.store, self.router, self.embedder = store, router, embedder
        self.normalizer, self.policy = normalizer, policy
        self._install, self.telemetry = install, telemetry
        self._skip = [re.compile(p) for p in policy.skip_patterns]
        self._lock = threading.Lock()  # one learning step at a time keeps merges simple

    # ---- learning ---------------------------------------------------------
    def _skip_reason(self, request: str) -> str | None:
        if not self.policy.enabled:
            return "disabled"
        if len(request) > self.policy.max_request_chars:
            return "too_long"
        if any(p.search(request) for p in self._skip):
            return "sensitive_pattern"
        n = len(re.findall(r"\d[\d,]*(?:\.\d+)?", request))
        if n == 0:
            return "no_numbers"
        if n > self.policy.max_numbers:
            return "too_many_numbers"
        return None

    def consider(self, request: str, answer: str) -> str:
        """Try to learn from one (request, answer) pair. Returns the outcome."""
        why = self._skip_reason(request)
        if why:
            return self._done(f"skipped:{why}")
        with self._lock:
            actions = {a.name: a for a in self.store.list()}
            known = [Known(a.name, a.templates[0].text, a.compute, tuple(a.params))
                     for a in self._nearest(request, actions)]
            proposal = self.normalizer.propose(request, answer, known)
            if isinstance(proposal, str):
                return self._done(f"rejected:{proposal}")
            v = verify(request, answer, proposal)
            if isinstance(v, str):
                return self._done(f"rejected:{v}")
            # The same operation under another name, or with other role names, is a new
            # phrasing of the existing action. Decided by computing, not by trusting the model.
            for old in actions.values():
                if not old.enabled:
                    continue
                rename = equivalent_under_renaming(v.expression, list(v.values), old.compute, old.params)
                if rename is not None:
                    return self._add_phrasing(old, v.template, rename)
            name = v.action
            if name in actions:  # a generic name reused for a different operation
                n = 2
                while f"{v.action}_{n}" in actions:
                    n += 1
                name = f"{v.action}_{n}"
            if self.store.count() >= self.policy.max_actions:
                return self._done("skipped:store_full")
            vec = self.embedder.encode([mask_template(v.template)])
            action = self.store.add(Action(
                name=name, compute=v.expression, params=v.params, response=v.response,
                templates=[Template(v.template, "probation", note="learned from one request")],
                note="learned"), [vec[0]])
            self._install(action)
            return self._done(f"learned:{action.name}")

    def _nearest(self, request: str, actions: dict[str, Action]) -> list[Action]:
        out: list[Action] = []
        for task, _ in self.router.rank(request):
            name = task.split("#")[0]
            if name in actions and actions[name].enabled and actions[name] not in out:
                out.append(actions[name])
            if len(out) == 3:
                break
        return out

    def _add_phrasing(self, old: Action, template: str, rename: dict[str, str]) -> str:
        for new, old_name in rename.items():
            template = template.replace("{" + new + "}", "{@" + old_name + "}")
        template = template.replace("{@", "{")
        existing = old.template(template)
        if existing is not None:
            return self._done("skipped:rejected_phrasing" if existing.status == "rejected" else "skipped:known_phrasing")
        vec = self.embedder.encode([mask_template(template)])[0]
        self.store.add_template(old.name, template, vec, "probation", "learned from one request")
        self._install(self.store.get(old.name))  # type: ignore[arg-type]
        return self._done(f"merged:{old.name}")

    def _done(self, outcome: str) -> str:
        key = outcome if outcome.split(":")[0] in ("skipped", "rejected") else outcome.split(":")[0]
        self.telemetry.record_learning(key)
        return outcome

    # ---- probation --------------------------------------------------------
    def observe(self, name: str, template: str, args: dict | None, agrees: bool) -> str | None:
        """Record one comparison with the LLM for a phrasing; promote or reject it
        if warranted. Returns "promoted", "demoted" or None."""
        action = self.store.get(name)
        t = action.template(template) if action else None
        if action is None or t is None or t.status == "rejected":
            return None
        args = args or {}
        if t.status == "active":
            if agrees:
                return None
            # an audit of a live phrasing found a disagreement: back to probation
            self.store.record_observation(name, template, args, False)
            self.store.set_template_status(name, template, "probation", "audit disagreement")
            self._install(self.store.get(name))  # type: ignore[arg-type]
            self.telemetry.record_learning("demoted")
            return "demoted"
        agree_n, dis = self.store.record_observation(name, template, args, agrees)
        if dis >= self.policy.reject_after:
            self.store.set_template_status(name, template, "rejected", f"disagreed with the LLM {dis} times")
            self._install(self.store.get(name))  # type: ignore[arg-type]
            self.telemetry.record_learning("demoted")
            return "demoted"
        if agree_n >= self.policy.min_agreements and dis == 0:
            if self.policy.promotion == "auto":
                self.store.set_template_status(name, template, "active", f"agreed on {agree_n} parameter sets")
                self._install(self.store.get(name))  # type: ignore[arg-type]
                self.telemetry.record_learning("promoted")
                return "promoted"
            if "ready" not in t.note:
                self.store.set_template_status(name, template, "probation",
                                               f"ready for approval: agreed on {agree_n} parameter sets")
        return None

    def approve(self, name: str, template: str | None = None) -> None:
        """Make a phrasing (or every non-rejected phrasing of an action) answer requests."""
        action = self.store.get(name)
        if action is None:
            raise KeyError(name)
        for t in action.templates:
            if (template is None or t.text == template) and t.status != "rejected":
                self.store.set_template_status(name, t.text, "active", "approved by a person")
        self._install(self.store.get(name))  # type: ignore[arg-type]
        self.telemetry.record_learning("promoted")

    def reject(self, name: str, template: str | None = None, note: str = "rejected by a person") -> None:
        action = self.store.get(name)
        if action is None:
            raise KeyError(name)
        for t in action.templates:
            if template is None or t.text == template:
                self.store.set_template_status(name, t.text, "rejected", note)
        self._install(self.store.get(name))  # type: ignore[arg-type]
        self.telemetry.record_learning("demoted")
