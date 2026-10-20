"""Declarative configuration: define tools in YAML, no code needed.

    version: 1
    min_similarity: 0.5
    embedder: {type: sentence-transformers}
    upstream: {base_url: http://localhost:11434/v1, model: llama3}
    tools:
      - name: percent_of
        examples: ["What is X percent of Y?", "Calculate X% of Y."]
        slots:
          - {name: pct, after: "%|percent", min: 0, max: 100}
          - {name: base, before: "of"}
        compute: "base * pct / 100"
        format: "{value:.2f}"
        tests:
          - {text: "What is 15% of 240?", expect: 36}
          - {text: "What is 15% of that?", expect: refuse}

A tool is either a ``compute`` expression (see safe_expr.py, no code execution)
or ``python: "package.module:function"`` for anything richer. The ``python``
form imports and runs code named in the config, so only load configs you trust.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .embedders import EmbedderLike, make_embedder
from .router import MODES, ComputationRouter, Slot, TextSlot, _FIELD_RE
from .safe_expr import ExprError, compile_expr, free_names
from .telemetry import Telemetry

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """The configuration is invalid. The message lists every problem found."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _regex(v: str, what: str) -> str:
    try:
        re.compile(v)
    except re.error as exc:
        raise ValueError(f"{what} is not a valid regular expression: {exc}") from exc
    return v


class SlotCfg(_Strict):
    name: str
    before: str = ""
    after: str = ""
    min: float | None = None
    max: float | None = None
    integer: bool = False
    scale: float | None = None  # multiply the number as written, e.g. 0.01 to turn 5 (percent) into 0.05

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError("must be an identifier (letters, digits, underscore)")
        return v

    @field_validator("before", "after")
    @classmethod
    def _re(cls, v: str) -> str:
        return _regex(v, "pattern") if v else v

    @model_validator(mode="after")
    def _anchor(self) -> "SlotCfg":
        if not self.before and not self.after:
            raise ValueError("give 'before' or 'after' so the number can be found")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min is greater than max")
        return self


class TextSlotCfg(_Strict):
    name: str
    pattern: str

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError("must be an identifier")
        return v

    @field_validator("pattern")
    @classmethod
    def _re(cls, v: str) -> str:
        return _regex(v, "pattern")


class ToolTest(_Strict):
    text: str
    expect: float | Literal["refuse"]


class ToolCfg(_Strict):
    name: str
    description: str = ""
    examples: list[str] = Field(min_length=1)
    slots: list[SlotCfg] = Field(min_length=1)
    text_slots: list[TextSlotCfg] = []
    cues: list[str] = []
    compute: str | None = None
    python: str | None = None
    format: str = "{value}"
    mode: Literal["active", "shadow", "off"] = "active"
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    tests: list[ToolTest] = []

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError("must be an identifier")
        return v

    @field_validator("cues")
    @classmethod
    def _cues(cls, v: list[str]) -> list[str]:
        return [_regex(c, "cue") for c in v]

    @model_validator(mode="after")
    def _check(self) -> "ToolCfg":
        if (self.compute is None) == (self.python is None):
            raise ValueError("give exactly one of 'compute' and 'python'")
        names = [s.name for s in self.slots] + [t.name for t in self.text_slots]
        if len(set(names)) != len(names):
            raise ValueError("slot names must be unique")
        if self.compute is not None:
            numeric = [s.name for s in self.slots]
            try:
                compile_expr(self.compute, numeric)
            except ExprError as exc:
                raise ValueError(f"compute: {exc}") from exc
        if self.python is not None and not re.match(r"^[\w.]+:[\w.]+$", self.python):
            raise ValueError("python must look like 'package.module:function'")
        for f in _FIELD_RE.finditer(self.format):
            if f.group("name") != "value" and f.group("name") not in names:
                raise ValueError(f"format refers to unknown field {{{f.group('name')}}}")
        return self


class UpstreamCfg(_Strict):
    base_url: str
    api_key_env: str | None = None
    model: str | None = None  # overrides the model named in each request
    timeout_s: float = 60.0


class EmbedderCfg(_Strict):
    type: Literal["sentence-transformers", "hashing", "openai"] = "sentence-transformers"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None


class TelemetryCfg(_Strict):
    log_file: str | None = None
    log_text: bool = False  # off by default: records hold a hash, not the request text
    ring_size: int = Field(default=1000, ge=1)


class ServerCfg(_Strict):
    api_key_env: str | None = None      # env var holding the bearer token clients must send to /v1/chat/completions
    admin_token_env: str | None = None  # env var holding the token for /v1/aicache/* endpoints (disabled if unset)
    max_body_bytes: int = Field(default=1_000_000, ge=1_000)
    max_text_chars: int = Field(default=4000, ge=1)  # longer requests skip the router and go to the LLM


class Config(_Strict):
    version: Literal[1] = 1
    min_similarity: float = Field(default=0.5, ge=-1.0, le=1.0)
    ambiguity_margin: float = Field(default=0.03, ge=0.0, le=1.0)
    embedder: EmbedderCfg = EmbedderCfg()
    upstream: UpstreamCfg | None = None
    telemetry: TelemetryCfg = TelemetryCfg()
    server: ServerCfg = ServerCfg()
    tools: list[ToolCfg] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> "Config":
        names = [t.name for t in self.tools]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate tool names: {', '.join(sorted(dup))}")
        return self


def load_config(source: str | Path | dict) -> Config:
    """Load and validate a config from a YAML path, a YAML string, or a dict."""
    if isinstance(source, dict):
        raw: Any = source
    else:
        text = Path(source).read_text() if isinstance(source, Path) or (
            isinstance(source, str) and "\n" not in source and Path(source).exists()) else str(source)
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("the config must be a mapping at the top level")
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"  {loc}: {err['msg'].removeprefix('Value error, ')}")
        raise ConfigError("invalid config:\n" + "\n".join(lines)) from None


def _slot(cfg: SlotCfg) -> Slot:
    def validate(v: float, c: SlotCfg = cfg) -> bool:
        if c.min is not None and v < c.min:
            return False
        if c.max is not None and v > c.max:
            return False
        return not (c.integer and v != int(v))

    transform = (lambda v, s=cfg.scale: v * s) if cfg.scale is not None else None
    needs_validate = cfg.min is not None or cfg.max is not None or cfg.integer
    return Slot(cfg.name, cfg.before, cfg.after, transform, validate if needs_validate else None)


def _python_tool(spec: str):
    mod, attr = spec.split(":", 1)
    try:
        obj: Any = importlib.import_module(mod)
        for part in attr.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot import tool {spec!r}: {exc}") from exc
    if not callable(obj):
        raise ConfigError(f"{spec!r} is not callable")
    return obj


def build_router(cfg: Config, embedder: EmbedderLike | None = None,
                 telemetry: Telemetry | None = None) -> ComputationRouter:
    emb = embedder or make_embedder(cfg.embedder.type, model=cfg.embedder.model,
                                    base_url=cfg.embedder.base_url, api_key_env=cfg.embedder.api_key_env)
    tele = telemetry or Telemetry(ring_size=cfg.telemetry.ring_size, log_file=cfg.telemetry.log_file,
                                  log_text=cfg.telemetry.log_text)
    router = ComputationRouter(emb, cfg.min_similarity, tele, cfg.ambiguity_margin)
    for t in cfg.tools:
        numeric = [s.name for s in t.slots]
        if t.compute is not None:
            expr = compile_expr(t.compute, numeric)
            fn = (lambda _e=expr, _n=tuple(numeric), **kw: _e(**{k: kw[k] for k in _n}))
        else:
            fn = _python_tool(t.python)  # type: ignore[arg-type]
        router.add_task(
            t.name, fn, examples=t.examples, slots=[_slot(s) for s in t.slots],
            text_slots=[TextSlot(x.name, x.pattern) for x in t.text_slots], cues=t.cues,
            description=t.description, mode=t.mode, min_similarity=t.min_similarity, format=t.format)
    return router


# ---- golden tests and lint ----------------------------------------------------

@dataclass
class TestOutcome:
    __test__ = False  # not a pytest class
    tool: str
    text: str
    expect: Any
    passed: bool
    detail: str


def run_golden_tests(cfg: Config, router: ComputationRouter) -> list[TestOutcome]:
    """Run every ``tests:`` entry in the config. Tools in shadow or off mode are
    tested as if active, so the compute logic is checked before rollout. Test
    runs are not counted in telemetry."""
    saved_tele, saved_modes = router.telemetry, {n: t.mode for n, t in router.tasks.items()}
    router.telemetry = Telemetry(ring_size=1)
    for n in saved_modes:
        router.set_mode(n, "active")
    out: list[TestOutcome] = []
    try:
        for tool in cfg.tools:
            for tc in tool.tests:
                r = router.route(tc.text)
                if tc.expect == "refuse":
                    ok = r.source != "computation"
                    detail = "refused" if ok else f"computed {r.value!r} but should have refused"
                else:
                    ok = (r.source == "computation" and r.task == tool.name
                          and abs(float(r.value) - float(tc.expect)) <= max(1e-6, 1e-9 * abs(float(tc.expect))))
                    detail = ("ok" if ok else
                              f"got source={r.source} task={r.task} value={r.value!r} reason={r.reason or '-'}")
                out.append(TestOutcome(tool.name, tc.text, tc.expect, ok, detail))
    finally:
        router.telemetry = saved_tele
        for n, m in saved_modes.items():
            router.set_mode(n, m)
    return out


@dataclass
class Issue:
    level: str  # "warning" or "info"
    tool: str
    message: str


def lint_config(cfg: Config) -> list[Issue]:
    issues: list[Issue] = []
    for t in cfg.tools:
        if len(t.examples) < 3:
            issues.append(Issue("warning", t.name, "fewer than 3 example phrasings; identification will be fragile"))
        if not any(x.expect != "refuse" for x in t.tests):
            issues.append(Issue("warning", t.name, "no positive tests; add requests that should compute"))
        if not any(x.expect == "refuse" for x in t.tests):
            issues.append(Issue("warning", t.name, "no 'refuse' tests; add requests that must NOT compute"))
        if t.mode == "shadow":
            issues.append(Issue("info", t.name, "shadow mode: computed answers are compared with the LLM, never served"))
        if t.mode == "off":
            issues.append(Issue("info", t.name, "mode is off: this tool is ignored"))
    by_sig: dict[int, list[ToolCfg]] = {}
    for t in cfg.tools:
        if t.mode != "off":
            by_sig.setdefault(len(t.slots) + len(t.text_slots), []).append(t)
    for group in by_sig.values():
        for t in group:
            if len(group) > 1 and not t.cues:
                others = ", ".join(o.name for o in group if o is not t)
                issues.append(Issue(
                    "warning", t.name,
                    f"takes the same number of parameters as {others} and has no 'cues'; embeddings can "
                    "confuse tools that differ only by operation (this caused 20 wrong hits in 1500 in our "
                    "benchmark). Add cues, or confirm with `aicache eval`."))
    seen: dict[str, str] = {}
    for t in cfg.tools:
        for ex in t.examples:
            key = ex.strip().lower()
            if key in seen and seen[key] != t.name:
                issues.append(Issue("warning", t.name, f"example {ex!r} also appears in {seen[key]}"))
            seen.setdefault(key, t.name)
    return issues
