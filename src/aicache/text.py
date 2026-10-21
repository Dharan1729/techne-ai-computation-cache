"""Text handling for learned actions.

A *template* is a request with its variable values replaced by named
placeholders: ``What is {pct}% of {base}?``. Three operations matter:

* ``mask_numbers``: replace every number in a request with ``#``. A request and
  a template reduce to the same shape (``What is #% of #?``), so they can be
  compared by embedding without the values getting in the way. This is the
  cheap, deterministic equivalent of "remove the variable values".
* ``align``: check that a template really is the request with its numbers
  abstracted, and read the values out. A model's template is never trusted
  until it reproduces the request exactly.

A learned template answers a later request only if that request *is* the
template with different numbers (``align``). Where numbers sit is never
inferred from surrounding words: inferred patterns turned out too generic and
bound requests for other operations.
"""

from __future__ import annotations

import re

NUMBER = r"\$?\d[\d,]*(?:\.\d+)?"
_NUMBER_RE = re.compile(NUMBER)
_PLACEHOLDER_RE = re.compile(r"\$?\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
_WORD_RE = re.compile(r"[A-Za-z]+|[%+*/=:×-]")

MAX_TEMPLATE_CHARS = 400


def mask_numbers(text: str) -> str:
    """``"What is 15% of 240?"`` -> ``"What is #% of #?"``"""
    return _NUMBER_RE.sub("#", text)


def template_params(template: str) -> list[str]:
    """Placeholder names in the order they appear."""
    return [m.group("name") for m in _PLACEHOLDER_RE.finditer(template)]


def mask_template(template: str) -> str:
    """The template with each placeholder (and a ``$`` in front of it) as ``#``,
    so it matches what ``mask_numbers`` makes of a real request."""
    return _PLACEHOLDER_RE.sub("#", template)


def check_template(template: str) -> str | None:
    """None if the template is usable, else the reason it is not."""
    if not template or len(template) > MAX_TEMPLATE_CHARS:
        return "template_size"
    names = template_params(template)
    if not names:
        return "template_no_placeholders"
    if len(set(names)) != len(names):
        return "template_repeated_placeholder"
    if _NUMBER_RE.search(_PLACEHOLDER_RE.sub("", template)):
        return "template_has_literal_number"  # a value was left in the template
    return None


def _literal_regex(literal: str) -> str:
    """Regex for a literal run of template text. Whitespace, including at the
    edges, matches one or more spaces; everything else must match exactly."""
    if not literal:
        return ""
    words = literal.split()
    if not words:
        return r"\s+"
    body = r"\s+".join(re.escape(w) for w in words)
    return (r"\s+" if literal[0].isspace() else "") + body + (r"\s+" if literal[-1].isspace() else "")


def align(template: str, request: str) -> dict[str, float] | None:
    """If ``template`` is ``request`` with its numbers abstracted, return the
    values by placeholder name, else None. Case is ignored, and so are
    trailing ``? . !``."""
    if check_template(template) is not None:
        return None
    parts: list[str] = []
    pos = 0
    for m in _PLACEHOLDER_RE.finditer(template):
        parts.append(_literal_regex(template[pos:m.start()]))
        parts.append(rf"\$?(?P<{m.group('name')}>\d[\d,]*(?:\.\d+)?)")
        pos = m.end()
    parts.append(_literal_regex(template[pos:].rstrip(" ?.!")))
    pattern = re.compile(r"\s*" + "".join(parts) + r"[\s?.!]*", re.IGNORECASE)
    m = pattern.fullmatch(request.strip())
    if not m:
        return None
    try:
        return {k: float(v.replace(",", "")) for k, v in m.groupdict().items()}
    except ValueError:  # pragma: no cover - the number pattern prevents this
        return None
