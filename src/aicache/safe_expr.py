"""A small, safe arithmetic expression language for config-defined tools.

``compute: "base * pct / 100"`` in a config file is parsed here. It never
calls ``eval``. Only arithmetic, comparisons, conditionals, a short list of math
functions and the declared parameter names are allowed. Attribute access,
subscripts, comprehensions, lambdas, imports and arbitrary calls are rejected
when the expression is compiled, so a bad config fails at load time and not on
a live request.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Iterable

MAX_LENGTH = 500
MAX_NODES = 200
MAX_POWER = 1000  # cap for the exponent in x ** y, to avoid huge integers

_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_CMP = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "exp": math.exp,
}
CONSTANTS = {"pi": math.pi, "e": math.e, "True": True, "False": False}


class ExprError(ValueError):
    """The expression is not allowed or cannot be evaluated."""


def _check(node: ast.AST, names: set[str]) -> None:
    for child in ast.walk(node):
        if isinstance(child, (ast.Expression, ast.Load, ast.operator, ast.unaryop,
                              ast.cmpop, ast.boolop)):
            continue
        if isinstance(child, ast.BinOp):
            if not isinstance(child.op, tuple(_BINOPS) + (ast.Pow,)):
                raise ExprError(f"operator {type(child.op).__name__} is not allowed")
        elif isinstance(child, ast.UnaryOp):
            if type(child.op) not in _UNARY:
                raise ExprError("unary operator is not allowed")
        elif isinstance(child, ast.Compare):
            if any(type(o) not in _CMP for o in child.ops):
                raise ExprError("comparison operator is not allowed")
        elif isinstance(child, (ast.BoolOp, ast.IfExp)):
            continue
        elif isinstance(child, ast.Constant):
            if not isinstance(child.value, (int, float, bool)):
                raise ExprError("only numeric and boolean constants are allowed")
        elif isinstance(child, ast.Name):
            if child.id not in names and child.id not in CONSTANTS and child.id not in FUNCTIONS:
                raise ExprError(f"unknown name {child.id!r}")
        elif isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in FUNCTIONS:
                raise ExprError("only these functions are allowed: " + ", ".join(sorted(FUNCTIONS)))
            if child.keywords:
                raise ExprError("keyword arguments are not allowed")
        else:
            raise ExprError(f"{type(child).__name__} is not allowed in an expression")


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ExprError(f"{node.id!r} is a function, not a value")
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_POWER:
                raise ExprError("exponent too large")
            return left ** right
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.BoolOp):
        vals = (_eval(v, env) for v in node.values)
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)
    if isinstance(node, ast.Call):
        fn = FUNCTIONS[node.func.id]  # type: ignore[attr-defined]
        return fn(*[_eval(a, env) for a in node.args])
    raise ExprError(f"{type(node).__name__} is not allowed")  # pragma: no cover


def compile_expr(source: str, names: Iterable[str]) -> Callable[..., Any]:
    """Compile ``source``. The result takes the named parameters as keyword
    arguments. Raises ``ExprError`` at compile time if the expression is not
    allowed or uses a name that is not in ``names``."""
    if len(source) > MAX_LENGTH:
        raise ExprError(f"expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"syntax error: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExprError("expression is too complex")
    allowed = set(names)
    _check(tree, allowed)

    def run(**kwargs: Any) -> Any:
        missing = allowed - set(kwargs)
        if missing:
            raise ExprError(f"missing values for: {', '.join(sorted(missing))}")
        return _eval(tree, kwargs)

    run.__doc__ = source
    return run


def free_names(source: str) -> set[str]:
    """Names an expression refers to, excluding constants and functions."""
    tree = ast.parse(source.strip(), mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(CONSTANTS) - set(FUNCTIONS)
