"""Conditions of sequence flows, evaluated by simpleeval over the fields of the state."""

import dataclasses
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import override

from pydantic import BaseModel
from simpleeval import (
    BASIC_ALLOWED_ATTRS,
    DEFAULT_FUNCTIONS,
    InvalidExpression,
    SimpleEval,
)

__all__ = ["condition", "fields"]


def fields(kind: type, /) -> tuple[str, ...] | None:
    """The fields of a pydantic model or dataclass type, None for any other type."""
    if issubclass(kind, BaseModel):
        return tuple(kind.model_fields)

    if dataclasses.is_dataclass(kind):
        return tuple(f.name for f in dataclasses.fields(kind))

    return None


class _Attributes(Mapping[type | None, Collection[str]]):
    """The `allowed_attrs` of simpleeval: its basic attributes and the fields of models and dataclasses."""

    @override
    def __getitem__(self, kind: type | None, /) -> Collection[str]:
        if kind is not None and (names := fields(kind)) is not None:
            return names

        return BASIC_ALLOWED_ATTRS[kind]

    @override
    def __iter__(self) -> Iterator[type | None]:
        return iter(BASIC_ALLOWED_ATTRS)

    @override
    def __len__(self) -> int:
        return len(BASIC_ALLOWED_ATTRS)


_ATTRIBUTES = _Attributes()


def condition(
    text: str, functions: Mapping[str, Callable[..., object]], /
) -> Callable[[object], bool]:
    """Parses a condition into a predicate over a state, a pydantic model or dataclass.

    simpleeval evaluates the condition with the fields of the state as names,
    its default functions and `functions`, and access to the basic attributes of builtins
    and the fields of nested models and dataclasses.

    Raises:
        ValueError: If the text is no expression or a function is forbidden.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass(frozen=True)
        ... class Review:
        ...     score: float
        ...     status: str
        >>> holds = condition("score >= 0.7 and short(status)", {"short": lambda text: len(text) < 5})
        >>> holds(Review(score=0.9, status="done")), holds(Review(score=0.9, status="pending"))
        (True, False)
        >>> condition("score >=", {})
        Traceback (most recent call last):
        ValueError: Invalid condition 'score >='
    """
    functions = {**DEFAULT_FUNCTIONS, **functions}

    try:
        tree = SimpleEval.parse(text)
        _ = SimpleEval(functions=functions)
    except (SyntaxError, InvalidExpression) as error:
        raise ValueError(f"Invalid condition {text!r}") from error

    def holds(state: object) -> bool:
        names = {name: getattr(state, name) for name in fields(type(state)) or ()}
        evaluator = SimpleEval(
            functions=functions, names=names, allowed_attrs=_ATTRIBUTES
        )

        return bool(evaluator.eval(text, previously_parsed=tree))

    return holds
