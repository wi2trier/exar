"""Conversions between the changes and errors of exar and those of cbrkit."""

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager

import cbrkit.errors
import cbrkit.model

from ..errors import Conflict, Refused
from ..typing import Add, Changes, Remove, Update

__all__ = ["from_changes", "to_changes", "translated"]


def to_changes[K, V](
    changes: Changes[K, V], /, expected: Mapping[K, V] | None = None
) -> cbrkit.model.Changes[K, V]:
    """The changes of a write to a memory as cbrkit changes of an application, without a rule.

    Examples:
        >>> to_changes(Changes[int, str](items=[Add(key=1, value="a", reason="new")])).items[0]
        Add(rule=None, reason='new', score=None, related=(), op='add', key=1, value='a')
    """
    items: list[
        cbrkit.model.Add[K, V] | cbrkit.model.Update[K, V] | cbrkit.model.Remove[K]
    ] = []

    for item in changes.items:
        match item:
            case Add(key=key, value=value, reason=reason):
                items.append(cbrkit.model.Add(key=key, value=value, reason=reason))
            case Update(key=key, value=value, reason=reason):
                items.append(cbrkit.model.Update(key=key, value=value, reason=reason))
            case Remove(key=key, reason=reason):
                items.append(cbrkit.model.Remove(key=key, reason=reason))

    return cbrkit.model.Changes.model_construct(
        items=items, applied=changes.applied, expected=expected
    )


def from_changes[K, V, W](
    changes: cbrkit.model.Changes[K, V], value: Callable[[V], W], /
) -> Changes[K, W]:
    """cbrkit changes as the changes of a memory, keeping their reasons, with each value converted.

    Examples:
        >>> from_changes(cbrkit.model.Changes(items=[cbrkit.model.Add(key=1, value="a", reason="new")]), str.upper)
        Changes(items=[Add(op='add', key=1, value='A', reason='new')], applied=False)
    """
    items: list[Add[K, W] | Update[K, W] | Remove[K]] = []

    for item in changes.items:
        match item:
            case cbrkit.model.Add():
                items.append(
                    Add(key=item.key, value=value(item.value), reason=item.reason)
                )
            case cbrkit.model.Update():
                items.append(
                    Update(key=item.key, value=value(item.value), reason=item.reason)
                )
            case cbrkit.model.Remove():
                items.append(Remove(key=item.key, reason=item.reason))

    return Changes(items=items, applied=changes.applied)


@contextmanager
def translated() -> Generator[None]:
    """Raises the conflicts and refusals of cbrkit as those of exar, so hosts catch one hierarchy.

    Examples:
        >>> with translated():
        ...     raise cbrkit.errors.Conflict("changed")
        Traceback (most recent call last):
        ...
        exar.errors.Conflict: changed
    """
    try:
        yield
    except cbrkit.errors.Conflict as error:
        raise Conflict(str(error)) from error
    except cbrkit.errors.Refused as error:
        raise Refused(str(error)) from error
