"""What tools and agents declare that calling them changes, below `exar.agents` so the tools can build actions with its depth.

The functions are public as part of `exar.tools`.
"""

import typing
from collections.abc import Callable

from .typing import Effect

__all__ = ["EFFECTS", "declared", "effect", "effect_of"]

EFFECTS: tuple[Effect, ...] = typing.get_args(Effect.__value__)


def effect[F: Callable[..., object]](effect: Effect, /) -> Callable[[F], F]:
    """Declares what calling a tool changes, read by policies through `effect_of`.

    Examples:
        >>> @effect("write")
        ... async def save(text: str) -> str:
        ...     return text
        >>> effect_of(save)
        'write'
    """

    def decorate(func: F) -> F:
        vars(func)["effect"] = effect

        return func

    return decorate


def effect_of(obj: object, /) -> Effect:
    """What calling a tool or invoking an agent changes, failing closed to `"execute"` if undeclared.

    The effect is the `effect` attribute of the object, as `effect` declares it on functions.
    Anything else may change the environment, so policies treat it as `"execute"`.

    Raises:
        ValueError: If the declared effect is unknown.

    Examples:
        >>> effect_of(len)
        'execute'
    """
    return declared(getattr(obj, "effect", "execute"), EFFECTS, "effect", obj)


def declared[T](
    value: object, options: tuple[T, ...], /, kind: str, owner: object
) -> T:
    """The option a value declares, such as the effect of a tool or the operation of an agent.

    Raises:
        ValueError: If the value is none of the options.

    Examples:
        >>> declared("write", EFFECTS, "effect", "save")
        'write'
    """
    for option in options:
        if option == value:
            return option

    raise ValueError(f"Unknown {kind} {value!r} of {owner!r}")
