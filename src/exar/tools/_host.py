"""Functions of the host that serialized forms reference by name, which validation binds.

`HostFactory` is one that returns what the host owns, such as its HTTP client (`HostClient`) or database engine (`HostEngine`),
which the tools check when they run, so this module imports neither HTTPX nor SQLAlchemy.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Self

from pydantic import GetCoreSchemaHandler, ValidationInfo
from pydantic_core import CoreSchema, core_schema

from ..errors import Unbound

__all__ = [
    "Client",
    "Engine",
    "HostClient",
    "HostEngine",
    "HostFactory",
    "HostFunction",
]


@dataclass(slots=True, frozen=True)
class HostFunction[F]:
    """A function of the host that serialized forms reference by its name only.

    Validating a name binds the function of the same name from the mapping that the pydantic validation context
    holds under the class, such as `{HostEngine: engines}`, which `exar.spec.load` provides from its arguments.
    Subclasses such as `HostFactory` add how the function is called.

    Args:
        name: Name of the function, the key of its mapping.
        func: The bound function, None until the host binds it.
    """

    name: str
    func: F | None = field(default=None, compare=False)

    def bound(self) -> F:
        """The bound function.

        Raises:
            exar.errors.Unbound: If no function is bound to the name.
        """
        if self.func is None:
            raise Unbound(
                f"{type(self).__name__} {self.name!r} is not bound to a function"
            )

        return self.func

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def bind(value: Self | str, info: ValidationInfo) -> Self:
            ref = cls(value) if isinstance(value, str) else value
            funcs: Mapping[str, F] = (info.context or {}).get(cls, {})

            if ref.func is None and ref.name in funcs:
                return replace(ref, func=funcs[ref.name])

            return ref

        name = core_schema.str_schema(min_length=1)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.with_info_after_validator_function(bind, name),
            python_schema=core_schema.with_info_after_validator_function(
                bind,
                core_schema.union_schema([core_schema.is_instance_schema(cls), name]),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda ref: ref.name
            ),
            ref=cls.__name__,
        )


@dataclass(slots=True, frozen=True)
class HostFactory[T](HostFunction[Callable[[], T]]):
    """A function of the host that returns what the host owns, which a tool uses and never closes."""

    def __call__(self) -> T:
        """What the bound function returns.

        Raises:
            exar.errors.Unbound: If no function is bound to the name.
        """
        return self.bound()()


class Client(Protocol):
    """What the host's HTTP client offers without importing HTTPX, which `httpx2.AsyncClient` satisfies.

    The tools check it is one when they run.
    """

    async def aclose(self) -> None:
        """Closes the connections."""


@dataclass(slots=True, frozen=True)
class HostClient(HostFactory[Client]):
    """The HTTP client of the host, serialized by its name only.

    The URL policy, proxies, and redirect limit of a fetch are the client's, since HTTPX configures them there.
    """


class Engine(Protocol):
    """What the host's database engine offers without importing SQLAlchemy, which `sqlalchemy.Engine` satisfies.

    The tools check it is one when they run.
    """

    def dispose(self) -> None:
        """Closes the pooled connections."""


@dataclass(slots=True, frozen=True)
class HostEngine(HostFactory[Engine]):
    """The SQLAlchemy engine of the host, serialized by its name only."""
