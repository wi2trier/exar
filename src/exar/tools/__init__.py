"""Conventions for tools as plain typed async callables, served unchanged by any agent framework.

A tool declares what it changes with the `effect` decorator, raises `Retry` for input the model can fix,
and handles lists of items with `run_batch`.
Tools that need configuration are serializable classes deriving from `BaseTool`, whose `__call__` is the tool,
and whose `tool` turns it into the plain function that frameworks register (`as_tool`), with the wording of its schema as data.
A `SchemaTool` is a tool whose input schema is data, such as an operation of an API,
and a configured class that offers several tools, such as `exar.tools.sql.Sql`, lists them with `tools()`.
Framework integrations read every tool as a `SchemaTool` (`schema_tool`), translate `Retry` with `translate_retry`,
ask policies about the `tool_action` of a call, and announce its effect as the MCP annotations of `effect_hints`.
The text helpers that render and fit tool output are in `exar.text`.
"""

import contextlib
import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import UnionType
from typing import Annotated, Any, ClassVar, Literal, TypedDict, get_args, get_origin

from pydantic import Field, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

from .._effect import effect, effect_of
from .._lazy import extra_hint, require
from ..agents import depth
from ..helpers import bounded_gather, frozendict, get_name
from ..text import os_reason
from ..typing import Action, Effect, JsonEntry, Tool

__all__ = [
    "MAX_BATCH_ITEMS",
    "BaseTool",
    "Batch",
    "BatchData",
    "EffectHints",
    "ItemFailure",
    "Retry",
    "SchemaTool",
    "as_tool",
    "effect",
    "effect_hints",
    "effect_of",
    "render_batch",
    "retry_on",
    "run_batch",
    "schema_tool",
    "tool_action",
    "translate_retry",
]


MAX_BATCH_ITEMS = 10
"""Default cap on the items one list argument of a tool may carry."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaTool:
    """A tool whose input schema is data instead of a signature, such as an operation of an API.

    It is a `exar.typing.Tool` itself: calling it calls `call`, `__name__` is its name, and `effect` what it changes.

    Args:
        name: Name of the tool.
        description: What the tool does, for the model.
        schema: JSON schema of the arguments, an object.
        effect: What calling it changes.
        call: The tool, called with the arguments as keywords.
    """

    name: str
    description: str
    schema: Mapping[str, object]
    effect: Effect
    call: Callable[..., Awaitable[object]]

    @property
    def __name__(self) -> str:
        """The name of the tool, as functions have it."""
        return self.name

    async def __call__(self, **arguments: object) -> object:
        """Calls the tool with the arguments as keywords."""
        return await self.call(**arguments)


class Retry(Exception):
    """A tool failure the model can fix by calling the tool again with other input.

    Framework integrations translate it into their own retry signal,
    such as `ModelRetry` in pydantic-ai or a tool error over MCP.
    """


@contextlib.contextmanager
def retry_on(*errors: type[Exception]) -> Generator[None]:
    """Turns errors the model can fix, such as a refused write or an unknown key, into `Retry`.

    Examples:
        >>> with retry_on(KeyError):
        ...     {}["missing"]
        Traceback (most recent call last):
        exar.tools.Retry: 'missing'
    """
    try:
        yield
    except errors as error:
        raise Retry(str(error)) from error


def translate_retry[**P, R](
    func: Callable[P, Awaitable[R]], error: Callable[[str], Exception], /
) -> Callable[P, Awaitable[R]]:
    """Wraps a tool so that `Retry` becomes the retry signal of a framework, such as `pydantic_ai.ModelRetry`.

    Examples:
        >>> import asyncio
        >>> async def fail() -> None:
        ...     raise Retry("try again")
        >>> asyncio.run(translate_retry(fail, RuntimeError)())
        Traceback (most recent call last):
        RuntimeError: try again
    """

    @functools.wraps(func)
    async def call(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)
        except Retry as retry:
            raise error(str(retry)) from retry

    return call


def as_tool[**P, R](
    call: Callable[P, Awaitable[R]],
    /,
    *,
    name: str,
    effect: Effect,
    description: str | None = None,
    arguments: Mapping[str, str] | None = None,
    guidance: str = "",
    notes: str = "",
    max_items: int | None = None,
) -> Callable[P, Awaitable[R]]:
    """Turns a callable, such as the bound `__call__` of a configured tool, into a tool, a plain function with a name, an effect, and optionally other wording.

    The description of the tool is `description`, else the docstring of the call
    with the `guidance` leading its body and the `notes` following it,
    `arguments` replace the descriptions of parameters by their names,
    and `max_items` caps every list parameter.
    The call keeps its signature, so the schema of the tool stays the one of its typed parameters.

    Raises:
        ValueError: If `arguments` names a parameter the call does not have.

    Examples:
        >>> class Echo:
        ...     async def __call__(self, text: str) -> str:
        ...         return text
        >>> tool = as_tool(Echo().__call__, name="say", effect="read", description="Repeats the text.", arguments={"text": "What to say."})
        >>> tool.__name__, tool.__doc__, effect_of(tool)
        ('say', 'Repeats the text.', 'read')
        >>> inspect.signature(tool).parameters["text"].annotation.__metadata__[0].description
        'What to say.'
        >>> class Sum:
        ...     async def __call__(self, numbers: list[int]) -> int:
        ...         return sum(numbers)
        >>> tool = as_tool(Sum().__call__, name="sum", effect="read", max_items=2)
        >>> inspect.signature(tool).parameters["numbers"].annotation.__metadata__[0].metadata
        [MaxLen(max_length=2)]
    """
    signature = inspect.signature(call, eval_str=True)

    if unknown := sorted((arguments or {}).keys() - signature.parameters.keys()):
        raise ValueError(f"{name} has no parameters {', '.join(unknown)}")

    if description is None and (guidance or notes):
        summary, _, body = (inspect.getdoc(call) or "").partition("\n\n")
        lead = f"{guidance}\n" if guidance else ""
        end = f"\n\n{notes}" if notes else ""
        description = f"{summary}\n\n{lead}{body}{end}"

    parameters = [
        parameter.replace(annotation=Annotated[parameter.annotation, *fields])
        if (fields := _fields(parameter, arguments or {}, max_items))
        else parameter
        for parameter in signature.parameters.values()
    ]

    @functools.wraps(call)
    async def tool(*args: P.args, **kwargs: P.kwargs) -> R:
        return await call(*args, **kwargs)

    tool.__name__ = tool.__qualname__ = name
    tool.__doc__ = description or inspect.getdoc(call)
    vars(tool)["__signature__"] = signature.replace(parameters=parameters)
    annotations = {
        **{parameter.name: parameter.annotation for parameter in parameters},
        "return": signature.return_annotation,
    }
    # The annotate function, since `functools.wraps` of other wrappers copies that and not the annotations.
    tool.__annotate__ = lambda _format: annotations  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[missing-attribute]

    # As the decorator `effect` declares it, which the parameter shadows.
    vars(tool)["effect"] = effect

    return tool


def _fields(
    parameter: inspect.Parameter, arguments: Mapping[str, str], max_items: int | None
) -> list[FieldInfo]:
    """The fields `as_tool` adds to the annotation of a parameter, its description and its cap."""
    fields: list[FieldInfo] = []

    if (description := arguments.get(parameter.name)) is not None:
        fields.append(Field(description=description))

    if max_items is not None and _is_list(parameter.annotation):
        fields.append(Field(max_length=max_items))

    return fields


def _is_list(annotation: object) -> bool:
    """Whether a type is a list, also if annotated or optional.

    Examples:
        >>> _is_list(Annotated[list[int] | None, "items"]), _is_list(int | None)
        (True, False)
    """
    if get_origin(annotation) is Annotated:
        return _is_list(get_args(annotation)[0])

    if isinstance(annotation, UnionType):
        return any(_is_list(member) for member in get_args(annotation))

    return get_origin(annotation) is list


def tool_action(
    name: str,
    effect: Effect,
    args: object = None,
    metadata: Mapping[str, JsonEntry] | None = None,
    *,
    target: str | None = None,
) -> Action:
    """The action of calling a tool, which a policy decides on, at the depth of the running agent (`exar.agents.depth`).

    The arguments become its details as JSON, and the metadata is what the tool declares about itself,
    such as the metadata of a pydantic-ai tool, so a policy can decide by more than the effect.
    The target is the key or path the call writes to, such as the key of a memory entry.

    Examples:
        >>> tool_action("mail", "write", {"to": "ada"})
        Action(kind='tool', name='mail', effect='write', target=None, depth=0, details={'to': 'ada'}, metadata={})
    """
    return Action(
        kind="tool",
        name=name,
        effect=effect,
        target=target,
        depth=depth(),
        details=to_jsonable_python(args, fallback=str),
        metadata=metadata or {},
    )


def schema_tool(tool: Tool, /) -> SchemaTool:
    """A tool as a `SchemaTool`, the one form every framework integration serves.

    A `SchemaTool` stays as it is, and a plain function gets the schema and description of its signature and docstring
    as pydantic-ai reads them, its name, and its effect, while its call validates the arguments,
    an invalid one raising `Retry` for the model to fix.
    It needs the `pydantic-ai` extra for a plain function.

    Examples:
        >>> import asyncio
        >>> @effect("read")
        ... async def add(a: int, b: int) -> int:
        ...     '''Adds two numbers.'''
        ...     return a + b
        >>> served = schema_tool(add)
        >>> served.name, served.description, served.effect, sorted(served.schema["properties"])
        ('add', 'Adds two numbers.', 'read', ['a', 'b'])
        >>> asyncio.run(served(a=1, b="2"))
        3
    """
    if isinstance(tool, SchemaTool):
        return tool

    with extra_hint("schema_tool", "pydantic-ai"):
        from pydantic_ai import Tool as PydanticAITool

    name = get_name(tool)
    function = PydanticAITool(tool, name=name)
    validator = function.function_schema.validator

    async def call(**arguments: object) -> object:
        try:
            validated: dict[str, object] = validator.validate_python(arguments)
        except ValidationError as error:
            raise Retry(str(error)) from error

        return await tool(**validated)

    return SchemaTool(
        name=name,
        description=function.tool_def.description or "",
        schema=function.tool_def.parameters_json_schema,
        effect=effect_of(tool),
        call=call,
    )


class EffectHints(TypedDict):
    """The behavior hints of the MCP tool annotations, by their names in the specification."""

    readOnlyHint: bool
    destructiveHint: bool
    idempotentHint: bool
    openWorldHint: bool


def effect_hints(effect: Effect, /) -> EffectHints:
    """The MCP tool annotations a tool of an effect is announced with.

    Examples:
        >>> effect_hints("write")
        {'readOnlyHint': False, 'destructiveHint': True, 'idempotentHint': False, 'openWorldHint': False}
    """
    reads = effect == "read"

    return {
        "readOnlyHint": reads,
        "destructiveHint": not reads,
        "idempotentHint": reads,
        "openWorldHint": effect == "execute",
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseTool(ABC):
    """The base of a configured tool, a keyword-only dataclass whose `__call__` is the tool and whose wording is data.

    `tool` turns it into a plain function, whose effect is the class variable `effect`, `"read"` unless a class declares another.
    A class whose calls need an extra names it as the class variable `extra`, which `tool` checks before the first call imports it.
    A list argument is served item by item, such as with `exar.tools.batch.serve_batch`, at most `concurrency` at once.

    Args:
        name: Name of the tool.
        description: Description of the tool, that of `__call__` with the guidance and notes if None.
        arguments: Descriptions of the arguments by name, which replace those of `__call__`.
        guidance: What to add to the description before the body of `__call__`, such as when to prefer another tool.
        notes: What to add to the description after it, such as which entries are left out.
        max_items: Most items of a list argument, the cap of its schema.
        concurrency: Most items of a list argument served at once.
    """

    effect: ClassVar[Effect] = "read"
    extra: ClassVar[str | None] = None
    name: str
    description: str | None = None
    arguments: frozendict[str, str] = field(default_factory=frozendict[str, str])
    guidance: str = ""
    notes: str = ""
    max_items: int = MAX_BATCH_ITEMS
    concurrency: int = 4

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> object:
        """The tool, whose signature is its input schema."""

    def tool(self) -> Tool:
        """The tool as a plain function, named by `name` and described by `description`, `guidance`, `notes`, and `arguments`.

        Raises:
            exar.errors.MissingExtra: If the extra of the class is missing.
        """
        if self.extra is not None:
            require(type(self).__name__, self.extra)

        return as_tool(
            self.__call__,
            name=self.name,
            effect=self.effect,
            description=self.description,
            arguments=self.arguments,
            guidance=self.guidance,
            notes=self.notes,
            max_items=self.max_items,
        )

    def tools(self) -> Sequence[Tool]:
        """The one tool, see `tool`, as the classes offering several tools list theirs."""
        return (self.tool(),)


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemFailure:
    """An item of a batch that failed, standing at its position among the results.

    Args:
        item: Key of the item.
        reason: Why it failed, for the model to fix its input.
        kind: Tells a failure from a result when serialized.
    """

    item: str
    reason: str
    kind: Literal["failure"] = "failure"


type Batch[R] = Mapping[str, R | ItemFailure]
"""Results or failures by item key, in request order."""

type BatchData[R] = tuple[R | ItemFailure, ...]
"""One entry per distinct requested item, in request order, as the client receives the results of a batch."""


async def run_batch[I, R](
    items: Iterable[I],
    run: Callable[[I], Awaitable[R]],
    /,
    *,
    key: Callable[[I], str] = str,
    concurrency: int = 1,
) -> Batch[R]:
    """Runs a tool on every distinct item, so a list argument fails item by item.

    Items are deduplicated by key, keeping the first, and run at most `concurrency` at a time.
    A `Retry` of an item becomes its `ItemFailure`, and so does an `OSError`, such as a file that vanished meanwhile,
    by its reason alone, which names no host path.
    Only a batch whose items all fail raises.

    Raises:
        Retry: If every item failed.

    Examples:
        >>> import asyncio
        >>> async def double(item: int) -> int:
        ...     if item < 0:
        ...         raise Retry("negative")
        ...     return 2 * item
        >>> asyncio.run(run_batch([1, -1, 1], double))
        {'1': 2, '-1': ItemFailure(item='-1', reason='negative', kind='failure')}
    """
    unique: dict[str, I] = {}

    for item in items:
        unique.setdefault(key(item), item)

    async def attempt(entry: tuple[str, I]) -> R | ItemFailure:
        name, item = entry

        try:
            return await run(item)
        except Retry as error:
            return ItemFailure(item=name, reason=str(error))
        except OSError as error:
            return ItemFailure(item=name, reason=os_reason(error))

    outcomes = await bounded_gather(unique.items(), attempt, limit=concurrency)
    failures = [outcome for outcome in outcomes if isinstance(outcome, ItemFailure)]

    if failures and len(failures) == len(outcomes):
        reasons = [f"{failure.item}: {failure.reason}" for failure in failures]

        raise Retry(
            reasons[0]
            if len(reasons) == 1
            else "Every item failed:\n" + "\n".join(f"- {reason}" for reason in reasons)
        )

    return dict(zip(unique, outcomes, strict=True))


def render_batch[R](batch: Batch[R], /, render: Callable[[R], str] = str) -> str:
    """Renders the results of a batch side by side under a header per item.

    Examples:
        >>> print(render_batch({"a": 1, "b": ItemFailure(item="b", reason="missing")}))
        ==> a <==
        1
        <BLANKLINE>
        ==> b <==
        failed: missing
    """
    return "\n\n".join(
        f"==> {name} <==\nfailed: {outcome.reason}"
        if isinstance(outcome, ItemFailure)
        else f"==> {name} <==\n{render(outcome)}"
        for name, outcome in batch.items()
    )
