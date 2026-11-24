"""Serialized forms and JSON Schemas derived from classes and their docstrings.

Descriptions come from Google-style docstrings,
the summary of a class and the `Args` section for its fields,
so the code stays the only source of the schemas offered to users and language models.
Configurable classes are keyword-only dataclasses made `serializable`, or pydantic models deriving from `Model`.
A field of configurable kinds is `Open` to any implementation of a protocol in Python code,
and a field that only Python code sets is `PythonOnly`.
"""

import inspect
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SkipValidation,
    Tag,
    with_config,
)
from pydantic.json_schema import SkipJsonSchema

__all__ = ["Model", "Open", "PythonOnly", "describe", "serializable"]

T = TypeVar("T")

# a classic alias, since pydantic ignores field metadata such as `exclude` in `type` aliases
PythonOnly = Annotated[SkipValidation[SkipJsonSchema[T]], Field(exclude=True)]
"""A field that only Python code sets, such as a function, left out of serialized forms."""


def _open_tag(value: object) -> Literal["kind", "custom"]:
    kind = (
        value.get("kind")
        if isinstance(value, Mapping)
        else getattr(value, "kind", None)
    )

    return "custom" if kind is None else "kind"


type Open[K, P] = Annotated[
    Annotated[K, Tag("kind")]
    | Annotated[SkipJsonSchema[SkipValidation[P]], Tag("custom")],
    Discriminator(_open_tag),
]
"""The configurable kinds `K`, which name their `kind`, or any other implementation of the protocol `P`, which serialized forms and JSON Schemas only know as `K`."""


def _parse(doc: str) -> tuple[str, dict[str, str]]:
    """The description of a Google-style docstring and the descriptions of its arguments, continuation lines joined."""
    # griffe is imported on first use, since only schema generation needs it.
    from griffe import Docstring, DocstringSectionParameters, DocstringSectionText

    sections = Docstring(doc).parse("google", warnings=False)
    description = "\n\n".join(
        section.value
        for section in sections
        if isinstance(section, DocstringSectionText)
    )
    args = {
        param.name: " ".join(
            filter(None, map(str.strip, param.description.splitlines()))
        )
        for section in sections
        if isinstance(section, DocstringSectionParameters)
        for param in section.value
    }

    return description, args


def describe(schema: dict[str, Any], cls: type[Any]) -> None:
    """Fills the descriptions of a JSON Schema from the Google-style docstrings of a class.

    Arguments that a base class documents are inherited, so the fields of a base class are documented once.

    Examples:
        >>> class Base:
        ...     '''Base.
        ...
        ...     Args:
        ...         a: First.
        ...     '''
        >>> class Child(Base):
        ...     '''Child.
        ...
        ...     Args:
        ...         b: Second,
        ...             continued.
        ...     '''
        >>> schema = {"properties": {"a": {}, "b": {}}}
        >>> describe(schema, Child)
        >>> schema
        {'properties': {'a': {'description': 'First.'}, 'b': {'description': 'Second, continued.'}}, 'description': 'Child.'}
    """
    args: dict[str, str] = {}

    for base in reversed(cls.__mro__):
        if doc := base.__dict__.get("__doc__"):
            args |= _parse(doc)[1]

    if description := _parse(inspect.getdoc(cls) or "")[0]:
        schema["description"] = description

    for name, prop in schema.get("properties", {}).items():
        if name in args:
            prop["description"] = args[name]


class Model(BaseModel):
    """Base of the models of exar: immutable, strict about unknown fields, and described by their docstrings."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra=describe,
        arbitrary_types_allowed=True,
        defer_build=True,
    )


_SERIALIZABLE = ConfigDict(
    extra="forbid", json_schema_extra=describe, arbitrary_types_allowed=True
)


def serializable[C: type[Any]](cls: C) -> C:
    """Lets pydantic validate and serialize a keyword-only dataclass, described by its docstring.

    Unknown fields are rejected, and the JSON Schema takes its descriptions from the Google-style docstring.

    Raises:
        TypeError: If the class is no dataclass with keyword-only parameters.
    """
    dataclass_fields = getattr(cls, "__dataclass_fields__", None)

    if dataclass_fields is None:
        raise TypeError(f"{cls.__name__} is no dataclass")

    if positional := [
        name for name, f in dataclass_fields.items() if f.init and not f.kw_only
    ]:
        raise TypeError(
            f"{cls.__name__} takes positional parameters {positional}, declare it with kw_only=True"
        )

    return with_config(_SERIALIZABLE)(cls)
