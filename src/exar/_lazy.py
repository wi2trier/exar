"""Lazy exports for the names that need an optional dependency.

A module declares its lazy exports as plain `from .module import name` statements
in an `if TYPE_CHECKING:` block and calls `lazy_exports(__name__)` as its last statement.
Type checkers and pdoc read the block as ordinary imports,
while at runtime the module imports each name on first access.
So `import exar` never loads an optional dependency,
and a missing extra surfaces as an `AttributeError` that names it,
caused by a `exar.errors.MissingExtra`.

An exported name must not be the name of a submodule of its package,
because PEP 810 resolves `lazy from package import name` to the submodule `package.name` if one exists.

Migration to PEP 810 once exar requires Python 3.15:
replace each `if TYPE_CHECKING:` block and its `lazy_exports(__name__)` call
with the same statements prefixed by `lazy`, then delete `lazy_exports` and `_LazyModule`.
"""

import ast
import inspect
import os
import pkgutil
import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
from functools import cache
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

from .errors import MissingExtra

__all__ = ["EXTRAS", "extra_hint", "lazy_exports", "require"]

EXTRAS: Mapping[str, str] = {
    "a2a": "a2a",
    "anydoc": "anydoc",
    "cbrkit": "cbrkit",
    "chonkie": "chonkie",
    "claude_agent_sdk": "claude-agent-sdk",
    "cv2": "media",
    "docling": "docling",
    "duckdb": "table",
    "fastexcel": "table",
    "fastmcp": "mcp",
    "httpx2": "http",
    "kreuzberg": "kreuzberg",
    "markdownify": "web",
    "markitdown": "markitdown",
    "pdf_inspector": "pdf-inspector",
    "pdf_oxide": "pdf-oxide",
    "PIL": "ingest",
    "polars": "table",
    "pydantic_ai": "pydantic-ai",
    "pydantic_graph": "bpmn",
    "pydantic_evals": "eval",
    "pypdfium2": "media",
    "simpleeval": "bpmn",
    "sqlalchemy": "sql",
    "trafilatura": "web",
}
"""Top-level import name of an optional dependency mapped to the extra that installs it."""


@contextmanager
def extra_hint(feature: str, extra: str | None = None) -> Generator[None]:
    """Name the extra to install when an optional dependency is missing.

    Args:
        feature: The name that needs the dependency.
        extra: The extra that installs it, looked up by the import name of the dependency if None.
    """
    try:
        yield
    except MissingExtra:
        raise
    except ModuleNotFoundError as e:
        root = (e.name or "").partition(".")[0]

        if root == "exar" or (extra := extra or EXTRAS.get(root)) is None:
            raise

        raise _missing(feature, root, extra, e.name) from e


def require(feature: str, extra: str) -> None:
    """Checks that the dependencies `EXTRAS` maps to an extra are installed, without importing them.

    Args:
        feature: The name that needs the extra.
        extra: The extra.

    Raises:
        exar.errors.MissingExtra: If a dependency of the extra is missing.
    """
    for root, name in EXTRAS.items():
        if name == extra and not _installed(root):
            raise _missing(feature, root, extra, root)


@cache
def _installed(root: str) -> bool:
    """Whether a top-level module can be imported."""
    return find_spec(root) is not None


def _missing(feature: str, root: str, extra: str, name: str | None) -> MissingExtra:
    """The error of a feature whose dependency of an extra is missing."""
    return MissingExtra(
        f"{feature} requires {root!r}, install it with `pip install 'exar[{extra}]'`",
        name=name,
    )


@cache
def _exports(module_name: str) -> Mapping[str, tuple[str, str]]:
    """Map each lazy export of a module to its source module and attribute."""
    module = sys.modules[module_name]
    exports: dict[str, tuple[str, str]] = {}

    try:
        source = inspect.getsource(module)
    except OSError as e:
        raise ImportError(
            f"{module_name}: lazy exports are read from the module source, which is missing"
        ) from e

    for stmt in ast.parse(source).body:
        if not (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "TYPE_CHECKING"
        ):
            continue

        for node in stmt.body:
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module
                and all(alias.name != "*" for alias in node.names)
            ):
                raise ImportError(
                    f"{module_name}: lazy exports must be `from module import name` statements"
                )

            for alias in node.names:
                exports[alias.asname or alias.name] = (
                    "." * node.level + node.module,
                    alias.name,
                )

    if hasattr(module, "__path__") and (
        clashes := exports.keys()
        & {info.name for info in pkgutil.iter_modules(module.__path__)}
    ):
        raise ImportError(f"{module_name}: lazy exports shadow submodules {clashes}")

    return exports


class _LazyModule(ModuleType):
    """Module whose lazy exports are imported on first access."""

    def __getattr__(self, name: str) -> object:
        if name.startswith("__") or name not in (exports := _exports(self.__name__)):
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {name!r}",
                name=name,
                obj=self,
            )

        source, attr = exports[name]

        try:
            with extra_hint(f"{self.__name__}.{name}"):
                module = import_module(source, self.__package__)
        except MissingExtra as e:
            raise AttributeError(str(e), name=name, obj=self) from e

        value = getattr(module, attr)
        setattr(self, name, value)

        return value

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *_exports(self.__name__)})


def lazy_exports(module_name: str) -> None:
    """Import the names of a module's `if TYPE_CHECKING:` block on first access.

    Setting the environment variable `EXAR_EAGER_IMPORTS` imports them right away,
    skipping those whose extra is missing, which tools like pdoc need.

    Args:
        module_name: `__name__` of the calling module.
    """
    module = sys.modules[module_name]
    module.__class__ = _LazyModule

    if os.environ.get("EXAR_EAGER_IMPORTS"):
        for name in _exports(module_name):
            with suppress(AttributeError):
                getattr(module, name)
