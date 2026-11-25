"""The configuration of `WebFetch` and `WebSearch`, which their calls read without importing the public classes."""

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass

from ...schema import PythonOnly
from .. import BaseTool
from .._host import HostClient
from ..extract import HtmlExtractor
from .backends import AnySearchBackend
from .session import HostSession

__all__ = ["WebFetchFields", "WebSearchFields"]


def _bytes(size: int) -> str:
    return f"{size} bytes"


@dataclass(slots=True, frozen=True, kw_only=True)
class WebFetchFields(BaseTool, ABC):
    """The configuration of `exar.tools.web.WebFetch`, which its calls read."""

    client: HostClient
    session: HostSession
    timeout_seconds: float = 20.0
    max_response_bytes: int = 5_000_000
    max_chars: int = 40_000
    max_line_chars: int = 2000
    user_agent: str | None = None
    fast: bool = False
    html: PythonOnly[HtmlExtractor | None] = None
    size: PythonOnly[Callable[[int], str]] = _bytes


@dataclass(slots=True, frozen=True, kw_only=True)
class WebSearchFields(BaseTool, ABC):
    """The configuration of `exar.tools.web.WebSearch`, which its calls read."""

    client: HostClient
    session: HostSession
    backend: AnySearchBackend
    timeout_seconds: float = 20.0
    max_response_bytes: int = 1_000_000
