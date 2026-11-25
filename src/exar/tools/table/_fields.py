"""The configuration of `Table`, which its queries read without importing the public class."""

from abc import ABC
from dataclasses import dataclass

from ...memory.mounts import Mounts
from ...schema import PythonOnly
from .. import BaseTool
from ..common import MAX_DOCUMENT_CHARS, Hint, no_hint

__all__ = ["TableFields"]


@dataclass(slots=True, frozen=True, kw_only=True)
class TableFields(BaseTool, ABC):
    """The configuration of `exar.tools.table.Table`, which its queries read."""

    mounts: Mounts
    preview_rows: int = 5
    max_rows: int = 100_000
    max_cell_chars: int = 200
    max_named_columns: int = 10
    max_document_chars: int = MAX_DOCUMENT_CHARS
    timeout_seconds: float = 30.0
    threads: int = 4
    memory_limit: str = "1GB"
    hint: PythonOnly[Hint] = no_hint
