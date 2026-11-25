"""A text with metadata, the document of a `FileMemory`."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

import yaml

from ..helpers import frozendict
from ..typing import JsonEntry

__all__ = ["Document"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Document:
    """A text with metadata, stored as a file with the metadata as YAML frontmatter, whose string is its text.

    Conventional metadata are `confidence` from 0 to 1 and the provenance:
    `source`, `derived_from`, `generated_by`, and `verified_by`.

    Args:
        text: The content without frontmatter.
        meta: The metadata.

    Examples:
        >>> document = Document.parse("---\\nconfidence: 0.8\\n---\\nThe sky is blue.\\n")
        >>> document.text, dict(document.meta), str(document) == document.text
        ('The sky is blue.\\n', {'confidence': 0.8}, True)
        >>> Document.parse(document.dump()) == document
        True
        >>> Document.parse('---\\n{"source": "web"}\\n---\\nJSON works too').meta["source"]
        'web'
    """

    text: str
    meta: Mapping[str, JsonEntry] = field(default_factory=frozendict[str, JsonEntry])

    def __str__(self) -> str:
        """The text, which measures and indexes score, such as the default `text` of an `InMemoryMemory`."""
        return self.text

    @classmethod
    def parse(cls, content: str, /) -> Self:
        """Reads a document from text with optional YAML or JSON frontmatter.

        Text whose frontmatter is no valid mapping is kept whole as the text.
        """
        lines = content.splitlines(keepends=True)
        end = next(
            (i for i, line in enumerate(lines[1:], 1) if line.rstrip() == "---"), None
        )

        if not lines or lines[0].rstrip() != "---" or end is None:
            return cls(text=content)

        try:
            meta = yaml.safe_load("".join(lines[1:end])) or {}
        except yaml.YAMLError:
            return cls(text=content)

        if not isinstance(meta, Mapping):
            return cls(text=content)

        return cls(text="".join(lines[end + 1 :]), meta=frozendict(meta))

    def dump(self) -> str:
        """Writes the document as text with its metadata as YAML frontmatter, if any."""
        if not self.meta:
            return self.text

        header = yaml.safe_dump(dict(self.meta), sort_keys=False, allow_unicode=True)

        return f"---\n{header}---\n{self.text}"
