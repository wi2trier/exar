"""File formats the tools read in place, without the dependencies of the tools that read them."""

__all__ = ["DELIMITERS", "TABULAR_SUFFIXES"]

DELIMITERS: dict[str, str] = {".csv": ",", ".tsv": "\t"}
"""What separates the fields of each delimited format, decided by its suffix alone, never sniffed."""

TABULAR_SUFFIXES: frozenset[str] = frozenset(
    {*DELIMITERS, ".parquet", ".xlsx", ".xlsb", ".xls"}
)
"""Extensions a SQL query can be run against."""
