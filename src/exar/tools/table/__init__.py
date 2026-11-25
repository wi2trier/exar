"""A tool that queries tables with SQL, configured as a serializable class.

`Table` loads spreadsheets, delimited files, and Parquet files below mounts as Polars frames and runs DuckDB SQL over them,
so a large table costs a small result.
It types what a file stores as text, reports what it could not type, and guesses the header rows of a sheet that carries a title.
It needs the `table` extra.
"""

from ..formats import DELIMITERS, TABULAR_SUFFIXES
from .result import HeaderGuess, QueriedTable, TableResult, TextColumn
from .tool import (
    Table,
    TableFilePathsArg,
    TableLimitArg,
    TableQueriesArg,
    TableSheetArg,
    TableSkipAboveHeaderArg,
    TableSkipBelowHeaderArg,
)

__all__ = [
    "DELIMITERS",
    "TABULAR_SUFFIXES",
    "HeaderGuess",
    "QueriedTable",
    "Table",
    "TableFilePathsArg",
    "TableLimitArg",
    "TableQueriesArg",
    "TableResult",
    "TableSheetArg",
    "TableSkipAboveHeaderArg",
    "TableSkipBelowHeaderArg",
    "TextColumn",
]
