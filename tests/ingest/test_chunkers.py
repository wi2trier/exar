import pytest

from exar.ingest import (
    Chunker,
    MarkdownChunker,
    RecursiveChunker,
    TableChunker,
    TokenChunker,
    Whole,
)

TEXT = (
    "# Title\n\nFirst paragraph with some words.\n\n```py\nprint(1)\n```\n\n"
    "| a | b |\n|---|---|\n| 1 | 2 |\n\nSecond paragraph, longer than the first one.\n"
)


def test_whole() -> None:
    assert Whole()("") == []
    (chunk,) = Whole()(TEXT)
    assert (chunk.start, chunk.end, chunk.start_line, chunk.end_line) == (
        0,
        len(TEXT),
        1,
        13,
    )


@pytest.mark.parametrize(
    "chunker",
    [
        TokenChunker(size=32),
        RecursiveChunker(size=40, min_characters=1),
        MarkdownChunker(size=64),
    ],
)
def test_chonkie_spans_and_lines(chunker: Chunker) -> None:
    pytest.importorskip("chonkie")
    chunks = chunker(TEXT)

    assert len(chunks) > 1

    for chunk in chunks:
        assert TEXT[chunk.start : chunk.end] == chunk.text
        assert chunk.start_line == TEXT.count("\n", 0, chunk.start) + 1
        assert chunk.end_line == TEXT.count("\n", 0, chunk.end - 1) + 1
        assert chunk.tokens > 0


TABLE = "| a | b |\n|---|---|\n" + "".join(f"| {i} | {i} |\n" for i in range(5))


def test_table() -> None:
    pytest.importorskip("chonkie")
    header = "| a | b |\n|---|---|\n"
    table = TABLE
    chunks = TableChunker(size=2)(table)

    assert len(chunks) == 3

    for chunk in chunks:
        assert chunk.text.startswith(header)
        assert chunk.text.removeprefix(header) == table[chunk.start : chunk.end]

    assert TableChunker()("No table") == Whole()("No table")


@pytest.mark.parametrize(
    ("chunker", "plain", "text"),
    [
        (
            RecursiveChunker(size=64, overlap=8, min_characters=1),
            RecursiveChunker(size=64, min_characters=1),
            TEXT,
        ),
        (MarkdownChunker(size=64, overlap=8), MarkdownChunker(size=64), TEXT),
        (TableChunker(size=2, overlap=1), TableChunker(size=2), TABLE),
    ],
)
def test_overlap_appends_the_next_span(
    chunker: Chunker, plain: Chunker, text: str
) -> None:
    pytest.importorskip("chonkie")
    chunks, own_chunks = chunker(text), plain(text)

    assert len(chunks) == len(own_chunks) > 1

    for chunk, own, following in zip(
        chunks, own_chunks, [*own_chunks[1:], None], strict=True
    ):
        assert (chunk.start, chunk.end) == (own.start, own.end)
        assert chunk.text.startswith(own.text)

        if following is not None:
            context = chunk.text.removeprefix(own.text)
            assert context and text[following.start :].startswith(context)


def test_markdown_keeps_images_and_splits_large_blocks() -> None:
    pytest.importorskip("chonkie")
    text = "Intro ![alt](a.png) more.\n\n" + "word " * 100 + "\n\n" + TABLE
    chunks = MarkdownChunker(size=64, rows=2, min_characters=1)(text)

    assert chunks[0].text.startswith("Intro ![alt](a.png) more.")
    assert all(chunk.tokens <= 64 for chunk in chunks)
    assert sum(chunk.text.startswith("| a | b |") for chunk in chunks) == 3


def test_markdown_keeps_sections_that_fit_and_the_front_matter_alone() -> None:
    pytest.importorskip("chonkie")
    text = (
        "---\ntitle: x\n---\n\n# A\n\nIntro.\n\n## B\n\n- one\n- two\n\n## C\n\n"
        + "word " * 30
    )
    chunks = MarkdownChunker(size=64, min_characters=1)(text)

    assert [(chunk.text[:8], chunk.hierarchy) for chunk in chunks] == [
        ("---\ntitl", ()),
        ("# A\n\nInt", ("A",)),
        ("## C\n\nwo", ("A", "C")),
        ("word wor", ("A", "C")),
        ("word wor", ("A", "C")),
    ]
    assert "## B" in chunks[1].text and all(chunk.tokens <= 64 for chunk in chunks)


def test_markdown_context_prefixes_missing_headings_and_keeps_spans() -> None:
    pytest.importorskip("chonkie")
    text = "# A\n\n## B\n\n" + "word " * 120 + "\n\n" + TABLE
    chunks = MarkdownChunker(size=128, rows=None, min_characters=1, context=True)(text)

    assert chunks[0].text.startswith("# A\n\n## B\n\nword")
    assert all(chunk.text.startswith("# A\n## B\n\n") for chunk in chunks[1:])
    assert all(chunk.tokens <= 128 for chunk in chunks)
    assert all(
        text[chunk.start : chunk.end].strip() in chunk.text
        for chunk in chunks
        if "|" not in chunk.text
    )


@pytest.mark.parametrize(
    "chunker",
    [
        MarkdownChunker(size=64, rows=2, min_characters=1, context=True, overlap=8),
        MarkdownChunker(size=64, rows=None, min_characters=1),
        TableChunker(size=2, overlap=1),
        RecursiveChunker(size=32, overlap=4, min_characters=1),
    ],
)
def test_text_is_prefix_span_and_suffix(chunker: Chunker) -> None:
    pytest.importorskip("chonkie")
    text = "# A\n\n## B\n\n" + "word " * 30 + "\n\n" + TABLE

    for chunk in chunker(text):
        assert chunk.text == chunk.prefix + text[chunk.start : chunk.end] + chunk.suffix
