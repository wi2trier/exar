import pytest

from exar.text import LineClip, decode, fit_window, tagged, truncate_middle

GERMAN = "Benutzer = Jörg Müller\nBeschreibung = Übersicht der Geschäftsprozesse\n"


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32", "cp1252"]
)
def test_decode_round_trips(encoding: str) -> None:
    decoded = decode(GERMAN.encode(encoding))

    assert decoded is not None and decoded.text == GERMAN
    assert (decoded.encoding is None) is (encoding == "utf-8")


@pytest.mark.parametrize(
    "content",
    [
        GERMAN.encode("utf-16-be"),
        b"\x89PNG\r\n\x1a\n",
        b"PK\x03\x04",
        b"text\x00nul",
        b"\xef\xbb\xbf\xff invalid body after a BOM",
        b"\x81 undefined in UTF-8 and CP1252",
    ],
)
def test_decode_refuses_binary(content: bytes) -> None:
    assert decode(content) is None


def test_decode_keeps_terminal_escapes() -> None:
    assert decode(b"\x1b[31mERROR\x1b[0m page\x0cbreak\n") is not None


def test_tagged_escapes_only_its_own_tag() -> None:
    text = tagged("data", "Vec<u8> <div> </DATA> <data x> <database>", source="a&b")

    assert text == (
        '<data source="a&amp;b">\nVec<u8> <div> &lt;/DATA> &lt;data x> <database>\n</data>'
    )
    assert text.count("</data>") == 1


def test_tagged_escapes_whitespace_variants() -> None:
    assert tagged("data", "</ data> < /data> <\n/ DATA>") == (
        "<data>\n&lt;/ data> &lt; /data> &lt;\n/ DATA>\n</data>"
    )


def test_truncate_middle() -> None:
    text = "x" * 100

    assert truncate_middle(text, 100) == text
    assert (
        truncate_middle(text, 20, note=lambda _head, n: f"[{n} cut]")
        == "xxxxxx\n[91 cut]\nxxx"
    )
    # The note can name the line a read resumes at, after the kept head.
    lines = "\n".join("abcdefghij")
    resume = truncate_middle(
        lines, 12, note=lambda head, _: f"[{head.count('\n') + 2}]"
    )
    assert resume == "a\nb\n[3]\nj"
    assert truncate_middle(text, 3) == "[1…"


def test_fit_window_keeps_a_contiguous_prefix() -> None:
    # A later short line must not backfill past a dropped one,
    # since a caller resumes where the output stops.
    window = fit_window(["aaaa", "bbbbbbbb", "c"], 1, 10, LineClip(None))

    assert (window.content, window.end) == ("aaaa", 1)


def test_fit_window_keeps_the_first_line_however_long() -> None:
    window = fit_window(["x" * 500, "y"], 1, 10, LineClip(None))

    assert window.end == 1
    assert window.hints() == ["1 more line, call again with offset=2"]


def test_fit_window_measures_lines_as_clipped() -> None:
    window = fit_window(["x" * 500, "y"], 1, 20, LineClip(5))

    assert window.end == 2
    assert window.hints() == []
