import pytest

from exar.ingest import TokenChunker, fingerprint


def test_fingerprint_counts_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = fingerprint(TokenChunker())
    monkeypatch.setattr(TokenChunker, "revision", 1, raising=False)

    assert fingerprint(TokenChunker()) != initial
