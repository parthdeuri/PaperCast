"""
Shared test setup. Every test runs in mock mode with no network access and
with all on-disk state (logs, caches, vector store, checkpoints) redirected
into pytest's tmp_path, so tests never touch the real logs/, chroma_db/,
cache/ or checkpoints.sqlite.
"""
from types import SimpleNamespace

import pytest

from src import config, ingestion
from src.agents import rag as rag_agent
from src.models import PaperData

ABSTRACT = (
    "We introduce FooNet, a new model that reaches 85.3% accuracy on ImageNet. "
    "It runs 2.5x faster than the baseline while using fewer parameters."
)


def make_arxiv_result(arxiv_id: str):
    """A stand-in for arxiv.Result with pdf_url=None, which puts ingestion in
    abstract-only mode (no PDF download)."""
    return SimpleNamespace(
        title=f"FooNet {arxiv_id}",
        authors=[SimpleNamespace(name="Ada Lovelace")],
        summary=ABSTRACT,
        primary_category="cs.LG",
        pdf_url=None,
        get_short_id=lambda: arxiv_id,
    )


class FakeArxivClient:
    """Records every search and returns a canned result for each ID, unless
    the ID is listed in `missing` (no result) or `error` is set (raised)."""

    def __init__(self):
        self.calls = []
        self.missing = set()
        self.error = None
        self.pdf_url = None

    def results(self, search):
        self.calls.append(search)
        if self.error is not None:
            raise self.error
        arxiv_id = search.id_list[0]
        if arxiv_id in self.missing:
            return iter([])
        result = make_arxiv_result(arxiv_id)
        result.pdf_url = self.pdf_url
        return iter([result])


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "RUN_LOG_PATH", str(tmp_path / "logs" / "run_log.jsonl"))
    monkeypatch.setattr(config, "ARXIV_CACHE_DIR", str(tmp_path / "cache" / "arxiv"))
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(config, "CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite"))

    # No Chroma, no embedding model, no related-paper arXiv search.
    monkeypatch.setattr(rag_agent, "query_library", lambda *a, **k: [])
    monkeypatch.setattr(rag_agent, "search_related_arxiv", lambda *a, **k: [])
    monkeypatch.setattr(rag_agent, "add_paper_to_library", lambda *a, **k: None)


@pytest.fixture
def fake_arxiv(monkeypatch):
    client = FakeArxivClient()
    monkeypatch.setattr(ingestion, "ARXIV_CLIENT", client)
    return client


@pytest.fixture
def make_paper():
    def _make(paper_id="2306.07691", full_text="Some text. FooNet reaches 85.3% accuracy.", **kw):
        return PaperData(
            paper_id=paper_id, title=f"Paper {paper_id}", abstract="abstract",
            full_text=full_text, **kw,
        )
    return _make
