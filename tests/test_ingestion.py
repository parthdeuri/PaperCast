"""Ingestion: ID validation, retry behaviour, stable PDF IDs, token counting,
and the arXiv cache. No test touches the network."""
import os

import arxiv
import pymupdf
import pytest

from src import ingestion
from src.models import PaperData


@pytest.mark.parametrize("arxiv_id", [
    "2306.07691", "2306.07691v2", "1706.03762", "hep-th/9901001", "math.GT/0309136v1",
])
def test_valid_arxiv_ids(arxiv_id):
    assert ingestion._ARXIV_ID_RE.match(arxiv_id)


@pytest.mark.parametrize("arxiv_id", ["foo", "paper.pdf", "123", "2306.123", "../etc/passwd", ""])
def test_invalid_arxiv_ids(arxiv_id):
    assert not ingestion._ARXIV_ID_RE.match(arxiv_id)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(ingestion._fetch_arxiv_result.retry, "sleep", lambda s: None)


def test_not_found_is_not_retried(fake_arxiv, no_sleep):
    fake_arxiv.missing.add("2306.07691")
    paper = ingestion.fetch_paper("2306.07691")
    assert paper.extraction_mode == "failed"
    assert len(fake_arxiv.calls) == 1


def test_invalid_id_never_calls_arxiv(fake_arxiv, no_sleep):
    paper = ingestion.fetch_paper("not-an-id")
    assert paper.extraction_mode == "failed"
    assert fake_arxiv.calls == []


def test_rate_limit_is_retried_four_times(fake_arxiv, no_sleep):
    fake_arxiv.error = arxiv.HTTPError("http://export.arxiv.org", 0, 429)
    paper = ingestion.fetch_paper("2306.07691")
    assert paper.extraction_mode == "failed"
    assert len(fake_arxiv.calls) == 4


def test_bad_request_status_is_not_retried(fake_arxiv, no_sleep):
    fake_arxiv.error = arxiv.HTTPError("http://export.arxiv.org", 0, 400)
    ingestion.fetch_paper("2306.07691")
    assert len(fake_arxiv.calls) == 1


def make_pdf(path, lines):
    doc = pymupdf.open()
    page = doc.new_page()
    for i, text in enumerate(lines):
        page.insert_text((50, 60 + 14 * i), text)
    doc.save(str(path))
    doc.close()


LONG_LINES = [f"Line {i} of a test paper about FooNet reaching 85.3 percent accuracy." for i in range(30)]


def test_local_pdf_gets_hash_id_and_counts_tokens(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "copy" / "renamed.pdf"
    b.parent.mkdir()
    make_pdf(a, LONG_LINES)
    b.write_bytes(a.read_bytes())

    usage = {}
    paper = ingestion.fetch_paper(str(a), usage=usage)
    assert paper.paper_id.startswith("pdf-") and len(paper.paper_id) == len("pdf-") + 10
    assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0

    # Same bytes at another path -> same ID.
    assert ingestion.fetch_paper(str(b)).paper_id == paper.paper_id


def test_fetch_papers_sums_pdf_usage(tmp_path):
    paths = []
    for name in ("one.pdf", "two.pdf"):
        p = tmp_path / name
        make_pdf(p, LONG_LINES + [f"unique to {name}"])
        paths.append(str(p))

    usage = {}
    papers = ingestion.fetch_papers(paths, usage=usage)
    assert len(papers) == 2
    assert len({p.paper_id for p in papers}) == 2

    single = {}
    ingestion.fetch_paper(paths[0], usage=single)
    assert usage["input_tokens"] > single["input_tokens"] > 0


def test_cache_hit_makes_no_arxiv_call(fake_arxiv):
    cached = PaperData(
        paper_id="2306.07691", title="Cached", abstract="a", full_text="cached text",
        extraction_mode="full",
    )
    ingestion._write_cache(cached)

    paper = ingestion.fetch_paper("2306.07691")
    assert paper.title == "Cached"
    assert fake_arxiv.calls == []


def test_abstract_only_result_is_not_cached(fake_arxiv):
    first = ingestion.fetch_paper("2306.07691")
    assert first.extraction_mode == "abstract_only"
    ingestion.fetch_paper("2306.07691")
    assert len(fake_arxiv.calls) == 2  # not served from cache


def test_full_extraction_is_cached(fake_arxiv, tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, LONG_LINES)
    fake_arxiv.pdf_url = "http://example.test/paper.pdf"

    class FakeResponse:
        content = pdf.read_bytes()
        def raise_for_status(self):
            pass

    monkeypatch.setattr(ingestion.requests, "get", lambda *a, **k: FakeResponse())

    first = ingestion.fetch_paper("2306.07691")
    assert first.extraction_mode == "full"
    again = ingestion.fetch_paper("2306.07691")
    assert again == first
    assert len(fake_arxiv.calls) == 1


def test_corrupt_cache_is_a_miss(fake_arxiv):
    ingestion._write_cache(PaperData(
        paper_id="2306.07691", title="x", abstract="a", full_text="t", extraction_mode="full",
    ))
    with open(ingestion._cache_path("2306.07691"), "w", encoding="utf-8") as f:
        f.write("{not json")

    paper = ingestion.fetch_paper("2306.07691")
    assert paper.extraction_mode == "abstract_only"
    assert len(fake_arxiv.calls) == 1


@pytest.mark.parametrize("pdf_lines", [LONG_LINES, []], ids=["extracted", "too-little-text"])
def test_downloaded_pdf_is_deleted_after_extraction(fake_arxiv, tmp_path, monkeypatch, pdf_lines):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, pdf_lines)
    fake_arxiv.pdf_url = "http://example.test/paper.pdf"

    class FakeResponse:
        content = pdf.read_bytes()
        def raise_for_status(self):
            pass

    monkeypatch.setattr(ingestion.requests, "get", lambda *a, **k: FakeResponse())

    created = []
    real_mkstemp = ingestion.tempfile.mkstemp
    def recording_mkstemp(*a, **k):
        fd, path = real_mkstemp(*a, **k)
        created.append(path)
        return fd, path
    monkeypatch.setattr(ingestion.tempfile, "mkstemp", recording_mkstemp)

    ingestion.fetch_paper("2306.07691")
    assert len(created) == 1
    assert not os.path.exists(created[0])
