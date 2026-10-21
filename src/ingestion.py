"""
Paper ingestion: fetch metadata + text for up to MAX_PAPERS arxiv papers.

Failure modes this handles explicitly (this is the part tutorials skip):
  - Invalid/nonexistent arxiv ID -> skipped with a warning, pipeline continues
    with the remaining valid papers.
  - PDF exists but text extraction yields too little text (e.g. scanned
    images, weird layout) -> falls back to abstract-only mode rather than
    silently passing an empty string downstream.
  - Network/arxiv API failure -> retried with backoff before giving up.
  - Papers fetched in full are cached on disk (config.ARXIV_CACHE_DIR), so
    re-running the same paper skips both the arXiv call and the PDF download.
"""
import re
import os
import hashlib
import tempfile
import logging
import requests
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

from .models import PaperData
from . import config
from .llm_client import call_structured

logger = logging.getLogger(__name__)

try:
    import arxiv
except ImportError:  # pragma: no cover
    arxiv = None

try:
    import pymupdf as fitz  # PyMuPDF's modern import name
except ImportError:  # pragma: no cover
    fitz = None


class IngestionError(Exception):
    """A permanent failure (bad ID, paper not found) - retrying won't help."""
    pass


class TransientIngestionError(IngestionError):
    """A failure that may succeed on retry (rate limit, network, server error)."""
    pass


# New-style IDs (2306.07691, 2306.07691v2) and old-style IDs (hep-th/9901001,
# math.GT/0309136v1).
_ARXIV_ID_RE = re.compile(
    r"^(\d{4}\.\d{4,5}|[a-z][a-z\-]*(\.[A-Z]{2})?/\d{7})(v\d+)?$"
)


class PDFMetadata(BaseModel):
    title: str = Field(description="The title of the paper")
    authors: List[str] = Field(description="The authors of the paper")
    abstract: str = Field(description="The abstract of the paper")
    domain: str = Field(description="The broad scientific domain of the paper (e.g., Computer Science, Biology, Physics, Medicine, General Science)")


# Module-level singleton, reused across every arxiv call (ingestion AND rag.py's
# related-paper search) - this lets arxiv's own internal rate-limit delay
# tracking (delay_seconds) actually take effect across multiple calls in one
# run, instead of resetting every time a fresh Client() is constructed.
ARXIV_CLIENT = arxiv.Client(delay_seconds=5, num_retries=3) if arxiv else None


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=10, max=90),  # arXiv 429s need real cooldown
    # Only retry failures that might go away. A bad or unknown ID fails
    # immediately instead of waiting through ~1 minute of backoff.
    retry=retry_if_exception_type((TransientIngestionError, ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_arxiv_result(arxiv_id: str):
    if arxiv is None:
        raise IngestionError("`arxiv` package not installed")
    if not _ARXIV_ID_RE.match(arxiv_id):
        hint = " (looks like a PDF path, but no such file exists)" if arxiv_id.lower().endswith(".pdf") else ""
        raise IngestionError(f"'{arxiv_id}' is not a valid arXiv ID{hint}")
    search = arxiv.Search(id_list=[arxiv_id])
    client = ARXIV_CLIENT
    try:
        result = next(client.results(search))
    except StopIteration:
        raise IngestionError(f"No paper found for arxiv id '{arxiv_id}'")
    except arxiv.HTTPError as e:
        # 4xx (except 429 rate limiting) means the request itself is bad.
        if 400 <= e.status < 500 and e.status != 429:
            raise IngestionError(f"arxiv rejected '{arxiv_id}' (HTTP {e.status})")
        raise TransientIngestionError(f"arxiv API error for '{arxiv_id}': {e}")
    except Exception as e:
        raise TransientIngestionError(f"arxiv API error for '{arxiv_id}': {e}")
    return result


def _extract_intro_and_conclusion(pdf_path: str) -> str:
    """Best-effort extraction of intro + conclusion sections from a PDF.
    Falls back to whatever text we can get if section headers aren't found."""
    if fitz is None:
        return ""
    try:
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        logger.warning(f"PDF text extraction failed for {pdf_path}: {e}")
        return ""

    # Heuristic section splitting - looks for common header patterns.
    intro_match = re.search(
        r"(?im)^\s*(1\.?\s*)?introduction\s*$", full_text
    )
    conclusion_match = re.search(
        r"(?im)^\s*(\d+\.?\s*)?(conclusion|conclusions|discussion)\s*$", full_text
    )

    if intro_match and conclusion_match and conclusion_match.start() > intro_match.start():
        intro = full_text[intro_match.end():intro_match.end() + config.MAX_FULLTEXT_CHARS_PER_PAPER // 2]
        conclusion = full_text[conclusion_match.end():conclusion_match.end() + config.MAX_FULLTEXT_CHARS_PER_PAPER // 2]
        return f"[INTRODUCTION]\n{intro}\n\n[CONCLUSION]\n{conclusion}"

    # Couldn't find clean section boundaries - just take the head of the doc.
    return full_text[: config.MAX_FULLTEXT_CHARS_PER_PAPER]


def _cache_path(arxiv_id: str) -> str:
    return os.path.join(config.ARXIV_CACHE_DIR, f"{arxiv_id.replace('/', '_')}.json")


def _read_cache(arxiv_id: str):
    """Return the cached PaperData, or None. A missing or corrupt file is
    just a cache miss."""
    if not _ARXIV_ID_RE.match(arxiv_id):
        return None
    try:
        with open(_cache_path(arxiv_id), encoding="utf-8") as f:
            return PaperData.model_validate_json(f.read())
    except (OSError, ValueError):
        return None


def _write_cache(paper: PaperData) -> None:
    try:
        os.makedirs(config.ARXIV_CACHE_DIR, exist_ok=True)
        with open(_cache_path(paper.paper_id), "w", encoding="utf-8") as f:
            f.write(paper.model_dump_json())
    except OSError as e:
        logger.warning(f"Could not write arXiv cache for {paper.paper_id}: {e}")


def _extract_local_pdf(pdf_path: str, usage: dict = None) -> PaperData:
    """Extract text from a local PDF and use an LLM to infer its metadata and domain.
    If `usage` is given, the LLM call's token counts are added to it."""
    if fitz is None:
        raise IngestionError("PyMuPDF (fitz) is not installed, cannot read local PDFs")
    
    extracted = _extract_intro_and_conclusion(pdf_path)
    if not extracted:
        raise IngestionError(f"Could not extract any text from {pdf_path}")
        
    system_prompt = "You are a helpful assistant that extracts metadata from research papers."
    user_prompt = f"Extract the title, authors, abstract, and scientific domain from the following text extracted from a PDF. If you cannot find the exact abstract, summarize the introduction as the abstract.\n\nText:\n{extracted[:8000]}"
    
    def mock_metadata():
        return PDFMetadata(
            title=os.path.basename(pdf_path),
            authors=["Unknown Author"],
            abstract="Mock abstract for local PDF.",
            domain="Computer Science"
        )
        
    try:
        response = call_structured(system_prompt, user_prompt, PDFMetadata, mock_factory=mock_metadata)
        meta = response.parsed
        if usage is not None:
            usage["input_tokens"] = usage.get("input_tokens", 0) + response.input_tokens
            usage["output_tokens"] = usage.get("output_tokens", 0) + response.output_tokens
    except Exception as e:
        logger.warning(f"Failed to extract metadata via LLM for {pdf_path}: {e}")
        meta = mock_metadata()
        
    # Attempt to extract full text since we are reading a local file anyway.
    try:
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception:
        full_text = extracted

    combined_text = f"Title: {meta.title}\n\nAbstract: {meta.abstract}\n\n{full_text}"
    
    # Hash the file contents so the ID is stable across runs and doesn't
    # depend on where the file happens to live (e.g. a fresh temp dir).
    with open(pdf_path, "rb") as f:
        paper_id = "pdf-" + hashlib.sha256(f.read()).hexdigest()[:10]

    return PaperData(
        paper_id=paper_id,
        title=meta.title,
        authors=meta.authors,
        abstract=meta.abstract,
        full_text=combined_text[: config.MAX_FULLTEXT_CHARS_PER_PAPER],
        extraction_mode="full",
        char_count=len(combined_text),
        domain=meta.domain
    )


def fetch_paper(source_id: str, download_pdf: bool = True, usage: dict = None) -> PaperData:
    """Fetch a single paper. source_id can be an arxiv_id or a local PDF file path.
    Never raises for recoverable issues - instead returns a PaperData with 
    extraction_mode reflecting what actually happened. `usage`, if given,
    collects LLM token counts (only local PDFs make an LLM call here)."""
    source_id = source_id.strip()
    
    if source_id.lower().endswith(".pdf") and os.path.exists(source_id):
        try:
            return _extract_local_pdf(source_id, usage)
        except IngestionError as e:
            logger.error(str(e))
            return PaperData(
                paper_id=source_id,
                title="[FETCH FAILED]",
                abstract="",
                full_text="",
                extraction_mode="failed",
                char_count=0,
                domain="Unknown"
            )

    cached = _read_cache(source_id)
    if cached is not None:
        return cached

    try:
        result = _fetch_arxiv_result(source_id)
    except IngestionError as e:
        logger.error(str(e))
        return PaperData(
            paper_id=source_id,
            title="[FETCH FAILED]",
            abstract="",
            full_text="",
            extraction_mode="failed",
            char_count=0,
            domain="Unknown"
        )

    title = result.title.strip()
    authors = [a.name for a in result.authors]
    abstract = result.summary.strip().replace("\n", " ")
    
    # We'll use a simple heuristic for arxiv domains based on their primary category if possible
    # (Arxiv categories look like cs.AI, math.CO, etc)
    domain_heuristic = "General Science"
    if hasattr(result, 'primary_category') and result.primary_category:
        cat = result.primary_category.split('.')[0].lower()
        if cat in ('cs', 'stat'): domain_heuristic = "Computer Science / Stats"
        elif cat in ('math'): domain_heuristic = "Mathematics"
        elif cat in ('q-bio'): domain_heuristic = "Quantitative Biology"
        elif cat in ('physics', 'astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'nucl-ex', 'nucl-th', 'quant-ph'): domain_heuristic = "Physics"
        elif cat in ('q-fin'): domain_heuristic = "Quantitative Finance"
        elif cat in ('econ'): domain_heuristic = "Economics"

    full_text = ""
    mode = "abstract_only"

    if download_pdf and fitz is not None:
        try:
            if not result.pdf_url:
                raise IngestionError("No pdf_url available for this paper")
            pdf_resp = requests.get(result.pdf_url, timeout=30)
            pdf_resp.raise_for_status()
            pdf_path = os.path.join(tempfile.gettempdir(), f"{source_id.replace('/', '_')}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_resp.content)
            extracted = _extract_intro_and_conclusion(pdf_path)
            if len(extracted) >= config.ABSTRACT_ONLY_FALLBACK_THRESHOLD:
                full_text = extracted
                mode = "full"
        except Exception as e:
            logger.warning(f"PDF download/extraction failed for {source_id}, falling back to abstract: {e}")

    if mode == "abstract_only":
        full_text = abstract

    combined_text = f"Title: {title}\n\nAbstract: {abstract}\n\n{full_text}"

    paper = PaperData(
        paper_id=source_id,
        title=title,
        authors=authors,
        abstract=abstract,
        full_text=combined_text[: config.MAX_FULLTEXT_CHARS_PER_PAPER],
        extraction_mode=mode,
        char_count=len(combined_text),
        domain=domain_heuristic
    )
    # Only cache full extractions: an abstract-only result may just mean the
    # PDF download failed this time, and shouldn't stick forever.
    if mode == "full":
        _write_cache(paper)
    return paper


def fetch_papers(input_sources: List[str], download_pdf: bool = True, usage: dict = None) -> List[PaperData]:
    """Fetch 1-MAX_PAPERS papers. Invalid IDs or local files are dropped with a logged
    warning rather than failing the whole batch. If `usage` is given, it gets
    the summed input_tokens/output_tokens of any LLM calls made while ingesting."""
    if not (config.MIN_PAPERS <= len(input_sources) <= config.MAX_PAPERS):
        raise ValueError(
            f"Expected {config.MIN_PAPERS}-{config.MAX_PAPERS} input sources, got {len(input_sources)}"
        )

    from concurrent.futures import ThreadPoolExecutor
    papers = []
    # One usage dict per source so worker threads never share a counter.
    usages = [{"input_tokens": 0, "output_tokens": 0} for _ in input_sources]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_paper, source, download_pdf, u)
                   for source, u in zip(input_sources, usages)]
        for future, source in zip(futures, input_sources):
            try:
                paper = future.result()
                if paper.extraction_mode == "failed":
                    logger.warning(f"Skipping '{source}' - could not ingest.")
                    continue
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Error fetching '{source}': {e}")

    if usage is not None:
        usage["input_tokens"] = usage.get("input_tokens", 0) + sum(u["input_tokens"] for u in usages)
        usage["output_tokens"] = usage.get("output_tokens", 0) + sum(u["output_tokens"] for u in usages)

    if not papers:
        raise ValueError("None of the provided input sources could be fetched.")

    return papers
