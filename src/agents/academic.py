"""
The Academic agent: reads one paper at a time and produces a dense summary
with claims tied to specific quotes. Runs once per paper (up to 3), so a
2-3 paper run produces 2-3 independent summaries for the Skeptic to compare.
"""
import logging

from ..models import PaperData, PaperSummary, Citation
from ..llm_client import call_structured
from ..cost_tracker import track_node

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are "The Academic" - an enthusiastic AI researcher explaining a paper's \
core contribution. Read the paper text provided and produce:
- A dense 3-5 sentence summary of the methodology and findings.
- 2-4 key claims the paper makes.
- For EACH key claim, include at least one short (<25 words) VERBATIM quote copied \
directly from the provided paper text that supports it. Do not paraphrase the quote - \
copy it exactly as written. If you cannot find a supporting quote in the text, do not \
invent one; omit the citation for that claim instead.
"""


def _mock_summary(paper: PaperData) -> PaperSummary:
    # Deterministic mock used only in MOCK_MODE, for testing without an API key.
    first_sentence = paper.abstract.split(".")[0][:80] or "the paper's core idea"
    return PaperSummary(
        paper_id=paper.paper_id,
        summary=f"[MOCK] This paper investigates {first_sentence}. The authors propose a method "
        f"and report empirical results supporting their central hypothesis.",
        key_claims=["[MOCK] Claim 1 about the method", "[MOCK] Claim 2 about the results"],
        citations=[
            Citation(
                quote=paper.abstract[:60] if paper.abstract else "mock quote",
                paper_id=paper.paper_id,
                claim="[MOCK] supporting claim",
            )
        ],
    )


def _process_paper(paper: PaperData, run_log: list) -> PaperSummary:
    with track_node(f"academic_agent[{paper.paper_id}]", run_log) as ctx:
        user_prompt = (
            f"Paper text (paper_id={paper.paper_id}):\n\n{paper.full_text}"
        )
        response = call_structured(
            SYSTEM_PROMPT,
            user_prompt,
            PaperSummary,
            mock_factory=lambda p=paper: _mock_summary(p),
        )
        ctx["input_tokens"] = response.input_tokens
        ctx["output_tokens"] = response.output_tokens
        summary = response.parsed
        # Defensive: force paper_id to match, in case the model drifts.
        summary.paper_id = paper.paper_id
        for c in summary.citations:
            c.paper_id = paper.paper_id
        return summary


def academic_node(state: dict) -> dict:
    papers = state["papers"]
    run_log = state.setdefault("run_log", [])
    
    from concurrent.futures import ThreadPoolExecutor
    
    summaries = []
    first_error = None
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_process_paper, paper, run_log) for paper in papers]
        for future in futures:
            try:
                summaries.append(future.result())
            except Exception as e:
                # Log the error but don't crash the whole pipeline if one paper fails.
                logger.warning(f"Error processing paper: {e}")
                first_error = first_error or e

    # Every paper failing is not a partial result - downstream nodes would
    # just crash on empty summaries, so stop here with the real cause.
    if not summaries:
        raise ValueError(f"Academic agent failed for every paper: {first_error}")

    return {"summaries": summaries, "run_log": run_log}
