"""
Shared data models for PaperCast.

These are the Pydantic schemas used both as (a) structured-output targets for
LLM calls (so we never rely on regex-parsing free text out of a model
response) and (b) the LangGraph shared state that flows between nodes.
"""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

class PaperData(BaseModel):
    """Everything we extracted for a single paper."""
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str
    full_text: str  # abstract + intro + conclusion (or abstract-only fallback)
    extraction_mode: Literal["full", "abstract_only", "failed"] = "full"
    char_count: int = 0
    domain: str = "General Science"


# --------------------------------------------------------------------------
# Citation grounding
# --------------------------------------------------------------------------

class Citation(BaseModel):
    """A quote an agent claims comes from a specific paper."""
    quote: str = Field(description="Short verbatim quote (<25 words) from the paper")
    paper_id: str = Field(description="paper_id of the source this quote is from")
    claim: str = Field(description="The claim this quote is being used to support")


class VerifiedCitation(Citation):
    """A citation after we've checked it actually appears in the source text."""
    verified: bool
    match_score: float = 0.0  # 0-1 fuzzy match confidence


# --------------------------------------------------------------------------
# RAG: cross-referencing against prior work
# --------------------------------------------------------------------------

class RAGMatch(BaseModel):
    """A related-work hit, either from the local library built across past
    runs, or from a live arXiv search - never from the papers in this run."""
    paper_id: str
    title: str
    snippet: str
    source: Literal["library", "arxiv_search"]
    for_paper_id: str = Field(description="Which input paper this match was found for")
    similarity: Optional[float] = None  # only set for library matches (cosine similarity)


# --------------------------------------------------------------------------
# Agent outputs (structured, so parsing never silently fails)
# --------------------------------------------------------------------------

class PaperSummary(BaseModel):
    paper_id: str
    summary: str = Field(description="Dense 3-5 sentence summary of methodology + findings")
    key_claims: List[str] = Field(description="2-4 core claims made by the paper")
    citations: List[Citation] = Field(default_factory=list)


class SkepticCritique(BaseModel):
    per_paper_flaws: dict[str, List[str]] = Field(
        default_factory=dict, description="paper_id -> list of methodological flaws"
    )
    cross_paper_conflicts: List[str] = Field(
        default_factory=list,
        description="Contradictions or tensions between papers, only when 2+ papers",
    )
    external_literature_notes: List[str] = Field(
        default_factory=list,
        description="How this paper's claims relate to prior work found via RAG "
        "(supported, contradicted, or extended by literature outside the input set)",
    )


class DialogueLine(BaseModel):
    speaker: Literal["Host", "Academic", "Skeptic"]
    text: str
    cites_paper_ids: List[str] = Field(default_factory=list)


class DebateScript(BaseModel):
    lines: List[DialogueLine]


# --------------------------------------------------------------------------
# Cost / latency tracking
# --------------------------------------------------------------------------

class NodeRunLog(BaseModel):
    node_name: str
    started_at: float
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    error: Optional[str] = None


# --------------------------------------------------------------------------
# LangGraph shared state
# --------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    input_sources: List[str]
    papers: List[PaperData]
    summaries: List[PaperSummary]
    verified_citations: List[VerifiedCitation]
    critique: SkepticCritique
    rag_context: List[RAGMatch]
    draft_script: DebateScript
    final_script: DebateScript
    human_feedback: Optional[str]
    human_approved: bool
    human_in_loop: bool  # toggle: whether to pause for review before finalizing
    run_log: List[NodeRunLog]
    grounding_report: dict
    script_grounding_report: dict  # flags on the Host's script (see script_grounding.py)
    # Set by the review node when the reviewer asks for a rewrite; the Host
    # reads it, then clears it so the regenerate loop ends.
    regenerate_feedback: Optional[str]
