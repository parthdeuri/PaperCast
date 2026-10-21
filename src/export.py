"""
Transcript export: turn a finished script into a Markdown or JSON document.

Shared by the CLI (--transcript) and the dashboard's download buttons, so the
two always produce the same files. Both formats carry the source papers and
the grounding results, so a saved transcript says how far to trust it.
"""
import json
from typing import Dict, List, Optional

from .models import DebateScript, PaperData


def number_papers(papers: List[PaperData]) -> Dict[str, int]:
    """paper_id -> 1-based number, so scripts can cite papers as [1], [2]
    instead of long IDs (a PDF's ID is a hash, and old runs used file paths)."""
    return {p.paper_id: i + 1 for i, p in enumerate(papers)}


def source_label(paper: PaperData) -> str:
    """Short human-readable origin of a paper, e.g. 'arXiv 2306.07691'."""
    pid = paper.paper_id
    if pid.startswith("pdf-") or pid.lower().endswith(".pdf"):
        return "Uploaded PDF"
    return f"arXiv {pid}"


def cite_marks(paper_ids: List[str], numbers: Dict[str, int]) -> str:
    """'[1][2]' for the given IDs; an ID that isn't in this run shows as [?]."""
    return "".join(f"[{numbers[pid]}]" if pid in numbers else "[?]" for pid in paper_ids)


def script_to_markdown(
    script: DebateScript,
    papers: List[PaperData],
    grounding_report: Optional[dict] = None,
    script_grounding_report: Optional[dict] = None,
) -> str:
    numbers = number_papers(papers)
    out = ["# PaperCast Transcript", "", "## Source papers", ""]
    for p in papers:
        authors = ", ".join(p.authors) if p.authors else "unknown authors"
        mode = "" if p.extraction_mode == "full" else f" ({p.extraction_mode.replace('_', ' ')})"
        out.append(f"{numbers[p.paper_id]}. **{p.title}** - {authors} - {source_label(p)}{mode}")

    out += ["", "## Transcript", ""]
    for line in script.lines:
        refs = f" {cite_marks(line.cites_paper_ids, numbers)}" if line.cites_paper_ids else ""
        out.append(f"**{line.speaker}:** {line.text}{refs}")
        out.append("")

    if grounding_report:
        out += [
            "## Grounding",
            "",
            f"- Citations verified: {grounding_report.get('verified_citations', 0)}"
            f"/{grounding_report.get('total_citations', 0)}"
            f" (rate {grounding_report.get('grounding_rate', 1.0):.0%})",
        ]
        for f in grounding_report.get("flagged", []):
            out.append(f"- Unverified quote in {cite_marks([f['paper_id']], numbers)}: \"{f['quote']}\"")
        out.append("")

    if script_grounding_report:
        flagged = script_grounding_report.get("flagged_lines", [])
        out += [
            "## Script check",
            "",
            f"- Lines flagged: {len(flagged)}/{script_grounding_report.get('total_lines', 0)}",
        ]
        for f in flagged:
            out.append(f"- Line {f['index'] + 1} ({f['speaker']}): {'; '.join(f['reasons'])}")
        out.append("")

    return "\n".join(out)


def script_to_json(
    script: DebateScript,
    papers: List[PaperData],
    grounding_report: Optional[dict] = None,
    script_grounding_report: Optional[dict] = None,
) -> str:
    data = {
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "extraction_mode": p.extraction_mode,
            }
            for p in papers
        ],
        "lines": [line.model_dump() for line in script.lines],
        "grounding_report": grounding_report or {},
        "script_grounding_report": script_grounding_report or {},
    }
    return json.dumps(data, indent=2, ensure_ascii=False)
