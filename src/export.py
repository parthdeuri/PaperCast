"""
Transcript export: turn a finished script into a Markdown or JSON document.

Shared by the CLI (--transcript) and the dashboard's download buttons, so the
two always produce the same files. Both formats carry the source papers and
the grounding results, so a saved transcript says how far to trust it.
"""
import json
from typing import List, Optional

from .models import DebateScript, PaperData


def script_to_markdown(
    script: DebateScript,
    papers: List[PaperData],
    grounding_report: Optional[dict] = None,
    script_grounding_report: Optional[dict] = None,
) -> str:
    out = ["# PaperCast Transcript", "", "## Source papers", ""]
    for p in papers:
        authors = ", ".join(p.authors) if p.authors else "unknown authors"
        out.append(f"- **{p.title}** ({p.paper_id}) - {authors} [{p.extraction_mode}]")

    out += ["", "## Transcript", ""]
    for line in script.lines:
        refs = f" _(refs: {', '.join(line.cites_paper_ids)})_" if line.cites_paper_ids else ""
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
            out.append(f"- Unverified quote in {f['paper_id']}: \"{f['quote']}\"")
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
