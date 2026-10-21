"""
Script grounding: check the Host's finished script against the source papers.

Citation grounding (citation_grounding.py) verifies the Academic's quotes, but
the Host then writes a whole new script from the summaries, and can invent
paper IDs or statistics along the way. Like citation grounding, this is plain
string logic, not an LLM call, so the check can't be talked out of the truth.

Two checks per line:
  (a) every id in cites_paper_ids is a paper from this run;
  (b) every number in the text (decimals, percentages, "3.5x", larger
      integers) appears in the full_text of a cited paper - or of any paper
      when the line cites none.

It only flags, never blocks: the report goes to the human reviewer, who
decides. Bare integers of 10 or less ("three papers", "2 baselines") and the
digits of the paper IDs themselves are ignored, since they are not claims.
"""
import re
from typing import Dict, List

from .models import DebateScript, PaperData

# A number, an optional decimal part, and an optional % or x suffix.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%|x\b)?")


def _strip_thousands(text: str) -> str:
    """'1,000' -> '1000', so both spellings compare equal."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def _numbers_to_check(text: str, paper_ids: List[str]) -> List[str]:
    """Numbers in `text` worth verifying against a paper."""
    # Remove the paper IDs themselves (e.g. "2306.07691") before looking for
    # numbers. Skipping any number that merely appears inside an ID would
    # also skip real claims like "23%" whenever a paper's ID starts with 23.
    for pid in sorted(paper_ids, key=len, reverse=True):
        text = text.replace(pid, " ")
    found = []
    for m in _NUMBER_RE.finditer(_strip_thousands(text)):
        number, suffix = m.group(1), m.group(2)
        if "." not in number and not suffix and int(number) <= 10:
            continue  # "three papers", "2 baselines"
        if number not in found:
            found.append(number)
    return found


def _appears_in(number: str, source: str) -> bool:
    pattern = r"(?<![\d.])" + re.escape(number) + r"(?!\d|\.\d)"
    return re.search(pattern, _strip_thousands(source)) is not None


def check_script(script: DebateScript, papers: List[PaperData]) -> dict:
    """Return {"total_lines": int, "flagged_lines": [{index, speaker, text, reasons}]}."""
    papers_by_id: Dict[str, PaperData] = {p.paper_id: p for p in papers}
    all_ids = list(papers_by_id)
    flagged = []

    for i, line in enumerate(script.lines):
        reasons = []

        for pid in line.cites_paper_ids:
            if pid not in papers_by_id:
                reasons.append(f"cites unknown paper '{pid}'")

        known_cited = [pid for pid in line.cites_paper_ids if pid in papers_by_id]
        sources = known_cited or all_ids
        for number in _numbers_to_check(line.text, all_ids):
            if not any(_appears_in(number, papers_by_id[pid].full_text) for pid in sources):
                where = ", ".join(known_cited) if known_cited else "any paper"
                reasons.append(f"number '{number}' not found in {where}")

        if reasons:
            flagged.append(
                {"index": i, "speaker": line.speaker, "text": line.text, "reasons": reasons}
            )

    return {"total_lines": len(script.lines), "flagged_lines": flagged}
