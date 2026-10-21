"""Transcript export: Markdown and JSON contents."""
import json

from src.export import script_to_markdown, script_to_json
from src.models import DebateScript, DialogueLine


def sample(make_paper):
    script = DebateScript(lines=[
        DialogueLine(speaker="Host", text="Welcome.", cites_paper_ids=[]),
        DialogueLine(speaker="Academic", text="It reaches 99.9%.", cites_paper_ids=["2306.07691"]),
    ])
    papers = [make_paper("2306.07691", authors=["Ada Lovelace"])]
    grounding = {"total_citations": 2, "verified_citations": 1, "grounding_rate": 0.5,
                 "hallucinated_citations": 1,
                 "flagged": [{"paper_id": "2306.07691", "quote": "made up", "match_score": 0.1}]}
    script_report = {"total_lines": 2, "flagged_lines": [
        {"index": 1, "speaker": "Academic", "text": "It reaches 99.9%.",
         "reasons": ["number '99.9' not found in 2306.07691"]}]}
    return script, papers, grounding, script_report


def test_markdown_contents(make_paper):
    md = script_to_markdown(*sample(make_paper))
    assert "**Host:** Welcome." in md
    assert "**Academic:** It reaches 99.9%. [1]" in md
    assert "1. **Paper 2306.07691** - Ada Lovelace - arXiv 2306.07691" in md
    assert "Unverified quote in [1]" in md
    assert "1/2" in md and "50%" in md  # citation grounding
    assert "made up" in md
    assert "Line 2 (Academic)" in md  # script check, 1-based


def test_json_contents(make_paper):
    data = json.loads(script_to_json(*sample(make_paper)))
    assert [l["speaker"] for l in data["lines"]] == ["Host", "Academic"]
    assert data["lines"][1]["cites_paper_ids"] == ["2306.07691"]
    assert data["papers"][0]["paper_id"] == "2306.07691"
    assert data["grounding_report"]["grounding_rate"] == 0.5
    assert data["script_grounding_report"]["flagged_lines"][0]["index"] == 1


def test_export_without_reports(make_paper):
    script, papers, _, _ = sample(make_paper)
    assert "## Grounding" not in script_to_markdown(script, papers)
    assert json.loads(script_to_json(script, papers))["grounding_report"] == {}


def test_refs_use_numbers_not_ids(make_paper):
    from src.export import cite_marks, number_papers, source_label
    papers = [make_paper("2306.07691"), make_paper("pdf-3fa91c2e7b")]
    numbers = number_papers(papers)
    assert cite_marks(["pdf-3fa91c2e7b", "2306.07691", "9999.99999"], numbers) == "[2][1][?]"
    assert source_label(papers[0]) == "arXiv 2306.07691"
    assert source_label(papers[1]) == "Uploaded PDF"
    assert source_label(make_paper("C:/Temp/old upload.pdf")) == "Uploaded PDF"
    md = script_to_markdown(
        DebateScript(lines=[DialogueLine(speaker="Host", text="Hi.", cites_paper_ids=["pdf-3fa91c2e7b"])]),
        papers,
    )
    [line] = [l for l in md.splitlines() if l.startswith("**Host:**")]
    assert line == "**Host:** Hi. [2]"
