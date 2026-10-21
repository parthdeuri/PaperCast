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
    assert "**Academic:** It reaches 99.9%. _(refs: 2306.07691)_" in md
    assert "Paper 2306.07691" in md and "Ada Lovelace" in md
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
