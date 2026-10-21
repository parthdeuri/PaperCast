"""Script grounding: unknown paper IDs and unsupported numbers are flagged."""
from src.models import DebateScript, DialogueLine
from src.script_grounding import check_script


def script(*lines):
    return DebateScript(lines=[DialogueLine(speaker=s, text=t, cites_paper_ids=c) for s, t, c in lines])


def test_invented_paper_id_is_flagged(make_paper):
    papers = [make_paper("2306.07691")]
    report = check_script(script(("Academic", "It works well.", ["9999.99999"])), papers)
    assert report["total_lines"] == 1
    [flag] = report["flagged_lines"]
    assert flag["index"] == 0 and flag["speaker"] == "Academic"
    assert "9999.99999" in flag["reasons"][0]


def test_unsupported_number_is_flagged(make_paper):
    papers = [make_paper("2306.07691", full_text="FooNet reaches 85.3% accuracy.")]
    report = check_script(
        script(("Academic", "It reaches 92.7% accuracy.", ["2306.07691"])), papers
    )
    [flag] = report["flagged_lines"]
    assert any("92.7" in r for r in flag["reasons"])


def test_supported_number_passes(make_paper):
    papers = [make_paper("2306.07691", full_text="FooNet reaches 85.3% accuracy, 2.5x faster.")]
    report = check_script(
        script(("Academic", "It reaches 85.3% accuracy and is 2.5x faster.", ["2306.07691"])),
        papers,
    )
    assert report["flagged_lines"] == []


def test_number_is_checked_against_cited_paper_only(make_paper):
    papers = [
        make_paper("2306.07691", full_text="Accuracy was 85.3%."),
        make_paper("2310.06825", full_text="Accuracy was 71.2%."),
    ]
    report = check_script(script(("Academic", "It hit 71.2%.", ["2306.07691"])), papers)
    assert len(report["flagged_lines"]) == 1


def test_uncited_line_checks_all_papers(make_paper):
    papers = [
        make_paper("2306.07691", full_text="Accuracy was 85.3%."),
        make_paper("2310.06825", full_text="Accuracy was 71.2%."),
    ]
    report = check_script(script(("Host", "One paper hit 71.2%.", [])), papers)
    assert report["flagged_lines"] == []


def test_paper_id_digits_and_small_integers_are_ignored(make_paper):
    papers = [make_paper("2306.07691", full_text="No numbers here.")]
    report = check_script(
        script(("Host", "Paper 2306.07691 compares 3 baselines on 2 datasets.", ["2306.07691"])),
        papers,
    )
    assert report["flagged_lines"] == []


def test_number_overlapping_a_paper_id_is_still_checked(make_paper):
    # "23" and "0769" both occur inside the ID 2306.07691, but are claims here.
    papers = [make_paper("2306.07691", full_text="No numbers here.")]
    report = check_script(
        script(("Academic", "2306.07691 is 23% better, using 0769 GPUs.", ["2306.07691"])), papers
    )
    [flag] = report["flagged_lines"]
    assert any("'23'" in r for r in flag["reasons"])
    assert any("'0769'" in r for r in flag["reasons"])


def test_thousands_separators_match(make_paper):
    papers = [make_paper("2306.07691", full_text="Trained on 1,500 examples.")]
    report = check_script(script(("Academic", "It used 1500 examples.", ["2306.07691"])), papers)
    assert report["flagged_lines"] == []
