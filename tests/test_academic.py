"""Academic node: one failing paper is tolerated, all failing is an error."""
import pytest

from src.agents import academic


def boom(*args, **kwargs):
    raise RuntimeError("model unavailable")


def test_all_papers_failing_raises(monkeypatch, make_paper):
    monkeypatch.setattr(academic, "call_structured", boom)
    with pytest.raises(ValueError, match="failed for every paper: model unavailable"):
        academic.academic_node({"papers": [make_paper("2306.07691"), make_paper("2310.06825")]})


def test_one_failing_paper_is_skipped(monkeypatch, make_paper):
    real = academic.call_structured

    def flaky(system, user, schema, mock_factory=None):
        if "2310.06825" in user:
            raise RuntimeError("model unavailable")
        return real(system, user, schema, mock_factory=mock_factory)

    monkeypatch.setattr(academic, "call_structured", flaky)
    out = academic.academic_node({"papers": [make_paper("2306.07691"), make_paper("2310.06825")]})
    assert [s.paper_id for s in out["summaries"]] == ["2306.07691"]
