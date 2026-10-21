"""End-to-end graph tests in mock mode: unattended run, the review step's
approve/edit/regenerate paths, and resuming a paused run from SQLite."""
import uuid

from langgraph.types import Command

from src.graph import build_graph, run_graph

INPUTS = {"input_sources": ["2306.07691"], "human_in_loop": True}


def new_config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_unattended_run_finishes_without_pause(fake_arxiv):
    app = build_graph()
    steps = []
    values, payload = run_graph(
        app, {"input_sources": ["2306.07691"], "human_in_loop": False}, new_config(),
        on_step=lambda name, secs: steps.append(name),
    )

    assert payload is None
    assert values["final_script"].lines
    assert steps == [
        "ingest", "academic_agent", "grounding", "rag_lookup",
        "skeptic_agent", "host_agent", "script_grounding", "review",
    ]
    assert values["script_grounding_report"]["total_lines"] == len(values["draft_script"].lines)
    names = {r.node_name for r in values["run_log"]}
    assert "script_grounding" in names and "ingestion" in names


def test_review_approve_with_edits(fake_arxiv):
    app = build_graph()
    config = new_config()
    values, payload = run_graph(app, INPUTS, config)

    assert payload["type"] == "human_review_required"
    assert "script_grounding_report" in payload
    assert "final_script" not in values

    edited = [{"speaker": "Host", "text": "Edited line.", "cites_paper_ids": []}]
    values, payload = run_graph(
        app, Command(resume={"action": "approve", "edited_lines": edited}), config
    )
    assert payload is None
    assert [l.text for l in values["final_script"].lines] == ["Edited line."]


def test_review_accepts_legacy_approved_payload(fake_arxiv):
    app = build_graph()
    config = new_config()
    values, _ = run_graph(app, INPUTS, config)
    draft = values["draft_script"]

    values, payload = run_graph(app, Command(resume={"approved": True}), config)
    assert payload is None
    assert values["final_script"] == draft


def test_regenerate_runs_host_again_then_approve(fake_arxiv):
    app = build_graph()
    config = new_config()
    run_graph(app, INPUTS, config)

    steps = []
    values, payload = run_graph(
        app, Command(resume={"action": "regenerate", "feedback": "More energy"}), config,
        on_step=lambda name, secs: steps.append(name),
    )
    # Review sent the graph back through the Host, then paused again.
    assert steps == ["review", "host_agent", "script_grounding"]
    assert payload is not None
    assert any("Revised per feedback: More energy" in l["text"] for l in payload["draft_script"])
    assert values.get("regenerate_feedback") in (None, "")

    values, payload = run_graph(app, Command(resume={"action": "approve"}), config)
    assert payload is None
    assert any("Revised per feedback" in l.text for l in values["final_script"].lines)


def test_sqlite_checkpoint_resumes_from_new_graph_instance(fake_arxiv, tmp_path):
    db = str(tmp_path / "cp.sqlite")
    config = new_config()

    first = build_graph(db)
    try:
        _, payload = run_graph(first, INPUTS, config)
        assert payload is not None
    finally:
        first.checkpointer.conn.close()

    # A fresh graph over the same file, as after a Streamlit restart.
    second = build_graph(db)
    try:
        state = second.get_state(config)
        assert state.next == ("review",)
        assert state.tasks[0].interrupts

        values, payload = run_graph(second, Command(resume={"action": "approve"}), config)
        assert payload is None
        assert values["final_script"].lines
    finally:
        second.checkpointer.conn.close()
