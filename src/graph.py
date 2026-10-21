"""
Graph wiring:

  ingest -> academic -> grounding -> rag_lookup -> skeptic -> host
         -> script_grounding -> review -> (END | back to host)

The `review` node is where human-in-the-loop lives, and it is OPTIONAL,
controlled by `state["human_in_loop"]` (bool), set per-run in main.py.

  - human_in_loop=True  -> the graph pauses at `review` via LangGraph's
    `interrupt()`, exposing the draft script for editing (e.g. in the
    Streamlit dashboard). The graph resumes once a human responds with
    {"action": "approve", "edited_lines": [...]?} or
    {"action": "regenerate", "feedback": "..."}. Regenerate sends the
    graph back to the Host with the feedback; the Host clears it, so each
    request produces exactly one new draft, which is then reviewed again.
  - human_in_loop=False -> `review` passes the draft straight through as
    final_script with no pause. Useful for unattended/scheduled runs where
    you trust the pipeline and just want output.

A checkpointer is required for `interrupt()` to work at all - without it,
LangGraph has nowhere to persist state across the pause. build_graph() uses
an in-memory one by default (CLI, tests) or a SQLite file when given a path
(dashboard), so a paused review survives a page refresh or restart.
"""
import sqlite3
import time
from datetime import datetime, timedelta, timezone

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .models import PipelineState, DebateScript, DialogueLine
from .ingestion import fetch_papers
from .cost_tracker import track_node
from .agents.academic import academic_node
from .agents.grounding import grounding_node, script_grounding_node
from .agents.rag import rag_node
from .agents.skeptic import skeptic_node
from .agents.host import host_node

DEFAULT_REGENERATE_FEEDBACK = "Please write a fresh version of the script."


def ingest_node(state: PipelineState) -> dict:
    run_log = state.setdefault("run_log", [])
    usage = {}
    with track_node("ingestion", run_log) as ctx:
        papers = fetch_papers(state["input_sources"], usage=usage)
        # Local PDFs make an LLM call for their metadata; arXiv papers don't.
        ctx["input_tokens"] = usage.get("input_tokens", 0)
        ctx["output_tokens"] = usage.get("output_tokens", 0)
    return {"papers": papers, "run_log": run_log}


def review_node(state: PipelineState) -> dict:
    draft = state["draft_script"]

    if not state.get("human_in_loop", False):
        # Toggle is off: pass straight through, no pause.
        return {"final_script": draft, "human_approved": True}

    # Toggle is on: pause here. `interrupt()` halts execution and surfaces
    # this payload to whatever is driving the graph (CLI prompt, Streamlit
    # dashboard, etc). Execution resumes when the caller invokes the graph
    # again with a Command(resume=<decision>).
    decision = interrupt(
        {
            "type": "human_review_required",
            "draft_script": [line.model_dump() for line in draft.lines],
            "grounding_report": state.get("grounding_report", {}),
            "script_grounding_report": state.get("script_grounding_report", {}),
            "instructions": "Return {'action': 'approve'} to accept as-is, "
            "{'action': 'approve', 'edited_lines': [...]} with a replacement line "
            "list in the same shape as draft_script, or {'action': 'regenerate', "
            "'feedback': '...'} to have the Host write a new draft.",
        }
    )

    # The older payload was {"approved": True[, "edited_lines": [...]]}; with
    # no "action" key that still means approve.
    decision = decision if isinstance(decision, dict) else {}
    action = decision.get("action", "approve")

    if action == "regenerate":
        feedback = (decision.get("feedback") or "").strip() or DEFAULT_REGENERATE_FEEDBACK
        return {"regenerate_feedback": feedback, "human_feedback": feedback}

    if decision.get("edited_lines"):
        final = DebateScript(lines=[DialogueLine(**l) for l in decision["edited_lines"]])
    else:
        final = draft

    return {"final_script": final, "human_approved": True, "human_feedback": str(decision)}


def _after_review(state: PipelineState) -> str:
    return "host_agent" if state.get("regenerate_feedback") else END


def build_graph(checkpoint_db: str = None):
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("academic_agent", academic_node)
    graph.add_node("grounding", grounding_node)
    graph.add_node("rag_lookup", rag_node)
    graph.add_node("skeptic_agent", skeptic_node)
    graph.add_node("host_agent", host_node)
    graph.add_node("script_grounding", script_grounding_node)
    graph.add_node("review", review_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "academic_agent")
    graph.add_edge("academic_agent", "grounding")
    graph.add_edge("grounding", "rag_lookup")
    graph.add_edge("rag_lookup", "skeptic_agent")
    graph.add_edge("skeptic_agent", "host_agent")
    graph.add_edge("host_agent", "script_grounding")
    graph.add_edge("script_grounding", "review")
    graph.add_conditional_edges("review", _after_review, {"host_agent": "host_agent", END: END})

    # Explicitly allow our Pydantic model classes through the checkpoint
    # serializer. Without this, LangGraph still works today but emits a
    # deprecation warning on every checkpoint write/read - future LangGraph
    # versions will refuse to (de)serialize unregistered types entirely.
    _model_classes = [
        "PaperData", "Citation", "VerifiedCitation", "PaperSummary",
        "SkepticCritique", "DialogueLine", "DebateScript", "NodeRunLog", "RAGMatch",
    ]
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("src.models", name) for name in _model_classes],
    )
    if checkpoint_db:
        from langgraph.checkpoint.sqlite import SqliteSaver
        # check_same_thread=False: Streamlit runs each rerun on a different thread.
        checkpointer = SqliteSaver(sqlite3.connect(checkpoint_db, check_same_thread=False), serde=serde)
    else:
        checkpointer = MemorySaver(serde=serde)
    return graph.compile(checkpointer=checkpointer)


def prune_old_threads(checkpointer, max_age_days: float) -> int:
    """Delete every run (thread) whose newest checkpoint is older than
    `max_age_days`. Returns how many were deleted. Whole threads are
    deleted, never single checkpoints, so a kept run is always complete."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    newest = {}
    for tup in checkpointer.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        ts = datetime.fromisoformat(tup.checkpoint["ts"])
        if thread_id not in newest or ts > newest[thread_id]:
            newest[thread_id] = ts

    old = [t for t, ts in newest.items() if ts < cutoff]
    for thread_id in old:
        checkpointer.delete_thread(thread_id)

    # SQLite keeps freed pages until VACUUM, so the file wouldn't shrink.
    conn = getattr(checkpointer, "conn", None)
    if old and conn is not None:
        conn.execute("VACUUM")
    return len(old)


def run_graph(app, inputs_or_command, config, on_step=None):
    """Run (or resume) the graph until it finishes or pauses for review.

    `inputs_or_command` is the initial state dict, or a Command(resume=...).
    `on_step(node_name, seconds)` is called as each node finishes, so the
    CLI and dashboard can show live progress.

    Returns (state_values, interrupt_payload). interrupt_payload is the
    dict passed to interrupt() if the graph is now paused, else None.
    """
    interrupt_payload = None
    last = time.time()
    for chunk in app.stream(inputs_or_command, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"][0].value
            continue
        now = time.time()
        for node_name in chunk:
            if on_step:
                on_step(node_name, now - last)
        last = now
    return app.get_state(config).values, interrupt_payload
