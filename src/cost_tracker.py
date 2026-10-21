"""
Cost and latency tracking, node by node.

Wraps each graph node call, timing it and (when a real LLM response with a
`usage` field is available) computing an estimated dollar cost from the
pricing table in config.py. Every run appends JSONL entries to logs/run_log.jsonl
so the Streamlit dashboard can read history across multiple runs, not just
the most recent one.
"""
import time
import json
import contextlib
from typing import Optional

from . import config
from .models import NodeRunLog


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = config.PRICING_PER_MTOK.get(model)
    if rates is None:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


@contextlib.contextmanager
def track_node(node_name: str, run_log: list, model: str = config.MODEL_NAME):
    """Context manager used inside each graph node:

        with track_node("academic_agent", state["run_log"]) as ctx:
            response = llm.invoke(...)
            ctx["input_tokens"] = response.usage_metadata["input_tokens"]
            ctx["output_tokens"] = response.usage_metadata["output_tokens"]

    Appends a NodeRunLog to run_log on exit (success or failure) and persists
    it to disk immediately, so a crash mid-pipeline doesn't lose earlier
    node's cost data.
    """
    start = time.time()
    ctx = {"input_tokens": 0, "output_tokens": 0}
    error: Optional[str] = None
    try:
        yield ctx
    except Exception as e:
        error = str(e)
        raise
    finally:
        duration = time.time() - start
        cost = estimate_cost(model, ctx["input_tokens"], ctx["output_tokens"])
        entry = NodeRunLog(
            node_name=node_name,
            started_at=start,
            duration_s=round(duration, 3),
            input_tokens=ctx["input_tokens"],
            output_tokens=ctx["output_tokens"],
            estimated_cost_usd=round(cost, 6),
            model=model,
            error=error,
        )
        run_log.append(entry)
        _persist(entry)


def _persist(entry: NodeRunLog):
    import os
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(config.RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def load_all_runs() -> list:
    """Read the full run history for the dashboard."""
    import os
    if not os.path.exists(config.RUN_LOG_PATH):
        return []
    entries = []
    with open(config.RUN_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
