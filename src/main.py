"""
CLI entry point.

Usage:
    python -m src.main --inputs 2306.07691 path/to/paper.pdf --human-in-loop
    python -m src.main --inputs 2306.07691 --no-human-in-loop --transcript outputs/episode.md

If --human-in-loop is set and the graph pauses, this CLI prompts you right
in the terminal to approve the draft or ask the Host to regenerate it with
your feedback - useful for testing without the Streamlit dashboard. In
practice you'd more likely drive the paused state from the dashboard (see
dashboard/app.py), which is why review_node's `interrupt()` payload is plain
JSON-serializable data, not tied to any one frontend.
"""
import argparse
import json
import os
import sys
import uuid

from .graph import build_graph, run_graph
from .export import script_to_markdown, script_to_json
from langgraph.types import Command


def _print_progress(node_name: str, seconds: float):
    print(f"✓ {node_name} ({seconds:.1f}s)")


def _write_transcript(path: str, result: dict):
    args = (
        result["final_script"], result["papers"],
        result.get("grounding_report"), result.get("script_grounding_report"),
    )
    text = script_to_json(*args) if path.lower().endswith(".json") else script_to_markdown(*args)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run(input_sources: list[str], human_in_loop: bool, audio_path: str = None, transcript_path: str = None):
    # The Windows console defaults to a legacy code page that can't print
    # the progress check mark or scripts with non-ASCII characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result, payload = run_graph(
            app, {"input_sources": input_sources, "human_in_loop": human_in_loop},
            config, on_step=_print_progress,
        )

        while payload is not None:
            print("\n=== HUMAN REVIEW REQUIRED ===")
            for line in payload["draft_script"]:
                print(f"[{line['speaker']}] {line['text']}")
            print("\nGrounding report:", json.dumps(payload["grounding_report"], indent=2))
            flagged = payload.get("script_grounding_report", {}).get("flagged_lines", [])
            if flagged:
                print("\nScript lines flagged by the script check:")
                for f in flagged:
                    print(f"  line {f['index'] + 1} [{f['speaker']}]: {'; '.join(f['reasons'])}")
            choice = input("\n[a]pprove or [r]egenerate? [a]: ").strip().lower()
            if choice.startswith("r"):
                feedback = input("Feedback for the Host: ").strip()
                resume_value = {"action": "regenerate", "feedback": feedback}
            else:
                resume_value = {"action": "approve"}
            result, payload = run_graph(app, Command(resume=resume_value), config, on_step=_print_progress)
    except ValueError as e:
        print(f"\n❌ {e}")
        print("Check your inputs and API keys. If ingestion failed, arXiv may be rate limiting "
              "(HTTP 429) - wait a minute or two and try again.")
        return None

    print("\n=== FINAL SCRIPT ===")
    for line in result["final_script"].lines:
        refs = f" (refs: {', '.join(line.cites_paper_ids)})" if line.cites_paper_ids else ""
        print(f"[{line.speaker}] {line.text}{refs}")

    total_cost = sum(r.estimated_cost_usd for r in result["run_log"])
    total_time = sum(r.duration_s for r in result["run_log"])
    print(f"\n=== COST/LATENCY ===\nTotal estimated cost: ${total_cost:.4f}  |  Total time: {total_time:.2f}s")
    print(f"Grounding rate: {result['grounding_report']['grounding_rate']:.0%}")
    script_report = result.get("script_grounding_report")
    if script_report:
        print(f"Script check: {len(script_report['flagged_lines'])}/{script_report['total_lines']} lines flagged")

    if transcript_path:
        _write_transcript(transcript_path, result)
        print(f"Transcript written to {transcript_path}")

    if audio_path:
        print(f"\n=== GENERATING AUDIO ===\nRendering to {audio_path} (this can take a while)...")
        from .tts import render_script_to_audio
        try:
            render_script_to_audio(result["final_script"], audio_path)
            print(f"Done: {audio_path}")
        except Exception as e:
            print(f"Audio generation failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="PaperCast")
    parser.add_argument("--inputs", nargs="+", required=True, help="1-3 arXiv IDs or local PDF file paths")
    parser.add_argument("--human-in-loop", dest="hil", action="store_true", default=False)
    parser.add_argument("--no-human-in-loop", dest="hil", action="store_false")
    parser.add_argument("--audio", dest="audio_path", default=None,
                         help="If set, render the final script to this .wav path using Kokoro TTS "
                              "(e.g. --audio outputs/episode.wav)")
    parser.add_argument("--transcript", dest="transcript_path", default=None,
                         help="If set, save the final script to this file; the format follows the "
                              "extension (.md or .json), e.g. --transcript outputs/episode.md")
    args = parser.parse_args()
    run(args.inputs, args.hil, args.audio_path, args.transcript_path)


if __name__ == "__main__":
    main()
