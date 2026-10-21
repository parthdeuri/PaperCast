"""
Streamlit dashboard - three tabs:
  1. "Run Pipeline" - kick off a run with a toggle for human-in-loop, and if
     it pauses for review, edit/approve the draft script right here (or ask
     the Host to regenerate it with feedback). Once a final script exists,
     download the transcript, generate audio (Kokoro / Edge TTS) and play it
     inline.
  2. "Cost & Latency" - reads logs/run_log.jsonl (persisted across every run,
     not just the current session) and renders per-node cost/latency charts.
  3. "Library Management" - view/delete papers in the local RAG library.

The sidebar shows which LLM provider/model is active and whether you're in
MOCK_MODE, so it's obvious at a glance why output looks like [MOCK].

Runs are checkpointed to a SQLite file, and the run's thread id lives in the
URL (?thread=...). Refreshing the page - or restarting Streamlit - while a
review is paused restores the review screen instead of losing the run.

Run with: streamlit run dashboard/app.py
"""
import sys
import os
import json
import uuid
import hashlib
import html
import shutil
import tempfile
import time
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph, run_graph, prune_old_threads
from src.cost_tracker import load_all_runs
from src.export import script_to_markdown, script_to_json, number_papers, source_label
from src import config as podcast_config
from langgraph.types import Command

st.set_page_config(page_title="PaperCast", layout="wide")


def _remove_stale_temp_files(max_age_hours: float = 24):
    """Upload folders and downloaded PDFs are deleted right after use; this
    catches any left behind by a crash or a killed server."""
    cutoff = time.time() - max_age_hours * 3600
    tmp = tempfile.gettempdir()
    for name in os.listdir(tmp):
        if not name.startswith("papercast_"):
            continue
        path = os.path.join(tmp, name)
        try:
            if os.path.getmtime(path) < cutoff:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        except OSError:
            pass


@st.cache_resource
def get_app():
    # One graph (and one SQLite connection) shared by every session and rerun.
    # Housekeeping runs here because this executes once per server start.
    app = build_graph(podcast_config.CHECKPOINT_DB)
    try:
        prune_old_threads(app.checkpointer, podcast_config.CHECKPOINT_RETENTION_DAYS)
        _remove_stale_temp_files()
    except Exception as e:
        # Cleanup is best-effort; never let it stop the dashboard loading.
        print(f"[dashboard] startup cleanup failed: {e}")
    return app


app = get_app()


def thread_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def restore_run(thread_id):
    """Rebuild the page state from the checkpoint for `thread_id`: a paused
    review comes back as the review screen, a finished run as its result."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value, None
        if state.values.get("final_script"):
            return None, state.values
    except Exception:
        pass
    return None, None


if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
    st.session_state.pending_interrupt = None
    st.session_state.result = None
    thread_from_url = st.query_params.get("thread")
    if thread_from_url:
        st.session_state.thread_id = thread_from_url
        st.session_state.pending_interrupt, st.session_state.result = restore_run(thread_from_url)
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "audio_error" not in st.session_state:
    st.session_state.audio_error = None
for _key in ("job", "notice", "last_steps", "audio_notice"):
    if _key not in st.session_state:
        st.session_state[_key] = None


def apply_outcome(values, payload):
    if payload is not None:
        st.session_state.pending_interrupt = payload
        st.session_state.result = None
    else:
        st.session_state.pending_interrupt = None
        st.session_state.result = values


PIPELINE_LABEL = "Running: ingest → academic → grounding → rag → skeptic → host → script check"

# ---------------------------------------------------------------------------
# Jobs. Anything slow (a pipeline run, a resume, an audio render) is queued
# as st.session_state.job by a button callback and then executed during the
# script run that follows. While a job exists every start button is disabled
# and a Stop button is shown. Stop works because Streamlit interrupts a
# running script at its next st.* call - here, the progress update after
# each pipeline step or audio line - so it takes effect once the step in
# progress (e.g. one LLM call) finishes.
# ---------------------------------------------------------------------------
def set_notice(kind, text):
    st.session_state.notice = (kind, text) if text else None


def start_pipeline():
    """Run button callback: validate the inputs, then queue the run."""
    if st.session_state.job:
        return
    ids = [p.strip() for p in st.session_state.get("papers_input", "").split(",") if p.strip()]
    pdf_entries = [p for p in ids if p.lower().endswith(".pdf")]
    if pdf_entries:
        # Typed paths would make the server read arbitrary local files.
        set_notice("error", f"'{pdf_entries[0]}' looks like a PDF path. Paths can't be typed here - "
                            "upload the file with the uploader instead.")
        return

    sources = list(ids)
    upload_dir = None
    uploads = st.session_state.get("uploads") or []
    if uploads:
        # A fresh folder per run, so same-named uploads never collide.
        upload_dir = tempfile.mkdtemp(prefix="papercast_")
        for i, uploaded_file in enumerate(uploads):
            path = os.path.join(upload_dir, f"{i}_{os.path.basename(uploaded_file.name)}")
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            sources.append(path)

    notice = None
    if len(sources) > podcast_config.MAX_PAPERS:
        notice = f"Only the first {podcast_config.MAX_PAPERS} papers are used."
        sources = sources[:podcast_config.MAX_PAPERS]
    set_notice("warning", notice)

    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_interrupt = None
    st.session_state.result = None
    st.session_state.audio_path = None  # new run - clear any previous episode's audio
    st.session_state.audio_error = None
    st.session_state.audio_notice = None
    st.session_state.last_steps = None
    st.session_state.job = {
        "kind": "pipeline", "label": PIPELINE_LABEL, "new_thread": True, "upload_dir": upload_dir,
        "input": {"input_sources": sources, "human_in_loop": st.session_state.get("hil", False)},
    }


def queue_resume(resume_value, label):
    if st.session_state.job:
        return
    set_notice(None, None)
    st.session_state.job = {"kind": "pipeline", "label": label, "input": Command(resume=resume_value)}


def approve_with_edits(draft_id, draft_lines):
    edited = [
        {
            "speaker": st.session_state[f"speaker_{draft_id}_{i}"],
            "text": st.session_state[f"text_{draft_id}_{i}"],
            "cites_paper_ids": line.get("cites_paper_ids", []),
        }
        for i, line in enumerate(draft_lines)
    ]
    queue_resume({"action": "approve", "edited_lines": edited}, "Finalizing the script...")


def regenerate_draft(draft_id):
    feedback = st.session_state.get(f"feedback_{draft_id}", "")
    queue_resume({"action": "regenerate", "feedback": feedback}, "Regenerating the script...")


def start_audio(engine, model_path, voices_path):
    if st.session_state.job:
        return
    st.session_state.audio_error = None
    st.session_state.audio_notice = None
    set_notice(None, None)
    st.session_state.job = {"kind": "audio", "engine": engine,
                            "model_path": model_path, "voices_path": voices_path}


def stop_job():
    """Stop button callback. Streamlit then interrupts the running script."""
    job = st.session_state.job
    st.session_state.job = None
    if not job:
        return
    if job.get("upload_dir"):
        shutil.rmtree(job["upload_dir"], ignore_errors=True)
    if job["kind"] == "pipeline":
        # Show whatever the run had reached (usually nothing usable).
        st.session_state.pending_interrupt, st.session_state.result = restore_run(st.session_state.thread_id)
        st.session_state.last_steps = None
        set_notice("info", "Pipeline stopped. The step that was running finishes in the background "
                           "and its result is discarded.")
    else:
        # Shown next to the audio controls, where the user clicked Stop.
        st.session_state.audio_notice = "Audio generation stopped."


def execute_pipeline_job(job):
    """Run (or resume) the graph with one progress line per finished step."""
    if job.get("new_thread"):
        st.query_params["thread"] = st.session_state.thread_id
    steps, error = [], None
    with st.status(job["label"], expanded=True) as status:
        def on_step(name, secs):
            steps.append(f"✓ {name} ({secs:.1f}s)")
            status.write(steps[-1])  # also where a Stop click interrupts the run
        try:
            values, payload = run_graph(app, job["input"], thread_config(), on_step=on_step)
        except Exception as e:
            error = e
    if job.get("upload_dir"):
        # Ingestion has read the uploads by now (their text lives in the
        # run's state), and nothing - not even a resumed review - opens them again.
        shutil.rmtree(job["upload_dir"], ignore_errors=True)

    st.session_state.job = None
    if error is not None:
        st.session_state.last_steps = ("Pipeline failed", steps, "error")
        set_notice("error", f"Pipeline failed: {error}")
    else:
        apply_outcome(values, payload)
        st.session_state.last_steps = ("Paused for review" if payload is not None else "Done", steps, "complete")
    st.rerun()


def execute_audio_job(job, script, slot):
    """Render the episode, showing progress in `slot` (an st.empty()). Putting
    a single element into that placeholder replaces whatever it held before -
    the previous episode's player, download button or messages - straight
    away, instead of leaving them on screen (stale, but still clickable)
    until the render finishes."""
    from src.tts import render_script_to_audio
    out_path = os.path.join("outputs", f"episode_{st.session_state.thread_id}.wav")
    total = len(script.lines)
    bar = slot.progress(0.0, text=f"Rendering audio with {job['engine']}... line 0/{total}")

    def on_line(done, n):
        bar.progress(done / n, text=f"Rendering audio with {job['engine']}... line {done}/{n}")

    try:
        render_script_to_audio(
            script, out_path, model_path=job["model_path"], voices_path=job["voices_path"],
            tts_engine=job["engine"], on_line=on_line,
        )
        st.session_state.audio_path = out_path
        st.session_state.audio_error = None
    except Exception as e:
        st.session_state.audio_path = None
        st.session_state.audio_error = str(e)
    st.session_state.job = None
    st.rerun()


# ---------------------------------------------------------------------------
# Presentation helpers. Citations show as short numbered chips ([1], [2]);
# hovering one shows the paper's full title and ID, so long IDs (a PDF's
# hash, or a file path from an old run) never clutter the script.
# ---------------------------------------------------------------------------
SPEAKERS = {"Host": ("🎙️", "#8b5cf6"), "Academic": ("🎓", "#0ea5e9"), "Skeptic": ("🧐", "#f97316")}

st.markdown(
    """<style>
.pc-line{display:flex;flex-wrap:wrap;gap:.25rem .75rem;padding:.6rem .85rem;margin:.4rem 0;
  border-radius:.6rem;background:rgba(128,128,128,.08);border-left:3px solid var(--pc-c);}
.pc-speaker{font-weight:600;color:var(--pc-c);min-width:7rem;}
.pc-text{flex:1 1 20rem;line-height:1.55;}
.pc-ref{display:inline-block;font-size:.72rem;font-weight:600;line-height:1.4;padding:0 .4rem;
  margin-left:.25rem;border-radius:.5rem;background:rgba(99,102,241,.18);cursor:help;vertical-align:1px;}
.pc-ref.pc-unknown{background:rgba(239,68,68,.22);}
.pc-flag{flex-basis:100%;font-size:.8rem;color:#ea580c;}
.pc-sources{margin:.25rem 0 .75rem;font-size:.9rem;}
.pc-sources div{margin:.2rem 0;}
.pc-muted{opacity:.65;font-size:.82rem;}
</style>""",
    unsafe_allow_html=True,
)


def ref_chips(paper_ids, papers) -> str:
    """HTML chips like [1][2] with the full title and ID as a hover tooltip."""
    numbers = number_papers(papers)
    by_id = {p.paper_id: p for p in papers}
    chips = []
    for pid in paper_ids:
        if pid in by_id:
            p = by_id[pid]
            label = source_label(p)
            tip = f"{p.title} - {label}" + ("" if pid in label else f" ({pid})")
            chips.append(f'<span class="pc-ref" title="{html.escape(tip)}">{numbers[pid]}</span>')
        else:
            tip = f"Not a paper in this run: {pid}"
            chips.append(f'<span class="pc-ref pc-unknown" title="{html.escape(tip)}">?</span>')
    return "".join(chips)


def with_numbers(text: str, papers) -> str:
    """Replace paper IDs inside a message (e.g. a grounding reason) with [n]."""
    for pid, n in sorted(number_papers(papers).items(), key=lambda kv: -len(kv[0])):
        text = text.replace(pid, f"[{n}]")
    return text


def render_sources(papers):
    rows = []
    for p in papers:
        mode = "" if p.extraction_mode == "full" else " · abstract only"
        rows.append(
            f"<div>{ref_chips([p.paper_id], papers)} <b>{html.escape(p.title)}</b> "
            f'<span class="pc-muted">{html.escape(source_label(p))}{mode}</span></div>'
        )
    st.markdown(f'<div class="pc-sources">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_script(lines, papers, flags_by_line=None):
    """The script as one block of speaker-coloured lines with citation chips."""
    flags_by_line = flags_by_line or {}
    blocks = []
    for i, line in enumerate(lines):
        icon, color = SPEAKERS.get(line.speaker, ("💬", "#64748b"))
        flag = ""
        if i in flags_by_line:
            reasons = with_numbers("; ".join(flags_by_line[i]["reasons"]), papers)
            flag = f'<div class="pc-flag">⚠️ {html.escape(reasons)}</div>'
        blocks.append(
            f'<div class="pc-line" style="--pc-c:{color}">'
            f'<div class="pc-speaker">{icon} {html.escape(line.speaker)}</div>'
            f'<div class="pc-text">{html.escape(line.text)}{ref_chips(line.cites_paper_ids, papers)}</div>'
            f"{flag}</div>"
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def run_papers():
    """Papers of the current run, read from its checkpoint (the review
    payload itself doesn't carry them)."""
    try:
        return app.get_state(thread_config()).values.get("papers", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# SIDEBAR: current provider/model/mock-mode status - always visible so
# "why is this [MOCK]" or "why is this free/paid" is never a mystery.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Current Configuration")
    st.text(f"Provider: {podcast_config.PROVIDER}")
    st.text(f"Model: {podcast_config.MODEL_NAME}")
    if podcast_config.MOCK_MODE:
        st.warning("MOCK_MODE is ON - no real LLM calls will be made. "
                    "Output will contain '[MOCK]' placeholder text.\n\n"
                    "Set the matching API key in your .env file to use a real model.")
    else:
        st.success("Using a real LLM - MOCK_MODE is off.")

    st.divider()
    st.subheader("🔊 Audio Generation")
    tts_engine = st.selectbox("TTS Engine", ["edge_tts", "kokoro"], disabled=st.session_state.job is not None)

    model_path = None
    voices_path = None

    if tts_engine == "kokoro":
        st.caption("Kokoro Model Files (for offline TTS)")
        model_path = st.text_input("kokoro model path", value="kokoro-v1.0.int8.onnx", key="model_path_input",
                                   disabled=st.session_state.job is not None)
        voices_path = st.text_input("voices path", value="voices-v1.0.bin", key="voices_path_input",
                                    disabled=st.session_state.job is not None)

    if tts_engine == "kokoro":
        if os.path.exists(model_path) and os.path.exists(voices_path):
            st.success("Model files found - audio generation is available.")
        else:
            st.warning(
                "Model files not found in the project root. Audio generation will fail until "
                "you download them (see README.md for the download commands)."
            )

st.title("🎙️ PaperCast")
st.caption("Turn 1-3 research papers into a debate between a Host, an Academic and a Skeptic - "
           "with every quote and number checked against the source papers.")

tab_run, tab_dashboard, tab_library = st.tabs(["🎙️ Run Pipeline", "📊 Cost & Latency", "📚 Library Management"])

# ---------------------------------------------------------------------------
# TAB 1: Run pipeline + human-in-the-loop review + audio generation
# ---------------------------------------------------------------------------
with tab_run:
    job = st.session_state.job
    busy = job is not None
    st.subheader("Generate a debate episode")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("arXiv IDs (comma-separated)", placeholder="2306.07691, 2310.06825",
                      key="papers_input", disabled=busy)
        st.file_uploader("Or upload PDF files", type=["pdf"], accept_multiple_files=True,
                         key="uploads", disabled=busy)
    with col2:
        st.toggle(
            "Human-in-the-loop review", value=False, key="hil", disabled=busy,
            help="When ON, the pipeline pauses before finalizing so you can edit, "
            "approve or regenerate the script here. When OFF, it runs straight through unattended.",
        )

    has_input = bool(st.session_state.get("papers_input", "").strip() or st.session_state.get("uploads"))
    run_col, stop_col, _ = st.columns([1.6, 1.1, 4])
    run_col.button("▶️ Run Pipeline", type="primary", disabled=busy or not has_input,
                   on_click=start_pipeline, width="stretch")
    if busy and job["kind"] == "pipeline":
        stop_col.button("⏹️ Stop", on_click=stop_job, width="stretch",
                        help="Stops after the step in progress finishes.")

    # Fixed slots, so nothing below shifts position between runs.
    notice_slot = st.container()
    progress_slot = st.container()
    # While a pipeline job runs this stays empty, which also clears the
    # previous review/result at once instead of leaving it on screen (and
    # clickable) until the run ends.
    body = st.empty()

    if st.session_state.notice:
        kind, text = st.session_state.notice
        getattr(notice_slot, kind)(text)

    if busy and job["kind"] == "pipeline":
        with progress_slot:
            execute_pipeline_job(job)  # reruns the page when done

    if st.session_state.last_steps:
        label, steps, state = st.session_state.last_steps
        with progress_slot.status(label, state=state, expanded=False):
            for line in steps:
                st.write(line)

    with body.container():
        # --- Human review UI (only shown when the graph is actually paused) ---
        if st.session_state.pending_interrupt:
            payload = st.session_state.pending_interrupt
            papers = run_papers()
            st.divider()
            st.subheader("⏸️ Review the draft")
            st.caption("Edit any line, then approve - or tell the Host what to change and regenerate.")

            gr = payload["grounding_report"]
            script_flags = payload.get("script_grounding_report", {}).get("flagged_lines", [])
            flags_by_line = {f["index"]: f for f in script_flags}
            c1, c2, c3 = st.columns(3)
            c1.metric("Quotes verified", f"{gr['verified_citations']}/{gr['total_citations']}",
                      help="Academic's quotes found in the source text")
            c2.metric("Unverified quotes", gr["hallucinated_citations"])
            c3.metric("Script lines flagged", len(script_flags),
                      help="Lines citing an unknown paper, or a number not found in the cited paper")
            if gr["flagged"]:
                with st.expander(f"⚠️ {len(gr['flagged'])} quote(s) could not be found in the source paper"):
                    for f in gr["flagged"]:
                        st.markdown(
                            f"{ref_chips([f['paper_id']], papers)} \"{html.escape(f['quote'])}\" "
                            f'<span class="pc-muted">match {f["match_score"]:.2f}</span>',
                            unsafe_allow_html=True,
                        )

            if papers:
                st.markdown("**Sources** <span class='pc-muted'>(hover a number for details)</span>",
                            unsafe_allow_html=True)
                render_sources(papers)

            # Widget keys include a hash of the draft, so a regenerated draft gets
            # fresh editors instead of inheriting the previous draft's edits.
            draft_lines = payload["draft_script"]
            draft_id = hashlib.md5(json.dumps(draft_lines, sort_keys=True).encode()).hexdigest()[:8]

            st.markdown("**Script**")
            for i, line in enumerate(draft_lines):
                cols = st.columns([1.5, 5])
                cols[0].selectbox(
                    "Speaker", list(SPEAKERS),
                    index=list(SPEAKERS).index(line["speaker"]),
                    key=f"speaker_{draft_id}_{i}", label_visibility="collapsed",
                    format_func=lambda s: f"{SPEAKERS[s][0]} {s}", disabled=busy,
                )
                cols[1].text_area(
                    "Line", value=line["text"], key=f"text_{draft_id}_{i}",
                    label_visibility="collapsed", height=68, disabled=busy,
                )
                cites = line.get("cites_paper_ids", [])
                notes = []
                if cites:
                    notes.append(f'<span class="pc-muted">cites</span>{ref_chips(cites, papers)}')
                if i in flags_by_line:
                    reasons = with_numbers("; ".join(flags_by_line[i]["reasons"]), papers)
                    notes.append(f'<span class="pc-flag">⚠️ {html.escape(reasons)}</span>')
                if notes:
                    cols[1].markdown(" &nbsp; ".join(notes), unsafe_allow_html=True)

            col_a, col_o = st.columns(2)
            col_a.button("✅ Approve (with my edits)", type="primary", disabled=busy,
                         on_click=approve_with_edits, args=(draft_id, draft_lines))
            col_o.button("📄 Approve original draft", disabled=busy,
                         on_click=queue_resume, args=({"action": "approve"}, "Finalizing the script..."))

            st.markdown("**Not happy with the draft?** Tell the Host what to change:")
            st.text_area(
                "Feedback for the Host", key=f"feedback_{draft_id}", label_visibility="collapsed",
                placeholder="e.g. Make the Skeptic push harder on the sample size.", disabled=busy,
            )
            st.button("🔁 Regenerate", disabled=busy, on_click=regenerate_draft, args=(draft_id,))

        # --- Final result ---
        if st.session_state.result:
            result = st.session_state.result
            papers = result.get("papers", [])
            script_report = result.get("script_grounding_report") or {}
            script_flags = script_report.get("flagged_lines", [])
            st.divider()
            st.subheader("🎬 Final script")

            total_cost = sum(r.estimated_cost_usd for r in result["run_log"])
            total_time = sum(r.duration_s for r in result["run_log"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cost (this run)", f"${total_cost:.4f}")
            m2.metric("Time (this run)", f"{total_time:.1f}s")
            m3.metric("Quotes verified", f"{result['grounding_report']['grounding_rate']:.0%}")
            m4.metric("Script lines flagged", len(script_flags))

            if papers:
                st.markdown("**Sources** <span class='pc-muted'>(hover a number for details)</span>",
                            unsafe_allow_html=True)
                render_sources(papers)
            render_script(result["final_script"].lines, papers, {f["index"]: f for f in script_flags})

            # --- Transcript downloads ---
            export_args = (
                result["final_script"], result["papers"],
                result.get("grounding_report"), result.get("script_grounding_report"),
            )
            d1, d2, _ = st.columns([1, 1, 2])
            d1.download_button(
                "⬇️ Transcript (.md)", data=script_to_markdown(*export_args),
                file_name="transcript.md", mime="text/markdown", on_click="ignore",
            )
            d2.download_button(
                "⬇️ Transcript (.json)", data=script_to_json(*export_args),
                file_name="transcript.json", mime="application/json", on_click="ignore",
            )

            # --- Audio generation ---
            st.divider()
            st.subheader("🔊 Audio Episode")

            files_ready = True
            if tts_engine == "kokoro":
                files_ready = os.path.exists(model_path) and os.path.exists(voices_path)

            gen_col, stop_audio_col, _ = st.columns([1.3, 1, 3.7])
            gen_col.button(
                "🎧 Generate Audio", type="primary", disabled=busy or not files_ready, width="stretch",
                on_click=start_audio, args=(tts_engine, model_path, voices_path),
            )
            audio_running = busy and job["kind"] == "audio"
            if audio_running:
                stop_audio_col.button("⏹️ Stop", key="stop_audio", on_click=stop_job, width="stretch",
                                      help="Stops after the line in progress finishes.")
            if not files_ready:
                st.caption("⚠️ Model files not found - see sidebar / README to download them first.")

            audio_out = st.empty()
            if audio_running:
                execute_audio_job(job, result["final_script"], audio_out)  # reruns the page when done

            with audio_out.container():
                if st.session_state.audio_notice:
                    st.info(st.session_state.audio_notice)
                if st.session_state.audio_error:
                    st.error(f"Audio generation failed: {st.session_state.audio_error}")

                if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                    st.success(f"Episode ready: `{st.session_state.audio_path}`")
                    with open(st.session_state.audio_path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        "⬇️ Download .wav", data=audio_bytes, on_click="ignore",
                        file_name=os.path.basename(st.session_state.audio_path), mime="audio/wav",
                    )

# ---------------------------------------------------------------------------
# TAB 2: Cost & latency dashboard, across ALL runs ever logged
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.header("Cost & Latency, All Runs")
    runs = load_all_runs()

    if not runs:
        st.info("No runs logged yet - run the pipeline in the first tab to populate this dashboard.")
    else:
        df = pd.DataFrame(runs)
        # Strip the per-paper suffix (e.g. "academic_agent[1234.5678]") so
        # multiple papers in one run group under one node category.
        df["node_group"] = df["node_name"].str.replace(r"\[.*\]", "", regex=True)

        total_cost = df["estimated_cost_usd"].sum()
        total_calls = len(df)
        avg_latency = df["duration_s"].mean()
        error_rate = (df["error"].notna()).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total spend (all time)", f"${total_cost:.4f}")
        m2.metric("Total node calls", total_calls)
        m3.metric("Avg latency/node", f"{avg_latency:.2f}s")
        m4.metric("Node error rate", f"{error_rate:.1%}")

        col_a, col_b = st.columns(2)
        with col_a:
            cost_by_node = df.groupby("node_group")["estimated_cost_usd"].sum().reset_index()
            fig1 = px.bar(cost_by_node, x="node_group", y="estimated_cost_usd",
                          title="Total Cost by Node", labels={"estimated_cost_usd": "USD", "node_group": "Node"})
            st.plotly_chart(fig1, width="stretch")
        with col_b:
            latency_by_node = df.groupby("node_group")["duration_s"].mean().reset_index()
            fig2 = px.bar(latency_by_node, x="node_group", y="duration_s",
                          title="Avg Latency by Node", labels={"duration_s": "seconds", "node_group": "Node"})
            st.plotly_chart(fig2, width="stretch")

        st.subheader("Raw run log")
        st.dataframe(df[["node_name", "duration_s", "input_tokens", "output_tokens", "estimated_cost_usd", "error"]],
                     width="stretch")

# ---------------------------------------------------------------------------
# TAB 3: Library Management
# ---------------------------------------------------------------------------
with tab_library:
    st.header("RAG Library Management")
    st.markdown("Manage the local Chroma database of previously processed papers.")
    
    from src.rag import list_library, delete_from_library, clear_library
    
    papers = list_library()
    if not papers:
        st.info("The RAG library is currently empty.")
    else:
        st.metric("Total Papers in Library", len(papers))
        
        st.subheader("Papers")
        for paper in papers:
            col1, col2 = st.columns([5, 1], vertical_alignment="center")
            origin = "Uploaded PDF" if paper["id"].startswith("pdf-") else f"arXiv {paper['id']}"
            col1.markdown(
                f"<b>{html.escape(paper['title'])}</b><br>"
                f'<span class="pc-muted" title="{html.escape(paper["id"])}">{html.escape(origin)}</span>',
                unsafe_allow_html=True,
            )
            if col2.button("Delete", key=f"del_{paper['id']}", disabled=st.session_state.job is not None):
                if delete_from_library(paper['id']):
                    st.success(f"Deleted {paper['id']}")
                    st.rerun()
                else:
                    st.error("Failed to delete paper.")
                    
        st.divider()
        if st.button("🗑️ Clear Entire Library", type="primary", disabled=st.session_state.job is not None):
            if clear_library():
                st.success("Library cleared.")
                st.rerun()
            else:
                st.error("Failed to clear library.")
