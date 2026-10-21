# PaperCast: AI Podcast Debate Generator

🚀 **[Live Demo](https://papercast-parth.streamlit.app/)**

Turns 1-3 research papers (via arXiv IDs or local PDF uploads) into a debate-style podcast script between three
personas (Academic, Skeptic, Host), with citation grounding to catch hallucinated quotes, an optional human-in-the-loop review step, and a cost/latency dashboard.

## Architecture

```
ingest → academic_agent → grounding → rag_lookup → skeptic_agent → host_agent → script_grounding → review → (finalize)
 (1-3     (per paper)      (verifies    (cross-refs    (critiques,    (writes the   (flags unknown       (optional pause,
 papers)                    citations    against local  incl. cross-   debate        paper IDs and        toggled by
                            against      library + live paper +       script)       unsupported          human_in_loop)
                            source)      arxiv search)  external lit)                numbers)                 |
                                                                          ^                                   |
                                                                          +---- "regenerate" with feedback ---+
```

The **RAG layer** (`rag_lookup`) queries two sources per paper: a persistent
local Chroma library of every paper this tool has processed (grows across
runs - each run's papers get added to it), and a live arXiv search for
related work never seen before. Both feed into the Skeptic's
`external_literature_notes` - "is this claim supported or contradicted by
work outside the papers given in this run?" Embeddings run locally via
sentence-transformers - free, no API key - and this whole step is
best-effort: if Chroma, the embedding model, or arXiv search fail for any
reason, it logs a warning and continues with empty RAG context rather than
breaking the pipeline.

Built with **LangGraph** for orchestration (typed shared state, not just a
linear chain of function calls), **Pydantic** structured outputs everywhere
(no regex-parsing free-text LLM responses), and **Streamlit** for the
review/dashboard UI.

## Key design decisions (useful for interviews)

- **Structured output over free text.** Every agent call uses
  `.with_structured_output()` against a Pydantic schema, so a malformed
  response fails loudly and specifically rather than silently producing
  garbage downstream.
- **Citation grounding is not an LLM call.** It's plain fuzzy-string-matching
  (`src/citation_grounding.py`) run against the actual extracted paper text.
  This is deliberate - you don't want to ask a model "did you just make this
  quote up?" and trust the answer. Verified independently instead.
- **Graceful degradation on ingestion.** A paper whose PDF can't be parsed
  cleanly (scanned images, weird layout) falls back to abstract-only mode
  rather than crashing or silently passing empty text into an agent.
- **The Host's script is checked too.** `script_grounding` (also plain
  string logic, `src/script_grounding.py`) flags lines that cite a paper
  that isn't in the run, or state a number that doesn't appear in the cited
  paper's text. It only flags - the reviewer decides.
- **Human-in-the-loop is optional and toggleable**, implemented via
  LangGraph's `interrupt()`/`Command(resume=...)` primitives. The reviewer
  can approve (with edits) or **regenerate** with feedback, which sends the
  graph back to the Host. When off, `review_node` passes straight through -
  useful for scheduled/unattended runs. The CLI uses an in-memory
  checkpointer; the dashboard uses SQLite (`checkpoints.sqlite`), so a paused
  review survives a page refresh or restart. Runs older than 7 days are
  deleted from it when the dashboard starts (`CHECKPOINT_RETENTION_DAYS`
  in `src/config.py`), and uploaded PDFs are deleted once ingested.
- **Cost/latency tracked per node**, not just per run, and persisted to
  `logs/run_log.jsonl` across every run so the dashboard shows trends, not
  just a single execution's numbers.
- **MOCK_MODE** lets you exercise the entire graph - ingestion, grounding,
  cost tracking, HITL pause/resume - without an LLM API key or spending
  money. It replaces only the LLM calls: fetching papers from arXiv and the
  RAG embedding model still need internet (the test suite stubs both).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`: set `PODCAST_PROVIDER` to `gemini` (recommended, free, no
credit card - get a key at https://aistudio.google.com/apikey) or `groq`
(also free, no credit card - https://console.groq.com/keys), and paste the
matching key. Leave all keys blank to run in MOCK_MODE (no real LLM calls,
useful for checking the wiring works).

Anthropic is also supported (`PODCAST_PROVIDER=anthropic`) but has no
permanent free tier - only occasional, non-guaranteed trial credits.

## Usage

### CLI

```bash
# unattended run, no review pause
python -m src.main --inputs 2306.07691 2310.06825 --no-human-in-loop

# with human review pause (prompts in terminal)
python -m src.main --inputs 2306.07691 --human-in-loop

# generate an audio episode too (Kokoro TTS via kokoro-onnx, free, local,
# works on Python 3.10-3.13)
python -m src.main --inputs 2306.07691 --no-human-in-loop --audio outputs/episode.wav

# save the transcript; the format follows the extension (.md or .json)
python -m src.main --inputs 2306.07691 --no-human-in-loop --transcript outputs/episode.md
```

The CLI prints a line as each pipeline step finishes. Fetched arXiv papers
are cached under `cache/arxiv/`, so re-running the same paper skips the
download (delete that folder to force a fresh fetch).

Audio generation needs the `kokoro-onnx` package (already in
`requirements.txt`) plus **two model files you download once manually**
(no auto-download - `kokoro-onnx` doesn't fetch them for you like the
torch-based `kokoro` package does):

```bash
curl -L -o kokoro-v1.0.int8.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Place both files in the project root (or pass custom paths to
`render_script_to_audio()` in `src/tts.py`). ~88MB + ~27MB, one-time
download, fully offline after that. Output is `.wav` (plays in any media
player) - see `convert_wav_to_mp3()` in `src/tts.py` if you want an actual
`.mp3` (requires `ffmpeg` + `pydub`).

> Why not the original `kokoro` PyPI package? It's torch-based and its
> releases cap out at Python <3.13. `kokoro-onnx` has no torch dependency
> and explicitly supports 3.10-3.13, at the cost of a one-time manual model
> download instead of an automatic one.

### Dashboard (recommended - has the review UI + charts)

```bash
streamlit run dashboard/app.py
```

Three tabs:
- **Run Pipeline** - enter arXiv IDs (or upload PDFs), toggle
  human-in-the-loop on/off, run; progress shows step by step. If paused, the
  script lines flagged by the script check are highlighted; edit lines
  inline and approve, approve the original draft, or regenerate with
  feedback. Refreshing the page while paused restores the review. Once a
  final script exists you can download the transcript (.md / .json), and a
  "Generate Audio" button renders it with Kokoro TTS and plays it inline
  (with a download button) - disabled automatically if the model files
  aren't found in the project root.
- **Cost & Latency** - reads `logs/run_log.jsonl` across all runs ever
  executed, with per-node cost/latency bar charts and an error rate metric.
- **Library Management** - view, delete, or clear the papers in the local
  RAG library.

The sidebar always shows the active provider/model and whether MOCK_MODE is
on, and lets you point at custom model file paths for audio generation.

## Features & Capabilities

- **Local PDF & arXiv Support**: Multi-paper ingestion (1-3 papers) via arXiv IDs or direct PDF uploads, with retry/backoff and abstract-only fallback.
- **Domain-Specific Heuristics**: Model determines the domain of the paper to generate specialized skeptic critiques.
- Three-agent pipeline (Academic → Skeptic → Host) with cross-paper conflict detection.
- **Parallelized Agent Calls**: papers are fetched in parallel during ingestion, and the Academic agent summarizes all papers in parallel.
- Citation grounding / anti-hallucination verification layer, plus a script check on the Host's output.
- Review step with approve / edit / regenerate-with-feedback, and transcript export (Markdown or JSON).
- **RAG layer**: cross-references each paper's claims against (a) a persistent
  local library of every paper this tool has processed, growing across runs,
  and (b) live arXiv search for related work never seen before. Feeds the
  Skeptic's `external_literature_notes`.
- **RAG Library Management**: View, delete, and clear the local Chroma database from the dashboard.
- Optional human-in-the-loop review, toggleable per run.
- Per-node cost and latency tracking with a Streamlit dashboard.
- **Docker Containerization**: Easily spin up the dashboard and dependencies using `docker-compose`.
- **Flexible LLM Support**: Use Gemini, Groq, Anthropic, or any OpenAI-compatible local model (Ollama, vLLM).
- Audio generation via **Edge TTS** (fast, cloud-based) or **Kokoro TTS** (free, local, per-persona voices).

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite runs in mock mode with arXiv and the RAG layer stubbed out, so it
needs no network or API keys, and it writes its logs, caches and checkpoints
into temporary folders - your real `logs/`, `chroma_db/` and `cache/` are
never touched.

## Project layout

```
src/
  models.py             # Pydantic schemas + LangGraph state
  config.py             # model choice, pricing table, mock mode
  ingestion.py           # arxiv fetch + PDF extraction + fallback
  citation_grounding.py  # anti-hallucination verification
  script_grounding.py    # checks the Host's script against the source papers
  export.py              # transcript export (Markdown / JSON)
  cost_tracker.py        # per-node cost/latency logging
  llm_client.py          # structured-output LLM wrapper + mock mode
  graph.py               # LangGraph wiring, incl. optional HITL interrupt
  main.py                # CLI entry point
  tts.py                 # Kokoro TTS audio rendering
  rag.py                 # Chroma library + arxiv search cross-referencing
  agents/
    academic.py
    grounding.py
    rag.py                # graph node wrapping rag.py
    skeptic.py
    host.py
dashboard/
  app.py                 # Streamlit: run pipeline + review UI + cost charts
tests/                   # pytest suite (mock mode, no network)
logs/
  run_log.jsonl          # append-only cost/latency history (created on first run)
cache/arxiv/             # cached arXiv papers (created on first run)
checkpoints.sqlite       # dashboard run checkpoints (created on first run)
```
