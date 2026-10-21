"""
Central configuration: model choice, API keys, and a pricing table used by
the cost tracker. Keeping prices here (rather than hardcoded in the tracker)
means updating a rate is a one-line change, and you can point at a different
provider/model without touching agent code.
"""
import os
from dotenv import load_dotenv

# Load variables from a .env file in the project root into os.environ.
# Without this, values in .env are just inert text - nothing reads them
# automatically. This must run before we read any os.environ.get() calls
# below. `override=False` (the default) means real environment variables
# you've exported in your shell still win over .env, which is the expected
# behavior.
load_dotenv()

# --- LLM provider ------------------------------------------------------------
# Pick one: "anthropic" (paid only, no permanent free tier), "gemini" (free,
# no credit card, via Google AI Studio), "groq" (free, no credit card,
# open-weight models), or "openai_compatible" (for local Ollama/vLLM). Set the matching API key env var below.
PROVIDER = os.environ.get("PODCAST_PROVIDER", "gemini")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# `or` instead of a get() default: docker-compose passes unset variables as
# empty strings, and an empty string should fall back to the default too.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "ollama"
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") or "http://localhost:11434/v1"

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.5-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "openai_compatible": "llama3.1",
}
MODEL_NAME = os.environ.get("PODCAST_MODEL") or _DEFAULT_MODELS.get(PROVIDER, "gemini-3.5-flash-lite")

_KEY_BY_PROVIDER = {"anthropic": ANTHROPIC_API_KEY, "gemini": GOOGLE_API_KEY, "groq": GROQ_API_KEY, "openai_compatible": OPENAI_API_KEY}
MOCK_MODE = os.environ.get("PODCAST_MOCK_MODE", "0") == "1" or not _KEY_BY_PROVIDER.get(PROVIDER)

# --- Pipeline limits ---------------------------------------------------------
MAX_PAPERS = 3
MIN_PAPERS = 1
MAX_FULLTEXT_CHARS_PER_PAPER = 12_000  # keep prompts bounded regardless of paper length
ABSTRACT_ONLY_FALLBACK_THRESHOLD = 500  # chars; below this, extraction is considered failed

# --- Cost tracking -----------------------------------------------------------
# $ per million tokens (input, output). Both free providers cost $0 while on
# their free tier - kept here mainly so switching to a paid tier later, or
# back to Anthropic, gives you accurate cost estimates without code changes.
PRICING_PER_MTOK = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    "gemini-3.6-flash": {"input": 0.00, "output": 0.00},        # free tier
    "gemini-3.5-flash-lite": {"input": 0.00, "output": 0.00},   # free tier
    "llama-3.3-70b-versatile": {"input": 0.00, "output": 0.00},  # Groq free tier
    "llama3.1": {"input": 0.00, "output": 0.00},  # Local Ollama
}

# --- Logging -----------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
RUN_LOG_PATH = os.path.join(LOG_DIR, "run_log.jsonl")

# --- Caching -----------------------------------------------------------------
# Fetched arXiv papers are cached as JSON so re-running the same paper skips
# the arXiv API call and the PDF download.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARXIV_CACHE_DIR = os.path.join(_ROOT, "cache", "arxiv")

# --- Checkpoints -------------------------------------------------------------
# SQLite file the dashboard uses for LangGraph checkpoints, so a paused
# review survives a page refresh or a Streamlit restart.
CHECKPOINT_DB = os.path.join(_ROOT, "checkpoints.sqlite")
# Runs whose last activity is older than this are deleted from the file when
# the dashboard starts, so it doesn't grow forever. A review left paused
# longer than this is lost.
CHECKPOINT_RETENTION_DAYS = 7

# --- RAG -----------------------------------------------------------------
# Local, persistent vector store of every paper this tool has ever
# processed (Chroma) - grows across runs. Embeddings via sentence-transformers
# run fully locally (no API calls, no rate limits), but the model itself
# (~80MB) downloads from Hugging Face on first use.
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 3  # matches to surface per paper, per source (library + live arxiv search)
