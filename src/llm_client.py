"""
Shared LLM client wrapper.

Two responsibilities:
  1. Always use structured output (.with_structured_output) so agent
     responses are validated Pydantic objects, never free text we regex out.
  2. Support MOCK_MODE (see config.py) so the entire graph can be exercised
     - including cost tracking and citation grounding - without an LLM API
     key. Mock mode only replaces the LLM calls: fetching papers from arXiv
     and the RAG embedding model still use the network unless they are
     stubbed out, as the test suite does. This is what lets you test the
     pipeline logic separately from "does the LLM produce a good debate
     script".
"""
from typing import Type, TypeVar
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


class StructuredResponse:
    """Normalizes the shape returned by real vs mock calls."""
    def __init__(self, parsed: BaseModel, input_tokens: int, output_tokens: int):
        self.parsed = parsed
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _get_real_llm(schema: Type[T]):
    if config.PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=config.MODEL_NAME, google_api_key=config.GOOGLE_API_KEY)
    elif config.PROVIDER == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=config.MODEL_NAME, api_key=config.GROQ_API_KEY)
    elif config.PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=config.MODEL_NAME, api_key=config.ANTHROPIC_API_KEY, max_tokens=2048)
    elif config.PROVIDER == "openai_compatible":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.MODEL_NAME, 
            api_key=config.OPENAI_API_KEY, 
            base_url=config.OPENAI_API_BASE
        )
    else:
        raise ValueError(f"Unknown PODCAST_PROVIDER: {config.PROVIDER!r}")
    return llm.with_structured_output(schema, include_raw=True)


def call_structured(system_prompt: str, user_prompt: str, schema: Type[T], mock_factory=None) -> StructuredResponse:
    """
    Call the LLM and get back a validated instance of `schema`.

    `mock_factory` is a zero-arg callable returning a valid `schema` instance,
    used only when config.MOCK_MODE is True. Each agent module supplies its
    own mock factory so mock output is realistic for that agent's job.
    """
    if config.MOCK_MODE:
        if mock_factory is None:
            raise ValueError("MOCK_MODE is on but no mock_factory was provided")
        parsed = mock_factory()
        # Rough token estimate for mock mode so the cost dashboard still has
        # plausible-looking numbers to render during a dry run.
        approx_in = max(len(system_prompt.split()) + len(user_prompt.split()), 1)
        approx_out = max(len(str(parsed.model_dump()).split()), 1)
        return StructuredResponse(parsed, approx_in, approx_out)

    structured_llm = _get_real_llm(schema)
    result = structured_llm.invoke(
        [("system", system_prompt), ("human", user_prompt)]
    )
    parsed = result["parsed"]
    raw = result["raw"]
    if parsed is None:
        raise ValueError(f"LLM failed to produce valid {schema.__name__}: {result.get('parsing_error')}")

    usage = getattr(raw, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return StructuredResponse(parsed, input_tokens, output_tokens)
