"""
The Host agent: takes the summaries + critique and produces the actual
conversational debate script, tagging each line with the speaker and which
paper(s) it references (used later for TTS voice assignment and for
displaying citations alongside the transcript in the review UI).

When the reviewer asks for a rewrite, the graph loops back here with
`regenerate_feedback` set: the feedback and the previous draft are added to
the prompt, and the field is cleared so the loop ends after one rewrite.
"""
from ..models import DebateScript, DialogueLine
from ..llm_client import call_structured
from ..cost_tracker import track_node

SYSTEM_PROMPT = """You are "The Host" of a podcast. You're given paper summaries and a skeptic's \
critique (possibly covering multiple papers). Write an engaging, natural-sounding conversational \
debate script between three speakers: "Host" (moderates, asks questions), "Academic" (explains \
and defends the findings), and "Skeptic" (raises the critiques). Aim for 10-20 lines total. \
Tag each line with cites_paper_ids listing which paper(s) that line is about, when applicable.
"""


def _mock_script(summaries: list, critique, feedback: str = None) -> DebateScript:
    lines = [
        DialogueLine(speaker="Host", text="[MOCK] Welcome back to the show - today we're debating some new research.", cites_paper_ids=[]),
    ]
    for s in summaries:
        lines.append(DialogueLine(speaker="Academic", text=f"[MOCK] The key finding in {s.paper_id} is exciting.", cites_paper_ids=[s.paper_id]))
        lines.append(DialogueLine(speaker="Skeptic", text=f"[MOCK] I have concerns about {s.paper_id}'s methodology.", cites_paper_ids=[s.paper_id]))
    if feedback:
        lines.append(DialogueLine(speaker="Host", text=f"[MOCK] Revised per feedback: {feedback}", cites_paper_ids=[]))
    lines.append(DialogueLine(speaker="Host", text="[MOCK] Great discussion - that wraps up today's episode.", cites_paper_ids=[]))
    return DebateScript(lines=lines)


def host_node(state: dict) -> dict:
    run_log = state.setdefault("run_log", [])
    summaries = state["summaries"]
    critique = state["critique"]

    summaries_text = "\n".join(f"- {s.paper_id}: {s.summary}" for s in summaries)
    flaws_text = "\n".join(
        f"- {pid}: {', '.join(flaws)}" for pid, flaws in critique.per_paper_flaws.items()
    )
    conflicts_text = "\n".join(f"- {c}" for c in critique.cross_paper_conflicts) or "None"
    lit_notes_text = "\n".join(f"- {n}" for n in critique.external_literature_notes) or "None"
    feedback = state.get("regenerate_feedback")

    with track_node("host_agent", run_log) as ctx:
        user_prompt = (
            f"Summaries:\n{summaries_text}\n\nFlaws raised by the Skeptic:\n{flaws_text}\n\n"
            f"Cross-paper conflicts:\n{conflicts_text}\n\n"
            f"Notes on related prior literature:\n{lit_notes_text}"
        )
        previous = state.get("draft_script")
        if feedback and previous is not None:
            previous_text = "\n".join(f"[{l.speaker}] {l.text}" for l in previous.lines)
            user_prompt += (
                f"\n\nReviewer feedback on the previous draft: {feedback}\n\n"
                f"Previous draft:\n{previous_text}\n\n"
                "Write a new script that addresses the feedback."
            )
        response = call_structured(
            SYSTEM_PROMPT,
            user_prompt,
            DebateScript,
            mock_factory=lambda: _mock_script(summaries, critique, feedback),
        )
        ctx["input_tokens"] = response.input_tokens
        ctx["output_tokens"] = response.output_tokens
        script = response.parsed

    # Clear the feedback so the review -> host loop only runs once per request.
    return {"draft_script": script, "regenerate_feedback": None, "run_log": run_log}
