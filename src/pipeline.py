"""End-to-end query pipeline: classify -> retrieve -> draft -> verify."""

from dataclasses import dataclass, field

from google.genai import types

from src.agents import client, FLASH_LITE_MODEL, classify_domain, draft_answer, verify_answer
from src.retrieve import retrieve

NO_DOMAIN_MATCH_MESSAGE = (
    "This question doesn't match any of the domains I support (HR, Finance, Support, "
    "or Legal), so I'm not able to look up an answer for it. Please rephrase your "
    "question or direct it to the appropriate team."
)

CITATION_SCORE_THRESHOLD = 0.40

ESCALATE_MESSAGE = (
    "I'm not confident enough in an answer to this question to send it as-is, so I'm "
    "escalating it to the relevant domain team for a human response."
)

CLARIFY_SYSTEM_PROMPT = """You are an enterprise knowledge assistant. A user's question could not be \
answered confidently from the retrieved context because it is ambiguous or underspecified.

Given the user's question and the retrieved context, ask ONE concise clarifying question that would help \
narrow down what the user actually means. Respond with only the clarifying question itself - no preamble, \
no explanation."""


@dataclass
class Answer:
    text: str
    domain: str
    confidence: str
    action: str  # "answer", "clarify", or "escalate"
    citations: list[str] = field(default_factory=list)


def _generate_clarifying_question(query: str, chunk_texts: list[str]) -> str:
    """Ask Gemini Flash-Lite what's ambiguous about `query` given the retrieved context."""
    context = "\n\n".join(f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunk_texts, start=1))
    user_turn = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.models.generate_content(
        model=FLASH_LITE_MODEL,
        contents=user_turn,
        config=types.GenerateContentConfig(
            system_instruction=CLARIFY_SYSTEM_PROMPT,
            max_output_tokens=256,
        ),
    )
    return response.text.strip()


def answer_query(query: str, history: list) -> Answer:
    """Run the full pipeline for `query` and return an Answer.

    `history` must be in Gemini's format: a list of
    {"role": "user" or "model", "parts": [{"text": ...}]} dicts.
    """
    domain = classify_domain(query)
    if domain == "OTHER":
        return Answer(
            text=NO_DOMAIN_MATCH_MESSAGE, domain=domain, confidence="low", action="escalate", citations=[]
        )

    chunks = retrieve(query, domain.lower())
    chunk_texts = [chunk["text"] for chunk in chunks]

    draft = draft_answer(query, chunk_texts, history)
    verification = verify_answer(draft, chunk_texts)
    confidence = verification.get("confidence", "low")
    action = verification.get("action")

    if action == "answer":
        citations = [chunk["source"] for chunk in chunks if chunk["score"] >= CITATION_SCORE_THRESHOLD]
        return Answer(text=draft, domain=domain, confidence=confidence, action="answer", citations=citations)

    if action == "clarify":
        clarifying_question = _generate_clarifying_question(query, chunk_texts)
        return Answer(
            text=clarifying_question, domain=domain, confidence=confidence, action="clarify", citations=[]
        )

    return Answer(text=ESCALATE_MESSAGE, domain=domain, confidence=confidence, action="escalate", citations=[])
