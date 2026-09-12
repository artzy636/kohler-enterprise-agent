"""Gemini-powered agent functions: domain classification, answer drafting, and verification.

Reads GEMINI_API_KEY from a .env file via python-dotenv.
"""

import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

FLASH_LITE_MODEL = "gemini-3.5-flash-lite"

MIN_CALL_INTERVAL_SECONDS = 4.2
_last_call_time = None


def _rate_limit() -> None:
    """Block until at least MIN_CALL_INTERVAL_SECONDS have passed since the last API call."""
    global _last_call_time
    now = time.monotonic()
    if _last_call_time is not None:
        remaining = MIN_CALL_INTERVAL_SECONDS - (now - _last_call_time)
        if remaining > 0:
            time.sleep(remaining)
    _last_call_time = time.monotonic()

DOMAINS = ["HR", "FINANCE", "SUPPORT", "LEGAL", "OTHER"]

CLASSIFY_DOMAIN_SYSTEM_PROMPT = """You are a routing classifier for an enterprise support assistant.

Classify the user's query into exactly one of the following domains:
- HR: employee benefits, leave, onboarding, performance reviews, workplace conduct, compensation
- FINANCE: expenses, budgets, invoices, procurement, payments, reimbursement, tax
- SUPPORT: product setup, troubleshooting, warranty, returns, device pairing
- LEGAL: contracts, compliance, data privacy, intellectual property, regulatory matters
- OTHER: anything that does not clearly fit HR, FINANCE, SUPPORT, or LEGAL

Respond with exactly one word: HR, FINANCE, SUPPORT, LEGAL, or OTHER.
Do not include punctuation, explanation, or any other text."""

DRAFT_ANSWER_SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the user's question using \
ONLY the information contained in the provided context chunks. Do not use outside knowledge and do not guess.

The conversation history may contain earlier turns in this exchange - use it to resolve follow-up questions \
(e.g. pronouns, "what about...", implied subjects) so your answer addresses what the user actually means.

If the context does not contain enough information to answer confidently, respond exactly with:
"I don't have enough information to answer confidently"
followed by a brief note on what information is missing, rather than guessing or inferring beyond the context.

If the answer depends on a variable (such as role, contract type, or category) and the context enumerates \
the specific values for each case, answer by listing all the relevant cases rather than asking the user to \
specify or declining to answer. Only treat a question as needing clarification when the missing information \
is about the user's own specific situation (e.g. which product they own, their exact travel destination, \
their purchase date) and is not itself enumerable from the context.

When you do answer, be concise. Write in clean prose without bracketed citations, chunk numbers, \
or other inline reference markers - do not mention "Chunk" or similar labels in your answer."""

VERIFY_ANSWER_SYSTEM_PROMPT = """You are a quality-control reviewer for an enterprise knowledge assistant. \
You will be given a draft answer and the source context chunks it was supposed to be based on.

Evaluate whether the draft is fully grounded in the source chunks (no unsupported claims, no facts absent \
from the context) and decide how confident and actionable the draft is.

Important rule: if the draft itself states that it doesn't have enough information, cannot answer \
confidently, or otherwise contains no substantive answer to the question, then action MUST be "escalate" - \
regardless of how well-reasoned or appropriate that refusal is. The "answer" action is only for drafts that \
provide an actual grounded answer to the question, never for drafts that correctly decline to answer. A \
well-written refusal is still not a confident answer.

"grounded" must be true only when action is "answer" and the draft provides a real, evidence-backed \
response supported by the source chunks. In every other case - "escalate" (whether because the draft \
refused to answer or because it contained ungrounded content) or "clarify" - "grounded" must be false, \
since there is no substantive answer being made that could be evaluated as grounded.

Respond with ONLY a single valid JSON object (no markdown formatting, no surrounding text) with exactly \
these fields:
{
  "grounded": <true only if action is "answer" and the draft is a real answer fully supported by the \
source chunks; false in every other case, including "escalate" and "clarify">,
  "confidence": <"high", "medium", or "low">,
  "action": <"answer" if the draft provides an actual grounded answer and is ready to send as-is, "clarify" \
if the user's question needs clarification before a confident answer is possible, or "escalate" if this \
should go to a human - including whenever the draft itself declines to answer or says it lacks enough \
information>,
  "reasoning": <a brief string explaining the evaluation>
}"""


def classify_domain(query: str) -> str:
    """Classify a query into HR, FINANCE, SUPPORT, LEGAL, or OTHER using Gemini Flash-Lite."""
    _rate_limit()
    response = client.models.generate_content(
        model=FLASH_LITE_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=CLASSIFY_DOMAIN_SYSTEM_PROMPT,
            max_output_tokens=16,
        ),
    )
    label = response.text.strip().upper()
    return label if label in DOMAINS else "OTHER"


def draft_answer(query: str, chunks: list[str], history: list[dict]) -> str:
    """Draft an answer to `query` grounded only in `chunks`, using Gemini Flash-Lite."""
    context = "\n\n".join(f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunks, start=1))
    user_turn = f"Context:\n{context}\n\nQuestion: {query}"

    _rate_limit()
    response = client.models.generate_content(
        model=FLASH_LITE_MODEL,
        contents=[*history, {"role": "user", "parts": [{"text": user_turn}]}],
        config=types.GenerateContentConfig(
            system_instruction=DRAFT_ANSWER_SYSTEM_PROMPT,
            max_output_tokens=16000,
            temperature=0.2,
        ),
    )
    return response.text


def verify_answer(draft: str, chunks: list[str]) -> dict:
    """Verify `draft` against `chunks` using Gemini Flash-Lite, returning a parsed JSON dict."""
    context = "\n\n".join(f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunks, start=1))
    user_turn = f"Source chunks:\n{context}\n\nDraft answer:\n{draft}"

    _rate_limit()
    response = client.models.generate_content(
        model=FLASH_LITE_MODEL,
        contents=user_turn,
        config=types.GenerateContentConfig(
            system_instruction=VERIFY_ANSWER_SYSTEM_PROMPT,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    text = response.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "grounded": False,
            "confidence": "low",
            "action": "escalate",
            "reasoning": "Verification response could not be parsed; escalating as a safe default.",
        }
