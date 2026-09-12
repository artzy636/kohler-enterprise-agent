"""Gemini-powered agent functions: domain classification, answer drafting, and verification.

Reads GEMINI_API_KEY from a .env file via python-dotenv.
"""

import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

FLASH_LITE_MODEL = "gemini-2.5-flash-lite"
FLASH_MODEL = "gemini-2.5-flash"

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

When you do answer, be concise and cite which part of the context supports your answer where useful."""

VERIFY_ANSWER_SYSTEM_PROMPT = """You are a quality-control reviewer for an enterprise knowledge assistant. \
You will be given a draft answer and the source context chunks it was supposed to be based on.

Evaluate whether the draft is fully grounded in the source chunks (no unsupported claims, no facts absent \
from the context) and decide how confident and actionable the draft is.

Respond with ONLY a single valid JSON object (no markdown formatting, no surrounding text) with exactly \
these fields:
{
  "grounded": <true or false - whether every claim in the draft is supported by the source chunks>,
  "confidence": <"high", "medium", or "low">,
  "action": <"answer" if the draft is ready to send as-is, "clarify" if the user's question needs \
clarification before a confident answer is possible, or "escalate" if this should go to a human>,
  "reasoning": <a brief string explaining the evaluation>
}"""


def classify_domain(query: str) -> str:
    """Classify a query into HR, FINANCE, SUPPORT, LEGAL, or OTHER using Gemini Flash-Lite."""
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
    """Draft an answer to `query` grounded only in `chunks`, using Gemini Flash."""
    context = "\n\n".join(f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunks, start=1))
    user_turn = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[*history, {"role": "user", "parts": [{"text": user_turn}]}],
        config=types.GenerateContentConfig(
            system_instruction=DRAFT_ANSWER_SYSTEM_PROMPT,
            max_output_tokens=16000,
        ),
    )
    return response.text


def verify_answer(draft: str, chunks: list[str]) -> dict:
    """Verify `draft` against `chunks` using Gemini Flash-Lite, returning a parsed JSON dict."""
    context = "\n\n".join(f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunks, start=1))
    user_turn = f"Source chunks:\n{context}\n\nDraft answer:\n{draft}"

    response = client.models.generate_content(
        model=FLASH_LITE_MODEL,
        contents=user_turn,
        config=types.GenerateContentConfig(
            system_instruction=VERIFY_ANSWER_SYSTEM_PROMPT,
            max_output_tokens=1024,
        ),
    )
    text = response.text.strip()
    return json.loads(text)
