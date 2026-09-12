"""Render an Answer into various output formats: JSON, Excel, XML, and a plain-text email draft."""

import json
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd

from src.pipeline import Answer

DOMAIN_LABELS = {
    "HR": "HR",
    "FINANCE": "Finance",
    "SUPPORT": "Support",
    "LEGAL": "Legal",
    "OTHER": "General",
}


def to_json(answer: Answer) -> str:
    """Return a JSON string of the Answer's fields."""
    return json.dumps({
        "text": answer.text,
        "domain": answer.domain,
        "confidence": answer.confidence,
        "citations": answer.citations,
    })


def to_excel(answer: Answer) -> bytes:
    """Build a single-row DataFrame from the Answer's fields and return it as xlsx bytes."""
    df = pd.DataFrame([{
        "text": answer.text,
        "domain": answer.domain,
        "confidence": answer.confidence,
        "citations": ", ".join(answer.citations),
    }])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Answer")
    return buffer.getvalue()


def to_xml(answer: Answer) -> str:
    """Return a simple XML string with each field as a tag."""
    citations_xml = "".join(f"<citation>{escape(c)}</citation>" for c in answer.citations)
    return (
        "<answer>"
        f"<text>{escape(answer.text)}</text>"
        f"<domain>{escape(answer.domain)}</domain>"
        f"<confidence>{escape(answer.confidence)}</confidence>"
        f"<citations>{citations_xml}</citations>"
        "</answer>"
    )


def to_email_draft(answer: Answer) -> str:
    """Return a formatted plain-text email with a subject, greeting, body, and citations line."""
    domain_label = DOMAIN_LABELS.get(answer.domain, answer.domain.title())
    lines = [
        f"Subject: {domain_label} Inquiry - Response",
        "",
        "Hello,",
        "",
        answer.text,
    ]
    if answer.citations:
        lines.append("")
        lines.append(f"Sources: {', '.join(answer.citations)}")
    lines.append("")
    lines.append("Best regards,")
    lines.append("Enterprise Knowledge Assistant")
    return "\n".join(lines)
