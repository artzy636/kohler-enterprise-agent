"""Streamlit chat UI for the KOHLER enterprise knowledge assistant."""

import json
from datetime import datetime

import streamlit as st

from src.formatters import DOMAIN_LABELS, to_email_draft, to_excel, to_json, to_xml
from src.pipeline import Answer, answer_query

MAX_HISTORY_MESSAGES = 8

OUTPUT_FORMATS = ["Plain Text", "JSON", "Excel", "XML", "Email Draft"]

CONFIDENCE_COLORS = {
    "high": "#2e7d32",
    "medium": "#f9a825",
    "low": "#c62828",
}

BADGE_STYLE = (
    "display:inline-block; padding:0.2rem 0.7rem; border-radius:999px; "
    "font-size:0.75rem; font-weight:600; letter-spacing:0.02em; color:#FFFFFF; "
    "background-color:{color};"
)

st.set_page_config(page_title="KOHLER Enterprise Assistant", page_icon="🚰", layout="centered")

st.markdown(
    """
    <div style="padding:1.25rem 0 1rem 0; border-bottom:1px solid #E0E0E0; margin-bottom:1.5rem;">
        <div style="color:#212121; font-weight:700; font-size:1.6rem; letter-spacing:0.12em;">KOHLER</div>
        <div style="color:#616161; font-size:0.95rem; letter-spacing:0.03em;">Enterprise Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def _to_gemini_history(messages: list[dict]) -> list[dict]:
    """Convert stored chat messages to Gemini's {"role", "parts"} format."""
    history = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        role = "model" if message["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [{"text": message["content"]}]})
    return history


def _render_status_badge(answer: Answer) -> None:
    if answer.action == "escalate":
        domain_label = DOMAIN_LABELS.get(answer.domain, answer.domain.title())
        label = f"Escalated to {domain_label} team"
        color = "#757575"
    elif answer.action == "clarify":
        label = "Needs clarification"
        color = "#1565C0"
    else:
        label = f"{answer.confidence.title()} Confidence"
        color = CONFIDENCE_COLORS.get(answer.confidence, "#757575")

    st.markdown(
        f'<span style="{BADGE_STYLE.format(color=color)}">{label}</span>',
        unsafe_allow_html=True,
    )


def _render_answer(answer: Answer, output_format: str, widget_key: str) -> None:
    _render_status_badge(answer)
    st.write("")

    if output_format == "Plain Text":
        st.markdown(answer.text)
    elif output_format == "JSON":
        pretty = json.dumps(json.loads(to_json(answer)), indent=2)
        st.code(pretty, language="json")
    elif output_format == "XML":
        st.code(to_xml(answer), language="xml")
    elif output_format == "Email Draft":
        st.code(to_email_draft(answer), language=None)
    elif output_format == "Excel":
        st.caption("Excel export ready for download.")
        st.download_button(
            label="Download Excel",
            data=to_excel(answer),
            file_name="kohler-answer.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=widget_key,
        )


output_format = st.selectbox("Output format", OUTPUT_FORMATS, key="output_format")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_answer(message["answer"], output_format, widget_key=f"excel_dl_{idx}")
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    history = _to_gemini_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = answer_query(prompt, history)
            except Exception as exc:
                st.error(f"Something went wrong while generating a response: {exc}")
                answer = None

        if answer is not None:
            _render_answer(answer, output_format, widget_key=f"excel_dl_new_{datetime.now().timestamp()}")
            st.session_state.messages.append({"role": "assistant", "content": answer.text, "answer": answer})

    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
