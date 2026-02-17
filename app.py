from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = """### System Instructions: RAG & Retriever Expert

Role:
- You are a knowledgeable, precise assistant focused on Retrieval-Augmented Generation (RAG) and retrievers.
- Use a clear, professional, and concise tone (not overly casual or chatty).

Behavior:
1. Understand the user query:
   - Detect whether they are asking about retriever selection, RAG concepts, or implementation details.
   - Infer where possible:
     - Dataset type (text, PDF, CSV, structured DB, knowledge base)
     - Query type (semantic vs keyword vs structured)
     - Goals (accuracy, latency, scalability, multi-domain, cost)

2. When recommending a retriever (or comparing retrievers), you MUST format the main recommendation using EXACTLY the structure below, with no extra bullet markers or headings inside it:

'''
When recommending a retriever:

Recommended Retriever: <retriever name>
Reason:
<why this retriever is best for the project, written as one or more full sentences on the following line(s)>
Secondary Options: 
- <Option 1> (pros/cons)
- <Option 2> (pros/cons)
Implementation Notes: <pre-processing or configuration tips>
'''

- Replace the angle-bracket sections with concrete content.
- You may add short explanation before or after this block if needed, but the block itself must appear exactly once and remain in this format.

3. Answer RAG questions:
   - Explain concepts clearly and technically, keeping the tone neutral and informative.
   - When helpful, briefly mention:
     - RAG architectures, pipelines, and trade-offs
     - Retriever types (dense, sparse, hybrid, BM25, etc.) and their use cases
   - Keep responses focused and avoid unnecessary storytelling or metaphors.

4. General rules:
   - Be explicit about assumptions.
   - Do NOT mention that you are using a PDF, knowledge base, or external document; answer as if using your own domain knowledge.
   - Do NOT include section/page-style citations such as "(Section 3.7.2)", page numbers, or similar references.
   - If the user asks about topics clearly outside RAG / retrievers (for example, general acronyms, unrelated ML concepts, or tooling that is not about retrieval), say plainly that you are a specialist assistant for RAG and retrievers and that this question is outside your scope.
   - If you do not have enough information to answer within that RAG/retriever scope, say so plainly and indicate what additional details you would need, without referring to any PDF or its sections.
   - Prefer concise, structured outputs over long paragraphs."""


APP_TITLE = "Ragify – RAG & Retriever Chatbot"
PDF_PATH = Path(__file__).with_name("RAG_Complete_Knowledge_Base.pdf")
MODEL_NAME = "gemini-2.5-flash"


def _get_gemini_api_key(user_input_key: str) -> str:
    key = (user_input_key or "").strip()
    if key:
        return key
    return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()


@st.cache_resource(show_spinner=False)
def _load_pdf_context_text(pdf_path: Path) -> str:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs).strip()


def _build_langchain_messages(
    chat_history: list[dict[str, Any]],
    user_text: str,
    pdf_context_text: str,
) -> list[Any]:
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(
            content=(
                "You must use the following PDF knowledge base as grounding context for your answers. "
                "If something is not covered there, say so and suggest what info would be needed.\n\n"
                f"PDF knowledge base ({PDF_PATH.name}):\n{pdf_context_text}"
            )
        ),
    ]

    for m in chat_history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_text))
    return messages


def _generate_assistant_reply(api_key: str, messages: list[Any]) -> str:
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )
    reply = model.invoke(messages)
    return (getattr(reply, "content", "") or "").strip()


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)

    # Initialize simple UI state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "pending_user_text" not in st.session_state:
        st.session_state.pending_user_text = None

    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your Google AI Studio / Gemini API key",
            help="Stored only in this session. You can also set GOOGLE_API_KEY in a .env file.",
        )

        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()

        st.caption(f"Model: `{MODEL_NAME}`")

    api_key = _get_gemini_api_key(api_key_input)
    if not api_key:
        st.info("Add your Gemini API key in the sidebar to start chatting.")
        st.stop()

    if not PDF_PATH.exists():
        st.error(f"Missing PDF at `{PDF_PATH}`")
        st.stop()

    with st.spinner("Loading PDF knowledge base..."):
        pdf_context_text = _load_pdf_context_text(PDF_PATH)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Main chat input; disabled while we are generating a response
    user_text = st.chat_input(
        "Ask me anything about RAG and retrievers…",
        disabled=st.session_state.is_generating,
    )
    if user_text:
        # Capture the new message and trigger a fresh run where generation happens
        st.session_state.pending_user_text = user_text
        st.session_state.is_generating = True
        st.rerun()

    pending_text = st.session_state.pending_user_text
    if not pending_text:
        return

    # Render the pending user message
    st.session_state.messages.append({"role": "user", "content": pending_text})
    with st.chat_message("user"):
        st.markdown(pending_text)

    # Generate assistant reply with a visible loader
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            lc_messages = _build_langchain_messages(
                chat_history=st.session_state.messages[:-1],
                user_text=pending_text,
                pdf_context_text=pdf_context_text,
            )
            with st.spinner("Generating response..."):
                assistant_text = _generate_assistant_reply(
                    api_key=api_key,
                    messages=lc_messages,
                )
            if not assistant_text:
                assistant_text = "I couldn’t generate a response. Try rephrasing your question."
        except Exception as exc:  # noqa: BLE001
            assistant_text = f"Something went wrong while calling the model: `{exc}`"

        placeholder.markdown(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    # Clear generation state and re-enable the input box on the next run
    st.session_state.pending_user_text = None
    st.session_state.is_generating = False
    st.rerun()


if __name__ == "__main__":
    main()
