import os
import tempfile
import time

import streamlit as st
from rag.pipeline import EnterpriseRAGPipeline

st.set_page_config(page_title="Enterprise RAG Chatbot", page_icon="🤖", layout="wide")

# ---------- THEME ----------
THEME_CSS = """
<style>
/* App background */
.stApp {
    background: linear-gradient(160deg, #0f172a 0%, #1e1b4b 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #312e81;
}

/* Headings */
h1, h2, h3 {
    color: #c7d2fe !important;
}

/* Chat rows: user messages on the right, bot on the left */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin-left: auto;
    max-width: 75%;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    max-width: 85%;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #1e293b;
    border: 1px solid #4f46e5;
    border-radius: 14px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #14b8a6, #2dd4bf);
    color: #0f172a;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: transform 0.1s ease, box-shadow 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(45, 212, 191, 0.35);
    color: #0f172a;
    border: none;
}

/* Expanders (sources panels) */
[data-testid="stExpander"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background: #1e293b;
    border: 1px dashed #6366f1;
    border-radius: 12px;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_backend():
    return EnterpriseRAGPipeline(vector_store_path="vector_store")


try:
    with st.spinner("Initializing AI Engine..."):
        pipeline = load_backend()
except Exception as e:
    st.error(f"Failed to start the RAG engine: {e}")
    st.stop()


def stream_text(text: str, delay: float = 0.015):
    """Yields words one by one to create a typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def render_web_sources(sources):
    if isinstance(sources, str):
        st.info(sources)
    elif isinstance(sources, list):
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                title = source.get("title", f"Source {i}")
                url = source.get("url", "#")
                snippet = source.get("snippet", "")
                st.markdown(f"**{i}. [{title}]({url})**")
                if snippet:
                    st.caption(f'"{snippet}"')
            elif isinstance(source, str) and source.startswith("http"):
                st.markdown(f"**{i}. [{source}]({source})**")
            else:
                st.write(str(source))


def render_local_sources(sources):
    with st.expander("📚 View Local Sources"):
        for doc in sources:
            st.write(f"- Page {doc.metadata.get('page', 'Unknown')}")


def render_web_sources_expander(sources):
    with st.expander("🌐 View Web Sources for Fact-Checking"):
        render_web_sources(sources)


# --- SIDEBAR UI ---
with st.sidebar:
    st.header("📄 Upload Knowledge")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Chunking, Embedding, and Indexing..."):
                total_chunks_added = 0
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    try:
                        chunks = pipeline.ingest_new_document(tmp_file_path)
                        total_chunks_added += chunks
                    finally:
                        os.unlink(tmp_file_path)

                st.success(f"✅ Successfully indexed {total_chunks_added} chunks!")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT UI ---
st.title("🤖 Hybrid RAG Chatbot")
st.caption("Upload PDFs in the sidebar, then ask questions. Powered by Groq LPU.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        source_type = msg.get("source_type", "local")

        if source_type == "error":
            st.error(msg["content"])
        else:
            st.write(msg["content"])

        if msg.get("sources"):
            if source_type == "web":
                render_web_sources_expander(msg["sources"])
            elif source_type == "local":
                render_local_sources(msg["sources"])

        if source_type == "web":
            st.info(
                "ℹ️ This answer was generated using external internet search because "
                "the information was not found in your uploaded PDFs."
            )

# --- NEW QUESTION ---
if prompt := st.chat_input("Ask a question about your documents..."):
    if not pipeline.faiss_store.docs:
        st.warning("⚠️ Please upload and process at least one PDF first.")
    else:
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("🧠 Processing query...", expanded=True) as status:
                st.write("Searching local knowledge base...")

                recent_history = st.session_state.messages[-4:]
                history_str = ""
                for m in recent_history:
                    role = "User" if m["role"] == "user" else "Assistant"
                    history_str += f"{role}: {m['content']}\n"

                answer, sources, source_type = pipeline.ask(
                    prompt,
                    chat_history=history_str,
                )

                if source_type == "web":
                    st.write("🌐 Local knowledge insufficient. Searching the web...")
                    status.update(
                        label="Response generated via Web Search!",
                        state="complete",
                        expanded=False,
                    )
                elif source_type == "local":
                    status.update(
                        label="Response generated via Local PDFs!",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(
                        label="Request failed",
                        state="error",
                        expanded=False,
                    )

            if source_type == "error":
                st.error(answer)
            else:
                st.write_stream(stream_text(answer))

                if sources:
                    if source_type == "web":
                        render_web_sources_expander(sources)
                    else:
                        render_local_sources(sources)

                if source_type == "web":
                    st.info(
                        "ℹ️ This answer was generated using external internet search because "
                        "the information was not found in your uploaded PDFs."
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "source_type": source_type,
        })