"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the singleton VectorStoreManager."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def initialise_session_state() -> None:
    """Initialise Streamlit session-state values."""
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel
# ---------------------------------------------------------------------------

def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Render document upload and ingestion controls."""

    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "📥 Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_files:
        with st.sidebar.status("Processing documents...", expanded=True):

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                file_paths = []

                for uploaded_file in uploaded_files:
                    file_path = temp_path / uploaded_file.name
                    file_path.write_bytes(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                try:
                    chunks = chunker.chunk_files(file_paths)

                    if not chunks:
                        st.sidebar.warning(
                            "No chunks were created from the uploaded files."
                        )
                    else:
                        result = store.ingest(chunks)
                        st.session_state.last_ingestion_result = result

                        st.sidebar.success(
                            f"{result.ingested} chunks added, "
                            f"{result.skipped} duplicates skipped"
                        )

                        if getattr(result, "errored", 0) > 0:
                            st.sidebar.error(
                                f"{result.errored} chunks had errors."
                            )

                except Exception as exc:
                    st.sidebar.error(f"Ingestion failed: {exc}")

        st.session_state.ingested_documents = store.list_documents()

    # Refresh document list even after a rerun
    st.session_state.ingested_documents = store.list_documents()

    st.sidebar.subheader("📚 Ingested Documents")

    documents = st.session_state.ingested_documents

    if not documents:
        st.sidebar.caption("No documents ingested yet.")
        return

    for index, document in enumerate(documents):
        source = document["source"]
        topic = document.get("topic", "")
        chunk_count = document.get("chunk_count", 0)

        st.sidebar.write(
            f"**{source}**  \n"
            f"Topic: {topic}  \n"
            f"Chunks: {chunk_count}"
        )

        if st.sidebar.button(
            "🗑 Remove",
            key=f"delete_{index}_{source}",
            use_container_width=True,
        ):
            deleted = store.delete_document(source)

            st.sidebar.success(
                f"Removed {deleted} chunks from {source}."
            )

            if st.session_state.selected_document == source:
                st.session_state.selected_document = None

            st.session_state.ingested_documents = store.list_documents()
            st.rerun()


# ---------------------------------------------------------------------------
# Corpus Statistics
# ---------------------------------------------------------------------------

def render_corpus_stats(store: VectorStoreManager) -> None:
    """Render corpus health information."""

    stats = store.get_collection_stats()

    st.sidebar.divider()
    st.sidebar.subheader("📊 Corpus Health")

    st.sidebar.metric(
        "Total Chunks",
        stats["total_chunks"],
    )

    topics = stats.get("topics", [])

    if topics:
        st.sidebar.write("**Topics:**")
        st.sidebar.caption(", ".join(topics))
    else:
        st.sidebar.caption("Topics: None")

    if stats.get("bonus_topics_present"):
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer
# ---------------------------------------------------------------------------

def render_document_viewer(store: VectorStoreManager) -> None:
    """Render the document viewer."""

    st.subheader("📄 Document Viewer")

    documents = st.session_state.ingested_documents

    if not documents:
        st.info(
            "No documents have been ingested yet. "
            "Upload a Markdown study note from the sidebar."
        )
        return

    sources = [document["source"] for document in documents]

    selected = st.selectbox(
        "Select document",
        sources,
        index=(
            sources.index(st.session_state.selected_document)
            if st.session_state.selected_document in sources
            else 0
        ),
    )

    st.session_state.selected_document = selected

    chunks = store.get_document_chunks(selected)

    st.caption(f"{len(chunks)} chunks in **{selected}**")

    with st.container(height=500):
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata

            topic = getattr(metadata, "topic", "")
            difficulty = getattr(metadata, "difficulty", "")
            chunk_type = getattr(metadata, "type", "")

            st.markdown(
                f"**Chunk {index}**  \n"
                f"`{topic}` · `{difficulty}` · `{chunk_type}`"
            )

            st.markdown(chunk.chunk_text)

            if index < len(chunks):
                st.divider()


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------

def render_chat_interface(graph) -> None:
    """Render the RAG chat interface."""

    st.subheader("💬 Interview Prep Chat")

    # ---------------------------------------------------------------
    # Filters
    # ---------------------------------------------------------------

    col_topic, col_diff = st.columns(2)

    with col_topic:
        topic_options = [
            "All",
            "CNN",
            "RNN",
            "LSTM",
            "Transformer",
            "GAN",
            "SOM",
            "BoltzmannMachine",
        ]

        selected_topic = st.selectbox(
            "Topic",
            topic_options,
        )

        st.session_state.topic_filter = (
            None if selected_topic == "All" else selected_topic
        )

    with col_diff:
        difficulty_options = [
            "All",
            "beginner",
            "intermediate",
            "advanced",
        ]

        selected_difficulty = st.selectbox(
            "Difficulty",
            difficulty_options,
        )

        st.session_state.difficulty_filter = (
            None
            if selected_difficulty == "All"
            else selected_difficulty
        )

    # ---------------------------------------------------------------
    # Chat History
    # ---------------------------------------------------------------

    chat_container = st.container(height=400)

    with chat_container:
        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)

                if message.get("no_context_found"):
                    st.warning(
                        "⚠️ No relevant content found in corpus."
                    )

    # ---------------------------------------------------------------
    # Chat Input
    # ---------------------------------------------------------------

    query = st.chat_input(
        "Ask about a deep learning topic..."
    )

    if not query:
        return

    # Add user message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    filters = {
        "topic": st.session_state.topic_filter,
        "difficulty": st.session_state.difficulty_filter,
    }

    graph_input = {
        "messages": [HumanMessage(content=query)],
        "filters": filters,
    }

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    try:
        with st.spinner("Searching the corpus and generating answer..."):

            result = graph.invoke(
                graph_input,
                config=config,
            )

        response = result.get("final_response")

        if response is None:
            response = result.get("answer", "No answer was returned.")

        # Convert response to string if necessary
        response_text = str(response)

        # Try to obtain sources from graph result
        sources = result.get("sources", [])

        if not sources:
            retrieved_chunks = result.get("retrieved_chunks", [])

            for chunk in retrieved_chunks:
                metadata = getattr(chunk, "metadata", None)

                if metadata:
                    source = getattr(
                        metadata,
                        "source",
                        "Unknown source",
                    )

                    if source not in sources:
                        sources.append(source)

        no_context_found = (
            "no relevant" in response_text.lower()
            or "no context" in response_text.lower()
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response_text,
                "sources": sources,
                "no_context_found": no_context_found,
            }
        )

    except Exception as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"❌ Error: {type(exc).__name__}: {exc}",
            }
        )

    st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""

    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")

    st.caption(
        "RAG-powered interview preparation — built with "
        "LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main area
    viewer_col, chat_col = st.columns(
        [1, 1],
        gap="large",
    )
    viewer_col, chat_col = st.columns(
        [1, 1],
        gap="large",
    )

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()