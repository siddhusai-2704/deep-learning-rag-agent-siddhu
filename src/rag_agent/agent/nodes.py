"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_state_value(state, key, default=None):
    """
    Safely read a value from either a dictionary-style state
    or an object-style state.
    """
    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.
    """

    original_query = ""

    try:
        messages = _get_state_value(state, "messages", [])

        # Find the latest human message.
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                original_query = message.content
                break

        if not original_query:
            return {
                "original_query": "",
                "rewritten_query": "",
            }

        settings = get_settings()
        llm = LLMFactory(settings).create()

        rewrite_prompt = (
            "Rewrite the following user question into a short, "
            "keyword-focused query for vector search. "
            "Keep the original meaning. "
            "Return only the rewritten search query.\n\n"
            f"User question: {original_query}"
        )

        result = llm.invoke(
            [HumanMessage(content=rewrite_prompt)]
        )

        rewritten_query = str(result.content).strip()

        if not rewritten_query:
            rewritten_query = original_query

        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
        }

    except Exception:
        # If rewriting fails, continue using the original question.
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.
    """

    try:
        rewritten_query = _get_state_value(
            state,
            "rewritten_query",
            "",
        )

        topic_filter = _get_state_value(
            state,
            "topic_filter",
            None,
        )

        difficulty_filter = _get_state_value(
            state,
            "difficulty_filter",
            None,
        )

        if not rewritten_query:
            return {
                "retrieved_chunks": [],
                "no_context_found": True,
            }

        manager = VectorStoreManager()

        chunks = manager.query(
            query_text=rewritten_query,
            topic_filter=topic_filter,
            difficulty_filter=difficulty_filter,
        )

        if not chunks:
            return {
                "retrieved_chunks": [],
                "no_context_found": True,
            }

        return {
            "retrieved_chunks": chunks,
            "no_context_found": False,
        }

    except Exception:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.
    """

    settings = get_settings()
    llm = LLMFactory(settings).create()

    no_context_found = _get_state_value(
        state,
        "no_context_found",
        False,
    )

    rewritten_query = _get_state_value(
        state,
        "rewritten_query",
        "",
    )

    original_query = _get_state_value(
        state,
        "original_query",
        "",
    )

    # -----------------------------------------------------------------------
    # Hallucination Guard
    # -----------------------------------------------------------------------

    if no_context_found:
        no_context_message = (
            "I was unable to find relevant information in the corpus "
            "for your query. This may mean the topic is not yet covered "
            "in the study material, or your query may need to be "
            "rephrased. Please try a more specific deep learning topic "
            "such as 'LSTM forget gate' or 'CNN pooling layers'."
        )

        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=rewritten_query,
        )

        return {
            "final_response": response,
            "messages": [
                AIMessage(content=no_context_message)
            ],
        }

    # -----------------------------------------------------------------------
    # Retrieved Context
    # -----------------------------------------------------------------------

    retrieved_chunks = _get_state_value(
        state,
        "retrieved_chunks",
        [],
    )

    if not retrieved_chunks:
        no_context_message = (
            "I was unable to find relevant information in the corpus "
            "for your query."
        )

        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=rewritten_query,
        )

        return {
            "final_response": response,
            "messages": [
                AIMessage(content=no_context_message)
            ],
        }

    context_parts = []
    sources = []
    scores = []

    for chunk in retrieved_chunks:
        # RetrievedChunk fields may be attributes or dictionary values.
        if isinstance(chunk, dict):
            chunk_text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            metadata = chunk.get("metadata", {})
        else:
            chunk_text = getattr(chunk, "text", "")
            score = getattr(chunk, "score", 0.0)
            metadata = getattr(chunk, "metadata", {})

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            try:
                metadata = vars(metadata)
            except Exception:
                metadata = {}

        topic = metadata.get(
            "topic",
            metadata.get("topic_name", "Unknown topic"),
        )

        filename = metadata.get(
            "filename",
            metadata.get(
                "file_name",
                metadata.get("source", "Unknown file"),
            ),
        )

        context_parts.append(
            f"[SOURCE: {topic} | {filename}]\n"
            f"{chunk_text}\n"
        )

        source_label = f"{topic} | {filename}"

        if source_label not in sources:
            sources.append(source_label)

        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            pass

    context = "\n".join(context_parts)

    # Calculate average retrieval confidence.
    if scores:
        confidence = sum(scores) / len(scores)
    else:
        confidence = 0.0

    # Keep confidence within 0 and 1.
    confidence = max(0.0, min(1.0, confidence))

    # -----------------------------------------------------------------------
    # Conversation History
    # -----------------------------------------------------------------------

    conversation_messages = _get_state_value(
        state,
        "messages",
        [],
    )

    try:
        trimmed_messages = trim_messages(
            conversation_messages,
            max_tokens=settings.max_context_tokens,
            strategy="last",
            token_counter=len,
            include_system=True,
            allow_partial=False,
        )
    except Exception:
        trimmed_messages = conversation_messages[-6:]

    # -----------------------------------------------------------------------
    # Build Prompt
    # -----------------------------------------------------------------------

    context_message = (
        "Use ONLY the information in the following retrieved context "
        "to answer the user's question.\n\n"
        "Retrieved context:\n"
        f"{context}"
    )

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=context_message),
    ]

    if trimmed_messages:
        prompt_messages.extend(trimmed_messages)

    prompt_messages.append(
        HumanMessage(content=original_query)
    )

    # -----------------------------------------------------------------------
    # Generate Answer
    # -----------------------------------------------------------------------

    result = llm.invoke(prompt_messages)

    answer = str(result.content).strip()

    if not answer:
        answer = (
            "I could not generate an answer from the retrieved "
            "study material."
        )

    ai_message = AIMessage(content=answer)

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=rewritten_query,
    )

    return {
        "final_response": response,
        "messages": [ai_message],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Decide whether retrieval succeeded.

    If relevant context was found, continue to generation.
    If no context was found, end the graph so the hallucination
    guard can return the no-context response.
    """

    no_context_found = _get_state_value(
        state,
        "no_context_found",
        False,
    )

    if no_context_found:
        return "end"

    return "generate"