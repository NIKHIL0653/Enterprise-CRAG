"""Unit tests for multi-turn conversational memory and query condensation."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agent.nodes import (
    _build_context,
    generate_answer_node,
    intent_route_node,
    retrieve_node,
)
from backend.core.config import get_settings
from backend.models.document import DocumentChunk
from backend.models.query import ChatMessage, QueryRequest, RAGState

settings = get_settings()


def test_query_request_with_history() -> None:
    """Ensure QueryRequest validates multi-turn conversation history."""
    req = QueryRequest(
        question="Can it be used for an ergonomic chair?",
        user_email="alice@acme.com",
        history=[
            ChatMessage(role="user", content="What is our remote work stipend?"),
            ChatMessage(role="assistant", content="Employees receive $500 for home office setup."),
        ],
    )
    assert len(req.history) == 2
    assert req.history[0].role == "user"
    assert "remote work" in req.history[0].content


@pytest.mark.asyncio
async def test_intent_route_bypasses_greeting_without_history() -> None:
    """Pure greeting without history should immediately return canned greeting."""
    state: RAGState = {
        "question": "hello",
        "user_email": "user@acme.com",
    }
    result = await intent_route_node(state)
    assert "WorkspaceRAG" in result.get("answer", "")
    assert result.get("is_relevant") is True


@pytest.mark.asyncio
async def test_intent_route_condenses_query_with_history() -> None:
    """When history is present, intent_route should condense ambiguous question into standalone search query."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Can the $500 remote work equipment stipend be used for an ergonomic desk chair?"
    mock_response.usage_metadata = MagicMock(prompt_token_count=35, candidates_token_count=18, total_token_count=53)
    mock_llm.generate_content_async = AsyncMock(return_value=mock_response)

    state: RAGState = {
        "question": "Can it be used for an ergonomic chair?",
        "user_email": "user@acme.com",
        "history": [
            {"role": "user", "content": "What is our remote work stipend?"},
            {"role": "assistant", "content": "Employees receive $500 for home office equipment."},
        ],
    }

    result = await intent_route_node(state, llm_client=mock_llm)
    assert result["question"] == "Can it be used for an ergonomic chair?"
    assert "remote work" in result["standalone_question"]
    assert result["token_usage"]["total_tokens"] == 53
    mock_llm.generate_content_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_node_uses_standalone_question() -> None:
    """Retrieve node should use standalone_question when available in state."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])

    state: RAGState = {
        "question": "Can it be used for that?",
        "standalone_question": "Can remote work stipend be used for ergonomic desk chair?",
        "user_email": "alice@acme.com",
    }

    await retrieve_node(state, retriever=mock_retriever, reranker=mock_reranker)

    mock_retriever.retrieve.assert_awaited_once_with(
        query="Can remote work stipend be used for ergonomic desk chair?",
        user_email="alice@acme.com",
        top_k=settings.TOP_K_RETRIEVAL,
        doc_ids=None,
    )


@pytest.mark.asyncio
async def test_generate_answer_injects_prior_conversation() -> None:
    """Ensure prior dialogue turns are prepended to the prompt in generate_answer_node."""
    now = datetime.now(timezone.utc)
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Yes, ergonomic chairs are covered under the remote stipend."
    mock_response.usage_metadata = MagicMock(prompt_token_count=40, candidates_token_count=15, total_token_count=55)
    mock_llm.generate_content_async = AsyncMock(return_value=mock_response)

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        doc_title="Remote Policy",
        source_url="http://drive/doc1",
        section_header="Home Office Equipment",
        content="Desks and ergonomic chairs qualify under the $500 stipend.",
        token_count=20,
        chunk_index=0,
        allowed_emails=["alice@acme.com"],
        allowed_domain="",
        created_at=now,
        updated_at=now,
    )

    state: RAGState = {
        "question": "Can it be used for an ergonomic chair?",
        "user_email": "alice@acme.com",
        "relevant_docs": [chunk],
        "history": [
            {"role": "user", "content": "What is our remote work stipend?"},
            {"role": "assistant", "content": "Employees get $500 annually."},
        ],
    }

    result = await generate_answer_node(state, llm_client=mock_llm)
    assert "ergonomic chairs" in result["answer"]
    # Verify the prompt sent to Gemini contained Prior Conversation
    call_args = mock_llm.generate_content_async.call_args[0][0]
    assert "### Prior Conversation" in call_args
    assert "What is our remote work stipend?" in call_args


def test_build_context_merges_contiguous_siblings() -> None:
    """Ensure contiguous chunks from the same document section are smoothly concatenated."""
    now = datetime.now(timezone.utc)
    chunk_0 = DocumentChunk(
        chunk_id="c_0",
        doc_id="doc_handbook",
        doc_title="Handbook",
        source_url="http://handbook",
        section_header="Benefits",
        content="First paragraph of benefits.",
        token_count=10,
        chunk_index=0,
        allowed_emails=["alice@acme.com"],
        created_at=now,
        updated_at=now,
    )
    chunk_1 = DocumentChunk(
        chunk_id="c_1",
        doc_id="doc_handbook",
        doc_title="Handbook",
        source_url="http://handbook",
        section_header="Benefits",
        content="Second contiguous paragraph of benefits.",
        token_count=10,
        chunk_index=1,
        allowed_emails=["alice@acme.com"],
        created_at=now,
        updated_at=now,
    )
    chunk_other = DocumentChunk(
        chunk_id="c_other",
        doc_id="doc_security",
        doc_title="Security",
        source_url="http://security",
        section_header="MFA",
        content="MFA is mandatory for all accounts.",
        token_count=10,
        chunk_index=0,
        allowed_emails=["alice@acme.com"],
        created_at=now,
        updated_at=now,
    )

    # Pass in reverse order to ensure sorting and contiguous stitching works
    context = _build_context([chunk_1, chunk_other, chunk_0])

    # Handbook chunks should be merged into a single block with one header
    assert context.count("[Doc: Handbook, §Benefits]") == 1
    assert "First paragraph of benefits.\n\nSecond contiguous paragraph of benefits." in context

    # Security chunk is kept separate with ---
    assert "[Doc: Security, §MFA]" in context
    assert "\n\n---\n\n" in context

