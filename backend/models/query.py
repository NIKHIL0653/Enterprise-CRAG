"""Pydantic v2 models for query requests, responses, and LangGraph state.

These models are used by:

* The FastAPI ``/query`` endpoint to validate incoming requests and shape
  outgoing responses.
* The LangGraph RAG pipeline to pass typed state between nodes.

Example::

    from backend.models.query import QueryRequest, QueryResponse

    req = QueryRequest(
        question="What was discussed in the last board meeting?",
        user_email="alice@acme.com",
    )
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.models.document import DocumentChunk


class QueryRequest(BaseModel):
    """Validated inbound query from an authenticated user.

    Attributes:
        question: The natural-language question to answer.  Must be between
            5 and 2 000 characters.
        user_email: Email address of the requesting user.  Used for ACL
            enforcement during retrieval.
        session_id: Optional opaque identifier for a conversation session.
            If omitted, the backend generates a fresh UUID for the response.
    """

    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural-language question to answer.",
    )
    user_email: EmailStr = Field(
        ...,
        description="Authenticated user's email address (used for ACL checks).",
    )
    session_id: str | None = Field(
        default=None,
        description="Existing conversation session ID, or None to start a new one.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        """Reject questions that consist solely of whitespace.

        Args:
            v: The raw ``question`` value provided by the caller.

        Returns:
            The stripped question string.

        Raises:
            ValueError: If the stripped question is empty.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be blank or contain only whitespace.")
        return stripped


class Citation(BaseModel):
    """A reference to a source document chunk included in a RAG answer.

    Citations allow the front-end to render clickable links back to the
    original Google Drive documents, with an excerpt shown as a preview.

    Attributes:
        doc_id: Identifier of the source document.
        doc_title: Human-readable title of the source document.
        source_url: Canonical URL to the Google Drive file.
        section_header: Section heading that precedes the cited chunk, if any.
        excerpt: Up to the first 200 characters of the cited chunk's content,
            used as a hover-preview in the UI.
    """

    doc_id: str
    doc_title: str
    source_url: str
    section_header: str | None = None
    excerpt: str = Field(
        ...,
        max_length=200,
        description="First ≤200 characters of the cited chunk content.",
    )

    @field_validator("excerpt", mode="before")
    @classmethod
    def truncate_excerpt(cls, v: str) -> str:
        """Ensure the excerpt does not exceed 200 characters.

        Args:
            v: The raw excerpt string.

        Returns:
            The excerpt, truncated to 200 characters if necessary.
        """
        return v[:200]


class QueryResponse(BaseModel):
    """Structured response returned by the ``/query`` endpoint.

    Attributes:
        answer: The generated natural-language answer.
        citations: Ordered list of source citations supporting the answer.
        faithfulness_score: Optional score (0–1) indicating how faithfully the
            answer is grounded in the retrieved context.  ``None`` if the
            faithfulness check was skipped.
        retrieval_latency_ms: Wall-clock time (ms) spent on retrieval + rerank.
        generation_latency_ms: Wall-clock time (ms) spent on LLM generation.
        session_id: Session identifier (echoed from request or newly created).
        trace_url: Optional URL to the Langfuse trace for this request,
            enabling one-click observability drill-down.
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    faithfulness_score: float | None = None
    retrieval_latency_ms: float
    generation_latency_ms: float
    session_id: str
    trace_url: str | None = None


class RAGState(TypedDict, total=False):
    """Mutable state dictionary threaded through the LangGraph RAG pipeline.

    Each LangGraph node reads fields it needs and writes the fields it
    produces.  Using ``total=False`` means all keys are optional by default,
    which prevents ``KeyError`` when a node hasn't yet populated a field.

    Keys:
        question: The original user question string.
        user_email: Authenticated user's email (passed through for ACL checks).
        retrieved_docs: Candidate :class:`~backend.models.document.DocumentChunk`
            objects returned by the vector-store + BM25 retrieval step.
        relevant_docs: Subset of ``retrieved_docs`` that passed the Cohere
            reranking and relevance-threshold filter.
        answer: The final generated answer string.
        citations: :class:`Citation` objects built from ``relevant_docs``.
        is_relevant: ``True`` if ``relevant_docs`` is non-empty and deemed
            sufficient to answer the question.
        is_faithful: ``True`` if the faithfulness self-check passed.
        trace_id: Langfuse trace ID for observability correlation.
        error: Human-readable error message if any node raised an exception;
            ``None`` on the happy path.
    """

    question: str
    user_email: str
    retrieved_docs: list[DocumentChunk]
    relevant_docs: list[DocumentChunk]
    answer: str
    citations: list[Citation]
    is_relevant: bool
    is_faithful: bool
    trace_id: str
    error: str | None
