"""LangGraph node functions for the workspace RAG pipeline.

Each node is an ``async`` function that accepts a ``RAGState`` TypedDict (and
bound dependencies via ``functools.partial``) and returns a dictionary of
state keys to update.

Nodes:
    retrieve_node           – Hybrid retrieval + Cohere reranking.
    grade_relevance_node    – Per-chunk Gemini relevance scoring.
    generate_answer_node    – Grounded answer generation with inline citations.
    verify_faithfulness_node – Faithfulness check of the generated answer.
    safe_fallback_node      – Returns a safe "insufficient documentation" reply.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

import structlog

from backend.core.config import get_settings
from backend.models import Citation, DocumentChunk, RAGState
from backend.retrieval import CohereReranker, HybridRetriever

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SAFE_ANSWER = (
    "I don't have sufficient documentation in the connected Google Workspace "
    "to answer your question. Please check that the relevant documents are "
    "shared with the RAG system."
)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM text response.

    Args:
        text: Raw text from Gemini that may contain extra prose around the JSON.

    Returns:
        Parsed dictionary from the first ``{...}`` block found.

    Raises:
        ValueError: If no valid JSON object is found in *text*.
    """
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return json.loads(match.group())


async def _gemini_generate(llm_client: Any, prompt: str) -> str:
    """Call the Gemini generate API and return the response text.

    Args:
        llm_client: A configured ``google.generativeai.GenerativeModel`` instance.
        prompt: Full prompt string to send.

    Returns:
        Stripped response text from the model.

    Raises:
        RuntimeError: If the Gemini call fails.
    """
    try:
        response = await llm_client.generate_content_async(prompt)
        return response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Conversational intent routing
# ---------------------------------------------------------------------------

_CONVERSATIONAL_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|goodbye|how are you|who are you|"
    r"what are you|what can you do|help me|tell me about yourself|"
    r"what is your name)\W*$",
    re.IGNORECASE,
)


async def intent_route_node(state: RAGState) -> dict[str, Any]:
    """Fast-path for conversational inputs that do not require document retrieval.

    Checks if the user's question matches common conversational patterns. If
    so, returns a canned response immediately, bypassing retrieval and all
    downstream LLM calls to save cost and latency.

    Args:
        state: Current RAGState containing the user's question.

    Returns:
        Dict with ``answer``, ``citations``, ``is_relevant``, ``is_faithful``
        pre-filled for conversational inputs, or empty dict to continue normal
        pipeline flow.
    """
    question = state.get("question", "").strip()
    if _CONVERSATIONAL_PATTERNS.match(question):
        log.bind(node="intent_route").info("conversational_bypass", question=question)
        return {
            "answer": (
                "I'm WorkspaceRAG, an AI assistant connected to your Google Workspace. "
                "Ask me anything about your indexed documents."
            ),
            "citations": [],
            "is_relevant": True,
            "is_faithful": True,
            "retrieved_docs": [],
            "relevant_docs": [],
        }
    return {"question": question}


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------


async def retrieve_node(
    state: RAGState,
    retriever: HybridRetriever,
    reranker: CohereReranker,
) -> dict[str, Any]:
    """Retrieve and rerank document chunks relevant to the user question.

    1. Calls ``HybridRetriever.retrieve`` with the state question and email.
    2. Passes raw results through ``CohereReranker.rerank`` to surface the
       most relevant chunks.

    Args:
        state: Current ``RAGState`` containing ``question`` and ``user_email``.
        retriever: Configured ``HybridRetriever`` instance (injected via partial).
        reranker: Configured ``CohereReranker`` instance (injected via partial).

    Returns:
        Dict with key ``retrieved_docs`` containing the reranked chunks.
    """
    node_log = log.bind(node="retrieve", question=state["question"])
    node_log.info("node_start")

    raw_chunks: list[DocumentChunk] = await retriever.retrieve(
        query=state["question"],
        user_email=state["user_email"],
        top_k=settings.TOP_K_RETRIEVAL,
    )

    reranked_chunks: list[DocumentChunk] = await reranker.rerank(
        query=state["question"],
        chunks=raw_chunks,
        top_n=settings.TOP_K_RERANK,
    )

    node_log.info("node_done", retrieved=len(reranked_chunks))
    return {"retrieved_docs": reranked_chunks}


# ---------------------------------------------------------------------------
# Node: grade_relevance
# ---------------------------------------------------------------------------


async def _grade_single_chunk(
    chunk: "DocumentChunk",
    question: str,
    llm_client: Any,
    node_log: Any,
) -> tuple["DocumentChunk", float, bool]:
    """Grade a single document chunk for relevance to the user question.

    This is a module-level helper used by ``grade_relevance_node`` to enable
    concurrent evaluation via ``asyncio.gather``.

    Args:
        chunk: The ``DocumentChunk`` to evaluate.
        question: The user's question string.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.
        node_log: Bound structlog logger from the calling node.

    Returns:
        A 3-tuple of ``(chunk, score, is_relevant)``.
    """
    prompt = (
        "You are a document relevance grader.\n"
        f"Query: {question}\n"
        f"Document excerpt: {chunk.content}\n"
        'Reply with a single JSON: {"relevant": true/false, "score": 0.0-1.0}'
    )
    try:
        raw = await _gemini_generate(llm_client, prompt)
        parsed = _extract_json(raw)
        score = float(parsed.get("score", 0.0))
        is_relevant = bool(parsed.get("relevant", False))
    except Exception as exc:
        node_log.warning(
            "relevance_grade_parse_failed",
            chunk_id=chunk.chunk_id,
            error=str(exc),
        )
        score = 0.0
        is_relevant = False
    return chunk, score, is_relevant


async def grade_relevance_node(
    state: RAGState,
    llm_client: Any,
) -> dict[str, Any]:
    """Grade each retrieved chunk for relevance to the user question.

    For every chunk in ``state["retrieved_docs"]`` a Gemini call is made using
    the relevance-grading prompt. Scores are averaged; chunks whose JSON reply
    has ``"relevant": true`` are retained in ``relevant_docs``. If the average
    score exceeds ``RELEVANCE_THRESHOLD`` **and** at least one chunk is
    relevant, ``is_relevant`` is set to ``True``.

    Args:
        state: Current ``RAGState`` with ``question`` and ``retrieved_docs``.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.

    Returns:
        Dict with keys ``is_relevant`` (bool) and ``relevant_docs``
        (list of relevant ``DocumentChunk``).
    """
    node_log = log.bind(node="grade_relevance", question=state["question"])
    node_log.info("node_start", num_docs=len(state["retrieved_docs"]))

    if not state["retrieved_docs"]:
        node_log.warning("no_retrieved_docs")
        return {"is_relevant": False, "relevant_docs": []}

    scores: list[float] = []
    relevant_docs: list[DocumentChunk] = []

    results = await asyncio.gather(
        *[
            _grade_single_chunk(chunk, state["question"], llm_client, node_log)
            for chunk in state["retrieved_docs"]
        ],
        return_exceptions=False,
    )
    for chunk, score, is_relevant_chunk in results:
        scores.append(score)
        if is_relevant_chunk:
            relevant_docs.append(chunk)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores, default=0.0)
    is_relevant = len(relevant_docs) > 0 and max_score >= settings.RELEVANCE_THRESHOLD

    node_log.info(
        "node_done",
        avg_score=avg_score,
        relevant_count=len(relevant_docs),
        is_relevant=is_relevant,
    )
    return {"is_relevant": is_relevant, "relevant_docs": relevant_docs}


# ---------------------------------------------------------------------------
# Node: generate_answer
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful enterprise knowledge assistant.\n"
    "Use ONLY the provided context to answer clearly, accurately, and concisely.\n"
    "Write in clean, natural prose using standard paragraphs or bullet points.\n"
    "Do NOT insert repetitive bracketed citations into every sentence. Mention the source document naturally.\n"
    "Avoid raw LaTeX markup or backslashed formatting for ordinary text and numbers.\n"
    "If context is insufficient, say 'I don\\'t have sufficient documentation to answer this.'"
)

_CITATION_PATTERN = re.compile(
    r"\[Doc:\s*(?P<title>[^,\]]+),\s*§(?P<section>[^\]]*)\]"
)


def _build_context(docs: list[DocumentChunk]) -> str:
    """Format relevant documents into a single context string for the LLM.

    Args:
        docs: List of ``DocumentChunk`` objects to include.

    Returns:
        Multi-block context string with per-chunk headers.
    """
    parts: list[str] = []
    for doc in docs:
        section = doc.section_header or "—"
        parts.append(
            f"[Doc: {doc.doc_title}, §{section}]\n"
            f"Source: {doc.source_url}\n"
            f"{doc.content}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_citations(
    answer: str,
    docs: list[DocumentChunk],
) -> list[Citation]:
    """Extract inline ``[Doc: ..., §...]`` references from the answer text.

    For each match the function looks up the corresponding ``DocumentChunk``
    by title and section to populate the ``Citation`` object with full
    metadata including an excerpt from the chunk.

    Args:
        answer: Generated answer text that may contain citation references.
        docs: Relevant document chunks (used as the metadata source).

    Returns:
        Deduplicated list of ``Citation`` objects found in *answer*.
    """
    # Build lookup by (title, section) for fast matching.
    doc_lookup: dict[tuple[str, str], DocumentChunk] = {}
    for doc in docs:
        key = (doc.doc_title.strip(), (doc.section_header or "—").strip())
        doc_lookup[key] = doc

    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []

    for match in _CITATION_PATTERN.finditer(answer):
        title = match.group("title").strip()
        section = match.group("section").strip()
        key = (title, section)
        if key in seen:
            continue
        seen.add(key)

        chunk = doc_lookup.get(key)
        if chunk:
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    source_url=chunk.source_url,
                    section_header=chunk.section_header,
                    excerpt=chunk.content[:200],
                )
            )
        else:
            # Citation referenced by LLM but not found in retrieved docs;
            # create a minimal stub so the reference is not silently lost.
            citations.append(
                Citation(
                    doc_id="unknown",
                    doc_title=title,
                    source_url="",
                    section_header=section if section != "—" else None,
                    excerpt="",
                )
            )

    return citations


async def generate_answer_node(
    state: RAGState,
    llm_client: Any,
) -> dict[str, Any]:
    """Generate a grounded, cited answer from the relevant document chunks.

    Constructs a context string from ``state["relevant_docs"]``, prepends the
    system instructions, and calls Gemini. The raw answer is then parsed for
    ``[Doc: ..., §...]`` citation markers to populate ``citations``.

    Args:
        state: Current ``RAGState`` with ``question`` and ``relevant_docs``.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.

    Returns:
        Dict with keys ``answer`` (str) and ``citations`` (list[Citation]).
    """
    node_log = log.bind(node="generate_answer", question=state["question"])
    node_log.info("node_start", num_relevant=len(state["relevant_docs"]))

    context = _build_context(state["relevant_docs"])
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"### Context\n{context}\n\n"
        f"### Question\n{state['question']}\n\n"
        "### Answer"
    )

    try:
        answer = await _gemini_generate(llm_client, full_prompt)
    except RuntimeError as exc:
        node_log.error("generation_failed", error=str(exc))
        return {
            "answer": _SAFE_ANSWER,
            "citations": [],
        }

    citations = _parse_citations(answer, state["relevant_docs"])
    if not citations and state["relevant_docs"]:
        seen_ids: set[str] = set()
        for doc in state["relevant_docs"]:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                citations.append(
                    Citation(
                        doc_id=doc.doc_id,
                        doc_title=doc.doc_title,
                        source_url=doc.source_url,
                        section_header=doc.section_header,
                        excerpt=doc.content[:200],
                    )
                )

    # Clean inline citation tags and awkward LaTeX formatting from the final answer text
    clean_ans = re.sub(r'\[Doc:\s*[^\]]+\]', '', answer)
    clean_ans = re.sub(r'\\text\{([^}]+)\}', r'\1', clean_ans)
    clean_ans = re.sub(r'\$([^$]+)\$', r'\1', clean_ans)
    clean_ans = re.sub(r'[ \t]+', ' ', clean_ans)
    clean_ans = re.sub(r'\s+([.,;:!?])', r'\1', clean_ans).strip()

    node_log.info("node_done", answer_length=len(clean_ans), num_citations=len(citations))
    return {"answer": clean_ans, "citations": citations}


# ---------------------------------------------------------------------------
# Node: verify_faithfulness
# ---------------------------------------------------------------------------


async def verify_faithfulness_node(
    state: RAGState,
    llm_client: Any,
) -> dict[str, Any]:
    """Check whether every factual claim in the answer is grounded in context.

    Sends the joined context and the generated answer to Gemini and asks for a
    JSON verdict containing ``faithful`` (bool), ``score`` (float), and
    ``issues`` (list of unsupported claims).

    Args:
        state: Current ``RAGState`` with ``relevant_docs`` and ``answer``.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.

    Returns:
        Dict with keys ``is_faithful`` (bool) and ``faithfulness_score`` (float).
    """
    node_log = log.bind(node="verify_faithfulness")
    node_log.info("node_start")

    context = _build_context(state["relevant_docs"])
    prompt = (
        "Given the context and the answer, check if every factual claim in the "
        "answer is supported by the context.\n\n"
        f"### Context\n{context}\n\n"
        f"### Answer\n{state['answer']}\n\n"
        'Return JSON: {"faithful": true/false, "score": 0.0-1.0, "issues": [list of unsupported claims]}'
    )

    try:
        raw = await _gemini_generate(llm_client, prompt)
        parsed = _extract_json(raw)
        is_faithful: bool = bool(parsed.get("faithful", False))
        faithfulness_score: float = float(parsed.get("score", 0.0))
        issues: list[str] = parsed.get("issues", [])
    except Exception as exc:
        node_log.warning("faithfulness_parse_failed", error=str(exc))
        # Default to not faithful so the fallback node is triggered.
        is_faithful = False
        faithfulness_score = 0.0
        issues = []

    node_log.info(
        "node_done",
        is_faithful=is_faithful,
        faithfulness_score=faithfulness_score,
        issues=issues,
    )
    return {"is_faithful": is_faithful, "faithfulness_score": faithfulness_score}


# ---------------------------------------------------------------------------
# Node: safe_fallback
# ---------------------------------------------------------------------------


async def safe_fallback_node(state: RAGState) -> dict[str, Any]:  # noqa: ARG001
    """Return a safe, user-facing fallback answer when grounding checks fail.

    This node is reached when either:
    - No relevant documents were found (``is_relevant`` is ``False``), or
    - The generated answer failed the faithfulness check.

    Args:
        state: Current ``RAGState`` (unused; signature required by LangGraph).

    Returns:
        Dict with ``answer`` set to the safe fallback string, empty
        ``citations``, and ``is_faithful`` set to ``True`` (the fallback answer
        itself is always faithful — it makes no factual claims).
    """
    log.bind(node="safe_fallback").info("node_triggered")
    return {
        "answer": _SAFE_ANSWER,
        "citations": [],
        "is_faithful": True,
    }
