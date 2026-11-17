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


async def _gemini_generate_with_usage(
    llm_client: Any,
    prompt: str,
) -> tuple[str, dict[str, int]]:
    """Call the Gemini generate API and return response text and token counts.

    Args:
        llm_client: A configured ``google.generativeai.GenerativeModel`` instance.
        prompt: Full prompt string to send.

    Returns:
        Tuple of (stripped response text, usage dict with prompt_tokens,
        completion_tokens, and total_tokens).

    Raises:
        RuntimeError: If the Gemini call fails.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        response = await llm_client.generate_content_async(prompt)
        text = response.text.strip()
        metadata = getattr(response, "usage_metadata", None)
        if metadata:
            usage["prompt_tokens"] = getattr(metadata, "prompt_token_count", 0) or 0
            usage["completion_tokens"] = getattr(metadata, "candidates_token_count", 0) or 0
            usage["total_tokens"] = getattr(metadata, "total_token_count", 0) or 0
        else:
            prompt_est = max(1, len(prompt.split()))
            comp_est = max(1, len(text.split()))
            usage["prompt_tokens"] = prompt_est
            usage["completion_tokens"] = comp_est
            usage["total_tokens"] = prompt_est + comp_est
        return text, usage
    except Exception as exc:
        raise RuntimeError(f"Gemini generation failed: {exc}") from exc


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
    text, _ = await _gemini_generate_with_usage(llm_client, prompt)
    return text


# ---------------------------------------------------------------------------
# Conversational intent routing
# ---------------------------------------------------------------------------

_CONVERSATIONAL_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|goodbye|how are you|who are you|"
    r"what are you|what can you do|help me|tell me about yourself|"
    r"what is your name)\W*$",
    re.IGNORECASE,
)


async def intent_route_node(
    state: RAGState,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Fast-path for conversational inputs and query condensation for multi-turn chat.

    1. Checks if the user's question matches common conversational patterns. If
       so and no prior history exists, returns a canned response immediately,
       bypassing retrieval and all downstream LLM calls to save cost and latency.
    2. If conversation history is present, invokes Gemini to reformulate ambiguous
       follow-ups into standalone search queries resolving pronouns and antecedents.

    Args:
        state: Current RAGState containing the user's question and history.
        llm_client: Optional Gemini GenerativeModel used for query reformulation.

    Returns:
        Dict with state updates (pre-filled answer for greeting, or
        ``standalone_question`` and updated ``token_usage``).
    """
    question = state.get("question", "").strip()
    history = state.get("history") or []

    if not history and _CONVERSATIONAL_PATTERNS.match(question):
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

    standalone_question = question
    token_usage = dict(state.get("token_usage") or {})

    # If history is present, condense follow-up into a standalone search query
    if history and llm_client:
        history_lines: list[str] = []
        for turn in history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {turn.get('content', '')}")
        history_text = "\n".join(history_lines)

        condense_prompt = (
            "You are an expert search query reformulation engine for an enterprise search system.\n"
            "Given the conversation history and a follow-up question, rewrite the follow-up question "
            "into a complete, standalone search query that preserves all necessary context, nouns, "
            "and entity references.\n"
            "If the question is already complete and standalone, return it unchanged.\n"
            "Do NOT answer the question. Output ONLY the standalone query.\n\n"
            f"### Conversation History\n{history_text}\n\n"
            f"### Follow-up Question\n{question}\n\n"
            "### Standalone Query:"
        )
        try:
            condensed, usage = await _gemini_generate_with_usage(llm_client, condense_prompt)
            if condensed and len(condensed) >= 3:
                standalone_question = condensed.strip().strip('"').strip("'")
                log.bind(node="intent_route").info(
                    "query_condensed",
                    original=question,
                    standalone=standalone_question,
                )
                token_usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0) + usage["prompt_tokens"]
                token_usage["completion_tokens"] = token_usage.get("completion_tokens", 0) + usage["completion_tokens"]
                token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + usage["total_tokens"]
        except Exception as exc:
            log.warning("query_condensation_failed", error=str(exc))

    return {
        "question": question,
        "standalone_question": standalone_question,
        "token_usage": token_usage,
    }


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------


async def retrieve_node(
    state: RAGState,
    retriever: HybridRetriever,
    reranker: CohereReranker,
) -> dict[str, Any]:
    """Retrieve and rerank document chunks relevant to the user question.

    1. Uses ``standalone_question`` (with antecedents resolved) if available.
    2. Calls ``HybridRetriever.retrieve`` with the search query and email.
    3. Passes raw results through ``CohereReranker.rerank`` to surface the
       most relevant chunks.

    Args:
        state: Current ``RAGState`` containing ``question`` and ``user_email``.
        retriever: Configured ``HybridRetriever`` instance (injected via partial).
        reranker: Configured ``CohereReranker`` instance (injected via partial).

    Returns:
        Dict with key ``retrieved_docs`` containing the reranked chunks.
    """
    search_query = state.get("standalone_question") or state["question"]
    node_log = log.bind(node="retrieve", question=state["question"], search_query=search_query)
    node_log.info("node_start")

    raw_chunks: list[DocumentChunk] = await retriever.retrieve(
        query=search_query,
        user_email=state["user_email"],
        top_k=settings.TOP_K_RETRIEVAL,
    )

    reranked_chunks: list[DocumentChunk] = await reranker.rerank(
        query=search_query,
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
    langfuse_client: Any = None,
) -> tuple["DocumentChunk", float, bool, dict[str, int]]:
    """Grade a single document chunk for relevance to the user question.

    This is a module-level helper used by ``grade_relevance_node`` to enable
    concurrent evaluation via ``asyncio.gather``.

    Args:
        chunk: The ``DocumentChunk`` to evaluate.
        question: The user's question string.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.
        node_log: Bound structlog logger from the calling node.
        langfuse_client: Optional Langfuse client for child generation tracing.

    Returns:
        A 4-tuple of ``(chunk, score, is_relevant, usage_dict)``.
    """
    prompt = (
        "You are a document relevance grader.\n"
        f"Query: {question}\n"
        f"Document excerpt: {chunk.content}\n"
        'Reply with a single JSON: {"relevant": true/false, "score": 0.0-1.0}'
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        if langfuse_client and hasattr(langfuse_client, "start_as_current_observation"):
            try:
                with langfuse_client.start_as_current_observation(
                    name="grade_relevance_chunk",
                    as_type="generation",
                    model=settings.GEMINI_MODEL,
                    input=prompt,
                ) as gen_span:
                    raw, usage = await _gemini_generate_with_usage(llm_client, prompt)
                    gen_span.update(
                        output=raw,
                        usage={
                            "input": usage["prompt_tokens"],
                            "output": usage["completion_tokens"],
                            "total": usage["total_tokens"],
                        },
                        model=settings.GEMINI_MODEL,
                    )
            except Exception:
                raw, usage = await _gemini_generate_with_usage(llm_client, prompt)
        else:
            raw, usage = await _gemini_generate_with_usage(llm_client, prompt)

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
    return chunk, score, is_relevant, usage


async def grade_relevance_node(
    state: RAGState,
    llm_client: Any,
    langfuse_client: Any = None,
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
        langfuse_client: Optional Langfuse client for child generation tracing.

    Returns:
        Dict with keys ``is_relevant`` (bool), ``relevant_docs``
        (list of relevant ``DocumentChunk``), and ``token_usage`` (dict).
    """
    node_log = log.bind(node="grade_relevance", question=state["question"])
    node_log.info("node_start", num_docs=len(state["retrieved_docs"]))

    token_usage = dict(state.get("token_usage") or {})

    if not state["retrieved_docs"]:
        node_log.warning("no_retrieved_docs")
        return {"is_relevant": False, "relevant_docs": [], "token_usage": token_usage}

    scores: list[float] = []
    relevant_docs: list[DocumentChunk] = []

    results = await asyncio.gather(
        *[
            _grade_single_chunk(chunk, state["question"], llm_client, node_log, langfuse_client)
            for chunk in state["retrieved_docs"]
        ],
        return_exceptions=False,
    )
    for chunk, score, is_relevant_chunk, chunk_usage in results:
        scores.append(score)
        if is_relevant_chunk:
            relevant_docs.append(chunk)
        token_usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0) + chunk_usage["prompt_tokens"]
        token_usage["completion_tokens"] = token_usage.get("completion_tokens", 0) + chunk_usage["completion_tokens"]
        token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + chunk_usage["total_tokens"]

    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores, default=0.0)
    is_relevant = len(relevant_docs) > 0 and max_score >= settings.RELEVANCE_THRESHOLD

    node_log.info(
        "node_done",
        avg_score=avg_score,
        relevant_count=len(relevant_docs),
        is_relevant=is_relevant,
        tokens=token_usage.get("total_tokens", 0),
    )
    return {"is_relevant": is_relevant, "relevant_docs": relevant_docs, "token_usage": token_usage}


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
    langfuse_client: Any = None,
) -> dict[str, Any]:
    """Generate a grounded, cited answer from the relevant document chunks.

    Constructs a context string from ``state["relevant_docs"]``, prepends the
    system instructions, and calls Gemini. The raw answer is then parsed for
    ``[Doc: ..., §...]`` citation markers to populate ``citations``.

    Args:
        state: Current ``RAGState`` with ``question`` and ``relevant_docs``.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.
        langfuse_client: Optional Langfuse client for child generation tracing.

    Returns:
        Dict with keys ``answer`` (str), ``citations`` (list[Citation]),
        and ``token_usage`` (dict).
    """
    node_log = log.bind(node="generate_answer", question=state["question"])
    node_log.info("node_start", num_relevant=len(state["relevant_docs"]))

    token_usage = dict(state.get("token_usage") or {})

    context = _build_context(state["relevant_docs"])
    history = state.get("history") or []
    history_block = ""
    if history:
        history_lines: list[str] = []
        for turn in history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {turn.get('content', '')}")
        history_block = "### Prior Conversation\n" + "\n".join(history_lines) + "\n\n"

    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{history_block}"
        f"### Context\n{context}\n\n"
        f"### Question\n{state['question']}\n\n"
        "### Answer"
    )

    try:
        if langfuse_client and hasattr(langfuse_client, "start_as_current_observation"):
            try:
                with langfuse_client.start_as_current_observation(
                    name="generate_answer",
                    as_type="generation",
                    model=settings.GEMINI_MODEL,
                    input=full_prompt,
                ) as gen_span:
                    answer, usage = await _gemini_generate_with_usage(llm_client, full_prompt)
                    gen_span.update(
                        output=answer,
                        usage={
                            "input": usage["prompt_tokens"],
                            "output": usage["completion_tokens"],
                            "total": usage["total_tokens"],
                        },
                        model=settings.GEMINI_MODEL,
                    )
            except Exception:
                answer, usage = await _gemini_generate_with_usage(llm_client, full_prompt)
        else:
            answer, usage = await _gemini_generate_with_usage(llm_client, full_prompt)

        token_usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0) + usage["prompt_tokens"]
        token_usage["completion_tokens"] = token_usage.get("completion_tokens", 0) + usage["completion_tokens"]
        token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + usage["total_tokens"]

    except RuntimeError as exc:
        node_log.error("generation_failed", error=str(exc))
        return {
            "answer": _SAFE_ANSWER,
            "citations": [],
            "token_usage": token_usage,
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

    node_log.info(
        "node_done",
        answer_length=len(clean_ans),
        num_citations=len(citations),
        tokens=token_usage.get("total_tokens", 0),
    )
    return {"answer": clean_ans, "citations": citations, "token_usage": token_usage}


# ---------------------------------------------------------------------------
# Node: verify_faithfulness
# ---------------------------------------------------------------------------


async def verify_faithfulness_node(
    state: RAGState,
    llm_client: Any,
    langfuse_client: Any = None,
) -> dict[str, Any]:
    """Check whether every factual claim in the answer is grounded in context.

    Sends the joined context and the generated answer to Gemini and asks for a
    JSON verdict containing ``faithful`` (bool), ``score`` (float), and
    ``issues`` (list of unsupported claims).

    Args:
        state: Current ``RAGState`` with ``relevant_docs`` and ``answer``.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.
        langfuse_client: Optional Langfuse client for child generation tracing.

    Returns:
        Dict with keys ``is_faithful`` (bool), ``faithfulness_score`` (float),
        and ``token_usage`` (dict).
    """
    node_log = log.bind(node="verify_faithfulness")
    node_log.info("node_start")

    token_usage = dict(state.get("token_usage") or {})

    context = _build_context(state["relevant_docs"])
    prompt = (
        "Given the context and the answer, check if every factual claim in the "
        "answer is supported by the context.\n\n"
        f"### Context\n{context}\n\n"
        f"### Answer\n{state['answer']}\n\n"
        'Return JSON: {"faithful": true/false, "score": 0.0-1.0, "issues": [list of unsupported claims]}'
    )

    try:
        if langfuse_client and hasattr(langfuse_client, "start_as_current_observation"):
            try:
                with langfuse_client.start_as_current_observation(
                    name="verify_faithfulness",
                    as_type="generation",
                    model=settings.GEMINI_MODEL,
                    input=prompt,
                ) as gen_span:
                    raw, usage = await _gemini_generate_with_usage(llm_client, prompt)
                    gen_span.update(
                        output=raw,
                        usage={
                            "input": usage["prompt_tokens"],
                            "output": usage["completion_tokens"],
                            "total": usage["total_tokens"],
                        },
                        model=settings.GEMINI_MODEL,
                    )
            except Exception:
                raw, usage = await _gemini_generate_with_usage(llm_client, prompt)
        else:
            raw, usage = await _gemini_generate_with_usage(llm_client, prompt)

        token_usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0) + usage["prompt_tokens"]
        token_usage["completion_tokens"] = token_usage.get("completion_tokens", 0) + usage["completion_tokens"]
        token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + usage["total_tokens"]

        parsed = _extract_json(raw)
        is_faithful: bool = bool(parsed.get("faithful", False))
        faithfulness_score: float = float(parsed.get("score", 0.0))
        issues: list[str] = parsed.get("issues", [])
    except Exception as exc:
        node_log.warning("faithfulness_parse_failed", error=str(exc))
        is_faithful = False
        faithfulness_score = 0.0
        issues = []

    node_log.info(
        "node_done",
        is_faithful=is_faithful,
        faithfulness_score=faithfulness_score,
        issues=issues,
        tokens=token_usage.get("total_tokens", 0),
    )
    return {
        "is_faithful": is_faithful,
        "faithfulness_score": faithfulness_score,
        "token_usage": token_usage,
    }


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
