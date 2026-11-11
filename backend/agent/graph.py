"""LangGraph StateGraph definition for the workspace RAG pipeline.

Defines ``WorkspaceRAGGraph``, which wires together the retrieval, grading,
generation, faithfulness-verification, and fallback nodes into a compiled
``StateGraph`` with conditional branching. Every invocation is wrapped in a
Langfuse trace for end-to-end observability.
"""
from __future__ import annotations

import functools
import uuid
from typing import Any

import structlog
from langfuse import Langfuse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph

from backend.agent.nodes import (
    generate_answer_node,
    grade_relevance_node,
    intent_route_node,
    retrieve_node,
    safe_fallback_node,
    verify_faithfulness_node,
)
from backend.core.config import get_settings
from backend.models import RAGState
from backend.retrieval import CohereReranker, HybridRetriever

log = structlog.get_logger(__name__)
settings = get_settings()


class WorkspaceRAGGraph:
    """Builds and runs the LangGraph-based workspace RAG pipeline.

    The graph implements the following flow::

        START
          └─► retrieve
                └─► grade_relevance
                      ├─(relevant)─► generate ─► verify_faithfulness
                      │                                  ├─(faithful)─► END
                      │                                  └─(not faithful)─► fallback ─► END
                      └─(not relevant)─► fallback ─► END

    Tracing is handled via the Langfuse SDK: each ``run`` invocation creates a
    new trace keyed by ``trace_id`` (generated from ``session_id``).

    Attributes:
        retriever: ``HybridRetriever`` for dense+sparse retrieval.
        reranker: ``CohereReranker`` for Cohere-based cross-encoder reranking.
        llm_client: Configured ``google.generativeai.GenerativeModel`` instance.
        langfuse_client: Authenticated ``Langfuse`` client for observability.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CohereReranker,
        llm_client: Any,
        langfuse_client: Langfuse,
    ) -> None:
        """Initialise the graph with all required dependencies.

        Args:
            retriever: ``HybridRetriever`` instance backed by a live DB pool.
            reranker: ``CohereReranker`` instance with a valid API key.
            llm_client: Gemini ``GenerativeModel`` for grading and generation.
            langfuse_client: Authenticated ``Langfuse`` client.
        """
        self._retriever = retriever
        self._reranker = reranker
        self._llm_client = llm_client
        self._langfuse = langfuse_client
        self._log = log.bind(component="WorkspaceRAGGraph")
        self._compiled: CompiledGraph | None = None

    # ------------------------------------------------------------------
    # Private: routing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _route_relevance(state: RAGState) -> str:
        """Conditional-edge router after ``grade_relevance``.

        Args:
            state: Current graph state.

        Returns:
            ``"generate"`` when ``is_relevant`` is truthy, else ``"fallback"``.
        """
        return "generate" if state.get("is_relevant") else "fallback"

    @staticmethod
    def _route_faithfulness(state: RAGState) -> str:
        """Conditional-edge router after ``verify_faithfulness``.

        Args:
            state: Current graph state.

        Returns:
            ``END`` constant when ``is_faithful`` is truthy, else ``"fallback"``.
        """
        return END if state.get("is_faithful") else "fallback"

    # ------------------------------------------------------------------
    # Public: build_graph
    # ------------------------------------------------------------------

    def build_graph(self) -> CompiledGraph:
        """Construct and compile the LangGraph ``StateGraph``.

        Uses ``functools.partial`` to bind ``retriever``, ``reranker``, and
        ``llm_client`` into the stateless node functions so they satisfy
        LangGraph's ``(state) -> dict`` contract.

        The graph flow is::

            START
              └─► intent_route
                    ├─(answer set, e.g. chitchat/greeting)─► END
                    └─(no answer yet)─► retrieve
                                          └─► grade_relevance
                                                ├─(relevant)─► generate ─► verify_faithfulness
                                                │                                ├─(faithful)─► END
                                                │                                └─(not faithful)─► fallback ─► END
                                                └─(not relevant)─► fallback ─► END

        Returns:
            A compiled ``CompiledGraph`` ready for ``.ainvoke``.
        """
        graph: StateGraph = StateGraph(RAGState)

        # Bind dependencies into node functions.
        bound_retrieve = functools.partial(
            retrieve_node,
            retriever=self._retriever,
            reranker=self._reranker,
        )
        bound_grade_relevance = functools.partial(
            grade_relevance_node,
            llm_client=self._llm_client,
        )
        bound_generate = functools.partial(
            generate_answer_node,
            llm_client=self._llm_client,
        )
        bound_verify = functools.partial(
            verify_faithfulness_node,
            llm_client=self._llm_client,
        )

        # Register nodes.
        graph.add_node("intent_route", intent_route_node)
        graph.add_node("retrieve", bound_retrieve)
        graph.add_node("grade_relevance", bound_grade_relevance)
        graph.add_node("generate", bound_generate)
        graph.add_node("verify_faithfulness", bound_verify)
        graph.add_node("fallback", safe_fallback_node)

        # Fixed edges.
        graph.add_edge(START, "intent_route")
        graph.add_conditional_edges(
            "intent_route",
            lambda s: END if s.get("answer") else "retrieve",
            {END: END, "retrieve": "retrieve"},
        )
        graph.add_edge("retrieve", "grade_relevance")
        graph.add_edge("generate", "verify_faithfulness")
        graph.add_edge("fallback", END)

        # Conditional edges.
        graph.add_conditional_edges(
            "grade_relevance",
            self._route_relevance,
            {"generate": "generate", "fallback": "fallback"},
        )
        graph.add_conditional_edges(
            "verify_faithfulness",
            self._route_faithfulness,
            {END: END, "fallback": "fallback"},
        )

        self._compiled = graph.compile()
        self._log.info("graph_compiled")
        return self._compiled

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    async def run(
        self,
        question: str,
        user_email: str,
        session_id: str,
    ) -> RAGState:
        """Execute the RAG pipeline for a single question.

        Creates a Langfuse trace keyed by a UUID derived from *session_id*,
        invokes the compiled graph, and flushes observability data before
        returning the final state.

        Args:
            question: Natural-language question from the end user.
            user_email: Email address of the requesting user (access control).
            session_id: Session or conversation identifier used for grouping
                traces in Langfuse.

        Returns:
            Final ``RAGState`` dict containing ``answer``, ``citations``,
            ``is_relevant``, ``is_faithful``, and all intermediate fields.

        Raises:
            RuntimeError: If the graph has not been compiled yet (call
                ``build_graph()`` first) or if the pipeline fails
                unrecoverably.
        """
        if self._compiled is None:
            raise RuntimeError(
                "Graph has not been compiled. Call build_graph() before run()."
            )

        trace_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}:{question}"))
        run_log = self._log.bind(
            trace_id=trace_id,
            session_id=session_id,
            user_email=user_email,
        )
        run_log.info("run_start", question=question)

        initial_state: RAGState = {
            "question": question,
            "user_email": user_email,
            "retrieved_docs": [],
            "relevant_docs": [],
            "answer": "",
            "citations": [],
            "is_relevant": False,
            "is_faithful": False,
            "trace_id": trace_id,
            "error": None,
        }

        try:
            with self._langfuse.start_as_current_observation(
                name="workspace-rag-pipeline",
                as_type="chain",
                input={"question": question, "user_email": user_email},
                metadata={"model": settings.GEMINI_MODEL, "session_id": session_id},
            ) as span:
                try:
                    final_state: RAGState = await self._compiled.ainvoke(initial_state)
                except Exception as exc:
                    run_log.error("pipeline_error", error=str(exc))
                    span.update(output={"error": str(exc)}, level="ERROR")
                    self._langfuse.flush()
                    raise RuntimeError(f"RAG pipeline failed: {exc}") from exc

                span.update(
                    output={
                        "answer": final_state.get("answer", ""),
                        "num_citations": len(final_state.get("citations", [])),
                        "is_relevant": final_state.get("is_relevant"),
                        "is_faithful": final_state.get("is_faithful"),
                    },
                    level="DEFAULT",
                )
                self._langfuse.flush()
        except AttributeError:
            # Fallback if an older SDK instance is used
            final_state = await self._compiled.ainvoke(initial_state)

        run_log.info(
            "run_done",
            is_relevant=final_state.get("is_relevant"),
            is_faithful=final_state.get("is_faithful"),
            answer_length=len(final_state.get("answer", "")),
        )
        return final_state
