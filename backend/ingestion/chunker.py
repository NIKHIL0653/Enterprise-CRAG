"""Semantic markdown chunker for RAG document ingestion.

Splits raw plain-text (originally from Google Docs) into overlapping token
windows, grouping content by detected section headings.  Each chunk is
represented as a :class:`~backend.models.document.DocumentChunk`.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import structlog
import tiktoken

from backend.models.document import Document, DocumentChunk

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Heading detection patterns
# ---------------------------------------------------------------------------

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
# ALL-CAPS line with more than 4 words (e.g. "SECTION FOUR OVERVIEW DATA")
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-&:,]{10,}$")


def _is_heading(line: str) -> bool:
    """Return True if *line* looks like a section heading.

    Detects both Markdown-style headings (``#``, ``##``, ``###``) and
    ALL-CAPS lines containing more than four words.

    Args:
        line: A single line of text (should be stripped).

    Returns:
        ``True`` when the line is detected as a heading.
    """
    stripped = line.strip()
    if _MARKDOWN_HEADING_RE.match(stripped):
        return True
    if _ALL_CAPS_HEADING_RE.match(stripped):
        words = stripped.split()
        return len(words) > 4
    return False


def _extract_heading_text(line: str) -> str:
    """Extract the display text from a heading line.

    For Markdown headings, strips the leading ``#`` characters.
    For ALL-CAPS headings, returns the line as-is (title-cased).

    Args:
        line: A heading line (already confirmed by :func:`_is_heading`).

    Returns:
        The cleaned heading string.
    """
    stripped = line.strip()
    md_match = _MARKDOWN_HEADING_RE.match(stripped)
    if md_match:
        return md_match.group(2).strip()
    return stripped.title()


# ---------------------------------------------------------------------------
# Chunk-ID generation
# ---------------------------------------------------------------------------


def _make_chunk_id(doc_id: str, section_header: str | None, chunk_index: int) -> str:
    """Generate a deterministic 16-character chunk identifier.

    Computes ``sha256(doc_id + section_header + chunk_index)`` and returns
    the first 16 hex characters.

    Args:
        doc_id: The parent document identifier.
        section_header: The heading under which this chunk falls, or ``None``.
        chunk_index: Zero-based position of this chunk within the document.

    Returns:
        A 16-character lowercase hex string.
    """
    raw = f"{doc_id}|{section_header or ''}|{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main chunker class
# ---------------------------------------------------------------------------


class SemanticMarkdownChunker:
    """Splits documents into overlapping token-window chunks grouped by section.

    Uses ``tiktoken`` with the ``cl100k_base`` encoding (compatible with most
    OpenAI and Gemini embedding models) for precise token counting.

    Example::

        chunker = SemanticMarkdownChunker(max_tokens=512, overlap_tokens=50)
        chunks = chunker.chunk_document(doc, raw_text)
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50) -> None:
        """Initialise the chunker with token-window settings.

        Args:
            max_tokens: Maximum number of tokens per chunk. Defaults to 512.
            overlap_tokens: Number of tokens shared between consecutive chunks
                within the same section to preserve context. Defaults to 50.

        Raises:
            ValueError: If *overlap_tokens* >= *max_tokens*.
        """
        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than "
                f"max_tokens ({max_tokens})."
            )
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")
        logger.info(
            "SemanticMarkdownChunker ready",
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document, raw_text: str) -> list[DocumentChunk]:
        """Chunk *raw_text* into overlapping token windows grouped by heading.

        Pipeline:
        1. Split the text into sections by detecting heading lines.
        2. For each section, encode the body text with tiktoken, then slide
           a ``max_tokens``-wide window with ``overlap_tokens`` overlap to
           produce sub-chunks.
        3. Each sub-chunk is wrapped in a :class:`DocumentChunk` with all
           metadata fields populated (``embedding`` is left as ``None``).

        Args:
            doc: The parent :class:`~backend.models.document.Document` whose
                metadata (permissions, URL, etc.) is copied to every chunk.
            raw_text: The raw plain-text export of the Google Doc.

        Returns:
            A list of :class:`~backend.models.document.DocumentChunk` objects
            ordered by their position in the source document.
        """
        log = logger.bind(
            method="chunk_document", doc_id=doc.doc_id, title=doc.title
        )
        log.info("Starting document chunking")

        sections = self._split_into_sections(raw_text)
        log.debug("Sections detected", count=len(sections))

        chunks: list[DocumentChunk] = []
        now = datetime.now(timezone.utc)
        global_chunk_index = 0

        for section_header, section_body in sections:
            if not section_body.strip():
                continue

            sub_chunks = self._window_split(section_body)
            for sub_text in sub_chunks:
                token_count = len(self._enc.encode(sub_text))
                chunk_id = _make_chunk_id(doc.doc_id, section_header, global_chunk_index)

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    source_url=doc.source_url,
                    section_header=section_header,
                    content=sub_text,
                    token_count=token_count,
                    chunk_index=global_chunk_index,
                    allowed_emails=list(doc.allowed_emails),
                    allowed_domain=doc.allowed_domain,
                    embedding=None,
                    created_at=now,
                    updated_at=now,
                )
                chunks.append(chunk)
                global_chunk_index += 1

        log.info(
            "Chunking complete",
            total_chunks=len(chunks),
            sections=len(sections),
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sections(
        self, text: str
    ) -> list[tuple[str | None, str]]:
        """Parse *text* into ``(heading, body)`` pairs.

        Iterates over lines; whenever a heading is detected the current section
        is finalised and a new one starts.  Content before the first heading is
        grouped under ``None``.

        Args:
            text: The full plain-text document content.

        Returns:
            Ordered list of ``(section_header, body_text)`` tuples.
        """
        sections: list[tuple[str | None, str]] = []
        current_header: str | None = None
        current_lines: list[str] = []

        for line in text.splitlines():
            if _is_heading(line):
                # Flush current section
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_header, body))
                current_header = _extract_heading_text(line)
                current_lines = []
            else:
                current_lines.append(line)

        # Flush final section
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_header, body))

        return sections

    def _window_split(self, text: str) -> list[str]:
        """Split *text* into overlapping token windows.

        Encodes the text with tiktoken, then slides a window of
        ``max_tokens`` tokens, advancing by ``(max_tokens - overlap_tokens)``
        each step.  Each window is decoded back to a string.

        Args:
            text: The section body text to split.

        Returns:
            Ordered list of string sub-chunks.
        """
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        step = self.max_tokens - self.overlap_tokens
        sub_chunks: list[str] = []

        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            window_tokens = tokens[start:end]
            sub_chunks.append(self._enc.decode(window_tokens))
            if end == len(tokens):
                break
            start += step

        return sub_chunks
