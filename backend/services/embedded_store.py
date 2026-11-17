"""Embedded local vector store using SQLite and NumPy.

Provides zero-dependency, local ACID storage with FTS5 BM25 text search and
vector cosine similarity. Enables WorkspaceRAG to operate fully locally
without requiring an external PostgreSQL/pgvector instance.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.logging import get_logger
from backend.models import DocumentChunk

log = get_logger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "embedded_rag.db"


class EmbeddedVectorStore:
    """Thread-safe SQLite + NumPy vector store with FTS5 full-text search."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create relational tables and FTS5 virtual tables."""
        with self._lock, self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    owner_email TEXT NOT NULL,
                    allowed_emails TEXT NOT NULL DEFAULT '[]',
                    allowed_domain TEXT,
                    last_modified TEXT NOT NULL,
                    synced_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    doc_title TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    section_header TEXT,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    allowed_emails TEXT NOT NULL DEFAULT '[]',
                    allowed_domain TEXT,
                    embedding BLOB,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    doc_title,
                    section_header,
                    content,
                    tokenize='porter unicode61'
                );

                CREATE TABLE IF NOT EXISTS rag_query_cache (
                    cache_id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    answer TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_query_cache_hash ON rag_query_cache(query_hash);
                """
            )

    async def upsert_document(
        self,
        doc_id: str,
        title: str,
        source_url: str = "",
        mime_type: str = "text/plain",
        owner_email: str = "local@system",
        allowed_emails: list[str] | None = None,
        allowed_domain: str | None = None,
    ) -> None:
        """Upsert document metadata."""
        now = datetime.now(timezone.utc).isoformat()
        allowed_json = json.dumps(allowed_emails or [])
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, title, source_url, mime_type, owner_email,
                    allowed_emails, allowed_domain, last_modified, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title,
                    source_url=excluded.source_url,
                    mime_type=excluded.mime_type,
                    allowed_emails=excluded.allowed_emails,
                    last_modified=excluded.last_modified,
                    synced_at=excluded.synced_at;
                """,
                (doc_id, title, source_url, mime_type, owner_email, allowed_json, allowed_domain, now, now),
            )

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Upsert chunks into both relational table and FTS5 search index."""
        if not chunks:
            return

        with self._lock, self._get_connection() as conn:
            for c in chunks:
                emb_blob = None
                if c.embedding:
                    emb_blob = np.array(c.embedding, dtype=np.float32).tobytes()

                allowed_json = json.dumps(c.allowed_emails or [])
                now_str = c.updated_at.isoformat() if c.updated_at else datetime.now(timezone.utc).isoformat()
                created_str = c.created_at.isoformat() if c.created_at else now_str

                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id, doc_id, doc_title, source_url, section_header,
                        content, token_count, chunk_index, allowed_emails, allowed_domain,
                        embedding, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        content=excluded.content,
                        section_header=excluded.section_header,
                        token_count=excluded.token_count,
                        embedding=excluded.embedding,
                        updated_at=excluded.updated_at;
                    """,
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.doc_title,
                        c.source_url,
                        c.section_header,
                        c.content,
                        c.token_count,
                        c.chunk_index,
                        allowed_json,
                        c.allowed_domain,
                        emb_blob,
                        created_str,
                        now_str,
                    ),
                )

                # Update FTS5 index (delete old entry first to prevent duplicate search matches)
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?;", (c.chunk_id,))
                conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, doc_id, doc_title, section_header, content)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (c.chunk_id, c.doc_id, c.doc_title, c.section_header or "", c.content),
                )

    async def dense_search(
        self,
        query_vector: list[float],
        top_k: int = 15,
        user_email: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """Perform vectorized cosine similarity search using NumPy."""
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        q_unit = q_vec / q_norm

        with self._lock, self._get_connection() as conn:
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                cursor = conn.execute(
                    f"""
                    SELECT chunk_id, doc_id, doc_title, source_url, section_header,
                           content, token_count, chunk_index, allowed_emails, allowed_domain,
                           embedding, created_at, updated_at
                    FROM document_chunks
                    WHERE embedding IS NOT NULL AND doc_id IN ({placeholders});
                    """,
                    tuple(doc_ids),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT chunk_id, doc_id, doc_title, source_url, section_header,
                           content, token_count, chunk_index, allowed_emails, allowed_domain,
                           embedding, created_at, updated_at
                    FROM document_chunks
                    WHERE embedding IS NOT NULL;
                    """
                )
            rows = cursor.fetchall()

        if not rows:
            return []

        # Vectorized batch cosine similarity
        scored_chunks: list[tuple[float, DocumentChunk]] = []
        for r in rows:
            raw_blob = r["embedding"]
            chunk_vec = np.frombuffer(raw_blob, dtype=np.float32)
            c_norm = np.linalg.norm(chunk_vec)
            if c_norm > 0:
                sim = float(np.dot(q_unit, chunk_vec / c_norm))
            else:
                sim = 0.0

            emails = json.loads(r["allowed_emails"]) if r["allowed_emails"] else []
            chunk = DocumentChunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                doc_title=r["doc_title"],
                source_url=r["source_url"],
                section_header=r["section_header"],
                content=r["content"],
                token_count=r["token_count"],
                chunk_index=r["chunk_index"],
                allowed_emails=emails,
                allowed_domain=r["allowed_domain"],
                embedding=chunk_vec.tolist(),
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            scored_chunks.append((sim, chunk))

        # Sort descending by cosine similarity
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    async def sparse_search(
        self,
        query: str,
        top_k: int = 15,
        user_email: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """Perform BM25 full-text search via SQLite FTS5."""
        clean_terms = [t for t in query.split() if t.isalnum()]
        if not clean_terms:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in clean_terms)

        with self._lock, self._get_connection() as conn:
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                cursor = conn.execute(
                    f"""
                    SELECT c.chunk_id, c.doc_id, c.doc_title, c.source_url, c.section_header,
                           c.content, c.token_count, c.chunk_index, c.allowed_emails, c.allowed_domain,
                           c.embedding, c.created_at, c.updated_at, bm25(chunks_fts) as rank
                    FROM chunks_fts f
                    JOIN document_chunks c ON f.chunk_id = c.chunk_id
                    WHERE chunks_fts MATCH ? AND c.doc_id IN ({placeholders})
                    ORDER BY rank ASC
                    LIMIT ?;
                    """,
                    (fts_query, *doc_ids, top_k),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT c.chunk_id, c.doc_id, c.doc_title, c.source_url, c.section_header,
                           c.content, c.token_count, c.chunk_index, c.allowed_emails, c.allowed_domain,
                           c.embedding, c.created_at, c.updated_at, bm25(chunks_fts) as rank
                    FROM chunks_fts f
                    JOIN document_chunks c ON f.chunk_id = c.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?;
                    """,
                    (fts_query, top_k),
                )
            rows = cursor.fetchall()

        results: list[DocumentChunk] = []
        for r in rows:
            raw_blob = r["embedding"]
            emb = np.frombuffer(raw_blob, dtype=np.float32).tolist() if raw_blob else None
            emails = json.loads(r["allowed_emails"]) if r["allowed_emails"] else []
            results.append(
                DocumentChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    doc_title=r["doc_title"],
                    source_url=r["source_url"],
                    section_header=r["section_header"],
                    content=r["content"],
                    token_count=r["token_count"],
                    chunk_index=r["chunk_index"],
                    allowed_emails=emails,
                    allowed_domain=r["allowed_domain"],
                    embedding=emb,
                    created_at=datetime.fromisoformat(r["created_at"]),
                    updated_at=datetime.fromisoformat(r["updated_at"]),
                )
            )
        return results

    async def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents and chunk counts."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT d.doc_id, d.title, d.source_url, d.mime_type, d.synced_at,
                       COUNT(c.chunk_id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.doc_id = c.doc_id
                GROUP BY d.doc_id;
                """
            )
            return [dict(r) for r in cursor.fetchall()]

    async def delete_document(self, doc_id: str) -> None:
        """Cascade delete a document and its FTS5 index entries."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?;", (doc_id,))
            conn.execute("DELETE FROM document_chunks WHERE doc_id = ?;", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?;", (doc_id,))

    async def exact_cache_lookup(self, query_hash: str, user_email: str) -> dict[str, Any] | None:
        """Lookup exact match by SHA-256 hash."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT answer, citations, tokens_used, created_at
                FROM rag_query_cache
                WHERE query_hash = ? AND user_email = ?
                LIMIT 1;
                """,
                (query_hash, user_email),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "answer": row["answer"],
                "citations": json.loads(row["citations"]) if isinstance(row["citations"], str) else row["citations"],
                "tokens_used": row["tokens_used"],
                "created_at": row["created_at"],
            }

    async def semantic_cache_lookup(
        self,
        user_email: str,
        query_vector: list[float],
        threshold: float = 0.16,
    ) -> dict[str, Any] | None:
        """Perform vectorized cosine distance match on cached query embeddings."""
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None

        q_unit = q_vec / q_norm

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT answer, citations, tokens_used, created_at, embedding
                FROM rag_query_cache
                WHERE user_email = ?;
                """,
                (user_email,),
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        best_hit: dict[str, Any] | None = None
        min_dist = float("inf")

        for r in rows:
            blob = r["embedding"]
            c_vec = np.frombuffer(blob, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                sim = float(np.dot(q_unit, c_vec / c_norm))
                dist = 1.0 - sim
            else:
                dist = 1.0

            if dist < threshold and dist < min_dist:
                min_dist = dist
                best_hit = {
                    "answer": r["answer"],
                    "citations": json.loads(r["citations"]) if isinstance(r["citations"], str) else r["citations"],
                    "tokens_used": r["tokens_used"],
                    "distance": round(min_dist, 4),
                    "created_at": r["created_at"],
                }

        return best_hit

    async def store_cache(
        self,
        cache_id: str,
        query_text: str,
        query_hash: str,
        user_email: str,
        embedding: list[float],
        answer: str,
        citations: list[dict[str, Any]] | None,
        tokens_used: int = 0,
    ) -> None:
        """Persist a newly generated answer into the query cache."""
        emb_blob = np.array(embedding, dtype=np.float32).tobytes()
        citations_json = json.dumps(citations or [])
        now_str = datetime.now(timezone.utc).isoformat()

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO rag_query_cache (
                    cache_id, query_text, query_hash, user_email, embedding,
                    answer, citations, tokens_used, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_id) DO UPDATE SET
                    answer=excluded.answer,
                    citations=excluded.citations,
                    embedding=excluded.embedding,
                    tokens_used=excluded.tokens_used,
                    created_at=excluded.created_at;
                """,
                (
                    cache_id,
                    query_text,
                    query_hash,
                    user_email,
                    emb_blob,
                    answer,
                    citations_json,
                    tokens_used,
                    now_str,
                ),
            )

    async def invalidate_cache(self, doc_id: str | None = None) -> int:
        """Invalidate query cache entries."""
        with self._lock, self._get_connection() as conn:
            if doc_id:
                cursor = conn.execute(
                    "DELETE FROM rag_query_cache WHERE citations LIKE ?;",
                    (f"%{doc_id}%",),
                )
            else:
                cursor = conn.execute("DELETE FROM rag_query_cache;")
            return cursor.rowcount


_GLOBAL_STORE: EmbeddedVectorStore | None = None


def get_embedded_store(db_path: str | Path | None = None) -> EmbeddedVectorStore:
    """Singleton provider for EmbeddedVectorStore."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None or db_path is not None:
        _GLOBAL_STORE = EmbeddedVectorStore(db_path)
    return _GLOBAL_STORE
