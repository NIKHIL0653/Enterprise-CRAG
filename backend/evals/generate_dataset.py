"""
Synthetic Dataset Generator for WorkspaceRAG.

Samples ingested document chunks from PostgreSQL (or local text files) and uses
Google Gemini to synthesize high-quality (question, ground_truth, category) evaluation
pairs. Outputs in JSONL format for direct consumption by `run_benchmarks.py` and
optionally uploads dataset items to Langfuse Datasets.

Usage:
    # Generate from PostgreSQL chunks
    python backend/evals/generate_dataset.py --limit 10 --output backend/evals/datasets/synthetic_dataset.jsonl

    # Generate and upload to Langfuse Dataset
    python backend/evals/generate_dataset.py --limit 10 --upload-langfuse --dataset-name "rag-eval-v1"

    # Offline / dry-run mode (using mock chunks for testing/CI)
    python backend/evals/generate_dataset.py --dry-run --output backend/evals/datasets/mock_dataset.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.config import get_settings
from backend.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

_SYNTHESIS_PROMPT = """You are an expert QA evaluation engineer creating ground-truth benchmarks for an Enterprise RAG system.

Given the following documentation chunk, generate a realistic user query that can be directly and factually answered using ONLY the provided text, along with its ground-truth answer.

Document Title: {doc_title}
Section: {section_header}
Content:
---
{content}
---

Requirements:
1. The question must be natural, realistic, and specific (as asked by an employee in an enterprise).
2. The ground truth answer must be clear, concise, and fully supported by the excerpt.
3. Categorize the query into one of: "policy", "engineering", "strategy", "security", or "general".
4. Respond ONLY with valid JSON in this exact structure:
{{
    "question": "<realistic question>",
    "ground_truth": "<factual ground-truth answer>",
    "category": "<category>"
}}
"""


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON block from LLM output."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group())


async def fetch_sample_chunks(
    db_pool: Any,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Sample chunks from the PostgreSQL document_chunks table."""
    query = """
        SELECT chunk_id, doc_id, doc_title, section_header, content
        FROM document_chunks
        ORDER BY RANDOM()
        LIMIT $1;
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, limit)
        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "doc_title": row["doc_title"],
                "section_header": row["section_header"] or "Overview",
                "content": row["content"],
            }
            for row in rows
        ]


def get_mock_chunks(limit: int = 5) -> list[dict[str, Any]]:
    """Return synthetic mock chunks for testing and dry-run environments."""
    samples = [
        {
            "chunk_id": "chunk_mock_1",
            "doc_id": "doc_hr_1",
            "doc_title": "Employee Benefits Handbook 2026",
            "section_header": "Health and Wellness Stipend",
            "content": (
                "Employees are eligible for a $500 quarterly wellness stipend that can be "
                "used for gym memberships, fitness trackers, or mental wellness apps."
            ),
        },
        {
            "chunk_id": "chunk_mock_2",
            "doc_id": "doc_eng_1",
            "doc_title": "Production Deployment Guide",
            "section_header": "Zero-Downtime Migration Policy",
            "content": (
                "All database schema migrations must be strictly backward compatible. Add columns "
                "with defaults or as nullable first, deploy code, and backfill asynchronously."
            ),
        },
        {
            "chunk_id": "chunk_mock_3",
            "doc_id": "doc_sec_1",
            "doc_title": "Information Security Architecture",
            "section_header": "Access Control and MFA",
            "content": (
                "Hardware security keys (FIDO2/WebAuthn) are mandatory for all production cloud "
                "console access and database administration."
            ),
        },
    ]
    return samples[:limit]


async def synthesize_qa_for_chunk(
    llm_client: Any,
    chunk: dict[str, Any],
) -> dict[str, Any] | None:
    """Use Gemini to generate a (question, ground_truth) pair from a chunk."""
    prompt = _SYNTHESIS_PROMPT.format(
        doc_title=chunk.get("doc_title", "Document"),
        section_header=chunk.get("section_header", "General"),
        content=chunk.get("content", ""),
    )
    try:
        response = await llm_client.generate_content_async(prompt)
        parsed = extract_json(response.text.strip())
        return {
            "category": parsed.get("category", "general"),
            "question": parsed.get("question", "").strip(),
            "ground_truth": parsed.get("ground_truth", "").strip(),
            "user_email": "eval@example.com",
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "doc_title": chunk.get("doc_title"),
        }
    except Exception as exc:
        log.warning("synthesis_failed_for_chunk", chunk_id=chunk.get("chunk_id"), error=str(exc))
        return None


async def generate_dataset(
    limit: int = 10,
    output_path: str = "backend/evals/datasets/synthetic_dataset.jsonl",
    dry_run: bool = False,
    upload_langfuse: bool = False,
    dataset_name: str = "rag-synthetic-evals",
) -> list[dict[str, Any]]:
    """
    Generate synthetic Q&A evaluation dataset from chunks.

    Args:
        limit: Maximum number of chunks to sample.
        output_path: Target JSONL destination.
        dry_run: If True, uses mock chunks and generates synthetic items without external calls.
        upload_langfuse: If True, uploads generated samples to Langfuse Datasets.
        dataset_name: Name of target dataset in Langfuse.

    Returns:
        List of generated dataset item dictionaries.
    """
    items: list[dict[str, Any]] = []

    if dry_run:
        log.info("generate_dataset_dry_run_mode", limit=limit)
        mock_chunks = get_mock_chunks(limit)
        for _i, chunk in enumerate(mock_chunks, 1):
            items.append(
                {
                    "category": "engineering" if "Production" in chunk["doc_title"] else "policy",
                    "question": f"What is described in {chunk['section_header']}?",
                    "ground_truth": chunk["content"],
                    "user_email": "eval@example.com",
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "doc_title": chunk["doc_title"],
                }
            )
    else:
        # Connect to DB and Gemini
        from backend.services.database import create_pool
        import google.generativeai as genai

        db_pool = await create_pool()
        try:
            chunks = await fetch_sample_chunks(db_pool, limit=limit)
        finally:
            await db_pool.close()

        if not chunks:
            log.warning("no_chunks_found_in_database")
            return []

        genai.configure(api_key=settings.GEMINI_API_KEY)
        llm = genai.GenerativeModel(settings.GEMINI_MODEL)

        for chunk in chunks:
            item = await synthesize_qa_for_chunk(llm, chunk)
            if item and item["question"] and item["ground_truth"]:
                items.append(item)

    # Save to JSONL
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    log.info("dataset_generation_saved", count=len(items), path=str(out))

    # Optional Langfuse dataset upload
    if upload_langfuse and items and not dry_run:
        try:
            from langfuse import Langfuse
            lf = Langfuse(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=settings.LANGFUSE_HOST,
            )
            for item in items:
                lf.create_dataset_item(
                    dataset_name=dataset_name,
                    input={"question": item["question"], "category": item["category"]},
                    expected_output=item["ground_truth"],
                    metadata={
                        "chunk_id": item.get("chunk_id"),
                        "doc_id": item.get("doc_id"),
                        "doc_title": item.get("doc_title"),
                    },
                )
            lf.flush()
            log.info("uploaded_to_langfuse_dataset", dataset=dataset_name, count=len(items))
        except Exception as exc:
            log.error("langfuse_dataset_upload_failed", error=str(exc))

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic evaluation dataset generator for WorkspaceRAG.")
    parser.add_argument("--limit", type=int, default=10, help="Number of chunks to sample.")
    parser.add_argument(
        "--output",
        type=str,
        default="backend/evals/datasets/synthetic_dataset.jsonl",
        help="Output path for .jsonl dataset.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate mock samples without DB or Gemini calls.")
    parser.add_argument("--upload-langfuse", action="store_true", help="Upload items to Langfuse Datasets.")
    parser.add_argument("--dataset-name", type=str, default="rag-synthetic-evals", help="Langfuse dataset name.")

    args = parser.parse_args()
    items = asyncio.run(
        generate_dataset(
            limit=args.limit,
            output_path=args.output,
            dry_run=args.dry_run,
            upload_langfuse=args.upload_langfuse,
            dataset_name=args.dataset_name,
        )
    )
    print(f"Generated {len(items)} evaluation samples saved to {args.output}")


if __name__ == "__main__":
    main()
