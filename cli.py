#!/usr/bin/env python3
"""WorkspaceRAG Command Line Interface (CLI).

Provides an interactive developer and DevOps CLI for inspecting, querying,
uploading documents, and running benchmark evaluations against the WorkspaceRAG system.

Usage:
    # Check backend connectivity & health
    python cli.py health

    # Query the knowledge base
    python cli.py query "What is the company's remote work equipment stipend?"

    # Upload & index a local document (PDF, Word, Markdown, CSV)
    python cli.py upload path/to/handbook.pdf

    # List indexed documents
    python cli.py docs

    # Run automated benchmark evaluation suite
    python cli.py eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_DEFAULT = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


def _print_banner() -> None:
    print("=" * 65)
    print("  WorkspaceRAG Enterprise CLI")
    print("=" * 65)


def cmd_health(args: argparse.Namespace) -> None:
    """Check backend health and service status."""
    url = f"{args.api}/api/v1/query/health"
    print(f"Connecting to {url}...")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Status:   OK (HTTP {resp.status})")
            print(f"Version:  {data.get('version', 'unknown')}")
            print("Backend is healthy and operational.")
    except urllib.error.URLError as e:
        print(f"Error: Could not connect to backend at {args.api}: {e.reason}")
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    """Execute a question against the RAG streaming API."""
    url = f"{args.api}/api/v1/query/"
    doc_ids = args.scope.split(",") if args.scope else None

    payload = {
        "question": args.question,
        "user_email": args.email,
        "session_id": f"cli-{int(time.time())}",
        "doc_ids": doc_ids,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"\nQuestion: {args.question}")
    if doc_ids:
        print(f"Scope:    {', '.join(doc_ids)}")
    print(f"User:     {args.email}")
    print("-" * 65)

    answer_parts: list[str] = []
    meta: dict[str, Any] = {}

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                chunk = decoded[6:].strip()
                if chunk == "[DONE]":
                    continue
                if chunk.startswith("{") and ("citations" in chunk or "latency_ms" in chunk):
                    try:
                        meta = json.loads(chunk)
                        continue
                    except Exception:
                        pass
                answer_parts.append(chunk)
                sys.stdout.write(chunk + " ")
                sys.stdout.flush()

        print("\n" + "-" * 65)

        # Print telemetry
        latency = meta.get("latency_ms")
        cache_hit = meta.get("cache_hit")
        tokens = meta.get("total_tokens", 0)
        cost_usd = meta.get("cost_usd", 0.0)
        citations = meta.get("citations", [])

        if cache_hit:
            print(f"Telemetry: CACHE HIT ({cache_hit}) | Saved: {meta.get('tokens_saved', 0)} tokens | Latency: {round(latency)}ms")
        else:
            print(f"Telemetry: Latency: {round(latency) if latency else 0}ms | Tokens: {tokens} | Cost: ${cost_usd:.6f}")

        if citations:
            print(f"\nSources ({len(citations)}):")
            for idx, c in enumerate(citations, start=1):
                title = c.get("doc_title", "Document")
                sec = c.get("section_header")
                sec_str = f" [§ {sec}]" if sec else ""
                print(f"  {idx}. {title}{sec_str}")
                if c.get("excerpt"):
                    print(f"     \"{c.get('excerpt')[:120]}...\"")

    except urllib.error.URLError as e:
        print(f"\nConnection error: {e.reason}")
        sys.exit(1)


def cmd_docs(args: argparse.Namespace) -> None:
    """List indexed documents."""
    url = f"{args.api}/api/v1/sync/documents"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            docs = json.loads(resp.read().decode("utf-8"))
            if not docs:
                print("No documents currently indexed in the knowledge base.")
                return

            print(f"\nIndexed Documents ({len(docs)} total):")
            print(f"{'Doc ID':<22} {'Chunks':<8} {'Title'}")
            print("-" * 65)
            for d in docs:
                doc_id = d.get("doc_id", "")[:20]
                chunks = str(d.get("chunk_count", 0))
                title = d.get("title", "Untitled")
                print(f"{doc_id:<22} {chunks:<8} {title}")
    except urllib.error.URLError as e:
        print(f"Error connecting to backend: {e.reason}")
        sys.exit(1)


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload and index a local file."""
    path = Path(args.file)
    if not path.exists() or not path.is_file():
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    url = f"{args.api}/api/v1/sync/upload"
    boundary = "----WorkspaceRAGBoundary" + str(int(time.time()))

    content_bytes = path.read_bytes()
    filename = path.name

    # Determine mime-type
    ext = path.suffix.lower()
    mime = "text/plain"
    if ext == ".pdf":
        mime = "application/pdf"
    elif ext == ".docx":
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext == ".csv":
        mime = "text/csv"
    elif ext in (".md", ".markdown"):
        mime = "text/markdown"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8") + content_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    print(f"Uploading and indexing '{filename}' ({len(content_bytes)} bytes)...")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            print("Upload complete!")
            print(f"  Doc ID:         {res.get('doc_id')}")
            print(f"  Chunks Created: {res.get('chunks_created')}")
            print(f"  Duration:       {res.get('duration_seconds')}s")
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"Upload failed (HTTP {e.code}): {err_msg}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        sys.exit(1)


def cmd_eval(args: argparse.Namespace) -> None:
    """Trigger the benchmark evaluation suite."""
    from backend.evals.run_benchmarks import run_evaluation
    import asyncio

    print(f"Running benchmark evaluation suite against {args.api}...")
    metrics = asyncio.run(run_evaluation(api_base_url=args.api, dataset_path=args.dataset))
    sys.exit(0 if metrics.get("passed") else 1)


def main() -> None:
    _print_banner()

    parser = argparse.ArgumentParser(
        description="WorkspaceRAG Enterprise Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api",
        default=API_DEFAULT,
        help=f"Base URL of the running WorkspaceRAG backend (default: {API_DEFAULT})",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # health
    p_health = subparsers.add_parser("health", help="Check backend connectivity and health status")
    p_health.set_defaults(func=cmd_health)

    # query
    p_query = subparsers.add_parser("query", help="Ask a question against the knowledge base")
    p_query.add_argument("question", help="The question string to ask")
    p_query.add_argument(
        "--email",
        default="rag-bot@personal-portfolio-471613.iam.gserviceaccount.com",
        help="User identity email for permission checking",
    )
    p_query.add_argument(
        "--scope",
        default=None,
        help="Comma-separated list of document IDs to scope search to",
    )
    p_query.set_defaults(func=cmd_query)

    # docs
    p_docs = subparsers.add_parser("docs", help="List all indexed documents and chunk statistics")
    p_docs.set_defaults(func=cmd_docs)

    # upload
    p_upload = subparsers.add_parser("upload", help="Upload and index a local file (PDF, Word, CSV, Markdown)")
    p_upload.add_argument("file", help="Path to local document file")
    p_upload.set_defaults(func=cmd_upload)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run the automated Ragas benchmark evaluation suite")
    p_eval.add_argument(
        "--dataset",
        default=None,
        help="Path to custom evaluation .jsonl dataset (defaults to golden_dataset.jsonl)",
    )
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
