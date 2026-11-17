"""
Pre-flight diagnostics and verification CLI for WorkspaceRAG.

Checks:
1. Python version compatibility (3.11+).
2. Configuration & environment variables.
3. PostgreSQL connectivity & vector extension status.
4. Google Gemini LLM & embedding API readiness.
5. Cohere Reranker API validity.
6. Langfuse Cloud observability credentials.

Usage:
    python -m backend.verify_setup
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.config import get_settings

# ANSI color formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_check(name: str, passed: bool, message: str, warning: bool = False) -> None:
    """Print a single check line with colored status tag."""
    if passed:
        tag = f"{GREEN}[PASS]{RESET}"
    elif warning:
        tag = f"{YELLOW}[WARN]{RESET}"
    else:
        tag = f"{RED}[FAIL]{RESET}"
    print(f"  {tag} {BOLD}{name:<32}{RESET} {message}")


async def check_python_version() -> bool:
    """Verify that Python version is 3.11 or greater."""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    passed = (v.major, v.minor) >= (3, 11)
    print_check(
        "Python Version",
        passed,
        f"Detected {ver_str} (requires >= 3.11)",
    )
    return passed


async def check_settings() -> tuple[bool, Any]:
    """Validate settings from environment or .env file."""
    try:
        settings = get_settings()
        print_check("Configuration & .env", True, f"Loaded settings for {settings.APP_NAME}")
        return True, settings
    except Exception as exc:
        print_check("Configuration & .env", False, f"Settings validation failed: {exc}")
        return False, None


async def check_database(database_url: str) -> bool:
    """Check PostgreSQL connection, pgvector extension, and tables."""
    try:
        import asyncpg
        conn = await asyncpg.connect(database_url, timeout=5)
        try:
            # Check vector extension
            ext_row = await conn.fetchrow(
                "SELECT extname FROM pg_extension WHERE extname = 'vector';"
            )
            has_vector = ext_row is not None

            # Check document count
            doc_count = 0
            try:
                row = await conn.fetchrow("SELECT COUNT(*) AS c FROM documents;")
                doc_count = row["c"] if row else 0
            except Exception:
                pass

            msg = f"Connected successfully. pgvector: {'installed' if has_vector else 'missing'}, documents: {doc_count}"
            print_check("PostgreSQL + pgvector", has_vector, msg)
            return has_vector
        finally:
            await conn.close()
    except Exception as exc:
        print_check("PostgreSQL + pgvector", False, f"Connection failed: {exc}")
        return False


async def check_gemini(api_key: str, model_name: str) -> bool:
    """Verify Gemini API connectivity with a lightweight test call."""
    if not api_key or api_key.startswith("test_"):
        print_check("Google Gemini API", False, "Missing or dummy GEMINI_API_KEY", warning=True)
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = await model.generate_content_async("Respond with 'OK'")
        text = response.text.strip()
        passed = len(text) > 0
        print_check("Google Gemini API", passed, f"Model {model_name} responded: '{text[:20]}'")
        return passed
    except Exception as exc:
        print_check("Google Gemini API", False, f"API test call failed: {exc}", warning=True)
        return False


async def check_cohere(api_key: str) -> bool:
    """Verify Cohere Reranker API key."""
    if not api_key or api_key.startswith("test_"):
        print_check("Cohere Reranker API", False, "Missing or dummy COHERE_API_KEY", warning=True)
        return False
    try:
        import cohere
        client = cohere.AsyncClient(api_key)
        res = await client.rerank(
            model="rerank-v3.5",
            query="test query",
            documents=["Document 1", "Document 2"],
            top_n=1,
        )
        passed = len(res.results) > 0
        print_check("Cohere Reranker API", passed, f"Successfully reranked candidate (score: {res.results[0].relevance_score:.3f})")
        return passed
    except Exception as exc:
        print_check("Cohere Reranker API", False, f"Rerank test failed: {exc}", warning=True)
        return False


async def check_langfuse(secret_key: str, public_key: str, host: str) -> bool:
    """Verify Langfuse credentials."""
    if not secret_key or secret_key.startswith("test_"):
        print_check("Langfuse Observability", False, "Missing or dummy LANGFUSE credentials", warning=True)
        return False
    try:
        from langfuse import Langfuse
        lf = Langfuse(secret_key=secret_key, public_key=public_key, host=host)
        auth_ok = lf.auth_check()
        print_check("Langfuse Observability", auth_ok, f"Authenticated with {host}")
        return auth_ok
    except Exception as exc:
        print_check("Langfuse Observability", False, f"Auth check failed: {exc}", warning=True)
        return False


async def run_diagnostics() -> int:
    """Run full diagnostic pre-flight suite and print summary."""
    print("\n" + "=" * 65)
    print(f" {BOLD}{CYAN}WorkspaceRAG Pre-Flight Setup Diagnostics{RESET}")
    print("=" * 65 + "\n")

    results = []

    # 1. Python
    py_ok = await check_python_version()
    results.append(py_ok)

    # 2. Settings
    cfg_ok, settings = await check_settings()
    results.append(cfg_ok)

    if settings:
        # 3. Database
        db_ok = await check_database(settings.DATABASE_URL)
        results.append(db_ok)

        # 4. Gemini
        gemini_ok = await check_gemini(settings.GEMINI_API_KEY, settings.GEMINI_MODEL)
        results.append(gemini_ok)

        # 5. Cohere
        cohere_ok = await check_cohere(settings.COHERE_API_KEY)
        results.append(cohere_ok)

        # 6. Langfuse
        lf_ok = await check_langfuse(
            settings.LANGFUSE_SECRET_KEY,
            settings.LANGFUSE_PUBLIC_KEY,
            settings.LANGFUSE_HOST,
        )
        results.append(lf_ok)

    print("\n" + "-" * 65)
    critical_ok = py_ok and cfg_ok
    if critical_ok:
        print(f"  {GREEN}{BOLD}CORE ENVIRONMENT CHECKS PASSED{RESET}")
        print(f"  Run '{CYAN}.\\start.ps1{RESET}' to launch the full container stack.")
    else:
        print(f"  {RED}{BOLD}CRITICAL SETUP CHECKS FAILED{RESET}")
        print("  Please check your configuration or dependencies above.")
    print("=" * 65 + "\n")

    return 0 if critical_ok else 1


def main() -> None:
    code = asyncio.run(run_diagnostics())
    sys.exit(code)


if __name__ == "__main__":
    main()
