import json
import os
import re
import time
import urllib.error
import urllib.request

import streamlit as st

# Configure page with expanded sidebar by default
st.set_page_config(
    page_title="Knowledge RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS styling for clean white theme, permanently fixed compact sidebar, and centered layout
st.markdown(
    """
    <style>
        /* Base typography and container */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 820px;
        }

        /* Permanently fixed sidebar on the left */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc !important;
            border-right: 1px solid #e2e8f0 !important;
            width: 255px !important;
            min-width: 255px !important;
            max-width: 255px !important;
            transform: none !important;
            visibility: visible !important;
        }

        /* Hide all collapse and resize controls so sidebar is permanently open */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarResizeHandle"] {
            display: none !important;
        }

        /* Compact sizing for objects and text in the sidebar */
        section[data-testid="stSidebar"] .block-container {
            padding: 1.25rem 0.85rem !important;
        }
        .sidebar-header {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: #0f172a !important;
            margin-bottom: 0.15rem !important;
            letter-spacing: -0.01em !important;
        }
        .sidebar-caption {
            font-size: 0.72rem !important;
            color: #64748b !important;
            line-height: 1.3 !important;
            margin-bottom: 0.75rem !important;
        }
        .doc-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 7px 9px;
            margin-bottom: 6px;
        }
        .doc-title {
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            color: #1e293b !important;
            word-break: break-word;
            line-height: 1.25 !important;
        }
        .doc-meta {
            font-size: 0.70rem !important;
            color: #64748b !important;
            margin-top: 2px;
        }
        .doc-meta a {
            color: #2563eb !important;
            text-decoration: none;
        }
        .doc-meta a:hover {
            text-decoration: underline;
        }

        /* Compact buttons and inputs in sidebar */
        section[data-testid="stSidebar"] button {
            font-size: 0.75rem !important;
            padding: 0.25rem 0.5rem !important;
            min-height: 1.9rem !important;
            border-radius: 5px !important;
        }
        section[data-testid="stSidebar"] .stTextInput label {
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            color: #475569 !important;
            margin-bottom: 2px !important;
        }
        section[data-testid="stSidebar"] input {
            font-size: 0.75rem !important;
            padding: 0.25rem 0.5rem !important;
            height: 1.9rem !important;
            border-radius: 5px !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.6rem 0 !important;
        }

        /* Hide unwanted headers/menus */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stDeployButton { display: none !important; }

        /* Minimal loading animation */
        @keyframes pulse-subtle {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }
        .rag-loader {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #64748b;
            font-size: 0.88rem;
            padding: 6px 0;
            animation: pulse-subtle 1.4s ease-in-out infinite;
        }
        .rag-loader-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #2563eb;
            display: inline-block;
        }
    </style>
    """,
    unsafe_allow_html=True
)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def check_backend_online() -> bool:
    """Check if the local FastAPI backend is responsive."""
    try:
        req = urllib.request.Request(f"{API_BASE}/api/v1/query/health")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


# Curated demo knowledge base used when Docker/backend is offline
_MOCK_DOCUMENTS = [
    {
        "title": "Engineering Onboarding & System Setup",
        "chunk_count": 14,
        "source_url": "https://drive.google.com/open?id=demo_eng_onboarding",
    },
    {
        "title": "Company Remote Work & Benefits Policy 2026",
        "chunk_count": 9,
        "source_url": "https://drive.google.com/open?id=demo_remote_policy",
    },
    {
        "title": "Production CRAG Architecture & Telemetry Specs",
        "chunk_count": 18,
        "source_url": "https://drive.google.com/open?id=demo_crag_specs",
    },
]

_MOCK_KNOWLEDGE = {
    "remote": {
        "answer": (
            "The company operates under a flexible remote-first policy for all full-time "
            "employees. Team members may work fully remotely from anywhere within approved "
            "timezones or utilize regional co-working hubs.\n\n"
            "Key benefits include:\n"
            "- **Home Office Stipend:** $1,500 annual allowance for desk, chair, and ergonomic equipment.\n"
            "- **Internet & Utility Subsidy:** $75 monthly recurring reimbursement.\n"
            "- **Core Collaboration Hours:** 10:00 AM to 3:00 PM EST for synchronous meetings."
        ),
        "citations": [
            {
                "doc_title": "Company Remote Work & Benefits Policy 2026",
                "source_url": "https://drive.google.com/open?id=demo_remote_policy",
                "section_header": "Workplace Flexibility > Stipends & Equipment",
                "excerpt": "Full-time employees receive a $1,500 annual home office allowance and $75/mo internet subsidy...",
            }
        ],
        "tokens": 284,
        "cost_usd": 0.000028,
    },
    "onboarding": {
        "answer": (
            "New engineering hires complete a structured 2-week onboarding roadmap:\n\n"
            "1. **Week 1 (Environment & Provisioning):** Hardware provisioning, GitHub organization access, "
            "and 1-on-1 pairing with a dedicated onboarding buddy.\n"
            "2. **Week 2 (First Pull Request):** Local Docker container setup, reproducing unit test suites, "
            "and deploying a starter bugfix to the staging environment."
        ),
        "citations": [
            {
                "doc_title": "Engineering Onboarding & System Setup",
                "source_url": "https://drive.google.com/open?id=demo_eng_onboarding",
                "section_header": "Developer Milestones > Two-Week Roadmap",
                "excerpt": "New engineers complete local setup and merge a first PR within their initial 10 business days...",
            }
        ],
        "tokens": 312,
        "cost_usd": 0.000031,
    },
    "architecture": {
        "answer": (
            "WorkspaceRAG is a Corrective RAG (CRAG) system built with Python 3.11, FastAPI, "
            "PostgreSQL 16 (pgvector), and Google Gemini. Key pillars include:\n\n"
            "- **Hybrid Search:** Combines 768-dimensional dense vectors with BM25 full-text search via Reciprocal Rank Fusion (RRF).\n"
            "- **Two-Stage Reranking:** Cohere cross-encoder reranks top-15 retrieved candidates down to top-5.\n"
            "- **Two-Tier Caching:** Tier 1 SHA-256 exact hash (<2ms) and Tier 2 pgvector cosine distance (<=0.16 threshold).\n"
            "- **Cost Observability:** Gemini token usage harvested from response metadata, emitting child generation spans in Langfuse."
        ),
        "citations": [
            {
                "doc_title": "Production CRAG Architecture & Telemetry Specs",
                "source_url": "https://drive.google.com/open?id=demo_crag_specs",
                "section_header": "Architecture Overview > Core Components",
                "excerpt": "Hybrid retrieval with Reciprocal Rank Fusion and Langfuse OTel distributed tracing...",
            }
        ],
        "tokens": 420,
        "cost_usd": 0.000039,
    },
}


def clean_text(text: str) -> str:
    """Clean unwanted formatting, repetitive inline citations, and awkward math tags."""
    if not text:
        return ""
    # 1. Remove repetitive inline [Doc: ..., §...] tags
    cleaned = re.sub(r'\[Doc:\s*[^\]]+\]', '', text)

    # 2. Clean LaTeX math formatting (e.g. \text{LayerNorm}, $N = 6$, $d_{\text{model}} = 512$)
    # First peel out \text{...} wrappers
    cleaned = re.sub(r'\\text\{([^}]+)\}', r'\1', cleaned)
    # Remove math dollar signs around plain expressions
    cleaned = re.sub(r'\$([^$]+)\$', r'\1', cleaned)
    # Clean subscripts like d_{\text{model}} -> d_model or d_model
    cleaned = re.sub(r'_\{([^}]+)\}', r'_\1', cleaned)

    # 3. Add clean line breaks before embedded section headings (e.g. ". Encoder Stack:" -> "\n\n**Encoder Stack**:")
    cleaned = re.sub(r'(?<=[.!?])\s+([A-Z][A-Za-z\s]{2,25}:)', r'\n\n**\1**', cleaned)

    # 4. Normalize multiple whitespace and clean punctuation spacing
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def fetch_documents():
    """Retrieve list of indexed documents from PostgreSQL via backend API."""
    try:
        url = f"{API_BASE}/api/v1/sync/documents"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return []


def trigger_sync():
    """Trigger a full document synchronization with Google Drive."""
    try:
        url = f"{API_BASE}/api/v1/sync/full"
        req = urllib.request.Request(url, data=b"", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}
    return None


# ── COMPACT FIXED SIDE PANEL: INDEXED DOCUMENTS ─────────────────────────────
with st.sidebar:
    backend_is_live = check_backend_online()
    if backend_is_live:
        st.markdown(
            '<div style="font-size: 0.72rem; color: #16a34a; font-weight: 500; margin-bottom: 8px;">'
            '● Live Backend Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-size: 0.72rem; color: #d97706; font-weight: 500; margin-bottom: 8px;">'
            '● Demo Mode (Docker Offline)</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-header">Indexed Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Documents in memory available for search.</div>', unsafe_allow_html=True)

    docs = fetch_documents() if backend_is_live else _MOCK_DOCUMENTS

    if docs:
        for doc in docs:
            title = doc.get("title", "Untitled Document")
            chunks = doc.get("chunk_count", 0)
            url = doc.get("source_url", "")

            link_html = f' · <a href="{url}" target="_blank">Drive</a>' if url else ''
            st.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-title">{title}</div>
                    <div class="doc-meta">{chunks} chunks{link_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.caption("No documents indexed yet.")

    if st.button("Sync from Drive", use_container_width=True):
        with st.spinner("Syncing..."):
            result = trigger_sync()
            if result and "error" not in result:
                st.success(f"Synced {result.get('processed', 0)} doc(s).")
                st.rerun()
            else:
                st.error("Sync failed.")

    st.markdown("<hr>", unsafe_allow_html=True)
    user_email = st.text_input(
        "User Identity",
        value="rag-bot@personal-portfolio-471613.iam.gserviceaccount.com",
        help="Permissions validated against this email."
    )


# ── CENTERED HEADING IN CHAT SPACE ──────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 0.5rem 0 1.5rem 0;">
        <h1 style="font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; color: #0f172a; margin: 0;">
            Knowledge RAG
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("Sources", expanded=False):
                for c in msg["citations"]:
                    title = c.get("doc_title", "Document")
                    url = c.get("source_url", "")
                    sec = c.get("section_header")
                    sec_str = f" - Section: {sec}" if sec else ""
                    if url:
                        st.markdown(f"- [{title}]({url}){sec_str}")
                    else:
                        st.markdown(f"- {title}{sec_str}")
        if "meta" in msg and msg["meta"]:
            meta = msg["meta"]
            latency = meta.get("latency_ms")
            cache_hit = meta.get("cache_hit")
            tokens_saved = meta.get("tokens_saved", 0)
            cost_saved = meta.get("cost_saved_usd", 0.0)
            total_tokens = meta.get("total_tokens", 0)
            cost_usd = meta.get("cost_usd", 0.0)

            parts = []
            if latency:
                parts.append(f"{round(latency)}ms")
            if cache_hit:
                if cost_saved > 0:
                    parts.append(f"{cache_hit} cache hit ({tokens_saved} tokens / ${cost_saved:.6f} saved)")
                else:
                    parts.append(f"{cache_hit} cache hit ({tokens_saved} tokens saved)")
            else:
                if total_tokens:
                    parts.append(f"{total_tokens} tokens")
                if cost_usd > 0:
                    parts.append(f"${cost_usd:.6f}")
            if parts:
                st.caption(" · ".join(parts))

def _generate_mock_response(query: str) -> tuple[str, list[dict], dict]:
    """Generate realistic mock response and telemetry simulating the CRAG pipeline."""
    normalized = query.lower().strip()

    if "asked_queries" not in st.session_state:
        st.session_state.asked_queries = set()

    is_cache_hit = normalized in st.session_state.asked_queries
    st.session_state.asked_queries.add(normalized)

    topic = "architecture"
    if any(k in normalized for k in ["remote", "work", "pto", "leave", "stipend", "policy", "benefit"]):
        topic = "remote"
    elif any(k in normalized for k in ["onboard", "hire", "start", "first week", "setup", "laptop"]):
        topic = "onboarding"
    elif any(k in normalized for k in ["arch", "tech", "langfuse", "rag", "crag", "rrf", "cohere", "pgvector", "cache"]):
        topic = "architecture"

    entry = _MOCK_KNOWLEDGE.get(topic, _MOCK_KNOWLEDGE["architecture"])
    answer = entry["answer"]
    citations = entry["citations"]
    tokens = entry["tokens"]
    cost_usd = entry["cost_usd"]

    if is_cache_hit:
        meta = {
            "latency_ms": 11,
            "cache_hit": "exact",
            "tokens_saved": tokens,
            "cost_saved_usd": round(tokens * 0.000075 / 1000, 6),
            "total_tokens": 0,
            "cost_usd": 0.0,
            "citations": citations,
        }
    else:
        meta = {
            "latency_ms": 230,
            "cache_hit": None,
            "tokens_saved": 0,
            "cost_saved_usd": 0.0,
            "total_tokens": tokens,
            "cost_usd": cost_usd,
            "citations": citations,
        }
    return answer, citations, meta


# Handle new user input
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Minimal loading feedback immediately while waiting for stream
        message_placeholder.markdown(
            '<div class="rag-loader"><span class="rag-loader-dot"></span>Searching knowledge base...</div>',
            unsafe_allow_html=True
        )

        full_raw_text = ""
        meta_payload = None
        citations = []

        if not check_backend_online():
            # Graceful Offline Demo Mode when Docker/backend is offline
            mock_ans, citations, meta_payload = _generate_mock_response(prompt)
            words = mock_ans.split(" ")
            delay = 0.003 if meta_payload.get("cache_hit") else 0.015
            for word in words:
                full_raw_text += word + " "
                message_placeholder.markdown(clean_text(full_raw_text))
                time.sleep(delay)
        else:
            try:
                req_data = json.dumps({
                    "question": prompt,
                    "user_email": user_email.strip() or "rag-bot@personal-portfolio-471613.iam.gserviceaccount.com",
                    "session_id": "streamlit-session"
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{API_BASE}/api/v1/query/",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )

                with urllib.request.urlopen(req, timeout=60) as resp:
                    for line in resp:
                        decoded = line.decode("utf-8").strip()
                        if not decoded or not decoded.startswith("data: "):
                            continue

                        data = decoded[6:].strip()
                        if data == "[DONE]":
                            continue

                        # Check for JSON metadata frame
                        if data.startswith("{") and ("citations" in data or "cache_hit" in data):
                            try:
                                meta_payload = json.loads(data)
                                citations = meta_payload.get("citations", [])
                                continue
                            except Exception:
                                pass

                        # Accumulate token
                        full_raw_text += data + " "
                        # Display cleaned streaming text
                        message_placeholder.markdown(clean_text(full_raw_text))

            except urllib.error.URLError as e:
                full_raw_text = f"Connection error: {e.reason}"
                message_placeholder.markdown(full_raw_text)
            except Exception as e:
                full_raw_text = f"Error: {str(e)}"
                message_placeholder.markdown(full_raw_text)

        cleaned_final_answer = clean_text(full_raw_text)
        message_placeholder.markdown(cleaned_final_answer)

        # Show citations cleanly
        if citations:
            with st.expander("Sources", expanded=False):
                for c in citations:
                    title = c.get("doc_title", "Document")
                    url = c.get("source_url", "")
                    sec = c.get("section_header")
                    sec_str = f" - Section: {sec}" if sec else ""
                    if url:
                        st.markdown(f"- [{title}]({url}){sec_str}")
                    else:
                        st.markdown(f"- {title}{sec_str}")

        # Show telemetry badge
        if meta_payload:
            latency = meta_payload.get("latency_ms")
            cache_hit = meta_payload.get("cache_hit")
            tokens_saved = meta_payload.get("tokens_saved", 0)
            cost_saved = meta_payload.get("cost_saved_usd", 0.0)
            total_tokens = meta_payload.get("total_tokens", 0)
            cost_usd = meta_payload.get("cost_usd", 0.0)

            parts = []
            if latency:
                parts.append(f"{round(latency)}ms")
            if cache_hit:
                if cost_saved > 0:
                    parts.append(f"{cache_hit} cache hit ({tokens_saved} tokens / ${cost_saved:.6f} saved)")
                else:
                    parts.append(f"{cache_hit} cache hit ({tokens_saved} tokens saved)")
            else:
                if total_tokens:
                    parts.append(f"{total_tokens} tokens")
                if cost_usd > 0:
                    parts.append(f"${cost_usd:.6f}")
            if parts:
                st.caption(" · ".join(parts))

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": cleaned_final_answer,
            "citations": citations,
            "meta": meta_payload,
        })
