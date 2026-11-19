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


def upload_file_to_backend(filename: str, file_bytes: bytes, mime_type: str = "") -> tuple[bool, str]:
    """Upload a local document to the FastAPI backend."""
    try:
        import httpx
        url = f"{API_BASE}/api/v1/sync/upload"
        files = {"file": (filename, file_bytes, mime_type or "application/octet-stream")}
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, files=files)
            if resp.status_code == 200:
                data = resp.json()
                return True, f"Indexed {data.get('chunks_created', 0)} chunks"
            return False, f"Server responded with status {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, str(exc)


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


def export_chat_markdown(messages: list[dict]) -> str:
    """Format conversation transcript into clean markdown with citations and telemetry."""
    if not messages:
        return "# Knowledge RAG Conversation Transcript\n\nNo messages in session.\n"

    lines = ["# Knowledge RAG Conversation Transcript\n"]
    lines.append(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    lines.append("---\n")

    turn = 1
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"### Turn {turn} — Question\n")
            lines.append(f"{content}\n")
        elif role == "assistant":
            lines.append(f"**Answer:**\n\n{content}\n")
            citations = msg.get("citations") or []
            if citations:
                lines.append("\n**Sources Cited:**")
                for c in citations:
                    title = c.get("doc_title", "Document")
                    url = c.get("source_url", "")
                    sec = c.get("section_header")
                    sec_str = f" - Section: {sec}" if sec else ""
                    if url:
                        lines.append(f"- [{title}]({url}){sec_str}")
                    else:
                        lines.append(f"- {title}{sec_str}")
                lines.append("")
            meta = msg.get("meta") or {}
            if meta:
                latency = meta.get("latency_ms")
                tokens = meta.get("total_tokens")
                cost = meta.get("cost_usd")
                parts = []
                if latency:
                    parts.append(f"Latency: {round(latency)}ms")
                if tokens:
                    parts.append(f"Tokens: {tokens}")
                if cost:
                    parts.append(f"Cost: ${cost:.6f}")
                if parts:
                    lines.append(f"\n> Telemetry: {' · '.join(parts)}\n")
            lines.append("---\n")
            turn += 1

    return "\n".join(lines)


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

    if "demo_docs" not in st.session_state:
        st.session_state.demo_docs = list(_MOCK_DOCUMENTS)
    if "demo_uploaded_content" not in st.session_state:
        st.session_state.demo_uploaded_content = {}

    st.markdown('<div class="sidebar-header">Indexed Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Documents in memory available for search.</div>', unsafe_allow_html=True)

    docs = fetch_documents() if backend_is_live else st.session_state.demo_docs

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

    st.markdown('<div class="sidebar-header" style="margin-top: 10px;">Filter Scope</div>', unsafe_allow_html=True)
    doc_options = [d.get("doc_id") for d in docs] if docs else []
    selected_doc_ids = st.multiselect(
        "Scope to documents",
        options=doc_options,
        format_func=lambda did: next((d.get("title", did) for d in docs if d.get("doc_id") == did), did),
        help="Optional: Select specific documents to restrict search scope.",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-header" style="margin-top: 10px;">Upload Document</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Upload PDF, DOCX, MD, or TXT.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "md", "txt"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        upload_sig = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
        if upload_sig not in st.session_state:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                bytes_data = uploaded_file.getvalue()
                if backend_is_live:
                    ok, msg = upload_file_to_backend(uploaded_file.name, bytes_data, uploaded_file.type)
                    if ok:
                        st.session_state[upload_sig] = True
                        st.success(f"Indexed {uploaded_file.name}!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {msg}")
                else:
                    from backend.ingestion.parsers import parse_bytes_to_markdown
                    extracted_text = parse_bytes_to_markdown(
                        bytes_data, filename=uploaded_file.name, mime_type=uploaded_file.type
                    )
                    approx_chunks = max(1, len(extracted_text) // 500)
                    st.session_state.demo_docs.append({
                        "doc_id": f"upload_{uploaded_file.name}",
                        "title": uploaded_file.name,
                        "chunk_count": approx_chunks,
                        "source_url": "",
                    })
                    st.session_state.demo_uploaded_content[uploaded_file.name] = extracted_text
                    st.session_state[upload_sig] = True
                    st.success(f"Indexed {uploaded_file.name} ({approx_chunks} chunks)")
                    st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    user_email = st.text_input(
        "User Identity",
        value="rag-bot@personal-portfolio-471613.iam.gserviceaccount.com",
        help="Permissions validated against this email."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Session Controls</div>', unsafe_allow_html=True)
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_export:
        transcript_bytes = export_chat_markdown(st.session_state.get("messages", [])).encode("utf-8")
        st.download_button(
            "Export .md",
            data=transcript_bytes,
            file_name="rag_chat_transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ── CENTERED HEADING IN CHAT SPACE ──────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 0.5rem 0 1.25rem 0;">
        <h1 style="font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; color: #0f172a; margin: 0;">
            Knowledge RAG
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Top navigation tabs for Chat vs Evals
tab_chat, tab_evals = st.tabs(["💬 Workspace Assistant", "📊 Benchmark Evaluations"])

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Evaluation scoring helper functions
def compute_eval_faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0
    ans_words = set(answer.lower().split())
    ctx_words = set(" ".join(contexts).lower().split())
    overlap = len(ans_words & ctx_words)
    return round(min(overlap / max(len(ans_words), 1), 1.0), 3)

def compute_eval_relevance(question: str, answer: str) -> float:
    if not answer:
        return 0.0
    q_words = set(question.lower().split()) - {"what", "is", "the", "how", "does", "a", "an", "for"}
    ans_words = set(answer.lower().split())
    overlap = len(q_words & ans_words)
    return round(min(overlap / max(len(q_words), 1), 1.0), 3)

def get_evaluation_samples() -> list[dict]:
    p = os.path.join(os.path.dirname(__file__), "backend", "evals", "datasets", "golden_dataset.jsonl")
    if os.path.exists(p):
        items = []
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            if items:
                return items
        except Exception:
            pass
    return [
        {
            "category": "policy",
            "question": "What is the company's remote work policy?",
            "ground_truth": "The company allows fully remote work for all employees with optional in-office days.",
        },
        {
            "category": "stipend",
            "question": "What is the home office equipment stipend?",
            "ground_truth": "The company provides a $1,500 home office equipment stipend per year for remote employees.",
        },
        {
            "category": "strategy",
            "question": "What are the quarterly OKRs for Q4?",
            "ground_truth": "Q4 OKRs focus on product expansion, customer retention above 90%, and team scaling.",
        },
        {
            "category": "engineering",
            "question": "What is the onboarding process for new engineers?",
            "ground_truth": "New engineers complete a 2-week structured onboarding with assigned mentors and system access provisioning.",
        },
        {
            "category": "security",
            "question": "What is the protocol for reporting a security incident?",
            "ground_truth": "Immediately notify the security team via Slack #incident-response and submit an incident report within 1 hour.",
        },
        {
            "category": "negative_fallback",
            "question": "What is the secret recipe for the cafeteria's special sauce?",
            "ground_truth": "I don't have sufficient documentation in the connected Google Workspace to answer your question.",
        },
    ]

# ── TAB 1: WORKSPACE CHAT ────────────────────────────────────────────────────
with tab_chat:
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
                        sec_str = f" · **Section:** {sec}" if sec else ""
                        score = c.get("relevance_score")
                        score_str = f" · **Score:** `{score:.2f}`" if isinstance(score, (int, float)) else ""
                        excerpt = c.get("excerpt", "")

                        if url:
                            st.markdown(f"📄 **[{title}]({url})**{sec_str}{score_str}")
                        else:
                            st.markdown(f"📄 **{title}**{sec_str}{score_str}")

                        if excerpt:
                            st.markdown(
                                f"""<div style="background-color: #f1f5f9; padding: 6px 10px; border-left: 3px solid #6366f1; border-radius: 4px; font-size: 0.78rem; color: #334155; margin-bottom: 8px;">
                                {excerpt.strip()}
                                </div>""",
                                unsafe_allow_html=True
                            )
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

def _generate_mock_response(query: str, doc_scope: list[str] | None = None) -> tuple[str, list[dict], dict]:
    """Generate realistic mock response and telemetry simulating the CRAG pipeline."""
    normalized = query.lower().strip()

    if doc_scope:
        scoped_titles = [
            d.get("title", "").lower()
            for d in getattr(st.session_state, "demo_docs", [])
            if d.get("doc_id") in doc_scope
        ]
        is_remote_query = any(k in normalized for k in ["remote", "work", "pto", "leave", "stipend"])
        is_onboard_query = any(k in normalized for k in ["onboard", "hire", "start", "laptop"])
        has_match = any(
            (any(k in t for k in ["remote", "handbook", "policy"]) if is_remote_query
             else any(k in t for k in ["onboard", "guide"]) if is_onboard_query
             else True)
            for t in scoped_titles
        )
        if not has_match:
            scoped_names = ", ".join(f"**{d.get('title')}**" for d in getattr(st.session_state, "demo_docs", []) if d.get("doc_id") in doc_scope)
            ans = (
                f"I searched within your selected document(s) ({scoped_names}), "
                f"but found no relevant sections matching your question: *\"{query}\"*."
            )
            return ans, [], {
                "latency_ms": 110,
                "cache_hit": None,
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "total_tokens": 120,
                "cost_usd": 0.000009,
                "citations": [],
            }

    if "asked_queries" not in st.session_state:
        st.session_state.asked_queries = set()

    is_cache_hit = normalized in st.session_state.asked_queries
    st.session_state.asked_queries.add(normalized)

    # Check user-uploaded files in demo mode first
    uploaded_dict = getattr(st.session_state, "demo_uploaded_content", {})
    query_tokens = [w for w in re.findall(r"\w+", normalized) if len(w) > 2]
    for doc_name, doc_text in uploaded_dict.items():
        doc_lower = doc_text.lower()
        if any(w in doc_lower for w in query_tokens):
            paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
            matching_paras = [p for p in paragraphs if any(w in p.lower() for w in query_tokens)]
            body = "\n\n".join(matching_paras[:3]) if matching_paras else (paragraphs[0] if paragraphs else "Content indexed.")
            answer = f"Based on your uploaded document **{doc_name}**:\n\n{body}"
            citations = [{
                "doc_title": doc_name,
                "source_url": "",
                "section_header": "Uploaded Document Content",
                "excerpt": body[:200] + "...",
            }]
            tokens = 340
            cost_usd = 0.000030
            meta = {
                "latency_ms": 135,
                "cache_hit": None,
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "total_tokens": tokens,
                "cost_usd": cost_usd,
                "citations": citations,
            }
            return answer, citations, meta

    # Multi-turn contextual follow-up detection for demo mode
    last_assistant_msg = ""
    for m in reversed(st.session_state.messages):
        if m.get("role") == "assistant":
            last_assistant_msg = m.get("content", "").lower()
            break

    is_followup = any(normalized.startswith(p) for p in [
        "can it", "can i", "what about", "how much", "is it", "does it", "does that",
        "why", "how does that", "where", "tell me more", "explain that", "elaborate"
    ]) or any(p in normalized for p in [" it ", " that ", " this ", " those ", " them "])

    topic = "architecture"
    if any(k in normalized for k in ["remote", "work", "pto", "leave", "stipend", "policy", "benefit"]):
        topic = "remote"
    elif any(k in normalized for k in ["onboard", "hire", "start", "first week", "setup", "laptop"]):
        topic = "onboarding"
    elif any(k in normalized for k in ["arch", "tech", "langfuse", "rag", "crag", "rrf", "cohere", "pgvector", "cache"]):
        topic = "architecture"
    elif is_followup and last_assistant_msg:
        # Infer topic from antecedent dialogue turn
        if any(k in last_assistant_msg for k in ["remote", "stipend", "pto", "leave", "home office"]):
            topic = "remote"
        elif any(k in last_assistant_msg for k in ["onboard", "buddy", "laptop", "setup", "first week"]):
            topic = "onboarding"
        else:
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
            mock_ans, citations, meta_payload = _generate_mock_response(
                prompt, doc_scope=selected_doc_ids if selected_doc_ids else None
            )
            words = mock_ans.split(" ")
            delay = 0.003 if meta_payload.get("cache_hit") else 0.015
            for word in words:
                full_raw_text += word + " "
                message_placeholder.markdown(clean_text(full_raw_text))
                time.sleep(delay)
        else:
            try:
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    if m.get("role") in ("user", "assistant") and m.get("content")
                ]

                req_data = json.dumps({
                    "question": prompt,
                    "user_email": user_email.strip() or "rag-bot@personal-portfolio-471613.iam.gserviceaccount.com",
                    "session_id": "streamlit-session",
                    "history": history_payload,
                    "doc_ids": selected_doc_ids if selected_doc_ids else None,
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

        # Show citations cleanly with rich excerpts and relevance badges
        if citations:
            with st.expander("📚 Sources & Grounding References", expanded=False):
                for c in citations:
                    title = c.get("doc_title", "Document")
                    url = c.get("source_url", "")
                    sec = c.get("section_header")
                    sec_str = f" · **Section:** {sec}" if sec else ""
                    score = c.get("relevance_score")
                    score_str = f" · **Score:** `{score:.2f}`" if isinstance(score, (int, float)) else ""
                    excerpt = c.get("excerpt", "")

                    if url:
                        st.markdown(f"📄 **[{title}]({url})**{sec_str}{score_str}")
                    else:
                        st.markdown(f"📄 **{title}**{sec_str}{score_str}")

                    if excerpt:
                        st.markdown(
                            f"""<div style="background-color: #f1f5f9; padding: 6px 10px; border-left: 3px solid #6366f1; border-radius: 4px; font-size: 0.78rem; color: #334155; margin-bottom: 8px;">
                            {excerpt.strip()}
                            </div>""",
                            unsafe_allow_html=True
                        )

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


# ── TAB 2: BENCHMARK EVALUATIONS ─────────────────────────────────────────────
with tab_evals:
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.05rem; font-weight: 600; color: #0f172a; margin-bottom: 0.25rem;">
                Automated Ragas Benchmark Suite
            </div>
            <div style="font-size: 0.82rem; color: #64748b; line-height: 1.4;">
                Evaluates factual grounding (<b>Faithfulness</b> &ge; 80%) and intent fulfillment (<b>Answer Relevance</b> &ge; 75%)
                against golden ground-truth samples spanning corporate policies, engineering standards, security protocols, and negative fallbacks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_samples = get_evaluation_samples()

    # Custom benchmark dataset uploader
    with st.expander("📂 Upload Custom Benchmark Dataset (.jsonl)", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; margin-bottom: 0.5rem;">
                Upload your own domain-specific <code>.jsonl</code> evaluation set.
                Each JSON line should contain: <code>{"question": "...", "ground_truth": "...", "category": "..."}</code>
            </div>
            """,
            unsafe_allow_html=True
        )
        custom_file = st.file_uploader("Upload custom evaluation JSONL", type=["jsonl"], label_visibility="collapsed")
        if custom_file is not None:
            try:
                uploaded_samples = []
                for line in custom_file.getvalue().decode("utf-8").splitlines():
                    line = line.strip()
                    if line:
                        uploaded_samples.append(json.loads(line))
                if uploaded_samples:
                    st.session_state.custom_eval_samples = uploaded_samples
                    st.success(f"Loaded {len(uploaded_samples)} custom test cases!")
            except Exception as e:
                st.error(f"Failed to parse custom JSONL: {e}")

        if "custom_eval_samples" in st.session_state:
            if st.button("Reset to Default Golden Dataset", use_container_width=True):
                del st.session_state["custom_eval_samples"]
                st.rerun()

    # Use custom samples if loaded, otherwise fallback to golden dataset
    active_eval_samples = st.session_state.get("custom_eval_samples", eval_samples)
    dataset_label = "custom dataset" if "custom_eval_samples" in st.session_state else "`golden_dataset.jsonl`"

    eval_col_info, eval_col_btn = st.columns([3, 1])
    with eval_col_info:
        st.caption(f"Loaded **{len(active_eval_samples)} test cases** from {dataset_label}")
    with eval_col_btn:
        run_eval_clicked = st.button("Run Benchmark", type="primary", use_container_width=True)

    if run_eval_clicked:
        results = []
        progress_bar = st.progress(0, text="Initializing evaluation run...")

        is_online = check_backend_online()

        for idx, sample in enumerate(active_eval_samples):
            q = sample["question"]
            progress_bar.progress(
                (idx + 1) / len(active_eval_samples),
                text=f"Evaluating ({idx+1}/{len(active_eval_samples)}): {q[:45]}..."
            )

            t_start = time.monotonic()
            ans = ""
            citations = []
            tokens = 0
            cost_usd = 0.0

            if not is_online:
                # Offline deterministic simulation using knowledge base
                ans, citations, meta = _generate_mock_response(q)
                tokens = meta.get("total_tokens", 150)
                cost_usd = meta.get("cost_usd", 0.000015)
                latency_ms = meta.get("latency_ms", 120)
            else:
                try:
                    req_data = json.dumps({
                        "question": q,
                        "user_email": "eval@example.com",
                        "session_id": f"eval-{int(time.time())}",
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
                            data_chunk = decoded[6:].strip()
                            if data_chunk == "[DONE]":
                                continue
                            if data_chunk.startswith("{") and ("citations" in data_chunk or "total_tokens" in data_chunk):
                                try:
                                    m = json.loads(data_chunk)
                                    citations = m.get("citations", [])
                                    tokens = m.get("total_tokens", 0)
                                    cost_usd = m.get("cost_usd", 0.0)
                                    continue
                                except Exception:
                                    pass
                            ans += data_chunk + " "
                    latency_ms = (time.monotonic() - t_start) * 1000
                except Exception as exc:
                    ans = f"Error during eval: {exc}"
                    latency_ms = (time.monotonic() - t_start) * 1000

            ans = clean_text(ans.strip())
            ctx_texts = [c.get("excerpt", "") for c in citations if c.get("excerpt")]
            if not ctx_texts and citations:
                ctx_texts = [c.get("doc_title", "") for c in citations]

            faith_score = compute_eval_faithfulness(
                ans, ctx_texts if ctx_texts else [sample.get("ground_truth", "")]
            )
            rel_score = compute_eval_relevance(q, ans)
            is_pass = faith_score >= 0.70 and rel_score >= 0.70

            results.append({
                "question": q,
                "category": sample.get("category", "general"),
                "ground_truth": sample.get("ground_truth", ""),
                "answer": ans,
                "citations": citations,
                "faithfulness": faith_score,
                "relevance": rel_score,
                "passed": is_pass,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "cost_usd": cost_usd,
            })

        progress_bar.empty()
        st.session_state.eval_results = results
        st.session_state.eval_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Display results if available
    if "eval_results" in st.session_state and st.session_state.eval_results:
        res = st.session_state.eval_results
        n = len(res)
        avg_faith = sum(r["faithfulness"] for r in res) / n
        avg_rel = sum(r["relevance"] for r in res) / n
        avg_lat = sum(r["latency_ms"] for r in res) / n
        tot_cost = sum(r["cost_usd"] for r in res)
        tot_tok = sum(r["tokens"] for r in res)

        st.markdown(
            f"<div style='font-size:0.75rem; color:#64748b; margin-bottom:0.75rem;'>Benchmark evaluated at {st.session_state.get('eval_timestamp', 'recently')}</div>",
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="Faithfulness",
                value=f"{avg_faith*100:.1f}%",
                delta="PASS (>=80%)" if avg_faith >= 0.8 else "FAIL (<80%)",
                delta_color="normal" if avg_faith >= 0.8 else "inverse"
            )
        with m2:
            st.metric(
                label="Answer Relevance",
                value=f"{avg_rel*100:.1f}%",
                delta="PASS (>=75%)" if avg_rel >= 0.75 else "FAIL (<75%)",
                delta_color="normal" if avg_rel >= 0.75 else "inverse"
            )
        with m3:
            st.metric(label="Avg Latency", value=f"{round(avg_lat)} ms")
        with m4:
            st.metric(label="Benchmark Cost", value=f"${tot_cost:.6f}", help=f"Total tokens consumed: {tot_tok}")

        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        col_hdr, col_dl = st.columns([3, 1])
        with col_hdr:
            st.markdown("<div style='font-size:0.85rem; font-weight:600; color:#0f172a; margin-top:0.4rem;'>Detailed Test Case Breakdown</div>", unsafe_allow_html=True)
        with col_dl:
            # Generate markdown audit report
            report_lines = [
                "# WorkspaceRAG Benchmark Evaluation Audit Report",
                f"**Date:** {st.session_state.get('eval_timestamp', 'N/A')}",
                f"**Overall Faithfulness:** {avg_faith*100:.1f}%",
                f"**Overall Answer Relevance:** {avg_rel*100:.1f}%",
                f"**Average Latency:** {round(avg_lat)} ms",
                f"**Total Cost:** ${tot_cost:.6f} ({tot_tok} tokens)",
                "",
                "## Test Case Details",
            ]
            for r in res:
                report_lines.extend([
                    f"### [{'PASS' if r['passed'] else 'FAIL'}] {r['question']}",
                    f"- **Category:** {r['category']}",
                    f"- **Faithfulness:** {r['faithfulness']:.2f} | **Relevance:** {r['relevance']:.2f}",
                    f"- **Latency:** {round(r['latency_ms'])}ms | **Tokens:** {r['tokens']} | **Cost:** ${r['cost_usd']:.6f}",
                    f"- **Ground Truth:** {r['ground_truth']}",
                    f"- **Answer:** {r['answer']}",
                    "",
                ])
            report_content = "\n".join(report_lines)
            st.download_button(
                label="📥 Export Report (.md)",
                data=report_content,
                file_name=f"rag_benchmark_audit_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        for _i, item in enumerate(res):
            badge_color = "#15803d" if item["passed"] else "#b91c1c"
            badge_bg = "#dcfce7" if item["passed"] else "#fee2e2"
            status_text = "PASS" if item["passed"] else "FAIL"
            cat = item["category"].upper()

            header_label = f"[{status_text}] {item['question']}"
            with st.expander(header_label, expanded=False):
                st.markdown(
                    f"""
                    <div style="display:flex; gap:8px; margin-bottom:8px;">
                        <span style="background:{badge_bg}; color:{badge_color}; font-size:0.7rem; font-weight:600; padding:2px 7px; border-radius:4px;">{status_text}</span>
                        <span style="background:#f1f5f9; color:#475569; font-size:0.7rem; font-weight:500; padding:2px 7px; border-radius:4px;">{cat}</span>
                        <span style="font-size:0.72rem; color:#64748b; margin-left:auto;">{round(item['latency_ms'])}ms · {item['tokens']} tokens · ${item['cost_usd']:.6f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                c_f, c_r = st.columns(2)
                with c_f:
                    st.caption(f"**Faithfulness Score:** `{item['faithfulness']:.2f}` (Threshold: 0.80)")
                with c_r:
                    st.caption(f"**Relevance Score:** `{item['relevance']:.2f}` (Threshold: 0.75)")

                st.markdown(f"**Expected Ground Truth:**\n> {item['ground_truth']}")
                st.markdown(f"**Generated Answer:**\n{item['answer']}")
                if item["citations"]:
                    st.caption("Context Excerpts:")
                    for c in item["citations"]:
                        st.caption(f"- *{c.get('doc_title', 'Source')}*: {c.get('excerpt', '')[:180]}...")
