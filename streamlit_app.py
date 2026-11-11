import json
import re
import urllib.request
import urllib.error
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

API_BASE = "http://localhost:8000"


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
        with urllib.request.urlopen(req, timeout=5) as resp:
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
    st.markdown('<div class="sidebar-header">Indexed Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Documents in memory available for search.</div>', unsafe_allow_html=True)

    docs = fetch_documents()

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
            parts = []
            if latency:
                parts.append(f"{round(latency)}ms")
            if cache_hit:
                parts.append(f"{cache_hit} cache hit")
            if tokens_saved:
                parts.append(f"{tokens_saved} tokens saved")
            if parts:
                st.caption(" · ".join(parts))

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
            parts = []
            if latency:
                parts.append(f"{round(latency)}ms")
            if cache_hit:
                parts.append(f"{cache_hit} cache hit")
            if tokens_saved:
                parts.append(f"{tokens_saved} tokens saved")
            if parts:
                st.caption(" · ".join(parts))

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": cleaned_final_answer,
            "citations": citations,
            "meta": meta_payload,
        })
