# app.py
"""
RAG Chatbot — Fixed last_sources storage & display
Stores retriever results as serializable dicts in st.session_state["last_sources"]
and displays chunk text + metadata (if present).
Keeps previous behaviors: recent-first, user prompt tokens, safe clearing, vibrant UI.
"""

import streamlit as st
import openai
import tiktoken
from datetime import datetime
import html
import tempfile
import os
from typing import List
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Chatbot — Final (fixed sources)", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helpers & Defaults
# -------------------------
DEFAULT_RATES = {
    "gpt-3.5-turbo": 0.002,
    "gpt-4o-mini": 0.003,
    "gpt-4o": 0.03,
}

def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def ensure_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # chronological
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "model_rates" not in st.session_state:
        st.session_state.model_rates = DEFAULT_RATES.copy()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(st.session_state.model_rates.keys())[0]
    if "_clear_input_next_run" not in st.session_state:
        st.session_state["_clear_input_next_run"] = False
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False
    # last_sources will be a list of dicts with keys: page_content, metadata, id, score
    if "last_sources" not in st.session_state:
        st.session_state["last_sources"] = []

def encoding_for_model(model_name):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str) -> int:
    enc = encoding_for_model(model)
    return len(enc.encode(text))

def calc_cost_from_tokens(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    rate_per_1k = st.session_state.model_rates.get(model, list(st.session_state.model_rates.values())[0])
    return (prompt_tokens + completion_tokens) * (rate_per_1k / 1000.0)

def make_sparkline_svg(values: List[float], width: int = 220, height: int = 44, stroke="#7c3aed") -> str:
    if not values:
        values = [0.0]
    mx = max(values); mn = min(values)
    rng = mx - mn if mx != mn else 1.0
    left, right, top, bottom = 6, 6, 6, 6
    inner_w = width - left - right; inner_h = height - top - bottom
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = left + (i / (n - 1 if n > 1 else 1)) * inner_w
        y = top + (1 - (v - mn) / rng) * inner_h
        pts.append(f"{x:.2f},{y:.2f}")
    pts_str = " ".join(pts)
    svg = f'''
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="sg" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="{stroke}" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <polyline fill="none" stroke="{stroke}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" points="{pts_str}" />
      <polygon points="{pts_str} {width-right},{height-bottom} {left},{height-bottom}" fill="url(#sg)" opacity="0.95"/>
    </svg>
    '''
    return svg

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# -------------------------
# Init & clear flag BEFORE widgets
# -------------------------
ensure_state()
if st.session_state.get("_clear_input_next_run", False):
    st.session_state["input_query"] = ""
    st.session_state["_clear_input_next_run"] = False

# -------------------------
# CSS (scoped)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial; }

/* header */
.rb-top { background: linear-gradient(90deg,#7c3aed,#06b6d4); color: white; padding:14px; border-radius:10px; margin-bottom:12px; box-shadow: 0 12px 36px rgba(12,18,45,0.14); }
.rb-top h1 { margin:0; font-size:18px; font-weight:700; }

/* small header styles */
.rb-card-header { display:inline-block; padding:8px 10px; margin-bottom:8px; border-radius:8px; }
.rb-card-header h3 { margin:0; font-size:15px; font-weight:700; color:#0b1724; padding-left:8px; position:relative; }
.rb-card-header.teal h3::before { content:""; position:absolute; left:-10px; top:0; bottom:0; width:6px; background: linear-gradient(180deg,#06b6d4,#7dd3fc); border-radius:3px; }
.rb-card-header.indigo h3::before { content:""; position:absolute; left:-10px; top:0; bottom:0; width:6px; background: linear-gradient(180deg,#7c3aed,#a78bfa); border-radius:3px; }
.rb-card-header.gold h3::before { content:""; position:absolute; left:-10px; top:0; bottom:0; width:6px; background: linear-gradient(180deg,#f59e0b,#fbbf24); border-radius:3px; }

/* card */
.rb-card { background: rgba(255,255,255,0.94); border-radius:12px; padding:10px; margin-bottom:12px; box-shadow: 0 8px 20px rgba(10,15,30,0.06); border:1px solid rgba(10,11,20,0.04); }

/* chat */
.rb-chat-scroll { max-height:70vh; overflow-y:auto; padding-right:6px; }
.rb-chat-pair { display:block; margin-bottom:10px; padding:8px 6px; border-bottom:1px dashed rgba(10,11,20,0.03); }
.rb-msg { padding:10px 12px; border-radius:10px; background: linear-gradient(180deg,#ffffff,#fbfdff); border:1px solid rgba(10,11,20,0.04); font-size:15px; white-space:pre-wrap; }
.rb-meta { color:#475569; font-size:12px; margin-top:6px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }

.rb-badge { padding:6px 10px; border-radius:999px; font-weight:700; font-size:13px; display:inline-block; background:rgba(12,18,45,0.06); }
.rb-cost { padding:6px 10px; border-radius:999px; font-weight:700; font-size:13px; display:inline-block; background: linear-gradient(90deg,#fde68a,#fb923c); color:#4b2e05; }

.details-inline summary { display:inline; font-weight:700; cursor:pointer; }
details > summary::-webkit-details-marker { display:none; }
details[open] > summary::after { content: " ▲"; color:#64748b; font-weight:500; }
details > summary::after { content: " ▼"; color:#64748b; font-weight:500; }

@media (max-width:900px) { .rb-chat-scroll { max-height:48vh; } }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="rb-top">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <h1 style="margin:0">RAG Chatbot - DocumentAI(fixed sources)</h1>
      <div style="opacity:0.95;font-size:13px">Follow Q&A.</div>
    </div>
    <div style="font-size:13px;opacity:0.95">Context Driven • Chat Assistant</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown('<div class="rb-card">', unsafe_allow_html=True)
    st.subheader("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Set here or via OPENAI_API_KEY env var")
    model_list = list(st.session_state.model_rates.keys())
    sel_idx = model_list.index(st.session_state.selected_model) if st.session_state.selected_model in model_list else 0
    selected_model = st.selectbox("Model", options=model_list, index=sel_idx, label_visibility="collapsed")
    st.session_state.selected_model = selected_model
    st.markdown("**Rates (USD per 1K tokens)**")
    for m in model_list:
        st.session_state.model_rates[m] = st.number_input(f"{m} ($/1K)", value=float(st.session_state.model_rates[m]), step=0.0001, format="%f", key=f"rate_{m}")
    st.write("")
    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.session_state.total_cost = 0.0
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Layout columns
# -------------------------
left, right = st.columns([3, 1], gap="small")

# LEFT: Upload, Chat Input, Conversation
with left:
    # Document area: uploader on same line as browse (uploader includes browse)
    with st.container():
        st.markdown('<div class="rb-card-header indigo"><h3>Document / Index</h3></div>', unsafe_allow_html=True)
        cols_u = st.columns([3, 1], gap="small")
        with cols_u[0]:
            uploaded = st.file_uploader("Upload PDF (optional)", type=["pdf"], help="Index a PDF for RAG retrieval")
        with cols_u[1]:
            st.write("")  # placeholder to keep same-line spacing
        # Process button next line
        cols_proc = st.columns([1, 5], gap="small")
        with cols_proc[0]:
            process_pressed = False
            if uploaded:
                process_pressed = st.button("Process Document")
            else:
                st.write("")
        with cols_proc[1]:
            st.write("")
        # Status line after processing
        if st.session_state.get("rag_ready", False):
            st.markdown('<div style="padding-top:6px;color:green;font-weight:700">Document indexed — RAG ready</div>', unsafe_allow_html=True)
        else:
            if uploaded:
                st.markdown('<div style="padding-top:6px;color:#475569">Click "Process Document" to index.</div>', unsafe_allow_html=True)

    # Chat input side-by-side
    with st.container():
        st.markdown('<div class="rb-card-header gold"><h3>Ask • Chat</h3></div>', unsafe_allow_html=True)
        cols_input = st.columns([5,1], gap="small")
        with cols_input[0]:
            user_query = st.text_input("Question", key="input_query", placeholder="Ask about the document or general question...", label_visibility="visible")
        with cols_input[1]:
            send_clicked = st.button("Send")

    # Conversation: recent-first pairs
    with st.container():
        st.markdown('<div class="rb-card-header teal"><h3>Conversation — recent first</h3></div>', unsafe_allow_html=True)
        st.markdown('<div class="rb-card"><div class="rb-chat-scroll">', unsafe_allow_html=True)

        history = st.session_state.history
        if not history:
            st.info("No messages yet — send a question to start.")
        else:
            pairs = []
            i = 0
            n = len(history)
            while i < n:
                e = history[i]
                if e.get("role") == "user":
                    user_e = e
                    assistant_e = None
                    if i + 1 < n and history[i+1].get("role") == "assistant":
                        assistant_e = history[i+1]
                        i += 2
                    else:
                        i += 1
                    pairs.append((user_e, assistant_e))
                else:
                    pairs.append((None, e))
                    i += 1

            for user_e, assistant_e in reversed(pairs):
                if user_e:
                    content = user_e.get("content","")
                    safe_content = html.escape(content).replace("\n","<br>")
                    ts = user_e.get("ts","")
                    prompt_tokens_user = user_e.get("prompt_tokens", 0) or 0
                    md_user = f'''
<div class="rb-chat-pair">
  <div style="font-weight:700;margin-bottom:6px">👤 You — <span style="font-weight:600;color:#475569;font-size:12px">{ts}</span></div>
  <div class="rb-msg">{safe_content}</div>
  <div class="rb-meta"><span class="rb-badge">Prompt tokens: {prompt_tokens_user}</span></div>
</div>
'''
                    st.markdown(md_user, unsafe_allow_html=True)

                if assistant_e:
                    content = assistant_e.get("content","")
                    ts = assistant_e.get("ts","")
                    pt = assistant_e.get("prompt_tokens",0) or 0
                    ct = assistant_e.get("completion_tokens",0) or 0
                    tt = assistant_e.get("total_tokens", pt + ct)
                    cost = assistant_e.get("cost", 0.0) or 0.0

                    threshold = 360
                    if len(content) > threshold:
                        excerpt = html.escape(content[:threshold]).replace("\n","<br>")
                        rest = html.escape(content[threshold:]).replace("\n","<br>")
                        message_html = f'''
<div class="rb-chat-pair">
  <div style="font-weight:700;margin-bottom:6px">🤖 Assistant — <span style="font-weight:600;color:#475569;font-size:12px">{ts}</span></div>
  <div class="rb-msg">{excerpt}... <details class="details-inline"><summary>Read more</summary><div style="margin-top:8px">{rest}</div></details></div>
  <div class="rb-meta"><span class="rb-cost">${cost:.6f}</span><span style="margin-left:8px;color:#475569;font-size:12px">Tokens: {tt}</span></div>
</div>
'''
                    else:
                        safe_content = html.escape(content).replace("\n","<br>")
                        message_html = f'''
<div class="rb-chat-pair">
  <div style="font-weight:700;margin-bottom:6px">🤖 Assistant — <span style="font-weight:600;color:#475569;font-size:12px">{ts}</span></div>
  <div class="rb-msg">{safe_content}</div>
  <div class="rb-meta"><span class="rb-cost">${cost:.6f}</span><span style="margin-left:8px;color:#475569;font-size:12px">Tokens: {tt}</span></div>
</div>
'''
                    st.markdown(message_html, unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

    # === Sources panel: now reads the serializable last_sources list stored in session_state ===
    if "last_sources" in st.session_state and st.session_state.get("last_sources"):
        with st.container():
            st.markdown('<div class="rb-card-header indigo"><h3>Sources (last answer)</h3></div>', unsafe_allow_html=True)
            # last_sources is a list of dicts we stored during retrieval
            for i, src in enumerate(st.session_state.get("last_sources", [])):
                page_content = src.get("page_content", "")
                metadata = src.get("metadata", {}) or {}
                src_id = src.get("id", None)
                score = src.get("score", None)
                snippet = html.escape(page_content[:700]).replace("\n", " ")
                # build a small metadata string for display
                meta_parts = []
                # common metadata keys: source, path, page, document_id, filename
                for key in ("source", "path", "filename", "page", "document_id"):
                    if key in metadata:
                        meta_parts.append(f"{key}: {metadata[key]}")
                if src_id:
                    meta_parts.append(f"id: {src_id}")
                if score is not None:
                    meta_parts.append(f"score: {score:.3f}")
                meta_str = " • ".join(meta_parts) if meta_parts else ""
                if meta_str:
                    st.markdown(f"**Source {i+1}:** {snippet}...  \n<small style='color:#475569'>{meta_str}</small>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Source {i+1}:** {snippet}...", unsafe_allow_html=True)

# RIGHT: summary, sparkline, last call
with right:
    with st.container():
        st.markdown('<div class="rb-card"><h3 style="margin:0 0 8px 0">Session Summary</h3>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-weight:700">Total cost (session)</div><div style="font-size:20px;color:#7c3aed;margin-top:6px">${st.session_state.total_cost:.6f}</div><div style="margin-top:8px">Model: {st.session_state.selected_model}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    assistant_costs = [m.get("cost",0.0) for m in st.session_state.history if m.get("role") == "assistant"]
    if assistant_costs and len(assistant_costs) == 1:
        assistant_costs = assistant_costs + assistant_costs
    svg = make_sparkline_svg(assistant_costs, width=220, height=48, stroke="#7c3aed")
    st.markdown('<div class="rb-card" style="margin-top:10px">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;margin-bottom:6px">Spend (session)</div>', unsafe_allow_html=True)
    st.markdown(svg, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="rb-card" style="margin-top:10px">', unsafe_allow_html=True)
    st.subheader("Last call details")
    if st.session_state.history:
        last_assistant = next((e for e in reversed(st.session_state.history) if e.get("role") == "assistant"), None)
        if last_assistant:
            st.write("Time:", last_assistant.get("ts"))
            st.write("Prompt tokens:", last_assistant.get("prompt_tokens", 0))
            st.write("Completion tokens:", last_assistant.get("completion_tokens", 0))
            st.write("Total tokens:", last_assistant.get("total_tokens", 0))
            st.write("Cost (last):", f"${last_assistant.get('cost', 0.0):.6f}")
        else:
            st.info("No assistant calls yet.")
    else:
        st.info("No calls yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Document processing (Process Document)
# -------------------------
if uploaded and ('process_pressed' in locals() and process_pressed):
    if not api_key:
        st.error("Please enter OpenAI API key in sidebar before processing the document.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        rag = RAGPipeline(api_key, model_name=st.session_state.selected_model)
        docs = rag.load_documents(file_path=tmp_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        rag.build_vectorstore(docs)
        st.session_state["rag"] = rag
        st.session_state["rag_ready"] = True
        # inline status message shown in UI on next render

# -------------------------
# Send / Query handling (with robust source serialization & token accounting)
# -------------------------
if 'send_clicked' in locals() and send_clicked:
    input_text = st.session_state.get("input_query", "").strip()
    if input_text != "":
        if "rag" not in st.session_state:
            st.error("RAG index not initialized. Upload & Process a document first.")
        elif not api_key:
            st.error("OpenAI API key missing (sidebar).")
        else:
            openai.api_key = api_key
            q = input_text

            # Retrieval — get documents and store serializable summary in session_state["last_sources"]
            rag: RAGPipeline = st.session_state.get("rag")
            if rag:
                retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 3})
                raw_docs = retriever.get_relevant_documents(q)
            else:
                raw_docs = []

            # Build a serializable list of sources (strip down to necessary fields)
            last_sources = []
            for d in raw_docs:
                # support different Document types by safe attribute access
                page_content = getattr(d, "page_content", None) or getattr(d, "content", "") or ""
                metadata = getattr(d, "metadata", None) or {}
                src_id = getattr(d, "id", None)
                score = getattr(d, "score", None)
                last_sources.append({
                    "page_content": page_content,
                    "metadata": metadata,
                    "id": src_id,
                    "score": score,
                })
            st.session_state["last_sources"] = last_sources

            # Build messages: system + (context documents) + user
            system_prompt = "You are a helpful assistant. Use provided excerpts to answer precisely; if absent, say you don't know."
            ctx = []
            for i, s in enumerate(last_sources):
                txt = s.get("page_content", "")
                if len(txt) > 1200:
                    txt = txt[:1200] + "..."
                ctx.append(f"Source {i+1}:\n{txt}")
            context_text = "\n\n---\n\n".join(ctx) if ctx else ""

            messages = [{"role":"system","content":system_prompt}]
            if context_text:
                messages.append({"role":"system","content": f"Relevant documents:\n\n{context_text}"})
            messages.append({"role":"user","content": q})

            # estimate prompt tokens for user message before API call
            try:
                prompt_tokens_est = sum(count_tokens(m["content"], st.session_state.selected_model) for m in messages)
            except Exception:
                prompt_tokens_est = count_tokens(q, st.session_state.selected_model)

            # Append user entry with prompt token estimate (so tokens visible immediately)
            user_entry = {"role":"user", "content": q, "ts": now_iso(), "prompt_tokens": prompt_tokens_est, "completion_tokens": 0, "total_tokens": prompt_tokens_est, "cost": 0.0}
            st.session_state.history.append(user_entry)

            # Call OpenAI
            try:
                resp = openai.ChatCompletion.create(model=st.session_state.selected_model, messages=messages, temperature=0.0)
                assistant_text = resp.choices[0].message["content"].strip()
                usage = resp.get("usage", {})
                prompt_tokens_api = usage.get("prompt_tokens")
                completion_tokens_api = usage.get("completion_tokens")
                total_tokens_api = usage.get("total_tokens")

                # choose authoritative tokens where available
                prompt_tokens = prompt_tokens_api if prompt_tokens_api is not None else prompt_tokens_est
                completion_tokens = completion_tokens_api if completion_tokens_api is not None else count_tokens(assistant_text, st.session_state.selected_model)
                total_tokens = total_tokens_api if total_tokens_api is not None else (prompt_tokens + completion_tokens)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
                assistant_text = f"Error generating response: {e}"
                prompt_tokens = prompt_tokens_est
                completion_tokens = 0
                total_tokens = prompt_tokens

            # cost and store assistant entry
            call_cost = calc_cost_from_tokens(prompt_tokens or 0, completion_tokens or 0, st.session_state.selected_model)
            assistant_entry = {
                "role":"assistant",
                "content": assistant_text,
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": total_tokens or 0,
                "cost": call_cost,
                "ts": now_iso(),
            }
            st.session_state.history.append(assistant_entry)
            st.session_state.total_cost += call_cost

            # update the last user message's prompt token with authoritative prompt_tokens
            for i in range(len(st.session_state.history)-1, -1, -1):
                if st.session_state.history[i].get("role") == "user":
                    st.session_state.history[i]["prompt_tokens"] = prompt_tokens
                    st.session_state.history[i]["completion_tokens"] = 0
                    st.session_state.history[i]["total_tokens"] = prompt_tokens
                    break

            # clear the input safely on next run
            st.session_state["_clear_input_next_run"] = True
            safe_rerun()

# -------------------------
# Footer explanation
# -------------------------
st.markdown("""
<div style="margin-top:12px;padding:12px;border-radius:10px;background:linear-gradient(180deg,#ffffff,#fbfdff);border:1px solid rgba(10,11,20,0.04)">
  <strong>How cost is calculated</strong>
  <div style="font-size:13px;color:#475569;margin-top:6px">
    We use OpenAI's returned <code>usage</code> (prompt_tokens/completion_tokens) when present. If not present we estimate tokens using <code>tiktoken</code>.
    <br><br>
    <strong>Formula:</strong> <code>cost = (prompt_tokens + completion_tokens) * (price_per_1k / 1000)</code>
  </div>
</div>
""", unsafe_allow_html=True)
