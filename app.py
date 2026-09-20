# app.py
# GetSetGo — Personal AI Second Brain
# it never forgets
# Streamlit UI connecting the ingestion and retrieval pipelines.
#
# SET mode : store text or PDF        → pipeline/ingest.py
# GET mode : semantic search + RAG    → pipeline/retrieve.py
# Learning : thumbs up/down feedback  → database/feedback_store.py
#
# Icons: Material Symbols (Google Fonts, self-imported) — no emojis anywhere.
# Run: streamlit run app.py

import re
import time
import base64
import os
import streamlit as st

from pipeline.ingest import ingest_text, ingest_pdf, count_memories
from pipeline.retrieve import ask_brain, SCOPE_FILTERS, DATE_WINDOWS
from database import feedback_store

USER_ID = "default"  # swap for the authenticated user's UUID once auth is wired up
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSET_DIR, "logo_small.png")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GetSetGo — it never forgets",  # title in browser tab
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "GSG",
    layout="centered",
    initial_sidebar_state="collapsed"
)


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str:
    with open(LOGO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


def icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    """Inline Material Symbols icon for use inside custom HTML (st.markdown)."""
    return (f'<span class="material-symbols-outlined" '
            f'style="font-size:{size}px;color:{color};">{name}</span>')


def alert(kind: str, message: str) -> None:
    """Custom HTML/CSS alert box (replaces st.success/error/warning/info) so the
    icon, color and spacing are fully controlled by our own design system."""
    meta = {
        "success": ("check_circle", "var(--good)", "var(--good-bg)", "var(--good-border)"),
        "error":   ("error", "var(--bad)", "var(--bad-bg)", "var(--bad-border)"),
        "warning": ("warning", "var(--warn)", "var(--warn-bg)", "var(--warn-border)"),
        "info":    ("info", "var(--accent)", "var(--accent-soft)", "rgba(124,156,255,.35)"),
    }
    ic, fg, bg, border = meta[kind]
    st.markdown(
        f'<div class="gsg-alert" style="color:{fg};background:{bg};border-color:{border};">'
        f'{icon(ic, 18, fg)}<span>{message}</span></div>',
        unsafe_allow_html=True
    )


def stat_cards(items) -> None:
    """items: list of (icon_name, label, value) rendered as an HTML/CSS grid."""
    cells = "".join(
        f'<div class="stat-card">{icon(ic, 20)}'
        f'<div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>'
        for ic, label, val in items
    )
    st.markdown(f'<div class="stat-grid">{cells}</div>', unsafe_allow_html=True)


# ── Design tokens & global styling ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,500,0,0&display=swap');

    :root{
        --bg:#0b0d12; --surface:#12151c; --surface-2:#171b24; --border:#242938;
        --text:#eef1f6; --text-dim:#9aa3b5; --text-faint:#6b7280;
        --accent:#7c9cff; --accent-2:#a78bfa; --accent-soft:rgba(124,156,255,.14);
        --good:#34d399; --good-bg:rgba(52,211,153,.12); --good-border:rgba(52,211,153,.35);
        --warn:#fbbf24; --warn-bg:rgba(251,191,36,.12); --warn-border:rgba(251,191,36,.35);
        --bad:#f87171;  --bad-bg:rgba(248,113,113,.12);  --bad-border:rgba(248,113,113,.35);
        --radius:12px;
    }

    .material-symbols-outlined{
        font-family:'Material Symbols Outlined'; font-weight:normal; font-style:normal;
        line-height:1; letter-spacing:normal; text-transform:none; display:inline-block;
        white-space:nowrap; word-wrap:normal; direction:ltr; vertical-align:middle;
        -webkit-font-smoothing:antialiased;
    }

    html, body, [class*="css"]{
        font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    }
    .stApp{ background:var(--bg); color:var(--text); }
    .main .block-container{ max-width:760px; padding-top:2rem; padding-bottom:4rem; }

    /* Header */
    .gsg-header{ display:flex; align-items:center; gap:.75rem; margin-bottom:.15rem; }
    .gsg-logo-tile{
        width:46px; height:46px; border-radius:12px; overflow:hidden; flex-shrink:0;
        background:#e4e2dd; box-shadow:0 4px 14px rgba(0,0,0,.35); border:1px solid var(--border);
    }
    .gsg-logo-tile img{ width:100%; height:100%; object-fit:cover; display:block; }
    .gsg-title{ font-size:1.65rem; font-weight:800; letter-spacing:-.03em; color:var(--text); line-height:1.1;}
    .gsg-subtitle{ font-size:.85rem; color:var(--text-dim); margin-top:.1rem; }
    .gsg-pill{
        display:inline-flex; align-items:center; gap:.4rem; font-size:.72rem; font-weight:600;
        color:var(--text-dim); background:var(--surface-2); border:1px solid var(--border);
        border-radius:999px; padding:.3rem .75rem .3rem .6rem; margin-top:.65rem;
    }
    .gsg-pill b{ color:var(--text); }
    .gsg-pill .material-symbols-outlined{ color:var(--accent); }

    /* Tabs -> mode switch */
    .stTabs [data-baseweb="tab-list"]{
        gap:4px; background:var(--surface); padding:5px; border-radius:12px; border:1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"]{
        height:2.4rem; border-radius:9px; font-weight:600; color:var(--text-dim); background:transparent;
    }
    .stTabs [aria-selected="true"]{
        background:var(--surface-2) !important; color:var(--text) !important;
        box-shadow:inset 0 0 0 1px var(--border);
    }

    /* Custom alert boxes (replace st.success/error/warning/info) */
    .gsg-alert{
        display:flex; align-items:flex-start; gap:.55rem; border:1px solid; border-radius:10px;
        padding:.7rem .9rem; font-size:.88rem; line-height:1.45; margin:.5rem 0;
    }
    .gsg-alert .material-symbols-outlined{ margin-top:1px; flex-shrink:0; }

    /* Stat grid (diagnostics + result metrics) */
    .stat-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:.6rem; margin:.5rem 0 .3rem 0; }
    .stat-card{
        background:var(--surface); border:1px solid var(--border); border-radius:10px;
        padding:.75rem .8rem; display:flex; flex-direction:column; gap:.15rem;
    }
    .stat-card .material-symbols-outlined{ color:var(--accent); font-size:20px; margin-bottom:.15rem; }
    .stat-value{ font-size:1.25rem; font-weight:700; color:var(--text); }
    .stat-label{ font-size:.72rem; color:var(--text-faint); font-weight:500; }

    /* Confidence badges */
    .badge{ display:inline-flex; align-items:center; gap:.3rem; font-size:.72rem; font-weight:700;
        padding:.22rem .6rem; border-radius:999px; white-space:nowrap; }
    .badge .material-symbols-outlined{ font-size:15px; }
    .badge-high{ color:var(--good); background:var(--good-bg); border:1px solid var(--good-border); }
    .badge-medium{ color:var(--warn); background:var(--warn-bg); border:1px solid var(--warn-border); }
    .badge-low{ color:var(--bad); background:var(--bad-bg); border:1px solid var(--bad-border); }

    /* Inline citation chip */
    .cite-chip{
        display:inline-flex; align-items:center; justify-content:center; min-width:1.15rem; height:1.15rem;
        padding:0 .3rem; font-size:.68rem; font-weight:800; color:var(--accent);
        background:var(--accent-soft); border:1px solid rgba(124,156,255,.4); border-radius:5px;
        text-decoration:none !important; vertical-align:2px; margin:0 1px;
    }
    .cite-chip:hover{ background:rgba(124,156,255,.28); }

    .gsg-answer{ font-size:1rem; line-height:1.65; }
    .gsg-answer p{ margin-bottom:.7rem; }

    .source-meta{ font-size:.78rem; color:var(--text-faint); display:flex; align-items:center; gap:.3rem; flex-wrap:wrap; }
    .source-meta .material-symbols-outlined{ font-size:14px; }
    .source-title{ display:flex; align-items:center; gap:.4rem; font-weight:700; }
    .source-title .material-symbols-outlined{ color:var(--accent); font-size:18px; }
    .source-snippet{ font-size:.87rem; color:var(--text-dim); margin-top:.35rem; line-height:1.5; }

    .gsg-section-label{ display:flex; align-items:center; gap:.4rem; font-size:.78rem; color:var(--text-faint);
        margin:.5rem 0 .4rem 0; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
    .gsg-section-label .material-symbols-outlined{ font-size:15px; }

    .gsg-heading{ display:flex; align-items:center; gap:.45rem; font-weight:700; font-size:1.02rem; margin:.2rem 0; }
    .gsg-heading .material-symbols-outlined{ color:var(--accent); font-size:19px; }

    /* Buttons */
    div.stButton > button{ border-radius:9px; font-weight:600; }
    div.stButton > button[kind="primary"]{
        background:linear-gradient(135deg,var(--accent),var(--accent-2)); border:none; color:#0b0d12;
    }
    div.stButton > button[kind="primary"]:hover{ filter:brightness(1.08); color:#0b0d12; }

    [data-testid="stFileUploader"]{ border:1px dashed var(--border); border-radius:12px; padding:1rem; background:var(--surface); }

    hr{ border-color:var(--border); margin:1.4rem 0; }
    .footer{ display:flex; align-items:center; justify-content:center; gap:.4rem;
        text-align:center; color:var(--text-faint); font-size:.78rem; margin-top:2.2rem; }
    .footer .material-symbols-outlined{ font-size:14px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
st.session_state.setdefault("last_query", None)
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("feedback_given", False)
st.session_state.setdefault("prefill_query", "")

# ── Header ────────────────────────────────────────────────────────────────────
count = count_memories(USER_ID)
logo_html = (
    f'<div class="gsg-logo-tile"><img src="{_logo_data_uri()}" alt="GetSetGo logo"/></div>'
    if os.path.exists(LOGO_PATH) else
    f'<div class="gsg-logo-tile" style="display:flex;align-items:center;justify-content:center;">{icon("psychology", 24)}</div>'
)
st.markdown(f"""
<div class="gsg-header">
    {logo_html}
    <div>
        <div class="gsg-title">GetSetGo</div>
        <div class="gsg-subtitle">GET your answers · SET your memories · GO</div>
    </div>
</div>
<div class="gsg-pill">{icon("database", 15)} <b>{count}</b> memories stored &nbsp;·&nbsp;
{icon("psychology_alt", 15)} learning from your feedback</div>
""", unsafe_allow_html=True)

st.write("")
tab_get, tab_set = st.tabs([":material/travel_explore: Get — Ask your brain",
                             ":material/add_circle: Set — Add a memory"])

# ══════════════════════════════════════════════════════════════════════════════
#  SET MODE — Data ingestion
# ══════════════════════════════════════════════════════════════════════════════
with tab_set:
    st.caption("Type a note or upload a PDF — it gets chunked, embedded, and stored semantically.")

    input_type = st.radio("Input type", ["Text", "PDF document"], horizontal=True, label_visibility="collapsed")

    category = st.selectbox(
        "Category",
        ["general", "movies", "books", "notes", "work", "recipes", "travel", "links", "document", "research"]
    )

    if input_type == "Text":
        user_input = st.text_area(
            "What do you want to remember?",
            placeholder='e.g. "Movies to watch: Interstellar, Coco, 3 Idiots, Tamasha"',
            height=140
        )
        if st.button("Save to brain", use_container_width=True, type="primary", icon=":material/bolt:"):
            if user_input.strip():
                with st.spinner("Chunking, embedding, and storing..."):
                    result = ingest_text(user_input.strip(), category=category, user_id=USER_ID)
                if result.startswith("OK:"):
                    alert("success", result[3:].strip())
                    st.balloons()
                else:
                    alert("error", result[6:].strip() if result.startswith("ERROR:") else result)
            else:
                alert("warning", "Please type something to save.")

    else:
        st.caption("Text will be extracted from all pages and stored semantically.")
        uploaded_file = st.file_uploader(
            "Upload a PDF", type=["pdf"],
            help="Max size: 200MB. Scanned PDFs (image-only) cannot be extracted."
        )
        if uploaded_file is not None:
            file_size_kb = round(uploaded_file.size / 1024, 1)
            alert("info", f"<b>{uploaded_file.name}</b> — {file_size_kb} KB &nbsp;·&nbsp; Category: <code>{category}</code>")
            if st.button("Process PDF", use_container_width=True, type="primary", icon=":material/picture_as_pdf:"):
                with st.spinner(f"Extracting and storing '{uploaded_file.name}'..."):
                    result = ingest_pdf(
                        pdf_bytes=uploaded_file.read(), filename=uploaded_file.name,
                        category=category, user_id=USER_ID
                    )
                if result.startswith("OK:"):
                    alert("success", result[3:].strip())
                    st.balloons()
                else:
                    alert("error", result[6:].strip() if result.startswith("ERROR:") else result)

# ══════════════════════════════════════════════════════════════════════════════
#  GET MODE — Semantic search + LLM generation
# ══════════════════════════════════════════════════════════════════════════════
with tab_get:

    # ── Filter bar ────────────────────────────────────────────────────────
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        scope = st.selectbox("Scope", list(SCOPE_FILTERS.keys()), key="scope_filter")
    with fcol2:
        date_range = st.selectbox("When", list(DATE_WINDOWS.keys()), key="date_filter")

    query = st.text_input(
        "Ask your brain...",
        value=st.session_state.prefill_query,
        placeholder='e.g. "What movies did I want to watch?" or "Summarise my travel notes"',
        key="query_box"
    )

    # ── Suggestion chips ─────────────────────────────────────────────────
    suggestions = [
        ("flight", "What are my upcoming travel itineraries?"),
        ("health_and_safety", "Show health protocols"),
        ("movie", "What movies did I want to watch?"),
        ("work", "Summarise my work notes"),
    ]
    st.markdown(f'<div class="gsg-section-label">{icon("bolt", 15)} Try asking</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(suggestions))
    for i, (ic_name, s) in enumerate(suggestions):
        if chip_cols[i].button(s, key=f"chip_{i}", use_container_width=True, icon=f":material/{ic_name}:"):
            st.session_state.prefill_query = s
            st.rerun()

    run_search = st.button("Search", use_container_width=True, type="primary", icon=":material/search:")

    if run_search:
        if query.strip():
            st.session_state.feedback_given = False

            # ── Staged, transparent progress instead of a blank spinner ──
            status = st.status("Searching your memories...", expanded=True)
            status.write(
                f":material/search: Searching **{count} stored memories** "
                f"(scope: *{scope}*, window: *{date_range}*)..."
            )
            time.sleep(0.25)

            result = ask_brain(query.strip(), user_id=USER_ID, scope=scope, date_range=date_range)

            n_sources = len(result["sources"])
            if n_sources:
                status.write(f":material/description: Found **{n_sources}** relevant source{'s' if n_sources != 1 else ''}...")
                time.sleep(0.15)
            status.write(":material/edit_note: Synthesizing response...")
            status.update(label="Done", state="complete", expanded=False)

            st.session_state.last_query = query.strip()
            st.session_state.last_result = result
        else:
            alert("warning", "Please type a question.")

    # ── Render last result (persists across feedback button clicks) ────────
    result = st.session_state.last_result
    if result:
        sources = result["sources"]

        def _badge(score):
            if score >= 0.70:
                return "badge-high", "check_circle", "High confidence"
            elif score >= 0.40:
                return "badge-medium", "info", "Moderate confidence"
            return "badge-low", "cancel", "Low confidence"

        # Turn [1], [2] markers from the LLM into clickable citation chips
        answer_html = result["answer"]
        answer_html = re.sub(r"\n", "<br>", answer_html)
        answer_html = re.sub(r"\[(\d+)\]", lambda m: f'<a class="cite-chip" href="#src-{m.group(1)}">{m.group(1)}</a>', answer_html)

        st.markdown(f'<div class="gsg-answer">{answer_html}</div>', unsafe_allow_html=True)

        # ── Feedback bar ─────────────────────────────────────────────────
        st.write("")
        fb1, fb2, fb3 = st.columns([6, 1, 1])
        fb1.markdown(f'<div style="padding-top:.4rem;color:var(--text-faint);font-size:.85rem;">'
                     f'{icon("forum", 15)} Was this answer helpful?</div>', unsafe_allow_html=True)
        up_clicked = fb2.button("", key="fb_up", use_container_width=True, icon=":material/thumb_up:")
        down_popover_slot = fb3

        if up_clicked and not st.session_state.feedback_given:
            feedback_store.record_feedback(USER_ID, st.session_state.last_query, sources, rating="up")
            st.session_state.feedback_given = True
            st.toast("Thanks! I'll trust these sources more next time.", icon=":material/thumb_up:")

        if hasattr(st, "popover"):
            with down_popover_slot.popover("", use_container_width=True, icon=":material/thumb_down:"):
                st.caption("What went wrong?")
                reason = st.radio(
                    "reason", ["Wrong source pulled", "Outdated information", "Hallucination"],
                    label_visibility="collapsed", key="down_reason"
                )
                if st.button("Submit", key="fb_down_submit", use_container_width=True, icon=":material/send:"):
                    feedback_store.record_feedback(
                        USER_ID, st.session_state.last_query, sources, rating="down", reason=reason
                    )
                    st.session_state.feedback_given = True
                    st.toast("Got it — I'll deprioritize these sources.", icon=":material/thumb_down:")
        else:
            if down_popover_slot.button("", key="fb_down_fallback", use_container_width=True, icon=":material/thumb_down:"):
                feedback_store.record_feedback(USER_ID, st.session_state.last_query, sources, rating="down")
                st.session_state.feedback_given = True
                st.toast("Got it — I'll deprioritize these sources.", icon=":material/thumb_down:")

        # ── Source cards (progressive disclosure) ──────────────────────────
        if sources:
            st.divider()
            st.markdown(f'<div class="gsg-heading">{icon("menu_book", 19)} Sources & grounding documents</div>',
                        unsafe_allow_html=True)

            for i, src in enumerate(sources):
                badge_cls, badge_icon, badge_txt = _badge(src["score"])
                type_icon = "picture_as_pdf" if src.get("type") == "pdf" else "description"
                snippet = src["content"][:160].strip()
                if len(src["content"]) > 160:
                    snippet += "…"

                st.markdown(f'<div id="src-{i+1}"></div>', unsafe_allow_html=True)
                with st.container(border=True):
                    hcol1, hcol2 = st.columns([5, 2])
                    with hcol1:
                        st.markdown(
                            f'<div class="source-title">[{i+1}] {icon(type_icon, 18)} <code>{src["source_name"]}</code></div>'
                            f'<div class="source-meta">{icon("sell", 13)} {src["category"]}'
                            f'&nbsp;·&nbsp;{icon("schedule", 13)} {src["date"]} at {src["time"]}</div>',
                            unsafe_allow_html=True
                        )
                    with hcol2:
                        st.markdown(
                            f'<div style="text-align:right" title="Raw similarity: {src["raw_score"]:.3f} · '
                            f'Learned adjustment: {src["learned_delta"]:+.3f}">'
                            f'<span class="badge {badge_cls}">{icon(badge_icon, 14)}{badge_txt}</span></div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"score: {src['score']:.3f}")

                    st.markdown(f'<div class="source-snippet">{snippet}</div>', unsafe_allow_html=True)
                    with st.expander("View full excerpt", icon=":material/expand_content:"):
                        st.write(src["content"])
                        st.caption(
                            f"Chunk {src['chunk_index'] + 1} of {src['total_chunks']} · "
                            f"doc id `{src['doc_id'][:8]}…`"
                        )

        elif result["error"]:
            alert("error", result["error"])

    # ── Search Diagnostics & Vector Metrics (bottom, collapsed) ────────────
    st.divider()
    with st.expander("Search Diagnostics & Vector Metrics", icon=":material/monitoring:"):
        if result and result["sources"]:
            scores = [s["score"] for s in result["sources"]]
            stat_cards([
                ("target", "Top match (adjusted)", f"{max(scores):.3f}"),
                ("equalizer", "Avg relevance", f"{round(sum(scores)/len(scores),3):.3f}"),
                ("folder_copy", "Sources used", len(scores)),
            ])
            st.caption(
                "Adjusted score = raw cosine similarity + learned weight from your past thumbs-up/down feedback. "
                "1.00 = perfect match · 0.00 = unrelated · ≥0.70 = strong semantic match."
            )
            st.table([
                {
                    "#": i + 1,
                    "source": s["source_name"],
                    "raw_similarity": s["raw_score"],
                    "learned_delta": s["learned_delta"],
                    "adjusted_score": s["score"],
                }
                for i, s in enumerate(result["sources"])
            ])
        else:
            st.caption("Run a search to see per-source vector metrics here.")

        st.markdown(f'<div class="gsg-heading" style="font-size:.92rem;margin-top:1rem;">'
                    f'{icon("model_training", 17)} Feedback-based learning summary</div>', unsafe_allow_html=True)
        stats = feedback_store.get_stats(USER_ID)
        stat_cards([
            ("how_to_vote", "Total ratings", stats["total_ratings"]),
            ("thumb_up", "Up", stats["ups"]),
            ("thumb_down", "Down", stats["downs"]),
        ])
        if stats["reason_breakdown"]:
            st.caption("Down-vote reasons: " + ", ".join(f"{k} ({v})" for k, v in stats["reason_breakdown"].items()))
        boosted = [r for r in stats["top_boosted"] if r["weight"] > 0]
        if boosted:
            st.caption("Most-trusted chunks: " + ", ".join(f"`{r['doc_id'][:8]}…` (+{r['weight']:.2f})" for r in boosted))
        penalized = [r for r in stats["top_penalized"] if r["weight"] < 0]
        if penalized:
            st.caption("Most-distrusted chunks: " + ", ".join(f"`{r['doc_id'][:8]}…` ({r['weight']:.2f})" for r in penalized))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<p class="footer">{icon("psychology", 14)} GetSetGo · Personal AI Second Brain · '
    'Powered by ChromaDB + Sentence Transformers + Groq LLaMA · learns from your feedback</p>',
    unsafe_allow_html=True
)
