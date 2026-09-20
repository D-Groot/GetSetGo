# pipeline/retrieve.py
# Retrieval + Generation Pipeline — based on brain.py's ask_brain().
#
# Pipeline flow:
#   Query → Embed → Similarity Search → Build Context → LLM → Answer + Citations
#
# What is identical to brain.py:
#   - embedder.encode([query]).tolist()
#   - collection.query with n_results=min(5, count)
#   - same prompt template
#   - temperature=0.3
#   - same citation format
#
# What is new vs brain.py:
#   - Returns a structured dict (answer + sources) instead of plain string
#   - relevance_score per source (0.0 to 1.0)
#   - source_name in citations (shows filename for PDFs)
#   - per-user isolation via user_id

from sentence_transformers import SentenceTransformer
from groq import Groq
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from database.chroma_client import get_collection
from database import feedback_store
from config import EMBED_MODEL, GROQ_API_KEY, GROQ_MODEL

load_dotenv()

# ── Same globals as brain.py ──────────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)
llm      = Groq(api_key=GROQ_API_KEY)

# ── Scope filter → ChromaDB `where` clause ────────────────────────────────────
SCOPE_FILTERS = {
    "All Memory"    : None,
    "PDF Documents" : {"type": "pdf"},
    "Text Notes"    : {"type": "text"},
    "Itineraries"   : {"category": "travel"},
}

# ── Date filter → lookback window ─────────────────────────────────────────────
DATE_WINDOWS = {
    "All Time"    : None,
    "Past 7 Days" : 7,
    "Past Month"  : 30,
}


def _within_window(date_str: str, days: int) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return True  # don't drop malformed dates
    return d >= datetime.now() - timedelta(days=days)


def ask_brain(query: str,
              user_id: str = "default",
              scope: str = "All Memory",
              date_range: str = "All Time") -> dict:
    """
    Full retrieval + generation pipeline.
    Based on brain.py's ask_brain() — returns structured dict for UI metrics.

    Args:
        query      : the user's question in natural language
        user_id    : searches only this user's ChromaDB collection
        scope      : one of SCOPE_FILTERS keys — narrows which memories are searched
        date_range : one of DATE_WINDOWS keys — narrows by recency

    Returns:
        {
            "answer"  : str,   full response text with citations
            "sources" : list,  each source dict with relevance + learned weight
            "error"   : str|None
        }
    """
    try:
        collection = get_collection(user_id)

        # ── brain.py: guard against empty DB ──────────────────────────
        if collection.count() == 0:
            return {
                "answer" : "No memories found yet. Switch to Set mode and add some information first!",
                "sources": [],
                "error"  : None
            }

        # ── Step 1: Embed the query (identical to brain.py) ───────────
        # MUST use same model as ingestion — same vector space = comparable
        query_vector = embedder.encode([query]).tolist()

        # ── Step 2: Similarity search, scoped by filter bar ────────────
        # Pull a wider pool than we need so client-side date filtering
        # still leaves enough candidates to fill the top 5.
        where_clause = SCOPE_FILTERS.get(scope)
        pool_size = min(20, collection.count())

        query_kwargs = dict(
            query_embeddings = query_vector,
            n_results        = pool_size,
            include          = ["documents", "metadatas", "distances"]
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        results = collection.query(**query_kwargs)

        if not results["documents"][0]:
            return {
                "answer" : "No memories matched your filters. Try widening the scope or date range.",
                "sources": [],
                "error"  : None
            }

        doc_ids   = results["ids"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        # ── Client-side date filter (kept simple/robust across chroma versions) ──
        window_days = DATE_WINDOWS.get(date_range)
        if window_days is not None:
            keep = [i for i, m in enumerate(metas) if _within_window(m.get("date", ""), window_days)]
            doc_ids   = [doc_ids[i] for i in keep]
            docs      = [docs[i] for i in keep]
            metas     = [metas[i] for i in keep]
            distances = [distances[i] for i in keep]

        if not docs:
            return {
                "answer" : "No memories matched your filters. Try widening the scope or date range.",
                "sources": [],
                "error"  : None
            }

        # ── Convert cosine distance → raw relevance score ───────────────
        # ChromaDB cosine distance: 0.0 = identical, 2.0 = completely opposite
        # score = 1 - (distance / 2) gives 0.0 to 1.0
        raw_scores = [round(1 - (d / 2), 3) for d in distances]

        # ── Apply learned feedback weights (thumbs up/down history) ────
        weights = feedback_store.get_weights(user_id, doc_ids)
        adjusted = [
            round(max(0.0, min(1.0, s + weights.get(i, 0.0))), 3)
            for s, i in zip(raw_scores, doc_ids)
        ]

        # ── Re-rank by adjusted (learned) score, then take top 5 ────────
        order = sorted(range(len(docs)), key=lambda i: adjusted[i], reverse=True)[:5]
        doc_ids   = [doc_ids[i] for i in order]
        docs      = [docs[i] for i in order]
        metas     = [metas[i] for i in order]
        raw_scores = [raw_scores[i] for i in order]
        adjusted   = [adjusted[i] for i in order]
        learned_deltas = [round(weights.get(i, 0.0), 3) for i in doc_ids]

        # ── Step 3: Build context (identical to brain.py) ─────────────
        context_parts = []
        for doc, meta in zip(docs, metas):
            ts = meta.get("timestamp", "unknown time")
            context_parts.append(f"[Stored on {ts}]\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

        # ── Step 4: Same prompt as brain.py ───────────────────────────
        prompt = f"""You are a personal memory assistant. The user has stored the following notes:

{context}

Based ONLY on the above stored notes, answer this question: {query}

If the answer is not in the notes, say so clearly. Always mention when the relevant note was stored.
When you state a fact drawn from note N (1-indexed in the order given above), append a citation
marker like [N] immediately after that sentence, matching the note's position in the list above."""

        response = llm.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.3
        )

        answer = response.choices[0].message.content

        # ── Build sources list for UI metrics + progressive disclosure ──
        sources = []
        for doc, meta, raw, adj, delta, doc_id in zip(docs, metas, raw_scores, adjusted, learned_deltas, doc_ids):
            sources.append({
                "doc_id"     : doc_id,
                "content"    : doc,
                "score"      : adj,          # learned/adjusted score drives the badge
                "raw_score"  : raw,          # pure cosine similarity, shown on hover
                "learned_delta": delta,      # how much feedback moved this source
                "timestamp"  : meta.get("timestamp", "?"),
                "date"       : meta.get("date", "?"),
                "time"       : meta.get("time", "?"),
                "source_name": meta.get("source_name", "text_input"),
                "category"   : meta.get("category", "general"),
                "type"       : meta.get("type", "text"),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1)
            })

        return {
            "answer" : answer,
            "sources": sources,
            "error"  : None
        }

    except Exception as e:
        return {
            "answer" : f"ERROR: {str(e)}",
            "sources": [],
            "error"  : str(e)
        }
