# brain.py
# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE FILE — the original working prototype.
# The production code is now split into:
#   pipeline/ingest.py   ← store_text() lives here
#   pipeline/retrieve.py ← ask_brain() lives here
#   database/chroma_client.py ← ChromaDB connection
#   config.py            ← all settings
#
# This file is kept as a reference. Do NOT import from it in production.
# ─────────────────────────────────────────────────────────────────────────────

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
import os
def 
load_dotenv()

embedder   = SentenceTransformer("all-MiniLM-L6-v2")
client     = chromadb.PersistentClient(path="./my_brain_db")
collection = client.get_or_create_collection(name="memories")
llm        = Groq(api_key=os.getenv("GROQ_API_KEY"))


def store_text(text: str) -> str:
    chunks    = [text.strip()]
    vectors   = embedder.encode(chunks).tolist()
    timestamp = datetime.now().isoformat()
    doc_id    = f"mem_{datetime.now().timestamp()}"
    collection.add(
        documents  = chunks,
        embeddings = vectors,
        metadatas  = [{"timestamp": timestamp, "source": "user_input"}],
        ids        = [doc_id]
    )
    return f"✅ Stored! Saved at {timestamp}"


def ask_brain(query: str) -> str:
    query_vector = embedder.encode([query]).tolist()
    results      = collection.query(
        query_embeddings = query_vector,
        n_results        = min(5, collection.count())
    )
    if not results["documents"][0]:
        return "🤔 No memories found yet."

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        ts = meta.get("timestamp", "unknown time")
        context_parts.append(f"[Stored on {ts}]\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a personal memory assistant. The user has stored the following notes:

{context}

Based ONLY on the above stored notes, answer this question: {query}

If the answer is not in the notes, say so clearly. Always mention when the relevant note was stored."""

    response = llm.chat.completions.create(
        model       = "llama-3.1-8b-instant",
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.3
    )
    answer    = response.choices[0].message.content
    citations = "\n\n---\n**📌 Sources used:**\n"
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        ts      = meta.get("timestamp", "?")
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        citations += f"\n{i+1}. *{ts}* — \"{preview}\""
    return answer + citations


def count_memories() -> int:
    return collection.count()
