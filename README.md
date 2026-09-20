# GetSetGo — Personal AI Second Brain

**GET** your answers · **SET** your memories · **GO**

A production-ready RAG (Retrieval Augmented Generation) application.
Store text and PDFs, search them semantically, get AI-generated answers with citations and accuracy scores.

---

## What it does

| Mode | What happens |
|------|-------------|
| **SET** | Type text or upload a PDF → parsed → chunked → embedded → stored in ChromaDB |
| **GET** | Ask a question → embedded → semantic search → LLM generates answer → citations + relevance scores shown |

---

## Quick start

### 1. Clone / unzip the project

```
cd getsetgo
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

> First run downloads the embedding model (~90MB). Subsequent runs are instant.

### 4. Set your API key

Copy `.env.example` to `.env`:

```
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at: https://console.groq.com

### 5. Run the app

```
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
getsetgo/
├── app.py                    ← Streamlit UI
├── config.py                 ← All settings in one place
├── brain.py                  ← Original prototype (reference only)
├── requirements.txt
├── .env.example
├── .gitignore
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py             ← SET pipeline: text + PDF ingestion
│   └── retrieve.py           ← GET pipeline: semantic search + LLM
│
├── database/
│   ├── __init__.py
│   └── chroma_client.py      ← ChromaDB connection (per-user ready)
│
└── db/                       ← Auto-created, stores your ChromaDB data
    └── default/              ← Default user's vector database
```

---

## How the pipelines work

### SET (Ingestion) Pipeline

```
Input (text or PDF)
    → Parse     (clean text, extract PDF pages)
    → Chunk     (split into 300-char overlapping pieces)
    → Embed     (convert each chunk to 384 numbers)
    → Store     (save vectors + metadata in ChromaDB)
```

### GET (Retrieval) Pipeline

```
User question
    → Embed     (convert query to 384 numbers, same model)
    → Search    (cosine similarity against all stored vectors)
    → Context   (assemble top 5 most relevant chunks)
    → Generate  (Groq LLM reads context, writes answer)
    → Output    (answer + citations + relevance scores)
```

---

## Accuracy metrics

Every GET response shows:
- **Top match score** — how relevant the best result is (0.0–1.0)
- **Average relevance** — across all retrieved sources
- **Source breakdown** — per-source score, filename, timestamp

Score guide:
- 🟢 ≥ 0.70 — strong semantic match
- 🟡 0.40–0.69 — moderate match
- 🔴 < 0.40 — weak match (answer may be unreliable)

---

## Tech stack (all free)

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| Vector DB | ChromaDB (local) |
| Embedding | sentence-transformers / all-MiniLM-L6-v2 |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| PDF parsing | PyMuPDF (fitz) |
| LLM | Groq (llama-3.1-8b-instant) |

---

## Roadmap

- [x] Text ingestion
- [x] PDF ingestion
- [x] Semantic search
- [x] LLM generation with citations
- [x] Relevance scores
- [ ] Authentication (Supabase) — Session B
- [ ] Memory history browser — Session C
- [ ] Full UI redesign — Session C
- [ ] Public deployment (Hugging Face Spaces) — Session D
