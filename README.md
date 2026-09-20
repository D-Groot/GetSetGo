<p align="center">
  <img src="assets/logo_small.png" alt="GetSetGo Logo" width="140">
</p>

<h1 align="center">GetSetGo</h1>

<p align="center">
  A personal AI second brain built with Retrieval-Augmented Generation (RAG).
</p>

<p align="center">
  Store your knowledge. Retrieve it intelligently. Use it when you need it.
</p>

---

## Overview

**GetSetGo** is a personal knowledge assistant that lets users store text and PDF documents, retrieve relevant information through semantic search, and generate grounded answers using a Large Language Model (LLM).

The system follows a simple workflow:

```text
User Knowledge
      |
      v
    SET
  Ingest and store
      |
      v
    GET
 Retrieve relevant context
      |
      v
    GO
 Generate a useful, grounded response
```

The application is designed around the idea of a **personal second brain**: instead of relying only on the model's general knowledge, the assistant retrieves information from the user's own stored knowledge before generating an answer.

## Why the name "GetSetGo"?

The name represents the three core stages of the application:

* **SET** — set your knowledge by storing notes and documents.
* **GET** — get relevant information from your stored memory when you ask a question.
* **GO** — move from retrieved knowledge to an actionable answer.

In short:

> **SET your knowledge. GET your context. GO with the answer.**

---

## Key Features

* Text and PDF knowledge ingestion
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Source-aware answers with citations
* Relevance scoring for retrieved sources
* User feedback-based retrieval adjustment
* Search scope and date filters
* Local ChromaDB persistence
* Clean Streamlit interface
* Modular ingestion, retrieval, database, and UI architecture

---

## How It Works

### 1. SET — Knowledge Ingestion

When a user adds text or a PDF, GetSetGo processes the information through an ingestion pipeline:

```text
Text / PDF
   |
   v
Parse
   |
   v
Chunk
   |
   v
Embed
   |
   v
Store in ChromaDB
```

**Parsing** extracts usable text from the input.

**Chunking** divides large text into smaller overlapping sections so individual pieces can be searched effectively.

**Embedding** converts each text chunk into a numerical vector that represents its semantic meaning.

**Vector storage** saves these embeddings and their metadata in ChromaDB.

### 2. GET — Retrieval and Generation

When the user asks a question:

```text
Question
   |
   v
Query Embedding
   |
   v
Semantic Search
   |
   v
Relevant Context
   |
   v
LLM Generation
   |
   v
Answer + Sources
```

The query is converted into an embedding and compared with stored vectors using cosine similarity.

The most relevant chunks are then provided as context to the LLM. The model generates an answer based on that retrieved context and includes source references.

### 3. Feedback-Based Retrieval

GetSetGo records user feedback on retrieved sources.

Positive feedback increases the future retrieval weight of a source, while negative feedback decreases it. This provides a lightweight learning mechanism without retraining the embedding model or LLM.

```text
Retrieved Source
       |
   User Feedback
       |
       v
  Trust Weight
       |
       v
Future Retrieval Ranking
```

---

## Core Concepts

| Concept               | Meaning                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------- |
| **RAG**               | Retrieval-Augmented Generation; retrieves relevant information before generating an answer. |
| **Embedding**         | A numerical representation of text that captures semantic meaning.                          |
| **Vector Database**   | A database optimized for storing and searching numerical representations of information.    |
| **Semantic Search**   | Search based on meaning rather than exact keyword matching.                                 |
| **Chunking**          | Splitting large documents into smaller sections for efficient retrieval.                    |
| **LLM**               | Large Language Model used to generate natural-language responses.                           |
| **Cosine Similarity** | A measure used to determine how semantically similar two vectors are.                       |
| **Relevance Score**   | A normalized indication of how closely a retrieved chunk matches the query.                 |

---

## Architecture

```text
                    +----------------------+
                    |      Streamlit UI    |
                    +----------+-----------+
                               |
                 +-------------+-------------+
                 |                           |
              SET MODE                    GET MODE
                 |                           |
                 v                           v
        +----------------+          +----------------+
        | Ingestion      |          | Retrieval      |
        | Pipeline       |          | Pipeline       |
        +-------+--------+          +-------+--------+
                |                           |
        Parse / Chunk                Query Embedding
                |                           |
            Embedding                Semantic Search
                |                           |
                +------------+--------------+
                             |
                             v
                      +-------------+
                      |  ChromaDB   |
                      | Vector Store|
                      +------+------+
                             |
                             v
                      Relevant Context
                             |
                             v
                      +-------------+
                      |     LLM     |
                      |    Groq     |
                      +------+------+
                             |
                             v
                    Answer + Citations

             Feedback ---> Retrieval Weights
```

---

## Technology Stack

| Layer                  | Technology                                 |
| ---------------------- | ------------------------------------------ |
| Interface              | Streamlit                                  |
| Programming Language   | Python                                     |
| Vector Database        | ChromaDB                                   |
| Embeddings             | Sentence Transformers — `all-MiniLM-L6-v2` |
| Text Chunking          | LangChain Text Splitters                   |
| PDF Processing         | PyMuPDF                                    |
| LLM                    | Groq API                                   |
| Local Feedback Storage | SQLite                                     |
| Configuration          | python-dotenv                              |

---

## Project Structure

```text
getsetgo/
├── app.py
├── config.py
├── brain.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── assets/
│   └── logo_small.png
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py
│   └── retrieve.py
│
└── database/
    ├── __init__.py
    ├── chroma_client.py
    └── feedback_store.py
```

### Module Responsibilities

* `app.py` — Streamlit application and user interface.
* `pipeline/ingest.py` — text/PDF parsing, chunking, embedding, and storage.
* `pipeline/retrieve.py` — query embedding, semantic retrieval, context construction, and LLM generation.
* `database/chroma_client.py` — ChromaDB connection and persistent vector storage.
* `database/feedback_store.py` — feedback history and retrieval-weight updates.
* `config.py` — central configuration for models, database paths, and chunking parameters.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/D-Groot/GetSetGo.git
cd GetSetGo
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file from the provided example:

```bash
cp .env.example .env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or API keys to the repository.

### 5. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Data and Privacy

GetSetGo stores embeddings and application data locally in the project's database directory.

The embedding model runs locally. However, retrieved context is sent to the configured Groq API for LLM generation. Do not store or submit sensitive information unless you understand the data handling policies of the services you use.

---

## Retrieval Quality

Each retrieved source has a relevance score derived from vector similarity and adjusted using the application's feedback mechanism.

The score should be interpreted as a **retrieval relevance signal**, not as a guarantee that an answer is factually correct.

For reliable responses, the retrieved sources should contain sufficient and accurate information about the user's question.

---

## Roadmap

* [x] Text ingestion
* [x] PDF ingestion
* [x] Semantic retrieval
* [x] RAG-based answer generation
* [x] Source citations
* [x] Relevance scoring
* [x] Feedback-based retrieval adjustment
* [x] Scope and date filtering
* [ ] User authentication
* [ ] Persistent conversation history
* [ ] Improved document management
* [ ] OCR support for scanned PDFs
* [ ] Production deployment
* [ ] Multi-user isolation with authenticated user IDs

---

## Repository

GitHub:

https://github.com/D-Groot/GetSetGo/

---

## License

This project is intended to be released under the **MIT License**.

The MIT License permits use, modification, distribution, and private or commercial use, subject to the license terms.

Add a `LICENSE` file containing the standard MIT License text before distributing the repository under this license.
