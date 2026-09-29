# RAG — Document Q&A

**Technical Challenge · AI Engineer · CIRES Technologies · Tanger Med Group**

A local Retrieval-Augmented Generation (RAG) system that indexes PDF documents and answers questions strictly from their content. Runs entirely on your machine — no cloud API required.

---

## Architecture

```
PDF upload
    │
    ▼
PDFPlumberLoader          ← extract text, one Document per page
    │
    ▼
RecursiveCharacterTextSplitter   ← 1 024-char chunks · 80-char overlap
    │
    ▼
FastEmbedEmbeddings       ← BAAI/bge-small-en-v1.5 dense vectors
    │
    ▼
ChromaDB  (./db)          ← persist vectors locally

━━━ Query time ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User question
    │
    ▼
FastEmbedEmbeddings       ← embed the question
    │
    ▼
ChromaDB retriever        ← similarity search · top-5 chunks · threshold 0.1
    │
    ▼
Mistral via Ollama        ← generate answer grounded in context only
    │
    ▼
Answer + deduplicated source references
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API server | Python · Flask |
| RAG framework | LangChain · LangChain Community |
| PDF extraction | PDFPlumber |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5) |
| Vector store | ChromaDB |
| LLM | Mistral via Ollama (local) |

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| Ollama | latest — [ollama.com](https://ollama.com/) |

---

## Setup

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> FastEmbed downloads its embedding model (~60 MB) on the first run.

### 3. Install Ollama and pull Mistral

Download Ollama from [https://ollama.com/](https://ollama.com/), then:

```bash
ollama pull mistral
```

---

## Run

**Terminal 1 — start Ollama:**
```bash
ollama serve
```

**Terminal 2 — start the application:**
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Usage

### Step 1 — Upload a PDF

1. Click the upload zone and select a `.pdf` file.
2. Click **Index document**.
3. Wait for: `✓ Document indexed — N page(s) · M chunks`

### Step 2 — Ask a question

1. Type your question in the text area (or press **Enter**).
2. Click **Get answer**.
3. The system retrieves the most relevant chunks, sends them to Mistral, and returns an answer with source references.

---

## Example

Using a Big Data introductory document:

**Question:** *What are the 3 Vs of Big Data?*

**Expected answer:** The model will cite Volume, Velocity, and Variety from the document and reference the page where that content appears.

**Off-topic question:** *What is the capital of France?*

**Expected response:** *"The document does not contain information about this topic."*

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve web UI |
| POST | `/pdf` | Upload and index a PDF |
| POST | `/ask` | Ask a question about indexed documents |

### POST /pdf

```json
// Response
{
  "status": "indexed",
  "filename": "document.pdf",
  "pages": 12,
  "chunks": 47
}
```

### POST /ask

```json
// Request
{ "query": "What is Big Data?" }

// Response
{
  "answer": "Big Data refers to...",
  "sources": [
    {
      "source": "document.pdf",
      "page": 3,
      "excerpt": "Big Data is defined as..."
    }
  ]
}
```

### Error codes

| Code | Meaning |
|---|---|
| 400 | Invalid input or no documents indexed |
| 500 | Vector store error |
| 503 | Ollama is not running |

---

## Reset the vector store

To clear all indexed documents and start fresh:

```bash
# Windows
rmdir /s /q db && mkdir db

# macOS / Linux
rm -rf db && mkdir db
```

---

## Project structure

```
RAG/
├── app.py              ← Flask API — upload, index, answer
├── index.html          ← Single-page web UI
├── requirements.txt    ← Direct Python dependencies
├── README.md           ← This file
├── .gitignore
├── pdf/                ← Uploaded PDFs (git-ignored)
├── db/                 ← ChromaDB vector store (git-ignored)
└── screenshots/        ← Demo screenshots
```

---

## Limitations

- **Session state:** the Ask panel unlocks only within the current browser session. Refreshing the page requires re-uploading the PDF (indexing is fast; the DB is already populated).
- **Accumulative indexing:** each upload adds to the same ChromaDB collection. Reset `./db` to start clean.
- **CPU inference:** without a GPU, Mistral may take 30–120 seconds per answer.
- **Hallucination risk:** the prompt strictly instructs Mistral to stay within the retrieved context, but LLMs can occasionally ignore this constraint.
