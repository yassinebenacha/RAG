# RAG — Document Q&A

A local Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask questions about its content. Runs entirely on your machine — no cloud API required.

**Stack:** Python · Flask · LangChain · PDFPlumber · FastEmbed · ChromaDB · Ollama (Mistral)

---

## Architecture

```
┌─────────────┐
│  PDF upload │
└──────┬──────┘
       │
       ▼
PDFPlumberLoader          ← extract text, page by page
       │
       ▼
RecursiveCharacterTextSplitter   ← 1 024-char chunks, 80-char overlap
       │
       ▼
FastEmbedEmbeddings       ← dense vector per chunk
       │
       ▼
ChromaDB (./db)           ← persist vectors locally

━━━ At query time ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User question
       │
       ▼
FastEmbedEmbeddings       ← embed the question
       │
       ▼
ChromaDB retriever        ← similarity search, top-5 chunks
       │
       ▼
Mistral via Ollama        ← generate answer grounded in context
       │
       ▼
Answer + source references
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| [Ollama](https://ollama.com/) | latest |
| Mistral model | `mistral:latest` |

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** FastEmbed will download its embedding model (~60 MB) on the first run.

### 4. Install Ollama

Download from [https://ollama.com/](https://ollama.com/) and follow the installer for your OS.

### 5. Pull the Mistral model

```bash
ollama pull mistral
```

---

## Running the application

### 1. Start Ollama (keep this terminal open)

```bash
ollama serve
```

### 2. Start the Flask server (in a second terminal, with venv activated)

```bash
python app.py
```

The server starts at **http://127.0.0.1:5000**

### 3. Open the web interface

Navigate to **http://127.0.0.1:5000** in your browser.

---

## Usage

### Upload a PDF

1. Click **"Click to select a PDF file"** and choose a `.pdf` file.
2. Click **"Index document"**.
3. Wait for confirmation: `Indexed — N page(s), M chunks`.

Indexing time depends on PDF length. A 20-page document typically takes 5–15 seconds.

### Ask a question

1. Type your question in the text area (or press **Enter** to submit).
2. Click **"Get answer"**.
3. The system retrieves the most relevant chunks, sends them to Mistral, and displays the answer with source references (filename, page number, excerpt).

### Example questions

For a Big Data introductory document:

- *What is Big Data?*
- *What are the 3 Vs of Big Data?*
- *What technologies are used for Big Data processing?*
- *What is the difference between batch and stream processing?*

### Off-topic questions

For questions not covered by the indexed document, the model will respond:

> "The document does not contain information about this topic."

---

## Project structure

```
RAG/
├── app.py              ← Flask API (upload, index, answer)
├── index.html          ← Single-page web UI
├── requirements.txt    ← Direct Python dependencies
├── README.md           ← This file
├── .gitignore
├── pdf/                ← Uploaded PDFs (not committed)
└── db/                 ← ChromaDB vector store (not committed)
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/`    | Serve web UI |
| POST | `/pdf` | Upload and index a PDF |
| POST | `/ask` | Ask a question about indexed documents |

### POST /pdf

**Input:** `multipart/form-data` with field `file`

**Output:**
```json
{
  "status": "indexed",
  "filename": "document.pdf",
  "pages": 12,
  "chunks": 47
}
```

### POST /ask

**Input:**
```json
{ "query": "What is Big Data?" }
```

**Output:**
```json
{
  "answer": "Big Data refers to...",
  "sources": [
    {
      "source": "document.pdf",
      "page": 3,
      "excerpt": "Big Data is defined as datasets that are too large..."
    }
  ]
}
```

---

## Configuration

Override the Ollama model via environment variable:

```bash
OLLAMA_MODEL=llama3 python app.py
```

### Reset the vector store

To clear all indexed documents and start fresh:

```bash
# Windows
rmdir /s /q db && mkdir db

# macOS / Linux
rm -rf db && mkdir db
```

---

## Known limitations

- **Session state only:** the Ask panel is unlocked in the browser after a successful upload in the same session. Refreshing the page hides it even if the ChromaDB index still exists. To re-enable, upload the PDF again (it will be re-indexed).
- **Accumulative indexing:** each upload adds chunks to the same ChromaDB collection. Uploading multiple PDFs will allow questions across all of them. Reset `./db` to start clean.
- **Local CPU inference:** response time varies with hardware. Expect 30–120 seconds per answer on CPU without a GPU.
- **Hallucination risk:** the prompt strictly instructs Mistral to stay within the retrieved context, but LLMs can occasionally ignore this constraint.
