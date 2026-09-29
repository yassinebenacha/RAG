"""
RAG — Document Q&A
Flask + LangChain + ChromaDB + Ollama (Mistral)

Endpoints:
  GET  /       — Serve the web UI
  POST /pdf    — Upload a PDF, extract, chunk, embed, store in ChromaDB
  POST /ask    — Retrieve relevant chunks and generate an answer with Mistral
"""

from flask import Flask, request, send_from_directory, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
import socket

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PDF_DIR = "pdf"
DB_DIR  = "db"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR,  exist_ok=True)

# ── LLM — Ollama / Mistral ───────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
llm = Ollama(model=OLLAMA_MODEL)

# ── Embeddings — FastEmbed ─────────────────────────────────────────────────────
# Model: BAAI/bge-small-en-v1.5 (downloaded on first run, ~60 MB)
embeddings = FastEmbedEmbeddings()

# ── Text splitter ──────────────────────────────────────────────────────────────
# 1024-char chunks, 80-char overlap to avoid cutting context at chunk boundaries.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
)

# ── Prompt ─────────────────────────────────────────────────────────────────────
# Strictly grounds the model in retrieved context — prevents hallucinations.
# Uses Mistral's instruction format: <s>[INST]...[/INST]</s>
RAG_PROMPT = PromptTemplate.from_template(
    "<s>[INST] You are a helpful assistant. "
    "Answer the question using ONLY the information provided in the context below. "
    "If the answer is not present in the context, reply with exactly: "
    "'The document does not contain information about this topic.' [/INST]</s>\n\n"
    "[INST]\nQuestion: {input}\n\nContext:\n{context}\n\nAnswer:\n[/INST]"
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page web UI."""
    base = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base, "index.html")


@app.route("/pdf", methods=["POST"])
def upload_pdf():
    """
    Upload a PDF and index it into ChromaDB.

    Pipeline: save → PDFPlumberLoader → RecursiveCharacterTextSplitter
              → FastEmbedEmbeddings → Chroma.persist()

    Input:  multipart/form-data  { file: <PDF> }
    Output: { status, filename, pages, chunks }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    # 1. Persist PDF to disk
    file_path = os.path.join(PDF_DIR, file.filename)
    file.save(file_path)

    # 2. Extract text — one Document per page
    loader = PDFPlumberLoader(file_path)
    docs   = loader.load_and_split()

    # 3. Split pages into overlapping chunks
    chunks = text_splitter.split_documents(docs)

    # 4. Embed and persist in ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vector_store.persist()

    return jsonify({
        "status":   "indexed",
        "filename": file.filename,
        "pages":    len(docs),
        "chunks":   len(chunks),
    })


@app.route("/ask", methods=["POST"])
def ask():
    """
    Retrieve the most relevant chunks from ChromaDB and generate an answer.

    Pipeline: embed query → similarity search (k=5) → stuff context → Mistral

    Input:  JSON { "query": "<question>" }
    Output: { answer, sources: [{ source, page, excerpt }] }
    """
    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()

    if not query:
        return jsonify({"error": "The 'query' field is required."}), 400

    # ── Load vector store ───────────────────────────────────────────────────────
    try:
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )
    except Exception as exc:
        return jsonify({
            "error": "Could not load the vector store. Try re-uploading your PDF.",
        }), 500

    if vector_store._collection.count() == 0:
        return jsonify({
            "error": "No documents indexed yet. Please upload a PDF first.",
        }), 400

    # ── Retrieve top-k relevant chunks ──────────────────────────────────────────
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.1},
    )

    # ── Quick Ollama health-check via TCP (2 s timeout) ──────────────────────────
    # A TCP connect to port 11434 is enough to confirm Ollama is up.
    # This avoids hanging the browser indefinitely when Ollama is stopped.
    try:
        with socket.create_connection(("localhost", 11434), timeout=2):
            pass
    except (socket.timeout, ConnectionRefusedError, OSError):
        return jsonify({
            "error": "LLM service unavailable. Make sure Ollama is running: ollama serve",
        }), 503

    # ── Generate answer ─────────────────────────────────────────────────────────
    doc_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    chain     = create_retrieval_chain(retriever, doc_chain)

    try:
        result = chain.invoke({"input": query})
    except Exception as exc:
        # Ollama crashed or timed out mid-generation
        return jsonify({
            "error": "LLM service unavailable. Make sure Ollama is running: ollama serve",
        }), 503

    # ── Format sources — deduplicated by (source, page) ────────────────────────
    seen    = set()
    sources = []
    for doc in result.get("context", []):
        raw_page    = doc.metadata.get("page")
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        page_num    = (raw_page + 1) if isinstance(raw_page, int) else None
        key         = (source_name, page_num)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":  source_name,
                "page":    page_num,
                "excerpt": doc.page_content[:250].strip(),
            })

    return jsonify({
        "answer":  result["answer"],
        "sources": sources,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
