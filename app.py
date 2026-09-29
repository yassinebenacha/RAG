"""
RAG — Document Q&A
Flask + LangChain + ChromaDB + Ollama (Mistral)

Endpoints:
  GET  /       — Serve web UI
  POST /pdf    — Upload a PDF, extract text, chunk, embed, store in ChromaDB
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

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PDF_DIR = "pdf"
DB_DIR  = "db"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR,  exist_ok=True)

# ── LLM — Ollama / Mistral ─────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
llm = Ollama(model=OLLAMA_MODEL)

# ── Embeddings — FastEmbed ─────────────────────────────────────────────────────
embeddings = FastEmbedEmbeddings()

# ── Text splitter ──────────────────────────────────────────────────────────────
# 1024-char chunks with 80-char overlap to preserve cross-boundary context.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
)

# ── Prompt ─────────────────────────────────────────────────────────────────────
# Strictly grounds the model in retrieved context to prevent hallucinations.
RAG_PROMPT = PromptTemplate.from_template(
    "<s>[INST] You are a helpful assistant. "
    "Answer the question using ONLY the information provided in the context below. "
    "If the answer is not present in the context, reply with exactly this sentence: "
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

    Input:  multipart/form-data  { file: <PDF> }
    Output: { status, filename, pages, chunks }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    # 1. Save the PDF to disk
    file_path = os.path.join(PDF_DIR, file.filename)
    file.save(file_path)

    # 2. Extract text page by page
    loader = PDFPlumberLoader(file_path)
    docs   = loader.load_and_split()

    # 3. Split into overlapping chunks
    chunks = text_splitter.split_documents(docs)

    # 4. Embed chunks and persist them in ChromaDB
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

    Input:  JSON { "query": "<question>" }
    Output: { answer, sources: [{ source, page, excerpt }] }
    """
    body  = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()

    if not query:
        return jsonify({"error": "The 'query' field is required"}), 400

    # Load persisted vector store and verify it is not empty
    try:
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )
        if vector_store._collection.count() == 0:
            raise ValueError("empty")
    except Exception:
        return jsonify({"error": "No documents indexed yet. Please upload a PDF first."}), 400

    # Retrieve the top-5 chunks that exceed the similarity threshold
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.1},
    )

    # Build the RAG chain: retriever → context injection → LLM
    doc_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    chain     = create_retrieval_chain(retriever, doc_chain)
    result    = chain.invoke({"input": query})

    # Format source metadata for the UI
    sources = []
    for doc in result.get("context", []):
        raw_page = doc.metadata.get("page")
        sources.append({
            "source":  os.path.basename(doc.metadata.get("source", "unknown")),
            "page":    (raw_page + 1) if isinstance(raw_page, int) else None,
            "excerpt": doc.page_content[:250].strip(),
        })

    return jsonify({
        "answer":  result["answer"],
        "sources": sources,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
