"""
RAG Pipeline API Server
=======================
A lightweight FastAPI wrapper around the ingestion and query pipeline.
Designed to be spawned by the Electron app and communicate via HTTP.

Optimized for macOS M1:
  - Single uvicorn worker (no multiprocessing overhead)
  - Lazy model loading on first use
  - Streaming responses to keep memory pressure low

Start manually (for testing):
    conda run -n langgraph python -u api_server.py

The Electron app spawns this automatically.
"""

import os
import sys
import uuid
import json
import datetime
import traceback
import threading
import queue
import io

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Ensure we can import from the same directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking a running ingestion
_ingestion_lock = threading.Lock()
_ingestion_thread = None
_ingestion_cancel = threading.Event()

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    target_path: str

class QueryRequest(BaseModel):
    question: str

class StopResponse(BaseModel):
    stopped: bool
    message: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_output_dir(target_path: str) -> str:
    """Creates a timestamped output directory under Processed_Output/."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Processed_Output")
    project_name = os.path.basename(os.path.abspath(target_path))
    project_name = project_name.replace(" ", "_").replace(".", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{project_name}_{timestamp}")

    subdirs = [
        "code", "text", "config", "pdf", "office",
        "media/images", "media/audio",
        "structured", "metadata",
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    return run_dir


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check — Electron polls this to know the server is ready."""
    return {"status": "ok", "pid": os.getpid()}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """
    Run the full RAG ingestion pipeline.
    Returns a streaming response (text/event-stream) with progress updates.
    Each line is a JSON object: {"event": "...", "data": {...}}
    """
    target_path = req.target_path

    if not os.path.exists(target_path):
        return JSONResponse(
            status_code=400,
            content={"error": f"Path does not exist: {target_path}"},
        )

    abs_target = os.path.abspath(target_path)

    # Use a queue to stream events from the ingestion thread to the response
    event_queue = queue.Queue()
    _ingestion_cancel.clear()

    def emit(event: str, data: dict):
        event_queue.put(json.dumps({"event": event, "data": data}) + "\n")

    def run_pipeline():
        global _ingestion_thread
        try:
            emit("status", {"message": "Initializing pipeline...", "step": "init"})

            # Lazy imports — only load heavy modules when actually ingesting
            from graph import create_ingestion_graph
            from config import PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE
            from utils import save_graph_image

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            emit("status", {"message": "Creating ingestion graph...", "step": "graph_init", "session_id": thread_id})

            app_graph = create_ingestion_graph()

            # Try to save graph image (non-critical)
            try:
                save_graph_image(app_graph, thread_id)
            except Exception:
                pass

            output_dir = _create_output_dir(abs_target)

            emit("status", {
                "message": f"Output directory: {output_dir}",
                "step": "output_dir",
                "output_dir": output_dir,
            })

            initial_state = {
                "session_id": thread_id,
                "target_path": abs_target,
                "project_tree": None,
                "output_dir": output_dir,
                "classified_files": None,
                "processed_documents": None,
                "processed_media": None,
                "processed_structured": None,
                "file_metadata": None,
                "pg_host": PG_HOST,
                "pg_port": PG_PORT,
                "pg_username": PG_USERNAME,
                "pg_password": PG_PASSWORD,
                "pg_database": PG_DATABASE,
                "records_inserted": 0,
                "current_step": None,
                "steps_completed": [],
                "has_error": False,
                "errors": [],
                "error_log_path": None,
                "debug_summary": None,
                "last_command": None,
                "last_stdout": None,
                "last_stderr": None,
            }

            emit("status", {"message": "Starting ingestion pipeline...", "step": "pipeline_start"})

            # ---- Run the full graph ----
            # We capture stdout to relay print statements from nodes
            old_stdout = sys.stdout
            captured = io.StringIO()

            class TeeWriter:
                """Writes to both the original stdout and a capture buffer."""
                def __init__(self, original, capture, emitter):
                    self.original = original
                    self.capture = capture
                    self.emitter = emitter
                def write(self, text):
                    self.original.write(text)
                    self.capture.write(text)
                    if text.strip():
                        self.emitter("log", {"text": text.rstrip()})
                def flush(self):
                    self.original.flush()
                    self.capture.flush()

            sys.stdout = TeeWriter(old_stdout, captured, emit)

            try:
                final_state = app_graph.invoke(initial_state, config)
            finally:
                sys.stdout = old_stdout

            # ---- Emit final result ----
            if final_state.get("has_error", False):
                emit("error", {
                    "message": "Ingestion failed",
                    "step": final_state.get("current_step", "unknown"),
                    "errors": final_state.get("errors", []),
                    "error_log_path": final_state.get("error_log_path"),
                })
            else:
                classified = final_state.get("classified_files") or {}
                file_counts = {cat: len(files) for cat, files in classified.items() if files}

                emit("complete", {
                    "message": "Ingestion completed successfully",
                    "session_id": thread_id,
                    "records_inserted": final_state.get("records_inserted", 0),
                    "steps_completed": final_state.get("steps_completed", []),
                    "file_counts": file_counts,
                    "output_dir": output_dir,
                })

        except Exception as e:
            emit("error", {
                "message": f"Pipeline exception: {str(e)}",
                "traceback": traceback.format_exc(),
            })
        finally:
            emit("done", {"message": "Stream finished"})
            _ingestion_thread = None

    # Start the pipeline in a background thread
    with _ingestion_lock:
        if _ingestion_thread and _ingestion_thread.is_alive():
            return JSONResponse(
                status_code=409,
                content={"error": "An ingestion is already running."},
            )
        _ingestion_thread = threading.Thread(target=run_pipeline, daemon=True)
        _ingestion_thread.start()

    def event_stream():
        while True:
            try:
                msg = event_queue.get(timeout=120)
                yield f"data: {msg}\n"
                # Check if this was the final message
                parsed = json.loads(msg)
                if parsed.get("event") == "done":
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'event': 'keepalive', 'data': {}})}\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    Run a RAG query: embed question → pgvector search → LLM answer.
    Returns JSON with the answer and source references.
    """
    question = req.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty."})

    try:
        # Lazy imports
        import psycopg2
        from config import (
            get_llm, get_embedding_model,
            PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE,
        )

        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USERNAME,
            password=PG_PASSWORD, dbname=PG_DATABASE, connect_timeout=10,
        )

        # Check which tables exist
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('document_chunks', 'media_files', 'structured_files')
        """)
        existing_tables = {row[0] for row in cur.fetchall()}
        cur.close()

        if not existing_tables:
            conn.close()
            return JSONResponse(
                status_code=404,
                content={"error": "No RAG tables found. Run an ingestion first."},
            )

        # Embed the query
        query_embedding = get_embedding_model().encode(question).tolist()
        formatted_emb = f"[{','.join(map(str, query_embedding))}]"

        retrieved_chunks = []
        cur = conn.cursor()

        # Search document_chunks
        if "document_chunks" in existing_tables:
            cur.execute("""
                SELECT filepath, content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 5
            """, (formatted_emb, formatted_emb))
            for row in cur.fetchall():
                retrieved_chunks.append({
                    "source": row[0],
                    "content": row[1],
                    "metadata": row[2] if isinstance(row[2], dict) else {},
                    "similarity": round(float(row[3]), 4),
                    "table": "document_chunks",
                })

        # Search media_files transcripts
        if "media_files" in existing_tables:
            cur.execute("""
                SELECT filepath, transcript, metadata,
                       1 - (transcript_embedding <=> %s::vector) AS similarity
                FROM media_files
                WHERE transcript_embedding IS NOT NULL
                ORDER BY transcript_embedding <=> %s::vector
                LIMIT 3
            """, (formatted_emb, formatted_emb))
            for row in cur.fetchall():
                retrieved_chunks.append({
                    "source": row[0],
                    "content": row[1] or "",
                    "metadata": row[2] if isinstance(row[2], dict) else {},
                    "similarity": round(float(row[3]), 4),
                    "table": "media_files",
                })

        # Search structured_files summaries
        if "structured_files" in existing_tables:
            cur.execute("""
                SELECT filepath, content, metadata,
                       1 - (summary_embedding <=> %s::vector) AS similarity
                FROM structured_files
                WHERE summary_embedding IS NOT NULL
                ORDER BY summary_embedding <=> %s::vector
                LIMIT 2
            """, (formatted_emb, formatted_emb))
            for row in cur.fetchall():
                retrieved_chunks.append({
                    "source": row[0],
                    "content": row[1][:2000] if row[1] else "",
                    "metadata": row[2] if isinstance(row[2], dict) else {},
                    "similarity": round(float(row[3]), 4),
                    "table": "structured_files",
                })

        cur.close()
        conn.close()

        if not retrieved_chunks:
            return {"answer": "No relevant results found in the knowledge base.", "sources": []}

        # Sort by similarity
        retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)

        # Build context for LLM
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}] (similarity: {chunk['similarity']}) — {chunk['source']}\n"
                f"{chunk['content'][:1500]}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        rag_prompt = f"""You are a helpful assistant answering questions based on retrieved documents.
Use ONLY the context below to answer. If the context doesn't contain enough information, say so.
Cite which source(s) you used by referencing [Source N].

--- RETRIEVED CONTEXT ---
{context_block}

--- USER QUESTION ---
{question}

Provide a clear, concise answer:"""

        try:
            response = get_llm().invoke(rag_prompt)
            answer = response.content
        except Exception as e:
            answer = f"LLM error: {str(e)}"

        # Build source references for the frontend
        sources = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            sources.append({
                "index": i,
                "source": chunk["source"],
                "similarity": chunk["similarity"],
                "table": chunk["table"],
                "preview": (chunk["content"][:200] + "...") if len(chunk["content"]) > 200 else chunk["content"],
            })

        return {
            "answer": answer,
            "sources": sources,
            "tables_searched": list(existing_tables),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/stop")
async def stop():
    """Kill any running ingestion process."""
    global _ingestion_thread
    _ingestion_cancel.set()

    if _ingestion_thread and _ingestion_thread.is_alive():
        # We can't forcibly kill a thread in Python, but setting the cancel
        # event will be checked by cooperative code. For a hard stop, the
        # Electron app kills the entire server process.
        return {"stopped": True, "message": "Cancel signal sent to ingestion."}

    return {"stopped": False, "message": "No ingestion was running."}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("RAG_API_PORT", "8787"))
    print(f"[RAG API] Starting server on http://localhost:{port}")
    print(f"[RAG API] PID: {os.getpid()}")
    sys.stdout.flush()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        # M1 optimization: single worker, no multiprocessing
        workers=1,
        timeout_keep_alive=300,
    )
