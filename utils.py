# utils.py
import os
import sys
import psycopg2
from config import (
    get_llm,
    get_embedding_model,
    PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE,
)


# ---------------------------------------------------------------------------
# Graph Visualization
# ---------------------------------------------------------------------------

def save_graph_image(app, thread_id="default"):
    """Saves a Mermaid PNG of the graph structure."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Graphs")
    os.makedirs(output_dir, exist_ok=True)

    try:
        png_data = app.get_graph().draw_mermaid_png()
        output_path = os.path.join(output_dir, f"graph_{thread_id}.png")
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"  📊 Graph saved to: {output_path}")
        try:
            from IPython.display import Image, display
            display(Image(png_data))
        except Exception:
            pass
    except Exception as e:
        print(f"  ⚠️  Could not generate graph image: {e}")


# ---------------------------------------------------------------------------
# Project / Directory Tree Builder
# ---------------------------------------------------------------------------

_TREE_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache", "dist", "build", ".egg-info"}

def build_project_tree(root_path: str, prefix: str = "", max_depth: int = 4, _depth: int = 0) -> str:
    """Builds a pretty-printed directory tree string."""
    if _depth > max_depth:
        return ""

    if os.path.isfile(root_path):
        return os.path.basename(root_path) + "\n"

    entries = sorted(os.listdir(root_path))
    entries = [e for e in entries if not e.startswith(".") and e not in _TREE_SKIP_DIRS]

    lines = []
    if _depth == 0:
        lines.append(os.path.basename(os.path.abspath(root_path)) + "/")

    for i, entry in enumerate(entries):
        full_path = os.path.join(root_path, entry)
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if os.path.isdir(full_path):
            lines.append(f"{prefix}{connector}{entry}/")
            subtree = build_project_tree(
                full_path,
                prefix=prefix + extension,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            if subtree:
                lines.append(subtree.rstrip("\n"))
        else:
            lines.append(f"{prefix}{connector}{entry}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File Stat Helpers
# ---------------------------------------------------------------------------

def get_file_stats(filepath: str) -> dict:
    """Returns creation date, modified date, and size for a file."""
    try:
        stat = os.stat(filepath)
        return {
            "file_size_bytes": stat.st_size,
            "file_created_date": _epoch_to_iso(getattr(stat, "st_birthtime", stat.st_ctime)),
            "file_modified_date": _epoch_to_iso(stat.st_mtime),
        }
    except Exception:
        return {"file_size_bytes": 0, "file_created_date": None, "file_modified_date": None}


def _epoch_to_iso(epoch: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# RAG Query Engine (Simple semantic search + LLM answer)
# ---------------------------------------------------------------------------

def query_rag():
    """Interactive RAG query loop: embed question → pgvector search → LLM answer."""
    print("\n" + "=" * 60)
    print("  🔍  RAG Query Mode")
    print("  Type your question and press Enter. Type 'exit' to quit.")
    print("=" * 60)

    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USERNAME,
            password=PG_PASSWORD, dbname=PG_DATABASE, connect_timeout=10,
        )
    except Exception as e:
        print(f"  ❌ Could not connect to PostgreSQL: {e}")
        return

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
        print("  ⚠️  No RAG tables found. Run an ingestion first.")
        conn.close()
        return

    print(f"  📦 Available tables: {', '.join(sorted(existing_tables))}")

    while True:
        print()
        query = input("  ❓ Query: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        # 1. Embed the query
        query_embedding = get_embedding_model().encode(query).tolist()
        formatted_emb = f"[{','.join(map(str, query_embedding))}]"

        retrieved_chunks = []
        cur = conn.cursor()

        # 2. Search document_chunks
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
                    "metadata": row[2],
                    "similarity": round(float(row[3]), 4),
                    "table": "document_chunks",
                })

        # 3. Search media_files transcripts
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
                    "metadata": row[2],
                    "similarity": round(float(row[3]), 4),
                    "table": "media_files",
                })

        # 4. Search structured_files summaries
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
                    "metadata": row[2],
                    "similarity": round(float(row[3]), 4),
                    "table": "structured_files",
                })

        cur.close()

        if not retrieved_chunks:
            print("  ⚠️  No relevant results found.")
            continue

        # 5. Sort by similarity and build context
        retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}] (similarity: {chunk['similarity']}) — {chunk['source']}\n"
                f"{chunk['content'][:1500]}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        # 6. Ask Gemini to answer using the retrieved context
        rag_prompt = f"""You are a helpful assistant answering questions based on retrieved documents.
Use ONLY the context below to answer. If the context doesn't contain enough information, say so.
Cite which source(s) you used by referencing [Source N].

--- RETRIEVED CONTEXT ---
{context_block}

--- USER QUESTION ---
{query}

Provide a clear, concise answer:"""

        try:
            response = get_llm().invoke(rag_prompt)
            answer = response.content
        except Exception as e:
            answer = f"LLM error: {e}"

        print(f"\n  💡 Answer:\n{'─' * 56}")
        print(f"  {answer}")
        print(f"{'─' * 56}")

        # 7. Print source references
        print(f"\n  📄 Sources ({len(retrieved_chunks)} retrieved):")
        for i, chunk in enumerate(retrieved_chunks, 1):
            sim = chunk["similarity"]
            bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
            print(f"    [{i}] {bar} {sim:.3f}  {chunk['source']}  ({chunk['table']})")

    conn.close()
    print("\n  👋 Exiting query mode.\n")
