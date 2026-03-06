<p align="center">
  <pre>
      ███╗   ███╗  █████╗  ██╗ ██╗   ██╗  █████╗
      ████╗ ████║ ██╔══██╗ ██║ ╚██╗ ██╔╝ ██╔══██╗
      ██╔████╔██║ ███████║ ██║  ╚████╔╝  ███████║
      ██║╚██╔╝██║ ██╔══██║ ██║   ╚██╔╝   ██╔══██║
      ██║ ╚═╝ ██║ ██║  ██║ ██║    ██║    ██║  ██║
      ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝    ╚═╝    ╚═╝  ╚═╝
  </pre>
</p>

<h3 align="center">Multi-Agent RAG Ingestion & Query Pipeline</h3>

<p align="center">
  <em>Powered by LangGraph · Gemini 2.5 Pro · pgvector · all-MiniLM-L6-v2</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-purple" alt="LangGraph">
  <img src="https://img.shields.io/badge/Gemini-2.5_Pro-orange?logo=google" alt="Gemini">
  <img src="https://img.shields.io/badge/pgvector-384_dim-green?logo=postgresql" alt="pgvector">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-teal?logo=fastapi" alt="FastAPI">
</p>

---

## What is Maiya?

Maiya is a **multi-agent RAG pipeline** that ingests virtually any file type — source code, documents, PDFs, images, audio, spreadsheets — and transforms them into a searchable, AI-powered knowledge base backed by PostgreSQL with pgvector.

The pipeline is orchestrated as a **LangGraph state machine** where each processing step is an independent agent node with built-in error handling, LLM-powered debugging, and automatic recovery routing.

### Supported File Types

| Category | Extensions | Processing Method |
|---|---|---|
| **Source Code** | `.py` `.js` `.ts` `.java` `.c` `.cpp` `.go` `.rs` `.rb` `.php` `.swift` `.kt` `.scala` `.sh` `.sql` `.css` `.vue` `.svelte` + more | Language-aware chunking via LangChain |
| **Text & Docs** | `.md` `.txt` `.rst` `.html` `.log` `.tex` | Recursive text splitting |
| **Configuration** | `.json` `.yaml` `.yml` `.toml` `.xml` `.env` `.ini` `.cfg` `.conf` | Direct read + text splitting |
| **PDFs** | `.pdf` | PyMuPDF extraction + Gemini Vision fallback for scanned pages |
| **Office** | `.docx` `.pptx` | python-docx / python-pptx extraction |
| **Images** | `.png` `.jpg` `.jpeg` `.gif` `.bmp` `.tiff` `.webp` | Gemini Vision OCR (or Tesseract) |
| **Audio** | `.mp3` `.wav` `.m4a` `.ogg` `.flac` `.aac` `.wma` | Gemini Audio transcription (or Whisper) |
| **Structured Data** | `.csv` `.tsv` `.xlsx` `.xls` | Pandas ingestion with column/type analysis |

---

## Architecture

Maiya's pipeline is a linear graph of seven agent nodes. Each node writes its results into a shared `RAGIngestionState` and includes a conditional error check — if any step fails, the graph short-circuits to `END` and produces an LLM-generated debug report.

```
START
  │
  ▼
┌─────────────────────────┐
│  1. Read & Classify      │  Walk target path, sort files by type
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  2. Process Text Docs    │  Extract from code, text, config, PDF, Office
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  3. Process Media        │  OCR images, transcribe audio
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  4. Process Structured   │  Read CSV/XLSX via Pandas
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  5. Generate Metadata    │  Gemini structured output per file
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  6. Setup PostgreSQL     │  Create/repair tables + pgvector extension
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  7. Vectorize & Store    │  Chunk → embed → INSERT into pgvector
└─────────┘
          ▼
         END
```

On error at any node, the pipeline calls Gemini to analyze the failure and writes a detailed debug log to the `Logs/` directory.

---

## Project Structure

```
maiya/
├── main.py              # CLI entry point (Ingest / Query / Exit)
├── api_server.py        # FastAPI server for Electron frontend
├── graph.py             # LangGraph state machine definition
├── nodes.py             # All 7 pipeline agent nodes
├── states.py            # RAGIngestionState TypedDict + FileMetadata schema
├── config.py            # Environment, model loaders, file type maps
├── utils.py             # Tree builder, file stats, RAG query engine
├── pg_tool.py           # Standalone CLI for browsing PostgreSQL tables
├── requirements.txt     # Python dependencies
└── gemini_models.txt    # Reference list of available Gemini model IDs
```

---

## Prerequisites

**Required:**

- Python 3.10+
- Conda (the project uses a `langgraph` conda environment)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- A Google API key with access to Gemini models

**Optional (for offline processing):**

- Tesseract OCR — alternative to Gemini Vision for image text extraction
- OpenAI Whisper — alternative to Gemini Audio for audio transcription

---

## Installation

### 1. Create and activate the conda environment

```bash
conda create -n langgraph python=3.11 -y
conda activate langgraph
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL with pgvector

Make sure PostgreSQL is running and the `pgvector` extension is available. If you haven't installed it yet:

```sql
-- Inside psql
CREATE EXTENSION IF NOT EXISTS vector;
```

Create a database for the project (or use an existing one):

```sql
CREATE DATABASE vector;
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your-google-api-key-here

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=admin
POSTGRES_PASSWORD=your-password
POSTGRES_DATABASE=vector

# Optional overrides
OCR_METHOD=gemini            # "gemini" or "tesseract"
TRANSCRIPTION_METHOD=gemini  # "gemini" or "whisper"
CHUNK_SIZE_TEXT=1000
CHUNK_OVERLAP_TEXT=200
CHUNK_SIZE_CODE=1500
CHUNK_OVERLAP_CODE=200
SIMILARITY_THRESHOLD=0.3
MAX_BINARY_MB=50
LOG_LEVEL=INFO
```

---

## Usage

### CLI Mode

Run the interactive CLI:

```bash
python main.py
```

You'll see three options:

```
  [1] Ingest  — Process and store files for RAG
  [2] Query   — Search your ingested knowledge base
  [3] Exit
```

**Ingesting files:** Select option `1` and provide the path to any file or folder. Maiya will classify every file, extract text, generate metadata, and store everything as 384-dimensional vectors in PostgreSQL.

**Querying:** Select option `2` to enter an interactive query loop. Type natural-language questions and Maiya will retrieve the most relevant chunks, pass them to Gemini, and return a cited answer.

### API Server Mode

For integration with a frontend (e.g., an Electron app), start the FastAPI server:

```bash
python api_server.py
```

The server runs on `http://localhost:8787` by default (configurable via the `RAG_API_PORT` environment variable).

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/ingest` | Start ingestion (streaming SSE response with progress) |
| `POST` | `/query` | Run a RAG query and get an answer with sources |
| `POST` | `/stop` | Send a cancellation signal to a running ingestion |

**Ingest request:**

```bash
curl -X POST http://localhost:8787/ingest \
  -H "Content-Type: application/json" \
  -d '{"target_path": "/path/to/your/project"}'
```

**Query request:**

```bash
curl -X POST http://localhost:8787/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the authentication system work?"}'
```

The `/ingest` endpoint returns a **Server-Sent Events stream** with real-time progress. Each event is a JSON object with an `event` field (`status`, `log`, `error`, `complete`, or `done`) and a `data` payload.

---

## How the Pipeline Works

### Step 1 — Read & Classify

Walks the target path and sorts every file into one of nine categories (code, text, config, pdf, office, structured, image, audio, unknown) based on its extension. Generates a project tree and saves a `_classification.json` summary.

### Step 2 — Process Text Documents

Extracts readable text from all text-based files. PDFs use PyMuPDF, with automatic Gemini Vision fallback for scanned/image-heavy pages. Office documents use python-docx and python-pptx. Code and config files are read directly.

### Step 3 — Process Media

Images are sent through OCR (Gemini Vision by default) to extract visible text or generate a description. Audio files are transcribed (Gemini Audio by default). Both originals and transcripts are saved locally.

### Step 4 — Process Structured Data

CSV, TSV, and Excel files are loaded via Pandas (capped at 500 rows by default). Column names, data types, and row counts are recorded. A preview JSON is saved alongside the original file.

### Step 5 — Generate Metadata

Each file's content is sent to Gemini with the project tree for context. Using LangChain's structured output, Gemini returns a typed metadata object containing a summary, topics, key entities, content category, and quality notes.

### Step 6 — Setup PostgreSQL

Creates three tables if they don't exist (or repairs them if the schema has drifted):

- `document_chunks` — chunked text with 384-dim embeddings
- `media_files` — binary data + transcript embeddings
- `structured_files` — tabular content + summary embeddings

### Step 7 — Vectorize & Store

Text documents are split using language-aware chunkers (code files get syntax-aware splitting). All chunks are embedded with `all-MiniLM-L6-v2` (384 dimensions, runs locally) and inserted into PostgreSQL via pgvector.

---

## Output Structure

Every ingestion run creates a timestamped folder under `Processed_Output/`:

```
Processed_Output/
└── my_project_20250305_143022/
    ├── _README.txt           # Human-readable summary
    ├── _overview.json        # Machine-readable run summary
    ├── _project_tree.txt     # Directory tree of the source
    ├── _classification.json  # File classification breakdown
    ├── code/                 # .extracted.txt, .chunks.json, .metadata.json
    ├── text/                 # Same structure as code/
    ├── config/               # Same structure as code/
    ├── pdf/                  # Same structure as code/
    ├── office/               # Same structure as code/
    ├── media/
    │   ├── images/           # Originals + .ocr.txt
    │   └── audio/            # Originals + .transcript.txt
    ├── structured/           # Originals + .preview.json
    └── metadata/
        └── all_metadata.json # Combined metadata for all files
```

---

## Database Management

Use `pg_tool.py` for interactive PostgreSQL inspection:

```bash
python pg_tool.py
```

This provides a rich terminal UI for connecting to your database, browsing tables, viewing all rows with formatted output, and deleting data when needed.

---

## Configuration Reference

All settings are configurable via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. API key for Gemini models |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `admin` | PostgreSQL username |
| `POSTGRES_PASSWORD` | — | PostgreSQL password |
| `POSTGRES_DATABASE` | `vector` | PostgreSQL database name |
| `OCR_METHOD` | `gemini` | Image OCR backend (`gemini` or `tesseract`) |
| `TRANSCRIPTION_METHOD` | `gemini` | Audio transcription backend (`gemini` or `whisper`) |
| `CHUNK_SIZE_TEXT` | `1000` | Characters per chunk for text files |
| `CHUNK_OVERLAP_TEXT` | `200` | Overlap between text chunks |
| `CHUNK_SIZE_CODE` | `1500` | Characters per chunk for code files |
| `CHUNK_OVERLAP_CODE` | `200` | Overlap between code chunks |
| `STRUCTURED_MAX_ROWS` | `500` | Max rows to read from CSV/XLSX |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity for query results |
| `MAX_BINARY_MB` | `50` | Max file size (MB) to store as binary in PostgreSQL |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `RAG_API_PORT` | `8787` | Port for the FastAPI server |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Orchestration | **LangGraph** | Multi-agent state graph with conditional routing |
| LLM | **Gemini 2.5 Pro** | Metadata generation, OCR, transcription, RAG answers |
| Embeddings | **all-MiniLM-L6-v2** | Local 384-dim sentence embeddings (fast on Apple Silicon) |
| Vector Store | **PostgreSQL + pgvector** | Cosine similarity search on embedded chunks |
| Text Splitting | **LangChain** | Language-aware recursive chunking |
| PDF Extraction | **PyMuPDF** | Fast text extraction with Gemini Vision fallback |
| API Layer | **FastAPI + Uvicorn** | Streaming SSE endpoint for frontend integration |
| Structured Data | **Pandas** | CSV/TSV/XLSX reading and analysis |

---

## Troubleshooting

**"POSTGRES_PASSWORD not set"** — Make sure your `.env` file is in the project root and contains a valid `POSTGRES_PASSWORD`.

**"pgvector extension not found"** — Install pgvector for your PostgreSQL version. On macOS with Homebrew: `brew install pgvector`.

**"GOOGLE_API_KEY not found"** — Gemini features (OCR, transcription, metadata, and query answers) all require a valid API key in your `.env` file.

**Pipeline errors** — When any node fails, Maiya automatically generates a detailed debug report in the `Logs/` directory with root-cause analysis and suggested fixes from Gemini.

**Conda environment issues** — `main.py` automatically re-launches itself inside the `langgraph` conda environment. Make sure the environment exists and has all dependencies installed.

---

<br/>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,50:3B82F6,100:06B6D4&height=120&section=footer&animation=fadeIn" width="100%"/>
</p>

<p align="center">
  <sub>Optimized for Apple Silicon. Optimized for M1 8GB RAM.</sub>
  <br/>
  <sub>Built with ❤️ by Alex Juma, using <b>Langgraph</b>, <b>Gemini</b>, and a whole lot of automation 👨🏾‍💻</sub>
</p>
