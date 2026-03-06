# config.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_pipeline")

# ---------------------------------------------------------------------------
# PostgreSQL Constants (read from .env)
# ---------------------------------------------------------------------------
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USERNAME = os.getenv("POSTGRES_USER", "admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
PG_DATABASE = os.getenv("POSTGRES_DATABASE", "vector")

if not PG_PASSWORD:
    logger.warning("POSTGRES_PASSWORD not set in .env — database operations may fail.")

# ---------------------------------------------------------------------------
# Media Processing Configuration (overridable via .env)
# ---------------------------------------------------------------------------
OCR_METHOD = os.getenv("OCR_METHOD", "gemini").lower()              # "gemini" or "tesseract"
TRANSCRIPTION_METHOD = os.getenv("TRANSCRIPTION_METHOD", "gemini").lower()  # "gemini" or "whisper"
MAX_BINARY_MB = int(os.getenv("MAX_BINARY_MB", "50"))

# Chunking config (tune for M1 8GB RAM)
CHUNK_SIZE_TEXT = int(os.getenv("CHUNK_SIZE_TEXT", "1000"))
CHUNK_OVERLAP_TEXT = int(os.getenv("CHUNK_OVERLAP_TEXT", "200"))
CHUNK_SIZE_CODE = int(os.getenv("CHUNK_SIZE_CODE", "1500"))
CHUNK_OVERLAP_CODE = int(os.getenv("CHUNK_OVERLAP_CODE", "200"))

# Structured data row cap
STRUCTURED_MAX_ROWS = int(os.getenv("STRUCTURED_MAX_ROWS", "500"))

# Query similarity threshold (0.0 to 1.0)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ---------------------------------------------------------------------------
# File Type Classifications
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm", ".log", ".tex"}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".r", ".sql", ".css", ".scss", ".less", ".vue", ".svelte",
}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".xml", ".env", ".ini", ".cfg", ".conf"}
PDF_EXTENSIONS = {".pdf"}
OFFICE_EXTENSIONS = {".docx", ".pptx"}
STRUCTURED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}

# Map code extensions → LangChain Language enum names (for code-aware splitting)
CODE_LANGUAGE_MAP = {
    ".py": "python", ".js": "js", ".ts": "ts", ".jsx": "js", ".tsx": "ts",
    ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "python", ".bash": "python",  # fallback splitter
    ".sql": "python", ".r": "python",    # fallback splitter
    ".html": "html", ".htm": "html", ".css": "python",
    ".scss": "python", ".less": "python",
    ".vue": "html", ".svelte": "html",
}

# MIME type map for Gemini multimodal API
MIME_TYPE_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}

# ---------------------------------------------------------------------------
# LLM & Embedding Model — Lazy Loading
# ---------------------------------------------------------------------------
# Models are loaded on first access, not at import time. This avoids loading
# the 384-dim embedding model and establishing LLM connections when only
# config constants are needed.

_llm = None
_genai_client = None
_embedding_model = None
_models_loaded = {"llm": False, "genai": False, "embedding": False}


def _connect_to_llm():
    """Establishes connection to Gemini via LangChain."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2,
        timeout=120,
        max_retries=4,
    )


def _connect_genai_client():
    """Creates a google-genai client for multimodal calls (vision, audio)."""
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found in .env — multimodal features will fail.")
        return None
    return genai.Client(api_key=api_key)


def _load_embedding_model():
    """Loads a local sentence transformer for vectorization (384-dim, fast on M1)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_llm():
    """Lazy accessor for the LLM singleton."""
    global _llm
    if not _models_loaded["llm"]:
        logger.info("Loading LLM (gemini-2.5-pro)...")
        _llm = _connect_to_llm()
        _models_loaded["llm"] = True
    return _llm


def get_genai_client():
    """Lazy accessor for the genai client singleton."""
    global _genai_client
    if not _models_loaded["genai"]:
        logger.info("Initializing Gemini genai client...")
        _genai_client = _connect_genai_client()
        _models_loaded["genai"] = True
    return _genai_client


def get_embedding_model():
    """Lazy accessor for the embedding model singleton."""
    global _embedding_model
    if not _models_loaded["embedding"]:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        _embedding_model = _load_embedding_model()
        _models_loaded["embedding"] = True
    return _embedding_model
