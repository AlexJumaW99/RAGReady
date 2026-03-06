# states.py
import threading
from typing import Any, Dict, List, TypedDict, Literal, Optional


class FileMetadata(TypedDict):
    """The structured output format expected from Gemini for file metadata."""
    summary: str
    topics: List[str]
    key_entities: List[str]
    content_category: Literal["code", "documentation", "data", "media", "configuration", "other", "unknown"]
    quality_notes: str

class RAGIngestionState(TypedDict):
    """Full state for the multi-agent RAG ingestion pipeline."""

    # ---- 1. Session & Input ----
    session_id: str
    target_path: str                          # File or folder path from user
    project_tree: Optional[str]               # Pretty-printed directory tree
    output_dir: str                           # Local output folder for reviewable data

    # ---- 2. File Classification ----
    classified_files: Optional[Dict[str, List[Dict[str, Any]]]]

    # ---- 3. Processed Outputs ----
    processed_documents: Optional[List[Dict[str, Any]]]
    processed_media: Optional[List[Dict[str, Any]]]
    processed_structured: Optional[List[Dict[str, Any]]]

    # ---- 4. Metadata (LLM-enriched) ----
    file_metadata: Optional[Dict[str, Dict[str, Any]]]

    # ---- 5. PostgreSQL Configuration ----
    pg_host: str
    pg_port: int
    pg_username: str
    pg_password: str
    pg_database: str

    # ---- 6. Execution Tracking ----
    records_inserted: int
    current_step: Optional[str]
    steps_completed: Optional[List[str]]

    # ---- 7. Error Handling ----
    has_error: bool
    errors: Optional[List[str]]
    error_log_path: Optional[str]
    debug_summary: Optional[str]

    # ---- 8. Command Outputs ----
    last_command: Optional[str]
    last_stdout: Optional[str]
    last_stderr: Optional[str]


# Global cancellation event shared between the API server and pipeline nodes.
# The api_server sets this event; nodes check it between processing steps.
cancellation_event = threading.Event()
