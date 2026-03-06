# main.py
import os
import sys
import uuid
import subprocess
import datetime
import re


# --- 1. CONDA ENVIRONMENT CHECK & RELAUNCH ---
def ensure_conda_env(env_name="langgraph"):
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env != env_name:
        print(f"🔄 Activating '{env_name}' environment and re-launching...")
        try:
            subprocess.check_call(
                ["conda", "run", "--no-capture-output", "-n", env_name, "python", __file__] + sys.argv[1:]
            )
        except Exception:
            print(f"❌ Error: Could not run in conda environment '{env_name}'.")
        sys.exit(0)

ensure_conda_env("langgraph")
# ---------------------------------------------

from graph import create_ingestion_graph
from config import PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE
from utils import save_graph_image, query_rag


def _create_output_dir(target_path: str) -> str:
    """
    Creates a timestamped output directory under Processed_Output/.

    Structure:
        Processed_Output/
        └── <project_name>_<YYYYMMDD_HHMMSS>/
            ├── code/
            ├── text/
            ├── config/
            ├── pdf/
            ├── office/
            ├── media/
            │   ├── images/
            │   └── audio/
            ├── structured/
            └── metadata/
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Processed_Output")

    # Name the folder after the project/file being ingested
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


def run_ingestion():
    """Runs the full RAG ingestion pipeline."""
    app = create_ingestion_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    save_graph_image(app, thread_id)

    print(f"\n  Session ID : {thread_id}")
    print(f"  PostgreSQL : {PG_USERNAME}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}")
    sys.stdout.flush()

    # Prompt user for file or folder path
    print("\n" + "=" * 60)
    target_path = input("  📂 Enter the path to the file or folder to ingest: ").strip()

    if not os.path.exists(target_path):
        print(f"  ❌ Error: Path '{target_path}' does not exist.")
        sys.exit(1)

    abs_target = os.path.abspath(target_path)

    # Create the local output directory
    output_dir = _create_output_dir(abs_target)
    print(f"  📁 Output Dir: {output_dir}")

    initial_state = {
        # Session
        "session_id": thread_id,
        "target_path": abs_target,
        "project_tree": None,
        "output_dir": output_dir,

        # Classification & Processing
        "classified_files": None,
        "processed_documents": None,
        "processed_media": None,
        "processed_structured": None,

        # Metadata
        "file_metadata": None,

        # PostgreSQL
        "pg_host": PG_HOST,
        "pg_port": PG_PORT,
        "pg_username": PG_USERNAME,
        "pg_password": PG_PASSWORD,
        "pg_database": PG_DATABASE,

        # Tracking
        "records_inserted": 0,
        "current_step": None,
        "steps_completed": [],

        # Errors
        "has_error": False,
        "errors": [],
        "error_log_path": None,
        "debug_summary": None,

        # Command outputs
        "last_command": None,
        "last_stdout": None,
        "last_stderr": None,
    }

    print(f"\n  🚀 Starting RAG Ingestion for: {target_path}\n")
    print("=" * 60)
    sys.stdout.flush()

    final_state = app.invoke(initial_state, config)

    # ---- Print Results ----
    print("\n" + "=" * 60)
    if final_state.get("has_error", False):
        print("  ❌ WORKFLOW STOPPED DUE TO ERROR")
        print(f"\n  Failed at step  : {final_state.get('current_step', 'N/A')}")
        for err in final_state.get("errors", []):
            print(f"    • {err}")
        log_path = final_state.get("error_log_path")
        if log_path:
            print(f"\n  📋 Debug log: {log_path}")
    else:
        print("  ✅ INGESTION COMPLETED SUCCESSFULLY")
        print(f"  Records Inserted : {final_state.get('records_inserted', 0)}")
        print(f"  Steps Completed  : {' → '.join(final_state.get('steps_completed', []))}")

        classified = final_state.get("classified_files") or {}
        for cat, files in classified.items():
            if files:
                print(f"    {cat:>12}: {len(files)} file(s)")

    print(f"\n  📂 Local output saved to:")
    print(f"     {output_dir}")
    print("=" * 60)
    sys.stdout.flush()


def _vlen(s):
    """Visible length, ignoring ANSI escape codes."""
    return len(re.sub(r'\033\[[0-9;]*m', '', s))

def _pad(content, width=62):
    """Right-pad content to exact visible width."""
    return content + ' ' * (width - _vlen(content))


# def print_banner():
#     """Display the Maiya startup banner."""
#     C = "\033[36m"    # cyan
#     M = "\033[35m"    # magenta
#     B = "\033[1m"     # bold
#     D = "\033[2m"     # dim
#     R = "\033[0m"     # reset
#     W = "\033[97m"    # bright white
#     Y = "\033[33m"    # yellow

#     banner = f"""
# {M}    ╔══════════════════════════════════════════════════════════════╗
#     ║                                                              ║
#     ║  {C}{B}  ███╗   ███╗  █████╗  ██╗ ██╗   ██╗  █████╗  {M}            ║
#     ║  {C}{B}  ████╗ ████║ ██╔══██╗ ██║ ╚██╗ ██╔╝ ██╔══██╗ {M}            ║
#     ║  {C}{B}  ██╔████╔██║ ███████║ ██║  ╚████╔╝  ███████║ {M}            ║
#     ║  {C}{B}  ██║╚██╔╝██║ ██╔══██║ ██║   ╚██╔╝   ██╔══██║ {M}            ║
#     ║  {C}{B}  ██║ ╚═╝ ██║ ██║  ██║ ██║    ██║    ██║  ██║ {M}            ║
#     ║  {C}{B}  ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝    ╚═╝    ╚═╝  ╚═╝ {M}            ║
#     ║                                                              ║
#     ║  {D}{W}  R A G   P I P E L I N E  {R}{M}                                ║
#     ║  {D}{W}  Multi-Agent Ingestion & Query System{R}{M}                      ║
#     ║                                                              ║
#     ║  {Y}▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀{M}  ║
#     ║  {D}  Powered by LangGraph | Gemini | pgvector{R}{M}                  ║
#     ╚══════════════════════════════════════════════════════════════╝{R}
# """
#     print(banner)

def print_banner():
    C = "\033[36m"; M = "\033[35m"; B = "\033[1m"
    D = "\033[2m";  R = "\033[0m";  W = "\033[97m"; Y = "\033[33m"

    row = lambda s: f"    ║{_pad(s)}║"

    banner = f"""
{M}    ╔══════════════════════════════════════════════════════════════╗
{row('')}
{row(f'  {C}{B}  ███╗   ███╗  █████╗  ██╗ ██╗   ██╗  █████╗  {M}')}
{row(f'  {C}{B}  ████╗ ████║ ██╔══██╗ ██║ ╚██╗ ██╔╝ ██╔══██╗ {M}')}
{row(f'  {C}{B}  ██╔████╔██║ ███████║ ██║  ╚████╔╝  ███████║ {M}')}
{row(f'  {C}{B}  ██║╚██╔╝██║ ██╔══██║ ██║   ╚██╔╝   ██╔══██║ {M}')}
{row(f'  {C}{B}  ██║ ╚═╝ ██║ ██║  ██║ ██║    ██║    ██║  ██║ {M}')}
{row(f'  {C}{B}  ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝    ╚═╝    ╚═╝  ╚═╝ {M}')}
{row('')}
{row(f'  {D}{W}  R A G   P I P E L I N E  {R}{M}')}
{row(f'  {D}{W}  Multi-Agent Ingestion & Query System{R}{M}')}
{row('')}
{row(f'  {Y}▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀{M}')}
{row(f'  {D}  Powered by LangGraph | Gemini | pgvector{R}{M}')}
{M}    ╚══════════════════════════════════════════════════════════════╝{R}
"""
    print(banner)


def main():
    print_banner()

    M = "\033[35m"
    C = "\033[36m"
    W = "\033[97m"
    D = "\033[2m"
    B = "\033[1m"
    R = "\033[0m"

    print(f"  {W}Select mode:{R}")
    print(f"    {C}[1]{R} Ingest  {D}— Process and store files for RAG{R}")
    print(f"    {C}[2]{R} Query   {D}— Search your ingested knowledge base{R}")
    print(f"    {C}[3]{R} Exit")
    print()

    choice = input(f"  {M}{B}>{R} Enter choice (1/2/3): ").strip()

    if choice == "1":
        run_ingestion()
    elif choice == "2":
        query_rag()
    elif choice == "3":
        print(f"\n  {D}Goodbye from Maiya.{R}\n")
        sys.exit(0)
    else:
        print(f"  {M}Invalid choice: '{choice}'. Please enter 1, 2, or 3.{R}")
        sys.exit(1)


if __name__ == "__main__":
    main()
