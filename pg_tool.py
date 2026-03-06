#!/usr/bin/env python3
"""
pg_tool.py — Interactive CLI for browsing and managing PostgreSQL tables.

Requirements:
    pip install psycopg2-binary rich

Usage:
    python pg_tool.py
"""

import getpass
import sys

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    sys.exit("psycopg2 is not installed. Run:  pip install psycopg2-binary")

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text
except ImportError:
    sys.exit("rich is not installed. Run:  pip install rich")

console = Console()


# ── Helpers ──────────────────────────────────────────────────────────────────

def prompt(label, default=None):
    """Prompt with an optional default shown in brackets."""
    suffix = f" [{default}]: " if default else ": "
    value = input(f"  {label}{suffix}").strip()
    return value or default


def connect(host, port, user, password, dbname):
    """Return a psycopg2 connection or exit on failure."""
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=dbname
        )
        conn.autocommit = False
        return conn
    except psycopg2.OperationalError as e:
        sys.exit(f"\n✗ Connection failed:\n  {e}")


def list_tables(conn):
    """Return a sorted list of user tables in the public schema."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name;"
        )
        return [row[0] for row in cur.fetchall()]


def pick_table(conn):
    """Let the user choose a table from the public schema."""
    tables = list_tables(conn)
    if not tables:
        print("\n  No tables found in the public schema.")
        return None

    print("\n  Available tables:")
    for i, t in enumerate(tables, 1):
        print(f"    {i}. {t}")

    while True:
        choice = input("\n  Select a table (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(tables):
            return tables[int(choice) - 1]
        if choice in tables:
            return choice
        print("  Invalid choice. Try again.")


def show_rows(conn, table):
    """Fetch and display every row in a table with a rich bordered grid."""
    safe = sql.Identifier(table)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(safe))
        count = cur.fetchone()[0]

        if count == 0:
            console.print(f"\n  Table [bold]'{table}'[/bold] is empty.", style="yellow")
            return

        cur.execute(sql.SQL("SELECT * FROM {}").format(safe))
        rows = cur.fetchall()
        headers = [desc[0] for desc in cur.description]

    # Build a Rich table with full borders around every cell
    rtable = Table(
        title=f"  {table}  ·  {count} row(s)",
        title_style="bold cyan",
        box=box.HEAVY_EDGE,
        show_lines=True,        # horizontal line between every row
        row_styles=["", "on grey11"],  # alternating row shading
        padding=(0, 1),
        expand=False,
    )

    # Add columns — no truncation, allow wrapping for very long values
    for h in headers:
        rtable.add_column(
            h,
            header_style="bold bright_white on blue",
            style="white",
            no_wrap=False,
            overflow="fold",
        )

    # Add data rows, converting every value to a string
    for row in rows:
        rtable.add_row(*(str(v) if v is not None else "NULL" for v in row))

    console.print()
    console.print(rtable)
    console.print()


def delete_rows(conn, table):
    """Delete all rows from a table after confirmation."""
    safe = sql.Identifier(table)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(safe))
        count = cur.fetchone()[0]

    if count == 0:
        console.print(f"\n  Table [bold]'{table}'[/bold] is already empty.", style="yellow")
        return

    console.print(f"\n  [bold red]⚠  This will delete ALL {count} row(s) from '{table}'.[/bold red]")
    confirm = input("  Type the table name to confirm: ").strip()

    if confirm != table:
        console.print("  Aborted — table name did not match.", style="yellow")
        return

    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {}").format(safe))
        conn.commit()
        console.print(f"  [green]✓ Deleted {count} row(s) from '{table}'.[/green]")
    except Exception as e:
        conn.rollback()
        console.print(f"  [red]✗ Delete failed: {e}[/red]")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print("  ╔══════════════════════════════════════╗", style="bold cyan")
    console.print("  ║       PostgreSQL Table Manager       ║", style="bold cyan")
    console.print("  ╚══════════════════════════════════════╝", style="bold cyan")
    console.print()

    # 1. Gather connection details
    print("  Enter connection details:\n")
    host = prompt("Host", "localhost")
    port = prompt("Port", "5432")
    user = prompt("User", "postgres")
    password = getpass.getpass("  Password: ")
    dbname = prompt("Database", "postgres")

    conn = connect(host, port, user, password, dbname)
    console.print(f"\n  [green]✓ Connected to '{dbname}' as '{user}' on {host}:{port}[/green]")

    # 2. Pick a table
    table = pick_table(conn)
    if table is None:
        conn.close()
        return

    print(f"\n  Selected table: {table}")

    # 3. Action menu
    while True:
        print("\n  ┌─────────────────────────────┐")
        print("  │  1. Show all rows            │")
        print("  │  2. Delete all rows           │")
        print("  │  3. Switch table              │")
        print("  │  4. Quit                      │")
        print("  └─────────────────────────────┘")

        action = input("\n  Choose an action: ").strip()

        if action == "1":
            show_rows(conn, table)
        elif action == "2":
            delete_rows(conn, table)
        elif action == "3":
            new_table = pick_table(conn)
            if new_table:
                table = new_table
                print(f"\n  Selected table: {table}")
        elif action == "4":
            break
        else:
            print("  Invalid choice.")

    conn.close()
    print("\n  Goodbye.\n")


if __name__ == "__main__":
    main()
