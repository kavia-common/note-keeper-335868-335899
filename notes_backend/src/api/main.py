"""FastAPI backend for the Notes app.

This service exposes a REST API for:
- Notes CRUD
- Search (full-text via SQLite FTS5)
- Tags listing

It is designed to work with the SQLite schema created by the notes_database container
(init_db.py), which provides:
- notes, tags, note_tags tables
- notes_fts FTS5 index and triggers

Environment variables (must be provided via container .env):
- SQLITE_DB: Path to the SQLite database file (e.g., /path/to/myapp.db)

CORS:
- By default, allows all origins (safe for dev). You may restrict by setting:
  - FRONTEND_ORIGINS: comma-separated list of allowed origins (e.g., http://localhost:3000,https://example.com)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("notes_backend")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _parse_origins(value: Optional[str]) -> List[str]:
    """Parse comma-separated origins string into list."""
    if not value:
        return ["*"]
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def _get_sqlite_db_path() -> str:
    """Return SQLite DB path from env.

    Raises:
        RuntimeError: if SQLITE_DB is not configured.
    """
    db_path = os.getenv("SQLITE_DB")
    if not db_path:
        raise RuntimeError(
            "SQLITE_DB environment variable is required to run notes_backend."
        )
    return db_path


@contextmanager
def _db_conn() -> sqlite3.Connection:
    """Context manager for SQLite connections.

    Contract:
      - Opens a sqlite3 connection to SQLITE_DB with Row factory enabled.
      - Enables foreign keys.
      - Commits on success; rolls back on error.
      - Always closes the connection.

    Errors:
      - Raises sqlite3.Error with original context (caller maps to HTTP errors).
    """
    db_path = _get_sqlite_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _rows_to_dicts(rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    """Convert sqlite3.Row list to list[dict]."""
    return [dict(r) for r in rows]


def _fetch_note_tags(conn: sqlite3.Connection, note_id: int) -> List[str]:
    """Return tag names for a note."""
    cur = conn.execute(
        """
        SELECT t.name
        FROM tags t
        JOIN note_tags nt ON nt.tag_id = t.id
        WHERE nt.note_id = ?
        ORDER BY t.name ASC
        """,
        (note_id,),
    )
    return [r["name"] for r in cur.fetchall()]


def _ensure_tag_ids(conn: sqlite3.Connection, tag_names: Sequence[str]) -> List[int]:
    """Ensure each tag exists and return their IDs.

    Invariant:
      - Tag names are treated case-sensitively, matching the DB UNIQUE constraint.
    """
    tag_ids: List[int] = []
    for name in tag_names:
        cleaned = name.strip()
        if not cleaned:
            continue
        conn.execute("INSERT OR IGNORE INTO tags(name) VALUES(?)", (cleaned,))
        row = conn.execute("SELECT id FROM tags WHERE name = ?", (cleaned,)).fetchone()
        if row is not None:
            tag_ids.append(int(row["id"]))
    # unique ids in stable order
    seen: set[int] = set()
    out: List[int] = []
    for tid in tag_ids:
        if tid not in seen:
            out.append(tid)
            seen.add(tid)
    return out


def _set_note_tags(conn: sqlite3.Connection, note_id: int, tag_names: Sequence[str]) -> None:
    """Replace tags for a given note with the provided set."""
    tag_ids = _ensure_tag_ids(conn, tag_names)
    conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
    for tag_id in tag_ids:
        conn.execute(
            "INSERT OR IGNORE INTO note_tags(note_id, tag_id) VALUES(?, ?)",
            (note_id, tag_id),
        )


def _note_row_to_model(conn: sqlite3.Connection, note_row: sqlite3.Row) -> Dict[str, Any]:
    """Build API note dict from DB row (including tags)."""
    note_id = int(note_row["id"])
    return {
        "id": note_id,
        "title": note_row["title"],
        "content": note_row["content"],
        "created_at": note_row.get("created_at") if isinstance(note_row, dict) else note_row["created_at"],
        "updated_at": note_row.get("updated_at") if isinstance(note_row, dict) else note_row["updated_at"],
        "tags": _fetch_note_tags(conn, note_id),
    }


class NoteCreateUpdateIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Note title")
    content: str = Field(..., min_length=0, description="Note content/body text")
    tags: Optional[List[str]] = Field(
        default=None, description="Optional list of tag names (strings)"
    )


class NoteOut(BaseModel):
    id: int = Field(..., description="Note ID")
    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Note content/body text")
    tags: List[str] = Field(default_factory=list, description="Tag names")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")


openapi_tags = [
    {"name": "Health", "description": "Service health checks."},
    {"name": "Notes", "description": "Notes CRUD, listing, filtering, and search."},
    {"name": "Tags", "description": "Tag listing."},
]

app = FastAPI(
    title="Notes Backend API",
    description="FastAPI backend for a fullstack Notes app (SQLite persistence, tags, and full-text search).",
    version="1.0.0",
    openapi_tags=openapi_tags,
)

frontend_origins = _parse_origins(os.getenv("FRONTEND_ORIGINS"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"], summary="Health check", operation_id="health_check")
def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
      {"message": "Healthy"} on success.
    """
    return {"message": "Healthy"}


@app.get(
    "/notes",
    tags=["Notes"],
    summary="List notes (optionally search/filter)",
    operation_id="list_notes",
    response_model=List[NoteOut],
)
def list_notes(
    q: Optional[str] = Query(
        default=None,
        description="Optional full-text query over title/content (uses FTS5).",
    ),
    tag: Optional[str] = Query(
        default=None,
        description="Optional tag filter (exact match on tag name).",
    ),
    limit: int = Query(default=100, ge=1, le=500, description="Max notes to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> List[Dict[str, Any]]:
    """List notes with optional full-text search and/or tag filtering.

    Contract:
      Inputs:
        - q: optional text to search
        - tag: optional tag name to filter
        - limit/offset: pagination
      Output:
        - List of notes sorted by updated_at DESC (and id DESC as tie-breaker)
      Errors:
        - 500 if database is unavailable or schema missing
    """
    logger.info("ListNotesFlow start q=%r tag=%r limit=%s offset=%s", q, tag, limit, offset)
    try:
        with _db_conn() as conn:
            params: List[Any] = []
            where_clauses: List[str] = []

            # Tag filter via join
            tag_join = ""
            if tag:
                tag_join = """
                JOIN note_tags nt ON nt.note_id = n.id
                JOIN tags t ON t.id = nt.tag_id
                """
                where_clauses.append("t.name = ?")
                params.append(tag)

            # Search via FTS - uses notes_fts rowid mapping to notes.id
            search_join = ""
            if q:
                search_join = """
                JOIN notes_fts f ON f.rowid = n.id
                """
                where_clauses.append("notes_fts MATCH ?")
                params.append(q)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            sql = f"""
            SELECT DISTINCT n.id, n.title, n.content, n.created_at, n.updated_at
            FROM notes n
            {tag_join}
            {search_join}
            {where_sql}
            ORDER BY n.updated_at DESC, n.id DESC
            LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
            result = [_note_row_to_model(conn, r) for r in rows]
            logger.info("ListNotesFlow end count=%s", len(result))
            return result
    except RuntimeError as e:
        logger.exception("ListNotesFlow config error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except sqlite3.Error as e:
        logger.exception("ListNotesFlow db error")
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.post(
    "/notes",
    tags=["Notes"],
    summary="Create a note",
    operation_id="create_note",
    response_model=NoteOut,
    status_code=status.HTTP_201_CREATED,
)
def create_note(payload: NoteCreateUpdateIn) -> Dict[str, Any]:
    """Create a new note.

    Contract:
      Inputs:
        - title/content required, tags optional
      Output:
        - The created note, including assigned id and tag list
      Errors:
        - 422 validation errors via Pydantic/FastAPI
        - 500 DB errors
    """
    logger.info("CreateNoteFlow start title_len=%s content_len=%s", len(payload.title), len(payload.content))
    try:
        with _db_conn() as conn:
            cur = conn.execute(
                "INSERT INTO notes(title, content) VALUES(?, ?)",
                (payload.title, payload.content),
            )
            note_id = int(cur.lastrowid)
            _set_note_tags(conn, note_id, payload.tags or [])
            row = conn.execute(
                "SELECT id, title, content, created_at, updated_at FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=500, detail="Failed to read created note")
            result = _note_row_to_model(conn, row)
            logger.info("CreateNoteFlow end note_id=%s", note_id)
            return result
    except sqlite3.Error as e:
        logger.exception("CreateNoteFlow db error")
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.put(
    "/notes/{note_id}",
    tags=["Notes"],
    summary="Update a note",
    operation_id="update_note",
    response_model=NoteOut,
)
def update_note(note_id: int, payload: NoteCreateUpdateIn) -> Dict[str, Any]:
    """Update an existing note (replaces title/content and tags).

    Contract:
      Inputs:
        - note_id: existing note id
        - payload: title/content required; tags optional (missing => clear)
      Output:
        - The updated note
      Errors:
        - 404 if note does not exist
        - 500 DB errors
    """
    logger.info("UpdateNoteFlow start note_id=%s", note_id)
    try:
        with _db_conn() as conn:
            existing = conn.execute("SELECT id FROM notes WHERE id = ?", (note_id,)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Note not found")

            conn.execute(
                "UPDATE notes SET title = ?, content = ? WHERE id = ?",
                (payload.title, payload.content, note_id),
            )
            _set_note_tags(conn, note_id, payload.tags or [])
            row = conn.execute(
                "SELECT id, title, content, created_at, updated_at FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Note not found")
            result = _note_row_to_model(conn, row)
            logger.info("UpdateNoteFlow end note_id=%s", note_id)
            return result
    except sqlite3.Error as e:
        logger.exception("UpdateNoteFlow db error")
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.delete(
    "/notes/{note_id}",
    tags=["Notes"],
    summary="Delete a note",
    operation_id="delete_note",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_note(note_id: int) -> Response:
    """Delete a note.

    Contract:
      Inputs:
        - note_id: existing note id
      Output:
        - 204 No Content on success
      Errors:
        - 404 if note does not exist
        - 500 DB errors
    """
    logger.info("DeleteNoteFlow start note_id=%s", note_id)
    try:
        with _db_conn() as conn:
            cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Note not found")
            logger.info("DeleteNoteFlow end note_id=%s", note_id)
            return Response(status_code=status.HTTP_204_NO_CONTENT)
    except sqlite3.Error as e:
        logger.exception("DeleteNoteFlow db error")
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.get(
    "/tags",
    tags=["Tags"],
    summary="List all tags",
    operation_id="list_tags",
    response_model=List[str],
)
def list_tags() -> List[str]:
    """List all tags.

    Output:
      - Sorted list of tag names (ascending).
    """
    logger.info("ListTagsFlow start")
    try:
        with _db_conn() as conn:
            cur = conn.execute("SELECT name FROM tags ORDER BY name ASC")
            tags = [r["name"] for r in cur.fetchall()]
            logger.info("ListTagsFlow end count=%s", len(tags))
            return tags
    except sqlite3.Error as e:
        logger.exception("ListTagsFlow db error")
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
