"""Schema state machine with idempotent migrations and backfill support."""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from typing import Any, Callable

from .types import (
    DimensionError,
    SQFoxError,
    SchemaError,
    SchemaState,
)

logger = logging.getLogger("sqfox.schema")


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

def detect_state(conn: sqlite3.Connection) -> SchemaState:
    """Detect the current schema state by inspecting existing tables.

    Checks for existence of sqfox-managed tables in sqlite_master.
    """
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    triggers = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }

    has_meta = "_sqfox_meta" in tables
    has_docs = "documents" in tables
    has_vec = "documents_vec" in tables
    has_fts = "documents_fts" in tables
    has_triggers = (
        "sqfox_fts_insert" in triggers
        and "sqfox_fts_delete" in triggers
        and "sqfox_fts_update" in triggers
    )

    if has_vec and has_fts and has_triggers:
        return SchemaState.ENRICHED
    if has_fts:
        return SchemaState.SEARCHABLE
    if has_vec:
        return SchemaState.INDEXED
    if has_meta and has_docs:
        return SchemaState.BASE
    return SchemaState.EMPTY


# ---------------------------------------------------------------------------
# Migration orchestration
# ---------------------------------------------------------------------------

def migrate_to(
    conn: sqlite3.Connection,
    target: SchemaState,
    *,
    vec_dimension: int | None = None,
) -> SchemaState:
    """Migrate schema from current state to target state.

    Migrations are idempotent — running them twice has no effect.
    Returns the achieved state.

    Note:
        INDEXED (vec0) and SEARCHABLE (FTS5) are independent capabilities.
        The numeric ordering in SchemaState is a convenience, not a strict
        dependency chain.  This function checks for the *existence* of
        individual tables rather than relying solely on numeric comparison,
        so that e.g. requesting INDEXED when FTS5 already exists (state ==
        SEARCHABLE) still creates the vec0 table correctly.

    Raises:
        SchemaError: If migration fails or target requires vec_dimension
                     but none is provided.
    """
    # Inspect actual table/trigger existence.  We do NOT rely on numeric
    # SchemaState comparison for the early-return because INDEXED and
    # SEARCHABLE are orthogonal capabilities:
    #   SEARCHABLE(3) > INDEXED(2), but that doesn't mean vec0 exists.
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    triggers = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }

    has_base = "_sqfox_meta" in tables and "documents" in tables
    has_vec = "documents_vec" in tables
    has_fts = "documents_fts" in tables
    has_triggers = (
        "sqfox_fts_insert" in triggers
        and "sqfox_fts_delete" in triggers
        and "sqfox_fts_update" in triggers
    )

    # --- BASE: core tables ---
    if not has_base and target >= SchemaState.BASE:
        _migrate_to_base(conn)

    # --- INDEXED: vec0 table ---
    needs_vec = target in (SchemaState.INDEXED, SchemaState.ENRICHED)

    if not has_vec and vec_dimension is not None:
        _migrate_to_indexed(conn, vec_dimension)
    elif needs_vec and not has_vec and vec_dimension is None:
        raise SchemaError(
            "vec_dimension is required to create the vector index"
        )

    # --- SEARCHABLE: FTS5 table ---
    if target >= SchemaState.SEARCHABLE and not has_fts:
        _migrate_to_searchable(conn)

    # --- ENRICHED: sync triggers (requires both vec0 and FTS5) ---
    if target >= SchemaState.ENRICHED and not has_triggers:
        _migrate_to_enriched(conn)

    return detect_state(conn)


# ---------------------------------------------------------------------------
# Individual migration steps
# ---------------------------------------------------------------------------

def _migrate_to_base(conn: sqlite3.Connection) -> None:
    """Create core tables: _sqfox_meta, documents."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _sqfox_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            content            TEXT    NOT NULL,
            content_lemmatized TEXT,
            metadata           TEXT    DEFAULT '{}',
            chunk_of           INTEGER REFERENCES documents(id),
            vec_indexed        INTEGER DEFAULT 0,
            fts_indexed        INTEGER DEFAULT 0,
            created_at         TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_vec_pending
            ON documents(vec_indexed) WHERE vec_indexed = 0
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_fts_pending
            ON documents(fts_indexed) WHERE fts_indexed = 0
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_chunk_of
            ON documents(chunk_of)
    """)
    conn.commit()
    logger.info("Migrated to BASE: created _sqfox_meta + documents tables")


def _migrate_to_indexed(conn: sqlite3.Connection, vec_dimension: int) -> None:
    """Create vec0 virtual table and store dimension in meta."""
    if not isinstance(vec_dimension, int) or vec_dimension <= 0:
        raise SchemaError(
            f"vec_dimension must be a positive integer, got {vec_dimension!r}"
        )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS documents_vec "
        f"USING vec0(embedding float[{vec_dimension}])"
    )
    conn.execute(
        "INSERT OR REPLACE INTO _sqfox_meta (key, value) VALUES (?, ?)",
        ("vec_dimension", json.dumps(vec_dimension)),
    )
    conn.commit()
    logger.info(
        "Migrated to INDEXED: created documents_vec with dimension=%d",
        vec_dimension,
    )


def _migrate_to_searchable(conn: sqlite3.Connection) -> None:
    """Create FTS5 virtual table for full-text search."""
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            content_lemmatized,
            content=documents,
            content_rowid=id,
            tokenize="unicode61"
        )
    """)
    conn.commit()
    logger.info("Migrated to SEARCHABLE: created documents_fts")


def _migrate_to_enriched(conn: sqlite3.Connection) -> None:
    """Create sync triggers for automatic FTS maintenance."""
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS sqfox_fts_insert
            AFTER INSERT ON documents
            WHEN new.content_lemmatized IS NOT NULL
        BEGIN
            INSERT INTO documents_fts(rowid, content_lemmatized)
                VALUES (new.id, new.content_lemmatized);
            UPDATE documents SET fts_indexed = 1 WHERE id = new.id;
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS sqfox_fts_delete
            AFTER DELETE ON documents
            WHEN old.content_lemmatized IS NOT NULL
        BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, content_lemmatized)
                VALUES ('delete', old.id, old.content_lemmatized);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS sqfox_fts_update
            AFTER UPDATE OF content_lemmatized ON documents
            WHEN new.content_lemmatized IS NOT NULL
        BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, content_lemmatized)
                SELECT 'delete', old.id, old.content_lemmatized
                WHERE old.content_lemmatized IS NOT NULL;
            INSERT INTO documents_fts(rowid, content_lemmatized)
                VALUES (new.id, new.content_lemmatized);
            UPDATE documents SET fts_indexed = 1 WHERE id = new.id;
        END
    """)
    conn.commit()
    logger.info("Migrated to ENRICHED: created FTS sync triggers")


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------

def get_stored_dimension(conn: sqlite3.Connection) -> int | None:
    """Get the stored vector dimension, or None if not set."""
    try:
        row = conn.execute(
            "SELECT value FROM _sqfox_meta WHERE key = ?", ("vec_dimension",)
        ).fetchone()
    except sqlite3.OperationalError:
        return None

    if row is None:
        return None
    return json.loads(row[0])


def validate_dimension(
    conn: sqlite3.Connection,
    dimension: int,
    *,
    commit: bool = True,
) -> None:
    """Check that the given dimension matches the stored dimension.

    On first call (no stored dimension), stores the dimension.
    On subsequent calls, raises DimensionError if mismatched.

    Args:
        commit: If False, do not commit (caller manages the transaction).
    """
    stored = get_stored_dimension(conn)

    if stored is None:
        # Ensure meta table exists
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _sqfox_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO _sqfox_meta (key, value) VALUES (?, ?)",
            ("vec_dimension", json.dumps(dimension)),
        )
        if commit:
            conn.commit()
        return

    if stored != dimension:
        raise DimensionError(expected=stored, got=dimension)


# ---------------------------------------------------------------------------
# Backfill functions
# ---------------------------------------------------------------------------

def backfill_vectors(
    conn: sqlite3.Connection,
    embed_fn: Callable[[list[str]], list[list[float]]],
    batch_size: int = 100,
) -> int:
    """Backfill vector embeddings for un-indexed documents.

    Returns the number of documents processed.
    Designed to be called incrementally (resumable).
    """
    processed = 0
    dimension_validated = False

    while True:
        rows = conn.execute(
            "SELECT id, content FROM documents "
            "WHERE vec_indexed = 0 LIMIT ?",
            (batch_size,),
        ).fetchall()

        if not rows:
            break

        texts = [row[1] for row in rows]
        embeddings = embed_fn(texts)

        if len(embeddings) != len(rows):
            raise SQFoxError(
                f"embed_fn returned {len(embeddings)} embeddings for "
                f"{len(rows)} texts — must return exactly one embedding per text"
            )

        # Validate dimension once, before any batch transaction.
        # commit=True ensures the dimension write completes its own
        # transaction so the subsequent BEGIN doesn't conflict.
        if not dimension_validated and embeddings:
            validate_dimension(conn, len(embeddings[0]), commit=True)
            dimension_validated = True

        conn.execute("BEGIN")
        for row, embedding in zip(rows, embeddings):
            vec_blob = struct.pack(f"{len(embedding)}f", *embedding)
            conn.execute(
                "INSERT OR REPLACE INTO documents_vec(rowid, embedding) "
                "VALUES (?, ?)",
                (row[0], vec_blob),
            )
            conn.execute(
                "UPDATE documents SET vec_indexed = 1 WHERE id = ?",
                (row[0],),
            )
        conn.commit()
        processed += len(rows)

    return processed


def backfill_fts(
    conn: sqlite3.Connection,
    lemmatize_fn: Callable[[str], str],
    batch_size: int = 100,
) -> int:
    """Backfill FTS index for un-indexed documents.

    Lemmatizes content and updates content_lemmatized column.
    If ENRICHED triggers are active, FTS is updated automatically.
    Otherwise, manually inserts into FTS table.

    Returns the number of documents processed.
    """
    state = detect_state(conn)
    has_triggers = state >= SchemaState.ENRICHED
    processed = 0

    while True:
        rows = conn.execute(
            "SELECT id, content FROM documents "
            "WHERE fts_indexed = 0 LIMIT ?",
            (batch_size,),
        ).fetchall()

        if not rows:
            break

        conn.execute("BEGIN")
        try:
            for row in rows:
                if row[1] is None:
                    # Mark as indexed so we don't re-process on next batch
                    conn.execute(
                        "UPDATE documents SET fts_indexed = 1 WHERE id = ?",
                        (row[0],),
                    )
                    continue
                lemmatized = lemmatize_fn(row[1])
                conn.execute(
                    "UPDATE documents SET content_lemmatized = ? WHERE id = ?",
                    (lemmatized, row[0]),
                )

                if not has_triggers:
                    # Manually insert into FTS
                    conn.execute(
                        "INSERT INTO documents_fts(rowid, content_lemmatized) "
                        "VALUES (?, ?)",
                        (row[0], lemmatized),
                    )
                    conn.execute(
                        "UPDATE documents SET fts_indexed = 1 WHERE id = ?",
                        (row[0],),
                    )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        processed += len(rows)

    return processed
