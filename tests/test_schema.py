"""Tests for sqfox schema: state detection, migrations, dimension validation, backfill."""

import sqlite3

import pytest

from sqfox.schema import (
    detect_state,
    migrate_to,
    validate_dimension,
    get_stored_dimension,
    backfill_vectors,
    backfill_fts,
)
from sqfox.types import DimensionError, SchemaError, SchemaState


@pytest.fixture
def conn(tmp_path):
    """Provide a fresh SQLite connection."""
    db_path = str(tmp_path / "test.db")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

class TestDetectState:
    def test_detect_empty(self, conn):
        assert detect_state(conn) == SchemaState.EMPTY

    def test_detect_base(self, conn):
        migrate_to(conn, SchemaState.BASE)
        assert detect_state(conn) == SchemaState.BASE

    def test_detect_indexed_stores_dimension(self, conn):
        migrate_to(conn, SchemaState.INDEXED, vec_dimension=4)
        # Without vec0 table, state is BASE (dimension stored in meta only)
        assert detect_state(conn) >= SchemaState.BASE
        dim = conn.execute(
            "SELECT value FROM _sqfox_meta WHERE key = 'vec_dimension'"
        ).fetchone()
        assert dim is not None

    def test_detect_searchable(self, conn):
        migrate_to(conn, SchemaState.SEARCHABLE)
        # Without vec0, FTS only = SEARCHABLE
        state = detect_state(conn)
        assert state == SchemaState.SEARCHABLE

    def test_detect_enriched(self, conn):
        migrate_to(conn, SchemaState.ENRICHED, vec_dimension=4)
        assert detect_state(conn) == SchemaState.ENRICHED


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

class TestMigrations:
    def test_migrate_empty_to_base(self, conn):
        result = migrate_to(conn, SchemaState.BASE)
        assert result >= SchemaState.BASE

        # Tables should exist
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "_sqfox_meta" in tables
        assert "documents" in tables

    def test_migrate_base_to_indexed(self, conn):
        migrate_to(conn, SchemaState.INDEXED, vec_dimension=384)

        # Check dimension stored in meta
        dim = get_stored_dimension(conn)
        assert dim == 384

    def test_migrate_to_searchable(self, conn):
        migrate_to(conn, SchemaState.SEARCHABLE)

        # FTS table should exist
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "documents_fts" in tables

    def test_migrate_idempotent(self, conn):
        migrate_to(conn, SchemaState.BASE)
        migrate_to(conn, SchemaState.BASE)  # Should not raise

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "documents" in tables

    def test_migrate_requires_dimension_for_indexed(self, conn):
        with pytest.raises(SchemaError, match="vec_dimension"):
            migrate_to(conn, SchemaState.INDEXED)

    def test_migrate_skip_if_already_at_target(self, conn):
        migrate_to(conn, SchemaState.BASE)
        # Calling migrate_to with lower target returns current
        result = migrate_to(conn, SchemaState.EMPTY)
        assert result == SchemaState.BASE

    def test_migrate_searchable_then_indexed(self, conn):
        """Dimension can be stored after SEARCHABLE (FTS5).

        This verifies that the orthogonal capabilities don't block each
        other despite SEARCHABLE(3) > INDEXED(2) numerically.
        """
        # First create SEARCHABLE (FTS5 only)
        migrate_to(conn, SchemaState.SEARCHABLE)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        assert "documents_fts" in tables

        # Now request INDEXED — dimension should be stored
        migrate_to(conn, SchemaState.INDEXED, vec_dimension=128)
        dim = get_stored_dimension(conn)
        assert dim == 128
        # FTS still exists
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        assert "documents_fts" in tables

    def test_migrate_indexed_then_enriched(self, conn):
        """ENRICHED can be reached from INDEXED (adds FTS + triggers)."""
        migrate_to(conn, SchemaState.INDEXED, vec_dimension=64)
        result = migrate_to(conn, SchemaState.ENRICHED, vec_dimension=64)
        assert result == SchemaState.ENRICHED

    def test_migrate_searchable_then_enriched(self, conn):
        """ENRICHED can be reached from SEARCHABLE."""
        migrate_to(conn, SchemaState.SEARCHABLE)
        result = migrate_to(conn, SchemaState.ENRICHED, vec_dimension=64)
        assert result == SchemaState.ENRICHED

    def test_migrate_empty_to_enriched(self, conn):
        """Direct jump from EMPTY to ENRICHED."""
        result = migrate_to(conn, SchemaState.ENRICHED, vec_dimension=64)
        assert result == SchemaState.ENRICHED

    def test_migrate_invalid_vec_dimension(self, conn):
        """Invalid vec_dimension raises SchemaError."""
        with pytest.raises(SchemaError, match="positive integer"):
            migrate_to(conn, SchemaState.INDEXED, vec_dimension=-1)
        with pytest.raises(SchemaError, match="positive integer"):
            migrate_to(conn, SchemaState.INDEXED, vec_dimension=0)


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------

class TestDimensionValidation:
    def test_dimension_first_call(self, conn):
        migrate_to(conn, SchemaState.BASE)
        validate_dimension(conn, 384)
        assert get_stored_dimension(conn) == 384

    def test_dimension_match(self, conn):
        migrate_to(conn, SchemaState.BASE)
        validate_dimension(conn, 384)
        validate_dimension(conn, 384)  # Should not raise

    def test_dimension_mismatch(self, conn):
        migrate_to(conn, SchemaState.BASE)
        validate_dimension(conn, 384)
        with pytest.raises(DimensionError) as exc_info:
            validate_dimension(conn, 768)
        assert exc_info.value.expected == 384
        assert exc_info.value.got == 768


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

class TestBackfill:
    def test_backfill_vectors(self, conn):
        migrate_to(conn, SchemaState.INDEXED, vec_dimension=4)

        # Insert some documents
        for i in range(5):
            conn.execute(
                "INSERT INTO documents (content) VALUES (?)",
                (f"Document {i}",),
            )
        conn.commit()

        # Mock embed function
        def mock_embed(texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        processed = backfill_vectors(conn, mock_embed, batch_size=2)
        assert processed == 5

        # Check all marked as indexed
        row = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE vec_indexed = 0"
        ).fetchone()
        assert row[0] == 0

    def test_backfill_fts(self, conn):
        migrate_to(conn, SchemaState.SEARCHABLE)

        # Insert some documents
        for i in range(3):
            conn.execute(
                "INSERT INTO documents (content) VALUES (?)",
                (f"Document number {i} about testing",),
            )
        conn.commit()

        # Mock lemmatize function
        def mock_lemmatize(text):
            return text.lower()

        processed = backfill_fts(conn, mock_lemmatize, batch_size=2)
        assert processed == 3

        # Check content_lemmatized populated
        rows = conn.execute(
            "SELECT content_lemmatized FROM documents WHERE content_lemmatized IS NOT NULL"
        ).fetchall()
        assert len(rows) == 3

    def test_backfill_resumable(self, conn):
        migrate_to(conn, SchemaState.INDEXED, vec_dimension=4)

        # Insert documents
        for i in range(5):
            conn.execute(
                "INSERT INTO documents (content) VALUES (?)",
                (f"Doc {i}",),
            )
        conn.commit()

        # Embed only first 2
        call_count = [0]
        def mock_embed(texts):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("Simulated failure")
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        try:
            backfill_vectors(conn, mock_embed, batch_size=2)
        except RuntimeError:
            pass

        # Should have 2 indexed
        row = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE vec_indexed = 1"
        ).fetchone()
        assert row[0] == 2

        # Resume: now embed remaining
        def mock_embed_ok(texts):
            return [[0.0, 1.0, 0.0, 0.0] for _ in texts]

        processed = backfill_vectors(conn, mock_embed_ok, batch_size=10)
        assert processed == 3

        # All should be indexed now
        row = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE vec_indexed = 0"
        ).fetchone()
        assert row[0] == 0

    def test_backfill_vectors_empty(self, conn):
        """backfill_vectors on empty table returns 0."""
        migrate_to(conn, SchemaState.INDEXED, vec_dimension=4)
        result = backfill_vectors(conn, lambda texts: [[0.0]*4]*len(texts))
        assert result == 0

    def test_backfill_fts_empty(self, conn):
        """backfill_fts on empty table returns 0."""
        migrate_to(conn, SchemaState.SEARCHABLE)
        result = backfill_fts(conn, lambda text: text.lower())
        assert result == 0
