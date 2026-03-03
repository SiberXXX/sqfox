"""Tests for pluggable vector backends."""

import os
import sqlite3
import struct

import pytest

from sqfox import SQFox, VectorBackend, VectorBackendError, SqliteHnswBackend
from sqfox.backends import SqliteVecBackend, get_backend
from sqfox.backends.registry import get_backend as registry_get_backend

from conftest import requires_sqlite_vec, requires_usearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NDIM = 8


def _embed(texts: list[str]) -> list[list[float]]:
    """Deterministic dummy embedder: each text → vector of its length."""
    return [[float(len(t))] * NDIM for t in texts]


def _embed_single(text: str) -> list[float]:
    return _embed([text])[0]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_none_returns_none(self):
        assert get_backend(None) is None

    @requires_sqlite_vec
    def test_sqlite_vec_alias(self):
        b = get_backend("sqlite-vec")
        assert isinstance(b, SqliteVecBackend)

    @requires_sqlite_vec
    def test_sqlite_vec_underscore_alias(self):
        b = get_backend("sqlite_vec")
        assert isinstance(b, SqliteVecBackend)

    @requires_usearch
    def test_usearch_alias(self):
        b = get_backend("usearch")
        assert isinstance(b, VectorBackend)

    def test_hnsw_alias(self):
        b = get_backend("hnsw")
        assert isinstance(b, SqliteHnswBackend)

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_backend("nonexistent-backend")

    def test_passthrough_instance(self):
        b = SqliteVecBackend()
        assert get_backend(b) is b


# ---------------------------------------------------------------------------
# SqliteVecBackend tests
# ---------------------------------------------------------------------------

@requires_sqlite_vec
class TestSqliteVecBackend:
    @pytest.fixture
    def backend_and_conn(self, tmp_path):
        """Create a SqliteVecBackend with a real sqlite-vec DB."""
        import sqlite_vec

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            f"CREATE VIRTUAL TABLE documents_vec "
            f"USING vec0(embedding float[{NDIM}])"
        )
        conn.commit()

        b = SqliteVecBackend()
        b.initialize(db_path, NDIM)
        b.set_writer_conn(conn)
        yield b, conn
        b.close()
        conn.close()

    def test_add_and_search(self, backend_and_conn):
        b, conn = backend_and_conn
        vecs = _embed(["hello", "world"])
        b.add([1, 2], vecs)
        conn.commit()

        results = b.search(_embed_single("hello"), 5, conn=conn)
        assert len(results) >= 1
        keys = [k for k, _ in results]
        assert 1 in keys or 2 in keys

    def test_remove(self, backend_and_conn):
        b, conn = backend_and_conn
        b.add([10], [_embed_single("test")])
        conn.commit()

        b.remove([10])
        conn.commit()

        results = b.search(_embed_single("test"), 5, conn=conn)
        keys = [k for k, _ in results]
        assert 10 not in keys

    def test_count(self, backend_and_conn):
        b, conn = backend_and_conn
        assert b.count() == 0
        b.add([1, 2, 3], _embed(["a", "bb", "ccc"]))
        conn.commit()
        assert b.count() == 3

    def test_search_requires_conn(self, backend_and_conn):
        b, _ = backend_and_conn
        with pytest.raises(ValueError, match="conn"):
            b.search(_embed_single("x"), 5)

    def test_flush_is_noop(self, backend_and_conn):
        b, _ = backend_and_conn
        b.flush()  # Should not raise


# ---------------------------------------------------------------------------
# USearchBackend tests
# ---------------------------------------------------------------------------

@requires_usearch
class TestUSearchBackend:
    @pytest.fixture
    def backend(self, tmp_path):
        from sqfox.backends.usearch import USearchBackend
        b = USearchBackend(metric="cos", dtype="f32")
        b.initialize(str(tmp_path / "test.db"), NDIM)
        yield b
        b.close()

    @pytest.fixture
    def memory_backend(self):
        from sqfox.backends.usearch import USearchBackend
        b = USearchBackend(metric="cos", dtype="f32")
        b.initialize(":memory:", NDIM)
        yield b
        b.close()

    def test_add_and_search(self, backend):
        import numpy as np
        # Use orthogonal-ish vectors for clearer results
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        backend.add([1, 2], [v1, v2])
        backend.flush()

        results = backend.search(v1, 2)
        assert len(results) >= 1
        # First result should be key=1 (closest to v1)
        assert results[0][0] == 1

    def test_remove(self, backend):
        v1 = [1.0] * NDIM
        backend.add([10], [v1])
        backend.flush()
        assert backend.count() == 1

        backend.remove([10])
        backend.flush()
        # USearch marks removed but count may not drop immediately
        # Search should not return the removed key
        results = backend.search(v1, 5)
        keys = [k for k, _ in results]
        assert 10 not in keys

    def test_count(self, backend):
        assert backend.count() == 0
        backend.add([1, 2, 3], [[float(i)] * NDIM for i in range(3)])
        assert backend.count() == 3

    def test_flush_creates_file(self, backend, tmp_path):
        backend.add([1], [[1.0] * NDIM])
        backend.flush()
        index_path = str(tmp_path / "test.db") + ".usearch"
        assert os.path.exists(index_path)

    def test_memory_mode_no_file(self, memory_backend):
        memory_backend.add([1], [[1.0] * NDIM])
        memory_backend.flush()
        # No file should be created for :memory: mode
        assert memory_backend._index_path is None

    def test_memory_search(self, memory_backend):
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        memory_backend.add([1, 2], [v1, v2])

        results = memory_backend.search(v1, 2)
        assert len(results) >= 1
        assert results[0][0] == 1

    def test_persistence(self, tmp_path):
        from sqfox.backends.usearch import USearchBackend

        db_path = str(tmp_path / "persist.db")
        v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        # Write
        b1 = USearchBackend()
        b1.initialize(db_path, NDIM)
        b1.add([42], [v])
        b1.flush()
        b1.close()

        # Read back
        b2 = USearchBackend()
        b2.initialize(db_path, NDIM)
        assert b2.count() == 1
        results = b2.search(v, 1)
        assert results[0][0] == 42
        b2.close()

    def test_verify_consistency_match(self, backend):
        backend.add([1, 2], [[1.0] * NDIM, [2.0] * NDIM])
        assert backend.verify_consistency(2) is True

    def test_verify_consistency_mismatch(self, backend):
        backend.add([1], [[1.0] * NDIM])
        assert backend.verify_consistency(5) is False

    def test_rebuild_from_blobs(self, tmp_path):
        from sqfox.backends.usearch import USearchBackend

        db_path = str(tmp_path / "rebuild.db")
        b = USearchBackend()
        b.initialize(db_path, NDIM)

        # Simulate blobs as stored in SQLite
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        blob1 = struct.pack(f"{NDIM}f", *v1)
        blob2 = struct.pack(f"{NDIM}f", *v2)

        b.rebuild_from_blobs([(1, blob1), (2, blob2)], NDIM)
        assert b.count() == 2

        results = b.search(v1, 2)
        assert results[0][0] == 1
        b.close()

    def test_empty_search(self, backend):
        results = backend.search([1.0] * NDIM, 5)
        assert results == []


# ---------------------------------------------------------------------------
# SqliteHnswBackend tests (pure Python, no external deps)
# ---------------------------------------------------------------------------

class TestSqliteHnswBackend:
    """Unit tests for pure-Python HNSW backend."""

    @pytest.fixture
    def backend_with_db(self, tmp_path):
        """SqliteHnswBackend wired to a real SQLite DB with documents table."""
        db_path = str(tmp_path / "hnsw.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  content TEXT,"
            "  embedding BLOB,"
            "  vec_indexed INTEGER DEFAULT 0"
            ")"
        )
        conn.commit()
        b = SqliteHnswBackend(M=4, ef_construction=32, ef_search=16)
        b.set_writer_conn(conn)
        b.initialize(db_path, NDIM)
        yield b, conn
        b.close()
        conn.close()

    def _store_vectors(self, conn, keys, vectors):
        """Helper: insert vectors into documents.embedding."""
        for key, vec in zip(keys, vectors):
            blob = struct.pack(f"{NDIM}f", *vec)
            conn.execute(
                "INSERT INTO documents (id, content, embedding, vec_indexed) "
                "VALUES (?, 'test', ?, 1)",
                (key, blob),
            )
        conn.commit()

    def test_add_and_search(self, backend_with_db):
        b, conn = backend_with_db
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._store_vectors(conn, [1, 2, 3], [v1, v2, v3])
        b.add([1, 2, 3], [v1, v2, v3])

        results = b.search(v1, 3, conn=conn)
        assert len(results) >= 1
        # Closest to v1 should be key=1
        assert results[0][0] == 1

    def test_search_empty_index(self, backend_with_db):
        b, conn = backend_with_db
        results = b.search([1.0] * NDIM, 5, conn=conn)
        assert results == []

    def test_search_requires_conn(self, backend_with_db):
        b, _ = backend_with_db
        with pytest.raises(ValueError, match="conn"):
            b.search([1.0] * NDIM, 5)

    def test_count(self, backend_with_db):
        b, conn = backend_with_db
        assert b.count() == 0
        vecs = [[float(i)] * NDIM for i in range(5)]
        self._store_vectors(conn, [1, 2, 3, 4, 5], vecs)
        b.add([1, 2, 3, 4, 5], vecs)
        assert b.count() == 5

    def test_remove(self, backend_with_db):
        b, conn = backend_with_db
        v = [1.0] * NDIM
        self._store_vectors(conn, [10], [v])
        b.add([10], [v])
        assert b.count() == 1

        b.remove([10])
        assert b.count() == 0
        results = b.search(v, 5, conn=conn)
        keys = [k for k, _ in results]
        assert 10 not in keys

    def test_flush_and_persistence(self, tmp_path):
        """Graph survives flush → close → reopen cycle."""
        db_path = str(tmp_path / "persist.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY, content TEXT, "
            "  embedding BLOB, vec_indexed INTEGER DEFAULT 0)"
        )
        conn.commit()

        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        blob1 = struct.pack(f"{NDIM}f", *v1)
        blob2 = struct.pack(f"{NDIM}f", *v2)
        conn.execute(
            "INSERT INTO documents VALUES (1, 'a', ?, 1)", (blob1,)
        )
        conn.execute(
            "INSERT INTO documents VALUES (2, 'b', ?, 1)", (blob2,)
        )
        conn.commit()

        # Write graph
        b1 = SqliteHnswBackend(M=4, ef_construction=16, ef_search=8)
        b1.set_writer_conn(conn)
        b1.initialize(db_path, NDIM)
        b1.add([1, 2], [v1, v2])
        b1.flush()
        b1.close()

        # Read back with fresh backend
        b2 = SqliteHnswBackend(M=4, ef_construction=16, ef_search=8)
        b2.set_writer_conn(conn)
        b2.initialize(db_path, NDIM)
        assert b2.count() == 2
        results = b2.search(v1, 2, conn=conn)
        assert len(results) >= 1
        assert results[0][0] == 1
        b2.close()
        conn.close()

    def test_verify_consistency(self, backend_with_db):
        b, conn = backend_with_db
        vecs = [[float(i)] * NDIM for i in range(3)]
        self._store_vectors(conn, [1, 2, 3], vecs)
        b.add([1, 2, 3], vecs)
        assert b.verify_consistency(3) is True
        assert b.verify_consistency(10) is False

    def test_rebuild_from_blobs(self, backend_with_db):
        b, conn = backend_with_db
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        blob1 = struct.pack(f"{NDIM}f", *v1)
        blob2 = struct.pack(f"{NDIM}f", *v2)

        b.rebuild_from_blobs([(1, blob1), (2, blob2)], NDIM)
        assert b.count() == 2

    def test_many_vectors(self, backend_with_db):
        """Insert 100 vectors and verify nearest-neighbor quality."""
        b, conn = backend_with_db
        import random as _rng
        _rng.seed(42)
        vecs = [[_rng.gauss(0, 1) for _ in range(NDIM)] for _ in range(100)]
        keys = list(range(1, 101))
        self._store_vectors(conn, keys, vecs)
        b.add(keys, vecs)
        b.flush()

        # Search for vec[0], it should return key=1 as closest
        results = b.search(vecs[0], 5, conn=conn)
        assert len(results) >= 1
        assert results[0][0] == 1  # exact match


# ---------------------------------------------------------------------------
# Engine integration with vector_backend
# ---------------------------------------------------------------------------

@requires_sqlite_vec
class TestEngineDefaultBackend:
    """Ensure default behavior (sqlite-vec) is unchanged."""

    def test_default_no_backend(self, tmp_path):
        db_path = str(tmp_path / "default.db")
        with SQFox(db_path) as db:
            db.ingest("hello world", embed_fn=_embed, wait=True)
            results = db.search("hello", embed_fn=_embed)
            assert len(results) >= 1
            # Default uses legacy sqlite-vec path (no explicit backend object)
            # but vector_backend_name reports "sqlite-vec" when extension is available
            assert db.vector_backend_name == "sqlite-vec"

    def test_explicit_sqlite_vec(self, tmp_path):
        db_path = str(tmp_path / "explicit.db")
        with SQFox(db_path, vector_backend="sqlite-vec") as db:
            db.ingest("hello world", embed_fn=_embed, wait=True)
            results = db.search("hello", embed_fn=_embed)
            assert len(results) >= 1
            assert db.vector_backend_name == "sqlite-vec"


@requires_usearch
class TestEngineUSearch:
    """Engine integration with USearch backend."""

    def test_usearch_ingest_and_search(self, tmp_path):
        db_path = str(tmp_path / "usearch.db")
        with SQFox(db_path, vector_backend="usearch") as db:
            db.ingest("hello world", embed_fn=_embed, wait=True)
            db.ingest("goodbye world", embed_fn=_embed, wait=True)
            results = db.search("hello", embed_fn=_embed)
            assert len(results) >= 1
            assert db.vector_backend_name == "usearch"

    def test_usearch_creates_index_file(self, tmp_path):
        db_path = str(tmp_path / "uf.db")
        with SQFox(db_path, vector_backend="usearch") as db:
            db.ingest("test", embed_fn=_embed, wait=True)
        assert os.path.exists(db_path + ".usearch")

    def test_usearch_diagnostics(self, tmp_path):
        db_path = str(tmp_path / "diag.db")
        with SQFox(db_path, vector_backend="usearch") as db:
            diag = db.diagnostics()
            assert diag["vector_backend"] == "usearch"

    def test_usearch_backup_copies_index(self, tmp_path):
        db_path = str(tmp_path / "src.db")
        backup_path = str(tmp_path / "backup.db")
        with SQFox(db_path, vector_backend="usearch") as db:
            db.ingest("backup test", embed_fn=_embed, wait=True)
            db.backup(backup_path)
        assert os.path.exists(backup_path)
        assert os.path.exists(backup_path + ".usearch")

    def test_usearch_memory_mode(self, tmp_path):
        with SQFox(":memory:", vector_backend="usearch") as db:
            db.ingest("memory test", embed_fn=_embed, wait=True)
            results = db.search("memory", embed_fn=_embed)
            assert len(results) >= 1

    def test_usearch_custom_instance(self, tmp_path):
        from sqfox.backends.usearch import USearchBackend
        backend = USearchBackend(metric="cos", dtype="f32", connectivity=32)
        db_path = str(tmp_path / "custom.db")
        with SQFox(db_path, vector_backend=backend) as db:
            db.ingest("custom backend", embed_fn=_embed, wait=True)
            results = db.search("custom", embed_fn=_embed)
            assert len(results) >= 1


class TestEngineHnsw:
    """Engine integration with SqliteHnswBackend (pure Python, no ext deps)."""

    def test_hnsw_ingest_and_search(self, tmp_path):
        db_path = str(tmp_path / "hnsw.db")
        with SQFox(db_path, vector_backend="hnsw") as db:
            db.ingest("hello world", embed_fn=_embed, wait=True)
            db.ingest("goodbye world", embed_fn=_embed, wait=True)
            results = db.search("hello", embed_fn=_embed)
            assert len(results) >= 1
            assert db.vector_backend_name == "hnsw"

    def test_hnsw_single_file(self, tmp_path):
        """No external index files should be created."""
        db_path = str(tmp_path / "single.db")
        with SQFox(db_path, vector_backend="hnsw") as db:
            db.ingest("test", embed_fn=_embed, wait=True)
        # Only .db file should exist, no .usearch or other sidecar
        siblings = os.listdir(tmp_path)
        assert siblings == ["single.db"]

    def test_hnsw_diagnostics(self, tmp_path):
        db_path = str(tmp_path / "diag.db")
        with SQFox(db_path, vector_backend="hnsw") as db:
            diag = db.diagnostics()
            assert diag["vector_backend"] == "hnsw"

    def test_hnsw_persistence_across_restart(self, tmp_path):
        db_path = str(tmp_path / "restart.db")

        # First session: ingest
        with SQFox(db_path, vector_backend="hnsw") as db:
            db.ingest("alpha bravo charlie", embed_fn=_embed, wait=True)
            db.ingest("delta echo foxtrot", embed_fn=_embed, wait=True)

        # Second session: search without re-ingesting
        with SQFox(db_path, vector_backend="hnsw") as db:
            results = db.search("alpha", embed_fn=_embed)
            assert len(results) >= 1

    def test_hnsw_custom_instance(self, tmp_path):
        backend = SqliteHnswBackend(M=8, ef_construction=64, ef_search=32)
        db_path = str(tmp_path / "custom.db")
        with SQFox(db_path, vector_backend=backend) as db:
            db.ingest("custom hnsw backend", embed_fn=_embed, wait=True)
            results = db.search("custom", embed_fn=_embed)
            assert len(results) >= 1
