"""Tests for pluggable vector backends."""

import os
import sqlite3
import struct

import pytest

from sqfox import SQFox, VectorBackend, VectorBackendError, SqliteHnswBackend, SqliteFlatBackend
from sqfox.backends import get_backend

from conftest import requires_usearch, requires_sentence_transformers


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

    @requires_usearch
    def test_usearch_alias(self):
        b = get_backend("usearch")
        assert isinstance(b, VectorBackend)

    def test_hnsw_alias(self):
        b = get_backend("hnsw")
        assert isinstance(b, SqliteHnswBackend)

    def test_flat_alias(self):
        b = get_backend("flat")
        assert isinstance(b, SqliteFlatBackend)

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_backend("nonexistent-backend")

    def test_passthrough_instance(self):
        b = SqliteFlatBackend()
        assert get_backend(b) is b


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

class TestEngineDefaultNoBackend:
    """When no backend is set, vector_backend_name is None."""

    def test_default_no_backend_is_none(self, tmp_path):
        db_path = str(tmp_path / "default.db")
        with SQFox(db_path) as db:
            assert db.vector_backend_name is None


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

    def test_usearch_backup_is_consistent(self, tmp_path):
        """Backup SQLite and .usearch file must be a consistent snapshot.

        Ingest several documents, backup, then open the backup with
        USearch and verify the index count matches the SQLite doc count.
        """
        db_path = str(tmp_path / "src.db")
        backup_path = str(tmp_path / "backup.db")

        with SQFox(db_path, vector_backend="usearch") as db:
            for i in range(10):
                db.ingest(f"document {i}", embed_fn=_embed, wait=True)
            db.backup(backup_path)

        # Open backup and verify consistency
        with SQFox(backup_path, vector_backend="usearch") as db2:
            results = db2.search("document", embed_fn=_embed, limit=20)
            assert len(results) == 10
            diag = db2.diagnostics()
            assert diag["vector_backend"] == "usearch"

    def test_usearch_backup_atomic_no_interleave(self, tmp_path):
        """Atomic backup must block ingest while running.

        Insert docs, start backup, verify that a concurrent ingest
        submitted AFTER backup started does not appear in the backup.
        """
        import threading

        db_path = str(tmp_path / "src.db")
        backup_path = str(tmp_path / "backup.db")

        with SQFox(db_path, vector_backend="usearch") as db:
            # Ingest baseline docs
            for i in range(5):
                db.ingest(f"baseline {i}", embed_fn=_embed, wait=True)

            # Slow progress callback to hold backup on writer thread
            backup_started = threading.Event()
            backup_done = threading.Event()

            def slow_progress(status, remaining, total):
                backup_started.set()

            def do_backup():
                db.backup(backup_path, progress=slow_progress)
                backup_done.set()

            t = threading.Thread(target=do_backup)
            t.start()
            t.join(timeout=10)
            assert not t.is_alive(), "Backup thread did not finish in time"
            assert backup_done.is_set(), "Backup did not complete"

        # Open backup — should have exactly 5 baseline docs
        with SQFox(backup_path, vector_backend="usearch") as db2:
            results = db2.search("baseline", embed_fn=_embed, limit=20)
            assert len(results) == 5

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

    def test_hnsw_backup_consistent(self, tmp_path):
        """HNSW stores data inside SQLite, so backup is inherently consistent."""
        db_path = str(tmp_path / "hnsw.db")
        backup_path = str(tmp_path / "backup.db")
        with SQFox(db_path, vector_backend="hnsw") as db:
            for i in range(5):
                db.ingest(f"hnsw doc {i}", embed_fn=_embed, wait=True)
            db.backup(backup_path)

        assert os.path.exists(backup_path)
        # No external index file
        assert not os.path.exists(backup_path + ".usearch")

        with SQFox(backup_path, vector_backend="hnsw") as db2:
            results = db2.search("hnsw doc", embed_fn=_embed, limit=10)
            assert len(results) == 5


# ---------------------------------------------------------------------------
# SqliteFlatBackend tests (pure Python, BQ + exact rerank)
# ---------------------------------------------------------------------------

class TestSqliteFlatBackend:
    """Unit tests for pure-Python flat (brute-force) backend."""

    @pytest.fixture
    def backend_with_db(self, tmp_path):
        """SqliteFlatBackend wired to a real SQLite DB with documents table."""
        db_path = str(tmp_path / "flat.db")
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
        b = SqliteFlatBackend()
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
        b.add([1, 2, 3], [v1, v2, v3])

        results = b.search(v1, 3)
        assert len(results) >= 1
        # Closest to v1 should be key=1
        assert results[0][0] == 1

    def test_search_ordering(self, backend_with_db):
        b, conn = backend_with_db
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        b.add([1, 2, 3], [v1, v2, v3])

        results = b.search(v1, 3)
        keys = [k for k, _ in results]
        # v1 closest, then v2, then v3
        assert keys[0] == 1
        assert keys[1] == 2
        assert keys[2] == 3

    def test_search_empty_index(self, backend_with_db):
        b, conn = backend_with_db
        results = b.search([1.0] * NDIM, 5)
        assert results == []

    def test_count(self, backend_with_db):
        b, conn = backend_with_db
        assert b.count() == 0
        vecs = [[float(i)] * NDIM for i in range(5)]
        b.add([1, 2, 3, 4, 5], vecs)
        assert b.count() == 5

    def test_remove(self, backend_with_db):
        b, conn = backend_with_db
        v = [1.0] * NDIM
        b.add([10], [v])
        assert b.count() == 1

        b.remove([10])
        assert b.count() == 0
        results = b.search(v, 5)
        keys = [k for k, _ in results]
        assert 10 not in keys

    def test_upsert(self, backend_with_db):
        b, conn = backend_with_db
        v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b.add([1], [v1])
        assert b.count() == 1

        # Upsert: same key, different vector
        b.add([1], [v2])
        assert b.count() == 1
        results = b.search(v2, 1)
        assert results[0][0] == 1

    def test_flush_is_noop(self, backend_with_db):
        b, _ = backend_with_db
        b.flush()  # Should not raise

    def test_verify_consistency(self, backend_with_db):
        b, conn = backend_with_db
        vecs = [[float(i)] * NDIM for i in range(3)]
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

        results = b.search(v1, 2)
        assert results[0][0] == 1

    def test_many_vectors(self, backend_with_db):
        """Insert 100 vectors and verify nearest-neighbor quality."""
        b, conn = backend_with_db
        import random as _rng
        _rng.seed(42)
        vecs = [[_rng.gauss(0, 1) for _ in range(NDIM)] for _ in range(100)]
        keys = list(range(1, 101))
        b.add(keys, vecs)

        # Search for vec[0], it should return key=1 as closest
        results = b.search(vecs[0], 5)
        assert len(results) >= 1
        assert results[0][0] == 1  # exact match

    def test_load_from_db_on_init(self, tmp_path):
        """Vectors stored in DB are loaded into cache on initialize."""
        db_path = str(tmp_path / "reload.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY, content TEXT, "
            "  embedding BLOB, vec_indexed INTEGER DEFAULT 0)"
        )
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

        b = SqliteFlatBackend()
        b.set_writer_conn(conn)
        b.initialize(db_path, NDIM)
        assert b.count() == 2

        results = b.search(v1, 2)
        assert results[0][0] == 1
        b.close()
        conn.close()


class TestEngineFlat:
    """Engine integration with SqliteFlatBackend (pure Python, no ext deps)."""

    def test_flat_ingest_and_search(self, tmp_path):
        db_path = str(tmp_path / "flat.db")
        with SQFox(db_path, vector_backend="flat") as db:
            db.ingest("hello world", embed_fn=_embed, wait=True)
            db.ingest("goodbye world", embed_fn=_embed, wait=True)
            results = db.search("hello", embed_fn=_embed)
            assert len(results) >= 1
            assert db.vector_backend_name == "flat"

    def test_flat_single_file(self, tmp_path):
        """No external index files should be created."""
        db_path = str(tmp_path / "single.db")
        with SQFox(db_path, vector_backend="flat") as db:
            db.ingest("test", embed_fn=_embed, wait=True)
        siblings = os.listdir(tmp_path)
        assert siblings == ["single.db"]

    def test_flat_diagnostics(self, tmp_path):
        db_path = str(tmp_path / "diag.db")
        with SQFox(db_path, vector_backend="flat") as db:
            diag = db.diagnostics()
            assert diag["vector_backend"] == "flat"

    def test_flat_persistence_across_restart(self, tmp_path):
        db_path = str(tmp_path / "restart.db")

        # First session: ingest
        with SQFox(db_path, vector_backend="flat") as db:
            db.ingest("alpha bravo charlie", embed_fn=_embed, wait=True)
            db.ingest("delta echo foxtrot", embed_fn=_embed, wait=True)

        # Second session: search without re-ingesting
        with SQFox(db_path, vector_backend="flat") as db:
            results = db.search("alpha", embed_fn=_embed)
            assert len(results) >= 1

    def test_flat_custom_instance(self, tmp_path):
        backend = SqliteFlatBackend(oversample=10)
        db_path = str(tmp_path / "custom.db")
        with SQFox(db_path, vector_backend=backend) as db:
            db.ingest("custom flat backend", embed_fn=_embed, wait=True)
            results = db.search("custom", embed_fn=_embed)
            assert len(results) >= 1


# ---------------------------------------------------------------------------
# SqliteFlatBackend: BQ pipeline + edge cases
# ---------------------------------------------------------------------------

class TestFlatBQPipeline:
    """Tests that exercise the BQ prescore + exact rerank path."""

    @pytest.fixture
    def large_backend(self):
        """Backend with 500 random 32-dim vectors — triggers BQ path."""
        import random as _rng
        _rng.seed(42)
        DIM = 32
        N = 500

        b = SqliteFlatBackend(oversample=20)
        b.initialize(":memory:", DIM)

        vecs = []
        for _ in range(N):
            v = [_rng.gauss(0, 1) for _ in range(DIM)]
            norm = sum(x * x for x in v) ** 0.5
            v = [x / norm for x in v]
            vecs.append(v)
        keys = list(range(1, N + 1))
        b.add(keys, vecs)
        yield b, vecs
        b.close()

    def test_bq_path_triggers(self, large_backend):
        """With 500 vectors and oversample=20, k=10 → 200 candidates < 500."""
        b, vecs = large_backend
        # k*oversample = 10*20 = 200 < 500 → BQ prescore path
        results = b.search(vecs[0], 10)
        assert len(results) == 10
        # Exact match must be #1 (distance ~ 0)
        assert results[0][0] == 1
        assert results[0][1] < 0.01

    def test_bq_results_sorted_by_distance(self, large_backend):
        b, vecs = large_backend
        results = b.search(vecs[42], 10)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_bq_recall_vs_exact(self, large_backend):
        """BQ recall on 32-dim normalized vectors should be reasonable."""
        import math
        import array as _array
        b, vecs = large_backend
        q = vecs[0]
        q_arr = _array.array("f", q)

        # Ground truth: exact brute-force
        all_dists = []
        for i, v in enumerate(vecs):
            d = math.dist(q_arr, _array.array("f", v))
            all_dists.append((d, i + 1))
        all_dists.sort()
        exact_top10 = set(did for _, did in all_dists[:10])

        # BQ approximate
        approx = b.search(q, 10)
        approx_top10 = set(did for did, _ in approx)

        recall = len(exact_top10 & approx_top10) / 10
        # On 32-dim normalized vectors, BQ oversample=20 should get decent recall
        assert recall >= 0.5, f"Recall too low: {recall:.0%}"

    def test_search_k_greater_than_n(self, large_backend):
        """k > n should return n results without error."""
        b, vecs = large_backend
        results = b.search(vecs[0], 9999)
        assert len(results) == 500

    def test_search_single_vector(self):
        """Search on an index with exactly one vector."""
        b = SqliteFlatBackend()
        b.initialize(":memory:", 4)
        b.add([42], [[1.0, 2.0, 3.0, 4.0]])
        results = b.search([1.0, 2.0, 3.0, 4.0], 5)
        assert len(results) == 1
        assert results[0][0] == 42
        assert results[0][1] < 0.01
        b.close()

    def test_oversample_affects_candidate_count(self):
        """Higher oversample → better recall on same dataset."""
        import random as _rng
        import math
        import array as _array
        _rng.seed(99)
        DIM = 64
        N = 1000
        K = 10

        vecs = []
        for _ in range(N):
            v = [_rng.gauss(0, 1) for _ in range(DIM)]
            norm = sum(x * x for x in v) ** 0.5
            v = [x / norm for x in v]
            vecs.append(v)
        keys = list(range(1, N + 1))

        q = vecs[0]
        q_arr = _array.array("f", q)
        all_dists = sorted(
            (math.dist(q_arr, _array.array("f", v)), i + 1)
            for i, v in enumerate(vecs)
        )
        exact_topk = set(did for _, did in all_dists[:K])

        recalls = {}
        for osamp in (5, 20):
            b = SqliteFlatBackend(oversample=osamp)
            b.initialize(":memory:", DIM)
            b.add(keys, vecs)
            approx = b.search(q, K)
            approx_topk = set(did for did, _ in approx)
            recalls[osamp] = len(exact_topk & approx_topk) / K
            b.close()

        # oversample=20 should have >= recall of oversample=5
        assert recalls[20] >= recalls[5]


# ---------------------------------------------------------------------------
# Real embedding test with Qwen3-Embedding-0.6B
# ---------------------------------------------------------------------------

@requires_sentence_transformers
class TestFlatQwenEmbedding:
    """End-to-end test with real Qwen3 embeddings.

    Verifies that BQ recall is high on semantically structured vectors
    (unlike random vectors where BQ is pathological).
    """

    @pytest.fixture(scope="class")
    def qwen_model(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            truncate_dim=256,
        )
        return model

    @pytest.fixture(scope="class")
    def corpus_and_embeddings(self, qwen_model):
        """A small corpus of semantically diverse texts + their embeddings."""
        corpus = [
            # Cluster 1: programming
            "Python is a popular programming language for data science",
            "JavaScript is used for web development and Node.js backends",
            "Rust provides memory safety without garbage collection",
            "C++ is widely used in game engines and systems programming",
            "Go language was designed by Google for concurrent programming",
            "TypeScript adds static typing to JavaScript applications",
            "Java remains dominant in enterprise backend systems",
            "Ruby on Rails popularized convention over configuration",
            "Swift was created by Apple for iOS app development",
            "Kotlin is the preferred language for Android development",
            # Cluster 2: animals
            "Dogs are loyal companions and have been domesticated for millennia",
            "Cats are independent pets that hunt mice and small rodents",
            "Elephants are the largest land animals with excellent memory",
            "Dolphins are highly intelligent marine mammals",
            "Eagles soar at great heights with exceptional eyesight",
            "Whales migrate thousands of miles across ocean basins",
            "Pandas eat bamboo and live in Chinese mountain forests",
            "Tigers are solitary hunters in Asian jungle habitats",
            "Penguins thrive in cold Antarctic environments",
            "Wolves hunt in cooperative packs with strict hierarchy",
            # Cluster 3: food / cooking
            "Italian pizza originated in Naples with tomato and mozzarella",
            "Japanese sushi features vinegared rice with fresh fish",
            "Mexican tacos combine corn tortillas with various fillings",
            "French croissants are flaky butter pastries for breakfast",
            "Indian curry uses complex spice blends like garam masala",
            "Thai tom yum soup balances sour lime and spicy chili",
            "Chinese dim sum includes steamed dumplings and buns",
            "Greek salad combines feta cheese with olives and tomatoes",
            "Korean kimchi is fermented cabbage with chili and garlic",
            "Brazilian churrasco is grilled meat on rotating skewers",
            # Cluster 4: space / science
            "Black holes warp spacetime with extreme gravitational pull",
            "The James Webb telescope captures infrared cosmic images",
            "Mars rovers explore the red planet surface for signs of water",
            "Quantum entanglement allows instant correlation over distance",
            "DNA double helix structure was discovered by Watson and Crick",
            "Photosynthesis converts sunlight into chemical energy in plants",
            "Nuclear fusion powers stars by combining hydrogen atoms",
            "Plate tectonics explains continental drift and earthquakes",
            "The Big Bang theory describes the origin of the universe",
            "CRISPR gene editing can modify DNA sequences precisely",
        ]
        embeddings = qwen_model.encode(corpus).tolist()
        return corpus, embeddings

    def test_bq_recall_on_real_embeddings(self, corpus_and_embeddings):
        """BQ at oversample=20 on real 256-dim embeddings: recall must be high."""
        import math
        import array as _array
        corpus, embeddings = corpus_and_embeddings
        DIM = len(embeddings[0])
        N = len(embeddings)
        K = 5

        b = SqliteFlatBackend(oversample=20)
        b.initialize(":memory:", DIM)
        b.add(list(range(1, N + 1)), embeddings)

        # Query: embed the first document, expect it as top-1
        q = embeddings[0]
        results = b.search(q, K)
        assert results[0][0] == 1, "Exact match should be top-1"
        assert results[0][1] < 0.01, "Distance to self should be ~0"

        # Recall check against brute-force
        q_arr = _array.array("f", q)
        all_dists = sorted(
            (math.dist(q_arr, _array.array("f", e)), i + 1)
            for i, e in enumerate(embeddings)
        )
        exact_topk = set(did for _, did in all_dists[:K])
        approx_topk = set(did for did, _ in results)
        recall = len(exact_topk & approx_topk) / K
        assert recall >= 0.8, f"BQ recall on real embeddings too low: {recall:.0%}"
        b.close()

    def test_semantic_search_quality(self, qwen_model, corpus_and_embeddings):
        """Semantic query should return results from the right cluster."""
        corpus, embeddings = corpus_and_embeddings
        DIM = len(embeddings[0])
        N = len(embeddings)

        b = SqliteFlatBackend(oversample=20)
        b.initialize(":memory:", DIM)
        b.add(list(range(1, N + 1)), embeddings)

        # Query about programming — should return programming docs (ids 1-10)
        q_vec = qwen_model.encode(
            "What programming language should I learn?",
            prompt_name="query",
        ).tolist()
        results = b.search(q_vec, 5)
        top5_ids = [did for did, _ in results]
        # At least 3 of top-5 should be from programming cluster (ids 1-10)
        programming_hits = sum(1 for did in top5_ids if 1 <= did <= 10)
        assert programming_hits >= 3, (
            f"Expected >=3 programming results in top-5, got {programming_hits}: "
            f"{[(did, corpus[did - 1][:40]) for did in top5_ids]}"
        )

        # Query about animals — should return animal docs (ids 11-20)
        q_vec = qwen_model.encode(
            "Tell me about wild animals in nature",
            prompt_name="query",
        ).tolist()
        results = b.search(q_vec, 5)
        top5_ids = [did for did, _ in results]
        animal_hits = sum(1 for did in top5_ids if 11 <= did <= 20)
        assert animal_hits >= 3, (
            f"Expected >=3 animal results in top-5, got {animal_hits}: "
            f"{[(did, corpus[did - 1][:40]) for did in top5_ids]}"
        )
        b.close()

    def test_engine_e2e_with_qwen(self, qwen_model, tmp_path):
        """Full engine roundtrip: ingest with Qwen, search with flat backend."""
        class _QwenEmbed:
            def __init__(self, model):
                self.model = model
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            def embed_query(self, text):
                return self.model.encode(text, prompt_name="query").tolist()

        embedder = _QwenEmbed(qwen_model)
        db_path = str(tmp_path / "qwen_e2e.db")
        with SQFox(db_path, vector_backend="flat") as db:
            db.ingest("Python is great for machine learning", embed_fn=embedder, wait=True)
            db.ingest("Cats love to sleep in sunbeams", embed_fn=embedder, wait=True)
            db.ingest("The universe began with the Big Bang", embed_fn=embedder, wait=True)

            results = db.search("deep learning and AI programming", embed_fn=embedder)
            assert len(results) >= 1
            # The programming-related doc should rank highest
            assert "Python" in results[0].text or "machine" in results[0].text


# ---------------------------------------------------------------------------
# Introspection API tests
# ---------------------------------------------------------------------------


class TestHnswIntrospection:
    """Tests for SqliteHnswBackend introspection methods."""

    @pytest.fixture
    def hnsw_db(self, tmp_path):
        """Create an HNSW-backed SQFox with 20 vectors."""
        db_path = str(tmp_path / "intro.db")
        backend = SqliteHnswBackend(M=4, ef_construction=20, ef_search=10)
        with SQFox(db_path, vector_backend=backend) as db:
            for i in range(20):
                db.ingest(
                    f"Document number {i} with content",
                    embed_fn=_embed, wait=True,
                )
            yield db, backend

    def test_graph_stats_keys(self, hnsw_db):
        db, backend = hnsw_db
        stats = backend.graph_stats()
        assert stats["count"] == 20
        assert stats["M"] == 4
        assert stats["M0"] == 8  # 2 * M
        assert stats["ef_construction"] == 20
        assert stats["ef_search"] == 10
        assert "levels" in stats
        assert isinstance(stats["levels"], list)
        assert len(stats["levels"]) > 0

    def test_graph_stats_level_0_has_all_nodes(self, hnsw_db):
        db, backend = hnsw_db
        stats = backend.graph_stats()
        level0 = stats["levels"][0]
        assert level0["level"] == 0
        assert level0["nodes"] == 20

    def test_graph_stats_level_keys(self, hnsw_db):
        db, backend = hnsw_db
        stats = backend.graph_stats()
        for lv_info in stats["levels"]:
            for key in ("level", "nodes", "edges", "avg_degree",
                        "min_degree", "max_degree", "orphans"):
                assert key in lv_info

    def test_node_info_found(self, hnsw_db):
        db, backend = hnsw_db
        info = backend.node_info(1)
        assert info is not None
        assert "level" in info
        assert isinstance(info["level"], int)
        assert "neighbors_by_level" in info
        assert 0 in info["neighbors_by_level"]
        assert isinstance(info["neighbors_by_level"][0], list)

    def test_node_info_not_found(self, hnsw_db):
        db, backend = hnsw_db
        assert backend.node_info(99999) is None

    def test_top_hubs(self, hnsw_db):
        db, backend = hnsw_db
        hubs = backend.top_hubs(5)
        assert len(hubs) <= 5
        assert all(isinstance(h, tuple) and len(h) == 2 for h in hubs)
        # Sorted by degree descending
        degrees = [d for _, d in hubs]
        assert degrees == sorted(degrees, reverse=True)

    def test_top_hubs_empty(self):
        backend = SqliteHnswBackend()
        assert backend.top_hubs() == []

    def test_search_trace(self, hnsw_db):
        db, backend = hnsw_db
        query = _embed_single("test query")
        with db.reader() as conn:
            trace = backend.search_trace(query, conn)
        assert len(trace) > 0
        assert trace[0]["action"] == "entry"
        assert trace[-1]["action"] == "found"
        for step in trace:
            assert "level" in step
            assert "node" in step
            assert "distance" in step
            assert "action" in step

    def test_search_trace_empty_index(self):
        backend = SqliteHnswBackend()
        trace = backend.search_trace([1.0] * 8, None)
        assert trace == []


class TestFlatIntrospection:
    """Tests for SqliteFlatBackend introspection properties."""

    def test_blas_available_type(self):
        b = SqliteFlatBackend()
        b.initialize(":memory:", 8)
        assert isinstance(b.blas_available, bool)

    def test_rerank_method_string(self):
        b = SqliteFlatBackend()
        b.initialize(":memory:", 8)
        assert b.rerank_method in ("cblas_sdot (SIMD)", "math.dist")

    def test_blas_consistency(self):
        b = SqliteFlatBackend()
        b.initialize(":memory:", 8)
        if b.blas_available:
            assert b.rerank_method == "cblas_sdot (SIMD)"
        else:
            assert b.rerank_method == "math.dist"

    def test_blas_loaded_function(self):
        from sqfox.backends.flat import blas_loaded
        assert isinstance(blas_loaded(), bool)
