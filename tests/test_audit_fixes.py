"""Tests for audit fixes — NaN guards, FTS5 sanitization, distance safety, etc."""

import math
import sqlite3
import struct

import pytest

from sqfox.backends.flat import SqliteFlatBackend
from sqfox.backends.hnsw import SqliteHnswBackend
from sqfox.backends.registry import get_backend
from sqfox.search import fts_search, vec_search
from sqfox.types import embed_for_query, embed_for_documents


NDIM = 8


# ---------------------------------------------------------------------------
# Flat backend: NaN guards
# ---------------------------------------------------------------------------

class TestFlatNaNGuards:
    """NaN/Inf vectors should be rejected or skipped gracefully."""

    @pytest.fixture
    def backend(self):
        b = SqliteFlatBackend()
        b.initialize(":memory:", NDIM)
        yield b
        b.close()

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf")])
    def test_add_bad_vector_skipped(self, bad_val):
        backend = SqliteFlatBackend()
        backend.initialize(":memory:", 3)
        backend.add([1], [[1.0, 2.0, 3.0]])
        backend.add([2], [[bad_val, 0.0, 0.0]])
        assert backend.count() == 1

    def test_add_all_nan_no_snapshot_change(self, backend):
        """If every vector is NaN, count stays 0."""
        bad1 = [float("nan")] * NDIM
        bad2 = [float("inf")] * NDIM
        backend.add([1, 2], [bad1, bad2])
        assert backend.count() == 0

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf")])
    def test_search_bad_query_returns_empty(self, bad_val):
        backend = SqliteFlatBackend()
        backend.initialize(":memory:", 3)
        backend.add([1], [[1.0, 0.0, 0.0]])
        result = backend.search([bad_val, 0.0, 0.0], 5)
        assert result == []

    def test_search_k_zero_returns_empty(self, backend):
        """search() with k=0 should return []."""
        backend.add([1], [[1.0] * NDIM])
        results = backend.search([1.0] * NDIM, 0)
        assert results == []

    def test_initialize_skips_nan_blobs(self, tmp_path):
        """initialize() should skip NaN vectors loaded from DB."""
        db_path = str(tmp_path / "nan_init.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY, content TEXT, "
            "  embedding BLOB, vec_indexed INTEGER DEFAULT 0)"
        )
        good = [1.0] * NDIM
        bad = [float("nan")] + [0.0] * (NDIM - 1)
        blob_good = struct.pack(f"{NDIM}f", *good)
        blob_bad = struct.pack(f"{NDIM}f", *bad)
        conn.execute(
            "INSERT INTO documents VALUES (1, 'ok', ?, 1)", (blob_good,)
        )
        conn.execute(
            "INSERT INTO documents VALUES (2, 'bad', ?, 1)", (blob_bad,)
        )
        conn.commit()

        b = SqliteFlatBackend()
        b.set_writer_conn(conn)
        b.initialize(db_path, NDIM)
        assert b.count() == 1  # only the good vector
        b.close()
        conn.close()

    def test_rebuild_from_blobs_skips_nan(self):
        """rebuild_from_blobs() should skip NaN vectors."""
        b = SqliteFlatBackend()
        b.initialize(":memory:", NDIM)

        good = struct.pack(f"{NDIM}f", *([1.0] * NDIM))
        bad = struct.pack(f"{NDIM}f", *([float("nan")] + [0.0] * (NDIM - 1)))
        b.rebuild_from_blobs([(1, good), (2, bad)], NDIM)
        assert b.count() == 1
        b.close()


# ---------------------------------------------------------------------------
# Flat backend: distance consistency
# ---------------------------------------------------------------------------

class TestFlatDistanceConsistency:
    """Distances from flat backend should be Euclidean L2 regardless of path."""

    def test_distance_matches_math_dist(self):
        """Distance returned by search should match math.dist."""
        b = SqliteFlatBackend()
        b.initialize(":memory:", NDIM)
        v1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        v2 = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        b.add([1, 2], [v1, v2])

        q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        results = b.search(q, 2)
        for doc_id, dist in results:
            vec = v1 if doc_id == 1 else v2
            expected = math.dist(q, vec)
            assert dist == pytest.approx(expected, abs=1e-3)
        b.close()


# ---------------------------------------------------------------------------
# search.py: vec_search NaN/negative distance guards
# ---------------------------------------------------------------------------

class TestVecSearchDistanceGuards:
    """vec_search should filter NaN, Inf, and negative distances."""

    @pytest.mark.parametrize("bad_dist", [float("nan"), -1.0, float("inf")])
    def test_backend_bad_distance_filtered(self, bad_dist, tmp_path):
        # Use the same mock backend pattern from the existing tests but parametrized
        from sqfox.search import vec_search

        class BadBackend:
            def search(self, query, k, **kw):
                return [(1, 0.5), (2, bad_dist), (3, 0.3)]

        db_path = str(tmp_path / "bad_dist.db")
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, content TEXT, metadata TEXT, chunk_of INTEGER, embedding BLOB, content_lemmatized TEXT, vec_indexed INTEGER DEFAULT 0, fts_indexed INTEGER DEFAULT 0)")
        conn.execute("INSERT INTO documents (id, content, metadata) VALUES (1, 'a', '{}')")
        conn.execute("INSERT INTO documents (id, content, metadata) VALUES (2, 'b', '{}')")
        conn.execute("INSERT INTO documents (id, content, metadata) VALUES (3, 'c', '{}')")
        conn.commit()

        results = vec_search(conn, [1.0, 0.0], limit=10, vector_backend=BadBackend())
        # Only finite non-negative distances should survive
        ids = [doc_id for doc_id, _ in results]
        assert 2 not in ids
        assert 1 in ids
        assert 3 in ids
        conn.close()


# ---------------------------------------------------------------------------
# FTS5 operator sanitization
# ---------------------------------------------------------------------------

class TestFTS5Sanitization:
    """FTS5 boolean operators AND/OR/NOT/NEAR should be neutralized."""

    @pytest.fixture
    def fts_conn(self, tmp_path):
        """Create a minimal FTS5 table for testing."""
        conn = sqlite3.connect(str(tmp_path / "fts.db"))
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE documents_fts "
            "USING fts5(content, content=documents, content_rowid=id)"
        )
        # Insert a document containing the word "and"
        conn.execute(
            "INSERT INTO documents (id, content) VALUES (1, 'cats and dogs')"
        )
        conn.execute(
            "INSERT INTO documents_fts (rowid, content) "
            "VALUES (1, 'cats and dogs')"
        )
        conn.execute(
            "INSERT INTO documents (id, content) VALUES (2, 'hello world')"
        )
        conn.execute(
            "INSERT INTO documents_fts (rowid, content) "
            "VALUES (2, 'hello world')"
        )
        conn.commit()
        yield conn
        conn.close()

    def test_query_with_AND_operator(self, fts_conn):
        """Query containing 'AND' should not crash FTS5."""
        results = fts_search(fts_conn, "cats AND dogs")
        # Should not raise, may or may not find results
        assert isinstance(results, list)

    def test_query_with_OR_operator(self, fts_conn):
        """Query containing 'OR' should not crash FTS5."""
        results = fts_search(fts_conn, "cats OR hello")
        assert isinstance(results, list)

    def test_query_with_NOT_operator(self, fts_conn):
        """Query containing 'NOT' should not crash FTS5."""
        results = fts_search(fts_conn, "NOT cats")
        assert isinstance(results, list)

    def test_query_with_NEAR_operator(self, fts_conn):
        """Query containing 'NEAR' should not crash FTS5."""
        results = fts_search(fts_conn, "cats NEAR dogs")
        assert isinstance(results, list)

    def test_query_with_leading_hyphen(self, fts_conn):
        """Query with leading hyphen (FTS5 NOT prefix) should not crash."""
        results = fts_search(fts_conn, "-cats dogs")
        assert isinstance(results, list)

    def test_query_with_special_chars(self, fts_conn):
        """Quotes, parens, asterisks should be stripped."""
        results = fts_search(fts_conn, '"cats" AND (dogs*)')
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# HNSW: double remove
# ---------------------------------------------------------------------------

class TestHnswDoubleRemove:
    """Removing the same key twice should not double-decrement count."""

    @pytest.fixture
    def backend_with_db(self, tmp_path):
        db_path = str(tmp_path / "hnsw_dr.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY, content TEXT, "
            "  embedding BLOB, vec_indexed INTEGER DEFAULT 0)"
        )
        conn.commit()
        b = SqliteHnswBackend(M=4, ef_construction=32, ef_search=16)
        b.set_writer_conn(conn)
        b.initialize(db_path, NDIM)
        yield b, conn
        b.close()
        conn.close()

    def test_double_remove_safe(self, backend_with_db):
        b, conn = backend_with_db
        v = [1.0] * NDIM
        blob = struct.pack(f"{NDIM}f", *v)
        conn.execute(
            "INSERT INTO documents (id, content, embedding, vec_indexed) "
            "VALUES (1, 'test', ?, 1)", (blob,)
        )
        conn.commit()
        b.add([1], [v])
        assert b.count() == 1

        b.remove([1])
        assert b.count() == 0

        b.remove([1])  # second remove — should be no-op
        assert b.count() == 0

    def test_remove_nonexistent_key(self, backend_with_db):
        b, conn = backend_with_db
        b.remove([999])  # never added — should be no-op
        assert b.count() == 0


# ---------------------------------------------------------------------------
# Registry: invalid type validation
# ---------------------------------------------------------------------------

class TestRegistryValidation:
    def test_invalid_type_raises_typeerror(self):
        """Passing an object without required methods should raise TypeError."""
        with pytest.raises(TypeError, match="missing"):
            get_backend(42)  # type: ignore

    def test_partial_protocol_raises_typeerror(self):
        """Object with only some methods should raise TypeError."""
        class _Partial:
            def initialize(self, *a): ...
            def search(self, *a): ...
            # missing add, flush, close

        with pytest.raises(TypeError, match="missing"):
            get_backend(_Partial())  # type: ignore


# ---------------------------------------------------------------------------
# types.py: embed_for_query guards
# ---------------------------------------------------------------------------

class TestEmbedForQueryGuards:
    def test_empty_outer_list_raises(self):
        """embed_fn returning [] should raise ValueError."""
        def bad_embed(texts):
            return []

        with pytest.raises(ValueError, match="empty"):
            embed_for_query(bad_embed, "hello")

    def test_empty_inner_vector_raises(self):
        """embed_fn returning [[]] should raise ValueError."""
        def bad_embed(texts):
            return [[]]

        with pytest.raises(ValueError, match="empty"):
            embed_for_query(bad_embed, "hello")

    def test_valid_embed_works(self):
        """Normal embed_fn should work fine."""
        def good_embed(texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

        result = embed_for_query(good_embed, "hello")
        assert result == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Flat backend: BLAS NaN distance filtering in search
# ---------------------------------------------------------------------------

class TestFlatBLASNaNFiltering:
    """Even if a stored vector somehow has bad norms_sq, search doesn't crash."""

    def test_corrupt_stored_vector_skipped_in_search(self):
        """Manually inject a corrupt vector, verify search skips it."""
        b = SqliteFlatBackend()
        b.initialize(":memory:", NDIM)
        # Add good vectors
        b.add([1, 2], [[1.0] * NDIM, [2.0] * NDIM])
        assert b.count() == 2

        # Manually corrupt one norms_sq entry
        snap = b._snap
        new_norms = list(snap.norms_sq)
        new_norms[1] = float("nan")  # corrupt the second vector's norm
        from sqfox.backends.flat import _Snapshot
        b._snap = _Snapshot(
            snap.bins, snap.vecs, snap.ids, new_norms,
            snap.id_set, snap.id_pos, snap.count,
        )

        # Search should still work (skip or handle corrupt entry)
        results = b.search([1.0] * NDIM, 5)
        # Should get at least the good vector
        assert len(results) >= 1
        b.close()


# ---------------------------------------------------------------------------
# Flat backend: upsert correctness (audit round 3, task #88)
# ---------------------------------------------------------------------------

class TestFlatUpsert:
    """Upsert must actually update vector data, not silently discard it."""

    def test_upsert_changes_vector(self):
        """After upsert, search should find the NEW vector, not the old one."""
        b = SqliteFlatBackend()
        b.initialize(":memory:", NDIM)

        # Insert initial vector (far from query)
        v_old = [0.0] * NDIM
        v_old[0] = 100.0
        b.add([1], [v_old])

        # Upsert same key with vector close to query
        v_new = [1.0] * NDIM
        b.add([1], [v_new])
        assert b.count() == 1

        # Search with query = [1.0]*NDIM — should match v_new (dist~0), not v_old
        results = b.search([1.0] * NDIM, 1)
        assert len(results) == 1
        assert results[0][0] == 1
        assert results[0][1] < 0.01  # distance ~ 0

        b.close()


# ---------------------------------------------------------------------------
# Flat backend: dimension validation (audit round 3, task #89)
# ---------------------------------------------------------------------------

class TestFlatDimensionValidation:
    """Wrong-dimension vectors should be rejected in add()."""

    def test_add_wrong_dim_skipped(self):
        b = SqliteFlatBackend()
        b.initialize(":memory:", NDIM)

        good = [1.0] * NDIM
        wrong = [1.0] * (NDIM + 2)  # too many dims
        b.add([1, 2], [good, wrong])
        assert b.count() == 1  # only good vector accepted

        b.close()

    def test_add_short_dim_skipped(self):
        b = SqliteFlatBackend()
        b.initialize(":memory:", NDIM)

        good = [1.0] * NDIM
        short = [1.0] * (NDIM - 2)  # too few dims
        b.add([1, 2], [good, short])
        assert b.count() == 1

        b.close()

    def test_initialize_wrong_blob_size_skipped(self, tmp_path):
        """Blob with wrong size should be skipped during initialize."""
        db_path = str(tmp_path / "dim.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY, content TEXT, "
            "  embedding BLOB, vec_indexed INTEGER DEFAULT 0)"
        )
        good_blob = struct.pack(f"{NDIM}f", *([1.0] * NDIM))
        bad_blob = struct.pack(f"{NDIM + 2}f", *([1.0] * (NDIM + 2)))
        conn.execute(
            "INSERT INTO documents VALUES (1, 'ok', ?, 1)", (good_blob,)
        )
        conn.execute(
            "INSERT INTO documents VALUES (2, 'bad', ?, 1)", (bad_blob,)
        )
        conn.commit()

        b = SqliteFlatBackend()
        b.set_writer_conn(conn)
        b.initialize(db_path, NDIM)
        assert b.count() == 1
        b.close()
        conn.close()


# ---------------------------------------------------------------------------
# Registry: remove and count validation (audit round 3, task #92)
# ---------------------------------------------------------------------------

class TestRegistryFullValidation:
    """Registry should validate remove and count methods too."""

    def test_missing_remove_raises(self):
        class _NoRemove:
            def set_writer_conn(self, conn): ...
            def initialize(self, *a): ...
            def search(self, *a): ...
            def add(self, *a): ...
            def flush(self): ...
            def count(self): ...
            def close(self): ...
            # missing remove

        with pytest.raises(TypeError, match="remove"):
            get_backend(_NoRemove())  # type: ignore

    def test_missing_count_raises(self):
        class _NoCount:
            def set_writer_conn(self, conn): ...
            def initialize(self, *a): ...
            def search(self, *a): ...
            def add(self, *a): ...
            def remove(self, *a): ...
            def flush(self): ...
            def close(self): ...
            # missing count

        with pytest.raises(TypeError, match="count"):
            get_backend(_NoCount())  # type: ignore

    def test_complete_protocol_passes(self):
        class _Complete:
            def set_writer_conn(self, conn): ...
            def initialize(self, *a): ...
            def search(self, *a): ...
            def add(self, *a): ...
            def remove(self, *a): ...
            def flush(self): ...
            def count(self): ...
            def close(self): ...

        result = get_backend(_Complete())  # type: ignore
        assert result is not None


# ---------------------------------------------------------------------------
# embed_for_documents guards (audit round 3, task #92)
# ---------------------------------------------------------------------------

class TestEmbedForDocumentsGuards:
    """embed_for_documents should validate result length and None entries."""

    def test_wrong_count_raises(self):
        """Returning fewer vectors than texts should raise."""
        def bad_embed(texts):
            return [[1.0, 2.0]]  # 1 vector for 2 texts

        with pytest.raises(ValueError, match="2 texts"):
            embed_for_documents(bad_embed, ["hello", "world"])

    def test_none_entry_raises(self):
        """Returning [None] for a text should raise."""
        def bad_embed(texts):
            return [None]

        with pytest.raises(ValueError, match="None or empty"):
            embed_for_documents(bad_embed, ["hello"])

    def test_empty_inner_raises(self):
        """Returning [[]] for a text should raise."""
        def bad_embed(texts):
            return [[]]

        with pytest.raises(ValueError, match="None or empty"):
            embed_for_documents(bad_embed, ["hello"])

    def test_valid_embed_works(self):
        """Normal embed_fn should work fine."""
        def good_embed(texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

        result = embed_for_documents(good_embed, ["a", "b"])
        assert len(result) == 2
        assert result[0] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Round 4 audit fixes
# ---------------------------------------------------------------------------

class TestEmbedForQueryValidation:
    """embed_for_query must validate Embedder.embed_query() returns."""

    def test_embedder_returns_none_raises(self):
        """Embedder.embed_query() returning None should raise."""
        class BadEmbedder:
            def embed_documents(self, texts):
                return [[1.0] * 8 for _ in texts]
            def embed_query(self, text):
                return None

        with pytest.raises(ValueError, match="None or empty"):
            embed_for_query(BadEmbedder(), "hello")

    def test_embedder_returns_empty_raises(self):
        """Embedder.embed_query() returning [] should raise."""
        class BadEmbedder:
            def embed_documents(self, texts):
                return [[1.0] * 8 for _ in texts]
            def embed_query(self, text):
                return []

        with pytest.raises(ValueError, match="None or empty"):
            embed_for_query(BadEmbedder(), "hello")

    def test_embedder_returns_valid(self):
        """Embedder.embed_query() returning valid vec should work."""
        class GoodEmbedder:
            def embed_documents(self, texts):
                return [[1.0] * 8 for _ in texts]
            def embed_query(self, text):
                return [1.0] * 8

        result = embed_for_query(GoodEmbedder(), "hello")
        assert len(result) == 8

    def test_callable_numpy_array_works(self):
        """Plain callable returning list-of-numpy should not crash."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        def embed(texts):
            return [np.array([1.0, 2.0, 3.0]) for _ in texts]

        result = embed_for_query(embed, "hello")
        assert len(result) == 3

    def test_callable_numpy_empty_raises(self):
        """Plain callable returning [np.array([])] should raise."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        def embed(texts):
            return [np.array([])]

        with pytest.raises(ValueError, match="empty or zero-length"):
            embed_for_query(embed, "hello")


class TestHnswUpsertLevel:
    """HNSW upsert must reuse existing level, not pick a new random one."""

    @pytest.fixture
    def backend(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "hnsw.db"))
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  content TEXT,"
            "  embedding BLOB,"
            "  vec_indexed INTEGER DEFAULT 0"
            ")"
        )
        conn.commit()
        b = SqliteHnswBackend()
        b.set_writer_conn(conn)
        b.initialize(str(tmp_path / "hnsw.db"), NDIM)
        yield b
        conn.close()

    def test_upsert_preserves_level(self, backend):
        """Upserting a key should keep its graph level, not randomize."""
        vec1 = [1.0] * NDIM
        backend.add([1], [vec1])
        level_before = backend._node_levels.get(1)

        # Upsert same key with different vector
        vec2 = [2.0] * NDIM
        backend.add([1], [vec2])
        level_after = backend._node_levels.get(1)

        assert level_before == level_after


class TestHnswDeserializeBounds:
    """_deserialize must reject truncated blobs."""

    @pytest.fixture
    def backend(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "hnsw.db"))
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  content TEXT,"
            "  embedding BLOB,"
            "  vec_indexed INTEGER DEFAULT 0"
            ")"
        )
        conn.commit()
        b = SqliteHnswBackend()
        b.set_writer_conn(conn)
        b.initialize(str(tmp_path / "hnsw.db"), NDIM)
        yield b
        conn.close()

    def test_bad_magic_raises(self, backend):
        with pytest.raises(ValueError, match="bad magic"):
            backend._deserialize(b"XXXX" + b"\x00" * 100)

    def test_truncated_header_raises(self, backend):
        with pytest.raises(ValueError, match="Truncated"):
            backend._deserialize(b"HNSW" + b"\x00" * 5)

    def test_truncated_nodes_raises(self, backend):
        """Build a valid header but truncate node data."""
        # Build a minimal valid blob, then truncate
        vec = [1.0] * NDIM
        backend.add([1], [vec])
        blob = backend._serialize()
        # Truncate 10 bytes off the end
        with pytest.raises(ValueError, match="Truncated"):
            backend._deserialize(blob[:-10])


class TestHnswBlobValidation:
    """rebuild_from_blobs must skip bad blobs."""

    @pytest.fixture
    def backend(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "hnsw.db"))
        conn.execute(
            "CREATE TABLE documents ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  content TEXT,"
            "  embedding BLOB,"
            "  vec_indexed INTEGER DEFAULT 0"
            ")"
        )
        conn.commit()
        b = SqliteHnswBackend()
        b.set_writer_conn(conn)
        b.initialize(str(tmp_path / "hnsw.db"), NDIM)
        yield b
        conn.close()

    def test_wrong_blob_size_skipped(self, backend):
        """Blobs with wrong size should be skipped, not crash rebuild."""
        good_blob = struct.pack(f"{NDIM}f", *([1.0] * NDIM))
        bad_blob = b"\x00" * 5  # wrong size
        rows = [(1, good_blob), (2, bad_blob), (3, good_blob)]
        backend.rebuild_from_blobs(rows, NDIM)
        assert backend.count() == 2  # only 2 good blobs


class TestRegistrySetWriterConn:
    """Registry must require set_writer_conn in custom backends."""

    def test_missing_set_writer_conn_rejected(self):
        class BadBackend:
            def initialize(self, db_path, ndim): ...
            def search(self, query, k, **kw): ...
            def add(self, keys, vectors): ...
            def remove(self, keys): ...
            def flush(self): ...
            def count(self): ...
            def close(self): ...
            # Missing set_writer_conn

        with pytest.raises(TypeError, match="set_writer_conn"):
            get_backend(BadBackend())


