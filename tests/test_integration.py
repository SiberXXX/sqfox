"""End-to-end integration tests: ingest → search through full stack."""

import struct
import time
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqfox import SQFox, SQFoxManager, SchemaState, Priority

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simple deterministic embeddings for testing.
# Each word maps to a direction in 4D space.
_WORD_VECTORS = {
    "database":   [1.0, 0.0, 0.0, 0.0],
    "sqlite":     [0.9, 0.1, 0.0, 0.0],
    "sql":        [0.8, 0.2, 0.0, 0.0],
    "server":     [0.0, 1.0, 0.0, 0.0],
    "network":    [0.0, 0.9, 0.1, 0.0],
    "python":     [0.0, 0.0, 1.0, 0.0],
    "code":       [0.0, 0.0, 0.9, 0.1],
    "sensor":     [0.0, 0.0, 0.0, 1.0],
    "temperature":[0.0, 0.1, 0.0, 0.9],
    "iot":        [0.1, 0.0, 0.0, 0.9],
}
_DEFAULT_VEC = [0.25, 0.25, 0.25, 0.25]


def mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic embedding: average of known word vectors."""
    results = []
    for text in texts:
        words = text.lower().split()
        vecs = [_WORD_VECTORS.get(w, _DEFAULT_VEC) for w in words]
        if not vecs:
            results.append(_DEFAULT_VEC[:])
            continue
        avg = [
            sum(v[i] for v in vecs) / len(vecs)
            for i in range(4)
        ]
        results.append(avg)
    return results


def simple_chunker(text: str) -> list[str]:
    """Split by double newline."""
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Single database end-to-end
# ---------------------------------------------------------------------------

class TestSingleDBEndToEnd:
    """Full cycle: create → ingest → search on one SQFox instance."""

    def test_ingest_and_fts_search(self, tmp_path):
        """Ingest documents, search by FTS only (no embed_fn)."""
        with SQFox(str(tmp_path / "fts.db")) as db:
            db.ensure_schema(SchemaState.SEARCHABLE)

            db.ingest("SQLite database configuration guide", wait=True)
            db.ingest("Python network server tutorial", wait=True)
            db.ingest("IoT sensor temperature monitoring", wait=True)

            # Wait for WAL visibility
            time.sleep(0.1)

            results = db.search("database")
            assert len(results) >= 1
            assert any("database" in r.text.lower() for r in results)

    def test_ingest_and_hybrid_search(self, tmp_path):
        """Ingest with embeddings, hybrid search."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        with SQFox(str(tmp_path / "hybrid.db")) as db:
            db.ingest(
                "SQLite database configuration and optimization",
                embed_fn=mock_embed,
                wait=True,
            )
            db.ingest(
                "Python network server deployment",
                embed_fn=mock_embed,
                wait=True,
            )
            db.ingest(
                "IoT sensor temperature monitoring system",
                embed_fn=mock_embed,
                wait=True,
            )

            time.sleep(0.1)

            # Hybrid search — should find database doc via both FTS and vectors
            results = db.search("database sql", embed_fn=mock_embed, limit=3)
            assert len(results) >= 1
            top = results[0]
            assert "database" in top.text.lower() or "sql" in top.text.lower()

    def test_ingest_with_chunker(self, tmp_path):
        """Ingest a long document that gets chunked."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        long_doc = (
            "SQLite database engine overview\n\n"
            "Python code for server deployment\n\n"
            "IoT sensor temperature readings"
        )

        with SQFox(str(tmp_path / "chunked.db")) as db:
            parent_id = db.ingest(
                long_doc,
                chunker=simple_chunker,
                embed_fn=mock_embed,
                wait=True,
            )

            time.sleep(0.1)

            # Parent + 3 chunks = 4 documents
            row = db.fetch_one("SELECT COUNT(*) FROM documents")
            assert row[0] == 4

            # Chunks should reference parent
            chunks = db.fetch_all(
                "SELECT id, chunk_of FROM documents WHERE chunk_of IS NOT NULL"
            )
            assert len(chunks) == 3
            assert all(c["chunk_of"] == parent_id for c in chunks)

            # Search should find the right chunk
            results = db.search("sensor temperature", embed_fn=mock_embed)
            assert len(results) >= 1
            assert "sensor" in results[0].text.lower() or "temperature" in results[0].text.lower()

    def test_ingest_with_metadata(self, tmp_path):
        """Metadata is stored and returned in search results."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        with SQFox(str(tmp_path / "meta.db")) as db:
            db.ingest(
                "SQLite database performance tuning",
                metadata={"source": "docs", "version": 3},
                embed_fn=mock_embed,
                wait=True,
            )

            time.sleep(0.1)

            results = db.search("database performance", embed_fn=mock_embed)
            assert len(results) >= 1
            assert results[0].metadata["source"] == "docs"
            assert results[0].metadata["version"] == 3

    def test_dimension_mismatch_caught(self, tmp_path):
        """Ingesting with different embedding dimensions raises error."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        def embed_4d(texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def embed_8d(texts):
            return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

        with SQFox(str(tmp_path / "dim.db")) as db:
            db.ingest("First doc", embed_fn=embed_4d, wait=True)
            with pytest.raises(Exception, match="dimension"):
                db.ingest("Second doc", embed_fn=embed_8d, wait=True)

    def test_search_empty_db(self, tmp_path):
        """Search on empty DB returns empty list, not error."""
        with SQFox(str(tmp_path / "empty.db")) as db:
            db.ensure_schema(SchemaState.SEARCHABLE)
            results = db.search("anything")
            assert results == []


# ---------------------------------------------------------------------------
# Schema auto-migration
# ---------------------------------------------------------------------------

class TestSchemaAutoMigration:
    """Schema should evolve automatically based on operations."""

    def test_ingest_creates_base_schema(self, tmp_path):
        with SQFox(str(tmp_path / "auto.db")) as db:
            db.ingest("Just text, no vectors", wait=True)

            time.sleep(0.05)
            row = db.fetch_one(
                "SELECT name FROM sqlite_master WHERE name = 'documents'"
            )
            assert row is not None

    def test_ingest_with_embed_creates_vec_table(self, tmp_path):
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        with SQFox(str(tmp_path / "autovec.db")) as db:
            db.ingest("Vector doc", embed_fn=mock_embed, wait=True)

            time.sleep(0.05)
            row = db.fetch_one(
                "SELECT name FROM sqlite_master WHERE name = 'documents_vec'"
            )
            assert row is not None


# ---------------------------------------------------------------------------
# Manager end-to-end
# ---------------------------------------------------------------------------

class TestManagerEndToEnd:
    """Multi-database scenario."""

    def test_multi_domain_ingest_and_search(self, tmp_path):
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        with SQFoxManager(tmp_path / "multi") as mgr:
            # IoT domain
            mgr.ingest_to(
                "sensors",
                "Temperature sensor reading 25 degrees",
                embed_fn=mock_embed,
                wait=True,
            )
            mgr.ingest_to(
                "sensors",
                "IoT sensor network monitoring",
                embed_fn=mock_embed,
                wait=True,
            )

            # Knowledge domain
            mgr.ingest_to(
                "knowledge",
                "SQLite database optimization guide",
                embed_fn=mock_embed,
                wait=True,
            )
            mgr.ingest_to(
                "knowledge",
                "Python code deployment best practices",
                embed_fn=mock_embed,
                wait=True,
            )

            time.sleep(0.1)

            # Search across all databases
            results = mgr.search_all("sensor temperature", embed_fn=mock_embed)
            assert len(results) >= 1
            # Top result should be from sensors domain
            db_name, top_result = results[0]
            assert db_name == "sensors"
            assert "sensor" in top_result.text.lower() or "temperature" in top_result.text.lower()

            # Search for database topic — should find results from knowledge
            results = mgr.search_all("database sql", embed_fn=mock_embed)
            assert len(results) >= 1
            knowledge_results = [r for name, r in results if name == "knowledge"]
            assert len(knowledge_results) >= 1

    def test_isolated_databases(self, tmp_path):
        """Documents in one DB are not visible in another."""
        with SQFoxManager(tmp_path / "iso") as mgr:
            mgr.ingest_to("db_a", "Unique content alpha", wait=True)
            mgr.ingest_to("db_b", "Unique content beta", wait=True)

            time.sleep(0.05)

            rows_a = mgr["db_a"].fetch_all("SELECT content FROM documents")
            rows_b = mgr["db_b"].fetch_all("SELECT content FROM documents")

            assert len(rows_a) == 1
            assert "alpha" in rows_a[0][0]
            assert len(rows_b) == 1
            assert "beta" in rows_b[0][0]


# ---------------------------------------------------------------------------
# Concurrent ingest + search
# ---------------------------------------------------------------------------

class TestConcurrentOperations:
    """Writes and reads happening simultaneously."""

    def test_search_during_ingest(self, tmp_path):
        """Search should work while ingest is ongoing."""
        import threading

        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        db_path = str(tmp_path / "concurrent.db")

        with SQFox(db_path) as db:
            # Pre-populate
            for i in range(5):
                db.ingest(
                    f"Document {i} about database configuration",
                    embed_fn=mock_embed,
                    wait=True,
                )

            time.sleep(0.1)

            errors = []
            search_results = []

            # Writer thread — keeps ingesting
            def writer():
                for i in range(10):
                    try:
                        db.ingest(
                            f"New document {i} about server network",
                            embed_fn=mock_embed,
                            wait=True,
                        )
                    except Exception as exc:
                        errors.append(exc)

            # Reader thread — keeps searching
            def reader():
                for _ in range(10):
                    try:
                        results = db.search("database", embed_fn=mock_embed)
                        search_results.append(len(results))
                        time.sleep(0.01)
                    except Exception as exc:
                        errors.append(exc)

            t_write = threading.Thread(target=writer)
            t_read = threading.Thread(target=reader)

            t_write.start()
            t_read.start()

            t_write.join(timeout=30)
            t_read.join(timeout=30)

            assert not errors, f"Errors during concurrent ops: {errors}"
            # Reader should have gotten results at least some of the time
            assert any(r > 0 for r in search_results)


# ---------------------------------------------------------------------------
# Embedder protocol (instruction-aware models like Qwen3, E5, BGE)
# ---------------------------------------------------------------------------

class MockInstructionEmbedder:
    """Simulates an instruction-aware model.

    Documents get plain embeddings, queries get a boosted version.
    This mimics how Qwen3/E5/BGE treat queries differently.
    """

    def __init__(self):
        self.doc_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += 1
        return mock_embed(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        # Boost first dimension to simulate instruction effect
        vec = mock_embed([text])[0]
        vec[0] += 0.1
        return vec


class TestEmbedderProtocol:
    """Embedder with embed_documents/embed_query is dispatched correctly."""

    def test_embedder_object_dispatches_correctly(self, tmp_path):
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        embedder = MockInstructionEmbedder()

        with SQFox(str(tmp_path / "embedder.db")) as db:
            db.ingest(
                "SQLite database configuration",
                embed_fn=embedder,
                wait=True,
            )
            db.ingest(
                "Python server deployment",
                embed_fn=embedder,
                wait=True,
            )

            time.sleep(0.1)

            results = db.search("database", embed_fn=embedder)

            # embed_documents was called during ingest
            assert embedder.doc_calls >= 1
            # embed_query was called during search
            assert embedder.query_calls >= 1
            assert len(results) >= 1

    def test_plain_callable_still_works(self, tmp_path):
        """Backward compatibility: plain function works as before."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        with SQFox(str(tmp_path / "plain.db")) as db:
            db.ingest(
                "SQLite database configuration",
                embed_fn=mock_embed,
                wait=True,
            )

            time.sleep(0.1)

            results = db.search("database", embed_fn=mock_embed)
            assert len(results) >= 1


# ---------------------------------------------------------------------------
# Reranker end-to-end
# ---------------------------------------------------------------------------

class TestRerankerEndToEnd:
    """Reranker integration through SQFox.search()."""

    def test_reranker_changes_order(self, tmp_path):
        """reranker_fn should re-score and reorder results."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        with SQFox(str(tmp_path / "rerank.db")) as db:
            db.ingest("Alpha document about databases", embed_fn=mock_embed, wait=True)
            db.ingest("Beta document about servers", embed_fn=mock_embed, wait=True)
            db.ingest("Gamma document about databases and servers", embed_fn=mock_embed, wait=True)
            time.sleep(0.1)

            def server_reranker(query, texts):
                return [10.0 if "servers" in t else 1.0 for t in texts]

            results = db.search(
                "database",
                embed_fn=mock_embed,
                reranker_fn=server_reranker,
                limit=3,
            )
            assert len(results) >= 1
            assert "servers" in results[0].text.lower()

    def test_reranker_with_rerank_top_n(self, tmp_path):
        """rerank_top_n controls how many candidates the reranker sees."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        seen = []

        def tracking_reranker(query, texts):
            seen.append(len(texts))
            return [1.0] * len(texts)

        with SQFox(str(tmp_path / "topn.db")) as db:
            for i in range(10):
                db.ingest(f"Document {i} about testing", embed_fn=mock_embed, wait=True)
            time.sleep(0.1)

            db.search(
                "testing",
                embed_fn=mock_embed,
                reranker_fn=tracking_reranker,
                limit=2,
                rerank_top_n=5,
            )
            assert seen
            assert seen[0] <= 5
