"""Tests for sqfox search: normalization, fusion, RRF, adaptive alpha."""

import pytest

from sqfox.search import (
    _min_max_normalize,
    score_fusion,
    rrf_fallback,
    adaptive_alpha,
    _std,
)


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

class TestMinMaxNormalize:
    def test_basic_normalization(self):
        results = [(1, 10.0), (2, 20.0), (3, 30.0)]
        norm = _min_max_normalize(results)
        assert norm[1] == pytest.approx(0.0)
        assert norm[2] == pytest.approx(0.5)
        assert norm[3] == pytest.approx(1.0)

    def test_identical_scores(self):
        results = [(1, 5.0), (2, 5.0), (3, 5.0)]
        norm = _min_max_normalize(results)
        # All-same scores → neutral 0.5 to avoid inflating importance
        assert all(v == 0.5 for v in norm.values())

    def test_empty(self):
        assert _min_max_normalize([]) == {}

    def test_single_result(self):
        results = [(1, 42.0)]
        norm = _min_max_normalize(results)
        assert norm[1] == 0.5

    def test_two_results(self):
        results = [(1, 0.0), (2, 1.0)]
        norm = _min_max_normalize(results)
        assert norm[1] == pytest.approx(0.0)
        assert norm[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

class TestScoreFusion:
    def test_balanced(self):
        fts = [(1, 10.0), (2, 5.0)]
        vec = [(1, 0.8), (2, 0.6)]
        result = score_fusion(fts, vec, alpha=0.5)
        # Doc 1 should rank first (highest in both)
        assert result[0][0] == 1

    def test_fts_only_alpha_zero(self):
        fts = [(1, 10.0), (2, 5.0)]
        vec = [(2, 0.9), (1, 0.1)]
        result = score_fusion(fts, vec, alpha=0.0)
        # With alpha=0, only FTS matters
        assert result[0][0] == 1

    def test_vec_only_alpha_one(self):
        fts = [(1, 10.0), (2, 5.0)]
        vec = [(2, 0.9), (1, 0.1)]
        result = score_fusion(fts, vec, alpha=1.0)
        # With alpha=1, only vec matters
        assert result[0][0] == 2

    def test_no_overlap(self):
        fts = [(1, 10.0)]
        vec = [(2, 0.8)]
        result = score_fusion(fts, vec, alpha=0.5)
        # Both docs should appear
        doc_ids = [r[0] for r in result]
        assert 1 in doc_ids
        assert 2 in doc_ids

    def test_one_empty(self):
        fts = [(1, 10.0), (2, 5.0)]
        vec = []
        result = score_fusion(fts, vec, alpha=0.5)
        # Only FTS results, but weighted by (1-alpha)
        assert len(result) == 2
        assert result[0][0] == 1

    def test_scores_in_zero_one_range(self):
        fts = [(1, 100.0), (2, 50.0), (3, 10.0)]
        vec = [(1, 0.95), (3, 0.80), (4, 0.70)]
        result = score_fusion(fts, vec, alpha=0.5)
        for _, score in result:
            assert 0.0 <= score <= 1.0

    def test_both_empty(self):
        result = score_fusion([], [])
        assert result == []


# ---------------------------------------------------------------------------
# RRF fallback
# ---------------------------------------------------------------------------

class TestRRF:
    def test_basic(self):
        fts = [(1, 10.0), (2, 5.0)]
        vec = [(1, 0.8), (3, 0.5)]
        result = rrf_fallback(fts, vec, k=60)
        # Doc 1 appears in both -> highest RRF score
        assert result[0][0] == 1

    def test_overlapping_boosts_rank(self):
        fts = [(1, 10.0), (2, 5.0)]
        vec = [(2, 0.9), (3, 0.5)]
        result = rrf_fallback(fts, vec, k=60)
        # Doc 2 in both lists -> should rank high
        doc_ids = [r[0] for r in result]
        # Doc 2 has RRF from rank 1 in FTS + rank 0 in vec
        # Doc 1 has RRF only from rank 0 in FTS
        # Since 1/(60+1) + 1/(60+2) > 1/(60+1), doc 2 should not beat doc 1
        # Actually doc 1: 1/61 = 0.01639
        # Doc 2: 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
        # So doc 2 wins!
        assert result[0][0] == 2

    def test_different_k(self):
        fts = [(1, 10.0), (2, 5.0)]
        vec = [(1, 0.8)]
        result_k2 = rrf_fallback(fts, vec, k=2)
        result_k100 = rrf_fallback(fts, vec, k=100)
        # With low k, rank differences matter more
        # Doc 1 should be top in both, but scores differ
        assert result_k2[0][0] == 1
        assert result_k100[0][0] == 1

    def test_rrf_empty_inputs(self):
        assert rrf_fallback([], []) == []
        assert len(rrf_fallback([(1, 0.5)], [])) == 1
        assert len(rrf_fallback([], [(1, 0.5)])) == 1

    def test_alpha_weighting(self):
        """alpha=0 means FTS-only, alpha=1 means vec-only."""
        fts = [(1, 10.0), (2, 5.0)]
        vec = [(3, 0.9), (4, 0.5)]
        # Alpha=0: only FTS results should appear with non-zero scores
        result_fts = rrf_fallback(fts, vec, alpha=0.0)
        fts_ids = {r[0] for r in result_fts if r[1] > 0}
        assert fts_ids == {1, 2}
        # Alpha=1: only vec results should appear with non-zero scores
        result_vec = rrf_fallback(fts, vec, alpha=1.0)
        vec_ids = {r[0] for r in result_vec if r[1] > 0}
        assert vec_ids == {3, 4}


# ---------------------------------------------------------------------------
# Adaptive alpha
# ---------------------------------------------------------------------------

class TestAdaptiveAlpha:
    def test_code_query_decreases_alpha(self):
        query = "camelCase function_name()"
        fts = [(i, 10.0 - i) for i in range(5)]
        vec = [(i, 0.9 - i * 0.1) for i in range(5)]
        alpha = adaptive_alpha(query, fts, vec, base_alpha=0.5)
        assert alpha < 0.5

    def test_question_query_increases_alpha(self):
        query = "how to configure database"
        # Use identical score distributions so std ratio doesn't adjust
        fts = [(i, 10.0 - i * 2.0) for i in range(5)]
        vec = [(i, 10.0 - i * 2.0) for i in range(5)]
        alpha = adaptive_alpha(query, fts, vec, base_alpha=0.5)
        assert alpha > 0.5

    def test_empty_fts_favors_vectors(self):
        query = "test query"
        fts = [(1, 5.0)]  # Only 1 result
        vec = [(i, 0.9 - i * 0.1) for i in range(5)]
        alpha = adaptive_alpha(query, fts, vec, base_alpha=0.5)
        assert alpha >= 0.8

    def test_empty_vec_favors_fts(self):
        query = "test query"
        fts = [(i, 10.0 - i) for i in range(5)]
        vec = [(1, 0.5)]  # Only 1 result
        alpha = adaptive_alpha(query, fts, vec, base_alpha=0.5)
        assert alpha <= 0.2

    def test_clamped_to_range(self):
        query = "how to camelCase function_name() test.run()"
        fts = [(1, 5.0)]
        vec = [(i, 0.9 - i * 0.1) for i in range(5)]
        alpha = adaptive_alpha(query, fts, vec, base_alpha=0.0)
        assert 0.0 <= alpha <= 1.0

    def test_normal_query_stays_near_base(self):
        query = "sqlite database configuration"
        fts = [(i, 10.0 - i) for i in range(5)]
        vec = [(i, 0.9 - i * 0.1) for i in range(5)]
        alpha = adaptive_alpha(query, fts, vec, base_alpha=0.5)
        assert 0.3 <= alpha <= 0.7

    def test_adaptive_alpha_boundary_values(self):
        """Alpha stays in [0, 1] range with extreme inputs."""
        fts = [(i, float(i)) for i in range(10)]
        vec = [(i, float(i)) for i in range(10)]
        a = adaptive_alpha("test", fts, vec, base_alpha=0.0)
        assert 0.0 <= a <= 1.0
        a = adaptive_alpha("test", fts, vec, base_alpha=1.0)
        assert 0.0 <= a <= 1.0


# ---------------------------------------------------------------------------
# Standard deviation helper
# ---------------------------------------------------------------------------

class TestStd:
    def test_basic(self):
        values = [2.0, 4.0, 6.0, 8.0]
        # Mean = 5, variance = (9+1+1+9)/4 = 5, std = sqrt(5) ≈ 2.236
        assert _std(values) == pytest.approx(2.236, abs=0.01)

    def test_identical_values(self):
        assert _std([5.0, 5.0, 5.0]) == 0.0

    def test_single_value(self):
        assert _std([5.0]) == 0.0

    def test_empty(self):
        assert _std([]) == 0.0


# ---------------------------------------------------------------------------
# Reranker in hybrid_search
# ---------------------------------------------------------------------------

class TestReranker:
    """Tests for reranker_fn integration in hybrid_search."""

    def _setup_db(self, tmp_path):
        """Create a DB with documents for search."""
        import sqlite3
        from sqfox.schema import migrate_to
        from sqfox.types import SchemaState

        db_path = str(tmp_path / "rerank.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        migrate_to(conn, SchemaState.SEARCHABLE)

        docs = [
            ("Python is great for scripting", "python great script"),
            ("Java is used in enterprise", "java use enterprise"),
            ("Rust is fast and safe", "rust fast safe"),
            ("Go is simple and concurrent", "go simple concurrent"),
            ("Python web frameworks rock", "python web framework rock"),
        ]
        for content, lemmatized in docs:
            cursor = conn.execute(
                "INSERT INTO documents (content, content_lemmatized) VALUES (?, ?)",
                (content, lemmatized),
            )
            doc_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO documents_fts(rowid, content_lemmatized) VALUES (?, ?)",
                (doc_id, lemmatized),
            )
            conn.execute(
                "UPDATE documents SET fts_indexed = 1 WHERE id = ?",
                (doc_id,),
            )
        conn.commit()
        return conn

    def test_reranker_reorders_results(self, tmp_path):
        """Reranker should reorder candidates by its scores."""
        from sqfox.search import hybrid_search

        conn = self._setup_db(tmp_path)

        def dummy_embed(texts):
            return [[0.0] * 4 for _ in texts]

        def reversing_reranker(query, texts):
            """Assign ascending scores so last candidate becomes first."""
            return [float(i) for i in range(len(texts))]

        results_no_rerank = hybrid_search(
            conn, "python", dummy_embed, limit=5,
        )
        results_reranked = hybrid_search(
            conn, "python", dummy_embed, limit=5,
            reranker_fn=reversing_reranker,
        )

        assert len(results_no_rerank) >= 2, "FTS should find at least 2 python docs"
        assert len(results_reranked) >= 2, "Reranked should also have at least 2"

        no_rerank_ids = [r.doc_id for r in results_no_rerank]
        reranked_ids = [r.doc_id for r in results_reranked]
        # With reversing reranker, the order must be reversed
        assert reranked_ids == list(reversed(no_rerank_ids)), (
            f"Reranker should reverse the order: "
            f"original={no_rerank_ids}, reranked={reranked_ids}"
        )

        conn.close()

    def test_reranker_scores_replace_fusion_scores(self, tmp_path):
        """Results should have reranker scores, not fusion scores."""
        from sqfox.search import hybrid_search

        conn = self._setup_db(tmp_path)

        def dummy_embed(texts):
            return [[0.0] * 4 for _ in texts]

        def fixed_reranker(query, texts):
            return [99.0 + i for i in range(len(texts))]

        results = hybrid_search(
            conn, "python", dummy_embed, limit=5,
            reranker_fn=fixed_reranker,
        )

        assert len(results) >= 1, "Search should return results"
        assert all(r.score >= 99.0 for r in results)

        conn.close()

    def test_reranker_failure_falls_back(self, tmp_path):
        """If reranker raises, fall back to fusion scores."""
        from sqfox.search import hybrid_search

        conn = self._setup_db(tmp_path)

        def dummy_embed(texts):
            return [[0.0] * 4 for _ in texts]

        def broken_reranker(query, texts):
            raise RuntimeError("reranker exploded")

        results = hybrid_search(
            conn, "python", dummy_embed, limit=5,
            reranker_fn=broken_reranker,
        )
        assert len(results) > 0
        conn.close()

    def test_reranker_wrong_length_ignored(self, tmp_path):
        """If reranker returns wrong number of scores, skip reranking."""
        from sqfox.search import hybrid_search

        conn = self._setup_db(tmp_path)

        def dummy_embed(texts):
            return [[0.0] * 4 for _ in texts]

        def bad_length_reranker(query, texts):
            return [1.0]

        results = hybrid_search(
            conn, "python", dummy_embed, limit=5,
            reranker_fn=bad_length_reranker,
        )
        assert len(results) > 0
        assert all(r.score <= 1.0 for r in results)
        conn.close()

    def test_rerank_top_n_controls_pool_size(self, tmp_path):
        """rerank_top_n limits how many candidates the reranker sees."""
        from sqfox.search import hybrid_search

        conn = self._setup_db(tmp_path)

        def dummy_embed(texts):
            return [[0.0] * 4 for _ in texts]

        seen_counts = []

        def counting_reranker(query, texts):
            seen_counts.append(len(texts))
            return [1.0] * len(texts)

        hybrid_search(
            conn, "python", dummy_embed, limit=1,
            reranker_fn=counting_reranker, rerank_top_n=3,
        )

        assert seen_counts
        assert seen_counts[0] <= 3
        conn.close()


# ---------------------------------------------------------------------------
# fts_search / vec_search unit tests
# ---------------------------------------------------------------------------

class TestFtsSearch:
    """Direct tests for fts_search (bypassing hybrid_search)."""

    def _make_fts_db(self, tmp_path):
        import sqlite3
        from sqfox.schema import migrate_to
        from sqfox.types import SchemaState

        db_path = str(tmp_path / "fts_unit.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        migrate_to(conn, SchemaState.SEARCHABLE)

        docs = [
            (1, "python programming language", "python program language"),
            (2, "java enterprise development", "java enterprise develop"),
            (3, "python web framework flask", "python web framework flask"),
        ]
        for doc_id, content, lemmatized in docs:
            conn.execute(
                "INSERT INTO documents (id, content, content_lemmatized, fts_indexed) "
                "VALUES (?, ?, ?, 1)",
                (doc_id, content, lemmatized),
            )
            conn.execute(
                "INSERT INTO documents_fts(rowid, content_lemmatized) VALUES (?, ?)",
                (doc_id, lemmatized),
            )
        conn.commit()
        return conn

    def test_fts_basic_match(self, tmp_path):
        from sqfox.search import fts_search
        conn = self._make_fts_db(tmp_path)
        results = fts_search(conn, "python", limit=10)
        assert len(results) == 2
        doc_ids = [r[0] for r in results]
        assert 1 in doc_ids
        assert 3 in doc_ids
        conn.close()

    def test_fts_no_match(self, tmp_path):
        from sqfox.search import fts_search
        conn = self._make_fts_db(tmp_path)
        results = fts_search(conn, "rust", limit=10)
        assert results == []
        conn.close()

    def test_fts_empty_query(self, tmp_path):
        from sqfox.search import fts_search
        conn = self._make_fts_db(tmp_path)
        assert fts_search(conn, "", limit=10) == []
        assert fts_search(conn, "   ", limit=10) == []
        conn.close()

    def test_fts_special_chars(self, tmp_path):
        from sqfox.search import fts_search
        conn = self._make_fts_db(tmp_path)
        # These should not crash FTS5
        # Quotes are stripped by sanitizer, so "python" becomes python
        results = fts_search(conn, '"python"', limit=10)
        assert len(results) == 2, f"Quoted query should match 2 docs, got {len(results)}"
        # Asterisk is stripped by sanitizer, so python* becomes python
        results = fts_search(conn, "python*", limit=10)
        assert len(results) == 2, f"Prefix query should match 2 docs, got {len(results)}"
        results = fts_search(conn, "---", limit=10)
        assert results == []
        conn.close()

    def test_fts_limit(self, tmp_path):
        from sqfox.search import fts_search
        conn = self._make_fts_db(tmp_path)
        results = fts_search(conn, "python", limit=1)
        assert len(results) == 1
        conn.close()

    def test_fts_scores_positive(self, tmp_path):
        from sqfox.search import fts_search
        conn = self._make_fts_db(tmp_path)
        results = fts_search(conn, "python", limit=10)
        for _, score in results:
            assert score > 0, "FTS scores should be positive (negated BM25)"
        conn.close()


class TestVecSearch:
    """Direct tests for vec_search with a Flat backend."""

    def _make_flat_backend(self):
        from sqfox.backends.flat import SqliteFlatBackend
        backend = SqliteFlatBackend()
        backend.initialize(":memory:", 4)
        vecs = [
            [1.0, 0.0, 0.0, 0.0],  # unit x
            [0.0, 1.0, 0.0, 0.0],  # unit y
            [0.7, 0.7, 0.0, 0.0],  # between x and y
        ]
        backend.add([1, 2, 3], vecs)
        return backend

    def test_vec_basic_search(self):
        from sqfox.search import vec_search
        import sqlite3
        backend = self._make_flat_backend()
        conn = sqlite3.connect(":memory:")
        results = vec_search(conn, [1.0, 0.0, 0.0, 0.0], limit=3, vector_backend=backend)
        assert len(results) == 3
        # Closest to [1,0,0,0] should be doc 1
        assert results[0][0] == 1
        conn.close()
        backend.close()

    def test_vec_scores_positive(self):
        from sqfox.search import vec_search
        import sqlite3
        backend = self._make_flat_backend()
        conn = sqlite3.connect(":memory:")
        results = vec_search(conn, [1.0, 0.0, 0.0, 0.0], limit=3, vector_backend=backend)
        for _, score in results:
            assert 0 < score <= 1.0, f"Vec score should be in (0, 1], got {score}"
        conn.close()
        backend.close()

    def test_vec_limit(self):
        from sqfox.search import vec_search
        import sqlite3
        backend = self._make_flat_backend()
        conn = sqlite3.connect(":memory:")
        results = vec_search(conn, [1.0, 0.0, 0.0, 0.0], limit=1, vector_backend=backend)
        assert len(results) == 1
        conn.close()
        backend.close()

    def test_vec_no_backend_returns_empty(self):
        from sqfox.search import vec_search
        import sqlite3
        conn = sqlite3.connect(":memory:")
        results = vec_search(conn, [1.0, 0.0, 0.0, 0.0], limit=3)
        assert results == []
        conn.close()
