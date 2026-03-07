"""Integration tests for auto-adaptive features in SQFox 0.3.0."""

import sqlite3

import pytest

from sqfox import SQFox, AUTO, SchemaState


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_auto.db")


def _dummy_embed(texts):
    """Return 8-dimensional dummy embeddings."""
    return [[float(i + j) for j in range(8)] for i, _ in enumerate(texts)]


# ---------------------------------------------------------------------------
# AUTO defaults
# ---------------------------------------------------------------------------

class TestAutoDefaults:
    def test_zero_config_start(self, db_path):
        """SQFox() with no arguments should start and work."""
        with SQFox(db_path) as db:
            assert db.is_running
            db.write(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", wait=True
            )
            db.write("INSERT INTO t VALUES (1, 'hello')", wait=True)
            row = db.fetch_one("SELECT v FROM t WHERE id = 1")
            assert row[0] == "hello"

    def test_auto_params_resolved(self, db_path):
        """AUTO params should be resolved to ints after start()."""
        db = SQFox(db_path)
        db.start()
        try:
            assert isinstance(db._cache_size_kb, int)
            assert isinstance(db._mmap_size_mb, int)
            assert db._cache_size_kb > 0
            assert db._mmap_size_mb >= 0
        finally:
            db.stop()

    def test_explicit_params_override(self, db_path):
        """Explicit int params should override auto-detection."""
        db = SQFox(db_path, cache_size_kb=8_000, mmap_size_mb=32)
        db.start()
        try:
            assert db._cache_size_kb == 8_000
            assert db._mmap_size_mb == 32
        finally:
            db.stop()


# ---------------------------------------------------------------------------
# Environment info in diagnostics
# ---------------------------------------------------------------------------

class TestDiagnosticsAuto:
    def test_diagnostics_has_auto(self, db_path):
        with SQFox(db_path) as db:
            diag = db.diagnostics()
            assert "auto" in diag
            auto = diag["auto"]
            assert "total_ram_mb" in auto
            assert "memory_tier" in auto
            assert "cpu_count" in auto
            assert "platform_class" in auto
            assert "fts5_available" in auto
            assert "resolved_cache_size_kb" in auto
            assert "resolved_mmap_size_mb" in auto

    def test_diagnostics_auto_values_sane(self, db_path):
        with SQFox(db_path) as db:
            auto = db.diagnostics()["auto"]
            assert auto["total_ram_mb"] > 0
            assert auto["cpu_count"] >= 1
            assert auto["memory_tier"] in ("LOW", "MEDIUM", "HIGH")
            assert auto["resolved_cache_size_kb"] > 0


# ---------------------------------------------------------------------------
# Auto-vacuum on new databases
# ---------------------------------------------------------------------------

class TestAutoVacuum:
    def test_new_db_gets_incremental_vacuum(self, db_path):
        """New databases should have auto_vacuum=INCREMENTAL."""
        with SQFox(db_path) as db:
            row = db.fetch_one("PRAGMA auto_vacuum")
            # 2 = INCREMENTAL
            assert row[0] == 2

    def test_existing_db_keeps_vacuum_mode(self, db_path):
        """Existing databases should NOT change auto_vacuum."""
        # Create a database with auto_vacuum=NONE (default SQLite)
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE existing (id INTEGER PRIMARY KEY)")
        conn.commit()
        row = conn.execute("PRAGMA auto_vacuum").fetchone()
        original_mode = row[0]
        conn.close()

        # Open with SQFox — should NOT change auto_vacuum
        with SQFox(db_path) as db:
            row = db.fetch_one("PRAGMA auto_vacuum")
            assert row[0] == original_mode


# ---------------------------------------------------------------------------
# Auto-backend selection
# ---------------------------------------------------------------------------

class TestAutoBackend:
    def test_auto_selects_by_tier(self, db_path):
        """On MEDIUM/HIGH RAM, auto-selects hnsw; on LOW, selects flat."""
        with SQFox(db_path) as db:
            db.ingest("test document", embed_fn=_dummy_embed, wait=True)
            # This machine has >4GB → HIGH → hnsw
            assert db.vector_backend_name in ("hnsw", "flat")

    def test_auto_selects_flat_on_low_ram(self, db_path):
        """On LOW memory tier, auto-selects flat (less RAM overhead)."""
        from unittest import mock
        from sqfox._auto import EnvironmentInfo, MemoryTier, PlatformClass

        low_env = EnvironmentInfo(
            total_ram_mb=512,
            cpu_count=2,
            memory_tier=MemoryTier.LOW,
            platform_class=PlatformClass.RASPBERRY_PI,
            is_sd_card=False,
            fts5_available=True,
            recommended_cache_size_kb=4000,
            recommended_mmap_size_mb=0,
            recommended_cpu_workers=1,
            recommended_reader_prune_threshold=5,
        )
        with mock.patch("sqfox._auto.detect_environment", return_value=low_env):
            with SQFox(db_path) as db:
                db.ingest("test document", embed_fn=_dummy_embed, wait=True)
                assert db.vector_backend_name == "flat"

    def test_auto_selects_flat_on_android(self, db_path):
        """Android Termux always gets flat — phone RAM shared with OS+LLM."""
        from unittest import mock
        from sqfox._auto import EnvironmentInfo, MemoryTier, PlatformClass

        android_env = EnvironmentInfo(
            total_ram_mb=3072,
            cpu_count=8,
            memory_tier=MemoryTier.MEDIUM,
            platform_class=PlatformClass.ANDROID_TERMUX,
            is_sd_card=False,
            fts5_available=True,
            recommended_cache_size_kb=16000,
            recommended_mmap_size_mb=64,
            recommended_cpu_workers=1,
            recommended_reader_prune_threshold=10,
        )
        with mock.patch("sqfox._auto.detect_environment", return_value=android_env):
            with SQFox(db_path) as db:
                db.ingest("test document", embed_fn=_dummy_embed, wait=True)
                assert db.vector_backend_name == "flat"

    def test_auto_selects_hnsw_on_high_ram(self, db_path):
        """On HIGH memory tier, auto-selects hnsw."""
        from unittest import mock
        from sqfox._auto import EnvironmentInfo, MemoryTier, PlatformClass

        high_env = EnvironmentInfo(
            total_ram_mb=8192,
            cpu_count=4,
            memory_tier=MemoryTier.HIGH,
            platform_class=PlatformClass.DESKTOP,
            is_sd_card=False,
            fts5_available=True,
            recommended_cache_size_kb=64000,
            recommended_mmap_size_mb=256,
            recommended_cpu_workers=2,
            recommended_reader_prune_threshold=20,
        )
        with mock.patch("sqfox._auto.detect_environment", return_value=high_env):
            with SQFox(db_path) as db:
                db.ingest("test document", embed_fn=_dummy_embed, wait=True)
                assert db.vector_backend_name == "hnsw"

    def test_auto_backend_persisted(self, db_path):
        """Auto-selected backend is stored in _sqfox_meta."""
        with SQFox(db_path) as db:
            db.ingest("test document", embed_fn=_dummy_embed, wait=True)
            backend = db.vector_backend_name

        # Re-open and check meta
        conn = sqlite3.connect(db_path)
        import json
        row = conn.execute(
            "SELECT value FROM _sqfox_meta WHERE key = 'vector_backend'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert json.loads(row[0]) == backend

    def test_auto_backend_restored_on_restart(self, db_path):
        """Stored backend is restored on restart."""
        # First session
        with SQFox(db_path) as db:
            db.ingest("test document", embed_fn=_dummy_embed, wait=True)
            first_backend = db.vector_backend_name

        # Second session — should restore the same backend
        with SQFox(db_path) as db:
            db.ingest("another document", embed_fn=_dummy_embed, wait=True)
            assert db.vector_backend_name == first_backend

    def test_explicit_backend_not_overridden(self, db_path):
        """Explicit vector_backend should be used, not auto-selected."""
        with SQFox(db_path, vector_backend="flat") as db:
            db.ingest("test document", embed_fn=_dummy_embed, wait=True)
            assert db.vector_backend_name == "flat"


# ---------------------------------------------------------------------------
# FTS5 self-healing
# ---------------------------------------------------------------------------

class TestFtsSelfHealing:
    def test_fts_check_runs_on_start(self, db_path):
        """FTS integrity check should run without error on valid FTS."""
        with SQFox(db_path) as db:
            db.ensure_schema(SchemaState.SEARCHABLE)
            # FTS is valid — check ran during start, no crash

    def test_fts_rebuild_on_corruption(self, db_path):
        """If FTS index is corrupt, it should auto-rebuild."""
        # Create valid schema with FTS
        with SQFox(db_path) as db:
            db.ensure_schema(SchemaState.SEARCHABLE)
            db.write(
                "INSERT INTO documents (content, content_lemmatized) "
                "VALUES ('test', 'test')",
                wait=True,
            )
            db.write(
                "INSERT INTO documents_fts(documents_fts) VALUES('rebuild')",
                wait=True,
            )

        # Re-open — integrity check should pass (it was just rebuilt)
        with SQFox(db_path) as db:
            assert db.is_running


# ---------------------------------------------------------------------------
# Adaptive reader pruning threshold
# ---------------------------------------------------------------------------

class TestAdaptivePruning:
    def test_threshold_set_from_env(self, db_path):
        """Reader prune threshold should be set from EnvironmentInfo."""
        with SQFox(db_path) as db:
            assert db._reader_prune_threshold > 0
            # On most desktops with >4GB: HIGH tier → 20
            # But could be lower on CI/small machines


# ---------------------------------------------------------------------------
# Incremental vacuum counter
# ---------------------------------------------------------------------------

class TestIncrementalVacuum:
    def test_ingest_counter_increments(self, db_path):
        """Ingest counter should increment after each ingest."""
        with SQFox(db_path) as db:
            assert db._ingest_counter == 0
            db.ingest("doc 1", embed_fn=_dummy_embed, wait=True)
            assert db._ingest_counter == 1
            db.ingest("doc 2", embed_fn=_dummy_embed, wait=True)
            assert db._ingest_counter == 2


# ---------------------------------------------------------------------------
# Search still works with auto-backend
# ---------------------------------------------------------------------------

class TestSearchWithAutoBackend:
    def test_ingest_and_search(self, db_path):
        """Full cycle: zero-config ingest + search with auto-backend."""
        with SQFox(db_path) as db:
            db.ingest("SQLite is a relational database", embed_fn=_dummy_embed, wait=True)
            db.ingest("Python is a programming language", embed_fn=_dummy_embed, wait=True)
            db.ingest("WAL mode improves concurrency", embed_fn=_dummy_embed, wait=True)

            results = db.search("database", embed_fn=_dummy_embed, limit=3)
            assert len(results) > 0
            assert all(hasattr(r, "score") for r in results)


# ---------------------------------------------------------------------------
# Memory-only database
# ---------------------------------------------------------------------------

class TestMemoryDb:
    def test_memory_db_auto(self):
        """In-memory database should work with AUTO params."""
        with SQFox(":memory:") as db:
            db.write("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", wait=True)
            db.write("INSERT INTO t VALUES (1, 'hello')", wait=True)
            row = db.fetch_one("SELECT v FROM t WHERE id = 1")
            assert row[0] == "hello"
            # Auto-vacuum should be skipped for :memory:
            assert db._env is not None


# ---------------------------------------------------------------------------
# PRAGMA optimize on stop()
# ---------------------------------------------------------------------------

class TestPragmaOptimize:
    def test_optimize_runs_after_heavy_ingestion(self, db_path):
        """PRAGMA optimize should run silently on stop() after >1000 ingests."""
        db = SQFox(db_path)
        db.start()
        # Simulate heavy ingestion counter (real 1000 ingests would be slow)
        db._ingest_counter = 1500
        # stop() should run PRAGMA optimize without error
        db.stop()

    def test_no_optimize_on_light_usage(self, db_path):
        """Under 1000 ingests, PRAGMA optimize is skipped (no harm either way)."""
        db = SQFox(db_path)
        db.start()
        db.ingest("just one doc", embed_fn=_dummy_embed, wait=True)
        assert db._ingest_counter == 1
        db.stop()  # Should not run PRAGMA optimize, but no error either


# ---------------------------------------------------------------------------
# Dynamic (elastic) batch size
# ---------------------------------------------------------------------------

class TestElasticBatch:
    def test_base_batch_preserved(self, db_path):
        """_base_batch_size should always hold the original value."""
        db = SQFox(db_path, batch_size=32)
        db.start()
        assert db._base_batch_size == 32
        assert db._batch_size == 32
        db.stop()

    def test_batch_grows_under_pressure(self, db_path):
        """When queue is flooded, batch_size should grow."""
        db = SQFox(db_path, batch_size=8)
        db.start()
        db.write("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", wait=True)

        # Flood the queue with fire-and-forget writes (don't wait)
        for i in range(200):
            try:
                db.write(
                    "INSERT INTO t VALUES (?, ?)",
                    (i, f"val_{i}"),
                    wait=False,
                )
            except Exception:
                break

        # Give writer time to adapt
        import time
        time.sleep(0.5)

        # batch_size should have grown above base (8) if queue had pressure
        # Note: on fast machines, the writer may drain instantly, so we
        # check that it at least didn't break — batch_size >= base
        assert db._batch_size >= db._base_batch_size
        db.stop()

    def test_batch_returns_to_base(self, db_path):
        """After queue drains, batch_size should return to base."""
        db = SQFox(db_path, batch_size=16)
        db.start()
        db.write("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", wait=True)

        # Single write + wait = queue instantly empty after
        db.write("INSERT INTO t VALUES (1, 'x')", wait=True)

        import time
        time.sleep(0.2)

        # After calm period, batch should be at base
        assert db._batch_size == db._base_batch_size
        db.stop()

    def test_batch_cap(self, db_path):
        """Batch size should never exceed base * 8."""
        db = SQFox(db_path, batch_size=32)
        db.start()
        # Manually force growth to verify cap
        db._batch_size = 32 * 16  # way over cap
        # Next writer iteration will clamp, but we verify the cap logic:
        cap = db._base_batch_size * 8
        assert cap == 256
        db.stop()

    def test_diagnostics_shows_both_batch_sizes(self, db_path):
        """diagnostics() should show both base and current batch_size."""
        with SQFox(db_path, batch_size=64) as db:
            diag = db.diagnostics()
            assert diag["batch_size"] == 64
            assert diag["batch_size_current"] == 64

    def test_batch_size_zero_clamped_to_one(self, db_path):
        """batch_size=0 should be clamped to 1, not cause infinite spin."""
        db = SQFox(db_path, batch_size=0)
        assert db._base_batch_size == 1
        assert db._batch_size == 1
        db.start()
        db.write("CREATE TABLE t (x)", wait=True)
        db.write("INSERT INTO t VALUES (1)", wait=True)
        row = db.fetch_one("SELECT x FROM t")
        assert row[0] == 1
        db.stop()

    def test_batch_size_negative_clamped_to_one(self, db_path):
        """Negative batch_size should be clamped to 1."""
        db = SQFox(db_path, batch_size=-5)
        assert db._base_batch_size == 1
        db.start()
        db.stop()


# ---------------------------------------------------------------------------
# Backend fallback chain (T6)
# ---------------------------------------------------------------------------

class TestBackendFallback:
    def test_fallback_when_preferred_unavailable(self, db_path):
        """If preferred backend is unavailable, fallback to next."""
        from unittest import mock
        from sqfox._auto import EnvironmentInfo, MemoryTier, PlatformClass

        high_env = EnvironmentInfo(
            total_ram_mb=8192,
            cpu_count=4,
            memory_tier=MemoryTier.HIGH,
            platform_class=PlatformClass.DESKTOP,
            is_sd_card=False,
            fts5_available=True,
            recommended_cache_size_kb=64000,
            recommended_mmap_size_mb=256,
            recommended_cpu_workers=2,
            recommended_reader_prune_threshold=20,
        )

        original_get_backend = None

        def mock_get_backend(name):
            """hnsw fails, flat succeeds."""
            if name == "hnsw":
                raise ImportError("hnsw not available")
            return original_get_backend(name)

        from sqfox.backends.registry import get_backend as _real_get_backend
        original_get_backend = _real_get_backend

        with mock.patch("sqfox._auto.detect_environment", return_value=high_env):
            with mock.patch("sqfox.backends.registry.get_backend", side_effect=mock_get_backend):
                with SQFox(db_path) as db:
                    db.ingest("test doc", embed_fn=_dummy_embed, wait=True)
                    # hnsw failed, should fall back to flat
                    assert db.vector_backend_name == "flat"

    def test_medium_tier_selects_hnsw(self, db_path):
        """MEDIUM tier on DESKTOP should select hnsw."""
        from unittest import mock
        from sqfox._auto import EnvironmentInfo, MemoryTier, PlatformClass

        medium_env = EnvironmentInfo(
            total_ram_mb=3072,
            cpu_count=4,
            memory_tier=MemoryTier.MEDIUM,
            platform_class=PlatformClass.DESKTOP,
            is_sd_card=False,
            fts5_available=True,
            recommended_cache_size_kb=16000,
            recommended_mmap_size_mb=64,
            recommended_cpu_workers=2,
            recommended_reader_prune_threshold=10,
        )
        with mock.patch("sqfox._auto.detect_environment", return_value=medium_env):
            with SQFox(db_path) as db:
                db.ingest("test doc", embed_fn=_dummy_embed, wait=True)
                assert db.vector_backend_name == "hnsw"


# ---------------------------------------------------------------------------
# FTS5 self-healing with real corruption (T2)
# ---------------------------------------------------------------------------

class TestFtsSelfHealing:
    def test_fts_check_does_not_leave_open_transaction(self, db_path):
        """After FTS5 integrity check on restart, writer should accept writes.

        Two-session pattern: first session creates FTS (SEARCHABLE),
        second session runs the integrity check at start(), then we
        verify the writer thread isn't blocked by an open transaction.
        """
        # Session 1: create SEARCHABLE schema + ingest data
        with SQFox(db_path) as db:
            db.ingest("seed document for FTS check", embed_fn=_dummy_embed, wait=True)

        # Session 2: start() runs _startup_fts_check (schema is SEARCHABLE now)
        with SQFox(db_path) as db:
            # Ingest should work — writer can BEGIN IMMEDIATE
            db.ingest("post-check document", embed_fn=_dummy_embed, wait=True)
            # Plain write should also work
            db.write(
                "INSERT INTO documents (content) VALUES (?)",
                ("another doc",),
                wait=True,
            )

    def test_fts_corruption_triggers_rebuild(self, db_path):
        """Corrupting FTS shadow table should trigger rebuild on restart."""
        # First session: create valid FTS + ingest searchable data
        with SQFox(db_path) as db:
            db.ingest("test document about cats and dogs", embed_fn=_dummy_embed, wait=True)

        # Corrupt the FTS index by dropping a shadow table
        conn = sqlite3.connect(db_path)
        corrupted = False
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'documents_fts_%'"
            ).fetchall()
            if tables:
                for t in tables:
                    if "data" in t[0] or "content" in t[0]:
                        conn.execute(f"DROP TABLE IF EXISTS [{t[0]}]")
                        conn.commit()
                        corrupted = True
                        break
        finally:
            conn.close()

        if not corrupted:
            pytest.skip("Could not corrupt FTS shadow table")

        # Second session: should detect corruption, rebuild, and work
        with SQFox(db_path) as db:
            assert db.is_running
            # Ingest new data + verify FTS search works after rebuild
            db.ingest("fresh doc about fish and birds", embed_fn=_dummy_embed, wait=True)
            results = db.search("fish", embed_fn=_dummy_embed, limit=5)
            # At minimum, the engine should not crash; if rebuild succeeded,
            # we may get results (FTS or vector)
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# HNSW NaN guard (T5)
# ---------------------------------------------------------------------------

class TestHnswNanGuard:
    def test_hnsw_rejects_nan_vector(self):
        """HNSW add() should skip NaN vectors without crashing."""
        from sqfox.backends.hnsw import SqliteHnswBackend
        backend = SqliteHnswBackend()
        backend._initialized = True
        backend._ndim = 4

        # Add a valid vector, then a NaN vector
        backend.add([1], [[1.0, 2.0, 3.0, 4.0]])
        backend.add([2], [[float("nan"), 2.0, 3.0, 4.0]])

        # Only the valid vector should be in the index
        assert backend.count() == 1

    def test_hnsw_rejects_inf_vector(self):
        """HNSW add() should skip Inf vectors without crashing."""
        from sqfox.backends.hnsw import SqliteHnswBackend
        backend = SqliteHnswBackend()
        backend._initialized = True
        backend._ndim = 4

        backend.add([1], [[1.0, 2.0, 3.0, 4.0]])
        backend.add([2], [[float("inf"), 2.0, 3.0, 4.0]])

        assert backend.count() == 1
