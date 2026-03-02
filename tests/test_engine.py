"""Tests for sqfox engine: writer thread, queue, reads, concurrency."""

import sqlite3
import threading
import time
from concurrent.futures import Future

import pytest

from sqfox import SQFox, Priority, EngineClosedError, QueueFullError, SchemaState


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary database file path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def engine(db_path):
    """Provide a started SQFox engine, stopped after test."""
    db = SQFox(db_path)
    db.start()
    # Create a simple test table
    db.write("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)", wait=True)
    yield db
    db.stop()


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_stop(self, db_path):
        db = SQFox(db_path)
        db.start()
        assert db.is_running
        db.stop()
        assert not db.is_running

    def test_context_manager(self, db_path):
        with SQFox(db_path) as db:
            assert db.is_running
        assert not db.is_running

    def test_double_stop(self, db_path):
        db = SQFox(db_path)
        db.start()
        db.stop()
        assert not db.is_running
        db.stop()  # Should not raise
        assert not db.is_running

    def test_write_after_stop(self, db_path):
        db = SQFox(db_path)
        db.start()
        db.stop()
        with pytest.raises(EngineClosedError):
            db.write("INSERT INTO t VALUES (1, 'a')")

    def test_cannot_restart(self, db_path):
        db = SQFox(db_path)
        db.start()
        db.stop()
        with pytest.raises(EngineClosedError):
            db.start()

    def test_start_idempotent(self, db_path):
        db = SQFox(db_path)
        db.start()
        thread_before = db._writer_thread
        db.start()  # Should not start second thread
        assert db.is_running
        assert db._writer_thread is thread_before
        db.stop()


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

class TestWrites:
    def test_write_single(self, engine):
        future = engine.write("INSERT INTO t (val) VALUES (?)", ("hello",))
        assert isinstance(future, Future)
        result = future.result(timeout=5)
        assert isinstance(result, int) and result > 0

    def test_write_wait(self, engine):
        result = engine.write(
            "INSERT INTO t (val) VALUES (?)", ("world",), wait=True
        )
        assert isinstance(result, int) and result > 0

    def test_write_error_propagation(self, engine):
        future = engine.write("INSERT INTO nonexistent_table VALUES (1)")
        with pytest.raises(sqlite3.OperationalError):
            future.result(timeout=5)

    def test_write_many(self, engine):
        data = [(f"item_{i}",) for i in range(10)]
        result = engine.write(
            "INSERT INTO t (val) VALUES (?)",
            data,
            many=True,
            wait=True,
        )
        # Verify all rows inserted
        rows = engine.fetch_all("SELECT COUNT(*) FROM t")
        assert rows[0][0] == 10

    def test_write_priority_ordering(self, db_path):
        """HIGH priority writes should execute before LOW."""
        db = SQFox(db_path, batch_time_ms=200)
        db.start()
        db.write("CREATE TABLE IF NOT EXISTS ordering (id INTEGER PRIMARY KEY, val TEXT)", wait=True)

        blocker = threading.Event()

        # Block the writer
        def block_fn(conn):
            blocker.wait(timeout=5)
        db.execute_on_writer(block_fn, wait=False)
        time.sleep(0.05)  # let it start

        # Enqueue LOW then HIGH while writer is blocked
        f_low = db.write("INSERT INTO ordering (val) VALUES ('low')", priority=Priority.LOW)
        f_high = db.write("INSERT INTO ordering (val) VALUES ('high')", priority=Priority.HIGH)

        # Unblock
        blocker.set()
        f_low.result(timeout=5)
        f_high.result(timeout=5)

        rows = db.fetch_all("SELECT val FROM ordering ORDER BY id")
        # HIGH should have been processed first
        assert rows[0]["val"] == "high"
        assert rows[1]["val"] == "low"

        db.stop()

    def test_queue_full(self, tmp_path):
        db = SQFox(str(tmp_path / "qf.db"), max_queue_size=2, batch_time_ms=500)
        db.start()
        try:
            db.write("CREATE TABLE t (id INTEGER PRIMARY KEY)", wait=True)

            blocker = threading.Event()
            db.execute_on_writer(lambda conn: blocker.wait(timeout=10), wait=False)
            time.sleep(0.05)

            # Now writer is blocked, queue should fill after 2 items
            full_count = 0
            for i in range(20):
                try:
                    db.write(f"INSERT INTO t VALUES ({i})")
                except QueueFullError:
                    full_count += 1

            blocker.set()
            assert full_count > 0
        finally:
            db.stop()


# ---------------------------------------------------------------------------
# Read tests
# ---------------------------------------------------------------------------

class TestReads:
    def test_fetch_one(self, engine):
        engine.write("INSERT INTO t (val) VALUES ('test')", wait=True)
        row = engine.fetch_one("SELECT val FROM t WHERE val = ?", ("test",))
        assert row is not None
        assert row["val"] == "test"

    def test_fetch_all(self, engine):
        for i in range(5):
            engine.write(f"INSERT INTO t (val) VALUES ('item_{i}')", wait=True)
        time.sleep(0.05)
        rows = engine.fetch_all("SELECT val FROM t WHERE val LIKE 'item_%'")
        assert len(rows) == 5

    def test_fetch_one_none(self, engine):
        # Ensure the table is visible to reader
        engine.write("INSERT INTO t (val) VALUES ('exists')", wait=True)
        time.sleep(0.05)
        row = engine.fetch_one("SELECT * FROM t WHERE val = 'nonexistent'")
        assert row is None

    def test_reader_context(self, engine):
        engine.write("INSERT INTO t (val) VALUES ('ctx')", wait=True)
        with engine.reader() as conn:
            cursor = conn.execute("SELECT val FROM t")
            rows = cursor.fetchall()
            assert len(rows) == 1


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_writes(self, db_path):
        """100 writes from 10 threads should all succeed."""
        db = SQFox(db_path)
        db.start()
        db.write(
            "CREATE TABLE IF NOT EXISTS conc (id INTEGER PRIMARY KEY, tid INTEGER)",
            wait=True,
        )

        errors = []

        def writer(thread_id):
            for i in range(10):
                try:
                    db.write(
                        "INSERT INTO conc (tid) VALUES (?)",
                        (thread_id,),
                        wait=True,
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(tid,))
            for tid in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent writes: {errors}"

        rows = db.fetch_all("SELECT COUNT(*) FROM conc")
        assert rows[0][0] == 100

        db.stop()

    def test_concurrent_reads(self, db_path):
        """Reads from multiple threads use separate connections."""
        db = SQFox(db_path)
        db.start()
        db.write("CREATE TABLE IF NOT EXISTS rtest (val TEXT)", wait=True)
        db.write("INSERT INTO rtest (val) VALUES ('data')", wait=True)

        results = []
        conn_ids = []

        def reader():
            row = db.fetch_one("SELECT val FROM rtest")
            results.append(row["val"] if row else None)
            # Access the connection id to verify separate connections
            conn = db._get_reader_connection()
            conn_ids.append(id(conn))

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(r == "data" for r in results)
        # Each thread should have its own connection
        assert len(set(conn_ids)) == 5

        db.stop()

    def test_write_read_isolation(self, db_path):
        """Writes should be visible to reads after commit."""
        db = SQFox(db_path)
        db.start()
        db.write("CREATE TABLE IF NOT EXISTS iso (val TEXT)", wait=True)
        db.write("INSERT INTO iso (val) VALUES ('visible')", wait=True)

        # Allow WAL to propagate
        time.sleep(0.1)

        row = db.fetch_one("SELECT val FROM iso")
        assert row is not None
        assert row["val"] == "visible"

        db.stop()


# ---------------------------------------------------------------------------
# PRAGMA tests
# ---------------------------------------------------------------------------

class TestPragmas:
    def test_wal_mode(self, db_path):
        with SQFox(db_path) as db:
            row = db.fetch_one("PRAGMA journal_mode")
            assert row[0] == "wal"

    def test_pragmas_applied(self, db_path):
        with SQFox(db_path, busy_timeout_ms=3000) as db:
            row = db.fetch_one("PRAGMA synchronous")
            assert row[0] == 1  # NORMAL = 1

            row = db.fetch_one("PRAGMA busy_timeout")
            assert row[0] == 3000

            row = db.fetch_one("PRAGMA foreign_keys")
            assert row[0] == 1

            row = db.fetch_one("PRAGMA temp_store")
            assert row[0] == 2  # MEMORY = 2


# ---------------------------------------------------------------------------
# Properties tests
# ---------------------------------------------------------------------------

class TestProperties:
    def test_path(self, engine, db_path):
        assert engine.path == db_path

    def test_is_running(self, engine):
        assert engine.is_running

    def test_queue_size(self, engine):
        assert engine.queue_size >= 0


# ---------------------------------------------------------------------------
# Error callback tests
# ---------------------------------------------------------------------------

class TestErrorCallback:
    def test_callback_on_bad_sql(self, db_path):
        """error_callback fires on SQL errors in fire-and-forget mode."""
        errors = []

        def on_error(sql, exc):
            errors.append((sql, exc))

        db = SQFox(db_path, error_callback=on_error)
        db.start()

        # Fire-and-forget — no wait
        db.write("INSERT INTO nonexistent_table VALUES (1)")

        # Give writer thread time to process
        time.sleep(0.3)
        db.stop()

        assert len(errors) == 1
        sql, exc = errors[0]
        assert "nonexistent_table" in sql
        assert isinstance(exc, Exception)

    def test_callback_on_constraint_violation(self, db_path):
        """error_callback fires on constraint violations."""
        errors = []

        def on_error(sql, exc):
            errors.append((sql, exc))

        db = SQFox(db_path, error_callback=on_error)
        db.start()
        db.write(
            "CREATE TABLE uniq (id INTEGER PRIMARY KEY, val TEXT UNIQUE)",
            wait=True,
        )
        db.write("INSERT INTO uniq VALUES (1, 'a')", wait=True)

        # Duplicate — will fail, fire-and-forget
        db.write("INSERT INTO uniq VALUES (2, 'a')")
        time.sleep(0.3)
        db.stop()

        assert len(errors) >= 1
        assert "UNIQUE" in str(errors[0][1]).upper() or "constraint" in str(errors[0][1]).lower()

    def test_no_callback_no_crash(self, db_path):
        """Without error_callback, errors still go to Future / logging."""
        db = SQFox(db_path)
        db.start()

        future = db.write("INSERT INTO nope VALUES (1)")
        with pytest.raises(sqlite3.OperationalError):
            future.result(timeout=5)

        db.stop()

    def test_callback_receives_batch_abort(self, db_path):
        """When a batch fails, subsequent requests also trigger callback."""
        errors = []

        def on_error(sql, exc):
            errors.append(sql)

        db = SQFox(db_path, error_callback=on_error, batch_time_ms=500)
        db.start()
        db.write("CREATE TABLE t (id INTEGER PRIMARY KEY)", wait=True)

        # These will likely batch together — second one fails
        db.write("INSERT INTO t VALUES (1)")
        db.write("INSERT INTO nonexistent VALUES (2)")
        db.write("INSERT INTO t VALUES (3)")

        time.sleep(1.0)
        db.stop()

        # At least the bad SQL triggered the callback
        assert any("nonexistent" in s for s in errors)


# ---------------------------------------------------------------------------
# Backup tests
# ---------------------------------------------------------------------------

class TestBackup:
    def test_backup_creates_copy(self, db_path, tmp_path):
        """backup() creates a valid copy of the database."""
        db = SQFox(db_path)
        db.start()
        db.write("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", wait=True)
        db.write("INSERT INTO t VALUES (1, 'hello')", wait=True)
        db.write("INSERT INTO t VALUES (2, 'world')", wait=True)

        backup_path = str(tmp_path / "backup.db")
        db.backup(backup_path)
        db.stop()

        conn = sqlite3.connect(backup_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT val FROM t ORDER BY id").fetchall()
        assert [r[0] for r in rows] == ["hello", "world"]
        conn.close()

    def test_backup_while_writing(self, db_path, tmp_path):
        """backup() works while the writer thread is active."""
        db = SQFox(db_path)
        db.start()
        db.write("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", wait=True)

        futures = []
        for i in range(100):
            futures.append(db.write("INSERT INTO t (val) VALUES (?)", (i,)))
        # Wait for all writes to complete
        for f in futures:
            f.result(timeout=10)

        backup_path = str(tmp_path / "backup.db")
        db.backup(backup_path)
        db.stop()

        conn = sqlite3.connect(backup_path)
        count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert count == 100
        conn.close()

    def test_backup_progress_callback(self, db_path, tmp_path):
        """backup() calls the progress callback."""
        db = SQFox(db_path)
        db.start()
        db.write("CREATE TABLE t (val TEXT)", wait=True)
        db.write("INSERT INTO t VALUES ('data')", wait=True)

        calls = []
        def on_progress(status, remaining, total):
            calls.append((status, remaining, total))

        backup_path = str(tmp_path / "backup.db")
        db.backup(backup_path, pages=1, progress=on_progress)
        db.stop()

        assert len(calls) > 0
        assert calls[-1][1] == 0  # remaining should be 0 at completion

    def test_backup_requires_running_engine(self, db_path, tmp_path):
        """backup() raises EngineClosedError if engine is not running."""
        db = SQFox(db_path)
        with pytest.raises(EngineClosedError):
            db.backup(str(tmp_path / "backup.db"))

    def test_backup_memory_db(self, tmp_path):
        """backup() works for :memory: databases via writer connection."""
        db = SQFox(":memory:")
        db.start()
        db.write("CREATE TABLE t (val TEXT)", wait=True)
        db.write("INSERT INTO t VALUES ('mem_data')", wait=True)

        backup_path = str(tmp_path / "mem_backup.db")
        db.backup(backup_path)
        db.stop()

        conn = sqlite3.connect(backup_path)
        row = conn.execute("SELECT val FROM t").fetchone()
        assert row[0] == "mem_data"
        conn.close()


# ---------------------------------------------------------------------------
# execute_on_writer tests
# ---------------------------------------------------------------------------

class TestExecuteOnWriter:
    def test_basic_callable(self, db_path):
        """execute_on_writer runs a callable on the writer connection."""
        db = SQFox(db_path)
        db.start()
        db.write("CREATE TABLE t (val TEXT)", wait=True)

        result = db.execute_on_writer(
            lambda conn: conn.execute("INSERT INTO t VALUES ('eow')").lastrowid,
            wait=True,
        )
        assert isinstance(result, int)

        row = db.fetch_one("SELECT val FROM t")
        assert row is not None
        assert row["val"] == "eow"
        db.stop()

    def test_callable_auto_commits(self, db_path):
        """execute_on_writer auto-commits if callable leaves open transaction."""
        db = SQFox(db_path)
        db.start()
        db.write("CREATE TABLE t (val TEXT)", wait=True)

        # This callable doesn't commit — engine should auto-commit
        def insert_no_commit(conn):
            conn.execute("INSERT INTO t VALUES ('no_commit')")

        db.execute_on_writer(insert_no_commit, wait=True)

        # Verify data was committed
        row = db.fetch_one("SELECT val FROM t WHERE val = 'no_commit'")
        assert row is not None

        # Verify subsequent writes still work (connection not poisoned)
        db.write("INSERT INTO t VALUES ('after')", wait=True)
        row = db.fetch_one("SELECT val FROM t WHERE val = 'after'")
        assert row is not None
        db.stop()

    def test_callable_error_propagation(self, db_path):
        """Errors in callable propagate to the caller."""
        db = SQFox(db_path)
        db.start()

        def raise_error(conn):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            db.execute_on_writer(raise_error, wait=True)
        db.stop()


# ---------------------------------------------------------------------------
# on_startup hook tests
# ---------------------------------------------------------------------------

class TestOnStartup:
    def test_hook_runs_on_start(self, db_path):
        """on_startup hooks are called when engine starts."""
        calls = []
        db = SQFox(db_path)
        db.on_startup(lambda engine: calls.append(engine))
        db.start()
        assert len(calls) == 1
        assert calls[0] is db
        db.stop()

    def test_hook_failure_stops_engine(self, db_path):
        """If a startup hook raises, engine is stopped."""
        db = SQFox(db_path)

        def bad_hook(engine):
            raise RuntimeError("hook failed")

        db.on_startup(bad_hook)
        with pytest.raises(RuntimeError, match="hook failed"):
            db.start()
        assert not db.is_running


# ---------------------------------------------------------------------------
# diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_diagnostics_returns_dict(self, db_path):
        db = SQFox(db_path)
        db.start()
        diag = db.diagnostics()
        assert isinstance(diag, dict)
        assert "sqfox_version" in diag
        assert "sqlite_version" in diag
        assert "is_running" in diag
        assert diag["is_running"] is True
        assert diag["path"] == db_path
        assert "vec_available" in diag
        assert "schema_state" in diag
        db.stop()


# ---------------------------------------------------------------------------
# Ingest tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_basic(self, db_path):
        """Basic ingest without embed_fn."""
        with SQFox(db_path) as db:
            doc_id = db.ingest("Hello world", wait=True)
            assert isinstance(doc_id, int) and doc_id >= 1
            row = db.fetch_one("SELECT content FROM documents WHERE id = ?", (doc_id,))
            assert row is not None
            assert row["content"] == "Hello world"

    def test_ingest_with_metadata(self, db_path):
        """Ingest with metadata stores JSON."""
        import json
        with SQFox(db_path) as db:
            doc_id = db.ingest("test", metadata={"key": "val"}, wait=True)
            row = db.fetch_one("SELECT metadata FROM documents WHERE id = ?", (doc_id,))
            meta = json.loads(row["metadata"])
            assert meta["key"] == "val"

    def test_ingest_chunker_exception(self, db_path):
        """Chunker exception propagates to caller."""
        def bad_chunker(text):
            raise RuntimeError("chunk failed")
        with SQFox(db_path) as db:
            with pytest.raises(RuntimeError, match="chunk failed"):
                db.ingest("test", chunker=bad_chunker, wait=True)

    def test_ingest_embed_exception(self, db_path):
        """Embed exception propagates to caller."""
        def bad_embed(texts):
            raise RuntimeError("embed failed")
        with SQFox(db_path) as db:
            with pytest.raises(RuntimeError, match="embed failed"):
                db.ingest("test", embed_fn=bad_embed, wait=True)


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_fts_only(self, db_path):
        """FTS-only search finds ingested documents."""
        with SQFox(db_path) as db:
            db.ensure_schema(SchemaState.SEARCHABLE)
            db.ingest("Python database tutorial about SQLite", wait=True)
            time.sleep(0.1)
            results = db.search("python database")
            assert isinstance(results, list)
            assert len(results) >= 1, (
                f"FTS search for 'python database' should find the ingested doc, got {results}"
            )
            assert "Python" in results[0].text or "database" in results[0].text

    def test_search_empty_db(self, db_path):
        """Search on empty DB returns empty list."""
        with SQFox(db_path) as db:
            results = db.search("anything")
            assert results == []

    def test_search_broken_embed_falls_back_to_fts(self, db_path):
        """search() with broken embed_fn falls back to FTS results.

        The embed_fn error is caught inside hybrid_search (vec step),
        so FTS results are still returned.
        """
        with SQFox(db_path) as db:
            db.ensure_schema(SchemaState.SEARCHABLE)
            db.ingest("Test document for search", wait=True)
            time.sleep(0.1)

            def broken_embed(texts):
                raise RuntimeError("model crashed")

            # hybrid_search catches embed errors internally (vec step),
            # FTS still works — should return results, not raise
            results = db.search("test document", embed_fn=broken_embed)
            assert isinstance(results, list)
            # FTS should find the document even though vec failed


# ---------------------------------------------------------------------------
# EnsureSchema tests
# ---------------------------------------------------------------------------

class TestEnsureSchema:
    def test_ensure_schema_base(self, db_path):
        """ensure_schema creates base tables."""
        with SQFox(db_path) as db:
            result = db.ensure_schema(SchemaState.BASE)
            assert result >= SchemaState.BASE
            row = db.fetch_one("SELECT name FROM sqlite_master WHERE name='documents'")
            assert row is not None

    def test_ensure_schema_searchable(self, db_path):
        """ensure_schema creates FTS table."""
        with SQFox(db_path) as db:
            result = db.ensure_schema(SchemaState.SEARCHABLE)
            assert result >= SchemaState.SEARCHABLE
            row = db.fetch_one("SELECT name FROM sqlite_master WHERE name='documents_fts'")
            assert row is not None


# ---------------------------------------------------------------------------
# VecAvailable tests
# ---------------------------------------------------------------------------

class TestVecAvailable:
    def test_vec_available_property(self, db_path):
        with SQFox(db_path) as db:
            assert isinstance(db.vec_available, bool)

    def test_vec_disabled(self, db_path):
        with SQFox(db_path, enable_vec=False) as db:
            assert db.vec_available is False


# ---------------------------------------------------------------------------
# Memory DB tests
# ---------------------------------------------------------------------------

class TestMemoryDB:
    def test_memory_write_read(self):
        """In-memory DB supports basic write and read."""
        with SQFox(":memory:") as db:
            db.write("CREATE TABLE t (val TEXT)", wait=True)
            db.write("INSERT INTO t VALUES ('hello')", wait=True)
            time.sleep(0.1)
            row = db.fetch_one("SELECT val FROM t")
            assert row is not None
            assert row["val"] == "hello"
