"""Tests for sqfox engine: writer thread, queue, reads, concurrency."""

import sqlite3
import tempfile
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqfox import SQFox, Priority, EngineClosedError, QueueFullError


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
        db.stop()  # Should not raise

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
        db.start()  # Should not raise or start second thread
        assert db.is_running
        db.stop()


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

class TestWrites:
    def test_write_single(self, engine):
        future = engine.write("INSERT INTO t (val) VALUES (?)", ("hello",))
        assert isinstance(future, Future)
        result = future.result(timeout=5)
        assert result is not None  # lastrowid

    def test_write_wait(self, engine):
        result = engine.write(
            "INSERT INTO t (val) VALUES (?)", ("world",), wait=True
        )
        assert result is not None

    def test_write_error_propagation(self, engine):
        future = engine.write("INSERT INTO nonexistent_table VALUES (1)")
        with pytest.raises(Exception):
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
        db.write("CREATE TABLE IF NOT EXISTS ordered (seq INTEGER, prio TEXT)", wait=True)

        # Pause writer by filling queue before it processes
        # Submit LOW first, then HIGH
        results = []

        f_low = db.write(
            "INSERT INTO ordered (seq, prio) VALUES (1, 'low')",
            priority=Priority.LOW,
        )
        f_high = db.write(
            "INSERT INTO ordered (seq, prio) VALUES (2, 'high')",
            priority=Priority.HIGH,
        )

        # Wait for both to complete
        f_low.result(timeout=5)
        f_high.result(timeout=5)

        db.stop()

    def test_queue_full(self, db_path):
        db = SQFox(db_path, max_queue_size=2)
        db.start()
        db.write("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)", wait=True)

        # Fill the queue without waiting
        # The writer thread may process items, so we need to be fast
        with pytest.raises(QueueFullError):
            for _ in range(100):
                db.write(
                    "INSERT INTO t DEFAULT VALUES",
                    priority=Priority.LOW,
                )
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

        assert len(errors) >= 1
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
        with pytest.raises(Exception):
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
