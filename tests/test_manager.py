"""Tests for sqfox manager: multi-database lifecycle, cross-search, drop."""

import time
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqfox import SQFoxManager, SQFox


@pytest.fixture
def base_dir(tmp_path):
    return tmp_path / "databases"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_context_manager(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            db = mgr["test"]
            assert db.is_running
        assert not db.is_running

    def test_start_stop(self, base_dir):
        mgr = SQFoxManager(base_dir)
        mgr.start()
        db = mgr["mydb"]
        assert db.is_running
        mgr.stop()
        assert not db.is_running

    def test_creates_base_dir(self, base_dir):
        assert not base_dir.exists()
        with SQFoxManager(base_dir) as mgr:
            mgr["test"]
        assert base_dir.exists()

    def test_db_file_created(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            mgr["sensors"]
        assert (base_dir / "sensors.db").exists()

    def test_custom_extension(self, base_dir):
        with SQFoxManager(base_dir, db_ext=".sqlite") as mgr:
            mgr["data"]
        assert (base_dir / "data.sqlite").exists()


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------

class TestAccess:
    def test_get_or_create_returns_same(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            db1 = mgr["test"]
            db2 = mgr["test"]
            assert db1 is db2

    def test_different_names_different_dbs(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            db1 = mgr["sensors"]
            db2 = mgr["knowledge"]
            assert db1 is not db2
            assert db1.path != db2.path

    def test_contains(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            assert "test" not in mgr
            mgr["test"]
            assert "test" in mgr

    def test_names_property(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            mgr["a"]
            mgr["b"]
            mgr["c"]
            assert sorted(mgr.names) == ["a", "b", "c"]

    def test_databases_property(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            mgr["x"]
            dbs = mgr.databases
            assert "x" in dbs
            assert isinstance(dbs["x"], SQFox)

    def test_default_kwargs_passed(self, base_dir):
        with SQFoxManager(base_dir, busy_timeout_ms=1234) as mgr:
            db = mgr["test"]
            row = db.fetch_one("PRAGMA busy_timeout")
            assert row[0] == 1234


# ---------------------------------------------------------------------------
# Write / Read isolation
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_databases_are_isolated(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            db1 = mgr["db1"]
            db2 = mgr["db2"]

            db1.write("CREATE TABLE t (val TEXT)", wait=True)
            db1.write("INSERT INTO t VALUES ('from_db1')", wait=True)

            db2.write("CREATE TABLE t (val TEXT)", wait=True)
            db2.write("INSERT INTO t VALUES ('from_db2')", wait=True)

            time.sleep(0.05)

            rows1 = db1.fetch_all("SELECT val FROM t")
            rows2 = db2.fetch_all("SELECT val FROM t")

            assert len(rows1) == 1
            assert rows1[0][0] == "from_db1"
            assert len(rows2) == 1
            assert rows2[0][0] == "from_db2"


# ---------------------------------------------------------------------------
# Drop
# ---------------------------------------------------------------------------

class TestDrop:
    def test_drop_stops_db(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            db = mgr["disposable"]
            assert db.is_running
            mgr.drop("disposable")
            assert not db.is_running
            assert "disposable" not in mgr

    def test_drop_with_delete(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            mgr["todelete"]
            db_file = base_dir / "todelete.db"
            assert db_file.exists()
            mgr.drop("todelete", delete_file=True)
            assert not db_file.exists()

    def test_drop_nonexistent(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            # Should not raise
            mgr.drop("nope")


# ---------------------------------------------------------------------------
# Cross-database search
# ---------------------------------------------------------------------------

class TestSearchAll:
    def test_search_all_empty(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            results = mgr.search_all("test")
            assert results == []

    def test_ingest_to(self, base_dir):
        with SQFoxManager(base_dir) as mgr:
            mgr.ingest_to(
                "docs",
                "This is a test document about databases",
                wait=True,
            )
            db = mgr["docs"]
            time.sleep(0.05)
            row = db.fetch_one("SELECT content FROM documents LIMIT 1")
            assert row is not None
            assert "test document" in row[0]
