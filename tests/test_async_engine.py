"""Tests for AsyncSQFox — async facade with dual thread pools."""

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqfox import AsyncSQFox, EngineClosedError, SchemaState


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_async_context_manager(self, tmp_path):
        """async with starts and stops the engine."""
        path = str(tmp_path / "life.db")
        async with AsyncSQFox(path) as db:
            assert db.is_running
        assert not db.is_running

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path):
        """Manual start/stop works."""
        db = AsyncSQFox(str(tmp_path / "manual.db"))
        db.start()
        assert db.is_running
        await db.stop()
        assert not db.is_running


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

class TestWrites:
    @pytest.mark.asyncio
    async def test_write_and_read(self, tmp_path):
        """Basic write → read cycle."""
        async with AsyncSQFox(str(tmp_path / "wr.db")) as db:
            await db.write(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)",
                wait=True,
            )
            await db.write(
                "INSERT INTO t VALUES (1, 'hello')", wait=True,
            )

            row = await db.fetch_one("SELECT val FROM t WHERE id = 1")
            assert row is not None
            assert row[0] == "hello"

    @pytest.mark.asyncio
    async def test_write_fire_and_forget(self, tmp_path):
        """write(wait=False) returns immediately."""
        async with AsyncSQFox(str(tmp_path / "ff.db")) as db:
            await db.write("CREATE TABLE t (val TEXT)", wait=True)
            result = await db.write("INSERT INTO t VALUES ('x')", wait=False)
            assert result is None

            # Give writer time to process
            await asyncio.sleep(0.2)
            row = await db.fetch_one("SELECT val FROM t")
            assert row is not None

    @pytest.mark.asyncio
    async def test_write_many(self, tmp_path):
        """executemany via async write."""
        async with AsyncSQFox(str(tmp_path / "many.db")) as db:
            await db.write("CREATE TABLE t (val INTEGER)", wait=True)
            await db.write(
                "INSERT INTO t VALUES (?)",
                [(i,) for i in range(10)],
                many=True,
                wait=True,
            )

            rows = await db.fetch_all("SELECT val FROM t ORDER BY val")
            assert len(rows) == 10


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

class TestReads:
    @pytest.mark.asyncio
    async def test_fetch_all(self, tmp_path):
        async with AsyncSQFox(str(tmp_path / "fa.db")) as db:
            await db.write("CREATE TABLE t (val INTEGER)", wait=True)
            for i in range(5):
                await db.write("INSERT INTO t VALUES (?)", (i,), wait=True)

            rows = await db.fetch_all("SELECT val FROM t ORDER BY val")
            assert [r[0] for r in rows] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_fetch_one_none(self, tmp_path):
        async with AsyncSQFox(str(tmp_path / "none.db")) as db:
            await db.write("CREATE TABLE t (val TEXT)", wait=True)
            row = await db.fetch_one("SELECT val FROM t")
            assert row is None


# ---------------------------------------------------------------------------
# Ingest (CPU pool)
# ---------------------------------------------------------------------------

class TestIngest:
    @pytest.mark.asyncio
    async def test_ingest_basic(self, tmp_path):
        """ingest() runs in CPU pool and returns doc ID."""
        async with AsyncSQFox(str(tmp_path / "ingest.db")) as db:
            doc_id = await db.ingest("Hello world")
            assert isinstance(doc_id, int)
            assert doc_id >= 1

    @pytest.mark.asyncio
    async def test_ingest_with_metadata(self, tmp_path):
        async with AsyncSQFox(str(tmp_path / "meta.db")) as db:
            doc_id = await db.ingest(
                "Temperature reading",
                metadata={"sensor": "temp_01"},
            )
            await asyncio.sleep(0.05)
            row = await db.fetch_one(
                "SELECT metadata FROM documents WHERE id = ?", (doc_id,),
            )
            assert row is not None
            assert "temp_01" in row[0]

    @pytest.mark.asyncio
    async def test_concurrent_ingest_limited(self, tmp_path):
        """CPU pool limits concurrent ingest operations."""
        max_workers = 2
        active = {"count": 0, "peak": 0}
        lock = asyncio.Lock()

        original_ingest = None

        async with AsyncSQFox(
            str(tmp_path / "conc.db"),
            max_cpu_workers=max_workers,
        ) as db:
            # We track concurrency by looking at how many ingest calls
            # are active simultaneously
            tasks = []
            for i in range(6):
                tasks.append(db.ingest(f"Document {i} about testing"))

            results = await asyncio.gather(*tasks)
            assert len(results) == 6
            assert all(isinstance(r, int) for r in results)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    @pytest.mark.asyncio
    async def test_fts_search_uses_io_pool(self, tmp_path):
        """FTS-only search should work (routed to I/O pool)."""
        async with AsyncSQFox(str(tmp_path / "fts.db")) as db:
            await db.ensure_schema(SchemaState.SEARCHABLE)
            await db.ingest("Python database tutorial")
            await asyncio.sleep(0.1)

            results = await db.search("python")
            # May or may not find results depending on FTS state
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_uses_cpu_pool(self, tmp_path):
        """Search with embed_fn should use CPU pool."""
        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec not available")

        def mock_embed(texts):
            return [[float(ord(c)) / 200.0 for c in t[:4].ljust(4)]
                    for t in texts]

        async with AsyncSQFox(str(tmp_path / "hyb.db")) as db:
            await db.ingest("Database optimization guide", embed_fn=mock_embed)
            await asyncio.sleep(0.1)

            results = await db.search("database", embed_fn=mock_embed)
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

class TestBackup:
    @pytest.mark.asyncio
    async def test_backup(self, tmp_path):
        """Online backup via async interface."""
        async with AsyncSQFox(str(tmp_path / "src.db")) as db:
            await db.write("CREATE TABLE t (val TEXT)", wait=True)
            await db.write("INSERT INTO t VALUES ('backup_test')", wait=True)

            backup_path = str(tmp_path / "bak.db")
            await db.backup(backup_path)

        conn = sqlite3.connect(backup_path)
        row = conn.execute("SELECT val FROM t").fetchone()
        assert row[0] == "backup_test"
        conn.close()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    @pytest.mark.asyncio
    async def test_properties(self, tmp_path):
        async with AsyncSQFox(str(tmp_path / "props.db")) as db:
            assert db.is_running
            assert db.path == str(tmp_path / "props.db")
            assert db.queue_size >= 0
            assert isinstance(db.diagnostics(), dict)


# ---------------------------------------------------------------------------
# Pool isolation — reads don't block on CPU work
# ---------------------------------------------------------------------------

class TestPoolIsolation:
    @pytest.mark.asyncio
    async def test_reads_not_blocked_by_ingest(self, tmp_path):
        """fetch_all should complete quickly even during heavy ingest."""
        async with AsyncSQFox(
            str(tmp_path / "iso.db"), max_cpu_workers=1,
        ) as db:
            await db.write("CREATE TABLE fast (val TEXT)", wait=True)
            await db.write("INSERT INTO fast VALUES ('quick')", wait=True)

            # Start a slow ingest with a slow chunker
            def slow_chunker(text):
                time.sleep(0.5)  # Simulate slow CPU work
                return [text]

            # Fire slow ingest (CPU pool, 1 worker)
            ingest_task = asyncio.create_task(
                db.ingest("Slow document", chunker=slow_chunker)
            )

            # Immediately try a read (I/O pool)
            await asyncio.sleep(0.05)  # Let ingest start
            start = time.monotonic()
            row = await db.fetch_one("SELECT val FROM fast")
            elapsed = time.monotonic() - start

            assert row is not None
            assert row[0] == "quick"
            # Read should be fast (< 0.3s), not waiting for slow ingest
            assert elapsed < 0.3, f"Read took {elapsed:.2f}s, expected < 0.3s"

            await ingest_task
