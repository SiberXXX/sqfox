"""Tests for AsyncSQFox — async facade with dual thread pools."""

import asyncio
import sqlite3
import time

import pytest

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
        """write(wait=False) returns an asyncio.Future immediately."""
        async with AsyncSQFox(str(tmp_path / "ff.db")) as db:
            await db.write("CREATE TABLE t (val TEXT)", wait=True)
            result = await db.write("INSERT INTO t VALUES ('x')", wait=False)
            assert isinstance(result, asyncio.Future)

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
# Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_write_bad_sql_raises(self, tmp_path):
        async with AsyncSQFox(str(tmp_path / "err.db")) as db:
            with pytest.raises(sqlite3.OperationalError):
                await db.write("INVALID SQL GIBBERISH", wait=True)

    @pytest.mark.asyncio
    async def test_operations_after_stop_raise(self, tmp_path):
        db = AsyncSQFox(str(tmp_path / "stopped.db"))
        db.start()
        await db.stop()
        with pytest.raises(EngineClosedError):
            await db.write("SELECT 1", wait=True)


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
            row = await db.fetch_one(
                "SELECT content FROM documents WHERE id = ?", (doc_id,),
            )
            assert row is not None
            assert row[0] == "Hello world"

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
        async with AsyncSQFox(
            str(tmp_path / "conc.db"),
            max_cpu_workers=max_workers,
        ) as db:
            tasks = []
            for i in range(6):
                tasks.append(db.ingest(f"Document {i} about testing"))
            results = await asyncio.gather(*tasks)
            assert len(results) == 6
            assert all(isinstance(r, int) for r in results)
            # Verify all documents were actually ingested
            count = await db.fetch_one("SELECT COUNT(*) FROM documents")
            assert count[0] == 6


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    @pytest.mark.asyncio
    async def test_fts_search_uses_io_pool(self, tmp_path):
        """FTS-only search should work (routed to I/O pool)."""
        from sqfox import SearchResult

        async with AsyncSQFox(str(tmp_path / "fts.db")) as db:
            await db.ensure_schema(SchemaState.SEARCHABLE)
            await db.ingest("Python database tutorial")
            await asyncio.sleep(0.1)

            results = await db.search("python")
            assert isinstance(results, list)
            assert len(results) >= 1, "FTS search should find the ingested document"
            assert isinstance(results[0], SearchResult)
            assert "Python" in results[0].text or "python" in results[0].text.lower()

    @pytest.mark.asyncio
    async def test_hybrid_search_uses_cpu_pool(self, tmp_path):
        """Search with embed_fn should use CPU pool."""
        from sqfox import SearchResult

        def mock_embed(texts):
            return [[float(ord(c)) / 200.0 for c in t[:4].ljust(4)]
                    for t in texts]

        async with AsyncSQFox(str(tmp_path / "hyb.db"), vector_backend="hnsw") as db:
            await db.ingest("Database optimization guide", embed_fn=mock_embed)
            await asyncio.sleep(0.1)

            results = await db.search("database", embed_fn=mock_embed)
            assert isinstance(results, list)
            assert len(results) >= 1, "Hybrid search should find the ingested document"
            assert isinstance(results[0], SearchResult)
            assert results[0].score > 0


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

    @pytest.mark.asyncio
    async def test_backup_rejects_async_progress(self, tmp_path):
        """backup() raises TypeError if progress is an async function."""
        async with AsyncSQFox(str(tmp_path / "bak2.db")) as db:
            await db.write("CREATE TABLE t (val TEXT)", wait=True)

            async def async_progress(status, remaining, total):
                pass  # pragma: no cover

            with pytest.raises(TypeError, match="synchronous"):
                await db.backup(
                    str(tmp_path / "bak2_out.db"),
                    progress=async_progress,
                )


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
            diag = db.diagnostics()
            assert isinstance(diag, dict)
            assert "sqfox_version" in diag
            assert diag["is_running"] is True
            assert diag["path"] == str(tmp_path / "props.db")

    @pytest.mark.asyncio
    async def test_vector_backend_name_property(self, tmp_path):
        """vector_backend_name reflects configured backend."""
        async with AsyncSQFox(str(tmp_path / "vec.db"), vector_backend="hnsw") as db:
            assert db.vector_backend_name == "hnsw"


# ---------------------------------------------------------------------------
# execute_on_writer tests
# ---------------------------------------------------------------------------

class TestExecuteOnWriter:
    @pytest.mark.asyncio
    async def test_execute_on_writer(self, tmp_path):
        """execute_on_writer runs callable on writer connection."""
        async with AsyncSQFox(str(tmp_path / "eow.db")) as db:
            await db.write("CREATE TABLE t (val TEXT)", wait=True)
            result = await db.execute_on_writer(
                lambda conn: conn.execute(
                    "INSERT INTO t VALUES ('from_writer')"
                ).lastrowid
            )
            assert isinstance(result, int)
            row = await db.fetch_one("SELECT val FROM t")
            assert row is not None
            assert row[0] == "from_writer"


# ---------------------------------------------------------------------------
# on_startup hook tests
# ---------------------------------------------------------------------------

class TestOnStartup:
    @pytest.mark.asyncio
    async def test_on_startup_hook(self, tmp_path):
        """on_startup hook runs during start."""
        hook_called = []
        db = AsyncSQFox(str(tmp_path / "hook.db"))
        db.on_startup(lambda engine: hook_called.append(True))
        async with db:
            assert len(hook_called) == 1


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
