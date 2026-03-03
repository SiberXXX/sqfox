# sqfox

<p align="center">
  <img src="https://raw.githubusercontent.com/SiberXXX/sqfox/main/sqfox.jpg" alt="sqfox mascot" width="300">
</p>

Embedded SQLite micro-framework for hybrid search (FTS5 + vectors), IoT and RAG.

No server. No Docker. No network. One `.db` file. Runs on anything — Raspberry Pi, industrial PCs, that old Celeron laptop collecting dust on a shelf.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**GitHub**: [github.com/SiberXXX/sqfox](https://github.com/SiberXXX/sqfox) | [README на русском](README_RU.md)

## Install

```bash
# Core only (zero dependencies, includes SqliteHnswBackend)
pip install sqfox

# With hybrid search (English)
pip install sqfox[search]

# With hybrid search (English + Russian)
pip install sqfox[search-ru]
```

## Quick Start

### Thread-Safe Writes + Reads

```python
from sqfox import SQFox

with SQFox("app.db") as db:
    db.write("CREATE TABLE sensors (ts TEXT, value REAL)", wait=True)

    # Non-blocking writes from any thread
    db.write("INSERT INTO sensors VALUES (?, ?)", ("2026-03-01T10:00:00", 23.5))

    # Reads — parallel, non-blocking
    rows = db.fetch_all("SELECT * FROM sensors ORDER BY ts DESC LIMIT 10")
```

All writes go through a single writer thread with batching. Multiple threads can read in parallel via WAL mode. PRAGMAs are tuned automatically.

### Hybrid Search: FTS + Vectors

```python
from sqfox import SQFox

def my_embed(texts):
    """Your embedding function — any model, any API."""
    raise NotImplementedError("Implement your embedding function")

with SQFox("knowledge.db") as db:
    db.ingest("Short note about SQLite WAL mode", embed_fn=my_embed, wait=True)

    results = db.search("database optimization", embed_fn=my_embed)
    for r in results:
        print(f"[{r.score:.3f}] {r.text}")
```

> **Always use a chunker for documents longer than ~500 words.** Without chunking, the entire document becomes a single vector — this kills search precision.

### Instruction-Aware Models (Qwen3, E5, BGE)

```python
from sentence_transformers import SentenceTransformer

class QwenEmbedder:
    def __init__(self, dim=256):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", truncate_dim=dim)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text, prompt_name="query").tolist()

embedder = QwenEmbedder()

with SQFox("rag.db") as db:
    db.ingest("Document text...", embed_fn=embedder, wait=True)
    results = db.search("query text", embed_fn=embedder)
```

sqfox detects `embed_documents` / `embed_query` methods and calls the right one automatically. Plain callables work too.

### Pluggable Vector Backends

By default, sqfox uses sqlite-vec (brute-force KNN). For larger datasets, plug in the built-in HNSW backend — pure Python, zero C dependencies, O(log N) search:

```python
from sqfox import SQFox, SqliteHnswBackend

# Default — sqlite-vec (brute-force, good up to ~50K vectors)
db = SQFox("app.db")

# HNSW — O(log N), pure Python, graph stored in SQLite as BLOB
db = SQFox("app.db", vector_backend="hnsw")

# HNSW with custom parameters
backend = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)
db = SQFox("app.db", vector_backend=backend)
```

The HNSW graph is stored alongside documents in the same `.db` file. SQLite is the source of truth — if the graph is lost or corrupted, it rebuilds automatically from embedding BLOBs on next startup.

**Write your own backend** — implement the `VectorBackend` protocol:

```python
from sqfox import VectorBackend

class MyBackend:
    def initialize(self, db_path: str, ndim: int) -> None: ...
    def add(self, keys: list[int], vectors: list[list[float]]) -> None: ...
    def remove(self, keys: list[int]) -> None: ...
    def search(self, query: list[float], k: int) -> list[tuple[int, float]]: ...
    def flush(self) -> None: ...
    def count(self) -> int: ...
    def close(self) -> None: ...

db = SQFox("app.db", vector_backend=MyBackend())
```

### Chunking

```python
from sqfox import SQFox, sentence_chunker, markdown_chunker, html_to_text

with SQFox("docs.db") as db:
    db.ingest(text, chunker=sentence_chunker(chunk_size=500, overlap=1), embed_fn=my_embed, wait=True)
    db.ingest(md_text, chunker=markdown_chunker(max_level=2), embed_fn=my_embed, wait=True)

    clean = html_to_text(raw_html)
    db.ingest(clean, chunker=sentence_chunker(), embed_fn=my_embed, wait=True)
```

Available: `sentence_chunker`, `paragraph_chunker`, `markdown_chunker`, `recursive_chunker`, `html_to_text`. Custom `(str) -> list[str]` callables also work.

### Multi-Database Manager

```python
from sqfox import SQFoxManager

with SQFoxManager("./databases") as mgr:
    mgr.ingest_to("sensors", "Temperature: 25.3C", embed_fn=my_embed, wait=True)
    mgr.ingest_to("manuals", "Replace bearing at 50k RPM", embed_fn=my_embed, wait=True)

    results = mgr.search_all("bearing vibration", embed_fn=my_embed)
    for db_name, r in results:
        print(f"[{db_name}] [{r.score:.3f}] {r.text}")

    mgr.drop("old_logs", delete_file=True)
```

### Metadata

```python
db.ingest(
    "Pressure sensor calibration procedure",
    metadata={"source": "manual_v3", "equipment": "PS-100"},
    embed_fn=my_embed, wait=True,
)

results = db.search("calibration", embed_fn=my_embed)
print(results[0].metadata)  # {"source": "manual_v3", "equipment": "PS-100"}
```

### Reranking (Cross-Encoder)

Two-stage retrieval: fast hybrid search retrieves candidates, then a cross-encoder re-scores them:

```python
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def my_reranker(query: str, texts: list[str]) -> list[float]:
    pairs = [(query, t) for t in texts]
    return reranker_model.predict(pairs).tolist()

results = db.search(
    "database optimization",
    embed_fn=my_embed,
    reranker_fn=my_reranker,
    rerank_top_n=20,
)
```

### Online Backup

```python
with SQFox("app.db") as db:
    db.backup("backup.db")

    def progress(status, remaining, total):
        print(f"Copied {total - remaining}/{total} pages")
    db.backup("backup.db", progress=progress)
```

### Async (FastAPI / asyncio)

`AsyncSQFox` — async facade with dual thread pools. I/O reads are never blocked by CPU-heavy embedding/reranking:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqfox import AsyncSQFox

db = AsyncSQFox("data.db", max_cpu_workers=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with db:
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/ingest")
async def ingest(text: str):
    doc_id = await db.ingest(text, embed_fn=my_embed)
    return {"id": doc_id}

@app.get("/search")
async def search(q: str):
    return await db.search(q, embed_fn=my_embed)
```

Two pools: **I/O pool** (default asyncio executor) for `fetch_one/fetch_all/backup`, **CPU pool** (limited `ThreadPoolExecutor`) for `ingest/search` with embeddings. Even with 200 concurrent ingests, only `max_cpu_workers` embeddings run at once.

> **Thread safety:** If your model doesn't support concurrent calls, set `max_cpu_workers=1`.

> **Single process:** Use `uvicorn app:app --workers 1`. For multi-worker, use PostgreSQL.

### Error Callback

```python
def on_error(sql: str, exc: Exception):
    with open("failed_queries.log", "a") as f:
        f.write(f"{sql}\n{exc}\n\n")

db = SQFox("app.db", error_callback=on_error)
```

Fires on syntax errors, constraint violations, batch aborts, and writer thread crashes. Without a callback, errors still go to the `Future` object and Python `logging`.

## How It Works

```
Writer Thread (single)          Reader Threads (many)
       |                              |
  PriorityQueue                 threading.local()
       |                              |
  Batching (N writes             One connection
  per transaction)               per thread
       |                              |
  Writer Connection             Reader Connections
       |                              |
       +--------- SQLite WAL ---------+
                     |
              One .db file
                     |
          +----------+----------+
          |                     |
     sqlite-vec            HNSW graph
     (brute KNN)         (BLOB in SQLite)
```

- **Writes**: All writes go through a `PriorityQueue` to a single writer thread. Batched into one transaction (`BEGIN IMMEDIATE`). No `database is locked` errors.
- **Reads**: Each thread gets its own connection via `threading.local()`. Parallel, non-blocking (WAL mode).
- **Search**: FTS5 (BM25) and vector backend (sqlite-vec or HNSW) run independently. Merged via Relative Score Fusion with adaptive alpha.
- **Vector backends**: `sqlite-vec` (brute-force, default) or `SqliteHnswBackend` (pure Python HNSW, O(log N), graph serialized as CSR BLOB in SQLite). Custom backends via the `VectorBackend` protocol.
- **Crash safety**: Embedding BLOBs are always stored in SQLite (source of truth). The HNSW graph is a rebuildable cache — if corrupted, it auto-recovers from BLOBs on startup.
- **Auto-PRAGMA**: WAL, `synchronous=NORMAL`, `busy_timeout=5000`, `temp_store=MEMORY`, `cache_size=-64000` (64 MB), `mmap_size=256MB`, `foreign_keys=ON`.
- **Schema**: Evolves automatically — `EMPTY → BASE → INDEXED → SEARCHABLE → ENRICHED`. No manual migrations. Idempotent. Resumable backfill.
- **Lemmatization**: Mixed RU+EN text handled per-word — pymorphy3 for Cyrillic, simplemma for Latin.

## Use Cases

| Scenario | Description | Key features |
|---|---|---|
| **DIY smart home / automation** | Old laptop or Celeron mini-PC on a shelf — boiler control, sensor logging, climate automation. No cloud, no subscription, just Python + sqfox on hardware you already own | Concurrent writes, WAL, batching, backup, Grafana via SQLite plugin |
| **Telegram / Discord bot** | Local knowledge base for bots, desktop tools, CLI utilities, offline assistants | AsyncSQFox, hybrid search, chunking, metadata, reranking |
| **Industrial IoT + Grafana** | Edge gateways (Raspberry Pi, Jetson, industrial PCs). Grafana reads the same `.db` via [frser-sqlite-datasource](https://grafana.com/grafana/plugins/frser-sqlite-datasource/) | Concurrent writes, WAL, batching, backup |
| **Self-diagnosing edge agent** | Two AsyncSQFox instances: telemetry (I/O pool) + knowledge base (CPU pool). Threshold triggers auto-RAG search | Dual-pool isolation, auto-RAG |
| **OBD-II smart mechanic** | ELM327 reads engine data, sqfox auto-searches service manual on DTC codes | See [`demo/obd_smart_mechanic.py`](demo/obd_smart_mechanic.py) |

## Limitations

sqfox is **not a replacement for PostgreSQL**. It is designed for a specific niche.

### When NOT to use sqfox

| Requirement | Use instead |
|---|---|
| Multiple processes writing to one DB | PostgreSQL, MySQL |
| More than ~500K vectors | pgvector, Qdrant, Milvus |
| Multi-server / distributed / HA | PostgreSQL, Elasticsearch, CockroachDB |
| High-frequency data (>10K points/sec) | InfluxDB, ClickHouse |

### Known limitations

- **sqlite-vec is pre-v1** — known memory leaks and segfaults on invalid input. Test on your data before production.
- **SqliteHnswBackend** — pure Python HNSW. Faster than brute-force on 10K+ vectors, but slower than C/Rust implementations (usearch, hnswlib). Acceptable for edge/embedded scenarios.
- **Lemmatization without context** — pymorphy3 picks the most frequent word form (~79% accuracy). Fine for search, not perfect.
- **`synchronous=NORMAL`** — last few seconds of commits may be lost on power failure. DB file won't corrupt. Use `synchronous=FULL` manually if you need guaranteed durability.
- **SD card wear** — batching reduces fsyncs, but plan card replacement for long-running IoT deployments.

### Sweet spot

Single process, multiple threads. Up to ~50K docs with sqlite-vec, ~100K+ with HNSW backend, or ~1M rows without vectors. Local/embedded deployment. Runs fine on dual-core Celerons, old Core 2 Duo laptops, Raspberry Pi — anything with Python 3.10+. Perfect for DIY home automation, boiler controllers, sensor logging, local knowledge bases. If you're thinking "maybe I need PostgreSQL" — you probably do. sqfox is for cases where PostgreSQL is overkill or impossible.

## Diagnostics

```python
with SQFox("app.db") as db:
    print(db.diagnostics())
```

```json
{
  "sqfox_version": "0.2.0",
  "python_version": "3.13.7",
  "platform": "Linux-6.1.0-rpi-aarch64",
  "sqlite_version": "3.50.4",
  "path": "app.db",
  "is_running": true,
  "vec_available": true,
  "vector_backend": "hnsw",
  "queue_size": 0,
  "schema_state": "ENRICHED"
}
```

## Platform Support

| Platform | Status |
|---|---|
| Linux x86-64 | works |
| Linux ARM64 (Raspberry Pi, industrial PCs) | works |
| macOS Intel / Apple Silicon | works |
| Windows x86-64 | works |
| Alpine Linux (musl) | **no** — no musllinux wheel, use `python:3.x-slim` |
| 32-bit / Windows ARM64 | **no** |

If sqlite-vec is unavailable, sqfox falls back to FTS-only search (or use `SqliteHnswBackend` which has no C dependencies).

## Demos

```bash
# IoT + RAG (requires sentence-transformers)
python demo/run_demo.py

# HNSW X-Ray — interactive graph inspector (requires sentence-transformers)
python demo/run_hnsw_xray.py

# Crash & Recovery — corrupt HNSW graph, auto-rebuild (requires sentence-transformers)
python demo/run_crash_recovery.py

# Smart home IoT emulation
python demo/run_smart_home.py

# OBD-II smart mechanic
python demo/run_smart_mechanic.py
```

## Requirements

- Python >= 3.10
- Core: zero dependencies (stdlib only)
- `[search]`: simplemma, sqlite-vec
- `[search-ru]`: + pymorphy3
- `SqliteHnswBackend`: included in core (pure Python, no extra deps)

## License

MIT
