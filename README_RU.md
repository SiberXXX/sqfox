# sqfox

<p align="center">
  <img src="https://raw.githubusercontent.com/SiberXXX/sqfox/main/sqfox.jpg" alt="sqfox" width="300">
</p>

Встраиваемый SQLite микрофреймворк для гибридного поиска (FTS5 + векторы), IoT и RAG.

Без сервера. Без Docker. Без сети. Один файл `.db`. Работает на чём угодно — Raspberry Pi, промышленные ПК, старый ноутбук на Celeron, который пылится на антресоли.

**GitHub**: [github.com/SiberXXX/sqfox](https://github.com/SiberXXX/sqfox) | [English README](README.md)

## Установка

```bash
# Только ядро (ноль зависимостей, включает SqliteHnswBackend)
pip install sqfox

# С гибридным поиском (английский)
pip install sqfox[search]

# С гибридным поиском (английский + русский)
pip install sqfox[search-ru]
```

## Быстрый старт

### Потокобезопасная запись и чтение

```python
from sqfox import SQFox

with SQFox("app.db") as db:
    db.write("CREATE TABLE sensors (ts TEXT, value REAL)", wait=True)

    # Неблокирующая запись из любого потока
    db.write("INSERT INTO sensors VALUES (?, ?)", ("2026-03-01T10:00:00", 23.5))

    # Чтение — параллельное, не блокирует запись
    rows = db.fetch_all("SELECT * FROM sensors ORDER BY ts DESC LIMIT 10")
```

Все записи идут через один writer-поток с батчингом. Несколько потоков читают параллельно через WAL. PRAGMA настраиваются автоматически.

### Гибридный поиск: FTS + векторы

```python
from sqfox import SQFox

def my_embed(texts):
    """Ваша функция эмбеддинга — любая модель, любой API."""
    raise NotImplementedError("Реализуйте вашу функцию эмбеддинга")

with SQFox("knowledge.db") as db:
    db.ingest("Краткая заметка про SQLite WAL", embed_fn=my_embed, wait=True)

    results = db.search("оптимизация базы данных", embed_fn=my_embed)
    for r in results:
        print(f"[{r.score:.3f}] {r.text}")
```

> **Всегда используйте чанкер для документов длиннее ~500 слов.** Без чанкинга весь документ становится одним вектором — это убивает точность поиска.

### Instruction-Aware модели (Qwen3, E5, BGE)

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
    db.ingest("Текст документа...", embed_fn=embedder, wait=True)
    results = db.search("текст запроса", embed_fn=embedder)
```

sqfox определяет наличие методов `embed_documents` / `embed_query` и вызывает нужный автоматически.

### Подключаемые векторные бэкенды

По умолчанию sqfox использует sqlite-vec (brute-force KNN). Для больших датасетов — встроенный HNSW-бэкенд, чистый Python, ноль C-зависимостей, O(log N) поиск:

```python
from sqfox import SQFox, SqliteHnswBackend

# По умолчанию — sqlite-vec (brute-force, хорош до ~50K векторов)
db = SQFox("app.db")

# HNSW — O(log N), чистый Python, граф хранится в SQLite как BLOB
db = SQFox("app.db", vector_backend="hnsw")

# HNSW с кастомными параметрами
backend = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)
db = SQFox("app.db", vector_backend=backend)
```

HNSW-граф хранится вместе с документами в одном файле `.db`. SQLite — источник истины: если граф потерян или повреждён, он автоматически пересобирается из embedding BLOB'ов при следующем запуске.

**Свой бэкенд** — реализуйте протокол `VectorBackend`:

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

### Чанкинг

```python
from sqfox import SQFox, sentence_chunker, markdown_chunker, html_to_text

with SQFox("docs.db") as db:
    db.ingest(text, chunker=sentence_chunker(chunk_size=500, overlap=1), embed_fn=my_embed, wait=True)
    db.ingest(md_text, chunker=markdown_chunker(max_level=2), embed_fn=my_embed, wait=True)

    clean = html_to_text(raw_html)
    db.ingest(clean, chunker=sentence_chunker(), embed_fn=my_embed, wait=True)
```

Доступные чанкеры: `sentence_chunker`, `paragraph_chunker`, `markdown_chunker`, `recursive_chunker`, `html_to_text`.

### Менеджер нескольких баз

```python
from sqfox import SQFoxManager

with SQFoxManager("./databases") as mgr:
    mgr.ingest_to("sensors", "Температура: 25.3C", embed_fn=my_embed, wait=True)
    mgr.ingest_to("manuals", "Замена подшипника при 50к об/мин", embed_fn=my_embed, wait=True)

    results = mgr.search_all("вибрация подшипник", embed_fn=my_embed)
    for db_name, r in results:
        print(f"[{db_name}] [{r.score:.3f}] {r.text}")
```

### Реранкинг (Cross-Encoder)

```python
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def my_reranker(query: str, texts: list[str]) -> list[float]:
    pairs = [(query, t) for t in texts]
    return reranker_model.predict(pairs).tolist()

results = db.search("оптимизация", embed_fn=my_embed, reranker_fn=my_reranker, rerank_top_n=20)
```

### Async (FastAPI / asyncio)

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

Два пула: **I/O-пул** для `fetch_one/fetch_all/backup`, **CPU-пул** для `ingest/search` с эмбеддингами. Даже при 200 параллельных запросах только `max_cpu_workers` эмбеддингов считаются одновременно.

> **Потокобезопасность моделей:** Если модель не поддерживает параллельные вызовы, ставьте `max_cpu_workers=1`.

> **Single process:** Используйте `uvicorn app:app --workers 1`. Для multi-worker — PostgreSQL.

## Как это работает

```
Writer-поток (один)             Reader-потоки (много)
       |                              |
  PriorityQueue                 threading.local()
       |                              |
  Батчинг (N записей            Один connection
  в одной транзакции)           на каждый поток
       |                              |
  Writer Connection             Reader Connections
       |                              |
       +--------- SQLite WAL ---------+
                     |
              Один файл .db
                     |
          +----------+----------+
          |                     |
     sqlite-vec            HNSW-граф
     (brute KNN)         (BLOB в SQLite)
```

- **Запись**: Все записи через `PriorityQueue` в один writer-поток. Батчинг в одной транзакции. Без `database is locked`.
- **Чтение**: Каждый поток получает свой connection через `threading.local()`. Параллельное, не блокирует запись.
- **Поиск**: FTS5 (BM25) и векторный бэкенд (sqlite-vec или HNSW) работают независимо. Слияние через Relative Score Fusion с адаптивным alpha.
- **Векторные бэкенды**: `sqlite-vec` (brute-force, по умолчанию) или `SqliteHnswBackend` (чистый Python HNSW, O(log N), граф сериализован как CSR BLOB в SQLite). Кастомные бэкенды через протокол `VectorBackend`.
- **Crash safety**: Embedding BLOBы всегда хранятся в SQLite (источник истины). HNSW-граф — пересобираемый кэш. При повреждении автоматически восстанавливается из BLOB'ов при запуске.
- **Авто-PRAGMA**: WAL, `synchronous=NORMAL`, `busy_timeout=5000`, `cache_size=64MB`, `mmap_size=256MB`, `foreign_keys=ON`.
- **Схема**: Эволюционирует автоматически — `EMPTY -> BASE -> INDEXED -> SEARCHABLE -> ENRICHED`. Без ручных миграций.
- **Лемматизация**: Смешанный RU+EN текст — pymorphy3 для кириллицы, simplemma для латиницы.

## Ограничения

sqfox — **не замена PostgreSQL**. Он для конкретной ниши.

| Требование | Используйте |
|---|---|
| Несколько процессов пишут в одну БД | PostgreSQL, MySQL |
| Больше ~500K векторов | pgvector, Qdrant, Milvus |
| Распределённая система / HA | PostgreSQL, Elasticsearch |
| >10K точек/сек | InfluxDB, ClickHouse |

**Зона комфорта**: один процесс, много потоков. До ~50K документов с sqlite-vec, ~100K+ с HNSW-бэкендом, или ~1M строк без векторов. Нормально работает на двухъядерных Celeron, старых Core 2 Duo, Raspberry Pi — на всём, где есть Python 3.10+. Идеален для самопального умного дома, управления котлом, логирования датчиков, локальных баз знаний. Если думаете «может мне нужен PostgreSQL» — скорее всего нужен. sqfox для случаев, когда PostgreSQL — оверкилл или невозможен.

## Демо

```bash
# IoT + RAG (требует sentence-transformers)
python demo/run_demo.py

# HNSW Рентген — интерактивный инспектор графа (требует sentence-transformers)
python demo/run_hnsw_xray.py

# Crash & Recovery — повреждение и восстановление HNSW (требует sentence-transformers)
python demo/run_crash_recovery.py

# Умный дом — IoT-эмуляция
python demo/run_smart_home.py

# OBD-II умный механик
python demo/run_smart_mechanic.py
```

## Зависимости

- Python >= 3.10
- Ядро: ноль зависимостей (только stdlib)
- `[search]`: simplemma, sqlite-vec
- `[search-ru]`: + pymorphy3
- `SqliteHnswBackend`: включён в ядро (чистый Python, без доп. зависимостей)

## Лицензия

MIT
