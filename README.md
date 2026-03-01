# sqfox

<p align="center">
  <img src="https://raw.githubusercontent.com/SiberXXX/sqfox/main/sqfox.jpg" alt="sqfox mascot" width="300">
</p>

Lightweight, thread-safe SQLite wrapper with hybrid search (FTS5 + vectors).

No server. No Docker. No network. One `.db` file. From ESP32 to industrial PCs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**GitHub**: [github.com/SiberXXX/sqfox](https://github.com/SiberXXX/sqfox)

## Install

```bash
# Core only (zero dependencies)
pip install sqfox

# With hybrid search (English)
pip install sqfox[search]

# With hybrid search (English + Russian)
pip install sqfox[search-ru]
```

## Quick Start

### Basic: Thread-Safe Writes + Reads

```python
from sqfox import SQFox

with SQFox("app.db") as db:
    # Create table
    db.write("CREATE TABLE sensors (ts TEXT, value REAL)", wait=True)

    # Non-blocking writes from any thread
    db.write("INSERT INTO sensors VALUES (?, ?)", ("2026-03-01T10:00:00", 23.5))
    db.write("INSERT INTO sensors VALUES (?, ?)", ("2026-03-01T10:01:00", 24.1))

    # Reads — parallel, non-blocking
    rows = db.fetch_all("SELECT * FROM sensors ORDER BY ts DESC LIMIT 10")
```

All writes go through a single writer thread with batching. Multiple threads can read in parallel via WAL mode. PRAGMA tuning (WAL, synchronous, busy_timeout, etc.) is applied automatically.

### Hybrid Search: FTS + Vectors

```python
from sqfox import SQFox

def my_embed(texts):
    """Your embedding function — any model, any API."""
    # return list of float vectors
    ...

def my_chunker(text):
    """Split document into chunks. IMPORTANT for large documents."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

with SQFox("knowledge.db") as db:
    # Ingest with chunking — each chunk gets its own vector and FTS entry
    db.ingest(long_document, chunker=my_chunker, embed_fn=my_embed, wait=True)

    # Short texts can be ingested without chunking
    db.ingest("Short note about SQLite WAL mode", embed_fn=my_embed, wait=True)

    # Hybrid search: FTS5 (exact terms) + vectors (semantics) + score fusion
    results = db.search("database optimization", embed_fn=my_embed)
    for r in results:
        print(f"[{r.score:.3f}] {r.text}")
```

> **Always use a chunker for documents longer than ~500 words.** Without chunking, the entire document becomes a single vector and a single FTS entry. This kills search precision — a 10-page document about many topics will match every query weakly instead of matching the right paragraph strongly.

### Instruction-Aware Models (Qwen3, E5, BGE)

Models that produce different embeddings for documents vs queries are supported natively:

```python
from sentence_transformers import SentenceTransformer

class QwenEmbedder:
    def __init__(self, dim=256):
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", truncate_dim=dim
        )

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

### Chunking

sqfox includes built-in chunkers — no need to write your own:

```python
from sqfox import SQFox, sentence_chunker, markdown_chunker, html_to_text

with SQFox("docs.db") as db:
    # Split by sentences with overlap (best for articles, documentation)
    db.ingest(text, chunker=sentence_chunker(chunk_size=500, overlap=1), embed_fn=my_embed, wait=True)

    # Split markdown by headers
    db.ingest(md_text, chunker=markdown_chunker(max_level=2), embed_fn=my_embed, wait=True)

    # HTML from web scraping: strip tags first, then chunk
    clean = html_to_text(raw_html)
    db.ingest(clean, chunker=sentence_chunker(), embed_fn=my_embed, wait=True)
```

Available chunkers: `sentence_chunker`, `paragraph_chunker`, `markdown_chunker`, `recursive_chunker`. Plus `html_to_text` preprocessor for web content. Custom callables `(str) -> list[str]` also work.

### Multi-Database Manager

```python
from sqfox import SQFoxManager

with SQFoxManager("./databases") as mgr:
    # Each domain — separate .db file, separate writer thread
    mgr.ingest_to("sensors", "Temperature: 25.3C", embed_fn=my_embed, wait=True)
    mgr.ingest_to("manuals", "Replace bearing at 50k RPM", embed_fn=my_embed, wait=True)

    # Cross-database search
    results = mgr.search_all("bearing vibration", embed_fn=my_embed)
    for db_name, r in results:
        print(f"[{db_name}] [{r.score:.3f}] {r.text}")

    # Cleanup
    mgr.drop("old_logs", delete_file=True)
```

### Metadata

```python
db.ingest(
    "Pressure sensor calibration procedure",
    metadata={"source": "manual_v3", "equipment": "PS-100"},
    embed_fn=my_embed,
    wait=True,
)

results = db.search("calibration", embed_fn=my_embed)
print(results[0].metadata)  # {"source": "manual_v3", "equipment": "PS-100"}
```

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
```

**Writes**: All writes from all threads go through a `PriorityQueue` to a single writer thread. The writer groups them into batches and executes in one transaction (`BEGIN IMMEDIATE`). This eliminates `database is locked` errors.

**Reads**: Each thread gets its own read connection via `threading.local()`. Reads run in parallel and don't block writes (WAL mode).

**Search**: FTS5 (lemmatized text, BM25 scoring) and sqlite-vec (vector KNN) run independently. Results are merged via Relative Score Fusion with adaptive alpha weighting.

## Auto-PRAGMA

Every connection is automatically configured:

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-64000;    -- 64 MB
PRAGMA mmap_size=268435456;  -- 256 MB
PRAGMA foreign_keys=ON;
```

## Schema State Machine

Schema evolves automatically based on what you do:

```
EMPTY → BASE → INDEXED → SEARCHABLE → ENRICHED
         |         |          |            |
     documents  + vec0     + FTS5      + triggers
     + metadata  table      table      (auto-sync)
```

No manual migrations. Idempotent. Resumable backfill for existing data.

## Multilingual Lemmatization

Mixed Russian + English text is handled per-word:

```python
from sqfox.tokenizer import lemmatize

lemmatize("настройка database для серверов running")
# → "настройка database для сервер run"
#    ^^^^^^^^                 ^^^^^^
#    pymorphy3 (RU)          simplemma (EN)
```

Each word is detected by script (Cyrillic/Latin) and sent to the right lemmatizer.

## Error Callback (Dead Letter Queue)

Fire-and-forget writes (`wait=False`) don't lose errors silently. Register a callback to catch failed writes:

```python
def on_error(sql: str, exc: Exception):
    # Log to file, send to Sentry, save to DLQ — your choice
    with open("failed_queries.log", "a") as f:
        f.write(f"{sql}\n{exc}\n\n")

db = SQFox("app.db", error_callback=on_error)
```

The callback receives the SQL statement and the exception. It fires on:
- Syntax errors
- Constraint violations (UNIQUE, NOT NULL, FK)
- Batch aborts (when one statement kills the whole batch)
- Writer thread crashes

Without a callback, errors still go to the `Future` object and Python `logging`.

## Diagnostics

Debug platform issues (e.g., sqlite-vec not loading on Alpine Linux):

```python
with SQFox("app.db") as db:
    print(db.diagnostics())
```

```json
{
  "sqfox_version": "0.1.0",
  "python_version": "3.13.7",
  "platform": "Linux-6.1.0-rpi-aarch64",
  "machine": "aarch64",
  "sqlite_version": "3.50.4",
  "vec_available": true,
  "sqlite_vec_version": "0.1.6",
  "simplemma_version": "1.1.2",
  "pymorphy3_version": "2.0.6"
}
```

## Platform Support

sqfox core (zero dependencies) works everywhere Python runs.

sqlite-vec (optional, for vector search) has prebuilt wheels for:

| Platform | pip install | Notes |
|---|---|---|
| Linux x86-64 | works | |
| Linux ARM64 | works | Raspberry Pi OS 64-bit, industrial PCs |
| macOS Intel | works | |
| macOS Apple Silicon | works | |
| Windows x86-64 | works | |
| Alpine Linux (musl) | **no** | No musllinux wheel. Use `python:3.x-slim` instead |
| Linux/Windows 32-bit | **no** | No 32-bit wheels |
| Windows ARM64 | **no** | No wheel available |

If sqlite-vec is not available, sqfox falls back to FTS-only search and logs a warning with platform details.

## Limitations (read this before choosing sqfox)

sqfox is **not a replacement for PostgreSQL**. It is designed for a specific niche. If your project doesn't fit — use a real database server.

### When NOT to use sqfox

| Requirement | sqfox | What to use instead |
|---|---|---|
| Multiple processes writing to one DB | **No.** Single-process only. The in-memory queue doesn't work across processes. | PostgreSQL, MySQL |
| More than ~100K vectors | **Slow.** sqlite-vec is brute-force O(n), no ANN index. 100K vectors @ 256 dim = ~100MB scan per query. | pgvector, Qdrant, Milvus |
| More than ~1M rows with complex queries | **Possible but not ideal.** SQLite is single-writer, no parallel writes. | PostgreSQL |
| Multi-server / distributed | **No.** One file, one machine. | PostgreSQL + pgvector, Elasticsearch |
| High-availability / replication | **No.** File copy is the only backup. | PostgreSQL, CockroachDB |
| Concurrent writes from multiple machines | **No.** SQLite doesn't support network access to the .db file. | PostgreSQL, MySQL |

### Known technical limitations

- **sqlite-vec is pre-v1.** There are known memory leaks (issue #265) and segfaults on invalid input (issue #245). Don't use it in production without testing on your data.
- **Brute-force vector search.** No HNSW, no IVF, no LSH. Every KNN query scans all vectors. Acceptable up to ~50-100K vectors depending on dimension and hardware.
- **Lemmatization without context.** pymorphy3 analyzes words in isolation. The word "stali" (Russian) can mean "steel" or "became" — pymorphy3 picks the most frequent one (~79% accuracy). For search this is usually fine, but not perfect.
- **`synchronous=NORMAL` loses recent data on power loss.** The database file won't be corrupted, but the last few seconds of committed transactions may be rolled back. If you need guaranteed durability, set `busy_timeout_ms` and use `synchronous=FULL` manually.
- **No async/await.** sqfox uses threads, not asyncio. The write queue is fast enough that `db.write()` doesn't block, but `db.search()` blocks the calling thread for the duration of the search.
- **SD card wear.** Frequent writes to SD cards degrade flash memory. Batching helps (fewer transactions = fewer fsyncs), but plan for card replacement in long-running IoT deployments.

### Sweet spot

sqfox works best when:
- Single process, multiple threads (web server, IoT gateway, desktop app)
- Up to ~50K documents with vectors, or ~1M rows without vectors
- Local/embedded deployment (no network DB needed)
- You need hybrid search (text + vectors) without infrastructure

If you're thinking "maybe I need PostgreSQL" — you probably do. sqfox is for cases where PostgreSQL is overkill or impossible (edge devices, offline systems, rapid prototyping).

## Use Cases

**IoT / Edge**: Write sensor data from 10+ threads without `database is locked`. Auto-batching for SD cards. Works on Raspberry Pi and industrial PCs (Intel Atom).

**RAG / AI**: Local vector + full-text search without PostgreSQL. Ingest documents, search with hybrid scoring. Supports instruction-aware embedding models (Qwen3, E5, BGE).

**Multi-domain**: Separate databases per domain (sensors, manuals, logs). Cross-search when needed. Backup = copy file. Delete domain = delete file.

## Requirements

- Python >= 3.10
- Core: zero dependencies (stdlib only)
- `[search]`: simplemma, sqlite-vec
- `[search-ru]`: + pymorphy3

## License

MIT

---

# sqfox (RU)

<p align="center">
  <img src="https://raw.githubusercontent.com/SiberXXX/sqfox/main/sqfox.jpg" alt="sqfox — скуфокс" width="300">
</p>

Легковесная, потокобезопасная обёртка над SQLite с гибридным поиском (FTS5 + векторы).

Без сервера. Без Docker. Без сети. Один файл `.db`. От ESP32 до промышленных ПК.

## Установка

```bash
# Только ядро (ноль зависимостей)
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
    # Создаём таблицу
    db.write("CREATE TABLE sensors (ts TEXT, value REAL)", wait=True)

    # Неблокирующая запись из любого потока
    db.write("INSERT INTO sensors VALUES (?, ?)", ("2026-03-01T10:00:00", 23.5))
    db.write("INSERT INTO sensors VALUES (?, ?)", ("2026-03-01T10:01:00", 24.1))

    # Чтение — параллельное, не блокирует запись
    rows = db.fetch_all("SELECT * FROM sensors ORDER BY ts DESC LIMIT 10")
```

Все записи идут через один writer-поток с батчингом. Несколько потоков читают параллельно через WAL. PRAGMA настраиваются автоматически.

### Гибридный поиск: FTS + векторы

```python
from sqfox import SQFox

def my_embed(texts):
    """Ваша функция эмбеддингов — любая модель, любой API."""
    # возвращает список float-векторов
    ...

def my_chunker(text):
    """Разбиваем документ на чанки. ВАЖНО для больших документов."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

with SQFox("knowledge.db") as db:
    # Загрузка с чанкингом — каждый чанк получает свой вектор и FTS-запись
    db.ingest(long_document, chunker=my_chunker, embed_fn=my_embed, wait=True)

    # Короткие тексты можно загружать без чанкинга
    db.ingest("Краткая заметка про SQLite WAL", embed_fn=my_embed, wait=True)

    # Гибридный поиск: FTS5 (точные слова) + векторы (семантика) + слияние скоров
    results = db.search("оптимизация базы данных", embed_fn=my_embed)
    for r in results:
        print(f"[{r.score:.3f}] {r.text}")
```

> **Всегда используйте чанкер для документов длиннее ~500 слов.** Без чанкинга весь документ становится одним вектором и одной FTS-записью. Это убивает точность поиска — документ на 10 страниц обо всём будет слабо матчиться на любой запрос, вместо того чтобы сильно матчиться на правильный абзац.

### Instruction-Aware модели (Qwen3, E5, BGE)

Модели, которые создают разные эмбеддинги для документов и запросов, поддерживаются нативно:

```python
from sentence_transformers import SentenceTransformer

class QwenEmbedder:
    def __init__(self, dim=256):
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", truncate_dim=dim
        )

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text, prompt_name="query").tolist()

embedder = QwenEmbedder()

with SQFox("rag.db") as db:
    db.ingest("Текст документа...", embed_fn=embedder, wait=True)
    results = db.search("текст запроса", embed_fn=embedder)
```

sqfox определяет наличие методов `embed_documents` / `embed_query` и вызывает нужный автоматически. Обычные функции тоже работают.

### Чанкинг

sqfox включает готовые чанкеры — не нужно писать свои:

```python
from sqfox import SQFox, sentence_chunker, markdown_chunker, html_to_text

with SQFox("docs.db") as db:
    # Разбиение по предложениям с перекрытием (статьи, документация)
    db.ingest(text, chunker=sentence_chunker(chunk_size=500, overlap=1), embed_fn=my_embed, wait=True)

    # Разбиение markdown по заголовкам
    db.ingest(md_text, chunker=markdown_chunker(max_level=2), embed_fn=my_embed, wait=True)

    # HTML из веб-скрапинга: сначала очистка, потом чанкинг
    clean = html_to_text(raw_html)
    db.ingest(clean, chunker=sentence_chunker(), embed_fn=my_embed, wait=True)
```

Доступные чанкеры: `sentence_chunker`, `paragraph_chunker`, `markdown_chunker`, `recursive_chunker`. Плюс `html_to_text` для веб-контента. Свои функции `(str) -> list[str]` тоже работают.

### Менеджер нескольких баз

```python
from sqfox import SQFoxManager

with SQFoxManager("./databases") as mgr:
    # Каждый домен — отдельный .db файл, отдельный writer-поток
    mgr.ingest_to("sensors", "Температура: 25.3C", embed_fn=my_embed, wait=True)
    mgr.ingest_to("manuals", "Замена подшипника при 50к об/мин", embed_fn=my_embed, wait=True)

    # Поиск по всем базам
    results = mgr.search_all("вибрация подшипник", embed_fn=my_embed)
    for db_name, r in results:
        print(f"[{db_name}] [{r.score:.3f}] {r.text}")

    # Удаление базы с файлом
    mgr.drop("old_logs", delete_file=True)
```

### Метаданные

```python
db.ingest(
    "Процедура калибровки датчика давления",
    metadata={"source": "manual_v3", "equipment": "PS-100"},
    embed_fn=my_embed,
    wait=True,
)

results = db.search("калибровка", embed_fn=my_embed)
print(results[0].metadata)  # {"source": "manual_v3", "equipment": "PS-100"}
```

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
```

**Запись**: Все записи из всех потоков идут через `PriorityQueue` в один writer-поток. Writer группирует их в батчи и выполняет в одной транзакции (`BEGIN IMMEDIATE`). Это устраняет ошибки `database is locked`.

**Чтение**: Каждый поток получает свой read-connection через `threading.local()`. Чтение параллельное и не блокирует запись (WAL).

**Поиск**: FTS5 (лемматизированный текст, BM25) и sqlite-vec (KNN по векторам) работают независимо. Результаты сливаются через Relative Score Fusion с адаптивным весом alpha.

## Автонастройка PRAGMA

Каждое соединение автоматически настраивается:

```sql
PRAGMA journal_mode=WAL;       -- Параллельное чтение/запись
PRAGMA synchronous=NORMAL;     -- Баланс скорости и надёжности
PRAGMA busy_timeout=5000;      -- 5 сек ожидания при блокировке
PRAGMA temp_store=MEMORY;      -- Временные таблицы в RAM
PRAGMA cache_size=-64000;      -- 64 MB кэш
PRAGMA mmap_size=268435456;    -- 256 MB memory-mapped I/O
PRAGMA foreign_keys=ON;        -- Внешние ключи
```

## Автомиграция схемы

Схема эволюционирует автоматически:

```
EMPTY -> BASE -> INDEXED -> SEARCHABLE -> ENRICHED
          |         |           |             |
      documents  + vec0      + FTS5       + триггеры
      + metadata  таблица     таблица     (авто-синк)
```

Без ручных миграций. Идемпотентно. Возобновляемый backfill существующих данных.

## Мультиязычная лемматизация

Смешанный русско-английский текст обрабатывается пословно:

```python
from sqfox.tokenizer import lemmatize

lemmatize("настройка database для серверов running")
# -> "настройка database для сервер run"
#    ^^^^^^^^                 ^^^^^^
#    pymorphy3 (RU)          simplemma (EN)
```

Каждое слово определяется по скрипту (кириллица/латиница) и отправляется в нужный лемматизатор.

## Error Callback (Dead Letter Queue)

Fire-and-forget записи (`wait=False`) не теряют ошибки молча. Зарегистрируйте callback для перехвата:

```python
def on_error(sql: str, exc: Exception):
    # Лог в файл, отправка в Sentry, сохранение в DLQ
    with open("failed_queries.log", "a") as f:
        f.write(f"{sql}\n{exc}\n\n")

db = SQFox("app.db", error_callback=on_error)
```

Callback получает SQL-запрос и исключение. Срабатывает при:
- Синтаксических ошибках
- Нарушениях ограничений (UNIQUE, NOT NULL, FK)
- Откате батча (когда один запрос убивает весь батч)
- Падении writer-потока

Без callback ошибки всё равно уходят в объект `Future` и стандартный `logging` Python.

## Диагностика

Отладка проблем платформы (например, sqlite-vec не грузится на Alpine Linux):

```python
with SQFox("app.db") as db:
    print(db.diagnostics())
```

```json
{
  "sqfox_version": "0.1.0",
  "python_version": "3.13.7",
  "platform": "Linux-6.1.0-rpi-aarch64",
  "machine": "aarch64",
  "sqlite_version": "3.50.4",
  "vec_available": true,
  "sqlite_vec_version": "0.1.6",
  "simplemma_version": "1.1.2",
  "pymorphy3_version": "2.0.6"
}
```

## Поддержка платформ

Ядро sqfox (ноль зависимостей) работает везде, где работает Python.

sqlite-vec (опционально, для векторного поиска) имеет готовые wheel для:

| Платформа | pip install | Примечания |
|---|---|---|
| Linux x86-64 | работает | |
| Linux ARM64 | работает | Raspberry Pi OS 64-bit, промышленные ПК |
| macOS Intel | работает | |
| macOS Apple Silicon | работает | |
| Windows x86-64 | работает | |
| Alpine Linux (musl) | **нет** | Нет musllinux wheel. Используйте `python:3.x-slim` |
| Linux/Windows 32-bit | **нет** | Нет 32-bit wheels |
| Windows ARM64 | **нет** | Нет wheel |

Если sqlite-vec недоступен, sqfox откатывается на FTS-only поиск и выводит предупреждение с деталями платформы.

## Ограничения (прочитайте перед выбором sqfox)

sqfox — **не замена PostgreSQL**. Он спроектирован для конкретной ниши. Если ваш проект не вписывается — используйте нормальный сервер БД.

### Когда НЕ использовать sqfox

| Требование | sqfox | Что использовать |
|---|---|---|
| Несколько процессов пишут в одну БД | **Нет.** Только single-process. Очередь в памяти не работает между процессами. | PostgreSQL, MySQL |
| Больше ~100K векторов | **Медленно.** sqlite-vec — brute-force O(n), нет ANN-индекса. 100K векторов @ 256 dim = ~100MB скан на запрос. | pgvector, Qdrant, Milvus |
| Больше ~1M строк со сложными запросами | **Можно, но не идеально.** SQLite — single-writer, нет параллельной записи. | PostgreSQL |
| Несколько серверов / распределённая система | **Нет.** Один файл, одна машина. | PostgreSQL + pgvector, Elasticsearch |
| Высокая доступность / репликация | **Нет.** Бэкап = копирование файла, и всё. | PostgreSQL, CockroachDB |
| Параллельная запись с нескольких машин | **Нет.** SQLite не поддерживает сетевой доступ к .db файлу. | PostgreSQL, MySQL |

### Известные технические ограничения

- **sqlite-vec — pre-v1.** Есть известные утечки памяти (issue #265) и сегфолты на невалидном вводе (issue #245). Не используйте в продакшене без тестирования на ваших данных.
- **Brute-force векторный поиск.** Нет HNSW, нет IVF, нет LSH. Каждый KNN-запрос сканирует все векторы. Приемлемо до ~50-100K векторов в зависимости от размерности и железа.
- **Лемматизация без контекста.** pymorphy3 анализирует слова изолированно. Слово "стали" может означать "сталь" или "стать" — pymorphy3 выбирает самый частотный вариант (~79% точности). Для поиска обычно достаточно, но не идеально.
- **`synchronous=NORMAL` теряет данные при обрыве питания.** Файл БД не побьётся, но последние несколько секунд коммитов могут откатиться. Если нужна гарантированная durability — используйте `synchronous=FULL` вручную.
- **Нет async/await.** sqfox использует потоки, не asyncio. `db.write()` не блокирует (очередь быстрая), но `db.search()` блокирует вызывающий поток на время поиска.
- **Износ SD-карт.** Частая запись на SD-карту убивает flash-память. Батчинг помогает (меньше транзакций = меньше fsync), но планируйте замену карт в длительных IoT-развёртываниях.

### Зона комфорта

sqfox работает лучше всего когда:
- Один процесс, много потоков (веб-сервер, IoT-шлюз, десктопное приложение)
- До ~50K документов с векторами, или ~1M строк без векторов
- Локальное/встроенное развёртывание (не нужна сетевая БД)
- Нужен гибридный поиск (текст + векторы) без инфраструктуры

Если вы думаете "может мне нужен PostgreSQL" — скорее всего нужен. sqfox для случаев, когда PostgreSQL — оверкилл или невозможен (edge-устройства, offline-системы, быстрое прототипирование).

## Сценарии использования

**IoT / Edge**: Запись с 10+ датчиков одновременно без `database is locked`. Автобатчинг для SD-карт. Работает на Raspberry Pi и промышленных ПК (Intel Atom).

**RAG / AI**: Локальный векторный + полнотекстовый поиск без PostgreSQL. Загрузка документов, гибридный поиск. Поддержка instruction-aware моделей (Qwen3, E5, BGE).

**Мульти-домен**: Отдельные базы на домен (датчики, документация, логи). Кросс-поиск при необходимости. Бэкап = копирование файла. Удаление домена = удаление файла.

## Зависимости

- Python >= 3.10
- Ядро: ноль зависимостей (только stdlib)
- `[search]`: simplemma, sqlite-vec
- `[search-ru]`: + pymorphy3

## Лицензия

MIT
