# sqfox

<p align="center">
  <img src="https://raw.githubusercontent.com/SiberXXX/sqfox/main/sqfox.jpg" alt="sqfox" width="300">
</p>

Встраиваемый SQLite микрофреймворк для гибридного поиска (FTS5 + векторы), IoT и RAG.

Без сервера. Без Docker. Без сети. Один файл `.db`. Работает на чём угодно — Raspberry Pi, промышленные ПК, старый ноутбук на Celeron, который пылится на антресоли.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**GitHub**: [github.com/SiberXXX/sqfox](https://github.com/SiberXXX/sqfox) | [English README](README.md)

## Установка

```bash
# Только ядро (ноль зависимостей, включает SqliteHnswBackend + SqliteFlatBackend)
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

Встроенные бэкенды — чистый Python, ноль C-зависимостей:

```python
from sqfox import SQFox, SqliteHnswBackend

# Flat — brute-force KNN, хорош до ~50K векторов
db = SQFox("app.db", vector_backend="flat")

# HNSW — O(log N), граф хранится в SQLite как BLOB
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
    def set_writer_conn(self, conn) -> None: ...
    def initialize(self, db_path: str, ndim: int) -> None: ...
    def add(self, keys: list[int], vectors: list[list[float]]) -> None: ...
    def remove(self, keys: list[int]) -> None: ...
    def search(self, query: list[float], k: int, **kwargs) -> list[tuple[int, float]]: ...
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

    mgr.drop("old_logs", delete_file=True)
```

### Метаданные

```python
db.ingest(
    "Процедура калибровки датчика давления",
    metadata={"source": "manual_v3", "equipment": "PS-100"},
    embed_fn=my_embed, wait=True,
)

results = db.search("калибровка", embed_fn=my_embed)
print(results[0].metadata)  # {"source": "manual_v3", "equipment": "PS-100"}
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

### Онлайн-бэкап

```python
with SQFox("app.db") as db:
    db.backup("backup.db")

    def progress(status, remaining, total):
        print(f"Скопировано {total - remaining}/{total} страниц")
    db.backup("backup.db", progress=progress)
```

### Vacuum

```python
with SQFox("app.db") as db:
    # Перезаписать файл БД, освободить место от удалённых строк
    db.vacuum()

    # Или записать компактную копию (без 2x места на диске)
    db.vacuum(into="compacted.db")
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

### Error Callback

```python
def on_error(sql: str, exc: Exception):
    with open("failed_queries.log", "a") as f:
        f.write(f"{sql}\n{exc}\n\n")

db = SQFox("app.db", error_callback=on_error)
```

Срабатывает при синтаксических ошибках, нарушениях ограничений, откатах батча и падении writer-потока. Без callback ошибки уходят в объект `Future` и Python `logging`.

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
     Flat-бэкенд          HNSW-граф
    (brute KNN)          (BLOB в SQLite)
```

- **Запись**: Все записи через `PriorityQueue` в один writer-поток. Батчинг в одной транзакции. Без `database is locked`.
- **Чтение**: Каждый поток получает свой connection через `threading.local()`. Параллельное, не блокирует запись.
- **Поиск**: FTS5 (BM25) и векторный бэкенд работают независимо. Слияние через Relative Score Fusion с адаптивным alpha.
- **Векторные бэкенды**: `SqliteFlatBackend` (brute-force) и `SqliteHnswBackend` (чистый Python HNSW, O(log N), граф сериализован как CSR BLOB в SQLite) — встроенные. `USearchBackend` — опциональный (pip install usearch). Кастомные бэкенды через протокол `VectorBackend`.
- **Crash safety**: Embedding BLOBы всегда хранятся в SQLite (источник истины). HNSW-граф — пересобираемый кэш. При повреждении автоматически восстанавливается из BLOB'ов при запуске.
- **Авто-адаптация**: Определяет RAM, CPU, платформу (Desktop / Raspberry Pi / Android Termux / SBC), SD-карту. PRAGMA настраиваются автоматически: LOW (<1 ГБ) → 4 МБ кэш, без mmap; MEDIUM (1–4 ГБ) → 16 МБ кэш, 64 МБ mmap; HIGH (>4 ГБ) → 64 МБ кэш, 256 МБ mmap. Векторный бэкенд авто-выбирается при первом `ingest()`. Инкрементальный vacuum работает в фоне. Ноль конфигурации.
- **Авто-PRAGMA**: WAL, `synchronous=NORMAL`, `busy_timeout=5000`, кэш/mmap настроены по RAM, `foreign_keys=ON`.
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

### Известные ограничения

- **SqliteHnswBackend** — чистый Python HNSW. Быстрее brute-force на 10K+ векторов, но медленнее C/Rust реализаций (usearch, hnswlib). Приемлемо для edge/embedded сценариев.
- **Лемматизация без контекста** — pymorphy3 выбирает самую частотную словоформу (~79% точность). Для поиска — нормально, не идеально.
- **`synchronous=NORMAL`** — последние несколько секунд коммитов могут потеряться при сбое питания. Файл БД не повредится. Для гарантированной durability: `PRAGMA synchronous=FULL`.
- **Износ SD-карт** — батчинг снижает количество fsync, но планируйте замену карты для долгих IoT-деплоев.

### Зона комфорта

Один процесс, много потоков. До ~50K документов с Flat-бэкендом, ~100K+ с HNSW, или ~1M строк без векторов. Нормально работает на двухъядерных Celeron, старых Core 2 Duo, Raspberry Pi — на всём, где есть Python 3.10+. Идеален для самопального умного дома, управления котлом, логирования датчиков, локальных баз знаний. Если думаете «может мне нужен PostgreSQL» — скорее всего нужен. sqfox для случаев, когда PostgreSQL — оверкилл или невозможен.

## Сценарии использования

| Сценарий | Описание | Ключевые фичи |
|---|---|---|
| **DIY умный дом / автоматизация** | Старый ноут или мини-ПК на Celeron — управление котлом, логирование датчиков, климат. Без облака, без подписки, просто Python + sqfox на железе, которое уже есть | Параллельная запись, WAL, батчинг, бэкап, Grafana через SQLite-плагин |
| **Telegram / Discord бот** | Локальная база знаний для ботов, десктопных утилит, CLI-ассистентов, офлайн-помощников | AsyncSQFox, гибридный поиск, чанкинг, метаданные, реранкинг |
| **Промышленный IoT + Grafana** | Edge-шлюзы (Raspberry Pi, Jetson, промышленные ПК). Grafana читает тот же `.db` через [frser-sqlite-datasource](https://grafana.com/grafana/plugins/frser-sqlite-datasource/) | Параллельная запись, WAL, батчинг, бэкап |
| **Самодиагностирующийся edge-агент** | Два AsyncSQFox: телеметрия (I/O-пул) + база знаний (CPU-пул). Порог срабатывает авто-RAG поиск | Двухпульная изоляция, авто-RAG |
| **OBD-II умный механик** | ELM327 читает данные двигателя, sqfox автоматически ищет в сервисном мануале по DTC-кодам | См. [`demo/obd_smart_mechanic.py`](demo/obd_smart_mechanic.py) |

## Диагностика

```python
with SQFox("app.db") as db:
    print(db.diagnostics())
```

```json
{
  "sqfox_version": "0.3.0",
  "python_version": "3.13.7",
  "platform": "Linux-6.1.0-rpi-aarch64",
  "sqlite_version": "3.50.4",
  "path": "app.db",
  "is_running": true,
  "vector_backend": "hnsw",
  "queue_size": 0,
  "schema_state": "ENRICHED",
  "auto": {
    "total_ram_mb": 512,
    "memory_tier": "LOW",
    "cpu_count": 4,
    "platform_class": "RASPBERRY_PI",
    "is_sd_card": false,
    "fts5_available": true,
    "resolved_cache_size_kb": 4000,
    "resolved_mmap_size_mb": 0
  }
}
```

## Авто-адаптация (0.3.0+)

sqfox **сам определяет** среду при запуске и настраивается:

| Что | Как | Fallback |
|---|---|---|
| RAM | `/proc/meminfo`, `sysctl`, `GlobalMemoryStatusEx` | 1 ГБ |
| CPU | `sched_getaffinity` (Docker-aware), `cpu_count()` | 1 |
| Платформа | `TERMUX_VERSION` env, `/proc/device-tree/model`, arch | Desktop |
| SD-карта | Эвристика по пути (`/media/`, `/mnt/sd` и т.д.) | Нет |
| FTS5 | Проба в `:memory:` | Только векторный поиск |

Настройка PRAGMA по тиру памяти:

| Тир | RAM | cache_size | mmap_size |
|---|---|---|---|
| LOW | < 1 ГБ | 4 МБ | 0 (отключён) |
| MEDIUM | 1–4 ГБ | 16 МБ | 64 МБ |
| HIGH | > 4 ГБ | 64 МБ | 256 МБ |

**Ноль конфигурации.** Просто `SQFox("data.db")` — работает на Raspberry Pi, Android-телефоне и 64 ГБ десктопе.

Для ручного переопределения:

```python
# Принудительные настройки (отключают авто для этих параметров)
db = SQFox(
    "sensors.db",
    cache_size_kb=8_000,    # вместо авто: 8 МБ
    mmap_size_mb=0,         # вместо авто: отключить mmap
)
```

Дополнительно:

- **SD-карты**: `synchronous=NORMAL` (по умолчанию) может потерять последнюю транзакцию при выдёргивании питания (файл не повредится). Для гарантий: `db.write("PRAGMA synchronous=FULL", wait=True)` после старта.
- **Диск**: sqfox включает инкрементальный auto-vacuum на новых базах и освобождает страницы в фоне при ingestion. Для ручной компактификации: `db.vacuum()` или `db.vacuum(into="compacted.db")`.
- **Android / Termux**: FTS5 зависит от сборки Python. Большинство сборок Kivy, BeeWare и Termux включают FTS5. Если FTS5 недоступен — sqfox определяет это и работает только с векторным поиском. Без падений, без конфигурации.
- **Graceful shutdown**: sqfox регистрирует `atexit`-хук — writer-поток штатно завершается при нормальном выходе процесса. При `SIGKILL` / выдёргивании питания WAL-журнал защищает файл БД.
- **Самовосстановление FTS5**: При каждом запуске sqfox проверяет целостность FTS-индекса. Если индекс повреждён — автоматически пересобирает.

## Поддержка платформ

| Платформа | Статус |
|---|---|
| Linux x86-64 | работает |
| Linux ARM64 (Raspberry Pi, промышленные ПК) | работает |
| macOS Intel / Apple Silicon | работает |
| Windows x86-64 | работает |
| Alpine Linux (musl) | работает |
| 32-bit / Windows ARM64 | работает (чистый Python, без C-расширений) |

Все векторные бэкенды — чистый Python без C-зависимостей, поэтому sqfox работает на любой платформе с Python 3.10+.

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
- Ядро: ноль зависимостей (только stdlib), включает `SqliteHnswBackend` и `SqliteFlatBackend`
- `[search]`: simplemma
- `[search-ru]`: + pymorphy3
- `[search-hnsw]`: + usearch, numpy (опциональный USearch-бэкенд)

## Лицензия

MIT
