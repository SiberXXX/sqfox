# Changelog

## 0.2.2

- Исправлена ссылка на русский README (404 на PyPI)

## 0.2.1

- Добавлена ссылка Changelog в PyPI project URLs

## 0.2.0

### Added

- **Pluggable vector backends** — `VectorBackend` protocol, параметр `vector_backend=` в `SQFox` / `AsyncSQFox`
- **SqliteHnswBackend** — чистый Python HNSW, O(log N), ноль C-зависимостей, граф хранится как CSR BLOB в SQLite
- **Crash recovery** — embedding BLOBы в SQLite = source of truth; при повреждении HNSW-графа автоматическая пересборка из BLOB'ов при запуске
- **Backend registry** — фабрика `get_backend()`, алиасы `"hnsw"`, `"sqlite-vec"`
- `VectorBackendError` — новый тип ошибки
- `vector_backend_name` property в `SQFox` / `AsyncSQFox`
- `vector_backend` поле в `diagnostics()`
- Колонка `embedding BLOB` + `vec_indexed` в таблице `documents`
- Демо: `run_hnsw_xray.py` — интерактивный инспектор HNSW-графа с трассировкой поиска
- Демо: `run_crash_recovery.py` — повреждение и восстановление HNSW-индекса

### Fixed

- **async_engine**: `write()` и `execute_on_writer()` больше не блокируют event loop на `_write_lock` (обёрнуты в `asyncio.to_thread`)
- **async_engine**: `wait=False` теперь возвращает `asyncio.Future` вместо `concurrent.futures.Future`
- **engine**: `flush()` бэкенда в `_do_ingest` обёрнут в try/except — ошибка flush не убивает writer thread
- **engine**: краш writer thread теперь резолвит pending futures из текущего батча
- **engine**: `stop()` — `backend.close()` вызывается ДО `writer_conn.close()`
- **hnsw**: re-insert после delete больше не оставляет ноду в `_deleted`
- **hnsw**: дублирующий `add()` не завышает `_count`
- **hnsw**: `_serialize()` пишет реальное количество активных нод, не потенциально устаревший `_count`
- **hnsw**: `_deleted` и `_node_levels` чистятся после `flush()`
- **hnsw**: corrupt BLOB при `initialize()` — warning + пустой граф вместо крэша

### Changed

- Eviction в HNSW vec-кэше через `itertools.islice` вместо полного списка ключей
- `rebuild_from_blobs` в USearch-бэкенде — батчевая обработка (2000 шт.) вместо одного гигантского списка
- `ensure_schema` в `AsyncSQFox` использует `asyncio.to_thread` вместо `loop.run_in_executor`

## 0.1.3

- Хардинг engine: улучшенная обработка ошибок, очистка кода
- Переписан README

## 0.1.2

- Тяжёлые вычисления (chunking, lemmatization, embedding) вынесены из writer thread в `ingest()`

## 0.1.1

- Исправлен URL маскота для PyPI

## 0.1.0

- Первый релиз: SQFox engine, AsyncSQFox, SQFoxManager
- Гибридный поиск: FTS5 (BM25) + sqlite-vec (KNN), Relative Score Fusion
- Чанкеры: sentence, paragraph, markdown, recursive, html_to_text
- Online backup, cross-encoder reranking
- Dual thread pool в AsyncSQFox (I/O + CPU)
- Автоматическая эволюция схемы, лемматизация RU+EN
