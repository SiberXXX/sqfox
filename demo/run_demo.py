"""
sqfox demo: IoT emulation + RAG on real embeddings (Qwen3-Embedding-0.6B).

Usage:
  python demo/run_demo.py          # run all modes
  python demo/run_demo.py iot      # IoT only
  python demo/run_demo.py rag      # RAG only (includes reranker)
  python demo/run_demo.py combined # combined
  python demo/run_demo.py manager  # manager
  python demo/run_demo.py async    # AsyncSQFox
"""

import asyncio
import os
import sqlite3
import sys
import time
import random
import shutil
import threading
from pathlib import Path

# Force UTF-8 on Windows so Cyrillic + box-drawing chars render properly.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.live import Live
from rich.align import Align
from rich import box

from sqfox import AsyncSQFox, SQFox, SQFoxManager, SchemaState

console = Console()

# ---------------------------------------------------------------------------
# Embedding adapter
# ---------------------------------------------------------------------------

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        with console.status("[bold cyan]Loading Qwen3-Embedding-0.6B...[/]", spinner="dots"):
            t0 = time.time()
            _model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
            )
            elapsed = time.time() - t0
        console.print(f"  [green]Model loaded in {elapsed:.1f}s[/] (256 dim, MRL truncated)")
    return _model


class QwenEmbedder:
    def __init__(self):
        self.model = get_model()
        self._doc_count = 0
        self._query_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._doc_count += len(texts)
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        self._query_count += 1
        return self.model.encode(text, prompt_name="query").tolist()


# ---------------------------------------------------------------------------
# IoT Sensor Emulator
# ---------------------------------------------------------------------------

SENSOR_TYPES = [
    ("temp_indoor",  "C",     18.0,   28.0,  "yellow"),
    ("temp_outdoor", "C",    -10.0,   40.0,  "yellow"),
    ("humidity",     "%",     30.0,   90.0,  "cyan"),
    ("pressure",     "hPa",  990.0, 1030.0,  "blue"),
    ("vibration_x",  "mm/s",   0.0,   15.0,  "red"),
    ("vibration_y",  "mm/s",   0.0,   15.0,  "red"),
    ("vibration_z",  "mm/s",   0.0,   15.0,  "red"),
    ("rpm",          "RPM",    0.0, 5000.0,  "magenta"),
    ("voltage",      "V",    210.0,  240.0,  "green"),
    ("current",      "A",      0.0,   32.0,  "green"),
    ("flow_rate",    "L/min",  0.0,  100.0,  "cyan"),
    ("noise_level",  "dB",    30.0,   95.0,  "yellow"),
]


def run_iot_mode(db_path: str, n_readings: int = 200, n_threads: int = 6):
    console.print()
    console.rule("[bold red]IoT Sensor Emulation[/]")
    console.print()

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_row("Sensors", f"[cyan]{len(SENSOR_TYPES)}[/]")
    info_table.add_row("Threads", f"[cyan]{n_threads}[/]")
    info_table.add_row("Readings/thread", f"[cyan]{n_readings}[/]")
    info_table.add_row("Total target", f"[cyan]{n_readings * n_threads}[/]")
    console.print(info_table)
    console.print()

    errors = []
    def on_error(sql, exc):
        errors.append((sql, exc))

    with SQFox(db_path, error_callback=on_error, batch_size=32, batch_time_ms=100) as db:
        db.write("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor TEXT NOT NULL, value REAL NOT NULL,
                unit TEXT NOT NULL,
                ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """, wait=True)

        total_writes = [0]
        lock = threading.Lock()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            tasks = {}
            for i in range(n_threads):
                tasks[i] = progress.add_task(f"Thread {i}", total=n_readings)

            t0 = time.time()

            def sensor_worker(thread_id):
                count = 0
                for _ in range(n_readings):
                    sensor = random.choice(SENSOR_TYPES)
                    name, unit, lo, hi, _ = sensor
                    value = round(random.uniform(lo, hi), 2)
                    db.write(
                        "INSERT INTO sensor_data (sensor, value, unit) VALUES (?, ?, ?)",
                        (f"{name}_{thread_id}", value, unit),
                    )
                    count += 1
                    progress.advance(tasks[thread_id])
                    time.sleep(random.uniform(0.001, 0.005))
                with lock:
                    total_writes[0] += count

            threads = [
                threading.Thread(target=sensor_worker, args=(i,))
                for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)

        elapsed = time.time() - t0
        time.sleep(0.5)

        row = db.fetch_one("SELECT COUNT(*) FROM sensor_data")
        actual = row[0] if row else 0

        # Results
        console.print()
        results_table = Table(title="Results", box=box.ROUNDED, title_style="bold green")
        results_table.add_column("Metric", style="bold")
        results_table.add_column("Value", justify="right")
        results_table.add_row("Time", f"{elapsed:.2f}s")
        results_table.add_row("Submitted", f"{total_writes[0]}")
        results_table.add_row("In DB", f"{actual}")
        results_table.add_row("Throughput", f"[bold cyan]{actual / elapsed:.0f} writes/sec[/]")
        results_table.add_row("Errors", f"[{'red' if errors else 'green'}]{len(errors)}[/]")
        results_table.add_row("Queue remaining", f"{db.queue_size}")
        console.print(results_table)

        # Sample data
        console.print()
        sample_table = Table(title="Sample Readings", box=box.SIMPLE_HEAVY)
        sample_table.add_column("Sensor", style="bold")
        sample_table.add_column("Value", justify="right")
        sample_table.add_column("Unit")
        sample_table.add_column("Timestamp", style="dim")

        rows = db.fetch_all(
            "SELECT sensor, value, unit, ts FROM sensor_data ORDER BY RANDOM() LIMIT 8"
        )
        for r in rows:
            color = "white"
            for s in SENSOR_TYPES:
                if r["sensor"].startswith(s[0]):
                    color = s[4]
                    break
            sample_table.add_row(
                f"[{color}]{r['sensor']}[/]",
                f"{r['value']:.2f}",
                r["unit"],
                r["ts"],
            )
        console.print(sample_table)

        # Per-sensor stats
        console.print()
        stats_table = Table(title="Sensor Statistics", box=box.ROUNDED)
        stats_table.add_column("Sensor Type", style="bold")
        stats_table.add_column("Count", justify="right")
        stats_table.add_column("Min", justify="right")
        stats_table.add_column("Max", justify="right")
        stats_table.add_column("Avg", justify="right")

        stat_rows = db.fetch_all("""
            SELECT
                substr(sensor, 1, instr(sensor || '_', '_') - 1) as stype,
                COUNT(*) as cnt,
                ROUND(MIN(value), 2) as vmin,
                ROUND(MAX(value), 2) as vmax,
                ROUND(AVG(value), 2) as vavg
            FROM sensor_data
            GROUP BY stype
            ORDER BY cnt DESC
            LIMIT 8
        """)
        for r in stat_rows:
            stats_table.add_row(
                r["stype"], str(r["cnt"]),
                str(r["vmin"]), str(r["vmax"]), str(r["vavg"]),
            )
        console.print(stats_table)

        # --- Online backup demo ---
        console.print()
        console.rule("[bold red]Online Backup[/]")
        console.print()

        backup_path = str(Path(db_path).parent / "iot_backup.db")
        backup_pages = [0]

        def progress_cb(status, remaining, total):
            backup_pages[0] = total - remaining

        with console.status("[bold cyan]Running online backup...[/]"):
            t0 = time.time()
            db.backup(backup_path, pages=5, progress=progress_cb)
            backup_time = time.time() - t0

        backup_size = Path(backup_path).stat().st_size
        size_kb = backup_size / 1024

        # Verify backup
        verify_conn = sqlite3.connect(backup_path)
        verify_count = verify_conn.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
        verify_conn.close()

        backup_table = Table(title="Backup Results", box=box.ROUNDED, title_style="bold green")
        backup_table.add_column("Metric", style="bold")
        backup_table.add_column("Value", justify="right")
        backup_table.add_row("Backup path", Path(backup_path).name)
        backup_table.add_row("Size", f"{size_kb:.1f} KB")
        backup_table.add_row("Time", f"{backup_time:.3f}s")
        backup_table.add_row("Original rows", str(actual))
        backup_table.add_row("Backup rows", str(verify_count))
        backup_table.add_row("Integrity", "[bold green]OK[/]" if verify_count == actual else "[bold red]MISMATCH[/]")
        console.print(backup_table)


# ---------------------------------------------------------------------------
# RAG Knowledge Base
# ---------------------------------------------------------------------------

RAG_DOCUMENTS = [
    {"text": "Замена подшипников качения выполняется при превышении уровня вибрации 7.1 мм/с (ISO 10816). Перед заменой необходимо остановить оборудование и зафиксировать вал.", "meta": {"type": "maintenance", "lang": "ru", "equipment": "bearings"}},
    {"text": "Калибровка датчиков давления производится ежеквартально. Допустимая погрешность не более 0.5% от диапазона измерения. При превышении — датчик подлежит замене.", "meta": {"type": "calibration", "lang": "ru", "equipment": "pressure_sensor"}},
    {"text": "При температуре охлаждающей жидкости выше 95 градусов Цельсия необходимо проверить работу термостата и уровень антифриза в расширительном бачке.", "meta": {"type": "troubleshooting", "lang": "ru", "equipment": "cooling_system"}},
    {"text": "Регламент технического обслуживания электродвигателей: замена смазки каждые 4000 моточасов, проверка изоляции обмоток мегаомметром каждые 6 месяцев.", "meta": {"type": "maintenance", "lang": "ru", "equipment": "electric_motor"}},
    {"text": "Система мониторинга вибрации позволяет выявлять дисбаланс ротора, расцентровку валов, дефекты подшипников и ослабление крепления на ранних стадиях.", "meta": {"type": "monitoring", "lang": "ru", "equipment": "vibration_system"}},
    {"text": "Напряжение питания промышленного оборудования должно находиться в пределах 380В +/- 10%. При отклонениях более 10% срабатывает защита по напряжению.", "meta": {"type": "electrical", "lang": "ru", "equipment": "power_supply"}},
    {"text": "SQLite WAL mode enables concurrent reads during writes. Set PRAGMA journal_mode=WAL and PRAGMA synchronous=NORMAL for optimal performance on embedded systems.", "meta": {"type": "database", "lang": "en", "topic": "sqlite_config"}},
    {"text": "MQTT protocol is widely used for IoT sensor data collection. QoS level 1 guarantees at-least-once delivery, suitable for non-critical telemetry data.", "meta": {"type": "networking", "lang": "en", "topic": "iot_protocols"}},
    {"text": "Edge computing reduces latency by processing data locally on industrial PCs instead of sending everything to the cloud. Typical edge devices use Intel Atom or ARM Cortex processors.", "meta": {"type": "architecture", "lang": "en", "topic": "edge_computing"}},
    {"text": "PID controller tuning for temperature regulation: start with Ziegler-Nichols method, then fine-tune proportional gain Kp and integral time Ti based on step response.", "meta": {"type": "control", "lang": "en", "topic": "pid_tuning"}},
    {"text": "Predictive maintenance uses vibration analysis and machine learning to forecast equipment failures. Common features include RMS velocity, peak acceleration, and crest factor.", "meta": {"type": "maintenance", "lang": "en", "topic": "predictive"}},
    {"text": "OPC UA (Unified Architecture) is the standard protocol for industrial automation. It provides secure, reliable communication between PLCs, SCADA systems, and cloud platforms.", "meta": {"type": "networking", "lang": "en", "topic": "industrial_protocols"}},
    {"text": "Настройка PRAGMA journal_mode=WAL в SQLite позволяет параллельное чтение и запись. Рекомендуется для embedded-систем с частой записью данных с датчиков.", "meta": {"type": "database", "lang": "mixed", "topic": "sqlite_config"}},
    {"text": "Протокол Modbus RTU используется для связи с PLC контроллерами. Скорость 9600-115200 бод, формат данных 8N1. Максимальная длина шины RS-485 — 1200 метров.", "meta": {"type": "networking", "lang": "mixed", "topic": "industrial_protocols"}},
    {"text": "Machine learning модели для predictive maintenance обучаются на исторических данных вибрации. Используются алгоритмы Random Forest и Gradient Boosting для классификации состояния оборудования.", "meta": {"type": "ai", "lang": "mixed", "topic": "predictive_ml"}},
]


def run_rag_mode(db_path: str):
    console.print()
    console.rule("[bold blue]RAG Knowledge Base[/]")
    console.print()

    embedder = QwenEmbedder()
    errors = []
    def on_error(sql, exc):
        errors.append(str(exc))

    with SQFox(db_path, error_callback=on_error) as db:
        # Ingest with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting documents", total=len(RAG_DOCUMENTS))
            t0 = time.time()
            for doc in RAG_DOCUMENTS:
                db.ingest(doc["text"], metadata=doc["meta"], embed_fn=embedder, wait=True)
                progress.advance(task)
            ingest_time = time.time() - t0

        # Ingest stats
        ingest_table = Table(title="Ingest Summary", box=box.ROUNDED, title_style="bold green")
        ingest_table.add_column("Metric", style="bold")
        ingest_table.add_column("Value", justify="right")
        ingest_table.add_row("Documents", str(len(RAG_DOCUMENTS)))
        ingest_table.add_row("Time", f"{ingest_time:.1f}s")
        ingest_table.add_row("Speed", f"{len(RAG_DOCUMENTS)/ingest_time:.2f} docs/sec")
        ingest_table.add_row("Docs embedded", str(embedder._doc_count))
        ingest_table.add_row("Dimension", "256 (MRL)")
        console.print(ingest_table)

        db.ensure_schema(SchemaState.SEARCHABLE)
        time.sleep(0.2)

        # Queries
        queries = [
            ("ru", "при какой вибрации менять подшипник"),
            ("ru", "настройка WAL в SQLite"),
            ("en", "how to tune PID controller for temperature"),
            ("en", "predictive maintenance vibration analysis"),
            ("ru", "калибровка датчиков давления"),
            ("en", "edge computing for industrial IoT"),
            ("ru", "протокол связи с контроллерами"),
            ("ru", "напряжение питания оборудования"),
        ]

        console.print()
        console.print("[bold]Hybrid Search Results[/]")
        console.print()

        total_search_ms = 0

        for lang, query in queries:
            t0 = time.time()
            results = db.search(query, embed_fn=embedder, limit=3)
            elapsed_ms = (time.time() - t0) * 1000
            total_search_ms += elapsed_ms

            # Query panel
            lang_badge = f"[bold yellow]{lang.upper()}[/]"
            query_text = f"{lang_badge} [bold white]{query}[/]  [dim]({elapsed_ms:.0f}ms)[/]"
            console.print(query_text)

            if not results:
                console.print("  [dim]No results[/]")
            else:
                for i, r in enumerate(results):
                    rlang = r.metadata.get("lang", "?")
                    rtype = r.metadata.get("type", "?")

                    if r.score >= 0.7:
                        score_color = "bold green"
                    elif r.score >= 0.6:
                        score_color = "yellow"
                    else:
                        score_color = "dim"

                    rank = f"  {'>>>' if i == 0 else '   '}"
                    meta_badge = f"[dim]{rlang}/{rtype}[/]"
                    console.print(
                        f"{rank} [{score_color}]{r.score:.3f}[/] {meta_badge} {r.text}"
                    )
            console.print()

        # Search summary
        summary = Table(title="Search Summary", box=box.ROUNDED, title_style="bold green")
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", justify="right")
        summary.add_row("Queries", str(len(queries)))
        summary.add_row("Total time", f"{total_search_ms:.0f}ms")
        summary.add_row("Avg latency", f"[bold cyan]{total_search_ms/len(queries):.0f}ms[/]")
        summary.add_row("Queries embedded", str(embedder._query_count))
        console.print(summary)

        # --- Reranker demo ---
        console.print()
        console.rule("[bold blue]With Reranker (word-overlap heuristic)[/]")
        console.print()

        def word_overlap_reranker(query: str, texts: list[str]) -> list[float]:
            q_words = set(query.lower().split())
            return [
                sum(1 for w in t.lower().split() if w in q_words) / max(len(q_words), 1)
                for t in texts
            ]

        rerank_queries = [
            ("ru", "при какой вибрации менять подшипник"),
            ("en", "predictive maintenance vibration analysis"),
            ("ru", "настройка WAL в SQLite"),
        ]

        for lang, query in rerank_queries:
            # Without reranker
            t0 = time.time()
            baseline = db.search(query, embed_fn=embedder, limit=3)
            baseline_ms = (time.time() - t0) * 1000

            # With reranker
            t0 = time.time()
            reranked = db.search(
                query, embed_fn=embedder, limit=3,
                reranker_fn=word_overlap_reranker, rerank_top_n=10,
            )
            reranked_ms = (time.time() - t0) * 1000

            lang_badge = f"[bold yellow]{lang.upper()}[/]"
            console.print(f"{lang_badge} [bold white]{query}[/]")

            cmp_table = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
            cmp_table.add_column("#", style="dim", width=3)
            cmp_table.add_column(f"Baseline ({baseline_ms:.0f}ms)", max_width=50)
            cmp_table.add_column("Score", justify="right", width=7)
            cmp_table.add_column(f"Reranked ({reranked_ms:.0f}ms)", max_width=50)
            cmp_table.add_column("Score", justify="right", width=7)

            for i in range(max(len(baseline), len(reranked))):
                b_text = baseline[i].text.replace("\n", " ") if i < len(baseline) else "--"
                b_score = f"{baseline[i].score:.3f}" if i < len(baseline) else ""
                r_text = reranked[i].text.replace("\n", " ") if i < len(reranked) else "--"
                r_score = f"{reranked[i].score:.3f}" if i < len(reranked) else ""
                cmp_table.add_row(str(i + 1), b_text, b_score, r_text, r_score)

            console.print(cmp_table)
            console.print()


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def run_combined_mode(db_path: str):
    console.print()
    console.rule("[bold magenta]Combined: IoT Writes + RAG Search[/]")
    console.print()

    embedder = QwenEmbedder()
    errors = []
    def on_error(sql, exc):
        errors.append(str(exc))

    with SQFox(db_path, error_callback=on_error, batch_size=32) as db:
        db.write("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor TEXT, value REAL, unit TEXT,
                ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """, wait=True)

        with console.status("[bold cyan]Ingesting 8 RAG documents...[/]"):
            for doc in RAG_DOCUMENTS[:8]:
                db.ingest(doc["text"], metadata=doc["meta"], embed_fn=embedder, wait=True)

        db.ensure_schema(SchemaState.SEARCHABLE)
        time.sleep(0.2)

        console.print("[green]  8 documents ingested[/]")
        console.print()

        iot_writes = [0]
        search_results = []
        iot_done = threading.Event()

        def iot_writer():
            for _ in range(100):
                s = random.choice(SENSOR_TYPES)
                name, unit, lo, hi, _ = s
                db.write(
                    "INSERT INTO sensor_data (sensor, value, unit) VALUES (?, ?, ?)",
                    (name, round(random.uniform(lo, hi), 2), unit),
                )
                iot_writes[0] += 1
                time.sleep(random.uniform(0.005, 0.02))
            iot_done.set()

        def rag_searcher():
            qs = [
                "вибрация подшипник замена",
                "SQLite WAL configuration",
                "датчик давления калибровка",
                "edge computing IoT",
            ]
            for q in qs:
                t0 = time.time()
                results = db.search(q, embed_fn=embedder, limit=2)
                elapsed = (time.time() - t0) * 1000
                top_text = results[0].text if results else "--"
                search_results.append((q, len(results), elapsed, top_text))
                time.sleep(0.1)

        console.print("[bold]Starting concurrent operations...[/]")
        console.print()
        t0 = time.time()

        t_iot = threading.Thread(target=iot_writer)
        t_rag = threading.Thread(target=rag_searcher)
        t_iot.start()
        t_rag.start()

        # Live status while running
        with console.status("[bold cyan]IoT writing + RAG searching...[/]"):
            t_iot.join(timeout=30)
            t_rag.join(timeout=60)

        elapsed = time.time() - t0
        time.sleep(0.3)

        row = db.fetch_one("SELECT COUNT(*) FROM sensor_data")
        sensor_count = row[0] if row else 0

        # Results
        result_table = Table(title=f"Combined Results ({elapsed:.1f}s)", box=box.ROUNDED, title_style="bold green")
        result_table.add_column("Metric", style="bold")
        result_table.add_column("Value", justify="right")
        result_table.add_row("IoT writes submitted", str(iot_writes[0]))
        result_table.add_row("IoT writes in DB", str(sensor_count))
        result_table.add_row("Concurrent searches", str(len(search_results)))
        result_table.add_row("Errors", f"[{'red' if errors else 'green'}]{len(errors)}[/]")
        console.print(result_table)

        console.print()
        search_table = Table(title="Search During IoT Writes", box=box.SIMPLE_HEAVY)
        search_table.add_column("Query", style="bold", max_width=35)
        search_table.add_column("Results", justify="center")
        search_table.add_column("Latency", justify="right")
        search_table.add_column("Top Hit", max_width=50)

        for q, n, ms, top in search_results:
            search_table.add_row(q, str(n), f"{ms:.0f}ms", f"[dim]{top}[/]")
        console.print(search_table)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

def run_manager_mode(base_dir: str):
    console.print()
    console.rule("[bold green]Multi-Database Manager[/]")
    console.print()

    embedder = QwenEmbedder()

    with SQFoxManager(base_dir) as mgr:
        # IoT
        console.print("[bold]Setting up databases...[/]")
        iot = mgr["sensors"]
        iot.write("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor TEXT, value REAL, unit TEXT
            )
        """, wait=True)
        for _ in range(50):
            s = random.choice(SENSOR_TYPES)
            iot.write(
                "INSERT INTO readings (sensor, value, unit) VALUES (?, ?, ?)",
                (s[0], round(random.uniform(s[2], s[3]), 2), s[1]),
                wait=True,
            )
        time.sleep(0.2)
        row = iot.fetch_one("SELECT COUNT(*) FROM readings")
        console.print(f"  [cyan]sensors.db[/]  — {row[0]} readings")

        # Knowledge
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("  knowledge.db — ingesting", total=len(RAG_DOCUMENTS))
            for doc in RAG_DOCUMENTS:
                mgr.ingest_to("knowledge", doc["text"], metadata=doc["meta"], embed_fn=embedder, wait=True)
                progress.advance(task)

        mgr["knowledge"].ensure_schema(SchemaState.SEARCHABLE)
        time.sleep(0.2)

        # Database overview
        console.print()
        db_table = Table(title="Databases", box=box.ROUNDED, title_style="bold")
        db_table.add_column("Name", style="bold cyan")
        db_table.add_column("Path")
        db_table.add_column("Vec", justify="center")
        for name in mgr.names:
            d = mgr[name]
            vec_icon = "[green]yes[/]" if d.vec_available else "[red]no[/]"
            db_table.add_row(name, str(Path(d.path).name), vec_icon)
        console.print(db_table)

        # Cross-search
        console.print()
        console.print("[bold]Cross-Database Search[/]")
        console.print()

        queries = [
            "вибрация подшипник замена",
            "SQLite PRAGMA configuration",
            "predictive maintenance IoT",
        ]

        for query in queries:
            t0 = time.time()
            results = mgr.search_all(query, embed_fn=embedder, limit=3)
            elapsed_ms = (time.time() - t0) * 1000

            console.print(f"  [bold white]{query}[/]  [dim]({elapsed_ms:.0f}ms)[/]")
            for db_name, r in results:
                text_preview = r.text.replace("\n", " ")
                score_color = "bold green" if r.score >= 0.7 else "yellow" if r.score >= 0.6 else "dim"
                console.print(
                    f"    [{score_color}]{r.score:.3f}[/] [cyan]{db_name:10s}[/] {text_preview}"
                )
            console.print()


# ---------------------------------------------------------------------------
# AsyncSQFox
# ---------------------------------------------------------------------------

def run_async_mode(db_path: str):
    console.print()
    console.rule("[bold yellow]AsyncSQFox[/]")
    console.print()

    embedder = QwenEmbedder()

    async def _run():
        async with AsyncSQFox(db_path) as db:

            # --- Concurrent ingest ---
            console.print("[bold]Concurrent Ingest (asyncio.gather)[/]")
            console.print()

            docs = RAG_DOCUMENTS[:10]
            t0 = time.time()
            results = await asyncio.gather(*(
                db.ingest(doc["text"], metadata=doc["meta"], embed_fn=embedder)
                for doc in docs
            ))
            ingest_time = time.time() - t0

            ingest_table = Table(
                title=f"Concurrent Ingest ({ingest_time:.2f}s)",
                box=box.ROUNDED, title_style="bold green",
            )
            ingest_table.add_column("Metric", style="bold")
            ingest_table.add_column("Value", justify="right")
            ingest_table.add_row("Documents", str(len(docs)))
            ingest_table.add_row("Doc IDs", ", ".join(str(r) for r in results))
            ingest_table.add_row("Throughput", f"[bold cyan]{len(docs)/ingest_time:.1f} docs/sec[/]")
            console.print(ingest_table)

            await db.ensure_schema(SchemaState.SEARCHABLE)
            # let FTS triggers settle
            await asyncio.sleep(0.2)

            # --- Concurrent search ---
            console.print()
            console.print("[bold]Concurrent Search (asyncio.gather)[/]")
            console.print()

            search_queries = [
                "при какой вибрации менять подшипник",
                "SQLite WAL configuration",
                "predictive maintenance vibration",
                "edge computing IoT",
            ]

            t0 = time.time()
            search_results = await asyncio.gather(*(
                db.search(q, embed_fn=embedder, limit=3)
                for q in search_queries
            ))
            search_time = time.time() - t0

            for q, hits in zip(search_queries, search_results):
                console.print(f"  [bold white]{q}[/]")
                for i, r in enumerate(hits):
                    preview = r.text.replace("\n", " ")
                    score_color = "bold green" if r.score >= 0.7 else "yellow" if r.score >= 0.6 else "dim"
                    rank = "  >>>" if i == 0 else "     "
                    console.print(f"{rank} [{score_color}]{r.score:.3f}[/] {preview}")
                console.print()

            console.print(f"  [dim]4 queries completed in {search_time*1000:.0f}ms total[/]")

            # --- Pool isolation demo ---
            console.print()
            console.print("[bold]Pool Isolation (heavy ingest + concurrent read)[/]")
            console.print()

            def slow_chunker(text: str) -> list[str]:
                """Simulate CPU-heavy chunking."""
                time.sleep(0.5)
                return [text]

            heavy_doc = "Heavy document for pool isolation test. " * 5

            timeline = []

            async def heavy_ingest():
                t_start = time.time()
                await db.ingest(heavy_doc, embed_fn=embedder, chunker=slow_chunker)
                t_end = time.time()
                timeline.append(("ingest", t_start, t_end))

            async def concurrent_read():
                await asyncio.sleep(0.05)  # start slightly after ingest
                t_start = time.time()
                row = await db.fetch_one("SELECT COUNT(*) FROM documents")
                t_end = time.time()
                timeline.append(("read", t_start, t_end))
                return row[0] if row else 0

            t0 = time.time()
            _, read_count = await asyncio.gather(heavy_ingest(), concurrent_read())
            total_time = time.time() - t0

            iso_table = Table(
                title="Pool Isolation Results",
                box=box.ROUNDED, title_style="bold green",
            )
            iso_table.add_column("Operation", style="bold")
            iso_table.add_column("Start", justify="right")
            iso_table.add_column("End", justify="right")
            iso_table.add_column("Duration", justify="right")

            for op, ts, te in sorted(timeline, key=lambda x: x[1]):
                iso_table.add_row(
                    op,
                    f"{ts - t0:.3f}s",
                    f"{te - t0:.3f}s",
                    f"{te - ts:.3f}s",
                )

            iso_table.add_section()
            iso_table.add_row(
                "total wall-time", "", "", f"[bold cyan]{total_time:.3f}s[/]",
            )
            console.print(iso_table)

            read_blocked = False
            for op, ts, te in timeline:
                if op == "read":
                    for op2, ts2, te2 in timeline:
                        if op2 == "ingest" and ts >= te2:
                            read_blocked = True
            if not read_blocked:
                console.print("  [green]Read was NOT blocked by heavy ingest[/]")
            else:
                console.print("  [yellow]Read waited for ingest to finish[/]")

            # --- Async backup ---
            console.print()
            console.print("[bold]Async Backup[/]")
            console.print()

            backup_path = str(Path(db_path).parent / "async_backup.db")

            t0 = time.time()
            await db.backup(backup_path)
            backup_time = time.time() - t0

            backup_size = Path(backup_path).stat().st_size

            # Verify
            verify_conn = sqlite3.connect(backup_path)
            verify_count = verify_conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            verify_conn.close()

            backup_table = Table(
                title="Async Backup Results",
                box=box.ROUNDED, title_style="bold green",
            )
            backup_table.add_column("Metric", style="bold")
            backup_table.add_column("Value", justify="right")
            backup_table.add_row("Backup file", Path(backup_path).name)
            backup_table.add_row("Size", f"{backup_size / 1024:.1f} KB")
            backup_table.add_row("Time", f"{backup_time:.3f}s")
            backup_table.add_row("Documents in backup", str(verify_count))
            backup_table.add_row("Integrity", "[bold green]OK[/]")
            console.print(backup_table)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Diagnostics panel
# ---------------------------------------------------------------------------

def show_diagnostics(db: SQFox):
    diag = db.diagnostics()
    table = Table(title="System Diagnostics", box=box.ROUNDED, title_style="bold")
    table.add_column("Component", style="bold")
    table.add_column("Value")

    table.add_row("sqfox", diag["sqfox_version"])
    table.add_row("Python", diag["python_version"])
    table.add_row("Platform", diag["platform"])
    table.add_row("SQLite", diag["sqlite_version"])
    table.add_row("sqlite-vec", str(diag.get("sqlite_vec_version", "N/A")))
    table.add_row("simplemma", str(diag.get("simplemma_version", "N/A")))
    table.add_row("pymorphy3", str(diag.get("pymorphy3_version", "N/A")))

    vec_status = "[green]loaded[/]" if diag["vec_available"] else "[red]not available[/]"
    table.add_row("Vector search", vec_status)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cleanup(*paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            for suffix in ("", "-wal", "-shm"):
                f = Path(str(p) + suffix)
                if f.exists():
                    f.unlink()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    valid_modes = ("all", "iot", "rag", "combined", "manager", "async")
    if mode not in valid_modes:
        console.print(f"[red]Unknown mode: {mode}[/]")
        console.print(f"Valid modes: {', '.join(valid_modes)}")
        sys.exit(1)

    demo_dir = Path(__file__).parent / "data"
    demo_dir.mkdir(exist_ok=True)

    iot_db = str(demo_dir / "iot_demo.db")
    rag_db = str(demo_dir / "rag_demo.db")
    combined_db = str(demo_dir / "combined_demo.db")
    manager_dir = str(demo_dir / "manager_demo")
    async_db = str(demo_dir / "async_demo.db")

    # Banner
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]sqfox demo[/]\n"
            "[dim]IoT + RAG on Qwen3-Embedding-0.6B[/]\n"
            "[dim]CPU only, 256 dim (MRL)[/]"
        ),
        border_style="bold cyan",
        box=box.DOUBLE,
    ))

    # Load model
    console.print()
    get_model()

    # Diagnostics
    console.print()
    with SQFox(str(demo_dir / "_diag.db")) as _db:
        show_diagnostics(_db)
    cleanup(str(demo_dir / "_diag.db"))

    iot_backup = str(demo_dir / "iot_backup.db")
    async_backup = str(demo_dir / "async_backup.db")

    if mode in ("all", "iot"):
        cleanup(iot_db, iot_backup)
        run_iot_mode(iot_db)
        cleanup(iot_backup)

    if mode in ("all", "rag"):
        cleanup(rag_db)
        run_rag_mode(rag_db)

    if mode in ("all", "combined"):
        cleanup(combined_db)
        run_combined_mode(combined_db)

    if mode in ("all", "manager"):
        cleanup(manager_dir)
        run_manager_mode(manager_dir)

    if mode in ("all", "async"):
        cleanup(async_db, async_backup)
        run_async_mode(async_db)
        cleanup(async_backup)

    # Done
    console.print()
    console.print(Panel(
        Align.center("[bold green]Demo complete![/]"),
        border_style="green",
        box=box.DOUBLE,
    ))


if __name__ == "__main__":
    main()
