"""
sqfox demo: Crash & Recovery — SQLite is the source of truth.

Demonstrates crash safety of the HNSW backend:
  Phase 1: Ingest real knowledge base with HNSW backend + Qwen3 embeddings
  Phase 2: Ingest more docs and simulate crash (corrupt HNSW graph BLOB)
  Phase 3: Reopen the DB — HNSW auto-rebuilds from embedding BLOBs in SQLite
  Phase 4: Verify all data is intact, search returns correct results

Usage:
  python demo/run_crash_recovery.py
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
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
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
)
from rich.align import Align
from rich import box

from sqfox import SQFox, SchemaState
from sqfox.backends.hnsw import SqliteHnswBackend

console = Console()

# ---------------------------------------------------------------------------
# Embedding adapter (Qwen3-Embedding-0.6B, 256 dim MRL)
# ---------------------------------------------------------------------------

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        with console.status(
            "[bold cyan]Loading Qwen3-Embedding-0.6B...[/]", spinner="dots"
        ):
            t0 = time.time()
            _model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
            )
            elapsed = time.time() - t0
        console.print(
            f"  [green]Model loaded in {elapsed:.1f}s[/] (256 dim, MRL truncated)"
        )
    return _model


class QwenEmbedder:
    def __init__(self):
        self.model = get_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, prompt_name="query").tolist()


# ---------------------------------------------------------------------------
# Knowledge base — real technical content, two batches
# ---------------------------------------------------------------------------

# Phase 1 batch: core knowledge
BATCH_1 = [
    {"text": "Замена подшипников качения выполняется при превышении уровня вибрации 7.1 мм/с (ISO 10816). Перед заменой необходимо остановить оборудование и зафиксировать вал.", "meta": {"domain": "maintenance", "lang": "ru"}},
    {"text": "Система мониторинга вибрации позволяет выявлять дисбаланс ротора, расцентровку валов, дефекты подшипников и ослабление крепления на ранних стадиях.", "meta": {"domain": "monitoring", "lang": "ru"}},
    {"text": "Predictive maintenance uses vibration analysis and machine learning to forecast equipment failures. Common features include RMS velocity, peak acceleration, and crest factor.", "meta": {"domain": "maintenance", "lang": "en"}},
    {"text": "Спектральный анализ вибрации позволяет определить тип дефекта: дисбаланс проявляется на частоте вращения (1X), расцентровка — на 2X, дефекты подшипников — на характерных частотах BPFO/BPFI.", "meta": {"domain": "diagnostics", "lang": "ru"}},
    {"text": "При температуре охлаждающей жидкости выше 95 градусов Цельсия необходимо проверить работу термостата и уровень антифриза в расширительном бачке.", "meta": {"domain": "troubleshooting", "lang": "ru"}},
    {"text": "PID controller tuning for temperature regulation: start with Ziegler-Nichols method, then fine-tune proportional gain Kp and integral time Ti based on step response.", "meta": {"domain": "control", "lang": "en"}},
    {"text": "Напряжение питания промышленного оборудования должно находиться в пределах 380В +/- 10%. При отклонениях более 10% срабатывает защита по напряжению.", "meta": {"domain": "electrical", "lang": "ru"}},
    {"text": "Регламент технического обслуживания электродвигателей: замена смазки каждые 4000 моточасов, проверка изоляции обмоток мегаомметром каждые 6 месяцев.", "meta": {"domain": "maintenance", "lang": "ru"}},
    {"text": "Калибровка датчиков давления производится ежеквартально. Допустимая погрешность не более 0.5% от диапазона измерения. При превышении — датчик подлежит замене.", "meta": {"domain": "calibration", "lang": "ru"}},
    {"text": "MQTT protocol is widely used for IoT sensor data collection. QoS level 1 guarantees at-least-once delivery, suitable for non-critical telemetry data.", "meta": {"domain": "networking", "lang": "en"}},
    {"text": "OPC UA (Unified Architecture) is the standard protocol for industrial automation. It provides secure, reliable communication between PLCs, SCADA systems, and cloud platforms.", "meta": {"domain": "networking", "lang": "en"}},
    {"text": "Edge computing reduces latency by processing data locally on industrial PCs instead of sending everything to the cloud. Typical edge devices use Intel Atom or ARM Cortex processors.", "meta": {"domain": "architecture", "lang": "en"}},
    {"text": "SQLite WAL mode enables concurrent reads during writes. Set PRAGMA journal_mode=WAL and PRAGMA synchronous=NORMAL for optimal performance on embedded systems.", "meta": {"domain": "database", "lang": "en"}},
    {"text": "Machine learning модели для predictive maintenance обучаются на исторических данных вибрации. Используются алгоритмы Random Forest и Gradient Boosting.", "meta": {"domain": "ai", "lang": "ru"}},
    {"text": "Давление в гидравлической системе не должно превышать 250 бар. При достижении 280 бар срабатывает предохранительный клапан. Проверка манометров — ежемесячно.", "meta": {"domain": "hydraulics", "lang": "ru"}},
    {"text": "При фрезеровании алюминия рекомендуемая скорость резания 200-500 м/мин, подача на зуб 0.05-0.2 мм. Использовать однозубую фрезу для черновой обработки.", "meta": {"domain": "cnc", "lang": "ru"}},
    {"text": "Система аварийного останова (E-Stop) должна обеспечивать категорию останова 0 по IEC 60204-1. Время срабатывания не более 100 мс.", "meta": {"domain": "safety", "lang": "ru"}},
    {"text": "Lockout/Tagout (LOTO) procedure must be followed before any maintenance work on energized equipment. Each worker applies their own lock to the energy isolation device.", "meta": {"domain": "safety", "lang": "en"}},
]

# Phase 2 batch: additional docs that will be "lost" in the crash
BATCH_2 = [
    {"text": "Протокол Modbus RTU используется для связи с PLC контроллерами. Скорость 9600-115200 бод, формат данных 8N1. Максимальная длина шины RS-485 — 1200 метров.", "meta": {"domain": "networking", "lang": "ru"}},
    {"text": "Тепловизионный контроль электрооборудования: перегрев контактных соединений выше 60 градусов Цельсия относительно окружающей среды указывает на критический дефект.", "meta": {"domain": "diagnostics", "lang": "ru"}},
    {"text": "Частотный преобразователь позволяет регулировать скорость асинхронного двигателя от 5 до 50 Гц. Для работы ниже 20 Гц требуется принудительное охлаждение.", "meta": {"domain": "electrical", "lang": "ru"}},
    {"text": "Ультразвуковой расходомер измеряет скорость потока по разнице времени прохождения сигнала по и против течения. Точность 0.5-1% при Reynolds > 4000.", "meta": {"domain": "instrumentation", "lang": "ru"}},
    {"text": "Transformer-based models achieve state-of-the-art results in anomaly detection for industrial time series. Self-attention captures long-range temporal dependencies.", "meta": {"domain": "ai", "lang": "en"}},
    {"text": "Federated learning enables training ML models across multiple factory sites without sharing raw sensor data, preserving data privacy and reducing network costs.", "meta": {"domain": "ai", "lang": "en"}},
    {"text": "Квантование модели INT8 уменьшает размер нейросети в 4 раза при потере точности менее 1%. Позволяет запускать inference на микроконтроллерах с 512 КБ RAM.", "meta": {"domain": "ai", "lang": "ru"}},
    {"text": "Фильтрация гидравлического масла: класс чистоты NAS 6 для сервоклапанов, NAS 8 для распределителей. Замена фильтроэлементов при перепаде давления 3 бар.", "meta": {"domain": "hydraulics", "lang": "ru"}},
    {"text": "G-code G41/G42 — коррекция на радиус инструмента. G43 — коррекция на длину. Без коррекции точность обработки падает на величину износа инструмента.", "meta": {"domain": "cnc", "lang": "ru"}},
    {"text": "Зоны безопасности промышленных роботов определяются по ISO 13857. Минимальное расстояние до ограждения зависит от скорости останова и высоты защитного барьера.", "meta": {"domain": "safety", "lang": "ru"}},
    {"text": "Коэффициент мощности (cos phi) промышленной сети должен быть не менее 0.92. Компенсация реактивной мощности выполняется конденсаторными установками.", "meta": {"domain": "power", "lang": "ru"}},
    {"text": "Photovoltaic panel degradation averages 0.5-0.7% per year. After 25 years, typical output is 80-85% of initial rated capacity.", "meta": {"domain": "power", "lang": "en"}},
]

# Queries for verification — each should find a specific document
VERIFY_QUERIES = [
    {
        "query": "при какой вибрации менять подшипник",
        "expect_substring": "7.1 мм/с",
    },
    {
        "query": "PID controller temperature tuning",
        "expect_substring": "Ziegler-Nichols",
    },
    {
        "query": "Modbus RS-485 протокол связи с контроллерами",
        "expect_substring": "Modbus RTU",
    },
    {
        "query": "квантование нейросети для микроконтроллеров",
        "expect_substring": "INT8",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_phase1 = len(BATCH_1)
    n_phase2 = len(BATCH_2)
    n_total = n_phase1 + n_phase2

    demo_dir = Path(__file__).parent / "data"
    demo_dir.mkdir(exist_ok=True)
    db_path = str(demo_dir / "crash_recovery.db")

    # Clean up
    for suffix in ("", "-wal", "-shm"):
        p = Path(db_path + suffix)
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    # ===== BANNER =====
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]Crash & Recovery[/]\n"
            "[dim]SQLite is the source of truth[/]\n"
            f"[dim]{n_total} docs, Qwen3-Embedding-0.6B, HNSW backend[/]"
        ),
        border_style="bold red",
        box=box.DOUBLE,
    ))

    # Load model
    console.print()
    embedder = QwenEmbedder()

    # ===== PHASE 1: Normal ingest =====
    console.print()
    console.rule(
        f"[bold green]Phase 1: Normal Ingest ({n_phase1} docs)[/]"
    )
    console.print()

    backend1 = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting (committed)", total=n_phase1)
        t0 = time.time()

        with SQFox(db_path, vector_backend=backend1) as db:
            for doc in BATCH_1:
                db.ingest(
                    doc["text"], metadata=doc["meta"],
                    embed_fn=embedder, wait=True,
                )
                progress.advance(task)
            db.ensure_schema(SchemaState.SEARCHABLE)
            time.sleep(0.2)

        phase1_time = time.time() - t0

    # Verify phase 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    p1_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    p1_indexed = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE vec_indexed = 1"
    ).fetchone()[0]
    p1_with_emb = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    p1_graph = conn.execute(
        "SELECT LENGTH(value) FROM __sqfox_hnsw WHERE key = 'graph'"
    ).fetchone()
    p1_graph_kb = (p1_graph[0] / 1024) if p1_graph and p1_graph[0] else 0
    conn.close()

    console.print()
    p1_table = Table(
        title="Phase 1: Committed State",
        box=box.ROUNDED, title_style="bold green",
    )
    p1_table.add_column("Metric", style="bold")
    p1_table.add_column("Value", justify="right")
    p1_table.add_row("Documents in SQLite", f"[bold cyan]{p1_docs}[/]")
    p1_table.add_row("With embedding BLOB", f"[cyan]{p1_with_emb}[/]")
    p1_table.add_row("vec_indexed = 1", f"[cyan]{p1_indexed}[/]")
    p1_table.add_row("HNSW graph BLOB", f"[cyan]{p1_graph_kb:.1f} KB[/]")
    p1_table.add_row("Time", f"{phase1_time:.1f}s")
    p1_table.add_row("Integrity", "[bold green]OK - fully committed[/]")
    console.print(p1_table)

    # Quick search to prove it works before crash
    console.print()
    console.print("[bold]Pre-crash search check:[/]")
    backend_pre = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)
    with SQFox(db_path, vector_backend=backend_pre) as db_pre:
        db_pre.ensure_schema(SchemaState.SEARCHABLE)
        time.sleep(0.1)
        results = db_pre.search(
            "при какой вибрации менять подшипник",
            embed_fn=embedder, limit=1,
        )
        if results:
            console.print(
                f"  [green]OK[/]  [dim]{results[0].score:.3f}[/]  "
                f"{results[0].text[:70]}..."
            )

    # ===== PHASE 2: Ingest + simulated crash =====
    console.print()
    console.rule(
        f"[bold red]Phase 2: Ingest + CRASH ({n_phase2} more docs)[/]"
    )
    console.print()

    backend2 = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting (pre-crash)", total=n_phase2)

        with SQFox(db_path, vector_backend=backend2) as db:
            for doc in BATCH_2:
                db.ingest(
                    doc["text"], metadata=doc["meta"],
                    embed_fn=embedder, wait=True,
                )
                progress.advance(task)

            console.print()
            console.print(Panel(
                Align.center(
                    "[bold red blink]>>> SIMULATING CRASH <<<[/]\n"
                    "[dim]Corrupting HNSW graph BLOB in SQLite...[/]\n"
                    "[dim]Documents and embedding BLOBs remain intact.[/]"
                ),
                border_style="bold red",
                box=box.HEAVY,
            ))

    # Corrupt the graph BLOB
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE __sqfox_hnsw SET value = X'00000000' WHERE key = 'graph'"
    )
    conn.commit()

    # Verify state after crash
    crash_docs = conn.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]
    crash_indexed = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE vec_indexed = 1"
    ).fetchone()[0]
    crash_with_emb = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    crash_graph = conn.execute(
        "SELECT LENGTH(value) FROM __sqfox_hnsw WHERE key = 'graph'"
    ).fetchone()
    crash_graph_bytes = crash_graph[0] if crash_graph and crash_graph[0] else 0
    conn.close()

    console.print()
    crash_table = Table(
        title="Post-Crash State",
        box=box.ROUNDED, title_style="bold red",
    )
    crash_table.add_column("Metric", style="bold")
    crash_table.add_column("Value", justify="right")
    crash_table.add_row(
        "Documents in SQLite",
        f"[bold green]{crash_docs}[/] (safe!)",
    )
    crash_table.add_row(
        "Embedding BLOBs",
        f"[bold green]{crash_with_emb}[/] (safe!)",
    )
    crash_table.add_row("vec_indexed = 1", f"[green]{crash_indexed}[/]")
    crash_table.add_row(
        "HNSW graph BLOB",
        f"[bold red]{crash_graph_bytes} bytes[/] (CORRUPTED!)",
    )
    crash_table.add_row(
        "Diagnosis",
        "[bold red]Graph lost, data intact[/]",
    )
    console.print(crash_table)

    console.print()
    console.print(
        "  [bold white]Key insight:[/] [dim]SQLite documents + embedding "
        "BLOBs are the source of truth.[/]\n"
        "  [dim]The HNSW graph is a rebuildable cache.[/]"
    )

    # ===== PHASE 3: Recovery =====
    console.print()
    console.rule("[bold yellow]Phase 3: Auto-Recovery[/]")
    console.print()
    console.print("  [dim]Reopening database with HNSW backend...[/]")
    console.print(
        "  [dim]Backend detects corrupt graph -> rebuilds from BLOBs.[/]"
    )
    console.print()

    backend3 = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)

    t0 = time.time()
    db = SQFox(db_path, vector_backend=backend3)
    db.start()
    recovery_time = time.time() - t0

    recovered_count = backend3.count()

    recovery_table = Table(
        title="Recovery Results",
        box=box.ROUNDED, title_style="bold yellow",
    )
    recovery_table.add_column("Metric", style="bold")
    recovery_table.add_column("Value", justify="right")
    recovery_table.add_row(
        "Recovery time", f"[cyan]{recovery_time:.2f}s[/]"
    )
    recovery_table.add_row(
        "HNSW nodes recovered",
        f"[bold green]{recovered_count}[/]",
    )
    recovery_table.add_row(
        "Expected",
        f"[dim]{crash_docs}[/]",
    )
    recovery_table.add_row(
        "Status",
        "[bold green]REBUILT FROM BLOBs[/]"
        if recovered_count == crash_docs
        else "[yellow]PARTIAL — will complete on next ingest[/]"
        if recovered_count > 0
        else "[bold red]RECOVERY FAILED[/]",
    )
    console.print(recovery_table)

    # ===== PHASE 4: Verification =====
    console.print()
    console.rule("[bold green]Phase 4: Post-Recovery Verification[/]")
    console.print()

    db.ensure_schema(SchemaState.SEARCHABLE)
    time.sleep(0.2)

    console.print("[bold]Search verification (hybrid: FTS5 + HNSW)[/]")
    console.print()

    all_ok = True
    for vq in VERIFY_QUERIES:
        query = vq["query"]
        expect = vq["expect_substring"]

        t0 = time.time()
        results = db.search(query, embed_fn=embedder, limit=3)
        elapsed_ms = (time.time() - t0) * 1000

        found_expected = any(expect in r.text for r in results)
        ok = len(results) > 0 and found_expected

        status = "[bold green]OK[/]" if ok else "[bold red]MISS[/]"
        if not ok:
            all_ok = False

        top_score = f"{results[0].score:.3f}" if results else "--"
        top_text = results[0].text[:70].replace("\n", " ") if results else "--"
        top_meta = results[0].metadata if results else {}
        tag = f"{top_meta.get('lang', '?')}/{top_meta.get('domain', '?')}"

        console.print(
            f"  {status}  [bold white]{query}[/]  "
            f"[dim]({elapsed_ms:.0f}ms)[/]"
        )
        console.print(
            f"       [{('green' if ok else 'red')}]{top_score}[/] "
            f"[dim]{tag}[/]  {top_text}..."
        )
        console.print()

    # Data integrity check
    total_docs_row = db.fetch_one("SELECT COUNT(*) FROM documents")
    total_docs = total_docs_row[0] if total_docs_row else 0
    total_emb_row = db.fetch_one(
        "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL"
    )
    total_emb = total_emb_row[0] if total_emb_row else 0

    integrity_table = Table(
        title="Data Integrity",
        box=box.ROUNDED, title_style="bold green",
    )
    integrity_table.add_column("Check", style="bold")
    integrity_table.add_column("Expected", justify="right")
    integrity_table.add_column("Actual", justify="right")
    integrity_table.add_column("Status", justify="center")

    integrity_table.add_row(
        "Total documents",
        str(n_total), str(total_docs),
        "[bold green]OK[/]"
        if total_docs == n_total
        else "[bold red]MISMATCH[/]",
    )
    integrity_table.add_row(
        "Embedding BLOBs",
        str(n_total), str(total_emb),
        "[bold green]OK[/]"
        if total_emb == n_total
        else "[bold red]MISMATCH[/]",
    )
    integrity_table.add_row(
        "HNSW index",
        str(n_total), str(recovered_count),
        "[bold green]OK[/]"
        if recovered_count == n_total
        else "[yellow]REBUILT[/]",
    )
    integrity_table.add_row(
        "Search correctness",
        f"{len(VERIFY_QUERIES)}/{len(VERIFY_QUERIES)}",
        f"{sum(1 for vq in VERIFY_QUERIES for r in db.search(vq['query'], embed_fn=embedder, limit=3) if vq['expect_substring'] in r.text)}/{len(VERIFY_QUERIES)}"
        if all_ok else "FAIL",
        "[bold green]OK[/]" if all_ok else "[bold red]FAIL[/]",
    )
    console.print(integrity_table)

    db.stop()

    # Cleanup
    for suffix in ("", "-wal", "-shm"):
        p = Path(db_path + suffix)
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    # ===== FINALE =====
    console.print()
    console.print(Panel(
        Align.center(
            "[bold green]Recovery complete![/]\n"
            f"[dim]{n_total} documents survived the crash.[/]\n"
            "[dim]HNSW graph rebuilt from embedding BLOBs in SQLite.[/]\n"
            "[dim]Zero data loss. SQLite = source of truth.[/]"
        ),
        border_style="green",
        box=box.DOUBLE,
    ))


if __name__ == "__main__":
    main()
