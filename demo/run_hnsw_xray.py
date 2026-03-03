"""
sqfox demo: HNSW X-Ray — interactive graph inspector.

Loads a real knowledge base (industrial maintenance, IoT, ML, protocols)
with Qwen3-Embedding-0.6B, builds an HNSW index, and drops into an
interactive REPL where you can:

  - Type any query to see its search path traced through HNSW layers
  - /stats   — show graph structure & density analysis
  - /node N  — inspect a specific node: its text, level, neighbors
  - /top     — show top-connected nodes (hubs in the graph)
  - /help    — list commands
  - /quit    — exit

Usage:
  python demo/run_hnsw_xray.py
"""

import array as _array
import math
import os
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
from rich.tree import Tree
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
)
from rich.prompt import Prompt
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
# Knowledge base — real technical content
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    # --- Вибрация и подшипники ---
    {"text": "Замена подшипников качения выполняется при превышении уровня вибрации 7.1 мм/с (ISO 10816). Перед заменой необходимо остановить оборудование и зафиксировать вал.", "meta": {"domain": "maintenance", "lang": "ru"}},
    {"text": "Система мониторинга вибрации позволяет выявлять дисбаланс ротора, расцентровку валов, дефекты подшипников и ослабление крепления на ранних стадиях.", "meta": {"domain": "monitoring", "lang": "ru"}},
    {"text": "Predictive maintenance uses vibration analysis and machine learning to forecast equipment failures. Common features include RMS velocity, peak acceleration, and crest factor.", "meta": {"domain": "maintenance", "lang": "en"}},
    {"text": "Спектральный анализ вибрации позволяет определить тип дефекта: дисбаланс проявляется на частоте вращения (1X), расцентровка — на 2X, дефекты подшипников — на характерных частотах BPFO/BPFI.", "meta": {"domain": "diagnostics", "lang": "ru"}},

    # --- Температура и охлаждение ---
    {"text": "При температуре охлаждающей жидкости выше 95 градусов Цельсия необходимо проверить работу термостата и уровень антифриза в расширительном бачке.", "meta": {"domain": "troubleshooting", "lang": "ru"}},
    {"text": "PID controller tuning for temperature regulation: start with Ziegler-Nichols method, then fine-tune proportional gain Kp and integral time Ti based on step response.", "meta": {"domain": "control", "lang": "en"}},
    {"text": "Тепловизионный контроль электрооборудования: перегрев контактных соединений выше 60 градусов Цельсия относительно окружающей среды указывает на критический дефект.", "meta": {"domain": "diagnostics", "lang": "ru"}},

    # --- Электрика ---
    {"text": "Напряжение питания промышленного оборудования должно находиться в пределах 380В +/- 10%. При отклонениях более 10% срабатывает защита по напряжению.", "meta": {"domain": "electrical", "lang": "ru"}},
    {"text": "Регламент технического обслуживания электродвигателей: замена смазки каждые 4000 моточасов, проверка изоляции обмоток мегаомметром каждые 6 месяцев.", "meta": {"domain": "maintenance", "lang": "ru"}},
    {"text": "Частотный преобразователь позволяет регулировать скорость асинхронного двигателя от 5 до 50 Гц. Для работы ниже 20 Гц требуется принудительное охлаждение.", "meta": {"domain": "electrical", "lang": "ru"}},

    # --- Датчики и калибровка ---
    {"text": "Калибровка датчиков давления производится ежеквартально. Допустимая погрешность не более 0.5% от диапазона измерения. При превышении — датчик подлежит замене.", "meta": {"domain": "calibration", "lang": "ru"}},
    {"text": "Ультразвуковой расходомер измеряет скорость потока по разнице времени прохождения сигнала по и против течения. Точность 0.5-1% при Reynolds > 4000.", "meta": {"domain": "instrumentation", "lang": "ru"}},

    # --- Протоколы и связь ---
    {"text": "MQTT protocol is widely used for IoT sensor data collection. QoS level 1 guarantees at-least-once delivery, suitable for non-critical telemetry data.", "meta": {"domain": "networking", "lang": "en"}},
    {"text": "OPC UA (Unified Architecture) is the standard protocol for industrial automation. It provides secure, reliable communication between PLCs, SCADA systems, and cloud platforms.", "meta": {"domain": "networking", "lang": "en"}},
    {"text": "Протокол Modbus RTU используется для связи с PLC контроллерами. Скорость 9600-115200 бод, формат данных 8N1. Максимальная длина шины RS-485 — 1200 метров.", "meta": {"domain": "networking", "lang": "ru"}},

    # --- Edge / embedded ---
    {"text": "Edge computing reduces latency by processing data locally on industrial PCs instead of sending everything to the cloud. Typical edge devices use Intel Atom or ARM Cortex processors.", "meta": {"domain": "architecture", "lang": "en"}},
    {"text": "SQLite WAL mode enables concurrent reads during writes. Set PRAGMA journal_mode=WAL and PRAGMA synchronous=NORMAL for optimal performance on embedded systems.", "meta": {"domain": "database", "lang": "en"}},
    {"text": "Настройка PRAGMA journal_mode=WAL в SQLite позволяет параллельное чтение и запись. Рекомендуется для embedded-систем с частой записью данных с датчиков.", "meta": {"domain": "database", "lang": "ru"}},

    # --- AI / ML ---
    {"text": "Machine learning модели для predictive maintenance обучаются на исторических данных вибрации. Используются алгоритмы Random Forest и Gradient Boosting для классификации состояния оборудования.", "meta": {"domain": "ai", "lang": "ru"}},
    {"text": "Transformer-based models achieve state-of-the-art results in anomaly detection for industrial time series. Self-attention captures long-range temporal dependencies.", "meta": {"domain": "ai", "lang": "en"}},
    {"text": "Federated learning enables training ML models across multiple factory sites without sharing raw sensor data, preserving data privacy and reducing network costs.", "meta": {"domain": "ai", "lang": "en"}},
    {"text": "Квантование модели INT8 уменьшает размер нейросети в 4 раза при потере точности менее 1%. Позволяет запускать inference на микроконтроллерах с 512 КБ RAM.", "meta": {"domain": "ai", "lang": "ru"}},

    # --- Гидравлика / пневматика ---
    {"text": "Давление в гидравлической системе не должно превышать 250 бар. При достижении 280 бар срабатывает предохранительный клапан. Проверка манометров — ежемесячно.", "meta": {"domain": "hydraulics", "lang": "ru"}},
    {"text": "Фильтрация гидравлического масла: класс чистоты NAS 6 для сервоклапанов, NAS 8 для распределителей. Замена фильтроэлементов при перепаде давления 3 бар.", "meta": {"domain": "hydraulics", "lang": "ru"}},
    {"text": "Пневматический привод: рабочее давление 4-6 бар, расход воздуха зависит от диаметра цилиндра и хода поршня. Обязательна установка влагоотделителя на входе.", "meta": {"domain": "pneumatics", "lang": "ru"}},

    # --- CNC / обработка ---
    {"text": "При фрезеровании алюминия рекомендуемая скорость резания 200-500 м/мин, подача на зуб 0.05-0.2 мм. Использовать однозубую фрезу для черновой обработки.", "meta": {"domain": "cnc", "lang": "ru"}},
    {"text": "G-code G41/G42 — коррекция на радиус инструмента. G43 — коррекция на длину. Без коррекции точность обработки падает на величину износа инструмента.", "meta": {"domain": "cnc", "lang": "ru"}},

    # --- Безопасность ---
    {"text": "Система аварийного останова (E-Stop) должна обеспечивать категорию останова 0 по IEC 60204-1. Время срабатывания не более 100 мс. Тестирование — еженедельно.", "meta": {"domain": "safety", "lang": "ru"}},
    {"text": "Lockout/Tagout (LOTO) procedure must be followed before any maintenance work on energized equipment. Each worker applies their own lock to the energy isolation device.", "meta": {"domain": "safety", "lang": "en"}},
    {"text": "Зоны безопасности промышленных роботов определяются по ISO 13857. Минимальное расстояние до ограждения зависит от скорости останова и высоты защитного барьера.", "meta": {"domain": "safety", "lang": "ru"}},

    # --- Энергетика ---
    {"text": "Коэффициент мощности (cos phi) промышленной сети должен быть не менее 0.92. Компенсация реактивной мощности выполняется конденсаторными установками с автоматическим регулированием.", "meta": {"domain": "power", "lang": "ru"}},
    {"text": "Photovoltaic panel degradation averages 0.5-0.7% per year. After 25 years, typical output is 80-85% of initial rated capacity. Regular cleaning improves efficiency by 5-10%.", "meta": {"domain": "power", "lang": "en"}},
]


# ---------------------------------------------------------------------------
# X-Ray: graph analysis helpers
# ---------------------------------------------------------------------------

def xray_graph(backend: SqliteHnswBackend, ndim: int) -> dict:
    """Extract detailed stats from the HNSW graph internals."""
    info = {
        "entry_point": backend._entry_point,
        "max_level": backend._max_level,
        "total_nodes": backend._count,
        "ndim": ndim,
        "M": backend._M,
        "M0": backend._M0,
        "ef_construction": backend._ef_construction,
        "ef_search": backend._ef_search,
        "levels": [],
    }

    for lv in range(len(backend._graphs)):
        g = backend._graphs[lv]
        g.merge()
        node_ids = sorted(g.all_node_ids() - backend._deleted)
        if not node_ids:
            info["levels"].append({
                "level": lv, "nodes": 0, "edges": 0,
                "avg_degree": 0, "min_degree": 0, "max_degree": 0,
                "orphans": 0,
            })
            continue

        degrees = []
        total_edges = 0
        orphans = 0
        for nid in node_ids:
            nbrs = [n for n in g.neighbors(nid) if n not in backend._deleted]
            degrees.append(len(nbrs))
            total_edges += len(nbrs)
            if not nbrs:
                orphans += 1

        info["levels"].append({
            "level": lv,
            "nodes": len(node_ids),
            "edges": total_edges,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "orphans": orphans,
        })

    return info


def trace_search_path(backend, query_vec, conn):
    """Trace the greedy search path through HNSW layers."""
    steps = []
    ep = backend._entry_point
    if ep is None:
        return steps

    q = _array.array("f", query_vec)
    current = ep
    current_vec = backend._get_vector(current, conn)
    if current_vec is None:
        return steps
    current_dist = math.dist(q, current_vec)

    steps.append({
        "level": backend._max_level,
        "node": current,
        "distance": current_dist,
        "action": "entry",
    })

    for level in range(backend._max_level, 0, -1):
        changed = True
        while changed:
            changed = False
            with backend._lock:
                nbrs = list(backend._neighbors(current, level))
            alive = [n for n in nbrs if n not in backend._deleted]
            vecs = backend._fetch_batch(alive, conn)
            for n, v in vecs.items():
                d = math.dist(q, v)
                if d < current_dist:
                    current_dist = d
                    current = n
                    changed = True
                    steps.append({
                        "level": level,
                        "node": current,
                        "distance": current_dist,
                        "action": "jump",
                    })
        steps.append({
            "level": level, "node": current,
            "distance": current_dist, "action": "drop",
        })

    results = backend._search_layer(q, current, backend._ef_search, 0, conn)
    if results:
        steps.append({
            "level": 0, "node": results[0][0],
            "distance": results[0][1], "action": "found",
        })

    return steps


def get_node_degree_map(backend):
    """Return {node_id: degree_on_L0} for all nodes."""
    if not backend._graphs:
        return {}
    g = backend._graphs[0]
    g.merge()
    result = {}
    for nid in g.all_node_ids():
        if nid not in backend._deleted:
            nbrs = [n for n in g.neighbors(nid) if n not in backend._deleted]
            result[nid] = len(nbrs)
    return result


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def show_stats(backend, ndim, db):
    """Display full graph stats."""
    info = xray_graph(backend, ndim)

    console.print()

    # Parameters
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Vectors", f"[bold cyan]{info['total_nodes']}[/]")
    summary.add_row("Dimensions", f"[cyan]{info['ndim']}[/]")
    summary.add_row("Max level", f"[cyan]{info['max_level']}[/]")
    summary.add_row("Entry point", f"[cyan]doc {info['entry_point']}[/]")
    summary.add_row("M", f"[dim]{info['M']}[/]")
    summary.add_row("M0 (layer 0)", f"[dim]{info['M0']}[/]")
    summary.add_row("ef_construction", f"[dim]{info['ef_construction']}[/]")
    summary.add_row("ef_search", f"[dim]{info['ef_search']}[/]")
    console.print(Panel(summary, title="[bold]Index Parameters[/]", border_style="cyan"))

    # Per-level table
    lt = Table(
        title="Per-Level Statistics", box=box.ROUNDED,
        title_style="bold green",
    )
    lt.add_column("Level", justify="center", style="bold")
    lt.add_column("Nodes", justify="right")
    lt.add_column("Edges", justify="right")
    lt.add_column("Avg Deg", justify="right")
    lt.add_column("Min", justify="right")
    lt.add_column("Max", justify="right")
    lt.add_column("Orphans", justify="right")
    lt.add_column("", justify="left")

    for lv_info in reversed(info["levels"]):
        lv = lv_info["level"]
        nodes = lv_info["nodes"]
        total = info["total_nodes"]
        bar_w = 25
        filled = int(nodes / total * bar_w) if total > 0 else 0
        bar = "[green]" + "#" * filled + "[/][dim]" + "." * (bar_w - filled) + "[/]"
        entry = " [bold yellow]<< ENTRY[/]" if lv == info["max_level"] else ""
        orphans = (
            f"[red]{lv_info['orphans']}[/]"
            if lv_info["orphans"] > 0
            else f"[green]{lv_info['orphans']}[/]"
        )
        lt.add_row(
            f"L{lv}", str(nodes), str(lv_info["edges"]),
            f"{lv_info['avg_degree']:.1f}",
            str(lv_info["min_degree"]), str(lv_info["max_degree"]),
            orphans, f"{bar}{entry}",
        )
    console.print(lt)

    # Tree hierarchy
    tree = Tree("[bold cyan]HNSW Layers[/]")
    for lv_info in reversed(info["levels"]):
        lv = lv_info["level"]
        total = info["total_nodes"]
        pct = lv_info["nodes"] / total * 100 if total > 0 else 0
        style = "bold yellow" if lv == info["max_level"] else "cyan"
        branch = tree.add(
            f"[{style}]Level {lv}[/]: {lv_info['nodes']} nodes "
            f"({pct:.1f}%), avg degree {lv_info['avg_degree']:.1f}"
        )
        if lv == info["max_level"] and info["entry_point"] is not None:
            branch.add(f"[yellow]Entry point: doc {info['entry_point']}[/]")
    console.print(Panel(tree, title="[bold]Layer Hierarchy[/]", border_style="green"))

    # Density & size
    blob_row = db.fetch_one(
        "SELECT LENGTH(value) FROM __sqfox_hnsw WHERE key = 'graph'"
    )
    blob_size = blob_row[0] if blob_row and blob_row[0] else 0
    total_edges = sum(lv["edges"] for lv in info["levels"])
    total = info["total_nodes"]

    dt = Table(
        title="Density & Size", box=box.ROUNDED,
        title_style="bold blue",
    )
    dt.add_column("Metric", style="bold")
    dt.add_column("Value", justify="right")
    dt.add_row("Total edges", str(total_edges))
    dt.add_row("Global avg degree", f"{total_edges / total:.1f}" if total else "0")
    dt.add_row("Graph BLOB", f"{blob_size / 1024:.1f} KB")
    dt.add_row("Bytes/vector", f"{blob_size / total:.0f}" if total else "0")
    dt.add_row("Projected 100K", f"{blob_size / total * 100_000 / 1024 / 1024:.1f} MB" if total else "0")
    console.print(dt)


def show_search_trace(query_text, steps, results, search_ms):
    """Render search path + results."""
    console.print()
    console.print(f"  [bold white]{query_text}[/]")
    console.print()

    if not steps:
        console.print("  [dim]Empty index[/]")
        return

    tt = Table(box=box.SIMPLE_HEAVY, title_style="bold magenta")
    tt.add_column("#", justify="center", style="dim", width=4)
    tt.add_column("Layer", justify="center", width=7)
    tt.add_column("Doc", justify="right", width=6)
    tt.add_column("Distance", justify="right", width=18)
    tt.add_column("", width=10)

    prev_dist = None
    for i, s in enumerate(steps):
        icons = {
            "entry": "[bold cyan]ENTER[/]",
            "jump":  "[bold green]JUMP[/]",
            "drop":  "[yellow]DROP[/]",
            "found": "[bold magenta]FOUND[/]",
        }
        if prev_dist is not None and s["distance"] < prev_dist:
            delta = prev_dist - s["distance"]
            d_str = f"[green]{s['distance']:.4f}[/] [dim](-{delta:.4f})[/]"
        else:
            d_str = f"{s['distance']:.4f}"
        prev_dist = s["distance"]

        tt.add_row(
            str(i + 1), f"L{s['level']}", str(s["node"]),
            d_str, icons.get(s["action"], "?"),
        )

    console.print(tt)

    # Summary line
    if len(steps) >= 2:
        d0, d1 = steps[0]["distance"], steps[-1]["distance"]
        pct = (1 - d1 / d0) * 100 if d0 > 0 else 0
        console.print(
            f"  [dim]{d0:.4f}[/] -> [bold green]{d1:.4f}[/]  "
            f"([green]{pct:.0f}% closer[/])  "
            f"[dim]{len(steps)} hops, L{steps[0]['level']}->L0[/]"
        )

    # Search results
    if results:
        console.print()
        console.print(f"  [dim]Top results ({search_ms:.0f}ms):[/]")
        for i, r in enumerate(results):
            preview = r.text[:90].replace("\n", " ")
            sc = "bold green" if r.score >= 0.7 else "yellow" if r.score >= 0.6 else "dim"
            meta = r.metadata
            tag = f"[dim]{meta.get('lang', '?')}/{meta.get('domain', '?')}[/]"
            console.print(
                f"    {'>>>' if i == 0 else '   '} [{sc}]{r.score:.3f}[/] "
                f"{tag} {preview}"
            )


def show_node(node_id, backend, db):
    """Inspect a single node: text, level, neighbors."""
    if node_id not in backend._node_levels:
        console.print(f"  [red]Node {node_id} not found[/]")
        return

    level = backend._node_levels[node_id]
    row = db.fetch_one(
        "SELECT content, metadata FROM documents WHERE id = ?", (node_id,)
    )
    text = row["content"] if row else "???"
    meta = row["metadata"] if row else ""

    console.print()
    console.print(Panel(
        f"[bold white]{text}[/]\n\n[dim]{meta}[/]",
        title=f"[bold]Doc {node_id}[/]  level={level}",
        border_style="cyan",
    ))

    # Neighbors per level
    for lv in range(level + 1):
        if lv < len(backend._graphs):
            g = backend._graphs[lv]
            nbrs = [n for n in g.neighbors(node_id) if n not in backend._deleted]
            nbr_str = ", ".join(str(n) for n in nbrs[:20])
            if len(nbrs) > 20:
                nbr_str += f" ... (+{len(nbrs) - 20})"
            console.print(
                f"  L{lv}: [cyan]{len(nbrs)}[/] neighbors  [dim]{nbr_str}[/]"
            )


def show_top_hubs(backend, db, n=10):
    """Show top connected nodes (hubs) on layer 0."""
    deg_map = get_node_degree_map(backend)
    if not deg_map:
        console.print("  [dim]No nodes[/]")
        return

    top = sorted(deg_map.items(), key=lambda x: x[1], reverse=True)[:n]

    console.print()
    ht = Table(
        title=f"Top {n} Hubs (Layer 0)", box=box.ROUNDED,
        title_style="bold yellow",
    )
    ht.add_column("#", justify="center", style="dim", width=4)
    ht.add_column("Doc", justify="right", width=6)
    ht.add_column("Degree", justify="right", width=8)
    ht.add_column("Level", justify="center", width=6)
    ht.add_column("Text", max_width=60)

    for i, (nid, deg) in enumerate(top):
        row = db.fetch_one("SELECT content FROM documents WHERE id = ?", (nid,))
        text = row["content"][:60].replace("\n", " ") if row else "?"
        lv = backend._node_levels.get(nid, 0)
        ht.add_row(str(i + 1), str(nid), str(deg), str(lv), f"[dim]{text}[/]")

    console.print(ht)


def show_help():
    console.print()
    console.print(Panel(
        "[bold]<query>[/]       — trace search path through HNSW + show results\n"
        "[bold]/stats[/]        — graph structure, density, size analysis\n"
        "[bold]/node N[/]       — inspect doc N: text, level, neighbors\n"
        "[bold]/top[/]          — top-10 most connected nodes (hubs)\n"
        "[bold]/help[/]         — this message\n"
        "[bold]/quit[/]         — exit",
        title="[bold]Commands[/]",
        border_style="cyan",
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    demo_dir = Path(__file__).parent / "data"
    demo_dir.mkdir(exist_ok=True)
    db_path = str(demo_dir / "hnsw_xray.db")

    # Clean up previous run
    for suffix in ("", "-wal", "-shm"):
        p = Path(db_path + suffix)
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    ndim = 256  # MRL truncated

    # Banner
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]HNSW X-Ray[/]\n"
            "[dim]Interactive Graph Inspector[/]\n"
            f"[dim]{len(KNOWLEDGE_BASE)} docs, Qwen3-Embedding-0.6B, 256 dim[/]"
        ),
        border_style="bold cyan",
        box=box.DOUBLE,
    ))

    # Load model
    console.print()
    embedder = QwenEmbedder()

    # Build index
    console.print()
    console.rule("[bold green]Building HNSW Index[/]")
    console.print()

    backend = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Ingesting knowledge base", total=len(KNOWLEDGE_BASE)
        )
        t0 = time.time()

        db = SQFox(db_path, vector_backend=backend)
        db.start()

        for doc in KNOWLEDGE_BASE:
            db.ingest(
                doc["text"], metadata=doc["meta"],
                embed_fn=embedder, wait=True,
            )
            progress.advance(task)

        ingest_time = time.time() - t0

    db.ensure_schema(SchemaState.SEARCHABLE)
    time.sleep(0.3)

    console.print()
    it = Table(title="Ingest Complete", box=box.ROUNDED, title_style="bold green")
    it.add_column("Metric", style="bold")
    it.add_column("Value", justify="right")
    it.add_row("Documents", str(len(KNOWLEDGE_BASE)))
    it.add_row("Dimensions", str(ndim))
    it.add_row("Backend", f"[bold cyan]{db.vector_backend_name}[/]")
    it.add_row("Time", f"{ingest_time:.1f}s")
    console.print(it)

    # Show initial stats
    show_stats(backend, ndim, db)

    # --- Interactive REPL ---
    console.print()
    console.rule("[bold magenta]Interactive Mode[/]")
    show_help()

    try:
        while True:
            console.print()
            try:
                raw = Prompt.ask("[bold cyan]xray[/]")
            except (EOFError, KeyboardInterrupt):
                break

            line = raw.strip()
            if not line:
                continue

            if line.lower() in ("/quit", "/exit", "/q"):
                break

            if line.lower() == "/help":
                show_help()
                continue

            if line.lower() == "/stats":
                show_stats(backend, ndim, db)
                continue

            if line.lower() == "/top":
                show_top_hubs(backend, db)
                continue

            if line.lower().startswith("/node"):
                parts = line.split()
                if len(parts) < 2 or not parts[1].isdigit():
                    console.print("  [dim]Usage: /node <id>[/]")
                    continue
                show_node(int(parts[1]), backend, db)
                continue

            if line.startswith("/"):
                console.print(f"  [red]Unknown command: {line}[/]  Type /help")
                continue

            # --- Search query ---
            query_vec = embedder.embed_query(line)
            conn = db._get_reader_connection()
            steps = trace_search_path(backend, query_vec, conn)

            t0 = time.time()
            results = db.search(line, embed_fn=embedder, limit=5)
            search_ms = (time.time() - t0) * 1000

            show_search_trace(line, steps, results, search_ms)

    finally:
        db.stop()
        # Cleanup
        for suffix in ("", "-wal", "-shm"):
            p = Path(db_path + suffix)
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    console.print()
    console.print("[dim]Done.[/]")


if __name__ == "__main__":
    main()
