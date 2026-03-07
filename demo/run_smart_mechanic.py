"""
sqfox demo: Self-Diagnosing CNC Machine — real-time telemetry + auto-RAG.

Simulates a CNC milling center with sensors (spindle temp, vibration, coolant
pressure, servo load). When an anomaly or alarm code is detected, the system
automatically searches the equipment manual and outputs a diagnosis with
actionable instructions.

Demonstrates AsyncSQFox dual-pool architecture:
  - I/O pool: fast telemetry writes + reads (never blocked)
  - CPU pool: heavy embedding + hybrid search (isolated)

Usage:
  python demo/run_smart_mechanic.py
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
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
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich import box

from sqfox import AsyncSQFox, SchemaState

console = Console()

# ---------------------------------------------------------------------------
# Embedding adapter (same pattern as run_demo.py)
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
        console.print(f"  [green]Model loaded in {elapsed:.1f}s[/] (256 dim)")
    return _model


class QwenEmbedder:
    def __init__(self):
        self.model = get_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, prompt_name="query").tolist()


# ---------------------------------------------------------------------------
# CNC Machine Simulator
# ---------------------------------------------------------------------------

# Sensors: (name, unit, normal_min, normal_max, alarm_lo, alarm_hi, color)
CNC_SENSORS = [
    ("spindle_temp",     "C",     35, 60,  None,  80, "yellow"),
    ("spindle_rpm",      "RPM",    0, 8000, None, 9000, "magenta"),
    ("spindle_load",     "%",      0, 70,   None,  90, "red"),
    ("vibration_x",      "mm/s", 0.0, 4.0,  None, 7.1, "red"),
    ("vibration_z",      "mm/s", 0.0, 3.5,  None, 7.1, "red"),
    ("coolant_pressure",  "bar",  3.0, 6.0,   2.0, 8.0, "cyan"),
    ("coolant_temp",      "C",    15,  35,   None,  45, "cyan"),
    ("servo_load_x",      "%",     0,  50,   None,  85, "green"),
    ("servo_load_y",      "%",     0,  50,   None,  85, "green"),
    ("servo_load_z",      "%",     0,  60,   None,  85, "green"),
    ("feed_rate",       "mm/min",  0, 5000,  None, None, "blue"),
    ("power_total",       "kW",  0.5, 8.0,  None,  12, "white"),
]

# Alarm codes (CNC controller style)
ALARM_CODES = {
    "AL-0107": "Spindle overheat detected by thermal sensor",
    "AL-0350": "Excessive vibration on X axis — possible tool imbalance",
    "AL-0410": "Coolant pressure below minimum threshold",
    "AL-0511": "Servo overload on Z axis during rapid traverse",
}

# Thresholds for monitoring
THRESHOLDS = {
    "spindle_temp":     (None, 80,  "C"),
    "vibration_x":      (None, 7.1, "mm/s"),
    "vibration_z":      (None, 7.1, "mm/s"),
    "coolant_pressure":  (2.0, None, "bar"),
    "servo_load_x":     (None, 85,  "%"),
    "servo_load_y":     (None, 85,  "%"),
    "servo_load_z":     (None, 85,  "%"),
    "spindle_load":     (None, 90,  "%"),
}


class CNCSimulator:
    """Simulates a CNC milling center with realistic sensor drift."""

    def __init__(self):
        self._tick = 0
        self._alarms_fired = set()
        # Base values ("idle machine just started")
        self._base = {s[0]: (s[2] + s[3]) / 2 for s in CNC_SENSORS}

    def read(self) -> dict[str, float]:
        self._tick += 1
        t = self._tick
        data = {}

        for name, unit, lo, hi, *_ in CNC_SENSORS:
            base = self._base[name]
            noise = random.uniform(-0.5, 0.5) * (hi - lo) * 0.05
            data[name] = round(base + noise, 2)

        # Phase 1 (tick 1-30): normal machining
        if t <= 30:
            data["spindle_rpm"] = round(4000 + random.uniform(-100, 100), 0)
            data["spindle_load"] = round(35 + random.uniform(-5, 10), 1)
            data["feed_rate"] = round(2000 + random.uniform(-50, 50), 0)

        # Phase 2 (tick 31-50): heavy cut, spindle heats up
        elif t <= 50:
            progress = (t - 30) / 20  # 0..1
            data["spindle_rpm"] = round(6000 + random.uniform(-50, 50), 0)
            data["spindle_load"] = round(55 + progress * 30 + random.uniform(-3, 3), 1)
            data["spindle_temp"] = round(55 + progress * 30 + random.uniform(-1, 2), 1)
            data["vibration_x"] = round(2.0 + progress * 4 + random.uniform(-0.3, 0.3), 2)
            data["power_total"] = round(5 + progress * 5, 1)

        # Phase 3 (tick 51-65): coolant fails, cascade
        elif t <= 65:
            progress = (t - 50) / 15
            data["spindle_temp"] = round(82 + progress * 10 + random.uniform(0, 2), 1)
            data["coolant_pressure"] = round(3.0 - progress * 2.5 + random.uniform(-0.1, 0.1), 2)
            data["coolant_pressure"] = max(0.3, data["coolant_pressure"])
            data["vibration_x"] = round(5.5 + progress * 3 + random.uniform(-0.2, 0.5), 2)
            data["spindle_load"] = round(75 + progress * 15, 1)
            data["servo_load_z"] = round(50 + progress * 40, 1)

        # Phase 4 (tick 66+): machine stops, cooling down
        else:
            progress = min((t - 65) / 20, 1.0)
            data["spindle_rpm"] = 0
            data["spindle_load"] = 0
            data["feed_rate"] = 0
            data["spindle_temp"] = round(90 - progress * 30 + random.uniform(-1, 1), 1)
            data["vibration_x"] = round(0.5 + random.uniform(-0.2, 0.2), 2)

        return data

    def get_alarms(self) -> list[tuple[str, str]]:
        new_alarms = []
        t = self._tick

        triggers = [
            (45, "AL-0350"),  # vibration during heavy cut
            (55, "AL-0107"),  # spindle overheat
            (58, "AL-0410"),  # coolant loss
            (62, "AL-0511"),  # servo overload
        ]
        for tick_at, code in triggers:
            if t >= tick_at and code not in self._alarms_fired:
                self._alarms_fired.add(code)
                new_alarms.append((code, ALARM_CODES[code]))

        return new_alarms


# ---------------------------------------------------------------------------
# Equipment manuals (knowledge base)
# ---------------------------------------------------------------------------

EQUIPMENT_MANUAL = [
    {
        "text": "AL-0107 Перегрев шпинделя: температура превысила 80 градусов Цельсия. "
                "Немедленные действия: 1) Остановить шпиндель командой M05. 2) Проверить давление "
                "СОЖ на манометре — норма 3-6 бар. 3) Осмотреть сопла подачи СОЖ на засор. "
                "4) Проверить уровень СОЖ в баке. Если температура выше 95°C — НЕ запускать "
                "шпиндель до полного остывания. Возможные причины: засор канала подачи СОЖ, "
                "износ переднего подшипника шпинделя (проверить радиальный люфт), "
                "перегрузка при обработке закалённой стали без снижения оборотов. "
                "См. раздел 4.2 руководства по обслуживанию.",
        "meta": {"code": "AL-0107", "type": "alarm", "section": "spindle"},
    },
    {
        "text": "AL-0350 Повышенная вибрация по оси X: уровень превысил 7.1 мм/с (порог ISO 10816). "
                "Немедленно снизить обороты шпинделя и подачу. Проверить: 1) Затяжку инструмента "
                "в цанге/патроне — момент затяжки согласно таблице на стр. 156. 2) Биение инструмента "
                "индикатором — допуск 0.01 мм. 3) Балансировку оправки при работе выше 6000 об/мин. "
                "4) Состояние подшипников шпинделя — замерить виброскорость на корпусе. "
                "При вибрации выше 11 мм/с — аварийная остановка.",
        "meta": {"code": "AL-0350", "type": "alarm", "section": "vibration"},
    },
    {
        "text": "AL-0410 Падение давления СОЖ ниже минимума (2.0 бар). Система охлаждения: "
                "проверить уровень в баке, состояние насоса, фильтр грубой очистки (засор каждые 200 "
                "моточасов). При давлении ниже 1.5 бар шпиндель автоматически останавливается. "
                "Замена фильтра: отключить насос, слить остатки из магистрали, заменить элемент "
                "(артикул CF-200), залить свежую СОЖ концентрацией 5-8%. "
                "Проверить уплотнения на муфте быстрого соединения — течь в районе шпинделя "
                "приводит к потере давления и попаданию СОЖ в подшипниковый узел.",
        "meta": {"code": "AL-0410", "type": "alarm", "section": "coolant"},
    },
    {
        "text": "AL-0511 Перегрузка сервопривода оси Z при ускоренном перемещении. Ток двигателя "
                "превысил 85% номинала. Проверить: 1) Смазку направляющих оси Z — автолубрикатор, "
                "уровень масла в бачке. 2) Натяжение ремня привода ШВП — прогиб 3-5 мм при усилии "
                "10 Н. 3) Люфт ШВП — допуск 0.005 мм, замерить индикатором. 4) Не зажат ли "
                "торможением стол или суппорт. При повторении ошибки — вызвать наладчика для "
                "проверки энкодера и драйвера сервопривода.",
        "meta": {"code": "AL-0511", "type": "alarm", "section": "servo"},
    },
    {
        "text": "Регламент обслуживания шпинделя: замена смазки подшипников каждые 4000 моточасов. "
                "Проверка радиального люфта при каждом ТО — допуск 0.002 мм. Замена подшипников "
                "при люфте свыше 0.005 мм или при устойчивом росте вибрации выше 4 мм/с на рабочих "
                "оборотах. Использовать только подшипники класса точности P4 (ABEC-7) или выше.",
        "meta": {"type": "maintenance", "section": "spindle"},
    },
    {
        "text": "Калибровка осей после замены ШВП: 1) Прогреть станок 30 минут на холостом ходу. "
                "2) Выполнить автоматический цикл компенсации люфта. 3) Проверить точность "
                "позиционирования лазерным интерферометром — допуск ±0.005 мм на 300 мм хода. "
                "4) Обновить таблицу коррекции шага в параметрах ЧПУ (P.1851-P.1899).",
        "meta": {"type": "calibration", "section": "axes"},
    },
    {
        "text": "Замена инструмента в магазине: 1) Перевести станок в режим MDI. 2) Вызвать нужную "
                "ячейку командой T## M06. 3) Проверить вылет инструмента — измерить привязку по Z "
                "через прибор Tool Setter. 4) Обновить таблицу коррекции инструмента. При смене "
                "типа обработки (черновая - чистовая) проверить биение нового инструмента.",
        "meta": {"type": "procedure", "section": "toolchange"},
    },
    {
        "text": "Концентрация СОЖ: рабочий диапазон 5-8% для универсальной эмульсии. Проверка "
                "рефрактометром еженедельно. При концентрации ниже 4% — риск коррозии заготовок "
                "и направляющих. Выше 10% — пенообразование, снижение теплоотвода, раздражение "
                "кожи оператора. Полная замена СОЖ каждые 3 месяца или при появлении запаха.",
        "meta": {"type": "maintenance", "section": "coolant"},
    },
    {
        "text": "Аварийная остановка (E-STOP): при нажатии красной кнопки все оси и шпиндель "
                "останавливаются немедленно. Для возобновления работы: 1) Устранить причину "
                "аварии. 2) Отжать кнопку E-STOP. 3) Нажать RESET на панели ЧПУ. 4) Выполнить "
                "референтное перемещение всех осей (Home / Zero Return). Без референтного "
                "перемещения координаты могут быть потеряны.",
        "meta": {"type": "procedure", "section": "safety"},
    },
    {
        "text": "Мониторинг нагрузки сервоприводов: нормальная нагрузка при фрезеровании "
                "алюминия 20-40%, стали 40-60%, нержавеющей стали 50-70%. Устойчивое превышение "
                "80% указывает на: затупленный инструмент, неверные режимы резания, механическую "
                "проблему (тугой ход направляющих, перетянутая гайка ШВП). "
                "Контроль: параметры диагностики D.001-D.008 на панели ЧПУ.",
        "meta": {"type": "diagnostics", "section": "servo"},
    },
]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run(backend=None):
    demo_dir = Path(__file__).parent / "data"
    demo_dir.mkdir(exist_ok=True)

    telemetry_path = str(demo_dir / "cnc_telemetry.db")
    knowledge_path = str(demo_dir / "cnc_manuals.db")

    # Clean previous run
    for p in (telemetry_path, knowledge_path):
        for suffix in ("", "-wal", "-shm"):
            f = Path(p + suffix)
            if f.exists():
                f.unlink()

    embedder = QwenEmbedder()
    sim = CNCSimulator()

    async with (
        AsyncSQFox(telemetry_path) as telemetry_db,
        AsyncSQFox(knowledge_path, max_cpu_workers=1, vector_backend=backend) as knowledge_db,
    ):
        # --- Load equipment manual ---
        console.print()
        console.rule("[bold cyan]Loading Equipment Manual[/]")
        console.print()

        t0 = time.time()
        for doc in EQUIPMENT_MANUAL:
            await knowledge_db.ingest(
                doc["text"], embed_fn=embedder, metadata=doc["meta"],
            )
        await knowledge_db.ensure_schema(SchemaState.SEARCHABLE)
        await asyncio.sleep(0.2)
        load_time = time.time() - t0

        manual_table = Table(box=box.ROUNDED, title_style="bold green")
        manual_table.add_column("Metric", style="bold")
        manual_table.add_column("Value", justify="right")
        manual_table.add_row("Documents", str(len(EQUIPMENT_MANUAL)))
        manual_table.add_row("Load time", f"{load_time:.1f}s")
        console.print(manual_table)

        # --- Create telemetry tables ---
        await telemetry_db.write("""
            CREATE TABLE IF NOT EXISTS sensor_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                data TEXT NOT NULL
            )
        """, wait=True)
        await telemetry_db.write("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                type TEXT NOT NULL,
                code TEXT,
                value REAL,
                diagnosis TEXT
            )
        """, wait=True)

        # --- Monitoring loop ---
        console.print()
        console.rule("[bold red]CNC Machine Monitoring[/]")
        console.print()

        sensor_header = "  [dim]{tick:>3}[/]  " \
            "[yellow]Spindle:{rpm:>5.0f}rpm {temp:>5.1f}C {load:>4.1f}%[/]  " \
            "[red]Vib:{vib:>4.2f}[/]  " \
            "[cyan]Cool:{cp:>4.2f}bar[/]  " \
            "[green]Servo Z:{sz:>4.1f}%[/]"

        alerts_log: list[dict] = []
        alert_cooldown: dict[str, int] = {}
        cooldown_ticks = 15
        total_ticks = 80

        with Live(console=console, refresh_per_second=8, transient=True) as live:
            for tick in range(1, total_ticks + 1):
                data = sim.read()

                # Fire-and-forget telemetry write (I/O pool)
                await telemetry_db.write(
                    "INSERT INTO sensor_log (data) VALUES (?)",
                    (json.dumps(data, ensure_ascii=False),),
                )

                # Build status line
                status = Text.from_markup(sensor_header.format(
                    tick=tick,
                    rpm=data.get("spindle_rpm", 0),
                    temp=data.get("spindle_temp", 0),
                    load=data.get("spindle_load", 0),
                    vib=data.get("vibration_x", 0),
                    cp=data.get("coolant_pressure", 0),
                    sz=data.get("servo_load_z", 0),
                ))
                live.update(status)

                # Check thresholds
                for sensor, (lo, hi, unit) in THRESHOLDS.items():
                    value = data.get(sensor)
                    if value is None:
                        continue
                    out_of_range = (lo is not None and value < lo) or \
                                   (hi is not None and value > hi)
                    if not out_of_range:
                        continue

                    # Cooldown: avoid expensive RAG search on every anomalous tick
                    last = alert_cooldown.get(sensor, -cooldown_ticks)
                    if tick - last < cooldown_ticks:
                        continue
                    alert_cooldown[sensor] = tick

                    # Auto-RAG: search manual (CPU pool — does NOT block telemetry writes)
                    query = f"{sensor.replace('_', ' ')} значение {value:.1f} {unit} причины неисправности"
                    t_search = time.time()
                    results = await knowledge_db.search(
                        query, embed_fn=embedder, limit=2,
                    )
                    search_ms = (time.time() - t_search) * 1000
                    diagnosis = results[0].text if results else None

                    threshold_str = f"{lo or '—'}..{hi or '—'} {unit}"

                    alerts_log.append({
                        "tick": tick,
                        "type": "threshold",
                        "sensor": sensor,
                        "value": f"{value:.1f} {unit}",
                        "norm": threshold_str,
                        "diagnosis": diagnosis,
                        "search_ms": search_ms,
                    })

                    await telemetry_db.write(
                        "INSERT INTO alerts (type, code, value, diagnosis) "
                        "VALUES (?, ?, ?, ?)",
                        ("threshold", sensor, value, diagnosis),
                    )

                # Check alarm codes
                for code, desc in sim.get_alarms():
                    query = f"alarm {code} {desc} troubleshooting"
                    t_search = time.time()
                    results = await knowledge_db.search(
                        query, embed_fn=embedder, limit=2,
                    )
                    search_ms = (time.time() - t_search) * 1000
                    diagnosis = results[0].text if results else None

                    alerts_log.append({
                        "tick": tick,
                        "type": "alarm",
                        "sensor": code,
                        "value": desc,
                        "norm": "--",
                        "diagnosis": diagnosis,
                        "search_ms": search_ms,
                    })

                    await telemetry_db.write(
                        "INSERT INTO alerts (type, code, value, diagnosis) "
                        "VALUES (?, ?, ?, ?)",
                        ("alarm", code, None, diagnosis),
                    )

                await asyncio.sleep(0.08)

        # --- Results ---
        console.print()
        console.rule("[bold green]Monitoring Complete[/]")
        console.print()

        # Telemetry stats
        row = await telemetry_db.fetch_one("SELECT COUNT(*) FROM sensor_log")
        log_count = row[0] if row else 0
        row = await telemetry_db.fetch_one("SELECT COUNT(*) FROM alerts")
        alert_count = row[0] if row else 0

        stats_table = Table(
            title="Telemetry Summary", box=box.ROUNDED, title_style="bold green",
        )
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")
        stats_table.add_row("Sensor readings logged", str(log_count))
        stats_table.add_row("Alerts triggered", str(alert_count))
        stats_table.add_row("Simulation ticks", str(total_ticks))
        stats_table.add_row("Telemetry DB", Path(telemetry_path).name)
        stats_table.add_row("Knowledge DB", Path(knowledge_path).name)
        console.print(stats_table)

        # Deduplicate alerts for display — show first occurrence per sensor
        seen = set()
        unique_alerts = []
        for a in alerts_log:
            key = a["sensor"]
            if key not in seen:
                seen.add(key)
                unique_alerts.append(a)

        if unique_alerts:
            console.print()
            console.print("[bold]Alerts & Auto-Diagnoses[/]")
            console.print()

            for a in unique_alerts:
                if a["type"] == "alarm":
                    title = f"[bold red]ALARM[/] [bold]{a['sensor']}[/]"
                    border = "red"
                else:
                    title = f"[bold yellow]THRESHOLD[/] [bold]{a['sensor']}[/]"
                    border = "yellow"

                parts = [
                    f"[white]tick {a['tick']}  value: {a['value']}[/]  "
                    f"[dim](search: {a['search_ms']:.0f}ms)[/]",
                ]
                if a.get("norm") and a["norm"] != "--":
                    parts.append(f"[dim]Norm: {a['norm']}[/]")
                if a.get("diagnosis"):
                    parts.append("")
                    parts.append(f"[bold green]Manual:[/]\n{a['diagnosis']}")

                console.print(Panel(
                    "\n".join(parts),
                    title=title,
                    border_style=border,
                    box=box.ROUNDED,
                    width=min(console.width, 90),
                ))
                console.print()

        # Architecture note
        console.print(Panel(
            Align.center(
                "[bold]Architecture: AsyncSQFox dual-pool[/]\n\n"
                "[yellow]I/O pool[/]  — telemetry writes (fire-and-forget)\n"
                "[yellow]          [/]  — sensor reads (fetch_one)\n"
                "[cyan]CPU pool[/] — embedding + hybrid search\n\n"
                "[dim]Heavy search does NOT block telemetry writes.[/]\n"
                "[dim]Both pools run in the same Python process.[/]"
            ),
            border_style="cyan",
            box=box.ROUNDED,
        ))


def main():
    parser = argparse.ArgumentParser(description="sqfox CNC Mechanic demo")
    parser.add_argument(
        "--backend", default=None,
        help="Vector backend: flat, hnsw, usearch",
    )
    args = parser.parse_args()

    backend_label = f"backend: {args.backend}" if args.backend else "backend: default"
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]sqfox demo: Self-Diagnosing CNC Machine[/]\n"
            "[dim]AsyncSQFox — dual-pool telemetry + auto-RAG[/]\n"
            f"[dim]Qwen3-Embedding-0.6B, 256 dim (MRL), {backend_label}[/]"
        ),
        border_style="bold red",
        box=box.DOUBLE,
    ))

    console.print()
    get_model()

    asyncio.run(run(backend=args.backend))

    console.print()
    console.print(Panel(
        Align.center("[bold green]Demo complete![/]"),
        border_style="green",
        box=box.DOUBLE,
    ))


if __name__ == "__main__":
    main()
