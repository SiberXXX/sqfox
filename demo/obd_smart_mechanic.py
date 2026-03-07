"""
OBD-II Smart Mechanic -- real-time car diagnostics with auto-RAG.

Reads live telemetry from an ELM327 adapter via python-obd,
logs everything to sqfox, and when a DTC (Check Engine code) appears,
automatically searches a local service manual for the diagnosis.

Hardware:
  - ELM327 adapter (Bluetooth/USB, ~$5-10)
  - Raspberry Pi / old laptop on the passenger seat

Dependencies:
  pip install obd sqfox[search-ru] sentence-transformers rich

Usage:
  # Simulation mode (default, no adapter needed):
  python obd_smart_mechanic.py

  # With real ELM327 adapter:
  python obd_smart_mechanic.py --real
  python obd_smart_mechanic.py --real --port /dev/ttyUSB0

  # Load your own service manual:
  python obd_smart_mechanic.py --load-manual path/to/manual.txt
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

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.live import Live
from rich.columns import Columns
from rich import box

from sqfox import AsyncSQFox, SchemaState, sentence_chunker

console = Console()


# ---------------------------------------------------------------------------
# Embedding adapter (swap for your model)
# ---------------------------------------------------------------------------

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        with console.status("[bold cyan]Loading Qwen3-Embedding-0.6B...[/]", spinner="dots"):
            t0 = time.time()
            _model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B", truncate_dim=256,
            )
            elapsed = time.time() - t0
        console.print(f"  [green]Model loaded in {elapsed:.1f}s (256 dim)[/]")
    return _model


class OBDEmbedder:
    """Instruction-aware embedder."""
    def __init__(self):
        self.model = get_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, prompt_name="query").tolist()


embedder = None  # initialized after model loads


# ---------------------------------------------------------------------------
# OBD connection (real or simulated)
# ---------------------------------------------------------------------------

WATCH_CMDS = [
    "RPM", "COOLANT_TEMP", "ENGINE_LOAD", "INTAKE_TEMP",
    "THROTTLE_POS", "TIMING_ADVANCE",
    "SHORT_FUEL_TRIM_1", "LONG_FUEL_TRIM_1",
]

THRESHOLDS = {
    "COOLANT_TEMP": (None, 105, "C"),
    "RPM": (None, 6500, "rpm"),
    "ENGINE_LOAD": (None, 95, "%"),
    "SHORT_FUEL_TRIM_1": (-25, 25, "%"),
    "LONG_FUEL_TRIM_1": (-20, 20, "%"),
}

SENSOR_LABELS = {
    "RPM": ("RPM", "", "magenta"),
    "COOLANT_TEMP": ("CoolT", "C", "red"),
    "ENGINE_LOAD": ("Load", "%", "yellow"),
    "INTAKE_TEMP": ("IntT", "C", "cyan"),
    "THROTTLE_POS": ("Thrtl", "%", "green"),
    "TIMING_ADVANCE": ("TimAd", "d", "blue"),
    "SHORT_FUEL_TRIM_1": ("STFT", "%", "white"),
    "LONG_FUEL_TRIM_1": ("LTFT", "%", "white"),
}


class OBDSimulator:
    """Fake OBD source for testing without hardware."""

    def __init__(self):
        self._tick = 0
        self._dtc_fired = False

    def read(self) -> dict[str, float]:
        self._tick += 1
        data = {
            "RPM": 800 + random.uniform(-50, 2000),
            "COOLANT_TEMP": 85 + random.uniform(-5, 10),
            "ENGINE_LOAD": 30 + random.uniform(-10, 40),
            "INTAKE_TEMP": 35 + random.uniform(-5, 15),
            "THROTTLE_POS": 15 + random.uniform(-5, 60),
            "TIMING_ADVANCE": 12 + random.uniform(-4, 8),
            "SHORT_FUEL_TRIM_1": random.uniform(-8, 8),
            "LONG_FUEL_TRIM_1": random.uniform(-5, 5),
        }
        if self._tick >= 40:
            data["COOLANT_TEMP"] = 108 + random.uniform(0, 5)
        return data

    def get_dtc(self) -> list[tuple[str, str]]:
        if self._tick >= 50 and not self._dtc_fired:
            self._dtc_fired = True
            return [("P0171", "System Too Lean (Bank 1)")]
        return []


class OBDReal:
    """Real ELM327 adapter via python-obd."""

    def __init__(self, port: str | None = None):
        import obd
        self._obd = obd
        self._conn = obd.OBD(port)
        if not self._conn.is_connected():
            raise RuntimeError(
                f"Cannot connect to ELM327 on {port or 'auto'}. "
                "Check adapter and port."
            )
        console.print(f"  [green]Connected: {self._conn.port_name()}[/]")

    def read(self) -> dict[str, float]:
        data = {}
        for name in WATCH_CMDS:
            cmd = getattr(self._obd.commands, name, None)
            if cmd is None:
                continue
            resp = self._conn.query(cmd)
            if not resp.is_null():
                data[name] = resp.value.magnitude
        return data

    def get_dtc(self) -> list[tuple[str, str]]:
        resp = self._conn.query(self._obd.commands.GET_DTC)
        if resp.is_null():
            return []
        return [(code, desc) for code, desc in resp.value]


# ---------------------------------------------------------------------------
# Sample manual
# ---------------------------------------------------------------------------

SAMPLE_MANUAL = [
    {
        "text": "Код ошибки P0171 (System Too Lean Bank 1): бедная топливовоздушная смесь. "
                "Наиболее частые причины: подсос воздуха за ДМРВ (MAF-сенсором), трещина в гофре "
                "впускного коллектора, падение давления в топливной рампе ниже 3.0 бар, загрязнение "
                "или неисправность ДМРВ.\n\n"
                "Первое действие: визуально проверить гофру от воздушного фильтра к дроссельной "
                "заслонке на наличие трещин и хомуты на герметичность. Проверить показания ДМРВ "
                "на холостом ходу (норма 8-10 кг/ч). Проверить давление в топливной рампе "
                "манометром (норма 3.0-4.0 бар). См. стр. 214.",
        "meta": {"code": "P0171", "type": "dtc"},
    },
    {
        "text": "Код ошибки P0300 (Random/Multiple Cylinder Misfire): пропуски зажигания "
                "в нескольких цилиндрах. Проверить свечи зажигания, высоковольтные провода, "
                "модуль зажигания. При пробеге >60000 км заменить свечи. Проверить компрессию "
                "в цилиндрах. Если пропуски в одном цилиндре -- поменять свечи/катушки местами "
                "для локализации неисправности.",
        "meta": {"code": "P0300", "type": "dtc"},
    },
    {
        "text": "Перегрев двигателя: температура охлаждающей жидкости выше 105 градусов. "
                "Немедленно снизить нагрузку, включить печку на максимум для отвода тепла. "
                "Возможные причины: неисправность термостата (заклинил в закрытом положении), "
                "утечка антифриза, неработающий вентилятор радиатора, забитый радиатор. "
                "Остановить двигатель при достижении 115 градусов. НЕ открывать крышку "
                "расширительного бачка на горячем двигателе -- ожог паром (стр. 87-89).",
        "meta": {"code": "overheat", "type": "threshold"},
    },
    {
        "text": "Датчик массового расхода воздуха (ДМРВ / MAF): калибровка и обслуживание. "
                "Загрязненный ДМРВ занижает показания расхода воздуха, что приводит к обеднению "
                "смеси (P0171/P0174). Очистка: снять датчик, промыть специальным очистителем MAF "
                "(не использовать WD-40 или карб-клинер). Замена при пробеге >100000 км. "
                "После очистки сбросить адаптацию ЭБУ.",
        "meta": {"code": "MAF", "type": "maintenance"},
    },
    {
        "text": "Топливные коррекции (Fuel Trims): Short Term Fuel Trim (STFT) отражает "
                "мгновенную коррекцию ЭБУ. Норма: -10%..+10%. Long Term Fuel Trim (LTFT) -- "
                "долговременная адаптация. Норма: -8%..+8%. Если LTFT > +15% -- бедная смесь "
                "(подсос воздуха или низкое давление топлива). LTFT < -15% -- богатая смесь "
                "(неисправность форсунки или датчика давления). При сбросе LTFT адаптация "
                "восстанавливается за 50-100 км пробега.",
        "meta": {"code": "fuel_trims", "type": "diagnostics"},
    },
    {
        "text": "Регламент замены ремня ГРМ: каждые 90000 км или 5 лет. При замене ремня "
                "обязательно заменить ролик-натяжитель и обводной ролик. Проверить помпу -- "
                "при люфте или течи заменить. Момент затяжки болта шкива коленвала: 90 Нм + "
                "доворот 90 градусов. Метки ГРМ совмещать по сервис-мануалу (стр. 156-162).",
        "meta": {"code": "timing_belt", "type": "maintenance"},
    },
    {
        "text": "Код ошибки P0420 (Catalyst System Efficiency Below Threshold): низкая "
                "эффективность каталитического нейтрализатора. Причины: износ катализатора "
                "(пробег >150000 км), неисправность второго лямбда-зонда, утечка в выпускной "
                "системе перед катализатором. Диагностика: сравнить осциллограммы первого и "
                "второго лямбда-зондов. Второй должен показывать ровную линию ~0.7В.",
        "meta": {"code": "P0420", "type": "dtc"},
    },
    {
        "text": "Давление в топливной рампе: норма 3.0-4.0 бар на холостом ходу. Проверка: "
                "манометр на штуцер рампы, завести двигатель. Ниже 2.5 бар -- неисправен "
                "топливный насос или забит фильтр. Выше 4.5 бар -- неисправен регулятор давления. "
                "После остановки двигателя давление должно удерживаться не менее 5 минут. "
                "Быстрое падение указывает на негерметичность форсунок или обратного клапана.",
        "meta": {"code": "fuel_pressure", "type": "diagnostics"},
    },
]


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

def build_gauges(data: dict[str, float], tick: int, max_ticks: int | None) -> Table:
    """Build a sensor gauges table."""
    tbl = Table(
        box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 1),
        expand=False, title_style="bold",
    )
    tbl.add_column("label", style="dim", width=7)
    tbl.add_column("value", justify="right", width=9)
    tbl.add_column("bar", width=14)
    tbl.add_column("sep", width=1)
    tbl.add_column("label", style="dim", width=7)
    tbl.add_column("value", justify="right", width=9)
    tbl.add_column("bar", width=14)

    items = list(SENSOR_LABELS.items())
    bar_len = 12
    for i in range(0, len(items), 2):
        left = []
        right = []
        for j, side in ((0, left), (1, right)):
            if i + j < len(items):
                key, (label, unit, color) = items[i + j]
                val = data.get(key, 0)
                lo, hi, _ = THRESHOLDS.get(key, (None, None, ""))
                alarm = (lo is not None and val < lo) or (hi is not None and val > hi)
                val_style = "bold red" if alarm else f"bold {color}"
                val_str = f"[{val_style}]{val:6.1f}[/] {unit}"

                ranges = {
                    "RPM": (0, 8000), "COOLANT_TEMP": (50, 130),
                    "ENGINE_LOAD": (0, 100), "INTAKE_TEMP": (0, 60),
                    "THROTTLE_POS": (0, 100), "TIMING_ADVANCE": (0, 30),
                    "SHORT_FUEL_TRIM_1": (-30, 30), "LONG_FUEL_TRIM_1": (-30, 30),
                }
                rmin, rmax = ranges.get(key, (0, 100))
                pct = max(0, min(1, (val - rmin) / (rmax - rmin))) if rmax != rmin else 0
                filled = int(pct * bar_len)
                ch = "!" if alarm else "#"
                clr = "red" if alarm else color
                bar_str = f"[{clr}]{ch * filled}[/][dim]{'.' * (bar_len - filled)}[/]"
                side.extend([f"[bold]{label}[/]", val_str, bar_str])
            else:
                side.extend(["", "", ""])

        tbl.add_row(*left, "", *right)

    return tbl


def build_alert_panel(alerts: list[dict]) -> Panel:
    """Build the alerts panel showing recent alerts with full diagnosis."""
    if not alerts:
        content = Align.center("[dim]No alerts yet[/]")
        return Panel(content, title="[bold green]Alerts[/]", border_style="green", box=box.ROUNDED)

    parts = []
    # Show last 3 alerts to fit on screen
    visible = alerts[-3:]
    for a in visible:
        ts = a["ts"]
        if a["type"] == "dtc":
            header = f"[bold red]CHECK ENGINE[/] [bold white]{a['code']}[/]"
            detail = a["desc"]
        else:
            header = f"[bold yellow]THRESHOLD[/] [bold white]{a['sensor']}[/]"
            detail = f"{a['value']:.1f} {a['unit']} (norm: {a['norm']})"

        text = Text()
        text.append(f"  [{ts}] ", style="dim")
        text.append_text(Text.from_markup(header))
        parts.append(text)
        parts.append(Text(f"    {detail}", style="white"))

        if a.get("diagnosis"):
            # Full text, word-wrapped by rich automatically
            parts.append(Text(f"    {a['diagnosis']}", style="green"))
        parts.append(Text(""))

    border = "red" if any(a["type"] == "dtc" for a in visible) else "yellow"
    return Panel(
        Group(*parts),
        title=f"[bold]Alerts ({len(alerts)} total)[/]",
        border_style=border,
        box=box.ROUNDED,
    )


def build_status_bar(tick: int, max_ticks: int | None, log_count: int, alert_count: int, is_simulation: bool = True) -> Text:
    """Build bottom status bar."""
    tick_str = f"{tick}/{max_ticks}" if max_ticks else str(tick)
    t = Text()
    t.append("  Tick: ", style="dim")
    t.append(tick_str, style="bold")
    t.append("  |  Records: ", style="dim")
    t.append(str(log_count), style="bold cyan")
    t.append("  |  Alerts: ", style="dim")
    t.append(str(alert_count), style="bold yellow" if alert_count else "bold green")
    t.append("  |  ", style="dim")
    if is_simulation:
        t.append("SIMULATION", style="dim magenta")
    else:
        t.append("LIVE", style="bold green")
    return t


def build_dashboard(
    data: dict[str, float],
    alerts: list[dict],
    tick: int,
    max_ticks: int | None,
    log_count: int,
    is_simulation: bool = True,
) -> Group:
    """Compose the full dashboard."""
    return Group(
        build_gauges(data, tick, max_ticks),
        build_alert_panel(alerts),
        build_status_bar(tick, max_ticks, log_count, len(alerts), is_simulation=is_simulation),
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

async def load_manual(knowledge_db: AsyncSQFox, path: str | None):
    """Load service manual into knowledge DB."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
        chunker = sentence_chunker(chunk_size=500, overlap=1)
        doc_id = await knowledge_db.ingest(
            text, chunker=chunker, embed_fn=embedder,
            metadata={"source": Path(path).name},
        )
        console.print(f"  [green]Loaded {path} -> doc {doc_id}[/]")
    else:
        for doc in SAMPLE_MANUAL:
            await knowledge_db.ingest(
                doc["text"], embed_fn=embedder, metadata=doc["meta"],
            )
        console.print(f"  [green]Loaded {len(SAMPLE_MANUAL)} manual entries[/]")

    await knowledge_db.ensure_schema(SchemaState.SEARCHABLE)
    await asyncio.sleep(0.2)


async def diagnose(knowledge_db: AsyncSQFox, query: str) -> str | None:
    """Search knowledge base for diagnosis. Runs on CPU pool."""
    results = await knowledge_db.search(query, embed_fn=embedder, limit=3)
    if not results:
        return None
    return results[0].text


async def run_monitor(
    telemetry_db: AsyncSQFox,
    knowledge_db: AsyncSQFox,
    obd_source,
    *,
    interval: float = 0.2,
    max_ticks: int | None = None,
    is_simulation: bool = True,
):
    """Main monitoring loop with live dashboard."""

    await telemetry_db.write("""
        CREATE TABLE IF NOT EXISTS obd_log (
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
            sensor TEXT,
            value REAL,
            diagnosis TEXT
        )
    """, wait=True)

    seen_dtcs: set[str] = set()
    alert_cooldown: dict[str, int] = {}
    cooldown_ticks = 20
    tick = 0
    log_count = 0
    alerts_log: list[dict] = []
    current_data: dict[str, float] = {}

    with Live(console=console, refresh_per_second=6, screen=False) as live:
        while True:
            tick += 1
            if max_ticks and tick > max_ticks:
                break

            data = obd_source.read()
            if not data:
                await asyncio.sleep(interval)
                continue

            current_data = data

            # Log (fire-and-forget, I/O pool)
            await telemetry_db.write(
                "INSERT INTO obd_log (data) VALUES (?)",
                (json.dumps(data, ensure_ascii=False),),
            )
            log_count += 1

            # Check thresholds
            for sensor, (lo, hi, unit) in THRESHOLDS.items():
                value = data.get(sensor)
                if value is None:
                    continue
                out = (lo is not None and value < lo) or (hi is not None and value > hi)
                if not out:
                    continue
                last = alert_cooldown.get(sensor, -cooldown_ticks)
                if tick - last < cooldown_ticks:
                    continue
                alert_cooldown[sensor] = tick

                query = f"{sensor.replace('_', ' ')} value {value:.1f} {unit} troubleshooting"
                diagnosis = await diagnose(knowledge_db, query)

                norm = f"{lo or '-inf'}..{hi or '+inf'} {unit}"
                alerts_log.append({
                    "ts": time.strftime("%H:%M:%S"),
                    "type": "threshold",
                    "sensor": sensor,
                    "value": value,
                    "unit": unit,
                    "norm": norm,
                    "code": None,
                    "desc": None,
                    "diagnosis": diagnosis,
                })
                await telemetry_db.write(
                    "INSERT INTO alerts (type, sensor, value, diagnosis) VALUES (?, ?, ?, ?)",
                    ("threshold", sensor, value, diagnosis),
                )

            # Check DTC
            for code, desc in obd_source.get_dtc():
                if code in seen_dtcs:
                    continue
                seen_dtcs.add(code)

                query = f"error code {code} {desc} causes diagnostics"
                diagnosis = await diagnose(knowledge_db, query)

                alerts_log.append({
                    "ts": time.strftime("%H:%M:%S"),
                    "type": "dtc",
                    "sensor": None,
                    "value": 0,
                    "unit": "",
                    "norm": "",
                    "code": code,
                    "desc": desc,
                    "diagnosis": diagnosis,
                })
                await telemetry_db.write(
                    "INSERT INTO alerts (type, sensor, value, diagnosis) VALUES (?, ?, ?, ?)",
                    ("dtc", code, None, diagnosis),
                )

            # Update dashboard
            live.update(build_dashboard(
                current_data, alerts_log, tick, max_ticks, log_count,
                is_simulation=is_simulation,
            ))

            await asyncio.sleep(interval)

    return alerts_log


async def main():
    global embedder

    parser = argparse.ArgumentParser(description="OBD-II Smart Mechanic")
    parser.add_argument("--real", action="store_true",
                        help="Use real ELM327 adapter (default: simulation)")
    parser.add_argument("--port", default=None,
                        help="ELM327 serial port (auto-detect if omitted)")
    parser.add_argument("--load-manual", default=None,
                        help="Path to service manual text file")
    parser.add_argument("--db-dir", default="./obd_data",
                        help="Directory for database files")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="Poll interval in seconds")
    parser.add_argument("--backend", default=None,
                        help="Vector backend: flat, hnsw, usearch")
    args = parser.parse_args()

    backend_label = f"backend: {args.backend}" if args.backend else "backend: default"
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]OBD-II Smart Mechanic[/]\n"
            f"[dim]Real-time diagnostics with auto-RAG, {backend_label}[/]"
        ),
        border_style="bold red",
        box=box.DOUBLE,
    ))
    console.print()

    get_model()
    embedder = OBDEmbedder()

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = str(db_dir / "telemetry.db")
    knowledge_path = str(db_dir / "manuals.db")

    async with (
        AsyncSQFox(telemetry_path) as telemetry_db,
        AsyncSQFox(knowledge_path, max_cpu_workers=1, vector_backend=args.backend) as knowledge_db,
    ):
        with console.status("[bold cyan]Loading service manual...[/]"):
            await load_manual(knowledge_db, args.load_manual)

        if args.real:
            obd_source = OBDReal(args.port)
            max_ticks = None
        else:
            console.print("  [dim]Simulation mode (use --real for ELM327)[/]")
            obd_source = OBDSimulator()
            max_ticks = 70

        console.print()
        alerts = await run_monitor(
            telemetry_db, knowledge_db, obd_source,
            interval=args.interval,
            max_ticks=max_ticks,
            is_simulation=not args.real,
        )

        # Final summary
        console.print()
        console.rule("[bold green]Session Complete[/]")
        console.print()

        row = await telemetry_db.fetch_one("SELECT COUNT(*) FROM obd_log")
        log_count = row[0] if row else 0
        row = await telemetry_db.fetch_one("SELECT COUNT(*) FROM alerts")
        alert_count = row[0] if row else 0

        stats = Table(title="Session Summary", box=box.ROUNDED, title_style="bold green")
        stats.add_column("Metric", style="bold")
        stats.add_column("Value", justify="right")
        stats.add_row("Telemetry records", str(log_count))
        stats.add_row("Alerts triggered", str(alert_count))
        stats.add_row("Data saved to", str(db_dir))
        console.print(stats)

        # Show all alerts with full text
        if alerts:
            console.print()
            for a in alerts:
                if a["type"] == "dtc":
                    title = f"[bold red]CHECK ENGINE: {a['code']}[/]"
                    detail = a["desc"]
                else:
                    title = f"[bold yellow]THRESHOLD: {a['sensor']}[/]"
                    detail = f"{a['value']:.1f} {a['unit']} (norm: {a['norm']})"

                content_parts = [f"[white]{detail}[/]"]
                if a.get("diagnosis"):
                    content_parts.append("")
                    content_parts.append(f"[bold green]Manual:[/]\n{a['diagnosis']}")

                border = "red" if a["type"] == "dtc" else "yellow"
                console.print(Panel(
                    "\n".join(content_parts),
                    title=f"{a['ts']}  {title}",
                    border_style=border,
                    box=box.ROUNDED,
                    width=min(console.width, 90),
                ))
                console.print()

        console.print(Panel(
            Align.center("[bold green]Done[/]"),
            border_style="green", box=box.DOUBLE,
        ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user. Exiting cleanly.[/]")
