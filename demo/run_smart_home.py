"""
sqfox demo: Smart Home — preference-aware automation with auto-RAG.

The system learns the owner's habits from natural-language preference records
and adapts the house automatically. When context changes (time, temperature,
motion, day of week), it searches the preference database and decides what
to do — no hardcoded rules, just semantic search over what the owner told it.

Demonstrates:
  - AsyncSQFox as a "personality database" for the house
  - Hybrid search (FTS + vectors) to match context → preference
  - Reranker for better precision on ambiguous situations
  - Concurrent sensor reads + heavy search (dual-pool architecture)

Usage:
  python demo/run_smart_home.py
"""

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
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
from rich.text import Text
from rich.align import Align
from rich import box

from sqfox import AsyncSQFox, SchemaState

console = Console()

# ---------------------------------------------------------------------------
# Embedding
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
# Owner preferences (the "personality" of the house)
# ---------------------------------------------------------------------------

OWNER_PREFERENCES = [
    # --- Morning ---
    {
        "text": "По будням просыпаюсь в 7:00. Люблю, когда за 10 минут до будильника "
                "на кухне включается чайник и начинает греться вода. Свет в спальне "
                "плавно разгорается до 40% яркости — тёплый, 2700K, чтобы не бить по глазам.",
        "meta": {"time": "morning", "day": "weekday", "room": "bedroom,kitchen"},
    },
    {
        "text": "В выходные встаю позже, около 9:30. Чайник не нужен сразу — сначала "
                "кофемашина готовит эспрессо (программа 2, двойной). Шторы открываются "
                "наполовину, свет не включать — хватает дневного.",
        "meta": {"time": "morning", "day": "weekend", "room": "bedroom,kitchen"},
    },
    {
        "text": "Утром в ванной включить тёплый пол за 20 минут до подъёма. "
                "Вентиляция на среднем режиме. Зеркало с подогревом — включить одновременно "
                "с тёплым полом, чтобы не запотело после душа.",
        "meta": {"time": "morning", "room": "bathroom"},
    },

    # --- Work from home ---
    {
        "text": "Когда работаю из дома (обычно с 9:00 до 18:00), в кабинете свет 100% "
                "нейтральный белый 4000K. Температура 22-23 градуса. Если температура "
                "поднимается выше 25 — включить кондиционер на 22 градуса, тихий режим. "
                "Не люблю сквозняки — жалюзи направлять вверх.",
        "meta": {"time": "day", "activity": "work", "room": "office"},
    },
    {
        "text": "Во время работы за компьютером шторы в кабинете закрыть наполовину — "
                "солнце бликует на мониторе с южной стороны. Если облачно — открыть полностью.",
        "meta": {"time": "day", "activity": "work", "room": "office"},
    },

    # --- Afternoon / cooking ---
    {
        "text": "Когда начинаю готовить (датчик движения на кухне + включена плита), "
                "вытяжка на автомате включается на средний режим. Свет над рабочей зоной "
                "100% яркость, над столом — 60%. Включить колонку на кухне — плейлист "
                "'Cooking Jazz' через Spotify.",
        "meta": {"time": "any", "activity": "cooking", "room": "kitchen"},
    },
    {
        "text": "Если температура на кухне выше 28 градусов во время готовки — "
                "открыть окно на микропроветривание (привод на окне) и усилить вытяжку. "
                "Кондиционер на кухне НЕ включать — он гоняет запахи по квартире.",
        "meta": {"time": "any", "activity": "cooking", "room": "kitchen"},
    },

    # --- Evening ---
    {
        "text": "Вечером после 20:00 в гостиной переключить свет в тёплый режим 2700K, "
                "яркость постепенно снижать до 30% к 22:00. Если включен телевизор — "
                "подсветка Ambilight за экраном, основной свет выключить для кинотеатра.",
        "meta": {"time": "evening", "room": "living_room"},
    },
    {
        "text": "Перед сном (около 23:00) включить ночной режим: все светильники плавно "
                "гаснут за 15 минут. Тёплый пол в ванной выключить. Термостат на 20 градусов "
                "ночной. Проверить — все окна закрыты, входная дверь заблокирована.",
        "meta": {"time": "night", "room": "all"},
    },
    {
        "text": "Если ночью встаю (датчик движения в коридоре после 23:00) — "
                "подсветка пола в коридоре 5% тёплый свет, не будить остальных. "
                "Через 3 минуты без движения — погасить.",
        "meta": {"time": "night", "room": "hallway"},
    },

    # --- Temperature ---
    {
        "text": "Зимой (ноябрь-март) температура в квартире 23 градуса днём, 20 ночью. "
                "Если на улице ниже -15, поднять до 24. Тёплый пол в ванной и коридоре "
                "включён постоянно на минимуме. Увлажнитель на 45% в спальне.",
        "meta": {"season": "winter", "room": "all"},
    },
    {
        "text": "Летом кондиционер включать только если выше 26 градусов в комнате. "
                "Целевая температура 23-24. На ночь можно 25 — прохладнее не люблю, "
                "простужаюсь. Вентилятор в спальне можно как альтернативу кондиционеру "
                "при температуре 25-27.",
        "meta": {"season": "summer", "room": "bedroom,living_room"},
    },

    # --- Guests ---
    {
        "text": "Когда приходят гости (несколько человек по датчику движения в прихожей, "
                "или я говорю боту 'гости') — свет в гостиной 80%, тёплый. Включить "
                "колонку — плейлист 'Chill Evening'. Вытяжка на тихом режиме. "
                "Термостат не менять — пусть кондиционер сам подстроится.",
        "meta": {"activity": "guests", "room": "living_room"},
    },

    # --- Away ---
    {
        "text": "Режим 'ушёл из дома': свет везде выключить, кондиционер в эко-режим "
                "(28 летом, 18 зимой), робот-пылесос запустить если не было уборки 2 дня. "
                "Камеры переключить в режим записи по движению. Имитация присутствия "
                "в отпуске — случайно включать свет в гостиной на 1-2 часа вечером.",
        "meta": {"activity": "away", "room": "all"},
    },

    # --- Special ---
    {
        "text": "Сценарий 'кинотеатр': в гостиной шторы закрыть полностью, основной "
                "свет выключить, подсветка за телевизором 10% синий. Звук переключить "
                "на саундбар, громкость 35%. Кондиционер тихий режим.",
        "meta": {"activity": "cinema", "room": "living_room"},
    },
    {
        "text": "Сценарий 'романтический ужин': кухня и гостиная — свет 20% тёплый "
                "янтарный, свечи (умная розетка гирлянды). Музыка — плейлист 'Jazz Ballads' "
                "на 20%. Вытяжка на минимуме.",
        "meta": {"activity": "romantic", "room": "kitchen,living_room"},
    },
]


# ---------------------------------------------------------------------------
# Smart home simulator
# ---------------------------------------------------------------------------

@dataclass
class RoomState:
    temp: float = 22.0
    humidity: float = 45.0
    light_pct: int = 0
    light_temp_k: int = 4000
    motion: bool = False
    window_open: bool = False
    ac_on: bool = False
    ac_target: float = 23.0

@dataclass
class HomeState:
    """Full state of the simulated apartment."""
    bedroom: RoomState = field(default_factory=RoomState)
    kitchen: RoomState = field(default_factory=RoomState)
    living_room: RoomState = field(default_factory=RoomState)
    office: RoomState = field(default_factory=RoomState)
    bathroom: RoomState = field(default_factory=RoomState)
    hallway: RoomState = field(default_factory=RoomState)

    # Appliances
    kettle_on: bool = False
    coffee_on: bool = False
    tv_on: bool = False
    music_playing: str = ""
    curtains_pct: dict = field(default_factory=lambda: {
        "bedroom": 100, "office": 100, "living_room": 100,
    })
    vacuum_running: bool = False
    door_locked: bool = True

    # Outside
    outside_temp: float = 18.0
    cloudy: bool = False
    season: str = "summer"


# Day phases with context triggers
DAY_TIMELINE = [
    {
        "time": "06:50", "label": "Pre-alarm (weekday)",
        "context": "weekday morning 06:50, owner wakes at 07:00, bedroom temp 21C, dark outside",
        "sim": lambda h: (
            setattr(h.bedroom, "motion", False),
            setattr(h, "outside_temp", 12.0),
            setattr(h.kitchen, "temp", 20.0),
        ),
    },
    {
        "time": "07:05", "label": "Owner wakes up",
        "context": "weekday 07:05, motion detected in bedroom, owner just woke up, bathroom needed",
        "sim": lambda h: (
            setattr(h.bedroom, "motion", True),
            setattr(h.bathroom, "motion", True),
        ),
    },
    {
        "time": "09:00", "label": "Start work from home",
        "context": "weekday 09:00, owner sits in office, computer turned on, sunny outside, south window glare",
        "sim": lambda h: (
            setattr(h.office, "motion", True),
            setattr(h.office, "temp", 24.5),
            setattr(h, "cloudy", False),
        ),
    },
    {
        "time": "12:30", "label": "Lunch break — cooking",
        "context": "12:30 lunch time, motion in kitchen, stove on, kitchen temperature rising to 29C",
        "sim": lambda h: (
            setattr(h.kitchen, "motion", True),
            setattr(h.kitchen, "temp", 29.0),
        ),
    },
    {
        "time": "14:00", "label": "Hot afternoon",
        "context": "14:00 summer afternoon, office temperature 27C, outside 32C, sunny, owner working",
        "sim": lambda h: (
            setattr(h.office, "temp", 27.0),
            setattr(h, "outside_temp", 32.0),
        ),
    },
    {
        "time": "18:30", "label": "Work done, relax",
        "context": "18:30 evening, owner finished work, moved to living room, wants to relax",
        "sim": lambda h: (
            setattr(h.office, "motion", False),
            setattr(h.living_room, "motion", True),
        ),
    },
    {
        "time": "19:00", "label": "Guests arrive",
        "context": "19:00 evening, multiple people detected in hallway, doorbell rang, guests arriving",
        "sim": lambda h: (
            setattr(h.hallway, "motion", True),
            setattr(h.living_room, "temp", 25.5),
        ),
    },
    {
        "time": "20:30", "label": "Movie time",
        "context": "20:30 evening, guests in living room, owner says 'cinema mode', TV turned on",
        "sim": lambda h: (
            setattr(h, "tv_on", True),
            setattr(h.living_room, "motion", True),
        ),
    },
    {
        "time": "23:00", "label": "Bedtime",
        "context": "23:00 night, guests left, owner going to bed, all rooms should prepare for sleep",
        "sim": lambda h: (
            setattr(h, "tv_on", False),
            setattr(h.living_room, "motion", False),
            setattr(h.bedroom, "motion", True),
        ),
    },
    {
        "time": "02:15", "label": "Night bathroom visit",
        "context": "02:15 night, motion detected in hallway, everyone sleeping, need dim light",
        "sim": lambda h: (
            setattr(h.hallway, "motion", True),
            setattr(h.bedroom, "motion", False),
        ),
    },
]

# What the system decides to do (simulated actions)
ACTION_ICONS = {
    "light": "[yellow]light[/]",
    "ac": "[cyan]ac[/]",
    "curtains": "[blue]curtains[/]",
    "kettle": "[red]kettle[/]",
    "coffee": "[red]coffee[/]",
    "music": "[magenta]music[/]",
    "ventilation": "[white]vent[/]",
    "floor_heating": "[yellow]floor[/]",
    "tv": "[blue]tv[/]",
    "vacuum": "[green]vacuum[/]",
    "lock": "[red]lock[/]",
    "window": "[cyan]window[/]",
    "thermostat": "[yellow]therm[/]",
}


def extract_actions(text: str) -> list[str]:
    """Pull action keywords from a preference text for display."""
    keywords = {
        "чайник": "kettle", "кофемашин": "coffee", "свет": "light",
        "яркость": "light", "кондиционер": "ac", "термостат": "thermostat",
        "штор": "curtains", "жалюзи": "curtains", "вытяжк": "ventilation",
        "тёплый пол": "floor_heating", "теплый пол": "floor_heating",
        "телевизор": "tv", "ambilight": "tv", "колонк": "music",
        "плейлист": "music", "музык": "music", "пылесос": "vacuum",
        "дверь": "lock", "заблокир": "lock", "окно": "window",
        "микропроветр": "window", "увлажнитель": "ac",
        "вентилятор": "ventilation",
    }
    found = []
    lower = text.lower()
    seen = set()
    for kw, action in keywords.items():
        if kw in lower and action not in seen:
            seen.add(action)
            found.append(action)
    return found


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def run():
    demo_dir = Path(__file__).parent / "data"
    demo_dir.mkdir(exist_ok=True)

    prefs_path = str(demo_dir / "home_prefs.db")
    log_path = str(demo_dir / "home_log.db")

    # Clean
    for p in (prefs_path, log_path):
        for suffix in ("", "-wal", "-shm"):
            f = Path(p + suffix)
            if f.exists():
                f.unlink()

    embedder = QwenEmbedder()
    home = HomeState()

    async with (
        AsyncSQFox(prefs_path, max_cpu_workers=1) as prefs_db,
        AsyncSQFox(log_path) as log_db,
    ):
        # --- Load preferences ---
        console.print()
        console.rule("[bold magenta]Loading Owner Preferences[/]")
        console.print()

        t0 = time.time()
        for pref in OWNER_PREFERENCES:
            await prefs_db.ingest(
                pref["text"], embed_fn=embedder, metadata=pref["meta"],
            )
        await prefs_db.ensure_schema(SchemaState.SEARCHABLE)
        await asyncio.sleep(0.2)
        load_time = time.time() - t0

        pref_table = Table(box=box.ROUNDED, title_style="bold green")
        pref_table.add_column("Metric", style="bold")
        pref_table.add_column("Value", justify="right")
        pref_table.add_row("Preferences loaded", str(len(OWNER_PREFERENCES)))
        pref_table.add_row("Time", f"{load_time:.1f}s")
        pref_table.add_row("Categories", "morning, work, cooking, evening, night, guests, away, scenes")
        console.print(pref_table)

        # --- Create log table ---
        await log_db.write("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                sim_time TEXT NOT NULL,
                context TEXT NOT NULL,
                preference TEXT,
                actions TEXT,
                search_ms REAL
            )
        """, wait=True)

        # --- Simulate day ---
        console.print()
        console.rule("[bold yellow]Simulating One Day[/]")
        console.print()

        all_decisions = []

        for event in DAY_TIMELINE:
            sim_time = event["time"]
            label = event["label"]
            context = event["context"]

            # Apply simulation state changes
            event["sim"](home)

            # Build context query
            t0 = time.time()
            results = await prefs_db.search(
                context, embed_fn=embedder, limit=3,
            )
            search_ms = (time.time() - t0) * 1000

            top_pref = results[0] if results else None
            pref_text = top_pref.text if top_pref else "—"
            score = top_pref.score if top_pref else 0
            actions = extract_actions(pref_text) if top_pref else []

            decision = {
                "time": sim_time,
                "label": label,
                "context": context,
                "preference": pref_text,
                "score": score,
                "actions": actions,
                "search_ms": search_ms,
                "meta": top_pref.metadata if top_pref else {},
            }
            all_decisions.append(decision)

            # Log to DB
            await log_db.write(
                "INSERT INTO decisions (sim_time, context, preference, actions, search_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                (sim_time, context, pref_text, json.dumps(actions), search_ms),
            )

            # Rich output
            score_color = "bold green" if score >= 0.65 else "yellow" if score >= 0.5 else "dim"

            # Actions line
            action_str = ""
            if actions:
                action_tags = "  ".join(
                    ACTION_ICONS.get(a, f"[white]{a}[/]") for a in actions
                )
                action_str = f"  Actions: {action_tags}"

            parts = [
                f"[dim]Context:[/] {context}",
                "",
                f"[{score_color}]{score:.3f}[/]  {pref_text}",
            ]
            if action_str:
                parts.append("")
                parts.append(action_str)

            if len(results) > 1:
                alt = results[1]
                parts.append(f"\n[dim]Alt: {alt.score:.3f}  {alt.text}[/]")

            console.print(Panel(
                "\n".join(parts),
                title=f"[bold white]{sim_time}[/]  [bold]{label}[/]  [dim]({search_ms:.0f}ms)[/]",
                border_style="magenta",
                box=box.ROUNDED,
                width=min(console.width, 100),
            ))
            await asyncio.sleep(0.1)

        # --- Summary ---
        console.print()
        console.rule("[bold green]Day Summary[/]")
        console.print()

        summary_table = Table(
            title="Decisions Log", box=box.ROUNDED, title_style="bold",
        )
        summary_table.add_column("Time", style="bold white", width=6)
        summary_table.add_column("Event", width=26)
        summary_table.add_column("Score", justify="right", width=6)
        summary_table.add_column("Actions", width=30)
        summary_table.add_column("ms", justify="right", width=5)

        for d in all_decisions:
            action_str = ", ".join(d["actions"]) if d["actions"] else "—"
            sc = d["score"]
            score_color = "bold green" if sc >= 0.65 else "yellow" if sc >= 0.5 else "dim"
            summary_table.add_row(
                d["time"],
                d["label"],
                f"[{score_color}]{sc:.2f}[/]",
                action_str,
                f"{d['search_ms']:.0f}",
            )

        console.print(summary_table)

        # Stats
        console.print()
        row = await log_db.fetch_one("SELECT COUNT(*) FROM decisions")
        total = row[0] if row else 0
        avg_ms = sum(d["search_ms"] for d in all_decisions) / len(all_decisions)

        all_actions = set()
        for d in all_decisions:
            all_actions.update(d["actions"])

        stats_table = Table(box=box.ROUNDED, title_style="bold green")
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")
        stats_table.add_row("Events processed", str(total))
        stats_table.add_row("Unique device types", str(len(all_actions)))
        stats_table.add_row("Devices triggered", ", ".join(sorted(all_actions)))
        stats_table.add_row("Avg search latency", f"[bold cyan]{avg_ms:.0f}ms[/]")
        stats_table.add_row("Preferences DB", Path(prefs_path).name)
        stats_table.add_row("Decisions log DB", Path(log_path).name)
        console.print(stats_table)

        # Architecture panel
        console.print()
        console.print(Panel(
            Align.center(
                "[bold]How it works[/]\n\n"
                "[dim]1. Context changes (time, sensors, motion)[/]\n"
                "[dim]2. System builds a natural-language query from context[/]\n"
                "[dim]3. AsyncSQFox searches owner preferences (FTS + vectors)[/]\n"
                "[dim]4. Top match -> extract actions -> execute[/]\n\n"
                "[bold yellow]No hardcoded rules.[/] Add a preference in plain text\n"
                "and the house adapts. Works in Russian, English, or mixed."
            ),
            border_style="magenta",
            box=box.ROUNDED,
        ))


def main():
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]sqfox demo: Smart Home[/]\n"
            "[dim]Preference-aware automation with auto-RAG[/]\n"
            "[dim]Qwen3-Embedding-0.6B, 256 dim (MRL)[/]"
        ),
        border_style="bold magenta",
        box=box.DOUBLE,
    ))

    console.print()
    get_model()
    asyncio.run(run())

    console.print()
    console.print(Panel(
        Align.center("[bold green]Demo complete![/]"),
        border_style="green",
        box=box.DOUBLE,
    ))


if __name__ == "__main__":
    main()
