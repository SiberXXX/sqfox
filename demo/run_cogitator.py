#!/usr/bin/env python3
"""
Слабый Когитатор: Ferrum Vigilans — Симулятор Орбитальной Обороны.

Демо SQFox в виде мини-игры по Warhammer 40K.

Ты — Техножрец Адептус Механикус на орбитальной оборонительной платформе
GAMMA-7 («Ferrum Vigilans»).  Корабли орков на подлёте.  Твоё единственное
оружие — тактическая база знаний Когитатора, работающая на SQFox.

Фичи SQFox в игре:
  - Гибридный поиск  (FTS5 + HNSW векторы)
  - Инжест новых данных  (STC-фрагменты между волнами)
  - Crash recovery  (психо-атака Вирдбоя)
  - SqliteHnswBackend  (чистый Python, ноль C-зависимостей)
  - Температура Когитатора  (HNSW O(log N))
  - Диагностика

Запуск:
    pip install rich sentence-transformers
    python demo/run_cogitator.py
"""

import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.align import Align
    from rich import box
except ImportError:
    print("pip install rich")
    sys.exit(1)

from sqfox import SQFox, SchemaState
from sqfox.backends.hnsw import SqliteHnswBackend

console = Console()


def cls():
    """Надёжная очистка экрана (Windows: cls, Unix: clear)."""
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")

# =====================================================================
#  ЭМБЕДДЕР
# =====================================================================

_model = None


def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            console.print(
                "[bold red]sentence-transformers не найдена.[/]\n"
                "  pip install sentence-transformers\n"
            )
            sys.exit(1)
        with console.status(
            "[bold cyan]Пробуждаю Дух Машины "
            "(Qwen3-Embedding-0.6B)...[/]",
            spinner="dots",
        ):
            t0 = time.time()
            _model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B", truncate_dim=256,
            )
            elapsed = time.time() - t0
        console.print(
            f"  [green]Дух Машины пробуждён за {elapsed:.1f}с[/]"
            "  (256 dim, Matryoshka)"
        )
    return _model


class QwenEmbedder:
    def __init__(self):
        self.model = get_model()

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text, prompt_name="query").tolist()


# =====================================================================
#  ТАКТИЧЕСКАЯ БАЗА ЗНАНИЙ  (инжестится в Когитатор)
# =====================================================================

TACTICAL_DATABASE = [
    # --- Профили кораблей ---
    {
        "text": (
            "Kill Kroozer (Убийца-Крейсер): тяжёлый орочий военный корабль с "
            "грубой но чрезвычайно толстой лобовой бронёй.  Вооружён тяжёлыми "
            "орудиями и носовым тараном.  УЯЗВИМОСТЬ: энергетические лучи "
            "лэнс-батарей обходят грубую броню, возбуждая молекулярные связи "
            "на резонансных частотах.  Снаряды макропушек отскакивают от "
            "толстых бронеплит практически без эффекта.  Рекомендация: "
            "продолжительный огонь лэнс-батарей на средней дистанции."
        ),
        "meta": {"type": "профиль", "ship": "Kill Kroozer"},
    },
    {
        "text": (
            "Brute Ram Ship (Брут-Таран): скоростной штурмовой корабль для "
            "таранных атак.  Скорость в 1.5 раза выше стандартных орочьих "
            "кораблей.  Усиленный нос из захваченных имперских бронеплит.  "
            "УЯЗВИМОСТЬ: торпедные залпы перехватывают и уничтожают его до "
            "дистанции тарана.  Если Таран достигнет ближней дистанции — "
            "катастрофические повреждения практически гарантированы.  Огонь "
            "макропушек неэффективен из-за малого профиля и высокой скорости.  "
            "Приоритет: уничтожить на максимальной дистанции торпедами."
        ),
        "meta": {"type": "профиль", "ship": "Brute Ram Ship"},
    },
    {
        "text": (
            "Onslaught (Натиск): лёгкий быстроходный корабль для стайных "
            "атак.  Минимальная броня, высокая скорость, лёгкое вооружение.  "
            "По отдельности слабые, опасны в стаях.  УЯЗВИМОСТЬ: залп "
            "макропушек разносит лёгкую обшивку.  Каждый макро-снаряд "
            "разлетается на сотни осколков при ударе, что смертельно для "
            "небронированных кораблей.  Лэнс-батареи — пустая трата на "
            "такие хрупкие цели.  Используйте макропушки."
        ),
        "meta": {"type": "профиль", "ship": "Onslaught"},
    },
    {
        "text": (
            "Terror Ship (Террор-Корабль): орочий авианосец, выпускающий "
            "волны истребителей Fighta-Bommerz.  Чем дольше живёт, тем больше "
            "истребителей выпускает — урон нарастает каждый ход.  "
            "УЯЗВИМОСТЬ: нова-пушка уничтожает авианосец вместе со всем "
            "истребительным облаком одним выстрелом.  Лэнс-батарея — "
            "второй вариант, пробивает до пусковых отсеков.  "
            "Приоритет: УНИЧТОЖИТЬ НЕМЕДЛЕННО, пока истребители не накопились."
        ),
        "meta": {"type": "профиль", "ship": "Terror Ship"},
    },
    {
        "text": (
            "Space Hulk (Космический Скиталец): гигантское сплавление "
            "заброшенных кораблей и обломков.  Экстремальная прочность "
            "корпуса, тяжёлое трофейное вооружение.  Поглощает огромный урон.  "
            "УЯЗВИМОСТЬ: только продолжительный огонь лэнс-батарей на "
            "ближней дистанции пробивает многослойный корпус.  Торпеды "
            "детонируют на внешней оболочке обломков с минимальным уроном "
            "внутренним системам.  Бой на несколько ходов.  Весь лэнс — сюда."
        ),
        "meta": {"type": "профиль", "ship": "Space Hulk"},
    },

    # --- Оружейные мануалы ---
    {
        "text": (
            "Макропушка (Macro-cannon): стандартное кинетическое оружие, "
            "стреляет тяжёлыми осколочными снарядами.  Боезапас неограничен "
            "(автозарядка).  Эффективна против лёгкой и средней брони.  "
            "Осколки при ударе создают дождь шрапнели, идеальный для срыва "
            "лёгкой обшивки.  Против тяжёлой брони бесполезна — снаряды "
            "не пробивают толстые орочьи плиты."
        ),
        "meta": {"type": "оружие", "weapon": "macro"},
    },
    {
        "text": (
            "Лэнс-батарея (Lance Battery): сфокусированное энергетическое "
            "оружие, стреляющее когерентными лучами.  Обходит физическую "
            "броню, возбуждая молекулярные связи.  Крайне эффективна против "
            "тяжелобронированных целей.  Ограниченный запас энергии требует "
            "аккуратного выбора целей.  Оружие выбора против Kill Крейсеров "
            "и Космических Скитальцев.  ВНИМАНИЕ: каждый выстрел расходует "
            "энергию реактора.  При истощении реактора — перегрузка: "
            "повреждение корпуса и временная недоступность лэнса."
        ),
        "meta": {"type": "оружие", "weapon": "lance"},
    },
    {
        "text": (
            "Торпеда (Torpedo): управляемый снаряд с самонаведением.  "
            "Лучше всего работает по приближающимся целям на большой "
            "дистанции.  Высокий урон при прямом попадании.  Ограниченный "
            "боезапас — каждый залп расходует невосполнимые боеголовки.  "
            "Идеальна для перехвата Брут-Таранов до дистанции тарана."
        ),
        "meta": {"type": "оружие", "weapon": "torpedo"},
    },
    {
        "text": (
            "Нова-пушка (Nova Cannon): опустошительное оружие с зоной "
            "поражения.  Стреляет зарядом имплозии, детонирующим в сфере "
            "уничтожения.  Всего 2 заряда на борту.  Предназначена для "
            "авианосцев и плотных построений.  Один выстрел может "
            "ликвидировать Террор-Корабль со всеми его истребителями."
        ),
        "meta": {"type": "оружие", "weapon": "nova"},
    },

    # --- Тактические доктрины ---
    {
        "text": (
            "ДОКТРИНА: Бой с Таранами.  При встрече с Брут-Таранами — "
            "атаковать на максимальной дистанции торпедными залпами.  "
            "Самонаведение эффективно перехватывает скоростные цели.  "
            "Если Таран прорвался сквозь торпеды — пустотные щиты последняя "
            "линия обороны.  Каждый щит поглощает один удар.  "
            "Без щитов — повреждения корпуса катастрофические.  Никогда "
            "не стрелять по Таранам из макропушек — скорость и малый "
            "профиль делают кинетическое наведение бесполезным."
        ),
        "meta": {"type": "доктрина", "topic": "анти-таран"},
    },
    {
        "text": (
            "ДОКТРИНА: Приоритет целей в смешанных построениях.  "
            "Порядок уничтожения: 1) Брут-Тараны (немедленная угроза "
            "столкновения), 2) Террор-Корабли (нарастающий урон "
            "истребителей), 3) Kill Крейсеры (тяжёлый огонь), "
            "4) Натиски (низкая индивидуальная угроза).  "
            "Если нова-пушка заряжена и Террор-Корабль на поле — "
            "приоритет Террору для предотвращения накопления истребителей."
        ),
        "meta": {"type": "доктрина", "topic": "приоритеты"},
    },

    # --- Боевые отчёты ---
    {
        "text": (
            "БОЕВОЙ ОТЧЁТ: Станция K-77.  Единственная лэнс-батарея "
            "уничтожила два Kill Крейсера подряд.  Энерголучи обошли "
            "2 метра грубой орочьей брони без проблем.  47 снарядов "
            "макропушки попали в третий Kill Крейсер — ноль пробитий.  "
            "Вывод: кинетическое оружие бесполезно против брони Kill "
            "Крейсера.  Только лэнс."
        ),
        "meta": {"type": "отчёт", "location": "K-77"},
    },
    {
        "text": (
            "БОЕВОЙ ОТЧЁТ: Оборона Алтаря-9.  Террор-Корабль прожил "
            "3 хода и выпустил 47 Fighta-Bommerz до уничтожения "
            "нова-пушкой.  Общий урон от истребителей: 340 единиц "
            "корпуса.  Вывод: каждый ход жизни Террор-Корабля = "
            "больше истребителей.  Уничтожать авианосцы немедленно."
        ),
        "meta": {"type": "отчёт", "location": "Алтарь-9"},
    },
    {
        "text": (
            "БОЕВОЙ ОТЧЁТ: Торпедный бой у Дамокла.  Три Брут-Тарана "
            "перехвачены на 40 000 км торпедными залпами — все уничтожены "
            "до дистанции тарана.  Четвёртый Таран увернулся от торпед и "
            "ударил в пустотный щит #2, мгновенно обрушив его.  Без "
            "щитов следующий таран был бы смертелен.  Вывод: торпеды "
            "обязательны против Таранов.  Всегда держать запас."
        ),
        "meta": {"type": "отчёт", "location": "Дамокл"},
    },

    # --- Дополнительные данные ---
    {
        "text": (
            "ТЕХЗАМЕТКА: Идентификация орочьих кораблей по сигнатуре.  "
            "Kill Крейсеры — низкочастотный гул двигателей на ауспексе "
            "за 80 000 км.  Брут-Тараны ускоряются по прямой без "
            "маневров уклонения.  Террор-Корабли — рассеянная тепловая "
            "сигнатура от множества пусковых ангаров.  Космический "
            "Скиталец — масса искажает гравитационные показания."
        ),
        "meta": {"type": "техзаметка", "topic": "идентификация"},
    },
    {
        "text": (
            "ПУСТОТНЫЕ ЩИТЫ: каждый генератор поглощает один удар "
            "любой мощности перед коллапсом.  Регенерация — один боевой "
            "цикл при отсутствии повреждений генератора.  Против "
            "Брут-Таранов щиты — последняя линия обороны."
        ),
        "meta": {"type": "оружие", "weapon": "щиты"},
    },
    {
        "text": (
            "РАЗВЕДКА: Орочьи Вирдбои — псайкеры, канализирующие "
            "коллективную психическую энергию WAAAGH!  Их атаки "
            "нарушают электронные системы и повреждают хранилища данных.  "
            "Нейропути когитатора особенно уязвимы.  При повреждении "
            "тактической базы все данные восстанавливаются из "
            "резервных эмбеддингов в SQLite."
        ),
        "meta": {"type": "разведка", "topic": "вирдбой"},
    },
]

# STC-фрагменты и перехваты (инжестятся между волнами)
STC_FRAGMENTS = [
    # После волны 2
    [
        {
            "text": (
                "STC-ФРАГМЕНТ: Древний шаблон производства усиленных "
                "эмиттеров пустотных щитов.  Регенерация щита возможна "
                "через перенаправление вспомогательного питания.  "
                "При полной зачистке волны — один щит восстанавливается."
            ),
            "meta": {"type": "stc", "origin": "Кузнечный мир"},
        },
        {
            "text": (
                "ПЕРЕХВАТ ОРК-СВЯЗИ: 'Паканы говорят шо большая "
                "палка-стрелялка прям скрозь броню бьёт!  Надо БЫСТРЕЙ "
                "таранить покамест не стрельнули!'  Анализ: орки знают "
                "об уязвимости к лэнс-батареям и пытаются контрить тараном."
            ),
            "meta": {"type": "перехват", "origin": "Орк-частота"},
        },
    ],
    # После волны 4
    [
        {
            "text": (
                "STC-ФРАГМЕНТ: Шаблон конверсии торпедных боеголовок "
                "в плазменные.  Конвертированные торпеды наносят на 30% "
                "больше урона по всем целям.  Ресурсы не требуются."
            ),
            "meta": {"type": "stc", "origin": "Архив Марса"},
        },
        {
            "text": (
                "БОЕВОЙ ОТЧЁТ: Орбита Армагеддона.  Смешанный орочий "
                "флот.  Ключевой вывод: Космические Скитальцы полностью "
                "игнорируют торпедный урон.  Их оболочка из обломков "
                "поглощает взрывы.  Только продолжительный огонь "
                "лэнс-батарей пробивает слои.  Скиталец требует 3+ "
                "попаданий лэнса для уничтожения."
            ),
            "meta": {"type": "отчёт", "location": "Армагеддон"},
        },
    ],
]

# Перехваченные орочьи переговоры (случайные, появляются между волнами)
ORK_INTERCEPTS = [
    (
        "ПЕРЕХВАТ: 'Зогнот, чё эта за блестяшка мигает?  "
        "— Не знаю, ткни пальцем.  — БАБАХ!  "
        "— Зогнот?  ЗОГНОТ?!'"
    ),
    (
        "ПЕРЕХВАТ: 'Капитан!  Они стреляют светом!  "
        "— Дык заслони бронёй!  "
        "— Свет СКРОЗЬ броню прошёл!  "
        "— ...а тогда заслони ДВУМЯ бронями!'"
    ),
    (
        "ПЕРЕХВАТ: 'Паканы!  У хумансов пушка шо кидает "
        "маленькое солнце!  Прям солнце!  И всё — БАБАХ!  "
        "— Красота!  Хочу такую!  "
        "— Нету больше тех кто рядом стоял, спросить не у кого.'"
    ),
    (
        "ПЕРЕХВАТ: 'Мекбой!  Чини движок!  "
        "— Щас, токо кувалду найду.  "
        "— КАКУЮ КУВАЛДУ?!  ЭТО ПЛАЗМЕННЫЙ РЕАКТОР!  "
        "— А чё, кувалда всегда помогала...'"
    ),
    (
        "ПЕРЕХВАТ: 'Почему корабль крутится?!  "
        "— Штурвал заклинило.  "
        "— Так выклини!  "
        "— Там грот сидит.  На штурвале.  Спит.  "
        "— ...ладно, крутимся.'"
    ),
    (
        "ПЕРЕХВАТ: 'Гляди, хумансы чё-то пишут на своих "
        "мигалках.  Может это послание?  "
        "— Да не, это они шаманят.  У них железный шаман "
        "в коробке живёт.  "
        "— А зачем?  "
        "— А хрен его знает, хумансы ж дурные.'"
    ),
    (
        "ПЕРЕХВАТ: 'WAAAGH!  Вперёд на станцию!  "
        "— А вдруг там опять эта пушка-которая-как-солнце?  "
        "— Тогда побежим назад.  "
        "— Назад нельзя, Варбосс сзади.  "
        "— ...WAAAGH вперёд, получается.'"
    ),
    (
        "ПЕРЕХВАТ: 'Я придумал!  Покрасим корабль в красный!  "
        "— Зачем?  "
        "— КРАСНЫЕ ЛЕТЯТ БЫСТРЕЙ!  "
        "— У нас нет красной краски.  "
        "— А скотчем замотаем?  "
        "— ...красным скотчем?  "
        "— НЕТ КРАСНОГО СКОТЧА!  "
        "— Тогда просто орём громче, может поможет.'"
    ),
    (
        "ПЕРЕХВАТ: 'Командир, шо делать, ихний луч "
        "прожёг дырку в машинном отделении!  "
        "— Заткни пальцем!  "
        "— Там ВАКУУМ!  "
        "— Ну тогда ДВУМЯ пальцами!'"
    ),
    (
        "ПЕРЕХВАТ: 'Слышь, а чё хумансы молятся перед "
        "стрельбой?  — Может пушка на молитвах работает?  "
        "— Как это?  — Ну вот мы орём WAAAGH и "
        "стреляем точнее.  У них наверно то же самое, "
        "только тихо и занудно.'"
    ),
    (
        "ПЕРЕХВАТ: 'Варбосс!  Ихняя торпеда летит!  "
        "— Стреляйте по торпеде!  "
        "— Мы по ней попали!  Она не взорвалась!  "
        "— Стреляйте ЕЩЁ!  "
        "— Она РАЗВЕРНУЛАСЬ И ЛЕТИТ ОБРАТНО!  "
        "— ...бежим?  "
        "— БЕЖИМ!'"
    ),
    (
        "ПЕРЕХВАТ: 'Доклад Варбоссу: на станции хумансов "
        "замечен Железный Бог в коробке.  Он ДУМАЕТ.  "
        "И когда думает — хумансы стреляют точнее.  "
        "Предложение: кинем камень в коробку.  "
        "— ПСИТКИЙ кинем!  Вирдбой, давай!'"
    ),
]


# =====================================================================
#  ИГРОВЫЕ ДАННЫЕ
# =====================================================================

@dataclass
class ShipType:
    name: str           # короткое имя для дисплея
    hp: int
    damage: int
    weakness: str       # ключ оружия для 2x урона
    resistance: str     # ключ оружия для 0.5x урона
    points: int
    art: str            # ASCII-арт
    spawn_damage: int = 0   # +урон/ход если жив (Террор-истребители)


SHIP_TYPES: dict[str, ShipType] = {
    "Onslaught": ShipType(
        "Натиск", hp=40, damage=10,
        weakness="macro", resistance="lance", points=50,
        art=r" /=>",
    ),
    "Brute Ram": ShipType(
        "Брут-Таран", hp=60, damage=40,
        weakness="torpedo", resistance="macro", points=150,
        art=r"<=###>",
    ),
    "Kill Kroozer": ShipType(
        "Kill Крейсер", hp=100, damage=22,
        weakness="lance", resistance="macro", points=200,
        art=r"<===[####]===>",
    ),
    "Terror Ship": ShipType(
        "Террор-Корабль", hp=70, damage=15,
        weakness="nova", resistance="torpedo", points=250,
        spawn_damage=12,
        art=r"<=={<<O>>}==>",
    ),
    "Space Hulk": ShipType(
        "Скиталец", hp=250, damage=35,
        weakness="lance", resistance="torpedo", points=500,
        art=r"<##=[XXXXX]=##>",
    ),
}

# per_turn: макс. выстрелов этим оружием за ход
# energy_cost: расход энергии реактора за выстрел (только для лэнса)
WEAPONS = {
    "macro":   {"name": "Макропушка",   "dmg": 25,  "stock": 999, "clr": "yellow",  "per_turn": 3, "energy_cost": 0},
    "lance":   {"name": "Лэнс",        "dmg": 40,  "stock": 999, "clr": "cyan",     "per_turn": 99, "energy_cost": 1},
    "torpedo": {"name": "Торпеда",      "dmg": 50,  "stock": 6,   "clr": "green",    "per_turn": 2, "energy_cost": 0},
    "nova":    {"name": "Нова-пушка",   "dmg": 120, "stock": 2,   "clr": "magenta",  "per_turn": 1, "energy_cost": 0},
}

WAVES = [
    [("Onslaught", 3)],
    [("Brute Ram", 1), ("Onslaught", 2)],
    [("Kill Kroozer", 1), ("Onslaught", 3)],
    [("Terror Ship", 1), ("Brute Ram", 2)],
    [("Kill Kroozer", 2), ("Terror Ship", 1), ("Onslaught", 2)],
    [("Space Hulk", 1), ("Kill Kroozer", 1), ("Brute Ram", 1), ("Onslaught", 1)],
]


# =====================================================================
#  СОСТОЯНИЕ ИГРЫ
# =====================================================================

@dataclass
class Ship:
    id: int
    type_key: str
    hp: int
    max_hp: int
    turns_alive: int = 0

    @property
    def info(self) -> ShipType:
        return SHIP_TYPES[self.type_key]


@dataclass
class State:
    hull: int = 100
    max_hull: int = 100
    shields: int = 3
    max_shields: int = 3
    score: int = 0
    wave: int = 0
    temp: float = 20.0
    weapon_stock: dict = field(default_factory=dict)
    ships: list = field(default_factory=list)
    kills: int = 0
    weakness_hits: int = 0
    max_queries: int = 2
    torpedo_up: bool = False
    log: list = field(default_factory=list)
    # Энергия реактора для лэнс-батареи
    reactor_energy: int = 8
    max_reactor: int = 8
    reactor_overloaded: bool = False  # True = лэнс недоступен 1 ход
    # Ярость Машинного Духа (одноразовая)
    rage_available: bool = True
    # Набор ship.id, по которым был запрос в когитатор
    queried_ships: set = field(default_factory=set)
    # Слепой огонь — множитель урона без запроса
    blind_fire_mult: float = 0.5
    # Статистика запросов
    total_queries: int = 0


# =====================================================================
#  РЕНДЕР  (один экран — без перемотки)
# =====================================================================

def _bar(cur, mx, w=20, full="green", mid="yellow", low="red"):
    r = max(0, min(1, cur / mx)) if mx > 0 else 0
    f = int(r * w)
    c = full if r > 0.6 else mid if r > 0.3 else low
    return f"[{c}]{'|' * f}[/][dim]{'.' * (w - f)}[/]"


def _temp_bar(t):
    r = min(1.0, t / 100.0)
    f = int(r * 12)
    c = "cyan" if t < 40 else "yellow" if t < 65 else "red" if t < 85 else "bold red"
    return f"[{c}]{'|' * f}[/][dim]{'.' * (12 - f)}[/] [{c}]{t:.0f}C[/]"


def _reactor_bar(energy, mx):
    r = max(0, min(1, energy / mx)) if mx > 0 else 0
    f = int(r * 10)
    c = "bold cyan" if r > 0.5 else "yellow" if r > 0.25 else "bold red"
    return f"[{c}]{'#' * f}[/][dim]{'.' * (10 - f)}[/] [{c}]{energy}/{mx}[/]"


def render_screen(st: State, phase: str = ""):
    """Рендерит интерфейс когитатора — 4 панели, 2 столбца, внешняя рамка."""
    cls()
    wide = console.width >= 100

    # ── 1. СТАНЦИЯ ─────────────────────────────────
    hull_bar = _bar(st.hull, st.max_hull, w=15)
    sh_icons = (
        "[bold cyan]@[/] " * st.shields
        + "[dim]_[/] " * (st.max_shields - st.shields)
    )
    temp = _temp_bar(st.temp)
    reactor = _reactor_bar(st.reactor_energy, st.max_reactor)
    reactor_status = ""
    if st.reactor_overloaded:
        reactor_status = " [bold red]ПЕРЕГРУЗКА![/]"

    rage_str = "[bold magenta]ГОТОВА[/]" if st.rage_available else "[dim]израсходована[/]"

    station = Panel(
        f"КОРПУС  {hull_bar}  {st.hull}/{st.max_hull}\n"
        f"ЩИТЫ   {sh_icons}\n"
        f"РЕАКТ  {reactor}{reactor_status}\n"
        f"ТЕМП   {temp}\n"
        f"ВОЛНА [bold]{st.wave}[/]/{len(WAVES)}  "
        f"УБИТО [bold]{st.kills}[/]  "
        f"СЧЁТ [bold white]{st.score}[/]\n"
        f"ЯРОСТЬ ДУХА: {rage_str}",
        title="[bold white]СТАНЦИЯ[/]",
        border_style="bold cyan",
        expand=True,
    )

    # ── 2. АУСПЕКС ────────────────────────────────
    if st.ships:
        asp = Table(
            show_header=True, box=box.SIMPLE_HEAVY,
            expand=True, padding=(0, 1),
        )
        asp.add_column("#", style="bold", width=3, justify="center")
        asp.add_column("", min_width=8)
        asp.add_column("Корабль", style="bold", min_width=10)
        asp.add_column("Прочность", min_width=16)
        asp.add_column("Урон", justify="right", width=6)
        asp.add_column("Инфо", width=5, justify="center")
        for s in st.ships:
            info = s.info
            hp_bar = _bar(s.hp, s.max_hp, w=8)
            dmg_str = str(info.damage)
            if info.spawn_damage > 0 and s.turns_alive > 0:
                dmg_str += f"[red]+{info.spawn_damage * s.turns_alive}[/]"
            queried = "[green]OK[/]" if s.id in st.queried_ships else "[dim]?[/]"
            asp.add_row(
                str(s.id), f"[red]{info.art}[/]",
                info.name, f"{hp_bar} {s.hp}/{s.max_hp}", dmg_str,
                queried,
            )
        auspex = Panel(
            asp,
            title="[bold yellow]АУСПЕКС — ВХОДЯЩИЕ[/]",
            border_style="yellow",
            expand=True,
        )
    else:
        auspex = Panel(
            "[bold green]Пустота чиста.[/]",
            title="[bold yellow]АУСПЕКС[/]",
            border_style="yellow",
            expand=True,
        )

    # ── 3. АРСЕНАЛ ─────────────────────────────────
    arm_lines = []
    for key, w in WEAPONS.items():
        stock = st.weapon_stock.get(key, 0)
        if key == "macro":
            s_str = "INF"
        elif key == "lance":
            e = st.reactor_energy
            s_str = f"E:{e}" if e > 0 else "[red]E:0[/]"
            if st.reactor_overloaded:
                s_str = "[red]ПЕРЕГРУЗКА[/]"
        elif stock <= 0:
            s_str = "[dim]---[/]"
        else:
            s_str = str(stock)

        dmg = w["dmg"]
        if st.torpedo_up and key == "torpedo":
            dmg = int(dmg * 1.3)

        pt = w["per_turn"]
        pt_str = f"/{pt}ход" if pt < 99 else ""

        arm_lines.append(
            f"[{w['clr']}][{key:>7}][/]  {w['name']:<12} "
            f"DMG:[bold]{dmg}[/]  x{s_str}{pt_str}"
        )
    arm_lines.append("")
    arm_lines.append(
        "[dim]  rage = Ярость Машинного Духа[/]" if st.rage_available
        else "[dim]  Ярость Духа израсходована[/]"
    )
    armory = Panel(
        "\n".join(arm_lines),
        title="[bold white]АРСЕНАЛ[/]",
        border_style="dim",
        expand=True,
    )

    # ── 4. СТАТУС ──────────────────────────────────
    log_parts: list[str] = []
    if phase:
        log_parts.append(f"[bold]{phase}[/]")
    if st.log:
        max_log = 10 if wide else max(4, console.height - 18)
        log_parts.extend(st.log[-max_log:])
    if not log_parts:
        log_parts.append("[dim]Дух Машины ожидает директив...[/]")
    status = Panel(
        "\n".join(log_parts),
        title="[bold green]СТАТУС[/]",
        border_style="dim green",
        expand=True,
    )

    # ── СБОРКА ─────────────────────────────────────
    if wide:
        top = Table.grid(expand=True)
        top.add_column(ratio=2)
        top.add_column(ratio=3)
        top.add_row(station, auspex)

        bottom = Table.grid(expand=True)
        bottom.add_column(ratio=2)
        bottom.add_column(ratio=3)
        bottom.add_row(armory, status)

        inner = Table.grid(expand=True)
        inner.add_column()
        inner.add_row(top)
        inner.add_row(bottom)
    else:
        inner = Table.grid(expand=True)
        inner.add_column()
        inner.add_row(station)
        inner.add_row(auspex)
        inner.add_row(armory)
        inner.add_row(status)

    # Внешняя рамка когитатора — по границам экрана
    console.print(Panel(
        inner,
        title=(
            "[bold cyan]"
            "+=== КОГИТАТОР «FERRUM VIGILANS» * "
            "MK.VII * ОРБИТАЛЬНАЯ ОБОРОНА ===+"
            "[/]"
        ),
        subtitle=(
            "[dim cyan]"
            "/// Адептус Механикус * "
            "Платформа GAMMA-7 * Сегментум Солар ///"
            "[/]"
        ),
        border_style="bold cyan",
        box=box.DOUBLE,
        width=console.width,
    ))


# =====================================================================
#  ИГРОВОЙ ДВИЖОК
# =====================================================================

class CogitatorGame:
    def __init__(self):
        self.st = State()
        self.db: SQFox | None = None
        self.backend: SqliteHnswBackend | None = None
        self.embedder: QwenEmbedder | None = None
        self.db_path = ""
        self._sid = 0
        self._intercept_idx = 0

    def setup(self):
        demo_dir = Path(__file__).parent / "data"
        demo_dir.mkdir(exist_ok=True)
        self.db_path = str(demo_dir / "cogitator.db")
        for suf in ("", "-wal", "-shm"):
            p = Path(self.db_path + suf)
            try:
                p.exists() and p.unlink()
            except OSError:
                pass

        for k, w in WEAPONS.items():
            self.st.weapon_stock[k] = w["stock"]

        console.print()
        self.embedder = QwenEmbedder()

        self.backend = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)
        self.db = SQFox(self.db_path, vector_backend=self.backend)
        self.db.start()

        console.print("\n[bold cyan]  Загрузка тактических архивов...[/]")
        for i, doc in enumerate(TACTICAL_DATABASE):
            self.db.ingest(
                doc["text"], metadata=doc["meta"],
                embed_fn=self.embedder, wait=True,
            )
            if (i + 1) % 5 == 0 or i == len(TACTICAL_DATABASE) - 1:
                console.print(
                    f"  [dim]{i + 1}/{len(TACTICAL_DATABASE)}[/]",
                    end="\r",
                )
        self.db.ensure_schema(SchemaState.SEARCHABLE)
        time.sleep(0.2)

        n = self.db.fetch_one("SELECT COUNT(*) FROM documents")[0]
        console.print(
            f"  [green]{n} записей загружено.[/]  "
            f"Бэкенд: [cyan]{self.db.vector_backend_name}[/]"
        )
        time.sleep(1)

    # --- Поиск ---

    def _search(self, query: str):
        """Поиск в базе. Греет когитатор. Помечает найденные корабли."""
        self.st.temp += random.uniform(8, 15)
        self.st.total_queries += 1

        t0 = time.time()
        results = self.db.search(query, embed_fn=self.embedder, limit=3)
        ms = (time.time() - t0) * 1000
        self.st.temp += ms * 0.05

        if self.st.temp > 90:
            self.st.log.append(
                "[bold red]ПЕРЕГРЕВ! Священные масла выкипают![/]"
            )

        if not results:
            self.st.log.append("[red]Данных не найдено.[/]")
            return

        self.st.log.append(f"[dim]Обработано за {ms:.0f}мс[/]")
        # Полный текст результата — Rich сам переносит строки в панели
        for i, r in enumerate(results):
            sc = "bold green" if r.score >= 0.7 else "yellow" if r.score >= 0.55 else "dim"
            txt = r.text.replace("\n", " ")
            mark = ">>>" if i == 0 else "   "
            self.st.log.append(f"{mark} [{sc}]{r.score:.3f}[/] {txt}")

            # Помечаем корабли, по которым нашлись данные
            if r.metadata:
                ship_meta = r.metadata.get("ship", "")
                for s in self.st.ships:
                    if s.type_key == ship_meta or s.info.name in txt:
                        self.st.queried_ships.add(s.id)
            # Пометка по упоминанию имени корабля в тексте
            for s in self.st.ships:
                if s.type_key.lower() in txt.lower() or s.info.name.lower() in txt.lower():
                    self.st.queried_ships.add(s.id)

    # --- Бой ---

    def _resolve(self, assignments: dict[int, str]):
        destroyed: set[int] = set()

        for sid, wkey in assignments.items():
            ship = next((s for s in self.st.ships if s.id == sid), None)
            if not ship or ship.hp <= 0:
                continue
            w = WEAPONS[wkey]
            info = ship.info
            dmg = w["dmg"]
            if self.st.torpedo_up and wkey == "torpedo":
                dmg = int(dmg * 1.3)

            # Слепой огонь — если корабль не был запрошен в когитаторе
            blind = sid not in self.st.queried_ships
            blind_tag = ""

            if wkey == info.weakness and not blind:
                dmg *= 2
                note = " [bold green]СЛАБОСТЬ! x2[/]"
                self.st.weakness_hits += 1
            elif wkey == info.resistance:
                dmg //= 2
                note = " [dim](отражено x0.5)[/]"
            else:
                note = ""

            if blind:
                dmg = int(dmg * self.st.blind_fire_mult)
                blind_tag = " [bold red]СЛЕПОЙ ОГОНЬ[/]"

            dmg = int(dmg * random.uniform(0.85, 1.15))

            # Расход энергии реактора (лэнс)
            if w["energy_cost"] > 0:
                self.st.reactor_energy -= w["energy_cost"]

            ship.hp -= dmg
            if ship.hp <= 0:
                ship.hp = 0
                destroyed.add(ship.id)
                self.st.score += info.points
                self.st.kills += 1
                self.st.log.append(
                    f"[{w['clr']}]{w['name']}[/] -> {info.name} [{sid}]: "
                    f"[bold]{dmg}[/]{note}{blind_tag} "
                    f"[bold green]УНИЧТОЖЕН +{info.points}[/]"
                )
            else:
                self.st.log.append(
                    f"[{w['clr']}]{w['name']}[/] -> {info.name} [{sid}]: "
                    f"[bold]{dmg}[/]{note}{blind_tag} HP:{ship.hp}/{ship.max_hp}"
                )
            if wkey not in ("macro", "lance"):
                self.st.weapon_stock[wkey] -= 1

        # Проверка перегрузки реактора после выстрелов
        if self.st.reactor_energy <= 0 and not self.st.reactor_overloaded:
            over_dmg = abs(self.st.reactor_energy) * 5 + 10
            self.st.hull -= over_dmg
            if self.st.hull < 0:
                self.st.hull = 0
            self.st.reactor_overloaded = True
            self.st.reactor_energy = 0
            self.st.log.append(
                f"[bold red]РЕАКТОР ПЕРЕГРУЖЕН! Выброс плазмы: "
                f"-{over_dmg} корпус! Лэнс заблокирован на ход![/]"
            )

        self.st.ships = [s for s in self.st.ships if s.id not in destroyed]

        # Ответный огонь
        for ship in self.st.ships:
            if ship.hp <= 0:
                continue
            ship.turns_alive += 1
            total = ship.info.damage
            if ship.info.spawn_damage > 0:
                total += ship.info.spawn_damage * ship.turns_alive
            if self.st.shields > 0:
                self.st.shields -= 1
                self.st.log.append(
                    f"[bold]{ship.info.name} [{ship.id}][/] стреляет! "
                    f"[cyan]Щит поглотил удар![/] ({self.st.shields} ост.)"
                )
            else:
                self.st.hull -= total
                if self.st.hull < 0:
                    self.st.hull = 0
                self.st.log.append(
                    f"[bold]{ship.info.name} [{ship.id}][/] стреляет! "
                    f"[bold red]{total} В КОРПУС![/] "
                    f"({self.st.hull}/{self.st.max_hull})"
                )

    def _resolve_rage(self):
        """Ярость Машинного Духа — одноразовый удар по всем живым кораблям."""
        self.st.rage_available = False
        self.st.temp += 30
        self.st.log.append("")
        self.st.log.append(
            "[bold magenta]========================================[/]"
        )
        self.st.log.append(
            "[bold magenta]    ЯРОСТЬ МАШИННОГО ДУХА![/]"
        )
        self.st.log.append(
            "[bold magenta]========================================[/]"
        )
        self.st.log.append(
            "[magenta]Когитатор содрогается!  Все орудия "
            "открывают огонь одновременно![/]"
        )
        self.st.log.append("")

        destroyed: set[int] = set()
        for ship in self.st.ships:
            if ship.hp <= 0:
                continue
            # 80-95% от текущего HP
            pct = random.uniform(0.80, 0.95)
            dmg = max(1, int(ship.hp * pct))
            ship.hp -= dmg

            if ship.hp <= 0:
                ship.hp = 0
                destroyed.add(ship.id)
                self.st.score += ship.info.points
                self.st.kills += 1
                self.st.log.append(
                    f"[bold magenta]ЯРОСТЬ[/] -> {ship.info.name} [{ship.id}]: "
                    f"[bold]{dmg}[/]  [bold green]РАСПЛАВЛЕН![/] "
                    f"+{ship.info.points}"
                )
            else:
                self.st.log.append(
                    f"[bold magenta]ЯРОСТЬ[/] -> {ship.info.name} [{ship.id}]: "
                    f"[bold]{dmg}[/]  HP:{ship.hp}/{ship.max_hp}"
                )

        self.st.ships = [s for s in self.st.ships if s.id not in destroyed]

        if self.st.temp > 90:
            self.st.log.append(
                "[bold red]КРИТИЧЕСКИЙ ПЕРЕГРЕВ! "
                "Когитатор на грани отключения![/]"
            )

    # --- События ---

    def _weirdboy_attack(self):
        self.st.log.clear()
        self.st.log.append("[bold red blink]!!! ПСИХО-АТАКА !!![/]")
        self.st.log.append(
            "[red]Вирдбой направил энергию WAAAGH! "
            "на нейропути Когитатора![/]"
        )
        self.st.log.append("[yellow]НЕЙРОПУТИ ПОВРЕЖДЕНЫ[/]")
        render_screen(self.st, "ПСИХО-АТАКА ВИРДБОЯ")
        time.sleep(2)

        self.db.stop()
        time.sleep(0.2)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE __sqfox_hnsw SET value = X'DEADBEEF' "
            "WHERE key = 'graph'"
        )
        conn.commit()
        conn.close()

        self.st.log.append("[yellow]Запуск Обряда Рекалибровки...[/]")
        render_screen(self.st, "ВОССТАНОВЛЕНИЕ")
        time.sleep(1)

        self.backend = SqliteHnswBackend(M=16, ef_construction=200, ef_search=64)
        self.db = SQFox(self.db_path, vector_backend=self.backend)
        t0 = time.time()
        self.db.start()
        rt = time.time() - t0
        self.db.ensure_schema(SchemaState.SEARCHABLE)
        time.sleep(0.2)

        recovered = self.backend.count()
        total = self.db.fetch_one("SELECT COUNT(*) FROM documents")[0]

        self.st.log.append(
            f"[bold green]ДУХ МАШИНЫ ВОССТАНОВЛЕН[/]  "
            f"{recovered}/{total} записей из backup-BLOB  "
            f"({rt:.2f}с)"
        )
        self.st.log.append(
            "[yellow]Омнисайя хранит.  Данные в сохранности.[/]"
        )
        self.st.temp = 25.0
        render_screen(self.st, "ВОССТАНОВЛЕНИЕ ЗАВЕРШЕНО")
        time.sleep(2)

    def _ingest_stc(self, group):
        self.st.log.append("[bold yellow]<<< ВХОДЯЩАЯ ПЕРЕДАЧА >>>[/]")
        for doc in group:
            self.db.ingest(
                doc["text"], metadata=doc["meta"],
                embed_fn=self.embedder, wait=True,
            )
            tp = doc["meta"].get("type", "данные")
            # Полный текст — Rich сам обрежет по ширине панели
            self.st.log.append(
                f"[green]+[/] [{tp}] {doc['text']}"
            )
        n = self.db.fetch_one("SELECT COUNT(*) FROM documents")[0]
        self.st.log.append(f"[dim]В базе теперь {n} записей.[/]")
        render_screen(self.st, "НОВЫЕ ДАННЫЕ ЗАГРУЖЕНЫ")
        time.sleep(1.5)

    def _show_ork_intercept(self):
        """Показать случайный перехват орочьей связи и заинжестить в базу."""
        if not ORK_INTERCEPTS:
            return
        msg = ORK_INTERCEPTS[self._intercept_idx % len(ORK_INTERCEPTS)]
        self._intercept_idx += 1

        # Инжест в базу SQFox — демонстрация live ingest
        self.db.ingest(
            msg,
            metadata={"type": "перехват", "origin": "Орк-частота"},
            embed_fn=self.embedder, wait=True,
        )
        n = self.db.fetch_one("SELECT COUNT(*) FROM documents")[0]

        self.st.log.append("[bold red]<<< ОРК-ЧАСТОТА >>>[/]")
        self.st.log.append(f"[yellow]{msg}[/]")
        self.st.log.append(
            f"[dim]Перехват сохранён в базу (всего {n} записей).[/]"
        )
        render_screen(self.st, "[red]ОРК-ПЕРЕХВАТ[/]")
        time.sleep(2)

    # --- Спавн ---

    def _spawn(self, wave_def):
        survivors = [s for s in self.st.ships if s.hp > 0]
        if survivors:
            self.st.log.append(
                f"[yellow]{len(survivors)} корабль(ей) "
                f"остались с прошлой волны![/]"
            )
        self.st.ships = list(survivors)
        for tk, cnt in wave_def:
            st = SHIP_TYPES[tk]
            for _ in range(cnt):
                self._sid += 1
                self.st.ships.append(
                    Ship(self._sid, tk, st.hp, st.hp)
                )

    # --- Основной цикл ---

    def run(self):
        cls()
        console.print(Panel(
            Align.center(
                "[bold white]"
                "  ___  ___  ___  ___  _ _  __  __\n"
                " | __|| __|| _ \\| _ \\| | ||  \\/  |\n"
                " | _| | _| |   /|   /| U || |\\/| |\n"
                " |_|  |___||_|_\\|_|_\\|___||_|  |_|\n"
                "\n   V I G I L A N S\n[/]\n"
                "[dim]Орбитальная Платформа GAMMA-7[/]\n"
                "[dim]Адептус Механикус * Терминал Когитатора[/]\n\n"
                "[bold red]ТРЕВОГА: Орочий флот на ауспексе.[/]\n"
                "[bold red]Боевые посты.  К бою.[/]\n\n"
                "[dim]Запрашивай Когитатор — ищи слабости врага.[/]\n"
                "[dim]Назначай оружие — уничтожай.  Выживи 6 волн.[/]\n\n"
                "[bold yellow]ВАЖНО:[/] Без запроса Когитатора — [bold red]СЛЕПОЙ ОГОНЬ[/] (50% урона, нет бонуса слабости)!\n"
                "[dim]Используй Когитатор чтобы раскрыть слабости и бить в полную силу.[/]\n"
            ),
            border_style="bold cyan",
            box=box.DOUBLE,
        ))

        self.setup()

        cls()
        console.print(Panel(
            "[bold]УПРАВЛЕНИЕ[/]\n\n"
            "[bold cyan]ФАЗА ЗАПРОСА[/] — "
            "набери вопрос для поиска в базе\n"
            '  [dim]напр.: "чем бить Kill Крейсер" '
            '/ "как уничтожить таран"[/]\n'
            "  [bold]done[/] — перейти к стрельбе\n\n"
            "[bold yellow]ФАЗА ОРУЖИЯ[/] — "
            "назначь оружие на цель\n"
            "  [dim]напр.: lance 3 / torpedo 1 / "
            "macro 5 / nova 4[/]\n"
            "  [bold magenta]rage[/] — [bold]Ярость Машинного Духа[/] "
            "(одноразовый удар по ВСЕМ врагам)\n"
            "  [bold]done[/] — огонь\n\n"
            "[bold red]СЛЕПОЙ ОГОНЬ:[/] если не запросить "
            "Когитатор по кораблю — урон 50%, нет бонуса слабости!\n"
            "[bold cyan]РЕАКТОР:[/] Лэнс расходует энергию реактора.  "
            "0 энергии = ПЕРЕГРУЗКА (урон корпусу, лэнс заблокирован на ход).",
            border_style="dim",
        ))
        try:
            Prompt.ask("[dim]Нажми Enter для начала[/]")
        except (EOFError, KeyboardInterrupt):
            return

        try:
            self._loop()
        finally:
            if self.db:
                try:
                    self.db.stop()
                except Exception:
                    pass
            for suf in ("", "-wal", "-shm"):
                p = Path(self.db_path + suf)
                try:
                    p.exists() and p.unlink()
                except OSError:
                    pass

    def _loop(self):
        for wi in range(len(WAVES)):
            self.st.wave = wi + 1
            self.st.log.clear()
            self.st.queried_ships.clear()  # Сброс запросов на новую волну
            self._spawn(WAVES[wi])

            # Охлаждение между волнами
            self.st.temp = max(20.0, self.st.temp - random.uniform(10, 20))

            # Восстановление реактора между волнами
            if self.st.reactor_overloaded:
                self.st.reactor_overloaded = False
                self.st.reactor_energy = max(2, self.st.reactor_energy)
                self.st.log.append(
                    "[cyan]Реактор стабилизирован. "
                    "Лэнс снова доступен.[/]"
                )
            # Частичное восстановление реактора
            regen = random.randint(1, 3)
            self.st.reactor_energy = min(
                self.st.max_reactor,
                self.st.reactor_energy + regen,
            )

            # ---------- ФАЗА ЗАПРОСОВ ----------
            queries_left = self.st.max_queries
            self.st.log.append(
                f"[dim]Запросов Когитатору: {queries_left}  "
                f"(без запроса = СЛЕПОЙ ОГОНЬ x{self.st.blind_fire_mult})[/]"
            )
            render_screen(
                self.st,
                f"[cyan]ФАЗА ЗАПРОСА (волна {self.st.wave})[/]  "
                f"-- набери вопрос или [bold]done[/]",
            )

            while queries_left > 0:
                try:
                    raw = Prompt.ask("[bold cyan]когитатор[/]")
                except (EOFError, KeyboardInterrupt):
                    return
                line = raw.strip()
                if not line:
                    continue
                if line.lower() in ("done", "d", "готово"):
                    break
                self._search(line)
                queries_left -= 1
                if queries_left > 0:
                    self.st.log.append(
                        f"[dim]Запросов осталось: {queries_left}[/]"
                    )
                else:
                    self.st.log.append(
                        "[yellow]Когитатор исчерпан. К оружию![/]"
                    )
                render_screen(
                    self.st,
                    f"[cyan]ЗАПРОС[/]  осталось: {queries_left}",
                )

            # Предупреждение о незапрошенных кораблях
            unqueried = [
                s for s in self.st.ships
                if s.id not in self.st.queried_ships and s.hp > 0
            ]
            if unqueried:
                names = ", ".join(
                    f"{s.info.name}[{s.id}]" for s in unqueried
                )
                self.st.log.append(
                    f"[bold red]СЛЕПОЙ ОГОНЬ по: {names} "
                    f"(50% урона, нет бонуса слабости)[/]"
                )

            # ---------- ФАЗА ОРУЖИЯ ----------
            assignments: dict[int, str] = {}
            fired_this_turn: dict[str, int] = {k: 0 for k in WEAPONS}
            self.st.log.append("[dim]Назначь: <оружие> <цель#>  (done=огонь)[/]")
            render_screen(
                self.st,
                "[yellow]ФАЗА ОРУЖИЯ[/]  -- "
                "[dim]lance 3 / torpedo 1 / macro 5 / rage[/]",
            )

            rage_used_this_turn = False

            while True:
                try:
                    raw = Prompt.ask("[bold yellow]оружие[/]")
                except (EOFError, KeyboardInterrupt):
                    return
                line = raw.strip().lower()
                if not line:
                    continue

                # Ярость Машинного Духа
                if line in ("rage", "ярость", "r"):
                    if not self.st.rage_available:
                        self.st.log.append(
                            "[red]Ярость Духа уже была "
                            "использована![/]"
                        )
                        render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                        continue
                    self.st.log.clear()
                    self._resolve_rage()
                    rage_used_this_turn = True
                    render_screen(
                        self.st, "[magenta]ЯРОСТЬ МАШИННОГО ДУХА[/]"
                    )
                    if not self.st.ships:
                        break
                    self.st.log.append(
                        "[dim]Назначить оружие на выживших?  "
                        "(done=огонь)[/]"
                    )
                    render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                    continue

                if line in ("done", "d", "fire", "f", "готово", "огонь"):
                    if not assignments and not rage_used_this_turn:
                        self.st.log.append(
                            "[red]Ничего не назначено! "
                            "Враг будет стрелять безнаказанно![/]"
                        )
                        render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                        c = Prompt.ask(
                            "[dim]  Всё равно? (y/n)[/]", default="n"
                        )
                        if c.lower() != "y":
                            continue
                    break

                parts = line.split()
                if len(parts) < 2:
                    self.st.log.append("[dim]Формат: <оружие> <цель#>[/]")
                    render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                    continue

                wk = parts[0]
                if wk not in WEAPONS:
                    # Попробуем по префиксу
                    mm = [k for k in WEAPONS if k.startswith(wk)]
                    if len(mm) == 1:
                        wk = mm[0]
                    else:
                        self.st.log.append(
                            f"[red]Нет оружия: {parts[0]}[/]  "
                            f"({', '.join(WEAPONS)})"
                        )
                        render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                        continue

                try:
                    tid = int(parts[1])
                except ValueError:
                    self.st.log.append("[dim]Цель -- число[/]")
                    render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                    continue

                ship = next(
                    (s for s in self.st.ships if s.id == tid), None
                )
                if not ship:
                    self.st.log.append(
                        f"[red]Цели [{tid}] нет на ауспексе[/]"
                    )
                    render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                    continue

                # Проверка лэнс: реактор перегружен?
                if wk == "lance" and self.st.reactor_overloaded:
                    self.st.log.append(
                        "[bold red]РЕАКТОР ПЕРЕГРУЖЕН! "
                        "Лэнс недоступен в этом ходу![/]"
                    )
                    render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                    continue

                # Проверка лэнс: хватает энергии?
                if wk == "lance":
                    lance_assigned = fired_this_turn.get("lance", 0)
                    energy_after = self.st.reactor_energy - (lance_assigned + 1)
                    if energy_after < -2:
                        self.st.log.append(
                            "[bold red]Реактор на пределе! "
                            "Энергии не хватит для ещё одного "
                            "выстрела лэнса![/]"
                        )
                        render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                        continue
                    if energy_after < 0:
                        self.st.log.append(
                            "[bold yellow]ВНИМАНИЕ: энергия реактора "
                            "уйдёт в минус! Будет ПЕРЕГРУЗКА "
                            f"(-{abs(energy_after) * 5 + 10} корпус)![/]"
                        )

                # Проверка per_turn лимита
                w_data = WEAPONS[wk]
                if fired_this_turn[wk] >= w_data["per_turn"]:
                    self.st.log.append(
                        f"[red]{w_data['name']} — лимит выстрелов "
                        f"за ход исчерпан! "
                        f"(макс. {w_data['per_turn']}/ход)[/]"
                    )
                    render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                    continue

                # Проверка боезапаса (торпеды, нова)
                if wk not in ("macro", "lance"):
                    stock = self.st.weapon_stock.get(wk, 0)
                    used = sum(
                        1 for v in assignments.values() if v == wk
                    )
                    if stock - used <= 0:
                        self.st.log.append(
                            f"[red]{WEAPONS[wk]['name']} — ПУСТО![/]"
                        )
                        render_screen(self.st, "[yellow]ОРУЖИЕ[/]")
                        continue

                if tid in assignments:
                    old_wk = assignments[tid]
                    fired_this_turn[old_wk] -= 1
                    self.st.log.append(
                        f"[dim]Замена {old_wk} -> {wk} "
                        f"на [{tid}][/]"
                    )

                assignments[tid] = wk
                fired_this_turn[wk] += 1
                w = WEAPONS[wk]
                self.st.log.append(
                    f"[{w['clr']}]{w['name']}[/] -> "
                    f"{ship.info.name} [{tid}]  [green]OK[/]"
                )
                render_screen(self.st, "[yellow]ОРУЖИЕ[/]")

            # ---------- РЕЗУЛЬТАТ БОЯ ----------
            if assignments:
                self.st.log.clear()
                self._resolve(assignments)

            if not self.st.ships:
                self.st.log.append(
                    "[bold green]Все контакты уничтожены.  "
                    "Пустота чиста.[/]"
                )
            else:
                self.st.log.append(
                    f"[yellow]{len(self.st.ships)} "
                    f"враг(ов) ещё живы.[/]"
                )

            render_screen(self.st, "[white]РЕЗУЛЬТАТ БОЯ[/]")
            time.sleep(1.5)

            # Смерть?
            if self.st.hull <= 0:
                self._game_over(False)
                return

            # ---------- МЕЖВОЛНОВЫЕ СОБЫТИЯ ----------

            # Регенерация щита при полной зачистке
            if not self.st.ships and self.st.shields < self.st.max_shields:
                self.st.shields += 1
                self.st.log.append(
                    "[cyan]Генератор щита перезарядился.  +1 щит.[/]"
                )
                render_screen(self.st, "[dim]Между волнами[/]")
                time.sleep(1)

            # Перехват орочьей связи — каждую волну
            if random.random() < 0.7:
                self._show_ork_intercept()

            # Волна 2 -> STC
            if wi == 1 and STC_FRAGMENTS:
                self._ingest_stc(STC_FRAGMENTS[0])

            # Волна 3 -> ВИРДБОЙ
            if wi == 2:
                self._weirdboy_attack()
                self.st.max_queries = 3
                self.st.log.append(
                    "[green]Когитатор оптимизирован.  "
                    "Запросов: 3 за ход.[/]"
                )

            # Волна 4 -> STC + апгрейд торпед
            if wi == 3 and len(STC_FRAGMENTS) > 1:
                self._ingest_stc(STC_FRAGMENTS[1])
                self.st.torpedo_up = True
                self.st.log.append(
                    "[bold yellow]ТОРПЕДЫ УЛУЧШЕНЫ!  +30% урона.[/]"
                )
                render_screen(self.st, "[yellow]АПГРЕЙД[/]")
                time.sleep(1.5)

            # Волна 5 -> предупреждение о шторме
            if wi == 4:
                self.st.log.append(
                    "[bold magenta]ЭЛЕКТРОМАГНИТНЫЙ ШТОРМ "
                    "ПРИБЛИЖАЕТСЯ[/]"
                )
                self.st.log.append(
                    "[dim]Ключевые слова могут не работать -- "
                    "полагайся на смысл запросов.[/]"
                )
                render_screen(self.st, "[magenta]ШТОРМ[/]")
                time.sleep(2)

            # Пауза
            if wi < len(WAVES) - 1:
                idioms = [
                    "Дух Машины мерно гудит...",
                    "Бинарные гимны эхом несутся по реле...",
                    "Дым благовоний курится вокруг когитатора...",
                ]
                self.st.log.append(f"[dim]{random.choice(idioms)}[/]")
                render_screen(self.st, "[dim]Подготовка...[/]")
                Prompt.ask("[dim]Enter -- след. волна[/]")

        # ПОБЕДА
        if self.st.hull > 0:
            self._game_over(True)

    # --- Финал ---

    def _game_over(self, victory):
        cls()

        hull_bonus = self.st.hull * 10 if victory else 0
        eff_bonus = self.st.weakness_hits * 50
        query_bonus = self.st.total_queries * 20
        self.st.score += hull_bonus + eff_bonus + query_bonus

        if victory:
            art = (
                "[bold green]"
                " _   _ ___ ___ _____ ___  ___  ___ ___ \n"
                "| | | |_ _/ __|_   _/ _ \\| _ \\|_ _/ _ \\\n"
                "| V | | || (__  | || (_) |   / | | (_) |\n"
                " \\_/  |___\\___| |_| \\___/|_|_\\|___\\___/\n"
                "[/]\n\n"
                "[bold white]Орочий флот разбит![/]\n"
                "[dim]Ferrum Vigilans выстоял.  "
                "Омнисайя доволен.[/]"
            )
            border = "bold green"
        else:
            art = (
                "[bold red]"
                " ___ ___ ___ ___ ___ ___ ___ _  _ ___ ___ \n"
                "|   \\ __/ __|_   _| _ \\ / _ \\ || | __| _ \\\n"
                "| |) | _\\__ \\ | | |   /| (_) || || _||  /\n"
                "|___/___|___/ |_| |_|_\\ \\___/ \\_, |___|_|_\\\n"
                "                               |__/\n"
                "[/]\n\n"
                "[bold red]Платформа уничтожена![/]\n"
                "[dim]Ferrum Vigilans замолчал.  "
                "Дух Машины покинул металл.[/]"
            )
            border = "bold red"

        console.print(Panel(
            Align.center(art),
            border_style=border,
            box=box.DOUBLE,
        ))

        st = Table(
            title="ИТОГОВЫЙ ОТЧЁТ КОГИТАТОРА",
            box=box.ROUNDED,
            title_style="bold white",
        )
        st.add_column("Метрика", style="bold")
        st.add_column("Значение", justify="right")
        st.add_row("Уничтожено кораблей", str(self.st.kills))
        st.add_row("Попаданий в слабость", str(self.st.weakness_hits))
        st.add_row("Запросов Когитатору", str(self.st.total_queries))
        st.add_row("Корпус", f"{self.st.hull}/{self.st.max_hull}")
        st.add_row("Бонус за корпус", f"+{hull_bonus}")
        st.add_row("Бонус за меткость", f"+{eff_bonus}")
        st.add_row("Бонус за запросы", f"+{query_bonus}")
        st.add_row("ИТОГО", f"[bold white]{self.st.score}[/]")
        console.print(st)

        if self.db:
            try:
                diag = self.db.diagnostics()
                dt = Table(
                    title="SQFox Диагностика",
                    box=box.ROUNDED,
                    title_style="bold cyan",
                )
                dt.add_column("Ключ", style="bold")
                dt.add_column("Значение", justify="right")
                for k, v in diag.items():
                    dt.add_row(str(k), str(v))
                console.print(dt)
            except Exception:
                pass


# =====================================================================

def main():
    try:
        CogitatorGame().run()
    except KeyboardInterrupt:
        console.print("\n[dim]Когитатор выключается...[/]")


if __name__ == "__main__":
    main()
