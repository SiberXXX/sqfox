"""Auto-adaptive environment detection and PRAGMA tuning.

Detects RAM, CPU count, platform class, and storage hints using only
the Python standard library.  All results are cached in a frozen
dataclass after a single detection pass at engine ``start()``.

Zero external dependencies.  Detection takes <50 ms.
"""

from __future__ import annotations

import logging
import os
import platform
import sqlite3
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("sqfox.auto")

# Sentinel: when used as a parameter default, means "auto-detect".
# Users who pass an explicit int/value override auto-detection.
AUTO = "auto"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryTier(Enum):
    LOW = auto()      # < 1 GB
    MEDIUM = auto()   # 1-4 GB
    HIGH = auto()     # > 4 GB


class PlatformClass(Enum):
    DESKTOP = auto()
    RASPBERRY_PI = auto()
    ANDROID_TERMUX = auto()
    UNKNOWN_SBC = auto()  # other ARM single-board computers


# ---------------------------------------------------------------------------
# Frozen environment snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Immutable snapshot of detected environment.  Created once at start()."""
    total_ram_mb: int
    cpu_count: int
    memory_tier: MemoryTier
    platform_class: PlatformClass
    is_sd_card: bool
    fts5_available: bool
    recommended_cache_size_kb: int
    recommended_mmap_size_mb: int
    recommended_cpu_workers: int
    recommended_reader_prune_threshold: int


# ---------------------------------------------------------------------------
# Detection probes (all stdlib, cross-platform)
# ---------------------------------------------------------------------------

def _detect_total_ram_mb() -> int:
    """Detect total system RAM in megabytes.

    Priority:
      1. Linux / Android (Termux): /proc/meminfo
      2. macOS: sysctl hw.memsize via ctypes
      3. Windows: GlobalMemoryStatusEx via ctypes
      4. Fallback: 1024 MB (safe MEDIUM assumption)
    """
    # Linux / Android / Raspberry Pi
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) // 1024  # kB -> MB
        except (OSError, ValueError, IndexError):
            pass

    # macOS
    if sys.platform == "darwin":
        try:
            import ctypes
            libc = ctypes.CDLL("libSystem.B.dylib")
            size = ctypes.c_uint64(0)
            sz = ctypes.c_size_t(8)
            if libc.sysctlbyname(
                b"hw.memsize",
                ctypes.byref(size),
                ctypes.byref(sz),
                None,
                0,
            ) == 0:
                return size.value // (1024 * 1024)
        except Exception:
            pass

    # Windows (broad except: corporate AV can block ctypes)
    if sys.platform == "win32":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            result = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(stat)
            )
            if result != 0:
                return stat.ullTotalPhys // (1024 * 1024)
        except Exception:
            pass

    logger.debug("Could not detect RAM, assuming 1024 MB")
    return 1024


def _detect_cpu_count() -> int:
    """Detect available CPU count.  Returns at least 1.

    Prefers ``os.sched_getaffinity`` (Linux) which respects cgroups /
    Docker CPU limits.  Falls back to ``os.cpu_count()`` elsewhere.
    """
    try:
        # Linux: respects cgroups / taskset / Docker --cpus
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        # Not available on macOS / Windows
        return max(1, os.cpu_count() or 1)


def _detect_platform_class() -> PlatformClass:
    """Classify platform: desktop, Raspberry Pi, Android/Termux, or SBC."""
    # Android / Termux
    if os.environ.get("TERMUX_VERSION") or os.environ.get(
        "PREFIX", ""
    ).startswith("/data/data/com.termux"):
        return PlatformClass.ANDROID_TERMUX

    # Raspberry Pi (Linux only)
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                if "raspberry" in f.read().lower():
                    return PlatformClass.RASPBERRY_PI
        except (OSError, UnicodeDecodeError):
            pass

        # ARM architecture on Linux → likely an SBC
        machine = platform.machine().lower()
        if machine.startswith("arm") or machine == "aarch64":
            return PlatformClass.UNKNOWN_SBC

    return PlatformClass.DESKTOP


def _is_sd_card_path(db_path: str) -> bool:
    """Heuristic: does the path look like removable / SD card storage?

    Note: ``/storage/emulated`` is Android's internal flash (UFS), not
    a physical SD card — so it is intentionally excluded.
    Android mounts physical SD cards at ``/storage/XXXX-XXXX`` where the
    label is typically a hex UUID (e.g. ``/storage/14FE-0C12``).
    """
    import re

    p = db_path.lower().replace("\\", "/")
    # Only match mount-root paths: /media/<user>/... and /run/media/<user>/...
    # Avoids false positives for /home/user/media/ or /var/www/media/.
    patterns = (
        "/mnt/external_sd",
    )
    if any(pat in p for pat in patterns):
        return True
    if p.startswith("/media/") or p.startswith("/run/media/"):
        return True
    # /mnt/sd/ or /mnt/sdcard but NOT /mnt/sda1, /mnt/sdb2 (SATA drives)
    if re.search(r"/mnt/sd(?:card|[^a-z]|$)", p):
        return True
    # Android SD card: /storage/<HEX-UUID>/  (e.g. /storage/14FE-0C12/)
    if re.match(r"/storage/[0-9a-f]{4}-[0-9a-f]{4}/", p):
        return True
    return False


def _check_fts5_available() -> bool:
    """Check if the SQLite build includes FTS5."""
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tier computation
# ---------------------------------------------------------------------------

def _classify_memory(ram_mb: int) -> MemoryTier:
    if ram_mb < 1024:
        return MemoryTier.LOW
    if ram_mb <= 4096:
        return MemoryTier.MEDIUM
    return MemoryTier.HIGH


def _recommend_pragmas(
    tier: MemoryTier,
    is_sd_card: bool,
    platform_class: PlatformClass,
) -> tuple[int, int]:
    """Return (cache_size_kb, mmap_size_mb) recommendations."""
    cache_map = {
        MemoryTier.LOW: 4_000,
        MemoryTier.MEDIUM: 16_000,
        MemoryTier.HIGH: 64_000,
    }
    mmap_map = {
        MemoryTier.LOW: 0,
        MemoryTier.MEDIUM: 64,
        MemoryTier.HIGH: 256,
    }
    cache = cache_map[tier]
    mmap = mmap_map[tier]

    if is_sd_card:
        mmap = 0

    if platform_class == PlatformClass.ANDROID_TERMUX and mmap > 64:
        mmap = 64

    return cache, mmap


def _recommend_cpu_workers(
    cpu_count: int,
    tier: MemoryTier,
    platform_class: PlatformClass,
) -> int:
    """Recommended max_cpu_workers for AsyncSQFox.

    Prioritises device safety over throughput:
    - Android / RPi / SBC → 1 (battery, thermal, no cooler)
    - LOW memory on any platform → 1 (context-switch + swap kill)
    - MEDIUM → up to 2 (old laptops, cheap VPS)
    - HIGH + Desktop → max(2, cpu_count // 2) (full power)

    Power users can always override with ``max_cpu_workers=N``.
    """
    if platform_class in (
        PlatformClass.ANDROID_TERMUX,
        PlatformClass.RASPBERRY_PI,
        PlatformClass.UNKNOWN_SBC,
    ):
        return 1
    if tier == MemoryTier.LOW:
        return 1
    if tier == MemoryTier.MEDIUM:
        return min(2, max(1, cpu_count // 2))
    # HIGH tier + Desktop/VPS
    return max(2, cpu_count // 2)


def _recommend_reader_prune_threshold(tier: MemoryTier) -> int:
    """Reader connections before pruning dead threads."""
    return {
        MemoryTier.LOW: 5,
        MemoryTier.MEDIUM: 10,
        MemoryTier.HIGH: 20,
    }[tier]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_environment(db_path: str = "") -> EnvironmentInfo:
    """Run all probes and return a frozen EnvironmentInfo.

    Called once at ``SQFox.start()``.  Typically takes <50 ms.
    """
    ram_mb = _detect_total_ram_mb()
    cpu_count = _detect_cpu_count()
    tier = _classify_memory(ram_mb)
    plat = _detect_platform_class()
    sd_card = _is_sd_card_path(db_path) if db_path else False
    fts5 = _check_fts5_available()
    cache_kb, mmap_mb = _recommend_pragmas(tier, sd_card, plat)
    workers = _recommend_cpu_workers(cpu_count, tier, plat)
    prune = _recommend_reader_prune_threshold(tier)

    env = EnvironmentInfo(
        total_ram_mb=ram_mb,
        cpu_count=cpu_count,
        memory_tier=tier,
        platform_class=plat,
        is_sd_card=sd_card,
        fts5_available=fts5,
        recommended_cache_size_kb=cache_kb,
        recommended_mmap_size_mb=mmap_mb,
        recommended_cpu_workers=workers,
        recommended_reader_prune_threshold=prune,
    )

    logger.info(
        "Environment: RAM=%dMB (%s), CPUs=%d, platform=%s, "
        "SD=%s, FTS5=%s -> cache=%dKB, mmap=%dMB",
        ram_mb, tier.name, cpu_count, plat.name, sd_card, fts5,
        cache_kb, mmap_mb,
    )
    return env


def resolve_param(user_value: Any, auto_value: Any) -> Any:
    """Return *user_value* if explicitly set (not AUTO), else *auto_value*."""
    if user_value == AUTO:
        return auto_value
    return user_value
