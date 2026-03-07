"""Unit tests for sqfox._auto — environment detection and PRAGMA tuning."""

import os
import sqlite3
import sys
from unittest import mock

import pytest

from sqfox._auto import (
    AUTO,
    EnvironmentInfo,
    MemoryTier,
    PlatformClass,
    _check_fts5_available,
    _classify_memory,
    _detect_cpu_count,
    _detect_platform_class,
    _detect_total_ram_mb,
    _is_sd_card_path,
    _recommend_cpu_workers,
    _recommend_pragmas,
    _recommend_reader_prune_threshold,
    detect_environment,
    resolve_param,
)


# ---------------------------------------------------------------------------
# resolve_param
# ---------------------------------------------------------------------------

class TestResolveParam:
    def test_auto_returns_detected(self):
        assert resolve_param(AUTO, 42) == 42

    def test_explicit_int_overrides(self):
        assert resolve_param(100, 42) == 100

    def test_explicit_zero_overrides(self):
        assert resolve_param(0, 42) == 0

    def test_explicit_string_not_auto(self):
        assert resolve_param("something", 42) == "something"

    def test_auto_sentinel_identity(self):
        assert AUTO is AUTO
        assert AUTO == "auto"

    def test_auto_from_json_config(self):
        """AUTO from JSON config (not interned) should still resolve."""
        import json
        auto_from_json = json.loads('"auto"')
        assert resolve_param(auto_from_json, 42) == 42

    def test_auto_from_concatenation(self):
        """Dynamically constructed 'auto' should still resolve."""
        dynamic = "au" + "to"
        assert resolve_param(dynamic, 42) == 42


# ---------------------------------------------------------------------------
# Memory tier classification
# ---------------------------------------------------------------------------

class TestClassifyMemory:
    def test_low_tier(self):
        assert _classify_memory(256) == MemoryTier.LOW
        assert _classify_memory(512) == MemoryTier.LOW
        assert _classify_memory(1023) == MemoryTier.LOW

    def test_medium_tier(self):
        assert _classify_memory(1024) == MemoryTier.MEDIUM
        assert _classify_memory(2048) == MemoryTier.MEDIUM
        assert _classify_memory(4096) == MemoryTier.MEDIUM

    def test_high_tier(self):
        assert _classify_memory(4097) == MemoryTier.HIGH
        assert _classify_memory(8192) == MemoryTier.HIGH
        assert _classify_memory(65536) == MemoryTier.HIGH


# ---------------------------------------------------------------------------
# PRAGMA recommendations
# ---------------------------------------------------------------------------

class TestRecommendPragmas:
    def test_low_tier_defaults(self):
        cache, mmap = _recommend_pragmas(
            MemoryTier.LOW, False, PlatformClass.DESKTOP
        )
        assert cache == 4_000
        assert mmap == 0

    def test_medium_tier_defaults(self):
        cache, mmap = _recommend_pragmas(
            MemoryTier.MEDIUM, False, PlatformClass.DESKTOP
        )
        assert cache == 16_000
        assert mmap == 64

    def test_high_tier_defaults(self):
        cache, mmap = _recommend_pragmas(
            MemoryTier.HIGH, False, PlatformClass.DESKTOP
        )
        assert cache == 64_000
        assert mmap == 256

    def test_sd_card_disables_mmap(self):
        _, mmap = _recommend_pragmas(
            MemoryTier.HIGH, True, PlatformClass.DESKTOP
        )
        assert mmap == 0

    def test_android_mmap_cap(self):
        _, mmap = _recommend_pragmas(
            MemoryTier.HIGH, False, PlatformClass.ANDROID_TERMUX
        )
        assert mmap == 64

    def test_android_low_mmap_unchanged(self):
        _, mmap = _recommend_pragmas(
            MemoryTier.MEDIUM, False, PlatformClass.ANDROID_TERMUX
        )
        assert mmap == 64  # already <= 64, no change


# ---------------------------------------------------------------------------
# SD card heuristic
# ---------------------------------------------------------------------------

class TestSdCardHeuristic:
    def test_media_path(self):
        assert _is_sd_card_path("/media/user/mysd/data.db") is True

    def test_mnt_sd(self):
        assert _is_sd_card_path("/mnt/sd0/data.db") is True

    def test_mnt_external_sd(self):
        assert _is_sd_card_path("/mnt/external_sd/data.db") is True

    def test_run_media(self):
        assert _is_sd_card_path("/run/media/user/card/data.db") is True

    def test_normal_path(self):
        assert _is_sd_card_path("/home/user/data.db") is False

    def test_storage_emulated_is_not_sd(self):
        # /storage/emulated is Android internal UFS flash, NOT SD card
        assert _is_sd_card_path("/storage/emulated/0/data.db") is False

    def test_android_uuid_sd_card(self):
        assert _is_sd_card_path("/storage/14FE-0C12/data.db") is True

    def test_android_uuid_lowercase(self):
        assert _is_sd_card_path("/storage/abcd-ef01/data.db") is True

    def test_windows_path(self):
        assert _is_sd_card_path("C:\\Users\\test\\data.db") is False

    def test_empty_path(self):
        assert _is_sd_card_path("") is False

    def test_sata_drive_sda_not_sd_card(self):
        # /mnt/sda1 is a SATA drive, NOT an SD card
        assert _is_sd_card_path("/mnt/sda1/data.db") is False

    def test_sata_drive_sdb_not_sd_card(self):
        assert _is_sd_card_path("/mnt/sdb2/data.db") is False

    def test_mnt_sdcard_is_sd_card(self):
        assert _is_sd_card_path("/mnt/sdcard/data.db") is True


# ---------------------------------------------------------------------------
# CPU workers recommendation
# ---------------------------------------------------------------------------

class TestRecommendCpuWorkers:
    """CPU workers depend on platform + tier, not just core count."""

    def test_android_always_one(self):
        assert _recommend_cpu_workers(8, MemoryTier.HIGH, PlatformClass.ANDROID_TERMUX) == 1

    def test_rpi_always_one(self):
        assert _recommend_cpu_workers(4, MemoryTier.MEDIUM, PlatformClass.RASPBERRY_PI) == 1

    def test_sbc_always_one(self):
        assert _recommend_cpu_workers(4, MemoryTier.MEDIUM, PlatformClass.UNKNOWN_SBC) == 1

    def test_low_tier_desktop_one(self):
        assert _recommend_cpu_workers(8, MemoryTier.LOW, PlatformClass.DESKTOP) == 1

    def test_medium_tier_desktop_capped_at_two(self):
        assert _recommend_cpu_workers(8, MemoryTier.MEDIUM, PlatformClass.DESKTOP) == 2

    def test_medium_tier_dual_core(self):
        assert _recommend_cpu_workers(2, MemoryTier.MEDIUM, PlatformClass.DESKTOP) == 1

    def test_high_tier_desktop_quad(self):
        assert _recommend_cpu_workers(4, MemoryTier.HIGH, PlatformClass.DESKTOP) == 2

    def test_high_tier_desktop_sixteen(self):
        assert _recommend_cpu_workers(16, MemoryTier.HIGH, PlatformClass.DESKTOP) == 8

    def test_high_tier_desktop_single_core(self):
        # Edge case: 1 core HIGH (weird but possible VPS)
        assert _recommend_cpu_workers(1, MemoryTier.HIGH, PlatformClass.DESKTOP) == 2


# ---------------------------------------------------------------------------
# Reader prune threshold
# ---------------------------------------------------------------------------

class TestReaderPruneThreshold:
    def test_low(self):
        assert _recommend_reader_prune_threshold(MemoryTier.LOW) == 5

    def test_medium(self):
        assert _recommend_reader_prune_threshold(MemoryTier.MEDIUM) == 10

    def test_high(self):
        assert _recommend_reader_prune_threshold(MemoryTier.HIGH) == 20


# ---------------------------------------------------------------------------
# FTS5 check
# ---------------------------------------------------------------------------

class TestFts5Check:
    def test_fts5_available(self):
        # Should be True in most standard Python builds
        result = _check_fts5_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

class TestPlatformDetection:
    def test_termux_env(self):
        with mock.patch.dict(os.environ, {"TERMUX_VERSION": "0.119"}):
            assert _detect_platform_class() == PlatformClass.ANDROID_TERMUX

    def test_termux_prefix(self):
        with mock.patch.dict(
            os.environ,
            {"PREFIX": "/data/data/com.termux/files/usr"},
            clear=False,
        ):
            # Remove TERMUX_VERSION if present
            env = os.environ.copy()
            env.pop("TERMUX_VERSION", None)
            env["PREFIX"] = "/data/data/com.termux/files/usr"
            with mock.patch.dict(os.environ, env, clear=True):
                assert _detect_platform_class() == PlatformClass.ANDROID_TERMUX

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_desktop_on_linux(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("TERMUX_VERSION", None)
            with mock.patch.dict(os.environ, env, clear=True):
                # On a normal Linux desktop, should be DESKTOP
                # (unless running on ARM SBC)
                result = _detect_platform_class()
                assert result in (
                    PlatformClass.DESKTOP,
                    PlatformClass.UNKNOWN_SBC,
                    PlatformClass.RASPBERRY_PI,
                )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_desktop_on_windows(self):
        result = _detect_platform_class()
        assert result == PlatformClass.DESKTOP


# ---------------------------------------------------------------------------
# RAM detection
# ---------------------------------------------------------------------------

class TestRamDetection:
    def test_returns_positive_int(self):
        ram = _detect_total_ram_mb()
        assert isinstance(ram, int)
        assert ram > 0

    def test_fallback_on_failure(self):
        # Mock all detection paths to fail
        with mock.patch("sys.platform", "unknown_os"):
            ram = _detect_total_ram_mb()
            assert ram == 1024  # fallback


# ---------------------------------------------------------------------------
# CPU count detection
# ---------------------------------------------------------------------------

class TestCpuCountDetection:
    def test_returns_positive_int(self):
        count = _detect_cpu_count()
        assert isinstance(count, int)
        assert count >= 1

    def test_fallback_when_getaffinity_missing(self):
        # On Windows, sched_getaffinity doesn't exist — the function
        # already hits the except branch.  On Linux, we mock it.
        if hasattr(os, "sched_getaffinity"):
            with mock.patch("os.sched_getaffinity", side_effect=AttributeError):
                count = _detect_cpu_count()
                assert count >= 1
        else:
            # Windows / macOS: fallback path is exercised natively
            count = _detect_cpu_count()
            assert count >= 1


# ---------------------------------------------------------------------------
# Full environment detection
# ---------------------------------------------------------------------------

class TestDetectEnvironment:
    def test_returns_frozen_dataclass(self):
        env = detect_environment()
        assert isinstance(env, EnvironmentInfo)
        # Frozen: should not allow mutation
        with pytest.raises(AttributeError):
            env.total_ram_mb = 999

    def test_with_db_path(self):
        env = detect_environment("/home/test/data.db")
        assert env.is_sd_card is False

    def test_with_sd_card_path(self):
        env = detect_environment("/media/user/sdcard/data.db")
        assert env.is_sd_card is True

    def test_all_fields_populated(self):
        env = detect_environment()
        assert env.total_ram_mb > 0
        assert env.cpu_count >= 1
        assert isinstance(env.memory_tier, MemoryTier)
        assert isinstance(env.platform_class, PlatformClass)
        assert isinstance(env.is_sd_card, bool)
        assert isinstance(env.fts5_available, bool)
        assert env.recommended_cache_size_kb > 0
        assert env.recommended_mmap_size_mb >= 0
        assert env.recommended_cpu_workers >= 1
        assert env.recommended_reader_prune_threshold > 0

    def test_consistency(self):
        """Two detect calls should return same values."""
        env1 = detect_environment("test.db")
        env2 = detect_environment("test.db")
        assert env1.total_ram_mb == env2.total_ram_mb
        assert env1.cpu_count == env2.cpu_count
        assert env1.memory_tier == env2.memory_tier
