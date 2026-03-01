"""Tests for sqfox tokenizer: language detection, tokenization, lemmatization."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from sqfox.tokenizer import detect_lang, tokenize, lemmatize, lemmatize_query


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestDetectLang:
    def test_detect_english(self):
        assert detect_lang("Hello world, this is a test") == "en"

    def test_detect_russian(self):
        assert detect_lang("Привет мир, это тест") == "ru"

    def test_detect_mixed_cyrillic_dominant(self):
        # More than 40% Cyrillic -> "ru"
        assert detect_lang("Привет hello мир world тест") == "ru"

    def test_detect_mixed_latin_dominant(self):
        assert detect_lang("Hello world testing один") == "en"

    def test_detect_empty(self):
        assert detect_lang("") == "unknown"

    def test_detect_numbers_only(self):
        assert detect_lang("12345 67890") == "unknown"

    def test_detect_long_text(self):
        # Should sample only 200 chars
        text = "a" * 1000
        assert detect_lang(text) == "en"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_tokenize_basic(self):
        tokens = tokenize("Hello World!")
        assert tokens == ["hello", "world"]

    def test_tokenize_unicode(self):
        tokens = tokenize("Привет мир")
        assert tokens == ["привет", "мир"]

    def test_tokenize_filters_short(self):
        tokens = tokenize("I am a test of x y z things")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "x" not in tokens
        assert "am" in tokens
        assert "test" in tokens
        assert "things" in tokens

    def test_tokenize_mixed_script(self):
        tokens = tokenize("Hello мир test тест")
        assert tokens == ["hello", "мир", "test", "тест"]

    def test_tokenize_numbers_included(self):
        tokens = tokenize("version 2023 release")
        assert "2023" in tokens

    def test_tokenize_empty(self):
        assert tokenize("") == []

    def test_tokenize_punctuation_only(self):
        assert tokenize("... !!! ???") == []


# ---------------------------------------------------------------------------
# Lemmatization
# ---------------------------------------------------------------------------

class TestLemmatize:
    def test_lemmatize_fallback_no_deps(self):
        """Without any deps, should still return tokens."""
        result = lemmatize("hello world test")
        assert len(result) > 0
        # Should contain word-like tokens
        words = result.split()
        assert len(words) >= 2

    def test_lemmatize_auto_detect_english(self):
        result = lemmatize("running tests quickly")
        assert len(result) > 0

    def test_lemmatize_auto_detect_russian(self):
        result = lemmatize("тестирование работает")
        assert len(result) > 0

    def test_lemmatize_explicit_lang(self):
        result = lemmatize("testing things", lang="en")
        assert len(result) > 0

    def test_lemmatize_unknown_lang_defaults(self):
        # Numbers-only text detected as "unknown", defaults to "en"
        result = lemmatize("12345", lang=None)
        # "12345" tokenizes as a word (5 chars), passes through
        assert "12345" in result

        result = lemmatize("12345 test", lang=None)
        assert "test" in result


class TestLemmatizeWithSimplemma:
    """Tests that require simplemma."""

    @pytest.fixture(autouse=True)
    def check_simplemma(self):
        try:
            import simplemma
        except ImportError:
            pytest.skip("simplemma not installed")

    def test_english_lemmatization(self):
        result = lemmatize("running tests configured", lang="en")
        words = result.split()
        # simplemma should reduce "running" -> "run", "tests" -> "test", etc.
        assert "run" in words or "running" in words
        assert "test" in words or "tests" in words


class TestLemmatizeWithPymorphy3:
    """Tests that require pymorphy3."""

    @pytest.fixture(autouse=True)
    def check_pymorphy3(self):
        try:
            import pymorphy3
        except ImportError:
            pytest.skip("pymorphy3 not installed")

    def test_russian_lemmatization(self):
        result = lemmatize("работающие тесты", lang="ru")
        words = result.split()
        # pymorphy3 should reduce "работающие" -> "работать" or similar
        assert len(words) >= 2


# ---------------------------------------------------------------------------
# Query lemmatization
# ---------------------------------------------------------------------------

class TestLemmatizeQuery:
    def test_preserves_fts5_operators(self):
        result = lemmatize_query("test AND running OR config")
        assert "AND" in result
        assert "OR" in result

    def test_preserves_not_operator(self):
        result = lemmatize_query("test NOT excluded")
        assert "NOT" in result

    def test_preserves_near_operator(self):
        result = lemmatize_query("test NEAR config")
        assert "NEAR" in result

    def test_basic_query(self):
        result = lemmatize_query("hello world")
        assert len(result) > 0
        words = result.split()
        assert len(words) >= 2
