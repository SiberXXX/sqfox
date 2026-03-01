"""Text tokenization and lemmatization with graceful degradation."""

from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger("sqfox.tokenizer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_CYRILLIC_RANGE = range(0x0400, 0x0500)
_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}

# Cache for lemmatizer instances (per language)
_lemmatizer_cache: dict[str, Callable[[str], str]] = {}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_lang(text: str) -> str:
    """Detect primary language of text using Unicode range heuristic.

    Returns:
        'ru' for Cyrillic-dominant text,
        'en' for Latin-dominant text (default),
        'unknown' for empty or non-alphabetic text.

    Samples up to 200 alpha characters for efficiency.
    """
    cyrillic_count = 0
    latin_count = 0
    sampled = 0

    for ch in text:
        if sampled >= 200:
            break
        if ch.isalpha():
            sampled += 1
            cp = ord(ch)
            if cp in _CYRILLIC_RANGE:
                cyrillic_count += 1
            elif cp < 0x0250:  # Basic Latin + Latin Extended
                latin_count += 1

    if sampled == 0:
        return "unknown"

    if cyrillic_count / sampled > 0.4:
        return "ru"
    return "en"


def detect_word_lang(word: str) -> str:
    """Detect language of a single word by its characters.

    Returns 'ru' if any character is Cyrillic, 'en' otherwise.
    Fast path for per-word language dispatch in mixed texts.
    """
    for ch in word:
        if ord(ch) in _CYRILLIC_RANGE:
            return "ru"
    return "en"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Split text into word tokens.

    Uses Unicode-aware word boundary splitting.
    Lowercases all tokens.  Filters tokens shorter than 2 characters.
    """
    return [
        m.group().lower()
        for m in _WORD_RE.finditer(text)
        if len(m.group()) >= 2
    ]


# ---------------------------------------------------------------------------
# Lemmatizer resolution
# ---------------------------------------------------------------------------

def _get_lemmatizer(lang: str) -> Callable[[str], str]:
    """Get the best available lemmatizer for the given language.

    Priority chain:
      RU: pymorphy3 -> simplemma -> raw passthrough
      EN: simplemma -> raw passthrough
      *:  simplemma -> raw passthrough
    """
    if lang in _lemmatizer_cache:
        return _lemmatizer_cache[lang]

    lemmatizer: Callable[[str], str]

    if lang == "ru":
        lemmatizer = _try_pymorphy3()
        if lemmatizer is None:
            lemmatizer = _try_simplemma(lang)
    else:
        lemmatizer = _try_simplemma(lang)

    _lemmatizer_cache[lang] = lemmatizer
    return lemmatizer


def _try_pymorphy3() -> Callable[[str], str] | None:
    """Try to create a pymorphy3-based lemmatizer for Russian."""
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()

        def _pymorphy_lemma(word: str) -> str:
            parsed = morph.parse(word)
            return parsed[0].normal_form if parsed else word

        logger.info("Using pymorphy3 for Russian lemmatization")
        return _pymorphy_lemma
    except ImportError:
        logger.debug("pymorphy3 not available")
        return None
    except Exception as exc:
        logger.warning(
            "pymorphy3 installed but failed to initialize (missing dictionary?): %s",
            exc,
        )
        return None


def _try_simplemma(lang: str) -> Callable[[str], str]:
    """Try to create a simplemma-based lemmatizer. Falls back to passthrough."""
    try:
        import simplemma

        def _simplemma_lemma(word: str) -> str:
            return simplemma.lemmatize(word, lang=lang)

        logger.info("Using simplemma for '%s' lemmatization", lang)
        return _simplemma_lemma
    except ImportError:
        logger.warning(
            "simplemma not installed, using raw passthrough for '%s'", lang
        )
        return _passthrough


def _passthrough(word: str) -> str:
    """Identity function — returns word unchanged."""
    return word


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _lemmatize_token(token: str, lang: str | None) -> str:
    """Lemmatize a single token, choosing the right lemmatizer.

    If lang is explicitly set, uses that language's lemmatizer.
    If lang is None, detects per-word language by character script.
    """
    if lang is not None:
        return _get_lemmatizer(lang)(token)
    word_lang = detect_word_lang(token)
    return _get_lemmatizer(word_lang)(token)


def lemmatize(text: str, lang: str | None = None) -> str:
    """Lemmatize all words in text, preserving word order.

    Args:
        text: Input text.
        lang: Language code ('ru', 'en').  If None, auto-detected
              per-word — Cyrillic words go through Russian lemmatizer,
              Latin words through English lemmatizer.

    Returns:
        Space-joined lemmatized tokens.
    """
    if lang == "unknown":
        lang = "en"

    tokens = tokenize(text)
    return " ".join(_lemmatize_token(token, lang) for token in tokens)


def lemmatize_query(query: str, lang: str | None = None) -> str:
    """Lemmatize a search query, suitable for FTS5 MATCH.

    Same as lemmatize() but preserves FTS5 operators when written in
    ALL CAPS in the original query (AND, OR, NOT, NEAR).
    Lowercase words like "not", "and", "or" are lemmatized normally —
    they are NOT treated as operators.
    """
    if lang == "unknown":
        lang = "en"

    # Extract original-case tokens to detect ALL-CAPS operators
    raw_matches = list(_WORD_RE.finditer(query))

    result = []
    for match in raw_matches:
        raw = match.group()
        if len(raw) < 2:
            continue
        # Only treat as FTS5 operator if written in ALL CAPS in original text
        if raw in _FTS5_OPERATORS:
            result.append(raw)
        else:
            result.append(_lemmatize_token(raw.lower(), lang))

    return " ".join(result)
