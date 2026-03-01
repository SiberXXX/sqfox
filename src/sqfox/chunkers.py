"""Built-in chunking strategies and text preprocessors for sqfox.

All chunkers return callables matching the ChunkerFn protocol:
    (text: str) -> list[str]

Usage:
    from sqfox.chunkers import sentence_chunker, paragraph_chunker, markdown_chunker
    from sqfox.chunkers import html_to_text

    clean = html_to_text(raw_html)
    db.ingest(clean, chunker=sentence_chunker(chunk_size=500, overlap=1))
    db.ingest(text, chunker=paragraph_chunker(max_size=1000))
    db.ingest(text, chunker=markdown_chunker())
"""

from __future__ import annotations

import re
from html import unescape as html_unescape
from typing import Callable

# ---------------------------------------------------------------------------
# Text preprocessors
# ---------------------------------------------------------------------------

# Tags that should produce a paragraph break when removed
_BLOCK_TAGS = re.compile(
    r"</?(?:p|div|section|article|aside|header|footer|nav|main"
    r"|h[1-6]|ul|ol|li|blockquote|pre|table|tr|thead|tbody"
    r"|figure|figcaption|details|summary|hr|br)\b[^>]*>",
    re.IGNORECASE,
)
# All remaining tags
_ANY_TAG = re.compile(r"<[^>]+>")
# Runs of whitespace that include at least one newline
_BLANK_LINES = re.compile(r"[ \t]*\n[ \t]*\n[ \t\n]*")


def html_to_text(html: str) -> str:
    """Convert HTML to clean plain text.

    - Strips all tags.
    - Converts block-level tags (p, div, h1-h6, li, tr, br, hr) to
      paragraph breaks so the chunker can split on them.
    - Decodes HTML entities (&amp; → &, etc.).
    - Normalizes whitespace.

    This is intentionally simple (regex-based, no DOM parser).
    For complex HTML with nested tables or heavy JS, use a proper
    library (BeautifulSoup, lxml) and pass the result to sqfox.

    Args:
        html: Raw HTML string.

    Returns:
        Clean plain text with paragraph breaks.
    """
    text = html
    # Remove script and style contents entirely
    text = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<style\b[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    # Remove HTML comments
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    # Block tags → newlines
    text = _BLOCK_TAGS.sub("\n\n", text)
    # All other tags → nothing
    text = _ANY_TAG.sub("", text)
    # Decode entities
    text = html_unescape(text)
    # Normalize
    text = _normalize(text)
    return text


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize whitespace: collapse runs, strip, unify line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse spaces/tabs (but not newlines) within lines
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Common abbreviations that should not end a sentence (EN + RU)
_ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "gen", "gov", "sgt", "cpl", "pvt", "capt", "lt", "col", "maj",
    "dept", "univ", "assn", "bros", "inc", "ltd", "co", "corp",
    "vs", "etc", "approx", "appt", "apt", "dept", "est", "min",
    "max", "misc", "no", "vol", "fig", "eq",
    # Russian
    "г", "гг", "т", "д", "пр", "ул", "р", "руб", "коп",
    "см", "мм", "м", "км", "кг", "гр", "мл", "л",
    "др", "проф", "акад", "доц", "канд",
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles RU and EN.

    Strategy: split on sentence-ending punctuation (.!?) followed by
    whitespace + uppercase/digit/quote, but protect abbreviations,
    decimals, and ellipses.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    sentences: list[str] = []
    current = ""

    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        current += ch

        # Check if this is a sentence-ending position
        if ch in ".!?" and i + 1 < length:
            # Ellipsis: ... — don't split
            if ch == "." and i + 2 < length and text[i + 1] == "." and text[i + 2] == ".":
                current += text[i + 1] + text[i + 2]
                i += 3
                continue

            # Look ahead: next non-space char
            j = i + 1
            while j < length and text[j] in " \t":
                j += 1

            if j >= length:
                # End of text
                i += 1
                continue

            next_ch = text[j]

            # Only split if next char is uppercase, digit, or quote
            if not (next_ch.isupper() or next_ch.isdigit() or next_ch in '"\'(«"'):
                i += 1
                continue

            # Check for abbreviation: word before the period
            if ch == ".":
                # Find the word before the dot
                word_start = len(current) - 2
                while word_start >= 0 and current[word_start].isalpha():
                    word_start -= 1
                word_before = current[word_start + 1:-1].lower()

                if word_before in _ABBREVS:
                    i += 1
                    continue

                # Decimal number: digit before dot
                if word_start >= 0 and current[word_start].isdigit():
                    i += 1
                    continue

                # Single letter abbreviation (like "U." "B." "D.")
                if len(word_before) == 1 and word_before.isalpha():
                    i += 1
                    continue

            # This is a sentence boundary
            sentences.append(current.strip())
            current = ""
            # Skip whitespace
            i = j
            continue

        i += 1

    if current.strip():
        sentences.append(current.strip())

    return sentences


# ---------------------------------------------------------------------------
# Recursive text splitter (core engine)
# ---------------------------------------------------------------------------

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def recursive_chunker(
    *,
    chunk_size: int = 500,
    overlap: int = 1,
    min_size: int = 50,
    separators: list[str] | None = None,
) -> Callable[[str], list[str]]:
    """Recursive text splitter — tries separators in order of priority.

    First tries to split on paragraph breaks, then line breaks,
    then sentence endings, then commas, then words.
    Overlap is in number of segments (not characters) carried to next chunk.

    Args:
        chunk_size: Target chunk size in characters.
        overlap:    Number of segments from end of previous chunk to prepend
                    to next chunk (sentence-level overlap).
        min_size:   Minimum chunk size. Tiny tail merged with previous.
        separators: Custom separator list (tried in order).

    Returns:
        ChunkerFn callable.
    """
    seps = separators if separators is not None else _DEFAULT_SEPARATORS

    def _chunk(text: str) -> list[str]:
        text = _normalize(text)
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        return _recursive_split(text, seps, chunk_size, overlap, min_size)

    return _chunk


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[str]:
    """Core recursive splitting logic."""
    # Find the best separator that actually splits this text
    best_sep = None
    for sep in separators:
        if sep in text:
            best_sep = sep
            break

    if best_sep is None:
        # No separator works — hard split
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(best_sep)
    parts = [p for p in parts if p.strip()]  # remove empties

    if not parts:
        return [text]

    # Group parts into chunks
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for part in parts:
        part_len = len(part) + len(best_sep)

        if current_len + part_len > chunk_size and current_parts:
            # Flush current
            chunk_text = best_sep.join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Overlap: carry last N parts
            if overlap > 0 and len(current_parts) >= overlap:
                current_parts = current_parts[-overlap:]
                current_len = sum(len(p) + len(best_sep) for p in current_parts)
            else:
                current_parts = []
                current_len = 0

        # If single part exceeds chunk_size, split it with finer separators
        if len(part) > chunk_size:
            if current_parts:
                chunk_text = best_sep.join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_parts = []
                current_len = 0

            remaining_seps = separators[separators.index(best_sep) + 1:]
            if remaining_seps:
                sub_chunks = _recursive_split(
                    part, remaining_seps, chunk_size, overlap, min_size
                )
                chunks.extend(sub_chunks)
            else:
                # Last resort: hard split
                chunks.extend(
                    part[i:i + chunk_size] for i in range(0, len(part), chunk_size)
                )
            continue

        current_parts.append(part)
        current_len += part_len

    # Flush remaining
    if current_parts:
        chunk_text = best_sep.join(current_parts).strip()
        if chunk_text:
            if chunks and len(chunk_text) < min_size:
                chunks[-1] = f"{chunks[-1]}{best_sep}{chunk_text}"
            else:
                chunks.append(chunk_text)

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Paragraph chunker
# ---------------------------------------------------------------------------

def paragraph_chunker(
    *,
    min_size: int = 50,
    max_size: int = 2000,
) -> Callable[[str], list[str]]:
    """Chunk by paragraphs (double newlines).

    Short paragraphs are merged up to max_size.
    Long paragraphs are split by sentences.

    Args:
        min_size: Minimum chunk size. Tiny chunks merged with neighbor.
        max_size: Maximum chunk size.

    Returns:
        ChunkerFn callable.
    """
    return recursive_chunker(
        chunk_size=max_size,
        overlap=0,
        min_size=min_size,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )


# ---------------------------------------------------------------------------
# Sentence chunker (with overlap)
# ---------------------------------------------------------------------------

def sentence_chunker(
    *,
    chunk_size: int = 500,
    overlap: int = 1,
    min_size: int = 50,
) -> Callable[[str], list[str]]:
    """Chunk by sentences with target size and overlap.

    Groups sentences into chunks of approximately chunk_size characters.
    Adjacent chunks share `overlap` sentences of context.

    Args:
        chunk_size: Target chunk size in characters.
        overlap:    Number of sentences shared between adjacent chunks.
        min_size:   Minimum chunk size. Tiny trailing chunks are merged.

    Returns:
        ChunkerFn callable.
    """
    def _chunk(text: str) -> list[str]:
        text = _normalize(text)
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        sentences = _split_sentences(text)
        if not sentences:
            return [text]
        if len(sentences) == 1:
            if len(text) <= chunk_size:
                return [text]
            # Single sentence too long — hard split by words
            return _recursive_split(
                text, [" "], chunk_size, 0, min_size
            )

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent) + 1  # +1 for space

            if current_len + sent_len > chunk_size and current:
                chunks.append(" ".join(current))

                # Sentence-level overlap
                if overlap > 0 and len(current) >= overlap:
                    current = current[-overlap:]
                    current_len = sum(len(s) + 1 for s in current)
                else:
                    current = []
                    current_len = 0

            current.append(sent)
            current_len += sent_len

        if current:
            last = " ".join(current)
            if chunks and len(last) < min_size:
                chunks[-1] = f"{chunks[-1]} {last}"
            else:
                chunks.append(last)

        return chunks if chunks else [text]

    return _chunk


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------

_MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def markdown_chunker(
    *,
    max_level: int = 2,
    max_size: int = 2000,
    include_header: bool = True,
    protect_code_blocks: bool = True,
) -> Callable[[str], list[str]]:
    """Chunk markdown by header sections.

    Splits at headers of level <= max_level. Protects code blocks
    from being split. Long sections are further split by the
    recursive chunker.

    Args:
        max_level:           Maximum header level to split at (1-6).
        max_size:            Maximum chunk size.
        include_header:      Include header line in its chunk.
        protect_code_blocks: Don't split inside ``` code blocks.

    Returns:
        ChunkerFn callable.
    """
    def _chunk(text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_size:
            return [text]

        # Protect code blocks by replacing with placeholders
        code_blocks: list[str] = []
        protected = text
        if protect_code_blocks:
            def _replace_code(m: re.Match) -> str:
                idx = len(code_blocks)
                code_blocks.append(m.group(0))
                return f"\x00CODE{idx}\x00"
            protected = _CODE_BLOCK_RE.sub(_replace_code, text)

        # Find splitting headers
        splits: list[tuple[int, str]] = []
        for match in _MARKDOWN_HEADER_RE.finditer(protected):
            level = len(match.group(1))
            if level <= max_level:
                splits.append((match.start(), match.group(0)))

        if not splits:
            # No headers — fall back to recursive chunker
            fallback = recursive_chunker(
                chunk_size=max_size, overlap=0, min_size=50,
            )
            return fallback(text)

        # Build sections
        sections: list[str] = []

        # Preamble before first header
        if splits[0][0] > 0:
            preamble = protected[:splits[0][0]].strip()
            if preamble:
                sections.append(preamble)

        for i, (pos, header) in enumerate(splits):
            end = splits[i + 1][0] if i + 1 < len(splits) else len(protected)
            section = protected[pos:end].strip()

            if not include_header:
                section = section[len(header):].strip()

            if section:
                sections.append(section)

        # Restore code blocks and split oversized sections
        chunks: list[str] = []
        sub_chunker = recursive_chunker(
            chunk_size=max_size, overlap=0, min_size=50,
        )
        for section in sections:
            # Restore code block placeholders
            if code_blocks:
                for idx, block in enumerate(code_blocks):
                    section = section.replace(f"\x00CODE{idx}\x00", block)

            if len(section) > max_size:
                chunks.extend(sub_chunker(section))
            else:
                chunks.append(section)

        return chunks if chunks else [text]

    return _chunk
