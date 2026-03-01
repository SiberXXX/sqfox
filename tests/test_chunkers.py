"""Tests for sqfox chunkers: paragraph, sentence, markdown, recursive."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from sqfox.chunkers import (
    html_to_text,
    paragraph_chunker,
    sentence_chunker,
    markdown_chunker,
    recursive_chunker,
    _split_sentences,
    _normalize,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_collapses_newlines(self):
        assert _normalize("a\n\n\n\nb") == "a\n\nb"

    def test_unifies_line_endings(self):
        assert _normalize("a\r\nb\rc") == "a\nb\nc"

    def test_collapses_spaces(self):
        assert _normalize("a   b\tc") == "a b c"

    def test_strips(self):
        assert _normalize("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_basic_english(self):
        text = "Hello world. This is a test. Final sentence."
        sentences = _split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."

    def test_basic_russian(self):
        text = "Привет мир. Это тест. Последнее предложение."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_question_exclamation(self):
        text = "What is this? It's amazing! Really."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_preserves_abbreviations(self):
        text = "Dr. Smith went to the meeting. He was late."
        sentences = _split_sentences(text)
        # "Dr." should not cause split, but ". He" should
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]

    def test_preserves_single_letter_abbrev(self):
        text = "Washington D.C. is the capital. It is big."
        sentences = _split_sentences(text)
        assert len(sentences) == 2
        assert "D.C." in sentences[0]

    def test_preserves_decimals(self):
        text = "Temperature is 36.6 degrees. Normal range."
        sentences = _split_sentences(text)
        assert len(sentences) == 2
        assert "36.6" in sentences[0]

    def test_ellipsis(self):
        text = "Hmm... I don't know. Maybe later."
        sentences = _split_sentences(text)
        assert len(sentences) == 2
        assert "..." in sentences[0]

    def test_empty(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_single_sentence(self):
        assert len(_split_sentences("Just one sentence.")) == 1

    def test_russian_abbreviations(self):
        text = "Проф. Иванов прибыл в г. Москву. Он опоздал."
        sentences = _split_sentences(text)
        assert len(sentences) == 2
        assert "Проф. Иванов" in sentences[0]


# ---------------------------------------------------------------------------
# Recursive chunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_short_text_not_split(self):
        chunker = recursive_chunker(chunk_size=1000)
        assert chunker("Short text.") == ["Short text."]

    def test_splits_paragraphs_first(self):
        text = "Para one content here.\n\nPara two content here.\n\nPara three content here."
        chunker = recursive_chunker(chunk_size=30, overlap=0, min_size=5)
        chunks = chunker(text)
        assert len(chunks) >= 3

    def test_falls_to_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunker = recursive_chunker(chunk_size=50, overlap=0, min_size=5)
        chunks = chunker(text)
        assert len(chunks) >= 2

    def test_hard_split_last_resort(self):
        text = "A" * 3000
        chunker = recursive_chunker(chunk_size=1000)
        chunks = chunker(text)
        assert all(len(c) <= 1000 for c in chunks)
        assert "".join(chunks) == text

    def test_overlap_carries_segments(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
        chunker = recursive_chunker(chunk_size=35, overlap=1, min_size=5)
        chunks = chunker(text)
        if len(chunks) >= 2:
            # Overlap: last part of chunk N should appear in chunk N+1
            assert any(
                part in chunks[1]
                for part in chunks[0].split("\n\n")[-1:]
            )

    def test_empty(self):
        chunker = recursive_chunker()
        assert chunker("") == []
        assert chunker("   ") == []


# ---------------------------------------------------------------------------
# Paragraph chunker
# ---------------------------------------------------------------------------

class TestParagraphChunker:
    def test_basic_split(self):
        text = "First paragraph with enough text.\n\nSecond paragraph also enough.\n\nThird paragraph here too."
        chunker = paragraph_chunker(min_size=10, max_size=40)
        chunks = chunker(text)
        assert len(chunks) >= 3

    def test_merges_within_max_size(self):
        text = "Short.\n\nAlso short.\n\nTiny."
        chunker = paragraph_chunker(min_size=5, max_size=2000)
        chunks = chunker(text)
        # All fit in one chunk
        assert len(chunks) == 1

    def test_splits_long_paragraphs(self):
        long_para = "This is a sentence. " * 100
        text = f"Short intro.\n\n{long_para}\n\nShort outro."
        chunker = paragraph_chunker(max_size=500)
        chunks = chunker(text)
        assert len(chunks) > 1

    def test_empty(self):
        chunker = paragraph_chunker()
        assert chunker("") == []

    def test_no_double_newlines(self):
        text = "Single block of text without paragraph breaks."
        chunker = paragraph_chunker()
        chunks = chunker(text)
        assert len(chunks) == 1

    def test_hard_split_no_sentences(self):
        text = "A" * 3000
        chunker = paragraph_chunker(max_size=1000)
        chunks = chunker(text)
        assert all(len(c) <= 1000 for c in chunks)


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------

class TestSentenceChunker:
    def test_basic_chunking(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunker = sentence_chunker(chunk_size=50, overlap=0, min_size=10)
        chunks = chunker(text)
        assert len(chunks) >= 2
        full = " ".join(chunks)
        assert "First" in full
        assert "Fifth" in full

    def test_sentence_overlap(self):
        sentences = [f"Sentence number {i} is here." for i in range(10)]
        text = " ".join(sentences)
        chunker = sentence_chunker(chunk_size=100, overlap=1, min_size=10)
        chunks = chunker(text)
        if len(chunks) >= 2:
            # Last sentence of chunk 0 should appear in chunk 1
            last_sent_0 = chunks[0].rsplit(".", 2)[-2] if "." in chunks[0] else ""
            if last_sent_0:
                assert last_sent_0.strip() in chunks[1]

    def test_single_sentence(self):
        chunker = sentence_chunker(chunk_size=1000)
        assert chunker("Just one sentence.") == ["Just one sentence."]

    def test_empty(self):
        chunker = sentence_chunker()
        assert chunker("") == []

    def test_tiny_tail_merged(self):
        text = "A very long first sentence that fills most of the chunk size easily. B."
        chunker = sentence_chunker(chunk_size=100, overlap=0, min_size=20)
        chunks = chunker(text)
        # "B." is < min_size, merged with previous
        assert len(chunks) == 1

    def test_long_single_sentence(self):
        text = "word " * 200  # ~1000 chars, no sentence ending
        chunker = sentence_chunker(chunk_size=100, min_size=10)
        chunks = chunker(text.strip())
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------

class TestMarkdownChunker:
    def test_basic_split(self):
        text = """# Title

Intro text here.

## Section One

Content of section one.

## Section Two

Content of section two.
"""
        chunker = markdown_chunker(max_level=2, max_size=50)
        chunks = chunker(text)
        assert len(chunks) >= 3

    def test_includes_header(self):
        text = "# Title\n\nContent here."
        chunker = markdown_chunker(include_header=True)
        chunks = chunker(text)
        assert any("# Title" in c for c in chunks)

    def test_excludes_header(self):
        text = "# Title\n\nContent here that is long enough to trigger splitting when max is low."
        chunker = markdown_chunker(include_header=False, max_size=50)
        chunks = chunker(text)
        assert all("# Title" not in c for c in chunks)
        assert any("Content" in c for c in chunks)

    def test_respects_max_level(self):
        text = """# H1

Content under H1.

## H2

Content under H2.

### H3

Content under H3.
"""
        chunker = markdown_chunker(max_level=2)
        chunks = chunker(text)
        # H3 should NOT cause a split — stays with H2 section
        h3_separate = any(c.strip().startswith("### H3") and len(c.strip()) < 30 for c in chunks)
        assert not h3_separate

    def test_no_headers_falls_back(self):
        text = "A long text. " * 200  # ~2600 chars, no headers
        chunker = markdown_chunker(max_size=500)
        chunks = chunker(text)
        assert len(chunks) >= 2

    def test_splits_long_sections(self):
        long_content = "This is a sentence. " * 100
        text = f"# Title\n\n{long_content}"
        chunker = markdown_chunker(max_size=500)
        chunks = chunker(text)
        assert len(chunks) > 1

    def test_preamble_before_first_header(self):
        text = "Some preamble text.\n\n# First Header\n\nContent of the first section is here."
        chunker = markdown_chunker(max_size=40)
        chunks = chunker(text)
        assert chunks[0] == "Some preamble text."

    def test_empty(self):
        chunker = markdown_chunker()
        assert chunker("") == []

    def test_russian_markdown(self):
        text = """# Заголовок

Введение к документу.

## Первый раздел

Содержимое первого раздела.

## Второй раздел

Содержимое второго раздела.
"""
        chunker = markdown_chunker(max_size=50)
        chunks = chunker(text)
        assert len(chunks) >= 3

    def test_protects_code_blocks(self):
        text = """# Title

Some text.

```python
def hello():
    # This has periods. And questions? And exclamations!
    print("Hello world.")
```

More text after code.
"""
        chunker = markdown_chunker(max_size=5000)
        chunks = chunker(text)
        # Code block should not be split
        full = "\n".join(chunks)
        assert '```python' in full
        assert 'print("Hello world.")' in full

    def test_code_block_not_split(self):
        code = "x = 1\n" * 50  # long code block
        text = f"# Title\n\n```python\n{code}```\n\nAfter code."
        chunker = markdown_chunker(max_size=100, protect_code_blocks=True)
        chunks = chunker(text)
        # Code block should stay intact even if > max_size
        any_has_full_code = any("x = 1" in c and "```" in c for c in chunks)
        assert any_has_full_code


# ---------------------------------------------------------------------------
# HTML preprocessor
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_strips_tags(self):
        html = "<p>Hello <b>world</b></p>"
        text = html_to_text(html)
        assert "Hello world" in text
        assert "<" not in text

    def test_block_tags_to_paragraphs(self):
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        text = html_to_text(html)
        assert "\n\n" in text
        assert "First paragraph." in text
        assert "Second paragraph." in text

    def test_br_to_newline(self):
        html = "Line one.<br>Line two.<br/>Line three."
        text = html_to_text(html)
        assert "Line one." in text
        assert "Line two." in text

    def test_removes_script(self):
        html = '<p>Text</p><script>alert("xss")</script><p>More</p>'
        text = html_to_text(html)
        assert "Text" in text
        assert "More" in text
        assert "alert" not in text
        assert "script" not in text

    def test_removes_style(self):
        html = '<style>body{color:red}</style><p>Content</p>'
        text = html_to_text(html)
        assert "Content" in text
        assert "color" not in text

    def test_removes_comments(self):
        html = '<!-- comment --><p>Visible</p>'
        text = html_to_text(html)
        assert "Visible" in text
        assert "comment" not in text

    def test_decodes_entities(self):
        html = "<p>Tom &amp; Jerry &lt;3 &quot;fun&quot;</p>"
        text = html_to_text(html)
        assert "Tom & Jerry" in text
        assert '<3' in text
        assert '"fun"' in text

    def test_headings_become_paragraphs(self):
        html = "<h1>Title</h1><p>Content.</p><h2>Subtitle</h2><p>More.</p>"
        text = html_to_text(html)
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        assert len(parts) >= 4

    def test_list_items(self):
        html = "<ul><li>One</li><li>Two</li><li>Three</li></ul>"
        text = html_to_text(html)
        assert "One" in text
        assert "Two" in text
        assert "Three" in text

    def test_table_rows(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        text = html_to_text(html)
        assert "A" in text
        assert "C" in text

    def test_nested_tags(self):
        html = '<div><p>Outer <span class="x">inner <em>deep</em></span> text.</p></div>'
        text = html_to_text(html)
        assert "Outer inner deep text." in text

    def test_empty(self):
        assert html_to_text("") == ""
        assert html_to_text("   ") == ""

    def test_plain_text_passthrough(self):
        plain = "Just plain text without any HTML."
        assert html_to_text(plain) == plain

    def test_real_world_snippet(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test</title></head>
        <body>
            <header><nav>Menu</nav></header>
            <main>
                <article>
                    <h1>Article Title</h1>
                    <p>First paragraph with <a href="#">link</a>.</p>
                    <p>Second paragraph with <strong>bold</strong> text.</p>
                    <ul>
                        <li>Item one</li>
                        <li>Item two</li>
                    </ul>
                </article>
            </main>
            <footer>Copyright 2026</footer>
            <script>var x = 1;</script>
        </body>
        </html>
        """
        text = html_to_text(html)
        assert "Article Title" in text
        assert "First paragraph with link." in text
        assert "bold" in text
        assert "Item one" in text
        assert "var x" not in text
        assert "<" not in text

    def test_works_with_chunker(self):
        """html_to_text output is compatible with chunkers."""
        html = "<h1>Title</h1><p>Content one.</p><p>Content two.</p>"
        text = html_to_text(html)
        chunker = paragraph_chunker(min_size=5, max_size=30)
        chunks = chunker(text)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) for c in chunks)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_paragraph_callable(self):
        c = paragraph_chunker()
        assert callable(c)
        assert isinstance(c("test"), list)

    def test_sentence_callable(self):
        c = sentence_chunker()
        assert callable(c)
        assert isinstance(c("Test sentence. Another."), list)

    def test_markdown_callable(self):
        c = markdown_chunker()
        assert callable(c)
        assert isinstance(c("# T\n\nC."), list)

    def test_recursive_callable(self):
        c = recursive_chunker()
        assert callable(c)
        assert isinstance(c("text"), list)

    def test_all_return_strings(self):
        for factory in [paragraph_chunker, sentence_chunker, markdown_chunker, recursive_chunker]:
            c = factory()
            result = c("Some text. More text.\n\nAnother paragraph.")
            assert all(isinstance(s, str) for s in result)
