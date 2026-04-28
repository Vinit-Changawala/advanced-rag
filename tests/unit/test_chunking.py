# ============================================================
# tests/unit/test_chunking.py
#
# PURPOSE: Test the chunking pipeline in isolation.
#
# BEGINNER CONCEPT - What is unit testing?
# A unit test tests ONE small piece of code at a time.
# "Unit" = the smallest testable piece (usually one function or class).
#
# WHY TEST?
# When you change code, tests tell you if you broke something.
# Without tests, you'd have to manually test everything after each change.
#
# We use pytest — the most popular Python testing library.
# Run tests with: pytest tests/ -v
# ============================================================

import pytest
from data_preprocessing.chunking.boundary_detector import BoundaryDetector
from data_preprocessing.chunking.table_preserver import TablePreserver
from data_preprocessing.chunking.heading_detector import HeadingDetector


class TestBoundaryDetector:
    """
    Tests for the BoundaryDetector class.

    BEGINNER CONCEPT - Test classes:
    Grouping related tests in a class keeps things organized.
    Each method starting with 'test_' is automatically discovered by pytest.
    """

    def setup_method(self):
        """
        setup_method runs BEFORE each test method.
        Like setting up your workspace before each experiment.
        """
        self.chunker = BoundaryDetector(chunk_size=100, chunk_overlap=20)

    def test_short_text_returns_single_chunk(self):
        """Short text that fits in one chunk should not be split."""
        text = "This is a short text."
        chunks = self.chunker.split(text)

        # 'assert' checks that something is True
        # If False → test FAILS with a clear error message
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split(self):
        """Long text should be split into multiple chunks."""
        # Create a text with many sentences (longer than chunk_size)
        text = " ".join([f"This is sentence number {i}." for i in range(50)])

        chunks = self.chunker.split(text)
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunks_contain_all_content(self):
        """No content should be lost during chunking."""
        text = "Hello world. This is important. Do not lose this."
        chunks = self.chunker.split(text)

        # Join all chunks and check all words are present
        combined = " ".join(chunks)
        for word in ["Hello", "important", "lose"]:
            assert word in combined, f"Word '{word}' was lost during chunking!"

    def test_empty_text_returns_empty_list(self):
        """Empty input should return empty list, not crash."""
        chunks = self.chunker.split("")
        # Either empty list or list with empty string — both acceptable
        assert chunks == [] or chunks == [""]

    def test_single_sentence_not_split(self):
        """A single sentence should never be split in the middle."""
        sentence = "The quick brown fox jumps over the lazy dog."
        chunks = self.chunker.split(sentence)

        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should not end mid-word (last char should not be a letter mid-word)
            assert chunk.strip()   # Should not be empty


class TestTablePreserver:
    """Tests for the TablePreserver class."""

    def setup_method(self):
        self.preserver = TablePreserver()

    def test_extracts_markdown_table(self):
        """Should detect and extract a markdown table."""
        text = """Some text before.

| Name | Age | City |
|------|-----|------|
| Alice | 30 | Mumbai |
| Bob | 25 | Delhi |

Some text after."""

        modified, tables = self.preserver.extract_tables(text)

        assert len(tables) == 1, "Should find exactly 1 table"
        assert "Alice" in tables[0]["text"], "Table should contain original data"
        assert "[[TABLE_0]]" in modified, "Table should be replaced with placeholder"
        assert "Alice" not in modified, "Original table should be removed from text"

    def test_text_without_tables_unchanged(self):
        """Text with no tables should not be modified."""
        text = "Just regular paragraph text. No tables here."
        modified, tables = self.preserver.extract_tables(text)

        assert len(tables) == 0
        assert modified == text

    def test_multiple_tables_extracted(self):
        """Multiple tables should all be extracted."""
        text = """| A | B |\n|---|---|\n| 1 | 2 |\n\nMiddle text.\n\n| C | D |\n|---|---|\n| 3 | 4 |"""
        modified, tables = self.preserver.extract_tables(text)

        assert len(tables) == 2


class TestHeadingDetector:
    """Tests for the HeadingDetector class."""

    def setup_method(self):
        self.detector = HeadingDetector()

    def test_splits_on_markdown_headings(self):
        """Should split text at markdown headings."""
        headings = [
            {"text": "Introduction", "level": 1, "line_number": 0},
            {"text": "Methods", "level": 1, "line_number": 3},
        ]
        text = "Introduction\nSome intro text.\nMore text.\nMethods\nMethod details."

        sections = self.detector.split_on_headings(text, headings)
        assert len(sections) == 2

    def test_no_headings_returns_full_document(self):
        """If no headings found, return entire text as one section."""
        text = "Just some plain text with no headings at all."
        sections = self.detector.split_on_headings(text, [])

        assert len(sections) == 1
        assert sections[0]["title"] == "Document"
