# ============================================================
# tests/unit/test_preprocessing.py
#
# Tests for all data_preprocessing sub-modules:
# - document_parser  (text cleaning)
# - structure_analyzer (heading/table detection)
# - summary_generator (LLM and extractive)
# - keyword_extractor
# - question_generator
# ============================================================

import pytest
from unittest.mock import MagicMock


# ── DocumentParser ────────────────────────────────────────────

class TestDocumentParser:

    def setup_method(self):
        from data_preprocessing.restructuring.document_parser import DocumentParser
        self.parser = DocumentParser()

    def _parse(self, text):
        return self.parser.parse({"content": text, "source": "test.txt"})

    def test_normalises_multiple_spaces(self):
        """Multiple consecutive spaces should be collapsed to one."""
        result = self._parse("Hello     world   here.")
        assert "     " not in result["content"]
        assert "Hello world here." in result["content"]

    def test_removes_excess_newlines(self):
        """More than 2 consecutive newlines should become exactly 2."""
        result = self._parse("Para one.\n\n\n\n\nPara two.")
        assert "\n\n\n" not in result["content"]

    def test_fixes_hyphenated_line_breaks(self):
        """'sen-\\ntence' should become 'sentence'."""
        result = self._parse("This is a sen-\ntence here.")
        assert "sentence" in result["content"]
        assert "sen-\ntence" not in result["content"]

    def test_normalises_curly_quotes(self):
        """Unicode curly quotes should become straight ASCII quotes."""
        result = self._parse("\u2018Hello\u2019 and \u201cWorld\u201d")
        assert "'" in result["content"] or '"' in result["content"]

    def test_removes_page_number_lines(self):
        """Standalone page numbers like 'Page 3 of 10' should be removed."""
        result = self._parse("Content here.\nPage 3 of 10\nMore content.")
        assert "Page 3 of 10" not in result["content"]

    def test_preserves_original_length_field(self):
        """Result should record original and cleaned character lengths."""
        result = self._parse("Short text.")
        assert "original_length" in result
        assert "cleaned_length" in result

    def test_empty_content_returns_gracefully(self):
        """Empty string content should not crash."""
        result = self._parse("")
        assert result["content"] == "" or result["content"] is not None

    def test_encoding_artifacts_replaced(self):
        """Common PDF encoding artifacts like â€™ should be replaced."""
        result = self._parse("It\u00e2\u0080\u0099s a test.")
        # After fix_encoding_issues runs, garbled chars should be gone
        assert "\u00e2\u0080\u0099" not in result["content"]


# ── StructureAnalyzer ─────────────────────────────────────────

class TestStructureAnalyzer:

    def setup_method(self):
        from data_preprocessing.restructuring.structure_analyzer import StructureAnalyzer
        self.analyzer = StructureAnalyzer()

    def _analyze(self, text):
        return self.analyzer.analyze({"content": text, "source": "test.txt"})

    def test_detects_markdown_headings(self):
        """Markdown # headings should be detected at correct levels."""
        text = "# Introduction\n\nSome text.\n\n## Background\n\nMore text."
        result = self._analyze(text)

        headings = result["headings"]
        levels = [h["level"] for h in headings]
        assert 1 in levels, "H1 heading should be detected"
        assert 2 in levels, "H2 heading should be detected"

    def test_detects_markdown_tables(self):
        """Pipe-separated tables should be identified."""
        text = "Before.\n\n| Col A | Col B |\n|-------|-------|\n| 1     | 2     |\n\nAfter."
        result = self._analyze(text)

        assert len(result["tables"]) >= 1, "Should detect one table"

    def test_detects_bullet_lists(self):
        """Bullet point lists should be detected."""
        text = "Intro.\n- Item one\n- Item two\n- Item three\nEnd."
        result = self._analyze(text)
        assert len(result["lists"]) >= 1

    def test_numbered_list_detected(self):
        """Numbered lists (1. Item) should be detected."""
        text = "Steps:\n1. First step\n2. Second step\n3. Third step\nDone."
        result = self._analyze(text)
        assert len(result["lists"]) >= 1

    def test_sections_built_from_headings(self):
        """Sections list should be built from detected headings."""
        text = "# Chapter 1\nChapter one text.\n\n# Chapter 2\nChapter two text."
        result = self._analyze(text)

        assert len(result["sections"]) >= 1

    def test_structure_detected_flag_set(self):
        """Result should have structure_detected=True."""
        result = self._analyze("Some text without structure.")
        assert result.get("structure_detected") is True

    def test_no_headings_in_plain_text(self):
        """Plain paragraph text should produce zero headings."""
        text = "This is just a paragraph. No headings here at all."
        result = self._analyze(text)
        assert len(result["headings"]) == 0

    def test_caps_heading_detected(self):
        """Short ALL-CAPS lines should be treated as headings."""
        text = "INTRODUCTION\n\nSome introductory text follows here."
        result = self._analyze(text)
        assert len(result["headings"]) >= 1


# ── SummaryGenerator ──────────────────────────────────────────

class TestSummaryGenerator:

    def test_extractive_fallback_with_no_llm(self):
        """Without LLM client, returns first 1-2 sentences of the text."""
        from data_preprocessing.metadata_creation.summary_generator import SummaryGenerator

        gen = SummaryGenerator(llm_client=None)
        text = "Python is a programming language. It is used for web development. And data science."
        summary = gen.generate(text)

        assert len(summary) > 0
        # Should start with the beginning of the text
        assert "Python" in summary

    def test_empty_text_returns_empty_string(self):
        """Empty input → empty string output."""
        from data_preprocessing.metadata_creation.summary_generator import SummaryGenerator

        gen = SummaryGenerator(llm_client=None)
        assert gen.generate("") == ""
        assert gen.generate("   ") == ""

    def test_short_text_is_returned_as_is(self):
        """Text shorter than max_length should be returned without truncation."""
        from data_preprocessing.metadata_creation.summary_generator import SummaryGenerator

        gen = SummaryGenerator(llm_client=None)
        short = "Short text."
        result = gen.generate(short, max_length=200)
        assert "Short" in result

    def test_llm_summary_calls_client(self):
        """When LLM client is provided, it should be called."""
        from data_preprocessing.metadata_creation.summary_generator import SummaryGenerator

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "A concise summary."
        gen = SummaryGenerator(llm_client=mock_client)
        summary = gen.generate("Some long text about an important topic.")

        mock_client.chat.completions.create.assert_called_once()
        assert "summary" in summary.lower() or len(summary) > 0

    def test_falls_back_to_extractive_on_llm_error(self):
        """If LLM raises an exception, fall back to extractive summary."""
        from data_preprocessing.metadata_creation.summary_generator import SummaryGenerator

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        gen = SummaryGenerator(llm_client=mock_client)

        text = "This is some content. It has multiple sentences."
        result = gen.generate(text)

        # Should still return something (extractive fallback)
        assert len(result) > 0


# ── KeywordExtractor ──────────────────────────────────────────

class TestKeywordExtractor:

    def setup_method(self):
        from data_preprocessing.metadata_creation.keyword_extractor import KeywordExtractor
        self.extractor = KeywordExtractor(max_keywords=10)

    def test_extracts_keywords_from_text(self):
        """Common important words should be extracted."""
        text = "Python programming language used for machine learning and data science projects."
        keywords = self.extractor.extract(text)

        assert len(keywords) > 0
        # 'python' should rank highly (appears once, not a stop word)
        assert "python" in keywords

    def test_stop_words_not_included(self):
        """Common stop words (the, and, is) should not appear in keywords."""
        text = "The quick brown fox jumps over the lazy dog and the cat."
        keywords = self.extractor.extract(text)

        stop_words = {"the", "and", "is", "a", "an"}
        for kw in keywords:
            assert kw not in stop_words, f"Stop word '{kw}' found in keywords"

    def test_empty_text_returns_empty_list(self):
        """Empty input returns empty list."""
        assert self.extractor.extract("") == []
        assert self.extractor.extract("   ") == []

    def test_respects_max_keywords_limit(self):
        """Should never return more than max_keywords."""
        text = " ".join(f"uniqueword{i}" for i in range(50))
        keywords = self.extractor.extract(text)
        assert len(keywords) <= 10

    def test_extract_with_scores_returns_dict(self):
        """extract_with_scores should return a dict with float scores."""
        text = "machine learning deep learning neural networks python tensorflow"
        scored = self.extractor.extract_with_scores(text)

        assert isinstance(scored, dict)
        for word, score in scored.items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_most_frequent_word_scores_highest(self):
        """The most repeated word should have the highest score."""
        text = "python python python machine learning"
        scored = self.extractor.extract_with_scores(text)

        if scored:
            top_word = max(scored, key=scored.get)
            assert top_word == "python"


# ── QuestionGenerator ─────────────────────────────────────────

class TestQuestionGenerator:

    def test_returns_empty_without_llm(self):
        """Without an LLM client, returns empty list."""
        from data_preprocessing.metadata_creation.question_generator import QuestionGenerator

        gen = QuestionGenerator(llm_client=None)
        questions = gen.generate("Python is a high-level programming language.")
        assert questions == []

    def test_empty_text_returns_empty(self):
        """Empty input returns empty list."""
        from data_preprocessing.metadata_creation.question_generator import QuestionGenerator

        gen = QuestionGenerator(llm_client=None)
        assert gen.generate("") == []
        assert gen.generate("  ") == []

    def test_llm_questions_are_questions(self):
        """LLM-generated questions should end with '?'."""
        from data_preprocessing.metadata_creation.question_generator import QuestionGenerator

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "What is Python?\nWhere is Python used?\nWhy is Python popular?"

        gen = QuestionGenerator(llm_client=mock_client, num_questions=3)
        questions = gen.generate("Python is a popular programming language.")

        assert len(questions) > 0
        for q in questions:
            assert q.endswith("?"), f"Expected question mark at end: '{q}'"

    def test_respects_num_questions_limit(self):
        """Should not return more than num_questions."""
        from data_preprocessing.metadata_creation.question_generator import QuestionGenerator

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "Q1?\nQ2?\nQ3?\nQ4?\nQ5?\nQ6?"

        gen = QuestionGenerator(llm_client=mock_client, num_questions=3)
        questions = gen.generate("Some text here.")
        assert len(questions) <= 3

    def test_falls_back_gracefully_on_llm_error(self):
        """If LLM raises, returns empty list without crashing."""
        from data_preprocessing.metadata_creation.question_generator import QuestionGenerator

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        gen = QuestionGenerator(llm_client=mock_client)
        questions = gen.generate("Some text here.")
        assert questions == []
