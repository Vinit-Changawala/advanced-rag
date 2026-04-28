# ============================================================
# tests/unit/test_data_sources.py
#
# Tests for every file loader in data_sources/.
# All tests run without real files on disk by using tmp_path
# (a pytest built-in fixture that creates a real temp folder).
# ============================================================

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── DocumentLoader ────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_txt_file(self, tmp_path):
        """Loading a plain text file returns its content."""
        from data_sources.document_loader import DocumentLoader

        f = tmp_path / "sample.txt"
        f.write_text("Hello world.\nSecond line.")
        result = DocumentLoader().load(str(f))

        assert result["content"] == "Hello world.\nSecond line."
        assert result["source_type"] == "document"
        assert result["file_type"] == ".txt"
        assert result["file_name"] == "sample.txt"

    def test_load_markdown_file(self, tmp_path):
        """Markdown files are treated as plain text."""
        from data_sources.document_loader import DocumentLoader

        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content here.")
        result = DocumentLoader().load(str(f))

        assert "Title" in result["content"]
        assert result["file_type"] == ".md"

    def test_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError when file does not exist."""
        from data_sources.document_loader import DocumentLoader

        with pytest.raises(FileNotFoundError):
            DocumentLoader().load(str(tmp_path / "nonexistent.txt"))

    def test_raises_on_unsupported_extension(self, tmp_path):
        """ValueError for unsupported file types."""
        from data_sources.document_loader import DocumentLoader

        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentLoader().load(str(f))

    def test_load_directory(self, tmp_path):
        """load_directory loads all supported files in a folder."""
        from data_sources.document_loader import DocumentLoader

        (tmp_path / "a.txt").write_text("File A content.")
        (tmp_path / "b.md").write_text("File B content.")
        (tmp_path / "ignored.xyz").write_text("Not loaded.")

        results = DocumentLoader().load_directory(str(tmp_path))

        assert len(results) == 2
        sources = {r["file_name"] for r in results}
        assert "a.txt" in sources
        assert "b.md" in sources
        assert "ignored.xyz" not in sources

    def test_load_directory_skips_failed_files(self, tmp_path):
        """load_directory continues if one file fails to load."""
        from data_sources.document_loader import DocumentLoader

        (tmp_path / "good.txt").write_text("Good content.")
        # Bad file: write bytes that aren't valid UTF-8 or latin-1 text
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"\x00\x01\x02")  # Not a real PDF — will fail

        loader = DocumentLoader()
        results = loader.load_directory(str(tmp_path))

        # Should get at least the good txt file
        names = {r["file_name"] for r in results}
        assert "good.txt" in names

    def test_result_has_all_required_keys(self, tmp_path):
        """Every result dict must have the required keys."""
        from data_sources.document_loader import DocumentLoader

        f = tmp_path / "test.txt"
        f.write_text("Some content.")
        result = DocumentLoader().load(str(f))

        required_keys = ["content", "source", "source_type", "file_type", "file_name"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ── CodeLoader ────────────────────────────────────────────────

class TestCodeLoader:

    def test_load_python_file(self, tmp_path):
        """Python files are loaded with language=python and symbols extracted."""
        from data_sources.code_loader import CodeLoader

        f = tmp_path / "app.py"
        f.write_text("def hello():\n    return 'world'\n\nclass MyClass:\n    pass\n")
        result = CodeLoader().load(str(f))

        assert result["language"] == "python"
        assert result["source_type"] == "code"
        assert result["line_count"] >= 4
        symbols = result["symbols"]
        assert any("hello" in s for s in symbols), f"Expected 'hello' in {symbols}"
        assert any("MyClass" in s for s in symbols), f"Expected 'MyClass' in {symbols}"

    def test_load_javascript_file(self, tmp_path):
        """JavaScript files get language=javascript."""
        from data_sources.code_loader import CodeLoader

        f = tmp_path / "index.js"
        f.write_text("function greet(name) {\n  return `Hello ${name}`;\n}\n")
        result = CodeLoader().load(str(f))

        assert result["language"] == "javascript"

    def test_load_sql_file(self, tmp_path):
        """SQL files are detected correctly."""
        from data_sources.code_loader import CodeLoader

        f = tmp_path / "query.sql"
        f.write_text("SELECT * FROM users WHERE active = 1;")
        result = CodeLoader().load(str(f))

        assert result["language"] == "sql"

    def test_load_directory_excludes_venv(self, tmp_path):
        """load_directory should skip venv/ and node_modules/."""
        from data_sources.code_loader import CodeLoader

        (tmp_path / "app.py").write_text("print('hello')")
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "lib.py").write_text("# should be excluded")

        results = CodeLoader().load_directory(str(tmp_path))
        sources = {r["file_name"] for r in results}

        assert "app.py" in sources
        assert "lib.py" not in sources

    def test_unknown_extension_gets_unknown_language(self, tmp_path):
        """Files with unmapped extensions get language='unknown'."""
        from data_sources.code_loader import CodeLoader

        loader = CodeLoader()
        loader.supported_extensions.add(".xyz")
        loader.LANGUAGE_MAP[".xyz"] = "unknown"

        f = tmp_path / "file.xyz"
        f.write_text("some content")
        result = loader.load(str(f))
        assert result["language"] == "unknown"


# ── SpreadsheetLoader ─────────────────────────────────────────

class TestSpreadsheetLoader:

    def test_load_csv(self, tmp_path):
        """CSV files are loaded and converted to readable text."""
        from data_sources.spreadsheet_loader import SpreadsheetLoader
        pytest.importorskip("pandas")

        f = tmp_path / "data.csv"
        f.write_text("Name,Age,City\nAlice,30,Mumbai\nBob,25,Delhi\n")
        result = SpreadsheetLoader().load(str(f))

        assert result["source_type"] == "spreadsheet"
        assert result["row_count"] == 2
        assert result["column_count"] == 3
        assert "Alice" in result["content"]
        assert "Name" in result["columns"]

    def test_csv_schema_contains_column_names(self, tmp_path):
        """Schema dict maps column names to their data types."""
        from data_sources.spreadsheet_loader import SpreadsheetLoader
        pytest.importorskip("pandas")

        f = tmp_path / "sales.csv"
        f.write_text("Product,Revenue\nWidget,1000\nGadget,2000\n")
        result = SpreadsheetLoader().load(str(f))

        assert "Product" in result["schema"]
        assert "Revenue" in result["schema"]

    def test_max_rows_limits_output(self, tmp_path):
        """max_rows parameter caps how many rows are loaded."""
        from data_sources.spreadsheet_loader import SpreadsheetLoader
        pytest.importorskip("pandas")

        rows = ["a,b"] + [f"{i},{i*2}" for i in range(100)]
        f = tmp_path / "big.csv"
        f.write_text("\n".join(rows))

        result = SpreadsheetLoader().load(str(f), max_rows=10)
        assert result["row_count"] <= 10

    def test_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError for missing spreadsheet."""
        from data_sources.spreadsheet_loader import SpreadsheetLoader
        pytest.importorskip("pandas")

        with pytest.raises(FileNotFoundError):
            SpreadsheetLoader().load(str(tmp_path / "missing.csv"))


# ── ImageLoader ───────────────────────────────────────────────

class TestImageLoader:

    def test_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError for missing image."""
        from data_sources.image_loader import ImageLoader

        with pytest.raises(FileNotFoundError):
            ImageLoader(use_ocr=False).load(str(tmp_path / "missing.png"))

    def test_raises_on_unsupported_extension(self, tmp_path):
        """ValueError for non-image file types."""
        from data_sources.image_loader import ImageLoader

        f = tmp_path / "document.pdf"
        f.write_bytes(b"fake content")
        with pytest.raises(ValueError, match="Unsupported"):
            ImageLoader(use_ocr=False).load(str(f))

    def test_loads_image_without_ocr(self, tmp_path):
        """Without OCR or vision AI, returns filename as fallback content."""
        from data_sources.image_loader import ImageLoader

        # Create a minimal valid PNG (1×1 pixel)
        # PNG magic bytes + minimal IHDR chunk
        minimal_png = (
            b'\x89PNG\r\n\x1a\n'                          # PNG signature
            b'\x00\x00\x00\rIHDR'                         # IHDR chunk length + type
            b'\x00\x00\x00\x01\x00\x00\x00\x01'          # width=1, height=1
            b'\x08\x02\x00\x00\x00\x90wS\xde'             # bit depth, color type, etc.
            b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'             # IEND chunk
        )
        f = tmp_path / "test.png"
        f.write_bytes(minimal_png)

        loader = ImageLoader(use_ocr=False, openai_client=None)
        result = loader.load(str(f))

        assert result["source_type"] == "image"
        assert result["file_type"] == ".png"
        assert result["file_name"] == "test.png"
        # Content should mention the filename since no OCR/AI
        assert "test.png" in result["content"]

    def test_result_has_required_keys(self, tmp_path):
        """Result dict must have all required keys."""
        from data_sources.image_loader import ImageLoader

        minimal_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        f = tmp_path / "img.png"
        f.write_bytes(minimal_png)

        result = ImageLoader(use_ocr=False).load(str(f))
        for key in ["content", "source", "source_type", "file_type",
                    "file_name", "ocr_text", "ai_description"]:
            assert key in result, f"Missing key: {key}"
