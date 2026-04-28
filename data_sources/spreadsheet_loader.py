# ============================================================
# data_sources/spreadsheet_loader.py
#
# PURPOSE: Load Excel and CSV files into readable text format.
#
# BEGINNER CONCEPT - Why is this hard?
# Spreadsheets have rows and columns. When you turn them into
# text for AI, you need to preserve the structure so the AI
# understands "column A is the date, column B is the value".
# We convert tables to readable text like:
# "Row 1: Name=Alice, Age=30, City=Mumbai"
# ============================================================

import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class SpreadsheetLoader:
    """
    Loads Excel (.xlsx, .xls) and CSV files into text.
    
    Usage:
        loader = SpreadsheetLoader()
        result = loader.load("data.csv")
        print(result["content"])
        print(result["schema"])  # Column names and types
    """
    
    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv"}
    
    def load(self, file_path: str, max_rows: int = 1000) -> Dict[str, Any]:
        """
        Load a spreadsheet file.
        
        Args:
            file_path: Path to the spreadsheet
            max_rows: Limit rows to prevent loading huge files into memory
                     (spreadsheets can have millions of rows!)
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if not path.exists():
            raise FileNotFoundError(f"Spreadsheet not found: {file_path}")
        
        logger.info(f"Loading spreadsheet: {path.name}")
        
        # Import pandas here (lazy import - only loads when needed)
        # This way if pandas isn't installed, other loaders still work
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas not installed. Run: pip install pandas openpyxl")
        
        # Load into a pandas DataFrame (like a spreadsheet in Python)
        if extension == ".csv":
            # Try different encodings (some CSVs use different character sets)
            df = self._load_csv(path, pd)
        elif extension == ".tsv":
            df = pd.read_csv(str(path), sep="\t", nrows=max_rows)
        elif extension in {".xlsx", ".xls"}:
            df = pd.read_excel(str(path), nrows=max_rows)
        
        # Limit rows to max_rows
        if len(df) > max_rows:
            logger.warning(f"Spreadsheet has {len(df)} rows, limiting to {max_rows}")
            df = df.head(max_rows)
        
        # Get schema information (column names and data types)
        schema = {
            col: str(dtype) 
            for col, dtype in df.dtypes.items()
        }
        
        # Convert DataFrame to readable text
        content = self._dataframe_to_text(df, path.name)
        
        return {
            "content": content,
            "source": str(file_path),
            "source_type": "spreadsheet",
            "file_type": extension,
            "file_name": path.name,
            "schema": schema,              # Column names -> data types
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }
    
    def _load_csv(self, path: Path, pd) -> "pd.DataFrame":
        """Try loading CSV with different encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                return pd.read_csv(str(path), encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise e
        
        raise ValueError(f"Could not decode CSV file: {path}")
    
    def _dataframe_to_text(self, df, filename: str) -> str:
        """
        Convert a pandas DataFrame into readable text.
        
        This makes the data searchable by AI. We represent each row
        as "Column: Value, Column: Value" which the AI can understand.
        
        Example output:
            Table: sales_data.csv
            Columns: Date, Product, Revenue, Units
            
            Row 1: Date=2024-01-01, Product=Widget A, Revenue=5000, Units=100
            Row 2: Date=2024-01-02, Product=Widget B, Revenue=3000, Units=60
        """
        lines = []
        
        # Header
        lines.append(f"Table: {filename}")
        lines.append(f"Columns: {', '.join(df.columns)}")
        lines.append(f"Total rows: {len(df)}")
        lines.append("")  # Blank line
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            lines.append("Numeric Summary:")
            for col in numeric_cols:
                lines.append(
                    f"  {col}: min={df[col].min():.2f}, "
                    f"max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}"
                )
            lines.append("")
        
        # Each row as text
        lines.append("Data rows:")
        for idx, row in df.iterrows():
            # Build "Column=Value, Column=Value" string for each row
            row_parts = [f"{col}={val}" for col, val in row.items()]
            lines.append(f"Row {idx + 1}: {', '.join(row_parts)}")
        
        return "\n".join(lines)
