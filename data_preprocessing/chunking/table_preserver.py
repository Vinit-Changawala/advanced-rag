# ============================================================
# data_preprocessing/chunking/table_preserver.py
#
# PURPOSE: Extract tables BEFORE chunking so they don't get split.
#
# THE PROBLEM:
# If you have a table with 10 rows and your chunk size is 200 tokens,
# a naive chunker will split the table in the middle.
# Row 1-5 end up in chunk A, Row 6-10 in chunk B.
# Now the AI can't understand the complete table.
#
# THE SOLUTION:
# 1. Extract all tables from the document
# 2. Replace them with a placeholder: [[TABLE_0]], [[TABLE_1]]
# 3. Chunk the rest of the document normally
# 4. Re-inject the tables as their own complete chunks
# ============================================================

import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class TablePreserver:
    """
    Extracts tables and replaces them with placeholders before chunking.
    
    Usage:
        preserver = TablePreserver()
        text_with_placeholders, tables = preserver.extract_tables(text)
        # text_with_placeholders: "... [[TABLE_0]] ... [[TABLE_1]] ..."
        # tables: [{"text": "| Col | ...", "index": 0}, ...]
    """
    
    # Placeholder template
    # [[TABLE_0]] is easy to find and replace later
    PLACEHOLDER_TEMPLATE = "[[TABLE_{index}]]"
    
    def extract_tables(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract all tables from text.
        
        Returns:
            Tuple of (modified_text, list_of_tables)
        """
        tables = []
        modified_text = text
        
        # Find markdown-style tables (| col | col |)
        # This regex finds multi-line table patterns
        table_pattern = re.compile(
            r"(\|[^\n]+\|\n"           # First row with pipes
            r"(?:\|[-:]+\|[-:\s|]*\n)?" # Optional separator row (|---|---|)
            r"(?:\|[^\n]+\|\n)+)",       # Subsequent rows
            re.MULTILINE
        )
        
        # Find all matches
        matches = list(table_pattern.finditer(modified_text))
        
        # Process in reverse order so that replacing doesn't shift positions
        # (if we replace from the beginning, the positions of later matches change)
        for i, match in enumerate(reversed(matches)):
            index = len(matches) - 1 - i
            table_text = match.group(0)
            placeholder = self.PLACEHOLDER_TEMPLATE.format(index=index)
            
            tables.insert(0, {  # Insert at beginning (we're going in reverse)
                "text": table_text,
                "index": index,
                "placeholder": placeholder,
                "row_count": table_text.count("\n"),
            })
            
            # Replace the table with the placeholder
            modified_text = modified_text[:match.start()] + f"\n{placeholder}\n" + modified_text[match.end():]
        
        if tables:
            logger.info(f"Extracted {len(tables)} tables, replaced with placeholders")
        
        return modified_text, tables
    
    def restore_tables(self, chunks: List[Dict], tables: List[Dict]) -> List[Dict]:
        """
        If a chunk contains a placeholder, replace it with the actual table.
        
        This is only used if you want the table INLINE in the chunk.
        Usually, tables become their own separate chunks.
        """
        for chunk in chunks:
            for table in tables:
                if table["placeholder"] in chunk.get("text", ""):
                    chunk["text"] = chunk["text"].replace(
                        table["placeholder"], 
                        table["text"]
                    )
        return chunks
