# ============================================================
# data_preprocessing/restructuring/structure_analyzer.py
#
# PURPOSE: Detect the structure of a document.
# - Where are the headings? (H1, H2, H3)
# - Where are the tables?
# - Where are the lists?
# - What sections does the document have?
#
# WHY DOES THIS MATTER?
# When chunking, we want to split at NATURAL BOUNDARIES,
# not randomly in the middle of a paragraph.
# Knowing the structure helps us chunk intelligently.
# ============================================================

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class StructureAnalyzer:
    """
    Analyzes document structure to identify headings, tables, and sections.
    
    Usage:
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(parsed_doc)
        print(result["headings"])   # List of heading objects
        print(result["tables"])     # List of table text blocks
    """
    
    def analyze(self, parsed_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the document structure.
        
        Returns the document dict with added structure information:
        - headings: list of {text, level, position}
        - tables: list of {text, position, caption}
        - lists: list of {items, position}
        - sections: list of {title, start, end}
        """
        content = parsed_document.get("content", "")
        
        headings = self._detect_headings(content)
        tables = self._detect_tables(content)
        lists = self._detect_lists(content)
        sections = self._build_sections(content, headings)
        
        result = parsed_document.copy()
        result.update({
            "headings": headings,
            "tables": tables,
            "lists": lists,
            "sections": sections,
            "structure_detected": True,
        })
        
        logger.info(
            f"Structure: {len(headings)} headings, "
            f"{len(tables)} tables, "
            f"{len(sections)} sections"
        )
        
        return result
    
    def _detect_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect headings in the document.
        
        We look for two patterns:
        1. Markdown-style: # Heading, ## Subheading
        2. ALL CAPS short lines: "INTRODUCTION" or "3. METHODOLOGY"
        3. Numbered sections: "1.", "1.1", "2.3.1"
        """
        headings = []
        lines = text.split("\n")
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            heading = None
            
            # Pattern 1: Markdown headings (### Title)
            md_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if md_match:
                level = len(md_match.group(1))  # Count the # symbols
                heading = {
                    "text": md_match.group(2),
                    "level": level,
                    "line_number": i,
                    "type": "markdown"
                }
            
            # Pattern 2: Numbered sections (1. Title, 1.1 Title, 2.3.1 Title)
            elif re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", stripped) and len(stripped) < 100:
                heading = {
                    "text": stripped,
                    "level": stripped.count(".") + 1,  # depth by dots
                    "line_number": i,
                    "type": "numbered"
                }
            
            # Pattern 3: Short ALL CAPS line (likely a heading)
            elif (stripped.isupper() and 
                  len(stripped) > 3 and 
                  len(stripped) < 80 and
                  len(stripped.split()) < 10):
                heading = {
                    "text": stripped,
                    "level": 2,  # Treat as H2
                    "line_number": i,
                    "type": "caps"
                }
            
            if heading:
                headings.append(heading)
        
        return headings
    
    def _detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect tables in text.
        
        We detect two styles:
        1. Markdown tables using | pipes: | Col1 | Col2 |
        2. Space-aligned tables (common in PDFs)
        """
        tables = []
        lines = text.split("\n")
        
        in_table = False
        table_start = 0
        table_lines = []
        
        for i, line in enumerate(lines):
            # Check if this line looks like a table row (has | characters)
            is_table_line = "|" in line and len(line.strip()) > 3
            
            if is_table_line and not in_table:
                # Start of a new table
                in_table = True
                table_start = i
                table_lines = [line]
            elif is_table_line and in_table:
                # Continuation of current table
                table_lines.append(line)
            elif not is_table_line and in_table:
                # End of table
                if len(table_lines) >= 2:  # Need at least 2 rows to be a table
                    tables.append({
                        "text": "\n".join(table_lines),
                        "start_line": table_start,
                        "end_line": i - 1,
                        "row_count": len(table_lines),
                    })
                in_table = False
                table_lines = []
        
        # Handle table at end of document
        if in_table and len(table_lines) >= 2:
            tables.append({
                "text": "\n".join(table_lines),
                "start_line": table_start,
                "end_line": len(lines) - 1,
                "row_count": len(table_lines),
            })
        
        return tables
    
    def _detect_lists(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect bullet lists and numbered lists.
        """
        lists = []
        lines = text.split("\n")
        
        in_list = False
        list_items = []
        list_start = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for bullet points or numbered list items
            is_list_item = (
                stripped.startswith("- ") or
                stripped.startswith("* ") or
                stripped.startswith("• ") or
                re.match(r"^\d+\.\s", stripped)  # "1. item"
            )
            
            if is_list_item and not in_list:
                in_list = True
                list_start = i
                list_items = [stripped]
            elif is_list_item and in_list:
                list_items.append(stripped)
            elif not is_list_item and in_list:
                if len(list_items) >= 2:
                    lists.append({
                        "items": list_items,
                        "start_line": list_start,
                        "count": len(list_items)
                    })
                in_list = False
                list_items = []
        
        return lists
    
    def _build_sections(self, text: str, headings: List[Dict]) -> List[Dict]:
        """
        Build a list of sections from the detected headings.
        
        A section = heading + all text until the next heading.
        """
        if not headings:
            return [{"title": "Document", "content": text}]
        
        lines = text.split("\n")
        sections = []
        
        for i, heading in enumerate(headings):
            # Start of this section = the heading line
            start = heading["line_number"]
            
            # End of this section = start of next heading (or end of document)
            if i + 1 < len(headings):
                end = headings[i + 1]["line_number"]
            else:
                end = len(lines)
            
            section_lines = lines[start:end]
            sections.append({
                "title": heading["text"],
                "level": heading["level"],
                "content": "\n".join(section_lines),
                "start_line": start,
                "end_line": end,
            })
        
        return sections
