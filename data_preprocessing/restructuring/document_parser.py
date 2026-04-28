# ============================================================
# data_preprocessing/restructuring/document_parser.py
#
# PURPOSE: Clean and normalize raw document text.
# "Restructuring" means taking messy raw text and making it clean.
#
# WHAT PROBLEMS DOES THIS SOLVE?
# When you extract text from PDFs, you often get:
# - Extra whitespace: "hello     world"
# - Broken lines: "This is a sen-\ntence"
# - Headers/footers: "Company Name | Page 3 | Confidential"
# - Garbled characters: "â€™" instead of "'"
# ============================================================

import re           # Regular expressions - powerful pattern matching
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Cleans and normalizes raw document text.
    
    Think of this as a "text cleaner" that runs before everything else.
    
    Usage:
        parser = DocumentParser()
        cleaned = parser.parse(raw_doc)
        print(cleaned["content"])  # Clean, normalized text
    """
    
    def parse(self, raw_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and clean a raw document.
        
        Args:
            raw_document: Dict with "content" key containing raw text
            
        Returns:
            Same dict but with cleaned "content"
        """
        content = raw_document.get("content", "")
        
        if not content:
            logger.warning(f"Empty content in: {raw_document.get('source')}")
            return raw_document
        
        # Apply cleaning steps in order
        content = self._fix_encoding_issues(content)
        content = self._fix_hyphenated_line_breaks(content)
        content = self._normalize_whitespace(content)
        content = self._remove_page_artifacts(content)
        content = self._normalize_quotes(content)
        
        # Create a new dict with cleaned content (don't modify original)
        # .copy() creates a shallow copy so we don't accidentally change the input
        result = raw_document.copy()
        result["content"] = content
        result["original_length"] = len(raw_document["content"])
        result["cleaned_length"] = len(content)
        
        return result
    
    def _fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding problems in extracted text.
        
        Some PDFs have garbled characters due to encoding issues.
        """
        # Common encoding artifacts from PDF extraction
        replacements = {
            "\u00e2\u0080\u0099": "'",
            "\u00e2\u0080\u009c": '"',
            "\u00e2\u0080\u009d": '"',
            "\u00e2\u0080\u0094": "\u2014",
            "\u00e2\u0080\u0093": "\u2013",
            "\x00": "",
            "\ufeff": "",
            "■": "₹",
            "●": "-",     # OCR reads bullet as filled circle
            "•": "-",
            "|": "",      # OCR reads borders/dividers as pipe
        }
        
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        # Remove isolated single characters on their own line (OCR logo/icon artifacts)
        text = re.sub(r'(?m)^\s*\S{1,2}\s*$', '', text)
        # Collapse multiple blank lines created by above
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _fix_hyphenated_line_breaks(self, text: str) -> str:
        """
        Fix words that were split across lines with a hyphen.
        
        PDFs often wrap long words with a hyphen:
        "This is a very long sen-
        tence that continues here."
        
        Should become: "This is a very long sentence that continues here."
        
        BEGINNER CONCEPT - What is regex?
        re.sub(pattern, replacement, text)
        pattern: describes what to find (using special characters)
        replacement: what to replace it with
        \\1 refers to the first "captured group" in the pattern
        """
        # Pattern: a word character, then a hyphen, then a newline,
        # then optional spaces, then more word characters
        return re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Clean up excessive whitespace.
        
        - Multiple spaces → single space
        - Multiple newlines → double newline (paragraph break)
        - Trailing spaces on lines → removed
        """
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        
        # Remove trailing whitespace from each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        
        # Replace 3+ newlines with exactly 2 (standard paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()
    
    def _remove_page_artifacts(self, text: str) -> str:
        """
        Remove PDF page headers and footers.
        
        Many PDFs repeat the company name, page number, or 
        document title on every page. This creates noise.
        
        We remove lines that match common patterns like:
        "Page 5 of 20" or "Company Name | Page 5"
        """
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip if it's a page number pattern
            if re.match(r"^[Pp]age\s+\d+\s*(of\s+\d+)?$", stripped):
                continue
            
            # Skip if it's mostly a number (isolated page number)
            if re.match(r"^\d+$", stripped):
                continue
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
    
    def _normalize_quotes(self, text: str) -> str:
        """Convert fancy/curly quotes to standard straight quotes."""
        quote_map = {
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark  
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
        }
        for fancy, plain in quote_map.items():
            text = text.replace(fancy, plain)
        return text
