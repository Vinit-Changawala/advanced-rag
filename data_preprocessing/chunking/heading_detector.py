# ============================================================
# data_preprocessing/chunking/heading_detector.py
# ============================================================

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HeadingDetector:
    """
    Splits text into sections based on detected headings.
    Each section = one heading + all its content until the next heading.

    Usage:
        detector = HeadingDetector()
        sections = detector.split_on_headings(text, headings)
        for section in sections:
            print(section["title"], len(section["text"]))
    """

    def split_on_headings(self, text: str,
                           headings: List[Dict]) -> List[Dict[str, Any]]:
        """
        Split text into sections based on heading positions.

        Args:
            text: The full document text
            headings: List of heading dicts (from StructureAnalyzer)
                      Each dict has "line_number", "text", "level"

        Returns:
            List of section dicts, each with "title", "text", "level"
        """
        if not headings:
            # No headings found — return the whole text as one section
            return [{"title": "Document", "text": text, "level": 1}]

        lines = text.split("\n")
        sections = []

        for i, heading in enumerate(headings):
            start = heading["line_number"]

            # This section ends where the next heading starts
            if i + 1 < len(headings):
                end = headings[i + 1]["line_number"]
            else:
                end = len(lines)

            section_text = "\n".join(lines[start:end]).strip()

            if section_text:
                sections.append({
                    "title": heading["text"],
                    "text": section_text,
                    "level": heading["level"],
                    "line_start": start,
                    "line_end": end,
                })

        logger.info(f"HeadingDetector: split into {len(sections)} sections")
        return sections
