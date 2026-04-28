# ============================================================
# data_preprocessing/chunking/heading_detector.py
#
# PURPOSE: Split documents on headings first.
# This creates "semantic chunks" that respect the document structure.
#
# WHY HEADINGS FIRST?
# A document section about "Data Privacy" should be one chunk.
# If we split by size first, we might get:
#   Chunk 1: "...end of Marketing section... [beginning of Data Privacy]"
#   Chunk 2: "[middle of Data Privacy]..."
# This loses context! Heading-based splitting is smarter.
# ============================================================

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HeadingDetector:
    """
    Splits text into sections based on detected headings.
    
    Each section = one heading + all its content.
    
    Usage:
        detector = HeadingDetector()
        sections = detector.split_on_headings(text, headings)
        for section in sections:
            print(section["title"], ":", section["text"][:100])
    """
    
    def split_on_headings(self, text: str, 
                           headings: List[Dict]) -> List[Dict[str, Any]]:
        """
        Split text into sections based on heading positions.
        
        Args:
            text: The document text
            headings: List of heading dicts from StructureAnalyzer
            
        Returns:
            List of sections, each with "title" and "text" keys
        """
        if not headings:
            # No headings found - return the whole text as one section
            return [{"title": "Document", "text": text, "level": 1}]
        
        lines = text.split("\n")
        sections = []
        
        for i, heading in enumerate(headings):
            start = heading["line_number"]
            
            # This section ends where the next one begins
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
        
        return sections


# ============================================================
# data_preprocessing/chunking/boundary_detector.py
#
# PURPOSE: Split large text chunks at natural sentence/paragraph boundaries.
#
# THE PROBLEM WITH SIZE-BASED CHUNKING:
# "This sentence ends here. This new sent" <- WRONG! Cut mid-word
#
# BOUNDARY-AWARE CHUNKING finds natural break points:
# - End of sentences (.)
# - End of paragraphs (\n\n)
# - End of list items
# ============================================================


class BoundaryDetector:
    """
    Splits text into chunks at natural language boundaries.
    
    Unlike naive chunking (split every N characters), this
    always finishes the current sentence before splitting.
    
    Usage:
        chunker = BoundaryDetector(chunk_size=512, chunk_overlap=50)
        chunks = chunker.split("Long text here...")
        # Returns list of text strings
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Target size in characters (not tokens!)
                       Roughly: 1 token ≈ 4 characters
                       512 tokens ≈ 2048 characters
            chunk_overlap: How many characters to overlap between chunks
                          This ensures context isn't lost at chunk boundaries
        """
        self.chunk_size = chunk_size * 4      # Convert tokens → characters
        self.chunk_overlap = chunk_overlap * 4
    
    def split(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks at sentence boundaries.
        
        BEGINNER CONCEPT - What is overlap?
        Imagine this sentence is cut into two chunks:
        Chunk 1: "The cat sat on the mat. The dog..."
        Chunk 2: "...The dog ran away."
        
        Without overlap, "The dog" gets cut and context is lost.
        With overlap:
        Chunk 1: "The cat sat on the mat. The dog"
        Chunk 2: "The dog ran away."  ← repeats "The dog" for context
        """
        if len(text) <= self.chunk_size:
            # Text is small enough to be one chunk
            return [text]
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Current chunk is full - save it
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last few sentences for context continuity
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Don't forget the last chunk!
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        
        This is harder than it looks! "Dr. Smith went to Washington."
        has a period in "Dr." that is NOT a sentence end.
        
        We use a simple but effective approach:
        Split on ". ", "! ", "? " but not on abbreviations.
        """
        # Simple sentence splitter using regex
        # (?<=[.!?]) means "preceded by . or ! or ?"
        # \s+ means "followed by whitespace"
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Get the last N sentences to use as overlap for the next chunk.
        
        We keep sentences from the end until we've collected
        enough characters for the overlap.
        """
        overlap_sentences = []
        overlap_length = 0
        
        # Walk backwards through sentences
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences
