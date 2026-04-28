# ============================================================
# data_preprocessing/metadata_creation/keyword_extractor.py
#
# PURPOSE: Extract the most important keywords from a chunk.
#
# KEYWORDS ARE USED FOR:
# 1. Keyword-based search (BM25) alongside vector search
# 2. Filtering: "show only chunks about 'authentication'"
# 3. Quick topic identification
#
# HOW IT WORKS (TF-IDF):
# TF = Term Frequency: How often does word X appear in THIS chunk?
# IDF = Inverse Document Frequency: How rare is word X across ALL chunks?
# TF-IDF score = high if word is COMMON in this chunk but RARE overall
# This gives us the words that are UNIQUELY important to this chunk.
# ============================================================

import re
import logging
import math
from typing import List, Dict
from collections import Counter

logger = logging.getLogger(__name__)


# Common English words that carry no meaning (called "stop words")
# We ignore these when extracting keywords
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "this", "that", "these", "those", "it", "its", "as",
    "if", "then", "than", "so", "yet", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "too",
    "very", "just", "because", "while", "although", "though", "however"
}


class KeywordExtractor:
    """
    Extracts important keywords from text chunks.
    
    Usage:
        extractor = KeywordExtractor()
        keywords = extractor.extract("Python is a programming language...")
        print(keywords)  # ["python", "programming", "language", ...]
    """
    
    def __init__(self, max_keywords: int = 10):
        self.max_keywords = max_keywords
    
    def extract(self, text: str) -> List[str]:
        """
        Extract the most important keywords from text.
        
        Uses simple word frequency + stop word removal.
        """
        if not text.strip():
            return []
        
        # Step 1: Tokenize (split into words)
        # \b means "word boundary", \w+ means "one or more word characters"
        words = re.findall(r"\b\w+\b", text.lower())
        
        # Step 2: Remove stop words and short words
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        # Step 3: Count word frequency
        word_counts = Counter(words)
        
        # Step 4: Get the most common words (these are our keywords)
        top_keywords = [word for word, count in word_counts.most_common(self.max_keywords)]
        
        return top_keywords
    
    def extract_with_scores(self, text: str) -> Dict[str, float]:
        """
        Extract keywords with their importance scores.
        
        Returns dict of {keyword: score} where score is normalized frequency.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        if not words:
            return {}
        
        word_counts = Counter(words)
        total = len(words)
        
        # Normalize scores to be between 0 and 1
        return {
            word: count / total
            for word, count in word_counts.most_common(self.max_keywords)
        }
