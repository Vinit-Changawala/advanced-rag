# ============================================================
# data_preprocessing/metadata_creation/summary_generator.py
#
# PURPOSE: Generate a short summary for each chunk using an LLM.
#
# WHY SUMMARIES?
# When searching, we can search the SUMMARY instead of the full text.
# Summaries are shorter → faster search.
# Summaries are more "semantic" → better match to user questions.
#
# EXAMPLE:
# Chunk text (200 words): "In Q3 2024, the company reported revenue of..."
# Summary (20 words): "Q3 2024 financial results showing revenue growth..."
# A question like "how did the company perform?" matches the summary better.
# ============================================================

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generates concise summaries for text chunks using an LLM.
    
    Usage:
        generator = SummaryGenerator(llm_client=openai_client)
        summary = generator.generate("Long chunk text here...")
        print(summary)  # "Short summary of the chunk..."
    """
    
    SUMMARY_PROMPT = """Summarize the following text in 1-2 sentences.
Be concise and capture the key information.
Only return the summary, nothing else.

Text:
{text}

Summary:"""
    
    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        """
        Args:
            llm_client: OpenAI client. If None, uses extractive summary.
            model: Which model to use (gpt-4o-mini is cheap and fast)
        """
        self.llm_client = llm_client
        self.model = model
    
    def generate(self, text: str, max_length: int = 150) -> str:
        """
        Generate a summary for the given text.
        
        Falls back to extractive summary if no LLM is available.
        
        Args:
            text: The chunk text to summarize
            max_length: Max character length of summary
        """
        if not text.strip():
            return ""
        
        # Use LLM if available (better quality)
        if self.llm_client:
            try:
                return self._llm_summary(text)
            except Exception as e:
                logger.warning(f"LLM summary failed, falling back to extractive: {e}")
        
        # Fallback: extractive summary (just use the first 2 sentences)
        return self._extractive_summary(text, max_length)
    
    def _llm_summary(self, text: str) -> str:
        """Generate summary using the LLM."""
        # Only send first 2000 characters to keep costs low
        truncated_text = text[:2000]
        
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.SUMMARY_PROMPT.format(text=truncated_text)
                }
            ],
            max_tokens=100,
            temperature=0.1  # Low temperature for consistent, factual summaries
        )
        
        return response.choices[0].message.content.strip()
    
    def _extractive_summary(self, text: str, max_length: int) -> str:
        """
        Simple extractive summary: use the first 1-2 sentences.
        
        This is a fallback when no LLM is available.
        Not as good as LLM summaries but better than nothing.
        """
        import re
        
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + " "
            else:
                break
        
        return summary.strip()
