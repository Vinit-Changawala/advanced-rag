# ============================================================
# data_preprocessing/metadata_creation/question_generator.py
#
# PURPOSE: Generate hypothetical questions that this chunk could answer.
#
# WHAT IS HyDE? (Hypothetical Document Embeddings)
# HyDE is a clever technique for better search:
#
# NORMAL RAG SEARCH:
# User asks: "What is the revenue growth?"
# System searches for chunks similar to: "What is the revenue growth?"
#
# HyDE APPROACH:
# For each chunk, we pre-generate questions it would answer.
# Chunk about revenue → ["What was Q3 revenue?", "How did sales grow?"]
# Now when user asks "What is the revenue growth?", it matches these
# pre-generated questions PERFECTLY → better retrieval!
#
# It's like writing the answer key before the exam.
# ============================================================

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates hypothetical questions that a text chunk would answer.
    
    These questions are stored as metadata and used during search
    to improve retrieval accuracy (HyDE technique).
    
    Usage:
        generator = QuestionGenerator(llm_client=openai_client)
        questions = generator.generate("Python is a high-level programming language...")
        # Returns: ["What is Python?", "What kind of language is Python?", ...]
    """
    
    QUESTION_PROMPT = """Given the following text, generate {n} questions that this text would answer.
Questions should be natural, as if asked by someone searching for this information.
Return ONLY the questions, one per line. No numbers, no explanations.

Text:
{text}

Questions:"""
    
    def __init__(self, llm_client=None, model: str = "gpt-4o-mini",
                 num_questions: int = 3):
        """
        Args:
            llm_client: OpenAI client for LLM-based generation
            model: LLM model to use
            num_questions: How many questions to generate per chunk
        """
        self.llm_client = llm_client
        self.model = model
        self.num_questions = num_questions
    
    def generate(self, text: str) -> List[str]:
        """
        Generate hypothetical questions for the given text.
        
        Returns:
            List of question strings
        """
        if not text.strip():
            return []
        
        if self.llm_client:
            try:
                return self._llm_generate(text)
            except Exception as e:
                logger.warning(f"LLM question generation failed: {e}")
        
        # Fallback: return empty list (questions are optional metadata)
        return []
    
    def _llm_generate(self, text: str) -> List[str]:
        """Generate questions using the LLM."""
        # Truncate to keep API costs low
        truncated = text[:1500]
        
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.QUESTION_PROMPT.format(
                        n=self.num_questions,
                        text=truncated
                    )
                }
            ],
            max_tokens=200,
            temperature=0.7  # Slightly creative to get diverse questions
        )
        
        # Parse the response: split by newlines, clean up
        raw = response.choices[0].message.content.strip()
        questions = [
            q.strip().lstrip("0123456789.-) ")  # Remove any numbering
            for q in raw.split("\n")
            if q.strip() and "?" in q  # Must end with a question mark
        ]
        
        return questions[:self.num_questions]  # Ensure we don't return too many
