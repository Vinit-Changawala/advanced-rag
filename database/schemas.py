# ============================================================
# database/schemas.py
#
# PURPOSE: Define the "shape" of data in our databases.
#
# BEGINNER CONCEPT - What are schemas?
# A schema is like a form template. It says:
# "Every chunk MUST have these fields: id, text, source, embedding"
# If you try to store something without required fields → error.
# This prevents messy, inconsistent data.
#
# We use Pydantic for validation (pip install pydantic).
# Pydantic checks that data types are correct automatically.
# ============================================================

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


def generate_id() -> str:
    """Generate a unique ID using UUID (Universally Unique Identifier)."""
    return str(uuid.uuid4())


class ChunkSchema(BaseModel):
    """
    Schema for a single text chunk stored in the database.
    
    BEGINNER CONCEPT - BaseModel:
    BaseModel is from Pydantic. When you extend it (class ChunkSchema(BaseModel)),
    Pydantic automatically:
    - Validates data types when you create an instance
    - Converts compatible types (e.g., "5" → 5 for int fields)
    - Provides helpful error messages if validation fails
    """
    
    # Required fields (must always be provided)
    chunk_id: str = Field(default_factory=generate_id, description="Unique ID")
    text: str = Field(..., description="The actual chunk text content")
    source: str = Field(..., description="Source file path")
    
    # Optional fields with defaults
    source_type: str = Field(default="document", description="document/code/image/spreadsheet")
    file_type: str = Field(default="", description="File extension (.pdf, .py, etc.)")
    chunk_index: int = Field(default=0, description="Position of chunk in source document")
    total_chunks: int = Field(default=1, description="Total chunks from this source")
    section_title: str = Field(default="", description="Section/heading this chunk belongs to")
    chunk_type: str = Field(default="text", description="text/table/code")
    
    # Metadata (added during preprocessing)
    summary: str = Field(default="", description="AI-generated summary of this chunk")
    keywords: List[str] = Field(default_factory=list, description="Important keywords")
    hypothetical_questions: List[str] = Field(
        default_factory=list,
        description="HyDE questions this chunk would answer"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    # Embedding (the vector representation - stored separately in vector DB)
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding (1536 floats for OpenAI)"
    )
    
    class Config:
        """Pydantic configuration."""
        # Allow extra fields (don't reject unknown keys)
        extra = "allow"


class QuerySchema(BaseModel):
    """Schema for a user query."""
    query_id: str = Field(default_factory=generate_id)
    query_text: str = Field(..., description="The user's question")
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None


class AnswerSchema(BaseModel):
    """Schema for the final answer returned to the user."""
    answer_id: str = Field(default_factory=generate_id)
    query_id: str = Field(..., description="Reference to the original query")
    answer_text: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default_factory=list, description="Source chunk IDs used")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluation_score: Optional[float] = None
    approved_by_human: bool = Field(default=False)
    latency_ms: Optional[int] = None
    token_count: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class EvaluationSchema(BaseModel):
    """Schema for storing evaluation results."""
    eval_id: str = Field(default_factory=generate_id)
    answer_id: str
    relevance_score: float = Field(ge=0.0, le=10.0)
    accuracy_score: float = Field(ge=0.0, le=10.0)
    completeness_score: float = Field(ge=0.0, le=10.0)
    clarity_score: float = Field(ge=0.0, le=10.0)
    overall_score: float = Field(ge=0.0, le=10.0)
    judge_reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class FeedbackSchema(BaseModel):
    """Schema for feedback loop signals."""
    feedback_id: str = Field(default_factory=generate_id)
    source: str  # "evaluation" or "human"
    issue_type: str  # "low_score", "hallucination", "irrelevant"
    details: str
    original_query: str
    suggested_fix: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
