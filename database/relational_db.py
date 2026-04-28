# ============================================================
# database/relational_db.py
#
# PURPOSE: Store structured metadata in PostgreSQL.
#
# BEGINNER CONCEPT - Vector DB vs Relational DB:
# Vector DB (Qdrant)    → "Find chunks SIMILAR to this question"
# Relational DB (Postgres) → "Find all chunks FROM this file"
#                           "Show me answers WITH score > 8"
#                           "Count queries per day"
#
# We use BOTH together:
# 1. Vector search finds the most relevant chunks
# 2. Relational DB filters/sorts/tracks everything else
#
# We use SQLAlchemy - a Python library that lets you write
# database queries in Python instead of raw SQL.
# ============================================================

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class RelationalDB:
    """
    PostgreSQL database interface for metadata, logs, and structured queries.

    Usage:
        db = RelationalDB(connection_string="postgresql://user:pass@localhost/ragdb")
        db.save_chunk_metadata(chunk)
        db.save_answer(answer)
        results = db.get_low_scored_answers(min_score=7.0)
    """

    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: PostgreSQL URL
            Format: "postgresql://username:password@host:port/database"
        """
        self.connection_string = connection_string
        self.engine = self._create_engine()
        self._create_tables()

    def _create_engine(self):
        """Create the SQLAlchemy database engine (the connection)."""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(
                self.connection_string,
                pool_size=5,        # Keep 5 connections open (reuse them)
                max_overflow=10,    # Allow 10 extra connections if busy
                pool_pre_ping=True  # Test connections before using (handles disconnects)
            )
            logger.info("Database engine created")
            return engine
        except ImportError:
            raise ImportError("sqlalchemy not installed. Run: pip install sqlalchemy psycopg2-binary")

    def _create_tables(self):
        """
        Create all database tables if they don't exist.

        BEGINNER CONCEPT - DDL (Data Definition Language):
        SQL commands that CREATE/ALTER/DROP tables.
        We write the table structure in Python using SQLAlchemy.
        """
        from sqlalchemy import (
            MetaData, Table, Column, String, Text,
            Float, Integer, Boolean, DateTime, JSON
        )

        metadata = MetaData()

        # Table: chunks — stores metadata for every processed chunk
        self.chunks_table = Table("chunks", metadata,
            Column("chunk_id", String(255), primary_key=True),
            Column("text", Text, nullable=False),
            Column("source", String(500)),
            Column("source_type", String(50)),
            Column("section_title", String(500)),
            Column("summary", Text),
            Column("keywords", JSON),           # JSON list of keywords
            Column("chunk_index", Integer, default=0),
            Column("total_chunks", Integer, default=1),
            Column("chunk_type", String(50), default="text"),
            Column("created_at", DateTime, default=datetime.now),
        )

        # Table: answers — every answer our system generates
        self.answers_table = Table("answers", metadata,
            Column("answer_id", String(255), primary_key=True),
            Column("query_id", String(255)),
            Column("query_text", Text),
            Column("answer_text", Text),
            Column("sources", JSON),            # List of source chunk IDs
            Column("confidence_score", Float, default=0.0),
            Column("evaluation_score", Float),
            Column("approved_by_human", Boolean, default=False),
            Column("latency_ms", Integer),
            Column("token_count", Integer),
            Column("timestamp", DateTime, default=datetime.now),
        )

        # Table: evaluations — scores for each answer
        self.evaluations_table = Table("evaluations", metadata,
            Column("eval_id", String(255), primary_key=True),
            Column("answer_id", String(255)),
            Column("relevance_score", Float),
            Column("accuracy_score", Float),
            Column("completeness_score", Float),
            Column("clarity_score", Float),
            Column("overall_score", Float),
            Column("judge_reasoning", Text),
            Column("timestamp", DateTime, default=datetime.now),
        )

        # Table: feedback — signals from the feedback loop
        self.feedback_table = Table("feedback", metadata,
            Column("feedback_id", String(255), primary_key=True),
            Column("source", String(50)),
            Column("issue_type", String(100)),
            Column("details", Text),
            Column("original_query", Text),
            Column("suggested_fix", Text),
            Column("timestamp", DateTime, default=datetime.now),
        )

        # Create all tables in the database
        metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

    def save_chunk_metadata(self, chunk: Dict[str, Any]):
        """Save chunk metadata to the chunks table."""
        from sqlalchemy import insert

        with self.engine.connect() as conn:
            # Use INSERT OR IGNORE to skip duplicates
            stmt = insert(self.chunks_table).prefix_with("OR IGNORE")
            conn.execute(stmt, {
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "source_type": chunk.get("source_type", ""),
                "section_title": chunk.get("section_title", ""),
                "summary": chunk.get("summary", ""),
                "keywords": chunk.get("keywords", []),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "chunk_type": chunk.get("chunk_type", "text"),
                "created_at": datetime.now(),
            })
            conn.commit()

    def save_answer(self, answer: Dict[str, Any]):
        """Save a generated answer to the answers table."""
        from sqlalchemy import insert

        with self.engine.connect() as conn:
            conn.execute(insert(self.answers_table), {
                "answer_id": answer.get("answer_id"),
                "query_id": answer.get("query_id"),
                "query_text": answer.get("query_text", ""),
                "answer_text": answer.get("answer_text", ""),
                "sources": answer.get("sources", []),
                "confidence_score": answer.get("confidence_score", 0.0),
                "evaluation_score": answer.get("evaluation_score"),
                "approved_by_human": answer.get("approved_by_human", False),
                "latency_ms": answer.get("latency_ms"),
                "token_count": answer.get("token_count"),
                "timestamp": datetime.now(),
            })
            conn.commit()

    def save_evaluation(self, evaluation: Dict[str, Any]):
        """Save an evaluation result."""
        from sqlalchemy import insert

        with self.engine.connect() as conn:
            conn.execute(insert(self.evaluations_table), evaluation)
            conn.commit()

    def save_feedback(self, feedback: Dict[str, Any]):
        """Save a feedback signal from the feedback loop."""
        from sqlalchemy import insert

        with self.engine.connect() as conn:
            conn.execute(insert(self.feedback_table), feedback)
            conn.commit()

    def get_low_scored_answers(self, threshold: float = 7.0,
                                limit: int = 100) -> List[Dict]:
        """
        Fetch answers that scored below the threshold.
        These are used by the feedback loop to improve the system.
        """
        from sqlalchemy import text

        query = text("""
            SELECT a.*, e.overall_score, e.judge_reasoning
            FROM answers a
            JOIN evaluations e ON a.answer_id = e.answer_id
            WHERE e.overall_score < :threshold
            ORDER BY e.overall_score ASC
            LIMIT :limit
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"threshold": threshold, "limit": limit})
            # Convert rows to dictionaries
            return [dict(row._mapping) for row in result]

    def get_answer_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics about system performance.
        Useful for dashboards and monitoring.
        """
        from sqlalchemy import text

        with self.engine.connect() as conn:
            stats = {}

            # Total answers
            r = conn.execute(text("SELECT COUNT(*) as total FROM answers"))
            stats["total_answers"] = r.fetchone()[0]

            # Average score
            r = conn.execute(text("SELECT AVG(overall_score) as avg FROM evaluations"))
            row = r.fetchone()
            stats["avg_score"] = round(row[0], 2) if row[0] else 0.0

            # Average latency
            r = conn.execute(text("SELECT AVG(latency_ms) as avg FROM answers WHERE latency_ms IS NOT NULL"))
            row = r.fetchone()
            stats["avg_latency_ms"] = round(row[0], 0) if row[0] else 0

            return stats
