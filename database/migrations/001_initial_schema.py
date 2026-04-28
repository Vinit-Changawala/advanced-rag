# ============================================================
# database/migrations/001_initial_schema.py
#
# PURPOSE: Create all database tables for the first time.
#
# BEGINNER CONCEPT — What is a database migration?
# Imagine you have a database already running in production with
# real data. You need to ADD a new column called "approved".
# You can't just delete the database and recreate it (that loses data!).
# Instead, you run a MIGRATION — a script that safely modifies
# the existing schema while preserving all the data.
#
# MIGRATION NAMING CONVENTION:
# 001_initial_schema.py  → First migration (creates all tables)
# 002_add_feedback_table.py → Second migration (adds a new table)
# 003_add_index_on_score.py → Third migration (performance improvement)
#
# Each migration has TWO functions:
# upgrade() → Apply the change (move forward)
# downgrade() → Undo the change (roll back if something goes wrong)
#
# We use Alembic for migration management.
# Commands:
#   alembic upgrade head       → Run all pending migrations
#   alembic downgrade -1       → Undo the last migration
#   alembic revision --autogenerate -m "description"  → Create new migration
# ============================================================

"""
Migration 001: Initial schema

Creates all core tables:
- chunks: stores metadata for every processed document chunk
- answers: stores every query-answer pair
- evaluations: stores LLM judge scores for each answer
- feedback: stores improvement signals from the feedback loop
"""

# revision identifiers (used by Alembic to track which migrations have run)
revision = "001"
down_revision = None   # None means this is the FIRST migration (no predecessor)
branch_labels = None
depends_on = None


def upgrade(engine=None):
    """
    Create all tables for the first time.

    Called by: alembic upgrade head
    Or manually: migration.upgrade(engine)
    """
    from sqlalchemy import (
        MetaData, Table, Column,
        String, Text, Float, Integer, Boolean, DateTime, JSON
    )
    from datetime import datetime

    metadata = MetaData()

    # ── Table: chunks ─────────────────────────────────────────
    # Stores metadata for every processed document chunk
    # (The actual text vectors are stored in Qdrant, not here)
    chunks = Table("chunks", metadata,
        Column("chunk_id",       String(255),  primary_key=True,
               doc="Unique identifier: source::chunk_N"),
        Column("text",           Text,         nullable=False,
               doc="The actual text content of this chunk"),
        Column("source",         String(500),  nullable=False,
               doc="Path to the original file"),
        Column("source_type",    String(50),   default="document",
               doc="document, code, image, or spreadsheet"),
        Column("file_type",      String(20),   default="",
               doc="File extension: .pdf, .py, .csv, etc."),
        Column("section_title",  String(500),  default="",
               doc="Heading/section this chunk belongs to"),
        Column("chunk_type",     String(50),   default="text",
               doc="text, table, or code"),
        Column("chunk_index",    Integer,      default=0,
               doc="Position of this chunk within its source document"),
        Column("total_chunks",   Integer,      default=1,
               doc="Total number of chunks from the same source"),
        Column("summary",        Text,         default="",
               doc="AI-generated summary of this chunk's content"),
        Column("keywords",       JSON,         default=list,
               doc="List of important keywords extracted from this chunk"),
        Column("created_at",     DateTime,     default=datetime.now),
    )

    # ── Table: answers ────────────────────────────────────────
    # Stores every query-answer pair the system generates
    answers = Table("answers", metadata,
        Column("answer_id",          String(255),  primary_key=True),
        Column("query_id",           String(255),  nullable=False),
        Column("query_text",         Text,         nullable=False,
               doc="The original user question"),
        Column("answer_text",        Text,         nullable=False,
               doc="The generated answer"),
        Column("sources",            JSON,         default=list,
               doc="List of source file paths used to generate this answer"),
        Column("confidence_score",   Float,        default=0.0),
        Column("evaluation_score",   Float,        nullable=True,
               doc="Overall score from LLM Judge (0-10)"),
        Column("approved_by_human",  Boolean,      default=False),
        Column("route_taken",        String(50),   default="direct",
               doc="direct, multi_agent, or human_review"),
        Column("latency_ms",         Integer,      nullable=True),
        Column("token_count",        Integer,      nullable=True),
        Column("timestamp",          DateTime,     default=datetime.now),
    )

    # ── Table: evaluations ────────────────────────────────────
    # Detailed scoring breakdown from the LLM Judge
    evaluations = Table("evaluations", metadata,
        Column("eval_id",            String(255),  primary_key=True),
        Column("answer_id",          String(255),  nullable=False,
               doc="Foreign key → answers.answer_id"),
        Column("relevance_score",    Float,        default=0.0),
        Column("accuracy_score",     Float,        default=0.0),
        Column("completeness_score", Float,        default=0.0),
        Column("clarity_score",      Float,        default=0.0),
        Column("overall_score",      Float,        default=0.0),
        Column("judge_reasoning",    Text,         default="",
               doc="LLM Judge's explanation of the scores"),
        Column("timestamp",          DateTime,     default=datetime.now),
    )

    # ── Table: feedback ───────────────────────────────────────
    # Improvement signals from the feedback loop
    feedback = Table("feedback", metadata,
        Column("feedback_id",    String(255),  primary_key=True),
        Column("source",         String(50),
               doc="evaluation or human"),
        Column("issue_type",     String(100),
               doc="accuracy, relevance, completeness, clarity"),
        Column("details",        Text,         default=""),
        Column("original_query", Text),
        Column("suggested_fix",  Text,         default=""),
        Column("timestamp",      DateTime,     default=datetime.now),
    )

    if engine:
        metadata.create_all(engine)
        print("✅ Migration 001: All tables created successfully")
    else:
        print("⚠️  No engine provided — tables not actually created")
        print("    Tables that would be created:", [t.name for t in metadata.sorted_tables])

    return metadata


def downgrade(engine=None):
    """
    Drop all tables created by this migration.

    DANGER: This deletes ALL data! Only use in development.
    Called by: alembic downgrade -1
    """
    from sqlalchemy import MetaData

    if engine:
        metadata = MetaData()
        metadata.reflect(engine)   # Load existing table definitions

        # Drop in reverse order (respect foreign key constraints)
        tables_to_drop = ["feedback", "evaluations", "answers", "chunks"]
        for table_name in tables_to_drop:
            if table_name in metadata.tables:
                metadata.tables[table_name].drop(engine)
                print(f"🗑️  Dropped table: {table_name}")

        print("✅ Migration 001 rolled back: all tables dropped")


# ── RUN DIRECTLY ─────────────────────────────────────────────
# python database/migrations/001_initial_schema.py
if __name__ == "__main__":
    import os
    from sqlalchemy import create_engine

    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://raguser:password@localhost:5432/ragdb"
    )

    print(f"Connecting to: {db_url}")
    engine = create_engine(db_url)

    print("Running upgrade...")
    upgrade(engine)
