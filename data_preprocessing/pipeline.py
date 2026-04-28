# ============================================================
# data_preprocessing/pipeline.py
#
# PURPOSE: Orchestrate all 3 stages of data preprocessing:
#   Stage 1 → Restructuring (parse and analyze structure)
#   Stage 2 → Chunking (split into pieces)
#   Stage 3 → Metadata creation (add summaries, keywords, questions)
#
# BEGINNER CONCEPT - What is a pipeline?
# A pipeline is like an assembly line in a factory.
# Raw document goes in → processed chunks with metadata come out.
# Each "station" (stage) does one specific job.
#
# FLOW:
# Raw file → [Restructure] → [Chunk] → [Add Metadata] → Ready for DB
# ============================================================

import logging
import time
from typing import List, Dict, Any, Optional

from .restructuring.document_parser import DocumentParser
from .restructuring.structure_analyzer import StructureAnalyzer
from .chunking.heading_detector import HeadingDetector
from .chunking.table_preserver import TablePreserver
from .chunking.boundary_detector import BoundaryDetector
from .metadata_creation.summary_generator import SummaryGenerator
from .metadata_creation.keyword_extractor import KeywordExtractor
from .metadata_creation.question_generator import QuestionGenerator

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    The main pipeline that processes raw documents into searchable chunks.
    
    Usage:
        pipeline = PreprocessingPipeline(llm_client=client)
        
        raw_doc = {
            "content": "Full text of the document...",
            "source": "report.pdf",
            "file_type": ".pdf"
        }
        
        chunks = pipeline.process(raw_doc)
        # chunks is a list of dictionaries, each ready to be stored in DB
    """
    
    def __init__(self, llm_client=None, config: Dict = None):
        """
        Initialize all pipeline stages.
        
        Args:
            llm_client: OpenAI/Anthropic client for AI-powered stages
            config: Optional settings (chunk_size, overlap, etc.)
        """
        # Default configuration values
        self.config = config or {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "min_chunk_size": 100,
        }
        
        # --- Stage 1: Restructuring ---
        self.parser = DocumentParser()
        self.structure_analyzer = StructureAnalyzer()
        
        # --- Stage 2: Chunking ---
        self.heading_detector = HeadingDetector()
        self.table_preserver = TablePreserver()
        self.boundary_detector = BoundaryDetector(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        
        # --- Stage 3: Metadata ---
        self.summary_generator = SummaryGenerator(llm_client=llm_client)
        self.keyword_extractor = KeywordExtractor()
        self.question_generator = QuestionGenerator(llm_client=llm_client)
        
        logger.info("PreprocessingPipeline initialized with all stages")
    
    def process(self, raw_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run a single document through all pipeline stages.
        
        Args:
            raw_document: Dict with at least "content" and "source" keys
            
        Returns:
            List of processed chunks, ready to be stored in the database
        """
        start_time = time.time()
        source = raw_document.get("source", "unknown")
        
        logger.info(f"Starting pipeline for: {source}")
        
        # ── STAGE 1: RESTRUCTURING ──
        logger.info(f"Stage 1: Restructuring {source}")
        
        # Parse the document (clean up the raw text)
        parsed = self.parser.parse(raw_document)
        
        # Analyze the structure (find headings, tables, sections)
        structured = self.structure_analyzer.analyze(parsed)
        
        # ── STAGE 2: CHUNKING ──
        logger.info(f"Stage 2: Chunking {source}")
        
        # Extract and protect tables first (they must not be split)
        content_with_table_markers, tables = self.table_preserver.extract_tables(
            structured["content"]
        )
        
        # Split on headings first (respects document structure)
        heading_chunks = self.heading_detector.split_on_headings(
            content_with_table_markers,
            structured.get("headings", [])
        )
        
        # Then split large heading-chunks on sentence boundaries
        raw_chunks = []
        for section in heading_chunks:
            sub_chunks = self.boundary_detector.split(section["text"])
            for chunk_text in sub_chunks:
                raw_chunks.append({
                    "text": chunk_text,
                    "section_title": section.get("title", ""),
                    "source": source,
                    "file_type": raw_document.get("file_type", ""),
                })
        
        # Re-inject the tables as their own atomic chunks
        for table in tables:
            if len(table["text"]) >= self.config["min_chunk_size"]:
                raw_chunks.append({
                    "text": table["text"],
                    "section_title": table.get("caption", "Table"),
                    "source": source,
                    "chunk_type": "table",
                    "file_type": raw_document.get("file_type", ""),
                })
        
        # Filter out chunks that are too small to be useful
        raw_chunks = [
            c for c in raw_chunks 
            if len(c["text"]) >= self.config["min_chunk_size"]
        ]
        
        logger.info(f"Stage 2 complete: {len(raw_chunks)} chunks created")
        
        # ── STAGE 3: METADATA CREATION ──
        logger.info(f"Stage 3: Adding metadata to {len(raw_chunks)} chunks")
        
        final_chunks = []
        for i, chunk in enumerate(raw_chunks):
            
            # Add a unique ID for this chunk
            chunk["chunk_id"] = f"{source}::chunk_{i}"
            chunk["chunk_index"] = i
            chunk["total_chunks"] = len(raw_chunks)
            
            # Add AI-generated metadata
            chunk["summary"] = self.summary_generator.generate(chunk["text"])
            chunk["keywords"] = self.keyword_extractor.extract(chunk["text"])
            chunk["hypothetical_questions"] = self.question_generator.generate(chunk["text"])
            
            final_chunks.append(chunk)
        
        elapsed = time.time() - start_time
        logger.info(f"Pipeline complete for {source}: {len(final_chunks)} chunks in {elapsed:.2f}s")
        
        return final_chunks
    
    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents through the pipeline.
        
        Args:
            documents: List of raw document dictionaries
            
        Returns:
            All chunks from all documents combined
        """
        all_chunks = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc.get('source')}")
            try:
                chunks = self.process(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {doc.get('source')}: {e}")
                # Continue with next document instead of crashing
                continue
        
        logger.info(f"Batch complete: {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
