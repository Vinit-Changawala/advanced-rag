# ============================================================
# database/vector_store.py
#
# PURPOSE: Store and search chunks using vector embeddings.
#
# BEGINNER CONCEPT - What is a vector database?
# Normal databases search by EXACT MATCH: "WHERE name = 'Alice'"
# Vector databases search by SIMILARITY: "Find things similar to..."
#
# HOW IT WORKS:
# 1. Convert text to a vector (list of 1536 numbers) using an embedding model
# 2. Store the vector in Qdrant
# 3. When searching, convert the query to a vector too
# 4. Find the stored vectors CLOSEST to the query vector
# 5. Return the text those vectors represent
#
# "Distance" between vectors = semantic similarity
# Vectors close together = similar meaning
# ============================================================

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Interface to Qdrant vector database for storing and searching chunks.
    
    Usage:
        store = VectorStore(
            host="localhost",
            port=6333,
            collection_name="rag_chunks",
            openai_client=client
        )
        
        # Store a chunk
        store.upsert(chunk_dict)
        
        # Search
        results = store.search("What is revenue growth?", top_k=5)
    """
    
    def __init__(self, host: str, port: int, collection_name: str,
                 openai_client=None,
                 embedding_client=None,
                 embedding_model: str = "text-embedding-3-small",
                 vector_size: int = 1536):
        """
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name for the collection (like a table)
            embedding_client: OpenAI client for embeddings (preferred parameter name).
                              Always an OpenAI client — even when using Mistral for chat,
                              embeddings still come from text-embedding-3-small.
            openai_client: Alias for embedding_client (backward compatibility).
            embedding_model: Which OpenAI embedding model to use
            vector_size: Number of dimensions per vector (must match Qdrant collection)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        # Accept either parameter name
        self.openai_client = embedding_client or openai_client
        self.embedding_model = embedding_model

        # Auto-detect vector size from HuggingFace embedder if not explicitly set
        if vector_size == 1536 and self.openai_client is not None:
            if hasattr(self.openai_client, "vector_size"):
                # HuggingFaceEmbedder exposes .vector_size
                self.vector_size = self.openai_client.vector_size
                logger.info(f"Auto-detected embedding dimensions: {self.vector_size}")
            else:
                self.vector_size = vector_size   # keep 1536 for OpenAI
        else:
            self.vector_size = vector_size

        # Connect to Qdrant
        self.client = self._connect()

        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _connect(self):
        """Create connection to Qdrant database."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            return client
        except ImportError:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Create the Qdrant collection if it doesn't exist yet."""
        from qdrant_client.models import VectorParams, Distance
        
        # Get existing collections
        existing = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE   # Cosine similarity is standard for text
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Convert text to a vector embedding.

        Supports two embedding backends:
        1. HuggingFaceEmbedder (default, free) — calls embedder.embed(text)
        2. OpenAI client (legacy) — calls openai_client.embeddings.create(...)

        The backend is detected automatically from the type of self.openai_client.
        """
        if self.openai_client is None:
            raise RuntimeError(
                "No embedding client configured. "
                "Pass embedding_client= when creating VectorStore."
            )

        # HuggingFace embedder: has an .embed() method
        if hasattr(self.openai_client, "embed"):
            return self.openai_client.embed(text)

        # Legacy OpenAI client: has .embeddings.create()
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        # Response contains a list of embeddings (we only sent 1 text)
        return response.data[0].embedding
    
    def upsert(self, chunk: Dict[str, Any]) -> str:
        """
        Store a chunk in the vector database.
        
        "Upsert" = INSERT if not exists, UPDATE if exists.
        Named because it handles both cases.
        
        Args:
            chunk: Dict with at minimum "text" and "chunk_id" keys
            
        Returns:
            The chunk_id
        """
        from qdrant_client.models import PointStruct
        
        # Generate embedding for the text
        # We embed both the text AND the summary+questions for better search
        text_to_embed = self._build_embeddable_text(chunk)
        embedding = self.embed_text(text_to_embed)
        
        # Build the payload (metadata stored alongside the vector)
        # The payload lets us filter results by source, type, etc.
        payload = {
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "source_type": chunk.get("source_type", ""),
            "section_title": chunk.get("section_title", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "summary": chunk.get("summary", ""),
            "keywords": chunk.get("keywords", []),
            "chunk_type": chunk.get("chunk_type", "text"),
        }
        
        # Create a point (the unit of data in Qdrant)
        # A "point" = vector + ID + payload
        point = PointStruct(
            id=self._hash_id(chunk["chunk_id"]),  # Qdrant needs integer IDs
            vector=embedding,
            payload=payload
        )
        
        # Upsert into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        return chunk["chunk_id"]
    
    def upsert_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Store multiple chunks efficiently in batches.
        
        Batching reduces the number of API calls → faster ingestion.
        """
        from qdrant_client.models import PointStruct
        
        logger.info(f"Upserting {len(chunks)} chunks in batches of {batch_size}")
        
        # Process in batches
        # range(start, stop, step) generates: 0, batch_size, 2*batch_size, ...
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []
            
            for chunk in batch:
                try:
                    text_to_embed = self._build_embeddable_text(chunk)
                    embedding = self.embed_text(text_to_embed)
                    
                    payload = {
                        "text": chunk.get("text", ""),
                        "source": chunk.get("source", ""),
                        "source_type": chunk.get("source_type", ""),
                        "section_title": chunk.get("section_title", ""),
                        "summary": chunk.get("summary", ""),
                        "keywords": chunk.get("keywords", []),
                    }
                    
                    points.append(PointStruct(
                        id=self._hash_id(chunk["chunk_id"]),
                        vector=embedding,
                        payload=payload
                    ))
                except Exception as e:
                    logger.error(f"Failed to embed chunk {chunk.get('chunk_id')}: {e}")
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            logger.info(f"Upserted batch {i//batch_size + 1}: {len(points)} points")
    
    def search(self, query: str, top_k: int = 5,
               filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for chunks most similar to the query.
        
        Args:
            query: The search query text
            top_k: Return the top K most similar chunks
            filter_dict: Optional filters, e.g., {"source_type": "document"}
            
        Returns:
            List of result dicts with "text", "score", and metadata
        """
        # Convert query to vector
        query_embedding = self.embed_text(query)
        
        # Build optional filter
        qdrant_filter = None
        if filter_dict:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            qdrant_filter = Filter(must=conditions)
        
        # Perform vector search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True  # Include the payload (metadata) in results
        )
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "chunk_id": str(result.id),
                "text": result.payload.get("text", ""),
                "source": result.payload.get("source", ""),
                "source_type": result.payload.get("source_type", ""),
                "section_title": result.payload.get("section_title", ""),
                "summary": result.payload.get("summary", ""),
                "score": result.score,  # Similarity score (0 to 1, higher = more similar)
            })
        
        return formatted
    
    def _build_embeddable_text(self, chunk: Dict) -> str:
        """
        Combine text + metadata into one string for embedding.
        
        Embedding the summary and questions alongside the main text
        improves search recall (the chance of finding the right chunk).
        """
        parts = [chunk.get("text", "")]
        
        if chunk.get("summary"):
            parts.append(f"Summary: {chunk['summary']}")
        
        if chunk.get("hypothetical_questions"):
            questions_str = " ".join(chunk["hypothetical_questions"])
            parts.append(f"Questions: {questions_str}")
        
        return " ".join(parts)
    
    def _hash_id(self, chunk_id: str) -> int:
        """
        Convert a string ID to an integer.
        Qdrant requires integer IDs for its points.
        
        We use Python's built-in hash() function.
        abs() makes it positive (hashes can be negative).
        """
        return abs(hash(chunk_id)) % (10**15)  # Keep within safe integer range
    
    def count(self) -> int:
        """Return total number of vectors stored."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.vectors_count or 0
        except Exception:
            return 0
    
    def delete_by_source(self, source: str):
        """
        Delete all chunks matching a source filename.
        Works with both full paths (/tmp/uuid_file.pdf) and clean names (file.pdf).
        Scans all chunks and deletes any where the basename matches.
        """
        import os

        def _clean(p):
            n = os.path.basename(p)
            return n[37:] if len(n) > 37 and n[36] == "_" else n

        clean_target = _clean(source)

        # Scroll through all points to find matching IDs
        ids_to_delete = []
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for point in results:
                stored = point.payload.get("source", "")
                if _clean(stored) == clean_target:
                    ids_to_delete.append(point.id)
            if next_offset is None:
                break
            offset = next_offset

        if ids_to_delete:
            from qdrant_client.models import PointIdsList
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids_to_delete),
            )
            logger.info(f"Deleted {len(ids_to_delete)} chunks for source: {clean_target}")
        else:
            logger.warning(f"No chunks found for source: {clean_target}")

    def source_exists(self, filename: str) -> bool:
        """
        Check if any chunks from this filename already exist in Qdrant.
        Used to prevent duplicate ingestion of the same file.
        
        Args:
            filename: Just the filename e.g. "RoBERTa.pdf" (not the full path)
        
        Returns:
            True if at least one chunk from this file exists
        """
        from qdrant_client.models import Filter, FieldCondition, MatchText
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="source",
                        match=MatchText(text=filename)
                    )]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results[0]) > 0
        except Exception:
            return False

    def list_sources(self) -> list:
        """
        Return a list of all unique source filenames currently in Qdrant.
        Used by the frontend to show what files are ingested.
        """
        try:
            all_sources = set()
            offset = None
            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["source"],
                    with_vectors=False,
                )
                for point in results:
                    src = point.payload.get("source", "")
                    if src:
                        import os
                        fname = os.path.basename(src)
                        # Strip UUID prefix (36 chars + underscore)
                        if len(fname) > 37 and fname[36] == "_":
                            fname = fname[37:]
                        all_sources.add(fname)
                if next_offset is None:
                    break
                offset = next_offset
            return sorted(list(all_sources))
        except Exception as e:
            logger.error(f"list_sources failed: {e}")
            return []