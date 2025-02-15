"""Module for handling data storage and retrieval."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    LargeBinary,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import sessionmaker

from .models import Base, BaseRegulationChunk
from .vector_store_base import VectorStore


class DatabaseManager:
    """Manages database operations for regulation chunks and embeddings."""

    def __init__(
        self,
        db_url: str,
        batch_size: int = 100,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize database and vector store connections."""
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.batch_size = batch_size
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

        # Create indices for faster querying
        self._create_indices()

    def _create_indices(self):
        """Create database indices for better query performance."""
        with self.engine.connect() as conn:
            # Index for agency lookups
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_agency "
                    "ON regulation_chunks (agency)"
                )
            )
            # Index for date-based queries
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_date " "ON regulation_chunks (date)"
                )
            )
            # Index for section lookups
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_section "
                    "ON regulation_chunks (section)"
                )
            )

    def _serialize_metadata(self, metadata: Dict) -> str:
        """Serialize metadata dictionary to JSON string."""
        if metadata is None:
            return None
        try:
            return json.dumps(metadata)
        except Exception as e:
            self.logger.warning(f"Failed to serialize metadata: {e}")
            return None

    def _deserialize_metadata(self, metadata_str: str) -> Dict:
        """Deserialize metadata string back to dictionary."""
        if metadata_str is None:
            return None
        try:
            return json.loads(metadata_str)
        except Exception as e:
            self.logger.warning(f"Failed to deserialize metadata: {e}")
            return None

    def store_chunks(
        self,
        chunks: List[Tuple[str, Dict]],
        embeddings: np.ndarray,
        metadata: Dict[str, str],
    ) -> None:
        """Store chunks and their embeddings."""
        session = self.Session()

        try:
            batch_size = min(50, len(chunks))

            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))

                # Prepare batch records with enriched metadata
                batch_records = [
                    BaseRegulationChunk(
                        agency=metadata["agency"],
                        title=metadata["title"],
                        chapter=metadata["chapter"],
                        date=metadata["date"],
                        chunk_text=chunk_text,
                        embedding=embeddings[j].tobytes(),
                        section=section_metadata.get("section"),
                        hierarchy=self._serialize_metadata(
                            section_metadata.get("hierarchy")
                        ),
                        # cross_references=json.dumps(section_metadata.get('cross_references', [])),
                        # definitions=json.dumps(section_metadata.get('definitions', [])),
                        # authority=json.dumps(section_metadata.get('enforcement_agencies', []))
                    )
                    for j, (chunk_text, section_metadata) in enumerate(
                        chunks[i:batch_end], start=i
                    )
                ]

                session.bulk_save_objects(batch_records)
                session.commit()
                session.expunge_all()

            # Vector store handling in separate try block
            if self.vector_store is not None:
                self.logger.info("Adding chunks to vector store")
                try:
                    # Process vector store in very small batches
                    vector_batch_size = 10  # Very small batch size

                    for i in range(0, len(chunks), vector_batch_size):
                        batch_end = min(i + vector_batch_size, len(chunks))

                        # Prepare metadata for batch
                        batch_metadata = [
                            {
                                "chunk_text": chunk_text,
                                "agency": metadata["agency"],
                                "title": metadata["title"],
                                "chapter": metadata["chapter"],
                                "date": metadata["date"],
                                "section": section_metadata.get("section"),
                                "hierarchy": section_metadata.get("hierarchy"),
                            }
                            for chunk_text, section_metadata in chunks[i:batch_end]
                        ]

                        try:
                            # Add vectors for this batch
                            self.vector_store.add_vectors(
                                embeddings[i:batch_end], batch_metadata
                            )
                            self.logger.debug(
                                f"Vector store batch {i//vector_batch_size + 1} added successfully"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed to add batch to vector store: {str(e)}"
                            )
                            # Continue with next batch instead of failing completely
                            continue

                except Exception as e:
                    self.logger.error(f"Error in vector store operations: {str(e)}")
                    # Don't raise - allow SQL storage to succeed even if vector store fails

        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing chunks: {str(e)}")
            raise
        finally:
            session.close()

    def search_similar(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar chunks with optional filtering."""
        # If vector store is available, use it for similarity search
        if self.vector_store is not None:
            results = self.vector_store.search(query_embedding, k=n_results)
            return [(meta["chunk_text"], score, meta) for score, meta in results]

        # Fallback to database search
        session = self.Session()
        try:
            # Build query with filters
            query = session.query(BaseRegulationChunk)
            if filters:
                for field, value in filters.items():
                    if hasattr(BaseRegulationChunk, field):
                        query = query.filter(
                            getattr(BaseRegulationChunk, field) == value
                        )

            results = []
            # Calculate similarities
            for chunk in query.all():
                chunk_embedding = np.frombuffer(chunk.embedding, dtype=np.float32)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )

                # Deserialize hierarchy when returning results
                hierarchy = self._deserialize_metadata(chunk.hierarchy)

                results.append(
                    (
                        chunk.chunk_text,
                        similarity,
                        {
                            "agency": chunk.agency,
                            "title": chunk.title,
                            "chapter": chunk.chapter,
                            "date": chunk.date,
                            "section": chunk.section,
                            "hierarchy": hierarchy,
                        },
                    )
                )

            # Sort by similarity and return top n_results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:n_results]

        finally:
            session.close()

    def backup_embeddings(self, backup_path: Path) -> None:
        """Backup embeddings to HDF5 file for persistence."""
        session = self.Session()
        try:
            chunks = session.query(BaseRegulationChunk).all()

            with h5py.File(backup_path, "w") as f:
                # Store embeddings
                embeddings = np.vstack(
                    [
                        np.frombuffer(chunk.embedding, dtype=np.float32)
                        for chunk in chunks
                    ]
                )
                f.create_dataset("embeddings", data=embeddings)

                # Store metadata
                dt = h5py.special_dtype(vlen=str)
                metadata = np.array(
                    [
                        (
                            chunk.agency,
                            chunk.title,
                            chunk.chapter,
                            chunk.date,
                            chunk.section,
                            chunk.hierarchy,
                        )
                        for chunk in chunks
                    ],
                    dtype=dt,
                )
                f.create_dataset("metadata", data=metadata)

        finally:
            session.close()

    def store_chunks_sql_only(
        self,
        chunks: List[Tuple[str, Dict]],
        embeddings: np.ndarray,
        metadata: Dict[str, str],
    ) -> None:
        """Store chunks and their embeddings in SQL database only."""
        session = self.Session()
        try:
            self.logger.info(f"Starting database transaction for {len(chunks)} chunks")

            # Process in smaller batches to manage memory
            batch_size = min(50, len(chunks))

            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))

                # Prepare batch records
                batch_records = [
                    BaseRegulationChunk(
                        agency=metadata["agency"],
                        title=metadata["title"],
                        chapter=metadata["chapter"],
                        date=metadata["date"],
                        chunk_text=chunk_text,
                        chunk_index=section_metadata.get("chunk_index", i),
                        embedding=embeddings[j].tobytes(),
                        section=section_metadata.get("section"),
                        hierarchy=self._serialize_metadata(
                            section_metadata.get("hierarchy")
                        ),
                        # cross_references=json.dumps(section_metadata.get('cross_references', [])),
                        # definitions=json.dumps(section_metadata.get('definitions', [])),
                        # authority=json.dumps(section_metadata.get('enforcement_agencies', []))
                    )
                    for j, (chunk_text, section_metadata) in enumerate(
                        chunks[i:batch_end], start=i
                    )
                ]

                # Insert batch
                self.logger.debug(f"Inserting batch of {len(batch_records)} chunks")
                session.bulk_save_objects(batch_records)
                session.commit()

                # Clear SQLAlchemy session to free memory
                session.expunge_all()

        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing chunks: {str(e)}")
            raise
        finally:
            session.close()
