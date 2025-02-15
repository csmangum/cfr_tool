"""Vector store implementations using FAISS."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .vector_store_base import VectorStore


class FaissStore(VectorStore):
    """FAISS-based vector store implementation for efficient similarity search.

    This class implements a vector store using Facebook AI Similarity Search (FAISS).
    It supports adding vectors with metadata, searching for similar vectors, and
    persistence to disk.

    Attributes:
        index: FAISS index for vector storage and search
        metadata: Dictionary mapping vector IDs to their metadata
    """

    def __init__(self, dimension: Optional[int] = 1536) -> None:
        """Initialize FAISS store.

        Args:
            dimension: Dimensionality of the vectors to be stored. If None,
                      initialization is deferred until first vector addition.
        """
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.metadata = {}

        if dimension is not None:
            # Initialize a new FAISS index with the dimension size
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)

            self.logger.info(f"Initialized FAISS store with dimension {dimension}")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """Add vectors and their metadata to the store.

        Args:
            vectors: Array of embedding vectors with shape (n_vectors, dimension)
            metadata: List of metadata dictionaries, one per vector

        Raises:
            ValueError: If vectors and metadata lengths don't match
            Exception: If FAISS operations fail
        """
        try:
            if len(vectors) != len(metadata):
                raise ValueError(
                    "Number of vectors must match number of metadata entries"
                )

            if self.index is None:
                dimension = vectors.shape[1]
                base_index = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIDMap(base_index)

            # Generate sequential IDs more efficiently
            start_id = len(self.metadata)
            ids = np.arange(start_id, start_id + len(vectors), dtype=np.int64)

            # Add vectors to FAISS index
            self.index.add_with_ids(vectors.astype(np.float32), ids)

            # Store metadata with integer keys
            self.metadata.update(zip(ids, metadata))

            self.logger.debug(f"Added {len(vectors)} vectors to FAISS index")

        except Exception as e:
            self.logger.error(f"Error adding vectors to FAISS: {str(e)}")
            raise

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        """Search for vectors similar to the query vector.

        Args:
            query_vector: Query embedding vector with shape (dimension,) or (1, dimension)
            k: Number of nearest neighbors to return

        Returns:
            List of tuples containing (similarity_score, metadata), sorted by
            decreasing similarity. Similarity is computed as 1/(1 + L2_distance).

        Raises:
            ValueError: If index is not initialized
            Exception: If FAISS operations fail
        """
        try:
            if self.index is None:
                raise ValueError("FAISS index not initialized")

            # Ensure query vector is 2D
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            # Convert to float32 for FAISS
            query_vector = query_vector.astype(np.float32)

            # Search the index
            distances, indices = self.index.search(query_vector, k)

            # Format results
            formatted_results = []
            for distance, idx in zip(distances[0], indices[0]):
                if (
                    idx == -1
                ):  # FAISS returns -1 for padding when there are fewer results
                    continue

                # Convert L2 distance to similarity score (1 / (1 + distance))
                similarity = 1 / (1 + distance)
                metadata = self.metadata.get(int(idx), {})
                formatted_results.append((similarity, metadata))

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching FAISS: {str(e)}")
            raise

    def save(self, path: Path) -> None:
        """Save the vector store to disk.

        Saves both the FAISS index and metadata to the specified directory.

        Args:
            path: Directory path where the store should be saved

        Raises:
            ValueError: If index is not initialized
            Exception: If save operations fail
        """
        try:
            if self.index is None:
                raise ValueError("No index to save")

            # Create directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = path / "index.faiss"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            import json

            metadata_path = path / "metadata.json"
            with open(metadata_path, "w") as f:
                # Convert integer keys to strings for JSON
                serializable_metadata = {str(k): v for k, v in self.metadata.items()}
                json.dump(serializable_metadata, f)

            self.logger.info(f"Saved FAISS store to {path}")

        except Exception as e:
            self.logger.error(f"Error saving FAISS store: {str(e)}")
            raise

    def load(self, path: Path) -> None:
        """Load a vector store from disk.

        Loads both the FAISS index and metadata from the specified directory.

        Args:
            path: Directory path containing the saved store

        Raises:
            FileNotFoundError: If index or metadata files are missing
            Exception: If load operations fail
        """
        try:
            index_path = path / "index.faiss"
            metadata_path = path / "metadata.json"

            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found at {metadata_path}")

            # Load FAISS index
            self.index = faiss.read_index(str(index_path))

            # Load metadata
            import json

            with open(metadata_path) as f:
                string_metadata = json.load(f)
                self.metadata = {int(k): v for k, v in string_metadata.items()}

            self.logger.info(f"Loaded FAISS store from {path}")

        except Exception as e:
            self.logger.error(f"Error loading FAISS store: {str(e)}")
            raise
