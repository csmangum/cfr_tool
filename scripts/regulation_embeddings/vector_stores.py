"""Vector store implementations using FAISS."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .vector_store_base import VectorStore


class FaissStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, dimension: int = 1536):  # Updated: 384 * 4 for enriched embeddings
        """Initialize FAISS store with dimension for enriched embeddings."""
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.metadata = {}
        
        if dimension is not None:
            # Initialize a new FAISS index with the enriched dimension size
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)
            
            self.logger.info(f"Initialized FAISS store with enriched dimension {dimension}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """
        Add vectors and metadata to the store.
        
        Args:
            vectors: Array of embedding vectors (n_vectors x dimension)
            metadata: List of metadata dictionaries for each vector
        """
        try:
            if self.index is None:
                # Initialize index if not already done
                dimension = vectors.shape[1]
                base_index = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIDMap(base_index)
            
            # Generate IDs for the vectors
            start_id = len(self.metadata)
            ids = np.array([i for i in range(start_id, start_id + len(vectors))], dtype=np.int64)
            
            # Add vectors to FAISS index
            self.index.add_with_ids(vectors.astype(np.float32), ids)
            
            # Store metadata separately
            for i, meta in zip(ids, metadata):
                self.metadata[int(i)] = meta
            
            self.logger.debug(f"Added {len(vectors)} vectors to FAISS index")
            
        except Exception as e:
            self.logger.error(f"Error adding vectors to FAISS: {str(e)}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples containing (similarity_score, metadata)
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
                if idx == -1:  # FAISS returns -1 for padding when there are fewer results
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
        """
        Save vector store to disk.
        
        Args:
            path: Path to save directory
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
            with open(metadata_path, 'w') as f:
                # Convert integer keys to strings for JSON
                serializable_metadata = {str(k): v for k, v in self.metadata.items()}
                json.dump(serializable_metadata, f)
            
            self.logger.info(f"Saved FAISS store to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving FAISS store: {str(e)}")
            raise
    
    def load(self, path: Path) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Path to load directory
        """
        try:
            # Load FAISS index
            index_path = path / "index.faiss"
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            import json
            metadata_path = path / "metadata.json"
            with open(metadata_path) as f:
                # Convert string keys back to integers
                string_metadata = json.load(f)
                self.metadata = {int(k): v for k, v in string_metadata.items()}
            
            self.logger.info(f"Loaded FAISS store from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading FAISS store: {str(e)}")
            raise 