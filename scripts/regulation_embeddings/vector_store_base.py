"""Base protocol for vector stores."""

from typing import List, Dict, Tuple, Protocol
from pathlib import Path
import numpy as np


class VectorStore(Protocol):
    """Protocol defining interface for vector stores."""
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """
        Add vectors and metadata to the store.
        
        Args:
            vectors: Array of embedding vectors
            metadata: List of metadata dictionaries for each vector
        """
        ...
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples containing (similarity_score, metadata)
        """
        ...
    
    def save(self, path: Path) -> None:
        """Save vector store to disk."""
        ...
    
    def load(self, path: Path) -> None:
        """Load vector store from disk."""
        ... 