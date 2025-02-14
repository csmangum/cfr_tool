"""Module for handling document embedding strategies."""

import warnings
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for document embedding strategies."""
    
    @abstractmethod
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text chunks."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 batch_size: int = 32,
                 device: Optional[str] = None):
        # Suppress the flash attention warning
        warnings.filterwarnings("ignore", message=".*flash attention.*")
        
        self.model = SentenceTransformer(model_name)
        if device:
            self.model = self.model.to(device)
        self.batch_size = batch_size
        self.device = device
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for chunks using batched processing."""
        # Use model's built-in batching
        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        return embeddings 