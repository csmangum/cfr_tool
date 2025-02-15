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
        self.base_dim = 384  # Base dimension for all-MiniLM-L6-v2
    
    def embed_text_with_metadata(self, 
                               text: str,
                               metadata: Dict[str, List[str]]) -> np.ndarray:
        """Generate enriched embedding including metadata fields."""
        # Get base text embedding
        text_embedding = self.model.encode(text, 
                                         batch_size=1,
                                         show_progress_bar=False,
                                         convert_to_numpy=True)
        
        # Generate embeddings for metadata fields
        cross_refs = " ".join(metadata.get("cross_references", []))
        definitions = " ".join(metadata.get("definitions", []))
        authority = " ".join(metadata.get("enforcement_agencies", []))
        
        # Create embeddings for non-empty metadata fields
        cross_refs_embedding = self.model.encode(cross_refs) if cross_refs else np.zeros(self.base_dim)
        definitions_embedding = self.model.encode(definitions) if definitions else np.zeros(self.base_dim)
        authority_embedding = self.model.encode(authority) if authority else np.zeros(self.base_dim)
        
        # Concatenate all embeddings
        enriched_embedding = np.concatenate([
            text_embedding,
            cross_refs_embedding,
            definitions_embedding,
            authority_embedding
        ])
        
        # Normalize the final embedding
        return enriched_embedding / np.linalg.norm(enriched_embedding)

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