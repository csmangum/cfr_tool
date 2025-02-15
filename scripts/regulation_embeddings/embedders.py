"""Module for handling document embedding strategies."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    """Abstract base class for document embedding strategies."""

    @abstractmethod
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text chunks."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generates embeddings using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        # Suppress the flash attention warning
        warnings.filterwarnings("ignore", message=".*flash attention.*")

        self.model = SentenceTransformer(model_name)
        if device:
            self.model = self.model.to(device)
        self.batch_size = batch_size
        self.device = device
        # Get embedding dimension dynamically from model
        self.base_dim = self.model.get_sentence_embedding_dimension()

    def _encode_metadata_field(self, field_values: List[str]) -> np.ndarray:
        """Helper method to encode a metadata field.

        Args:
            field_values: List of strings for a metadata field

        Returns:
            numpy array: Embedding for the field or zeros if empty
        """
        field_text = " ".join(field_values) if field_values else ""
        if not field_text:
            return np.zeros(self.base_dim)
        return self.model.encode(
            field_text, batch_size=1, show_progress_bar=False, convert_to_numpy=True
        )

    def embed_text_with_metadata(
        self, text: str, metadata: Dict[str, List[str]], enrich: bool = False
    ) -> np.ndarray:
        """Generate embedding, optionally enriched with metadata fields.

        Args:
            text: The text to embed
            metadata: Dictionary of metadata fields where each value is a list of strings
            enrich: Whether to include metadata in the embedding

        Returns:
            numpy array: Normalized embedding vector
        """
        text_embedding = self.model.encode(
            text, batch_size=1, show_progress_bar=False, convert_to_numpy=True
        )

        if not enrich:
            return text_embedding / np.linalg.norm(text_embedding)

        # Process metadata fields
        metadata_fields = {
            "cross_references": metadata.get("cross_references", []),
            "definitions": metadata.get("definitions", []),
            "enforcement_agencies": metadata.get("enforcement_agencies", []),
        }

        # Generate embeddings for each metadata field
        field_embeddings = [
            self._encode_metadata_field(field_values)
            for field_values in metadata_fields.values()
        ]

        # Concatenate base embedding with metadata embeddings
        enriched_embedding = np.concatenate([text_embedding] + field_embeddings)

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
            device=self.device,
        )
        return embeddings

    def close(self) -> None:
        """Clean up resources used by the embedder."""
        if hasattr(self, "model"):
            # Clear CUDA cache if using GPU
            if self.device and "cuda" in self.device:
                import torch

                torch.cuda.empty_cache()

            # Clear model from memory
            self.model = None
