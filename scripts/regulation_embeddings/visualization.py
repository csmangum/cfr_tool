"""Module for visualizing regulation embeddings."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import Config
from .models import BaseRegulationChunk


class EmbeddingVisualizer:
    """Class for visualizing regulation embeddings."""

    def __init__(self, config: Config):
        """Initialize visualizer with database configuration."""
        self.engine = create_engine(config.database.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)

    def get_embeddings_and_metadata(self) -> Tuple[np.ndarray, List[Dict]]:
        """Retrieve all embeddings and metadata from database."""
        session = self.Session()
        try:
            chunks = session.query(BaseRegulationChunk).all()

            # Convert binary embeddings to numpy arrays
            embeddings = np.vstack(
                [np.frombuffer(chunk.embedding, dtype=np.float32) for chunk in chunks]
            )

            # Collect metadata
            metadata = [
                {
                    "agency": chunk.agency,
                    "title": chunk.title,
                    "chapter": chunk.chapter,
                    "date": chunk.date,
                    "section": chunk.section,
                }
                for chunk in chunks
            ]

            return embeddings, metadata

        finally:
            session.close()

    def plot_embeddings(self, output_path: Path = None):
        """Create visualization of embeddings with metadata components."""
        embeddings, metadata = self.get_embeddings_and_metadata()

        # Split enriched embeddings into components
        base_embeddings = embeddings[:, :384]
        cross_refs_embeddings = embeddings[:, 384:768]
        definitions_embeddings = embeddings[:, 768:1152]
        authority_embeddings = embeddings[:, 1152:]

        # Create separate PCA visualizations for each component
        components = [
            ("Base Text", base_embeddings),
            ("Cross References", cross_refs_embeddings),
            ("Definitions", definitions_embeddings),
            ("Authority", authority_embeddings),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.ravel()

        for ax, (name, component_embeddings) in zip(axes, components):
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(component_embeddings)

            # Plot points colored by agency
            for agency in set(m["agency"] for m in metadata):
                mask = [m["agency"] == agency for m in metadata]
                ax.scatter(reduced[mask, 0], reduced[mask, 1], alpha=0.6, label=agency)

            ax.set_title(f"{name} Embeddings")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        plt.close()
