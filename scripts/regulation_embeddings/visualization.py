"""Module for visualizing regulation embeddings."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple
import seaborn as sns
from pathlib import Path
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import RegulationChunk
from .config import Config

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
            chunks = session.query(RegulationChunk).all()
            
            # Convert binary embeddings to numpy arrays
            embeddings = np.vstack([
                np.frombuffer(chunk.embedding, dtype=np.float32)
                for chunk in chunks
            ])
            
            # Collect metadata
            metadata = [{
                'agency': chunk.agency,
                'title': chunk.title,
                'chapter': chunk.chapter,
                'date': chunk.date,
                'section': chunk.section
            } for chunk in chunks]
            
            return embeddings, metadata
            
        finally:
            session.close()

    def plot_embeddings(self, output_path: Path = None):
        """Create PCA visualization of embeddings colored by agency."""
        # Get embeddings and metadata
        embeddings, metadata = self.get_embeddings_and_metadata()
        
        # Perform PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Get unique agencies and assign colors
        agencies = list(set(m['agency'] for m in metadata))
        color_palette = sns.color_palette('husl', n_colors=len(agencies))
        agency_to_color = dict(zip(agencies, color_palette))
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot points
        for agency in agencies:
            mask = [m['agency'] == agency for m in metadata]
            agency_points = embeddings_2d[mask]
            
            plt.scatter(
                agency_points[:, 0],
                agency_points[:, 1],
                c=[agency_to_color[agency]],
                label=agency,
                alpha=0.6
            )
        
        # Customize plot
        plt.title('Regulation Embeddings by Agency (PCA)')
        plt.xlabel(f'PC1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2%})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

        return embeddings_2d, pca.explained_variance_ratio_ 