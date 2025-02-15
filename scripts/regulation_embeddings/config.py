"""Configuration management for regulation embeddings."""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, DirectoryPath


class ChunkerConfig(BaseModel):
    """Configuration for document chunking."""
    max_chunk_length: int = Field(
        default=1000,
        description="Maximum length of text chunks in characters"
    )
    xml_tag_depth: str = Field(
        default=".//DIV8",
        description="XPath expression for finding document divisions"
    )


class EmbedderConfig(BaseModel):
    """Configuration for document embedding."""
    model_name: str = Field(
        default='all-MiniLM-L6-v2',
        description="Name of the sentence-transformer model to use"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device to use for embedding (cpu, cuda, etc.)"
    )


class DatabaseConfig(BaseModel):
    """Configuration for database storage."""
    db_url: str = Field(
        default="sqlite:///data/db/regulation_embeddings.db",
        description="SQLAlchemy database URL"
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for database operations"
    )


class ProcessingConfig(BaseModel):
    """Configuration for regulation processing."""
    data_dir: Path = Field(
        default=Path("data/agencies"),
        description="Root directory containing agency data"
    )
    xml_pattern: str = Field(
        default="*/xml/*.xml",
        description="Glob pattern for finding XML files"
    )
    log_file: str = Field(
        default="data/logs/embed_regulations.log",
        description="Path to log file"
    )


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    collection_name: str = Field(
        default="regulations",
        description="Name of the ChromaDB collection"
    )
    persist_directory: str = Field(
        default="data/chroma",
        description="Directory to persist ChromaDB data"
    )


class Config(BaseModel):
    """Main configuration for regulation embeddings."""
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        
        # Create default config if file doesn't exist
        if not path.exists():
            return cls()
            
        with open(path) as f:
            data = yaml.safe_load(f)
            if data is None:  # Empty file
                return cls()
        return cls.parse_obj(data) 