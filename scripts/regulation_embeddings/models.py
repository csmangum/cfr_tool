"""SQLAlchemy models for regulation data."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BaseRegulationChunk(Base):
    """Model for storing regulation chunks with basic embeddings."""
    
    __tablename__ = "regulation_chunks"

    id = Column(Integer, primary_key=True)
    agency = Column(String)
    title = Column(String)
    chapter = Column(String)
    date = Column(String)
    chunk_text = Column(String)
    chunk_index = Column(Integer)
    embedding = Column(LargeBinary)  # Stores base embedding dimension
    section = Column(String)
    hierarchy = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# class EnrichedRegulationChunk(Base):
#     """Model for storing regulation chunks with enriched embeddings."""
    
#     __tablename__ = "regulation_chunks"

#     id = Column(Integer, primary_key=True)
#     agency = Column(String)
#     title = Column(String)
#     chapter = Column(String)
#     date = Column(String)
#     chunk_text = Column(String)
#     chunk_index = Column(Integer)
#     embedding = Column(LargeBinary)  # Stores 4x base embedding dimension
#     section = Column(String)
#     hierarchy = Column(String)
#     cross_references = Column(String, nullable=True)  # Stored as JSON array
#     definitions = Column(String, nullable=True)       # Stored as JSON array
#     authority = Column(String, nullable=True)         # Stored as JSON array
#     created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc)) 