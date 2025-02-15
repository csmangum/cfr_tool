"""SQLAlchemy models for regulation data."""

from datetime import datetime
from sqlalchemy import Column, DateTime, String, LargeBinary, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RegulationChunk(Base):
    """Model for storing regulation chunks and their enriched embeddings."""
    
    __tablename__ = "regulation_chunks"

    id = Column(Integer, primary_key=True)
    agency = Column(String)
    title = Column(String)
    chapter = Column(String)
    date = Column(String)
    chunk_text = Column(String)
    chunk_index = Column(Integer)
    embedding = Column(LargeBinary)  # Now stores 1536-dimensional embeddings
    section = Column(String)
    hierarchy = Column(String)
    cross_references = Column(String)  # Store as JSON list
    definitions = Column(String)       # Store as JSON list
    authority = Column(String)         # Store as JSON list
    created_at = Column(DateTime, default=datetime.utcnow) 