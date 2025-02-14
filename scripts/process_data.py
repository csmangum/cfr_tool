"""
eCFR Data Processor

This module processes downloaded federal regulations from the Electronic Code of Federal 
Regulations (eCFR) and calculates various readability metrics. It stores the results in 
a SQLite database for further analysis.

Example usage:
    from process_data import process_agencies

    # Process all downloaded agency regulations
    process_agencies()

The module calculates various metrics including:
- Basic statistics (word count, sentence count, syllable count)
- Readability scores (Flesch Reading Ease, Gunning Fog, SMOG Index, etc.)
- Complexity measures (type-token ratio, polysyllabic words, etc.)

The processed data is stored in a SQLite database with the following structure:
    data/
    ├── db/
    │   └── regulations.db         # SQLite database with metrics
    └── logs/
        └── process_data.log      # Processing logs

The database schema includes detailed metrics for each regulation version,
with timestamps and agency identification.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import textstat
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sentence_transformers import SentenceTransformer
import numpy as np


def create_directories():
    """
    Create required directories for logs and database storage.

    Creates the following directory structure if it doesn't exist:
        data/
        ├── logs/    # For log files
        └── db/      # For SQLite database
    """
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/db").mkdir(parents=True, exist_ok=True)


# Create directories before setting up logging
create_directories()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/process_data.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Create SQLAlchemy base class
Base = declarative_base()


class RegulationMetrics(Base):
    """
    SQLAlchemy model representing regulation metrics in the database.

    Stores various readability and complexity metrics for each regulation version,
    including basic statistics, readability scores, and advanced complexity measures.

    Attributes:
        id (int): Primary key
        agency (str): Agency identifier/slug
        title (str): CFR title number
        chapter (str): Chapter number within title
        date (str): Version date of the regulation
        word_count (int): Total number of words
        flesch_reading_ease (float): Flesch Reading Ease score
        flesch_kincaid_grade (float): Flesch-Kincaid Grade Level
        gunning_fog (float): Gunning Fog Index
        smog_index (float): SMOG Index
        automated_readability_index (float): Automated Readability Index
        coleman_liau_index (float): Coleman-Liau Index
        linsear_write (float): Linsear Write Formula score
        dale_chall (float): Dale-Chall Readability score
        difficult_words (int): Count of difficult words
        sentence_count (int): Total number of sentences
        avg_sentence_length (float): Average words per sentence
        syllable_count (int): Total syllable count
        avg_syllables_per_word (float): Average syllables per word
        type_token_ratio (float): Ratio of unique words to total words
        polysyllabic_words (int): Count of words with 3+ syllables
        metadata_embedding (String): Metadata embedding as a string
        created_at (datetime): Timestamp of record creation
    """

    __tablename__ = "regulation_metrics"

    id = Column(Integer, primary_key=True)
    agency = Column(String)
    title = Column(String)
    chapter = Column(String)
    date = Column(String)
    word_count = Column(Integer)
    flesch_reading_ease = Column(Float)
    flesch_kincaid_grade = Column(Float)
    gunning_fog = Column(Float)
    smog_index = Column(Float)
    automated_readability_index = Column(Float)
    coleman_liau_index = Column(Float)
    linsear_write = Column(Float)
    dale_chall = Column(Float)
    difficult_words = Column(Integer)
    sentence_count = Column(Integer)
    avg_sentence_length = Column(Float)
    syllable_count = Column(Integer)
    avg_syllables_per_word = Column(Float)
    type_token_ratio = Column(Float)
    polysyllabic_words = Column(Integer)
    metadata_embedding = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_db():
    """
    Create the SQLite database and initialize tables.

    Returns:
        sqlalchemy.engine.Engine: Database engine instance
    """
    logger.info("Creating/connecting to database")
    engine = create_engine("sqlite:///data/db/regulations.db")
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")
    return engine


def extract_title_chapter(filename):
    """
    Extract title and chapter numbers from a regulation filename.

    Args:
        filename (str): Name of the regulation file

    Returns:
        tuple: (title, chapter) numbers as strings, or (None, None) if not found
    """
    match = re.search(r"title_(\d+)_chapter_([^_]+)", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def calculate_metrics(text):
    """
    Calculate comprehensive readability and complexity metrics for regulation text.

    Args:
        text (str): Plain text content of the regulation

    Returns:
        dict: Dictionary containing all calculated metrics including:
            - Basic statistics (word count, sentence count)
            - Readability scores (Flesch, Gunning Fog, etc.)
            - Complexity measures (type-token ratio, polysyllabic words)
    """
    word_count = len(text.split())
    sentence_count = textstat.sentence_count(text)
    syllable_count = textstat.syllable_count(text)

    # Calculate additional metrics:
    avg_sentence_length = word_count / sentence_count if sentence_count != 0 else 0
    avg_syllables_per_word = syllable_count / word_count if word_count != 0 else 0
    unique_words = len(set(text.split()))
    type_token_ratio = unique_words / word_count if word_count != 0 else 0

    # Count polysyllabic words (words with 3 or more syllables)
    polysyllabic_words = sum(
        1 for word in text.split() if textstat.syllable_count(word) >= 3
    )

    return {
        "word_count": word_count,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "linsear_write": textstat.linsear_write_formula(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
        "difficult_words": textstat.difficult_words(text),
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "syllable_count": syllable_count,
        "avg_syllables_per_word": avg_syllables_per_word,
        "type_token_ratio": type_token_ratio,
        "polysyllabic_words": polysyllabic_words,
    }


def process_agencies():
    """
    Process all downloaded agency regulations and store metrics in database.

    Walks through the agency directories, processes each regulation text file,
    calculates metrics, and stores results in the SQLite database. Handles errors
    gracefully and logs processing status.

    The function expects regulations to be in the following structure:
        data/agencies/{agency-slug}/text/*.txt
    """
    # Create directories first
    create_directories()

    logger.info("Starting agency regulation processing")
    engine = create_db()
    Session = sessionmaker(bind=engine)
    session = Session()

    data_dir = Path("data/agencies")
    logger.info(f"Reading from directory: {data_dir}")

    total_files = 0
    processed_files = 0

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for agency_dir in data_dir.iterdir():
        if not agency_dir.is_dir():
            continue

        agency_name = agency_dir.name
        text_dir = agency_dir / "text"

        if not text_dir.exists():
            logger.warning(f"No text directory found for agency: {agency_name}")
            continue

        logger.info(f"Processing agency: {agency_name}")

        for text_file in text_dir.glob("*.txt"):
            total_files += 1
            try:
                # Read text content
                text = text_file.read_text(encoding="utf-8")
                logger.debug(f"Processing file: {text_file.name}")

                # Extract title and chapter
                title, chapter = extract_title_chapter(text_file.name)
                if not title or not chapter:
                    logger.warning(
                        f"Could not extract title/chapter from: {text_file.name}"
                    )

                # Extract date from filename
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text_file.name)
                date = date_match.group(1) if date_match else None
                if not date:
                    logger.warning(f"Could not extract date from: {text_file.name}")

                # Calculate metrics
                metrics = calculate_metrics(text)

                # Generate text embedding
                text_vector = model.encode(text)

                # Generate metadata embeddings
                metadata = {}  # Placeholder for actual metadata extraction logic
                cross_ref_vec = model.encode(metadata.get("cross_references", [""])[0])
                definition_vec = model.encode(metadata.get("definitions", [""])[0])
                authority_vec = model.encode(metadata.get("authority", ""))

                # Merge metadata embeddings
                metadata_embedding = np.mean(
                    [cross_ref_vec, definition_vec, authority_vec], axis=0
                )
                final_embedding = np.concatenate([text_vector, metadata_embedding])

                # Create database record
                record = RegulationMetrics(
                    agency=agency_name,
                    title=title,
                    chapter=chapter,
                    date=date,
                    metadata_embedding=final_embedding.tolist(),  # Convert to list for storage
                    **metrics,
                )

                session.add(record)
                processed_files += 1

            except Exception as e:
                logger.error(f"Error processing file {text_file}: {str(e)}")
                continue

        # Commit after processing each agency
        try:
            session.commit()
            logger.info(f"Committed records for agency: {agency_name}")
        except Exception as e:
            logger.error(f"Error committing records for {agency_name}: {str(e)}")
            session.rollback()

    logger.info(
        f"Processing complete. Processed {processed_files} of {total_files} files"
    )
    session.close()


if __name__ == "__main__":
    process_agencies()
