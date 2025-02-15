"""
Regulation Text Embedder

This script processes regulation text files from agencies, chunks them into meaningful segments,
and creates embeddings for similarity search. It stores the embeddings and metadata in a SQLite 
database for later retrieval and searching.

The script handles:
- Reading regulation text files from agency directories
- Chunking documents into meaningful segments
- Creating embeddings using sentence-transformers
- Storing embeddings and metadata in SQLite
"""

import argparse
import logging
import os
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

# Suppress various warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Update existing environment settings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("FAISS_CPU_ONLY", "1")
os.environ["TORCH_ALLOW_TF32"] = "1"  # Better performance on Ampere+ GPUs

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from scripts.regulation_embeddings.config import Config
from scripts.regulation_embeddings.models import Base, BaseRegulationChunk
from scripts.regulation_embeddings.pipeline import ProcessingError, RegulationProcessor


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    return logger


def create_db(db_url: str):
    """Create the SQLite database and initialize tables."""
    # Get logger instance
    logger = logging.getLogger(__name__)

    logger.info("Creating/connecting to database")

    # Create database directory if it doesn't exist
    db_path = Path(db_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")
    return engine


def extract_metadata(filename):
    """
    Extract title, chapter, and date from regulation filename.

    Args:
        filename (str): Name of the regulation file

    Returns:
        tuple: (title, chapter, date) as strings, or (None, None, None) if not found
    """
    title_chapter_match = re.search(r"title_(\d+)_chapter_([^_]+)", filename)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)

    title = title_chapter_match.group(1) if title_chapter_match else None
    chapter = title_chapter_match.group(2) if title_chapter_match else None
    date = date_match.group(1) if date_match else None

    return title, chapter, date


def chunk_regulation_xml(xml_path: str) -> List[Tuple[str, dict]]:
    """
    Split regulation XML into meaningful chunks based on the document structure.

    Args:
        xml_path (str): Path to the XML file

    Returns:
        list: List of tuples containing (chunk_text, metadata)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks = []

    def clean_text(text: str) -> str:
        """Clean up text by removing extra whitespace and normalizing quotes."""
        if text is None:
            return ""
        return " ".join(text.split()).replace('"', '"').replace('"', '"')

    def process_section(section) -> dict:
        """Extract metadata from a section element."""
        metadata = {}
        # Get section number from HEAD
        head = section.find("HEAD")
        if head is not None:
            section_num = head.text.strip().replace("§", "").strip()
            metadata["section"] = section_num

        # Get hierarchy metadata if available
        hierarchy = section.get("hierarchy_metadata")
        if hierarchy:
            metadata["hierarchy"] = hierarchy

        # Extract additional metadata fields
        cross_references = section.findall(".//CROSSREF")
        if cross_references:
            metadata["cross_references"] = [
                clean_text(ref.text) for ref in cross_references if ref.text
            ]

        definitions = section.findall(".//DEF")
        if definitions:
            metadata["definitions"] = [
                clean_text(defn.text) for defn in definitions if defn.text
            ]

        enforcement_agencies = section.findall(".//ENFORCEMENT")
        if enforcement_agencies:
            metadata["enforcement_agencies"] = [
                clean_text(agency.text)
                for agency in enforcement_agencies
                if agency.text
            ]

        last_revision = section.find(".//LASTREV")
        if last_revision is not None:
            metadata["date_of_last_revision"] = clean_text(last_revision.text)

        intent = section.find(".//INTENT")
        if intent is not None:
            metadata["regulatory_intent"] = clean_text(intent.text)

        return metadata

    def extract_text_with_context(element) -> str:
        """Extract text from an element including its header for context."""
        texts = []
        head = element.find("HEAD")
        if head is not None:
            texts.append(head.text.strip())

        for child in element:
            if child.tag == "P":
                if child.text:
                    texts.append(clean_text(child.text))
            elif child.tag not in ["HEAD"]:  # Skip headers as they're already processed
                if child.text:
                    texts.append(clean_text(child.text))

        return " ".join(texts)

    # Process each major division (PART, SECTION, etc.)
    for div in root.findall(".//DIV8"):  # Find all section-level divisions
        section_metadata = process_section(div)

        # Create chunks based on paragraphs or subsections
        paragraphs = div.findall(".//P")
        if paragraphs:
            current_chunk = []
            current_length = 0
            max_chunk_length = 1000  # Adjust this value as needed

            for p in paragraphs:
                text = clean_text(p.text) if p.text else ""

                # If adding this paragraph would exceed max length, save current chunk
                if current_length + len(text) > max_chunk_length and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, section_metadata))
                    current_chunk = []
                    current_length = 0

                current_chunk.append(text)
                current_length += len(text)

            # Add any remaining text as final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, section_metadata))
        else:
            # If no paragraphs found, create one chunk from the whole section
            text = extract_text_with_context(div)
            if text.strip():
                chunks.append((text, section_metadata))

    return chunks


def process_regulations(
    config_path: Path,
    data_dir: Optional[Path] = None,
    single_file: Optional[Path] = None,
) -> None:
    """Process regulation files using the configured pipeline.

    Args:
        config_path: Path to configuration file
        data_dir: Optional override for data directory
        single_file: Optional path to process a single file

    Raises:
        ProcessingError: If processing fails
    """
    try:
        # Load configuration
        config = Config.from_yaml(config_path)

        # Override data directory if specified
        if data_dir:
            config.processing.data_dir = data_dir

        # Set up logging
        logger = setup_logging(config.processing.log_file)
        logger.info("Starting regulation processing")
        logger.info(f"Using configuration from {config_path}")

        # Initialize processor with context manager for cleanup
        with RegulationProcessor(config) as processor:
            if single_file:
                logger.info(f"Processing single file: {single_file}")
                processor.process_file(single_file)
            else:
                logger.info(f"Processing regulations from {config.processing.data_dir}")
                processor.process_directory(config.processing.data_dir)

            logger.info("Processing completed successfully")

    except ProcessingError as e:
        logger.error(f"Processing error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise ProcessingError(f"Failed to process regulations: {str(e)}") from e


def search_similar_chunks(query_text: str, db_url: str, n_results: int = 5):
    """
    Search for regulation chunks similar to the query text using cosine similarity
    and efficient batch processing.

    Args:
        query_text (str): The search query
        db_url (str): Database connection URL
        n_results (int): Number of results to return

    Returns:
        list: Top n_results similar chunks with metadata
    """
    # Load model and create embedding for query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query_text, normalize_embeddings=True)

    # Connect to database
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Process chunks in batches to avoid memory issues
        batch_size = 1000
        results = []

        # Get total count for progress tracking
        total_chunks = session.query(BaseRegulationChunk).count()

        for offset in tqdm(range(0, total_chunks, batch_size), desc="Searching chunks"):
            # Get batch of chunks
            chunks = (
                session.query(BaseRegulationChunk)
                .offset(offset)
                .limit(batch_size)
                .all()
            )

            # Convert embeddings to numpy array
            chunk_embeddings = np.vstack(
                [np.frombuffer(chunk.embedding, dtype=np.float32) for chunk in chunks]
            )

            # Normalize embeddings for cosine similarity
            chunk_embeddings = (
                chunk_embeddings
                / np.linalg.norm(chunk_embeddings, axis=1)[:, np.newaxis]
            )

            # Calculate cosine similarities for the batch
            similarities = np.dot(chunk_embeddings, query_embedding)

            # Add results with metadata
            batch_results = [
                (
                    chunks[i].chunk_text,
                    float(similarities[i]),
                    {
                        "agency": chunks[i].agency,
                        "title": chunks[i].title,
                        "chapter": chunks[i].chapter,
                        "date": chunks[i].date,
                        "section": chunks[i].section,
                        "hierarchy": chunks[i].hierarchy,
                    },
                )
                for i in range(len(chunks))
            ]

            results.extend(batch_results)

            # Keep only top n_results so far to save memory
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:n_results]

        return results

    finally:
        session.close()


def main():
    """Run the regulation embedding pipeline."""
    parser = argparse.ArgumentParser(
        description="Process and embed regulation documents."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single file instead of directory",
    )

    args = parser.parse_args()

    try:
        process_regulations(
            config_path=args.config,
            data_dir=args.data_dir,
            single_file=args.file,
        )
    except ProcessingError as e:
        print(f"Error: {str(e)}")
        exit(1)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        exit(130)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
