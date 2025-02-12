import logging
import re
from datetime import datetime
from pathlib import Path

import textstat
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# Create necessary directories first
def create_directories():
    """Create required directories if they don't exist"""
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
    created_at = Column(DateTime, default=datetime.utcnow)


def create_db():
    """Create the SQLite database and tables"""
    logger.info("Creating/connecting to database")
    engine = create_engine("sqlite:///data/db/regulations.db")
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")
    return engine


def extract_title_chapter(filename):
    """Extract title and chapter from filename"""
    match = re.search(r"title_(\d+)_chapter_([^_]+)", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def calculate_metrics(text):
    """Calculate various text metrics including additional complexity measures"""
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
    """Process all agency regulation files and store metrics"""
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

                # Create database record
                record = RegulationMetrics(
                    agency=agency_name,
                    title=title,
                    chapter=chapter,
                    date=date,
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
