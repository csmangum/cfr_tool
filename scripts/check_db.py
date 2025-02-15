"""Script to check database structure and content."""

import logging
from pathlib import Path
import sqlite3
import sys

from regulation_embeddings.config import Config
from regulation_embeddings.models import RegulationChunk
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler("data/logs/db_check.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_sqlite_directly(db_path: str):
    """Check SQLite database directly."""
    logger = logging.getLogger(__name__)
    
    try:
        # Remove sqlite:/// prefix if present
        if db_path.startswith('sqlite:///'):
            db_path = db_path[10:]
            
        logger.info(f"Checking SQLite database at: {db_path}")
        
        # Check if file exists
        if not Path(db_path).exists():
            logger.error(f"Database file does not exist: {db_path}")
            return False
            
        # Try to connect directly with sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Found tables: {[t[0] for t in tables]}")
        
        # Check regulation_chunks table
        cursor.execute("PRAGMA table_info(regulation_chunks);")
        columns = cursor.fetchall()
        logger.info("Table structure:")
        for col in columns:
            logger.info(f"  {col[1]}: {col[2]}")
            
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM regulation_chunks;")
        count = cursor.fetchone()[0]
        logger.info(f"Total rows: {count}")
        
        # Check sample row
        cursor.execute("SELECT * FROM regulation_chunks LIMIT 1;")
        sample = cursor.fetchone()
        if sample:
            logger.info("Sample row found")
            logger.info(f"  ID: {sample[0]}")
            logger.info(f"  Agency: {sample[1]}")
            logger.info(f"  Embedding size: {len(sample[6]) if sample[6] else 'None'} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking SQLite database: {str(e)}", exc_info=True)
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def check_sqlalchemy_connection(db_url: str):
    """Check database using SQLAlchemy."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Checking database with SQLAlchemy: {db_url}")
        
        # Create engine
        engine = create_engine(db_url)
        
        # Get inspector
        inspector = inspect(engine)
        
        # Check tables
        tables = inspector.get_table_names()
        logger.info(f"Found tables via SQLAlchemy: {tables}")
        
        # Check regulation_chunks columns
        columns = inspector.get_columns('regulation_chunks')
        logger.info("Table structure via SQLAlchemy:")
        for col in columns:
            logger.info(f"  {col['name']}: {col['type']}")
        
        # Try a session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check count
        count = session.query(RegulationChunk).count()
        logger.info(f"Total rows via SQLAlchemy: {count}")
        
        # Check sample
        sample = session.query(RegulationChunk).first()
        if sample:
            logger.info("Sample row found via SQLAlchemy:")
            logger.info(f"  ID: {sample.id}")
            logger.info(f"  Agency: {sample.agency}")
            logger.info(f"  Embedding size: {len(sample.embedding) if sample.embedding else 'None'} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking database via SQLAlchemy: {str(e)}", exc_info=True)
        return False
    finally:
        if 'session' in locals():
            session.close()

def main():
    """Run database checks."""
    logger = setup_logging()
    logger.info("Starting database checks...")
    
    try:
        # Load config
        config = Config.from_yaml(Path("config.yaml"))
        db_url = config.database.db_url
        
        # Run checks
        sqlite_ok = check_sqlite_directly(db_url)
        sqlalchemy_ok = check_sqlalchemy_connection(db_url)
        
        if sqlite_ok and sqlalchemy_ok:
            logger.info("All database checks passed successfully")
        else:
            logger.error("Some database checks failed")
            
    except Exception as e:
        logger.error(f"Error during database checks: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 