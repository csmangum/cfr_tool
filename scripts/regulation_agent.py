"""
Regulation Search and QA Agent

This module implements an LLM-powered agent that can search through regulation embeddings
and provide answers to questions using the OpenAI API. The agent combines semantic search
over regulation embeddings with GPT-4 to provide accurate, contextual answers about federal
regulations.

Features:
- Semantic search using sentence transformers embeddings
- Context-aware responses using OpenAI's GPT models
- Source citations for answers
- Diagnostic tools for system verification
- Configurable search parameters and filters

Example usage:
    from regulation_agent import RegulationAgent
    
    # Initialize the agent
    agent = RegulationAgent()
    
    # Run diagnostics to verify system
    agent.print_diagnostics()
    
    # Ask a question about regulations
    result = agent.answer_question(
        "What are the requirements for filing a FOIA request?",
        n_results=5  # Number of relevant chunks to consider
    )
    
    print(result["answer"])  # Print the answer
    
    # Print sources
    for _, score, metadata in result["search_results"]:
        print(f"Source: {metadata['agency']}, Title {metadata['title']}, "
              f"Chapter {metadata['chapter']}, Section {metadata.get('section', 'N/A')}")

Requirements:
- OpenAI API key (set in .env file as OPENAI_API_KEY)
- Pre-populated SQLite database with regulation embeddings (created by embed_regulations.py)
- Python packages: openai, sentence-transformers, sqlalchemy, numpy, python-dotenv

Database Schema:
The agent expects a SQLite database with regulation embeddings stored in the following format:
- Table: regulation_chunks
- Columns:
  - id: Primary key
  - agency: Agency identifier
  - title: CFR title number
  - chapter: Chapter number
  - date: Version date
  - chunk_text: The regulation text
  - chunk_index: Index within document
  - embedding: Binary vector embedding
  - section: Section number
  - hierarchy: JSON metadata about document structure

Configuration:
The agent can be configured with the following parameters:
- openai_api_key: OpenAI API key (defaults to environment variable)
- model_name: OpenAI model to use (default: gpt-4o-mini)
- embedding_model: SentenceTransformers model (default: all-MiniLM-L6-v2)
- db_url: Database URL (default: sqlite:///data/db/regulation_embeddings.db)
- temperature: OpenAI temperature setting (default: 0.0)
- max_tokens: Maximum response tokens (default: 1000)

The module includes diagnostic tools to verify:
- Database connectivity and content
- Embedding model functionality
- Search capability
- OpenAI API access
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from regulation_embeddings.models import RegulationChunk
from regulation_embeddings.storage import DatabaseManager
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/regulation_agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class RegulationAgent:
    """
    An LLM-powered agent for searching and answering questions about regulations.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        db_url: str = "sqlite:///data/db/regulation_embeddings.db",
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ):
        """
        Initialize the regulation agent.

        Args:
            openai_api_key: OpenAI API key (defaults to environment variable)
            model_name: OpenAI model to use
            embedding_model: SentenceTransformers model for embeddings
            db_url: URL for the regulation embeddings database
            temperature: Temperature for OpenAI API responses
            max_tokens: Maximum tokens in OpenAI API responses
        """
        # Initialize OpenAI client with API key from .env
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env file"
            )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize database connection
        self.db_manager = DatabaseManager(db_url=db_url)

        # System prompt template
        self.system_prompt = """You are a helpful assistant that answers questions about federal regulations. 
        You will be provided with relevant regulation excerpts to help answer the user's question.
        Always cite the specific agency, title, chapter, and section when providing information.
        If you're unsure about something, say so rather than making assumptions.
        Format your responses in a clear, easy-to-read manner."""

        # Add database verification
        try:
            engine = create_engine(db_url)
            Session = sessionmaker(bind=engine)
            session = Session()
            count = session.query(RegulationChunk).count()
            logger.info(f"Connected to database. Found {count} regulation chunks.")
            session.close()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for the search query."""
        return self.embedding_model.encode(query)

    def _search_regulations(
        self, query: str, n_results: int = 5, filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for relevant regulation chunks.

        Args:
            query: Search query
            n_results: Number of results to return
            filters: Optional filters for agency, title, etc.

        Returns:
            List of tuples containing (chunk_text, similarity_score, metadata)
        """
        try:
            query_embedding = self._create_query_embedding(query)
            results = self.db_manager.search_similar(
                query_embedding, n_results=n_results, filters=filters
            )

            if not results:
                logger.warning(f"No results found for query: {query}")
            else:
                logger.info(f"Found {len(results)} results for query: {query}")

            return results

        except Exception as e:
            logger.error(f"Error searching regulations: {e}")
            raise

    def _format_context(self, search_results: List[Tuple[str, float, Dict]]) -> str:
        """Format search results into context for the LLM."""
        context_parts = []

        for i, (text, score, metadata) in enumerate(search_results, 1):
            context_part = f"\nExcerpt {i}:\n"
            context_part += f"Agency: {metadata['agency']}\n"
            context_part += (
                f"Title: {metadata['title']}, Chapter: {metadata['chapter']}"
            )
            if metadata.get("section"):
                context_part += f", Section: {metadata['section']}"
            context_part += f"\nText: {text}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def answer_question(
        self, question: str, n_results: int = 5, filters: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Answer a question about regulations using relevant context.

        Args:
            question: User's question
            n_results: Number of regulation chunks to retrieve
            filters: Optional filters for search

        Returns:
            Dict containing answer and search results
        """
        try:
            # Search for relevant regulations
            search_results = self._search_regulations(
                question, n_results=n_results, filters=filters
            )

            if not search_results:
                return {
                    "answer": "I couldn't find any relevant regulations to answer your question.",
                    "search_results": [],
                }

            # Format context from search results
            context = self._format_context(search_results)

            # Create messages for OpenAI
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"""Please answer this question about regulations: {question}

Here are the relevant regulation excerpts to help you answer:
{context}

Please provide a clear answer, citing specific regulations where appropriate.""",
                },
            ]

            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return {
                "answer": response.choices[0].message.content,
                "search_results": search_results,
            }

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise

    def _run_diagnostics(self) -> Dict[str, any]:
        """
        Run diagnostic checks on database and search functionality.

        Returns:
            Dict containing diagnostic results
        """
        diagnostics = {}

        # Check database connection and contents
        try:
            engine = create_engine(self.db_manager.db_url)
            Session = sessionmaker(bind=engine)
            session = Session()

            # Get basic stats
            chunk_count = session.query(RegulationChunk).count()
            agency_count = session.query(RegulationChunk.agency).distinct().count()

            # Get sample of available agencies
            agencies = session.query(RegulationChunk.agency).distinct().limit(5).all()
            sample_agencies = [agency[0] for agency in agencies]

            # Get date range
            date_range = session.query(
                func.min(RegulationChunk.date), func.max(RegulationChunk.date)
            ).first()

            diagnostics["database"] = {
                "status": "connected",
                "total_chunks": chunk_count,
                "unique_agencies": agency_count,
                "sample_agencies": sample_agencies,
                "date_range": date_range,
            }

            # Test embedding generation
            try:
                test_query = "Test query for embedding"
                embedding = self._create_query_embedding(test_query)
                diagnostics["embedding"] = {
                    "status": "working",
                    "shape": embedding.shape,
                    "mean": float(embedding.mean()),
                    "std": float(embedding.std()),
                }
            except Exception as e:
                diagnostics["embedding"] = {"status": "error", "error": str(e)}

            # Test search functionality with a sample query
            try:
                test_results = self._search_regulations(
                    "FOIA request requirements", n_results=1
                )
                diagnostics["search"] = {
                    "status": "working",
                    "results_found": len(test_results) > 0,
                    "sample_result": test_results[0] if test_results else None,
                }
            except Exception as e:
                diagnostics["search"] = {"status": "error", "error": str(e)}

            session.close()

        except Exception as e:
            diagnostics["database"] = {"status": "error", "error": str(e)}

        return diagnostics

    def print_diagnostics(self):
        """Print formatted diagnostic information."""
        diagnostics = self._run_diagnostics()

        print("\n=== Regulation Agent Diagnostics ===\n")

        # Database status
        print("Database Status:")
        db_info = diagnostics.get("database", {})
        if db_info.get("status") == "connected":
            print(f"  ✓ Connected")
            print(f"  • Total chunks: {db_info['total_chunks']:,}")
            print(f"  • Unique agencies: {db_info['unique_agencies']}")
            print(f"  • Sample agencies: {', '.join(db_info['sample_agencies'])}")
            if db_info.get("date_range"):
                print(
                    f"  • Date range: {db_info['date_range'][0]} to {db_info['date_range'][1]}"
                )
        else:
            print(f"  ✗ Error: {db_info.get('error', 'Unknown error')}")

        # Embedding status
        print("\nEmbedding Status:")
        embed_info = diagnostics.get("embedding", {})
        if embed_info.get("status") == "working":
            print(f"  ✓ Working")
            print(f"  • Embedding shape: {embed_info['shape']}")
            print(f"  • Mean: {embed_info['mean']:.3f}")
            print(f"  • Std: {embed_info['std']:.3f}")
        else:
            print(f"  ✗ Error: {embed_info.get('error', 'Unknown error')}")

        # Search status
        print("\nSearch Status:")
        search_info = diagnostics.get("search", {})
        if search_info.get("status") == "working":
            print(f"  ✓ Working")
            print(f"  • Test query returned results: {search_info['results_found']}")
            if search_info.get("sample_result"):
                text, score, metadata = search_info["sample_result"]
                print(f"  • Sample result score: {score:.3f}")
                print(f"  • Sample result agency: {metadata.get('agency')}")
        else:
            print(f"  ✗ Error: {search_info.get('error', 'Unknown error')}")

        print("\n" + "=" * 35 + "\n")


def main():
    """Example usage of the RegulationAgent."""
    try:
        # Verify database exists
        db_path = Path("data/db/regulation_embeddings.db")
        if not db_path.exists():
            print(f"Database not found at {db_path}")
            print(
                "Please run embed_regulations.py first to create and populate the database."
            )
            return

        # Initialize agent
        agent = RegulationAgent()

        # Run diagnostics
        agent.print_diagnostics()

        # Prompt user to continue
        response = input("Continue with example questions? (y/n): ")
        if response.lower() != "y":
            return

        # Example questions
        questions = [
            "What are the requirements for filing a FOIA request?",
            "What are the safety requirements for commercial vehicles?",
            "How are medical devices classified by the FDA?",
        ]

        # Answer each question
        for question in questions:
            print(f"\nQuestion: {question}")
            print("-" * 80)

            try:
                result = agent.answer_question(question)
                print("\nAnswer:")
                print(result["answer"])

                print("\nSources:")
                if result["search_results"]:
                    for _, score, metadata in result["search_results"]:
                        print(
                            f"- {metadata['agency']}, Title {metadata['title']}, "
                            f"Chapter {metadata['chapter']}, Section {metadata.get('section', 'N/A')}"
                        )
                        print(f"  Relevance score: {score:.3f}")
                else:
                    print("No relevant sources found")

            except Exception as e:
                print(f"Error: {str(e)}")

            print("-" * 80)

    except Exception as e:
        print(f"Initialization error: {str(e)}")


if __name__ == "__main__":
    main()
