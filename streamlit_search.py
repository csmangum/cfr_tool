"""
Streamlit app for searching regulations using semantic similarity.

Run with: streamlit run scripts/streamlit_search.py
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.search_regulations import SAMPLE_QUESTIONS, RegulationSearcher

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []


def initialize_searcher() -> RegulationSearcher:
    """Initialize the regulation searcher with default paths."""
    # Use session state to cache the searcher
    if "searcher" not in st.session_state:
        st.session_state.searcher = RegulationSearcher(
            index_path="data/faiss/regulation_index.faiss",
            metadata_path="data/faiss/regulation_metadata.json",
            db_path="data/db/regulation_embeddings.db",
            model_name="all-MiniLM-L6-v2",
        )
    return st.session_state.searcher


def format_result(metadata: Dict, score: float, chunk_text: str) -> str:
    """Format a single search result."""
    agency = metadata.get("agency", "Unknown Agency")
    title = metadata.get("title", "Unknown Title")
    chapter = metadata.get("chapter", "Unknown Chapter")
    section = metadata.get("section", "N/A")
    date = metadata.get("date", "Unknown Date")

    return {
        "Score": f"{score:.3f}",
        "Agency": agency,
        "Title": title,
        "Chapter": chapter,
        "Section": section,
        "Date": date,
        "Text": chunk_text,
    }


def format_result_text(result: Dict) -> str:
    """Format a result dictionary as text."""
    return (
        f"Agency: {result['Agency']}\n"
        f"Title: {result['Title']}\n"
        f"Chapter: {result['Chapter']}\n"
        f"Section: {result['Section']}\n"
        f"Date: {result['Date']}\n"
        f"Score: {result['Score']}\n\n"
        f"Text:\n{result['Text']}"
    )


def main():
    st.set_page_config(page_title="Regulation Search", page_icon="📚", layout="wide")

    st.title("📚 Regulation Search")
    st.markdown(
        """
    Ask questions about federal regulations and get relevant answers based on semantic search.
    """
    )

    # Initialize searcher once
    try:
        searcher = initialize_searcher()
    except Exception as e:
        st.error(f"Error initializing search: {str(e)}")
        return

    # Sidebar
    with st.sidebar:
        st.header("Search Options")
        num_results = st.slider("Number of results", 1, 20, 5)

        # Add filters
        st.header("Filters")
        with st.expander("Agency Filters", expanded=False):
            # Get unique agencies from metadata
            agencies = sorted(
                set(
                    metadata.get("agency", "Unknown")
                    for metadata in searcher.metadata.values()
                )
            )
            selected_agencies = st.multiselect(
                "Select Agencies", options=agencies, default=None
            )

        with st.expander("Date Range", expanded=False):
            date_range = st.date_input(
                "Select Date Range",
                value=[pd.Timestamp.now().date(), pd.Timestamp.now().date()],
                help="Filter regulations by date",
            )

        # Similarity threshold
        min_score = st.slider(
            "Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Filter results by minimum similarity score",
        )

        st.header("Sample Questions")
        if st.button("Try a random question"):
            random_question = random.choice(SAMPLE_QUESTIONS)
            st.session_state.random_question = random_question
            # Force a rerun to update the main input
            st.rerun()

        st.header("Search History")
        if st.session_state.history:
            for i, past_query in enumerate(reversed(st.session_state.history[-10:])):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"• {past_query}")
                with col2:
                    if st.button("Rerun", key=f"rerun_{i}"):
                        st.session_state.random_question = past_query
                        st.rerun()

            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    # Main search interface
    query = st.text_input(
        "Enter your question about regulations:",
        value=st.session_state.get("random_question", ""),
        key="search_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search")
    with col2:
        clear_button = st.button("🗑️ Clear")

    if clear_button:
        st.session_state.random_question = ""
        st.rerun()

    # Add CSS for better styling
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
        }
        .search-box {
            margin-bottom: 2rem;
        }
        .result-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Add keyboard shortcut for search
    if query and (search_button or st.session_state.get("enter_pressed")):
        # Add to history
        if query not in st.session_state.history:
            st.session_state.history.append(query)

        try:
            with st.spinner("Searching regulations..."):
                start_time = time.time()
                # Simplified search call with only supported parameters
                results = searcher.search(query, n_results=num_results)

                # Filter results after search if min_score is set
                if min_score > 0:
                    results = [r for r in results if r[1] >= min_score]

                search_time = time.time() - start_time
                st.info(f"Search completed in {search_time:.2f} seconds")

            if results:
                st.success(f"Found {len(results)} relevant regulations")

                # Add export options
                export_format = st.radio(
                    "Export Format", options=["CSV", "JSON", "TXT"], horizontal=True
                )

                if st.button("Export Results"):
                    export_data = []
                    for metadata_item, score, chunk_text in results:
                        result_dict = format_result(metadata_item, score, chunk_text)
                        export_data.append(result_dict)

                    if export_format == "CSV":
                        df = pd.DataFrame(export_data)
                        st.download_button(
                            "Download CSV",
                            df.to_csv(index=False),
                            "regulation_results.csv",
                            "text/csv",
                        )
                    elif export_format == "JSON":
                        st.download_button(
                            "Download JSON",
                            json.dumps(export_data, indent=2),
                            "regulation_results.json",
                            "application/json",
                        )
                    else:  # TXT
                        text_output = "\n\n".join(
                            [
                                f"Result {i+1}:\n" + format_result_text(r)
                                for i, r in enumerate(export_data)
                            ]
                        )
                        st.download_button(
                            "Download TXT",
                            text_output,
                            "regulation_results.txt",
                            "text/plain",
                        )

                for idx, (metadata_item, score, chunk_text) in enumerate(results, 1):
                    with st.expander(
                        f"Result {idx} (Score: {score:.3f})", expanded=True
                    ):
                        result_dict = format_result(metadata_item, score, chunk_text)

                        # Display metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Agency:** {result_dict['Agency']}")
                            st.markdown(f"**Title:** {result_dict['Title']}")
                            st.markdown(f"**Chapter:** {result_dict['Chapter']}")
                        with col2:
                            st.markdown(f"**Section:** {result_dict['Section']}")
                            st.markdown(f"**Date:** {result_dict['Date']}")
                            st.markdown(f"**Score:** {result_dict['Score']}")

                        # Display text in a box with word wrap
                        st.markdown("**Regulation Text:**")
                        st.markdown(
                            f"""<div style='background-color: #1E1E1E; color: #E0E0E0; padding: 1rem; 
                            border-radius: 0.5rem; white-space: pre-wrap; word-wrap: break-word; 
                            font-family: monospace; line-height: 1.5;'>
{result_dict['Text'].strip()}
</div>""",
                            unsafe_allow_html=True,
                        )

                # Add visualization tab
                with st.expander("📊 Visualizations", expanded=False):
                    # Score distribution
                    scores = [score for _, score, _ in results]
                    fig = px.histogram(
                        scores,
                        title="Distribution of Similarity Scores",
                        labels={
                            "value": "Similarity Score",
                            "count": "Number of Results",
                        },
                    )
                    st.plotly_chart(fig)

                    # Agency breakdown
                    agencies = [metadata.get("agency") for metadata, _, _ in results]
                    fig = px.pie(names=agencies, title="Results by Agency")
                    st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error performing search: {str(e)}")
            st.error("Please try refining your search terms or adjusting filters")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit and FAISS vector search</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
