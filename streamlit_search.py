"""
Streamlit app for searching regulations using semantic similarity.

Run with: streamlit run scripts/streamlit_search.py
"""

import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Counter, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.search_regulations import SAMPLE_QUESTIONS, RegulationSearcher

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Add after the imports, before initialize_searcher()
RELEVANCE_SCORES = {
    "Not Relevant": 0,
    "Somewhat Relevant": 1,
    "Relevant": 2,
    "Very Relevant": 3,
}

QUALITY_SCORES = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize additional session state for evaluations
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []


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


def format_result(metadata: Dict, score: float, chunk_text: str) -> Dict:
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


def save_evaluation(
    query: str, results: List[Dict], ratings: Dict[int, Dict], feedback: str
):
    """Save evaluation data to a JSON file."""
    try:
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results": results,
            "ratings": ratings,
            "feedback": feedback,
        }

        st.session_state.evaluations.append(evaluation)

        # Create full path to evaluations directory
        project_root = Path(__file__).parent.parent
        eval_dir = project_root / "data" / "evaluations"
        
        # Create directories if they don't exist
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        eval_file = eval_dir / "search_evaluations.json"
        
        # Load existing evaluations
        existing_evals = []
        if eval_file.exists():
            try:
                with eval_file.open("r", encoding="utf-8") as f:
                    existing_evals = json.load(f)
            except json.JSONDecodeError:
                st.warning(f"Could not read existing evaluations from {eval_file}. Starting fresh.")
        
        # Append new evaluation
        existing_evals.append(evaluation)

        # Save updated evaluations
        with eval_file.open("w", encoding="utf-8") as f:
            json.dump(existing_evals, f, indent=2, ensure_ascii=False)
            
        st.success(f"Evaluation saved to {eval_file}")
        
    except Exception as e:
        st.error(f"Error saving evaluation: {str(e)}")
        st.error(f"Attempted to save to: {eval_file if 'eval_file' in locals() else 'unknown path'}")


def calculate_query_stats(evaluations: List[Dict]) -> Dict:
    """Calculate statistics about query performance."""
    query_stats = defaultdict(
        lambda: {
            "count": 0,
            "avg_relevance": 0.0,
            "avg_quality": 0.0,
            "total_ratings": 0,
        }
    )

    for eval in evaluations:
        query = eval["query"]
        query_stats[query]["count"] += 1

        relevance_sum = sum(
            RELEVANCE_SCORES[r["relevance"]] for r in eval["ratings"].values()
        )
        quality_sum = sum(
            QUALITY_SCORES[r["quality"]] for r in eval["ratings"].values()
        )
        num_ratings = len(eval["ratings"])

        current_stats = query_stats[query]
        current_stats["total_ratings"] += num_ratings
        current_stats["avg_relevance"] = (
            current_stats["avg_relevance"] * (current_stats["count"] - 1)
            + relevance_sum / num_ratings
        ) / current_stats["count"]
        current_stats["avg_quality"] = (
            current_stats["avg_quality"] * (current_stats["count"] - 1)
            + quality_sum / num_ratings
        ) / current_stats["count"]

    return dict(query_stats)


def analyze_feedback_themes(evaluations: List[Dict]) -> Counter:
    """Analyze common themes in feedback."""
    themes = Counter()

    # Keywords to look for in feedback
    theme_keywords = {
        "irrelevant": ["irrelevant", "unrelated", "wrong", "not relevant"],
        "outdated": ["outdated", "old", "expired", "not current"],
        "incomplete": ["incomplete", "partial", "missing", "not enough"],
        "helpful": ["helpful", "useful", "good", "excellent"],
        "unclear": ["unclear", "confusing", "vague", "hard to understand"],
    }

    for eval in evaluations:
        # Check overall feedback
        feedback_text = eval["feedback"].lower()

        # Check specific feedback for each result
        for rating in eval["ratings"].values():
            if rating["feedback"]:
                feedback_text += " " + rating["feedback"].lower()

        # Count themes
        for theme, keywords in theme_keywords.items():
            if any(keyword in feedback_text for keyword in keywords):
                themes[theme] += 1

    return themes


def auto_evaluate_result(result: Dict, query: str) -> Dict:
    """Automatically evaluate search result using an LLM."""
    try:
        # Format prompt for the LLM
        prompt = f"""Evaluate this search result for relevance and quality:
Query: {query}
Text: {result['Text']}
Metadata: Agency: {result['Agency']}, Title: {result['Title']}, Section: {result['Section']}

Rate using these scales:
Relevance: Not Relevant, Somewhat Relevant, Relevant, Very Relevant
Quality: Poor, Fair, Good, Excellent

Return your response in this format:
Relevance: [rating]
Quality: [rating]
Explanation: [your explanation]"""

        # Get LLM response
        response = st.session_state.searcher.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return {
            "auto_relevance": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result
        }
    except Exception as e:
        st.error(f"Auto-evaluation failed: {str(e)}")
        return None


def save_ai_evaluation(ai_rating: Dict):
    """Save AI evaluation results to a separate file."""
    try:
        project_root = Path(__file__).parent.parent
        eval_dir = project_root / "data" / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        ai_eval_file = eval_dir / "ai_evaluations.json"
        
        # Load existing evaluations
        existing_evals = []
        if ai_eval_file.exists():
            try:
                with ai_eval_file.open("r", encoding="utf-8") as f:
                    existing_evals = json.load(f)
            except json.JSONDecodeError:
                st.warning(f"Could not read existing AI evaluations. Starting fresh.")
        
        # Append new evaluation
        existing_evals.append(ai_rating)

        # Save updated evaluations
        with ai_eval_file.open("w", encoding="utf-8") as f:
            json.dump(existing_evals, f, indent=2, ensure_ascii=False)
            
        st.success(f"AI evaluation saved to {ai_eval_file}")
        
    except Exception as e:
        st.error(f"Error saving AI evaluation: {str(e)}")


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

        auto_evaluate = st.toggle("Enable AI Evaluation", value=False)

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

                # Add evaluation interface
                st.markdown("### Search Quality Evaluation")
                st.markdown(
                    "Please help us improve search quality by rating the results:"
                )

                # Store ratings in session state
                if "current_ratings" not in st.session_state:
                    st.session_state.current_ratings = {}

                for idx, (metadata_item, score, chunk_text) in enumerate(results, 1):
                    with st.expander(
                        f"Result {idx} (Score: {score:.3f})", expanded=True
                    ):
                        # Display result content
                        result_dict = format_result(metadata_item, score, chunk_text)

                        # Display metadata and content
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Agency:** {result_dict['Agency']}")
                            st.markdown(f"**Title:** {result_dict['Title']}")
                            st.markdown(f"**Chapter:** {result_dict['Chapter']}")
                        with col2:
                            st.markdown(f"**Section:** {result_dict['Section']}")
                            st.markdown(f"**Date:** {result_dict['Date']}")
                            st.markdown(f"**Score:** {result_dict['Score']}")

                        st.markdown("**Regulation Text:**")
                        st.markdown(
                            f"""<div style='background-color: #1E1E1E; color: #E0E0E0; padding: 1rem; 
                            border-radius: 0.5rem; white-space: pre-wrap; word-wrap: break-word; 
                            font-family: monospace; line-height: 1.5;'>
{result_dict['Text'].strip()}
</div>""",
                            unsafe_allow_html=True,
                        )

                        # Add rating interface
                        st.markdown("#### Rate this result")
                        col1, col2 = st.columns(2)

                        with col1:
                            relevance = st.select_slider(
                                f"Relevance for Result {idx}",
                                options=[
                                    "Not Relevant",
                                    "Somewhat Relevant",
                                    "Relevant",
                                    "Very Relevant",
                                ],
                                key=f"relevance_{idx}",
                            )

                        with col2:
                            quality = st.select_slider(
                                f"Content Quality for Result {idx}",
                                options=["Poor", "Fair", "Good", "Excellent"],
                                key=f"quality_{idx}",
                            )

                        specific_feedback = st.text_area(
                            "Specific feedback for this result (optional)",
                            key=f"feedback_{idx}",
                        )

                        # Store ratings in session state
                        st.session_state.current_ratings[idx] = {
                            "relevance": relevance,
                            "quality": quality,
                            "feedback": specific_feedback,
                        }

                # Overall feedback
                st.markdown("### Overall Feedback")
                overall_feedback = st.text_area(
                    "Please provide any overall feedback about the search results",
                    key="overall_feedback",
                )

                # Submit evaluation
                if st.button("Submit Evaluation"):
                    save_evaluation(
                        query=query,
                        results=[format_result(m, s, t) for m, s, t in results],
                        ratings=st.session_state.current_ratings,
                        feedback=overall_feedback,
                    )
                    st.success(
                        "Thank you for your feedback! Your evaluation has been saved."
                    )

                    # Clear current ratings
                    st.session_state.current_ratings = {}

                # Add enhanced evaluation statistics
                if st.session_state.evaluations:
                    with st.expander(
                        "View Detailed Evaluation Statistics", expanded=False
                    ):
                        st.markdown("### Search Quality Analysis")

                        # Basic metrics
                        total_evaluations = len(st.session_state.evaluations)
                        total_ratings = sum(
                            len(eval["ratings"])
                            for eval in st.session_state.evaluations
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Evaluations", total_evaluations)
                        with col2:
                            st.metric("Total Ratings", total_ratings)
                        with col3:
                            st.metric(
                                "Avg Ratings per Query",
                                f"{total_ratings/total_evaluations:.1f}",
                            )

                        # Query Performance Analysis
                        st.markdown("### Query Performance")
                        query_stats = calculate_query_stats(
                            st.session_state.evaluations
                        )

                        # Convert to DataFrame for visualization
                        query_df = pd.DataFrame(
                            [
                                {
                                    "Query": query,
                                    "Count": stats["count"],
                                    "Avg Relevance": stats["avg_relevance"],
                                    "Avg Quality": stats["avg_quality"],
                                    "Total Ratings": stats["total_ratings"],
                                }
                                for query, stats in query_stats.items()
                            ]
                        )

                        # Sort by average relevance
                        query_df = query_df.sort_values(
                            "Avg Relevance", ascending=False
                        )

                        # Show query performance table
                        st.dataframe(
                            query_df.style.format(
                                {"Avg Relevance": "{:.2f}", "Avg Quality": "{:.2f}"}
                            )
                        )

                        # Feedback Theme Analysis
                        st.markdown("### Feedback Themes")
                        themes = analyze_feedback_themes(st.session_state.evaluations)

                        if themes:
                            theme_df = pd.DataFrame(
                                [
                                    {"Theme": theme, "Count": count}
                                    for theme, count in themes.most_common()
                                ]
                            )

                            theme_chart = (
                                alt.Chart(theme_df)
                                .mark_bar()
                                .encode(
                                    x="Count:Q",
                                    y=alt.Y("Theme:N", sort="-x"),
                                    tooltip=["Theme", "Count"],
                                )
                                .properties(
                                    title="Common Feedback Themes",
                                    height=min(len(themes) * 40, 300),
                                )
                            )

                            st.altair_chart(theme_chart, use_container_width=True)

                        # Time-based Analysis
                        st.markdown("### Quality Trends Over Time")

                        trend_data = []
                        for eval in st.session_state.evaluations:
                            timestamp = pd.to_datetime(eval["timestamp"])
                            avg_relevance = np.mean(
                                [
                                    RELEVANCE_SCORES[r["relevance"]]
                                    for r in eval["ratings"].values()
                                ]
                            )
                            avg_quality = np.mean(
                                [
                                    QUALITY_SCORES[r["quality"]]
                                    for r in eval["ratings"].values()
                                ]
                            )

                            trend_data.append(
                                {
                                    "Date": timestamp,
                                    "Score": avg_relevance,
                                    "Type": "Relevance",
                                }
                            )
                            trend_data.append(
                                {
                                    "Date": timestamp,
                                    "Score": avg_quality,
                                    "Type": "Quality",
                                }
                            )

                        trend_df = pd.DataFrame(trend_data)

                        if not trend_df.empty:
                            trend_chart = (
                                alt.Chart(trend_df)
                                .mark_line(point=True)
                                .encode(
                                    x="Date:T",
                                    y="Score:Q",
                                    color="Type:N",
                                    tooltip=["Date", "Score", "Type"],
                                )
                                .properties(title="Quality Metrics Over Time")
                            )

                            st.altair_chart(trend_chart, use_container_width=True)

                        # Export Evaluation Data
                        st.markdown("### Export Evaluation Data")
                        if st.button("Download Evaluation Data"):
                            eval_df = pd.json_normalize(st.session_state.evaluations)
                            csv = eval_df.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                "regulation_search_evaluations.csv",
                                "text/csv",
                                key="download_eval",
                            )

                # Add in results display
                if auto_evaluate:
                    ai_rating = auto_evaluate_result(result_dict, query)
                    if ai_rating:
                        st.markdown("#### AI Evaluation")
                        st.write(ai_rating["auto_relevance"])
                        # Save the AI evaluation
                        save_ai_evaluation(ai_rating)

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
