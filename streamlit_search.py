"""
Streamlit app for searching regulations using semantic similarity.

Run with: streamlit run scripts/streamlit_search.py
"""

import json
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Counter, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

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

# Add this near the top of the file with other session state initializations
if "ai_evaluations" not in st.session_state:
    st.session_state.ai_evaluations = {}


def setup_openai_logging():
    """Setup logging for OpenAI API interactions."""
    try:
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a logger
        logger = logging.getLogger("openai")
        logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicates
        logger.handlers = []

        # Create a file handler with absolute path
        log_file = log_dir / f"openai_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.INFO)

        # Create a formatting for the logs
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

        # Log the initialization
        logger.info(f"OpenAI logging initialized. Logging to: {log_file}")

        return logger
    except Exception as e:
        st.error(f"Error setting up logging: {str(e)}")
        st.error(f"Current directory: {Path.cwd()}")
        st.error(f"File location: {Path(__file__)}")
        return None


class ChatCompletions:
    def __init__(self, client, logger):
        self.client = client
        self.logger = logger

    def create(self, **kwargs):
        """Log the chat completion request and response."""
        try:
            if self.logger:
                # Log the request
                self.logger.info(f"OpenAI Request: {kwargs}")

                # Make the API call
                response = self.client.chat.completions.create(**kwargs)

                # Log the response
                self.logger.info(f"OpenAI Response: {response}")

                return response
            else:
                # If logger failed to initialize, just make the API call
                return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if self.logger:
                self.logger.error(f"OpenAI Error: {str(e)}")
            raise


class Chat:
    def __init__(self, client, logger):
        self.completions = ChatCompletions(client, logger)


class LoggedOpenAI:
    """Wrapper around OpenAI client that logs all interactions."""

    def __init__(self):
        self.client = OpenAI()
        self.logger = setup_openai_logging()
        self.chat = Chat(self.client, self.logger)


def initialize_searcher() -> RegulationSearcher:
    """Initialize the regulation searcher with default paths."""
    # Use session state to cache the searcher
    if "searcher" not in st.session_state:
        try:
            # Define paths relative to project root
            index_path = "data/faiss/regulation_index.faiss"
            metadata_path = "data/faiss/regulation_metadata.json"
            db_path = "data/db/regulation_embeddings.db"

            # Check if files exist
            if not Path(index_path).exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {index_path}. "
                    "Please ensure you have run the embedding pipeline to generate the index."
                )
            if not Path(metadata_path).exists():
                raise FileNotFoundError(
                    f"Metadata file not found at {metadata_path}. "
                    "Please ensure you have run the embedding pipeline."
                )
            if not Path(db_path).exists():
                raise FileNotFoundError(
                    f"Database not found at {db_path}. "
                    "Please ensure you have run the data processing pipeline."
                )

            searcher = RegulationSearcher(
                index_path=index_path,
                metadata_path=metadata_path,
                db_path=db_path,
                model_name="all-MiniLM-L6-v2",
            )
            # Add logged OpenAI client
            searcher.client = LoggedOpenAI()
            st.session_state.searcher = searcher

        except Exception as e:
            st.error(f"Error initializing search: {str(e)}")
            st.error(
                "Please ensure all required files are present in the correct locations:"
            )
            st.error(f"- FAISS index: {index_path}")
            st.error(f"- Metadata file: {metadata_path}")
            st.error(f"- Database: {db_path}")
            raise

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

        eval_dir = Path("data/evaluations")
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_file = eval_dir / "search_evaluations.json"

        # Load existing evaluations
        existing_evals = []
        if eval_file.exists():
            try:
                with eval_file.open("r", encoding="utf-8") as f:
                    existing_evals = json.load(f)
            except json.JSONDecodeError:
                st.warning(
                    f"Could not read existing evaluations from {eval_file}. Starting fresh."
                )

        # Append new evaluation
        existing_evals.append(evaluation)

        # Save updated evaluations
        with eval_file.open("w", encoding="utf-8") as f:
            json.dump(existing_evals, f, indent=2, ensure_ascii=False)

        st.success(f"Evaluation saved to {eval_file}")

    except Exception as e:
        st.error(f"Error saving evaluation: {str(e)}")
        st.error(f"Current directory: {Path.cwd()}")
        st.error(f"File location: {Path(__file__)}")
        st.error(
            f"Attempted to save to: {eval_file if 'eval_file' in locals() else 'unknown path'}"
        )


def calculate_query_stats(evaluations: List[Dict]) -> Dict:
    """Calculate statistics about query performance."""
    try:
        # Initialize empty stats dictionary
        query_stats = defaultdict(
            lambda: {
                "count": 0,
                "avg_relevance": 0.0,
                "avg_quality": 0.0,
                "total_ratings": 0,
            }
        )

        if not evaluations:
            return {}

        for eval in evaluations:
            query = eval.get("query", "")
            if not query:  # Skip if query is empty
                continue

            ratings = eval.get("ratings", {})
            if not ratings:  # Skip if no ratings
                continue

            query_stats[query]["count"] += 1

            # Calculate relevance and quality only if there are valid ratings
            relevance_scores = [
                RELEVANCE_SCORES.get(r.get("relevance", "Not Relevant"), 0)
                for r in ratings.values()
                if r.get("relevance")
            ]

            quality_scores = [
                QUALITY_SCORES.get(r.get("quality", "Poor"), 0)
                for r in ratings.values()
                if r.get("quality")
            ]

            num_ratings = len(ratings)
            if num_ratings > 0:
                query_stats[query]["total_ratings"] += num_ratings

                # Calculate averages only if there are scores
                if relevance_scores:
                    avg_relevance = sum(relevance_scores) / len(relevance_scores)
                    query_stats[query]["avg_relevance"] = (
                        query_stats[query]["avg_relevance"]
                        * (query_stats[query]["count"] - 1)
                        + avg_relevance
                    ) / query_stats[query]["count"]

                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    query_stats[query]["avg_quality"] = (
                        query_stats[query]["avg_quality"]
                        * (query_stats[query]["count"] - 1)
                        + avg_quality
                    ) / query_stats[query]["count"]

        return dict(query_stats)
    except Exception as e:
        st.error(f"Error calculating query stats: {str(e)}")
        return {}


def analyze_feedback_themes(evaluations: List[Dict]) -> Counter:
    """Analyze common themes in feedback."""
    try:
        # Load evaluations from file if not provided
        if not evaluations:
            eval_file = Path("data/evaluations/search_evaluations.json")
            if eval_file.exists():
                with eval_file.open("r", encoding="utf-8") as f:
                    evaluations = json.load(f)

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
    except Exception as e:
        st.error(f"Error analyzing feedback themes: {str(e)}")
        return Counter()


def auto_evaluate_result(result: Dict, query: str) -> Dict:
    """Automatically evaluate search result using GPT-4."""
    try:
        # Optimized prompt with fewer tokens
        prompt = f"""Evaluate relevance and quality:
Q: {query}
Text: {result['Text'][:500]}... # Truncate long texts
Meta: {result['Agency']}, {result['Title']}, {result['Section']}

Return exactly:
{{"relevance": ["Not Relevant"|"Somewhat Relevant"|"Relevant"|"Very Relevant"], 
"quality": ["Poor"|"Fair"|"Good"|"Excellent"]}}
Brief explanation"""

        response = st.session_state.searcher.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,  # Limit response length
        )

        # Split response into dict and explanation
        response_text = response.choices[0].message.content
        dict_end = response_text.find("}") + 1
        ratings_dict = json.loads(response_text[:dict_end])
        explanation = response_text[dict_end:].strip()

        return {
            "ratings": ratings_dict,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": {
                k: v for k, v in result.items() if k != "Text"
            },  # Don't store full text
        }
    except Exception as e:
        st.error(f"AI evaluation failed: {str(e)}")
        return None


def save_ai_evaluation(ai_rating: Dict):
    """Save AI evaluation results to a separate file."""
    try:
        eval_dir = Path("data/evaluations")
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


def generate_overall_feedback(results: List[Dict], ai_evaluations: Dict) -> str:
    """Generate overall feedback based on AI evaluations of results."""
    try:
        # Collect only essential evaluation data
        evaluations_summary = []
        for idx, (metadata_item, score, _) in enumerate(results, 1):
            result_key = f"ai_eval_result_{metadata_item.get('section', '')}_{idx}"
            if result_key in ai_evaluations:
                eval_data = ai_evaluations[result_key]
                evaluations_summary.append(
                    {
                        "r": eval_data["ratings"]["relevance"],
                        "q": eval_data["ratings"]["quality"],
                    }
                )

        if not evaluations_summary:
            return "No AI evaluations available. Use evaluation buttons above."

        # Optimized prompt
        prompt = f"""Assess {len(evaluations_summary)} results:
{json.dumps(evaluations_summary)}

1. Overall quality/relevance
2. Key strengths/weaknesses
3. Improvement suggestions

Be concise."""

        response = st.session_state.searcher.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,  # Limit response length
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating feedback: {str(e)}"


def generate_random_question() -> str:
    """Generate a random question about US regulations using GPT-4."""
    try:
        prompt = """Generate a random, specific question about US federal regulations.
The question should be practical and focused on understanding regulatory requirements.
It should be a question that a business owner, compliance officer, or citizen might actually ask.

Format: Return only the question text, nothing else.

Examples:
- What are the OSHA requirements for emergency exits in a small retail store?
- What regulations govern the storage of hazardous materials in a research laboratory?
- What are the FDA labeling requirements for gluten-free products?"""

        response = st.session_state.searcher.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,  # Higher temperature for more variety
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating random question: {str(e)}")
        return random.choice(SAMPLE_QUESTIONS)  # Fallback to predefined questions


def analyze_advanced_metrics(evaluations: List[Dict]) -> Dict:
    """Calculate advanced metrics from evaluation data."""
    try:
        metrics = {
            "agency_performance": defaultdict(list),
            "time_of_day_stats": defaultdict(list),
            "query_length_impact": [],
            "response_times": [],
            "section_relevance": defaultdict(list),
            "topic_clusters": defaultdict(int),
            "user_engagement": {
                "feedback_rate": 0,
                "ai_eval_rate": 0,
                "detailed_feedback_rate": 0,
            },
        }

        if not evaluations:
            return metrics

        for eval in evaluations:
            # Time of day analysis
            timestamp = pd.to_datetime(eval.get("timestamp", ""))
            if timestamp:
                hour = timestamp.hour
                metrics["time_of_day_stats"][hour].append(1)

            # Query analysis
            query = eval.get("query", "")
            if query:
                query_length = len(query.split())

                # Process results
                results = eval.get("results", [])
                ratings = eval.get("ratings", {})

                if results and ratings:
                    # Calculate average relevance for this query
                    relevance_scores = []
                    for rating in ratings.values():
                        relevance = rating.get("relevance", "Not Relevant")
                        if relevance in RELEVANCE_SCORES:
                            relevance_scores.append(RELEVANCE_SCORES[relevance])

                    if relevance_scores:
                        avg_relevance = sum(relevance_scores) / len(relevance_scores)
                        metrics["query_length_impact"].append(
                            (query_length, avg_relevance)
                        )

                    # Process each result
                    for result in results:
                        if isinstance(result, dict):
                            agency = result.get("Agency", "Unknown")
                            section = result.get("Section", "Unknown")

                            # Add relevance score to agency performance
                            if agency and ratings:
                                first_rating = next(iter(ratings.values()))
                                relevance = first_rating.get(
                                    "relevance", "Not Relevant"
                                )
                                if relevance in RELEVANCE_SCORES:
                                    metrics["agency_performance"][agency].append(
                                        RELEVANCE_SCORES[relevance]
                                    )

                            # Add relevance score to section performance
                            if section and ratings:
                                metrics["section_relevance"][section].append(
                                    RELEVANCE_SCORES.get(relevance, 0)
                                )

            # User engagement
            has_feedback = bool(eval.get("feedback"))
            has_ai_eval = bool(eval.get("ai_evaluations"))
            has_detailed_feedback = any(
                r.get("feedback") for r in eval.get("ratings", {}).values()
            )

            metrics["user_engagement"]["feedback_rate"] += int(has_feedback)
            metrics["user_engagement"]["ai_eval_rate"] += int(has_ai_eval)
            metrics["user_engagement"]["detailed_feedback_rate"] += int(
                has_detailed_feedback
            )

        # Calculate averages and normalize
        total_evals = len(evaluations)
        if total_evals > 0:
            metrics["user_engagement"] = {
                k: v / total_evals for k, v in metrics["user_engagement"].items()
            }

        return metrics
    except Exception as e:
        st.error(f"Error calculating advanced metrics: {str(e)}")
        return {}


def visualize_advanced_metrics(metrics: Dict):
    """Create visualizations for advanced metrics."""
    st.markdown("## Advanced Analytics")

    # 1. Agency Performance Heatmap
    st.markdown("### Agency Performance Heatmap")
    try:
        agency_scores = {
            agency: np.mean(scores) if scores else 0
            for agency, scores in metrics.get("agency_performance", {}).items()
        }

        if agency_scores:
            agency_df = pd.DataFrame(
                list(agency_scores.items()), columns=["Agency", "Average Score"]
            ).sort_values("Average Score", ascending=False)

            fig = px.imshow(
                [agency_df["Average Score"]],
                labels=dict(x="Agency", y="", color="Score"),
                x=agency_df["Agency"],
                aspect="auto",
                title="Agency Performance Heatmap",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agency performance data available yet.")
    except Exception as e:
        st.error(f"Error creating agency performance heatmap: {str(e)}")

    # 2. Query Length Impact
    st.markdown("### Query Length Impact on Relevance")
    query_df = pd.DataFrame(
        metrics["query_length_impact"], columns=["Query Length", "Average Relevance"]
    )
    if not query_df.empty:
        fig = px.scatter(
            query_df,
            x="Query Length",
            y="Average Relevance",
            trendline="ols",
            title="Impact of Query Length on Result Relevance",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3. User Engagement Metrics
    st.markdown("### User Engagement Metrics")
    engagement_df = pd.DataFrame(
        {
            "Metric": ["Feedback Rate", "Detailed Feedback Rate", "AI Evaluation Rate"],
            "Rate": [
                metrics["user_engagement"]["feedback_rate"],
                metrics["user_engagement"]["detailed_feedback_rate"],
                metrics["user_engagement"]["ai_eval_rate"],
            ],
        }
    )

    fig = px.bar(
        engagement_df,
        x="Metric",
        y="Rate",
        title="User Engagement Rates",
        text_auto=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Section Performance
    st.markdown("### Section Performance")
    section_scores = {
        section: np.mean(scores)
        for section, scores in metrics["section_relevance"].items()
        if scores and len(scores) >= 5  # Only include sections with sufficient data
    }
    if section_scores:
        section_df = pd.DataFrame(
            list(section_scores.items()), columns=["Section", "Average Score"]
        ).sort_values("Average Score", ascending=False)

        fig = px.bar(
            section_df,
            x="Section",
            y="Average Score",
            title="Average Relevance Score by Section",
            labels={"Average Score": "Relevance Score"},
        )
        st.plotly_chart(fig, use_container_width=True)


def analyze_openai_logs(log_file: Path) -> Dict:
    """Analyze OpenAI API usage from logs."""
    try:
        metrics = {
            "requests_by_type": defaultdict(int),
            "token_usage": [],
            "response_times": [],
            "completion_tokens": [],
            "prompt_tokens": [],
            "total_cost": 0,  # Assuming GPT-4 pricing
            "hourly_usage": defaultdict(int),
            "error_rates": defaultdict(int),
        }

        # GPT-4 pricing per 1k tokens (approximate)
        PROMPT_COST = 0.03
        COMPLETION_COST = 0.06

        # Open file with UTF-8 encoding and error handling
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    if "OpenAI Request" in line or "OpenAI Response" in line:
                        timestamp = pd.to_datetime(line.split(" - ")[0])
                        if "Request" in line:
                            # Parse request data safely
                            try:
                                request_str = line.split("Request: ")[1].strip()
                                request_data = eval(request_str)
                                metrics["requests_by_type"][
                                    request_data.get("model", "unknown")
                                ] += 1
                                metrics["hourly_usage"][timestamp.hour] += 1
                            except Exception as e:
                                metrics["error_rates"][
                                    f"Request Parse Error: {str(e)}"
                                ] += 1
                                continue

                        elif "Response" in line:
                            # Parse response data safely
                            try:
                                response_str = line.split("Response: ")[1].strip()
                                response_data = eval(response_str)
                                if hasattr(response_data, "usage"):
                                    usage = response_data.usage
                                    metrics["completion_tokens"].append(
                                        usage.completion_tokens
                                    )
                                    metrics["prompt_tokens"].append(usage.prompt_tokens)
                                    metrics["token_usage"].append(usage.total_tokens)

                                    # Calculate cost
                                    prompt_cost = (
                                        usage.prompt_tokens / 1000
                                    ) * PROMPT_COST
                                    completion_cost = (
                                        usage.completion_tokens / 1000
                                    ) * COMPLETION_COST
                                    metrics["total_cost"] += (
                                        prompt_cost + completion_cost
                                    )
                            except Exception as e:
                                metrics["error_rates"][
                                    f"Response Parse Error: {str(e)}"
                                ] += 1
                                continue
                except Exception as e:
                    metrics["error_rates"][f"Line Parse Error: {str(e)}"] += 1
                    continue

        return metrics
    except Exception as e:
        st.error(f"Error analyzing OpenAI logs: {str(e)}")
        return {}


def visualize_openai_metrics(metrics: Dict):
    """Create visualizations for OpenAI API usage metrics."""
    st.markdown("## OpenAI API Usage Analytics")

    # 1. Token Usage Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        total_tokens = sum(metrics["token_usage"])
        st.metric("Total Tokens Used", f"{total_tokens:,}")
    with col2:
        avg_tokens = np.mean(metrics["token_usage"]) if metrics["token_usage"] else 0
        st.metric("Average Tokens per Request", f"{avg_tokens:.1f}")
    with col3:
        st.metric("Estimated Cost", f"${metrics['total_cost']:.2f}")

    # 2. Token Usage Distribution
    st.markdown("### Token Usage Distribution")
    token_df = pd.DataFrame(
        {
            "Prompt Tokens": metrics["prompt_tokens"],
            "Completion Tokens": metrics["completion_tokens"],
        }
    )

    fig = px.box(
        token_df.melt(),
        y="value",
        x="variable",
        title="Token Usage Distribution by Type",
        labels={"value": "Number of Tokens", "variable": "Token Type"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Hourly Usage Pattern
    st.markdown("### Hourly Usage Pattern")
    hourly_df = pd.DataFrame(
        list(metrics["hourly_usage"].items()), columns=["Hour", "Requests"]
    ).sort_values("Hour")

    fig = px.line(
        hourly_df,
        x="Hour",
        y="Requests",
        title="API Requests by Hour",
        labels={"Requests": "Number of Requests"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. Model Usage Distribution
    st.markdown("### Model Usage Distribution")
    model_df = pd.DataFrame(
        list(metrics["requests_by_type"].items()), columns=["Model", "Requests"]
    ).sort_values("Requests", ascending=False)

    fig = px.pie(
        model_df, values="Requests", names="Model", title="Requests by Model Type"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Error Analysis
    if metrics["error_rates"]:
        st.markdown("### Error Analysis")
        error_df = pd.DataFrame(
            list(metrics["error_rates"].items()), columns=["Error Type", "Count"]
        ).sort_values("Count", ascending=False)

        fig = px.bar(error_df, x="Error Type", y="Count", title="API Errors by Type")
        st.plotly_chart(fig, use_container_width=True)

    # Add export option
    if st.button("Export OpenAI Usage Data"):
        export_data = {
            "token_usage_summary": {
                "total_tokens": total_tokens,
                "average_tokens": avg_tokens,
                "estimated_cost": metrics["total_cost"],
            },
            "hourly_usage": dict(metrics["hourly_usage"]),
            "model_usage": dict(metrics["requests_by_type"]),
            "error_rates": dict(metrics["error_rates"]),
        }
        st.download_button(
            "Download JSON",
            json.dumps(export_data, indent=2),
            "openai_usage_metrics.json",
            "application/json",
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

        auto_evaluate = st.toggle("Enable AI Evaluation", value=False)

        st.header("Sample Questions")
        if st.button("🎲 Generate Random Question"):
            with st.spinner("Generating question..."):
                random_question = generate_random_question()
                st.session_state.random_question = random_question
                # Clear any previous search results and reset the page
                if "search_results" in st.session_state:
                    del st.session_state.search_results
                if "current_query" in st.session_state:
                    del st.session_state.current_query
                if "current_ratings" in st.session_state:
                    st.session_state.current_ratings = {}
                # Force a rerun to update the main input and clear results
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
                results = searcher.search(query, n_results=num_results)

                # Filter results after search if min_score is set
                if min_score > 0:
                    results = [r for r in results if r[1] >= min_score]

                search_time = time.time() - start_time
                st.info(f"Search completed in {search_time:.2f} seconds")

            if results:
                # Store search results and query in session state
                st.session_state["search_results"] = results
                st.session_state["current_query"] = query

        except Exception as e:
            st.error(f"Error performing search: {str(e)}")
            st.error("Please try refining your search terms or adjusting filters")

    # Display results (single display block)
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        query = st.session_state["current_query"]

        st.success(f"Found {len(results)} relevant regulations")

        # Add AI evaluation button for all results
        if st.button("🤖 Get AI Evaluations for All Results"):
            with st.spinner("Getting AI evaluations..."):
                for idx, (metadata_item, score, chunk_text) in enumerate(results, 1):
                    result_dict = format_result(metadata_item, score, chunk_text)
                    result_key = (
                        f"ai_eval_result_{metadata_item.get('section', '')}_{idx}"
                    )

                    # Only evaluate if not already evaluated
                    if result_key not in st.session_state.ai_evaluations:
                        ai_rating = auto_evaluate_result(result_dict, query)
                        if ai_rating:
                            st.session_state.ai_evaluations[result_key] = ai_rating
                            # Update the sliders with AI ratings
                            st.session_state[f"relevance_{idx}"] = ai_rating["ratings"][
                                "relevance"
                            ]
                            st.session_state[f"quality_{idx}"] = ai_rating["ratings"][
                                "quality"
                            ]
                            # Update the feedback text area with AI explanation
                            st.session_state[f"feedback_{idx}"] = (
                                f"AI Evaluation: {ai_rating['explanation']}"
                            )

            st.success("AI evaluations completed for all results!")

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
        st.markdown("Please help us improve search quality by rating the results:")

        # Store ratings in session state
        if "current_ratings" not in st.session_state:
            st.session_state.current_ratings = {}

        for idx, (metadata_item, score, chunk_text) in enumerate(results, 1):
            with st.expander(f"Result {idx} (Score: {score:.3f})", expanded=True):
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

                # Add AI evaluation section
                st.markdown("#### AI Evaluation")

                # Create unique keys for this result
                button_key = f"ai_eval_button_{idx}"
                result_key = f"ai_eval_result_{metadata_item.get('section', '')}_{idx}"

                col1, col2 = st.columns([1, 3])

                with col1:
                    if st.button("🤖 Get AI Evaluation", key=button_key):
                        with st.spinner("Getting AI evaluation..."):
                            ai_rating = auto_evaluate_result(result_dict, query)
                            if ai_rating:
                                st.session_state.ai_evaluations[result_key] = ai_rating
                                # Update the sliders with AI ratings
                                st.session_state[f"relevance_{idx}"] = ai_rating[
                                    "ratings"
                                ]["relevance"]
                                st.session_state[f"quality_{idx}"] = ai_rating[
                                    "ratings"
                                ]["quality"]
                                # Update the feedback text area with AI explanation
                                st.session_state[f"feedback_{idx}"] = (
                                    f"AI Evaluation: {ai_rating['explanation']}"
                                )

                with col2:
                    if result_key in st.session_state.ai_evaluations:
                        st.markdown("**AI Ratings Applied**")

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

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🤖 Generate Overall Feedback"):
                with st.spinner("Analyzing results..."):
                    ai_feedback = generate_overall_feedback(
                        results, st.session_state.ai_evaluations
                    )
                    st.session_state["overall_feedback"] = ai_feedback

        overall_feedback = st.text_area(
            "Overall feedback about the search results",
            key="overall_feedback",
            height=200,
        )

        # Submit evaluation
        if st.button("Submit Evaluation"):
            save_evaluation(
                query=query,
                results=[format_result(m, s, t) for m, s, t in results],
                ratings=st.session_state.current_ratings,
                feedback=overall_feedback,
            )
            st.success("Thank you for your feedback! Your evaluation has been saved.")

            # Clear current ratings
            st.session_state.current_ratings = {}

        # Add enhanced evaluation statistics
        eval_file = Path("data/evaluations/search_evaluations.json")
        if eval_file.exists():
            st.markdown("## Analytics Dashboard")

            # Basic Metrics Section
            with st.expander("🔢 Basic Metrics", expanded=True):
                try:
                    with eval_file.open("r", encoding="utf-8") as f:
                        evaluations = json.load(f)

                    total_evaluations = len(evaluations)
                    total_ratings = sum(len(eval["ratings"]) for eval in evaluations)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Evaluations", total_evaluations)
                    with col2:
                        st.metric("Total Ratings", total_ratings)
                    with col3:
                        st.metric(
                            "Avg Ratings per Query",
                            (
                                f"{total_ratings/total_evaluations:.1f}"
                                if total_evaluations > 0
                                else "0.0"
                            ),
                        )
                except Exception as e:
                    st.error(f"Error loading basic metrics: {str(e)}")

            # Query Performance Section
            with st.expander("📊 Query Performance Analysis", expanded=False):
                try:
                    st.markdown("### Query Performance")
                    query_stats = calculate_query_stats(evaluations)

                    if query_stats:
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

                        if not query_df.empty:
                            query_df = query_df.sort_values(
                                "Avg Relevance", ascending=False
                            )
                            st.dataframe(
                                query_df.style.format(
                                    {
                                        "Avg Relevance": "{:.2f}",
                                        "Avg Quality": "{:.2f}",
                                        "Count": "{:,}",
                                        "Total Ratings": "{:,}",
                                    }
                                )
                            )
                        else:
                            st.info("No query performance data available yet.")
                    else:
                        st.info(
                            "No evaluations available. Start rating search results to see performance metrics."
                        )
                except Exception as e:
                    st.error(f"Error loading query performance: {str(e)}")
                    st.error("Please ensure the evaluation data is properly formatted.")

            # Feedback Analysis Section
            with st.expander("💭 Feedback Analysis", expanded=False):
                try:
                    themes = analyze_feedback_themes(evaluations)
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
                except Exception as e:
                    st.error(f"Error loading feedback analysis: {str(e)}")

            # Time Trends Section
            with st.expander("📈 Quality Trends", expanded=False):
                try:
                    trend_data = []
                    for eval in evaluations:
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
                        trend_data.extend(
                            [
                                {
                                    "Date": timestamp,
                                    "Score": avg_relevance,
                                    "Type": "Relevance",
                                },
                                {
                                    "Date": timestamp,
                                    "Score": avg_quality,
                                    "Type": "Quality",
                                },
                            ]
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
                except Exception as e:
                    st.error(f"Error loading time trends: {str(e)}")

            # Advanced Analytics Section
            with st.expander("🔬 Advanced Analytics", expanded=False):
                try:
                    advanced_metrics = analyze_advanced_metrics(evaluations)
                    visualize_advanced_metrics(advanced_metrics)

                    if st.button("Download Advanced Metrics"):
                        metrics_json = json.dumps(advanced_metrics, indent=2)
                        st.download_button(
                            "Download JSON",
                            metrics_json,
                            "advanced_metrics.json",
                            "application/json",
                        )
                except Exception as e:
                    st.error(f"Error loading advanced analytics: {str(e)}")

            # OpenAI Usage Section
            with st.expander("🤖 OpenAI API Usage", expanded=False):
                try:
                    log_file = Path("data/logs/openai_20250215.log")
                    if log_file.exists():
                        openai_metrics = analyze_openai_logs(log_file)
                        visualize_openai_metrics(openai_metrics)
                    else:
                        st.warning("No OpenAI usage logs found.")
                except Exception as e:
                    st.error(f"Error loading OpenAI usage metrics: {str(e)}")

            # Export Section
            with st.expander("📥 Export Data", expanded=False):
                try:
                    if st.button("Download Complete Evaluation Data"):
                        eval_df = pd.json_normalize(evaluations)
                        csv = eval_df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "regulation_search_evaluations.csv",
                            "text/csv",
                            key="download_eval",
                        )
                except Exception as e:
                    st.error(f"Error preparing export: {str(e)}")

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
