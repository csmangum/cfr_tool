import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

# Set page config
st.set_page_config(
    page_title="Federal Regulations Analysis",
    page_icon="��",
    layout="wide",
    menu_items={},  # Empty dict removes the menu items
)


# Function to load data
@st.cache_data
def load_data():
    """Load and prepare regulation metrics data from SQLite database."""
    try:
        engine = create_engine("sqlite:///data/db/regulations.db")
        df = pd.read_sql_table("regulation_metrics", engine)

        # Load agency mapping
        with Path("data/agencies.json").open("r", encoding="utf-8") as f:
            agencies_data = json.load(f)

        # Create mapping dictionary
        mapping = {}
        major_agencies = {
            "Department of Defense": ["defense-department"],
            "Department of Agriculture": ["agriculture-department"],
            "Department of Commerce": ["commerce-department"],
            "Department of Education": ["education-department"],
            "Department of Energy": ["energy-department"],
            "Department of Health and Human Services": [
                "health-and-human-services-department"
            ],
            "Department of Homeland Security": ["homeland-security-department"],
            "Department of Housing and Urban Development": [
                "housing-and-urban-development-department"
            ],
            "Department of Interior": ["interior-department"],
            "Department of Justice": ["justice-department"],
            "Department of Labor": ["labor-department"],
            "Department of State": ["state-department"],
            "Department of Transportation": ["transportation-department"],
            "Department of Treasury": ["treasury-department"],
            "Department of Veterans Affairs": ["veterans-affairs-department"],
            "Other Agencies": [],
        }

        for agency in agencies_data.get("agencies", []):
            slug = agency.get("slug")
            assigned = False
            for major_name, major_slugs in major_agencies.items():
                if slug in major_slugs:
                    mapping[slug] = major_name
                    for child in agency.get("children", []):
                        child_slug = child.get("slug")
                        if child_slug:
                            mapping[child_slug] = major_name
                    assigned = True
                    break
            if not assigned:
                mapping[slug] = "Other Agencies"
                for child in agency.get("children", []):
                    child_slug = child.get("slug")
                    if child_slug:
                        mapping[child_slug] = "Other Agencies"

        # Map agencies to their groups
        df["agency_group"] = df["agency"].map(mapping)

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Main app
def main():
    # Remove sidebar filters and move to main content
    st.title("Federal Regulations Analysis Dashboard")
    st.write(
        "Interactive analysis of federal regulation complexity and metrics across agencies"
    )

    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the database connection.")
        return

    # Move filters to main content area
    st.write(
        """
    **Agency Selection**  
    Select specific agencies to compare their regulations. By default, all agencies are shown.
    """
    )
    selected_agencies = st.multiselect(
        "Select Agencies",
        options=sorted(df["agency_group"].unique()),
        default=sorted(df["agency_group"].unique()),
    )

    # Filter data based on selection
    filtered_df = df[df["agency_group"].isin(selected_agencies)]

    # Create two columns for metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Word Count Analysis")
        st.write(
            """
        This chart shows the total volume of regulations by agency, measured in word count.
        Longer bars indicate agencies with more extensive regulatory text.
        The summary statistics below show the distribution of document lengths within each agency.
        """
        )

        # Calculate total word count by agency and sort
        agency_wordcounts = (
            filtered_df.groupby("agency_group")["word_count"]
            .sum()
            .sort_values(ascending=True)
        )

        # Create sorted bar chart for word count
        fig_wordcount = px.bar(
            x=agency_wordcounts.values,
            y=agency_wordcounts.index,
            orientation="h",  # horizontal bars
            title="Total Word Count by Agency",
            labels={"x": "Total Word Count", "y": "Agency"},
        )
        fig_wordcount.update_layout(showlegend=False)
        st.plotly_chart(fig_wordcount, use_container_width=True)

        # Summary statistics
        st.write("Summary Statistics - Word Count")
        word_count_stats = (
            filtered_df.groupby("agency_group")["word_count"]
            .agg(["mean", "median", "std", "count"])
            .round(2)
        )
        st.dataframe(word_count_stats)

    with col2:
        st.subheader("Complexity Analysis")
        st.write(
            """
        This section analyzes the readability of regulations using various metrics:
        - **Flesch Reading Ease**: Higher scores (0-100) indicate easier reading
        - **Gunning Fog**: Estimates years of formal education needed (lower is simpler)
        - **SMOG Index**: Estimates years of education needed to understand the text
        - **Automated Readability**: Another grade-level estimate of text complexity
        
        The box plots show the distribution of complexity scores across documents within each agency.
        """
        )

        # Metric selector
        complexity_metric = st.selectbox(
            "Select Complexity Metric",
            options=[
                "flesch_reading_ease",
                "gunning_fog",
                "smog_index",
                "automated_readability_index",
            ],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        # Complexity distribution by agency
        fig_complexity = px.box(
            filtered_df,
            x="agency_group",
            y=complexity_metric,
            title=f"Distribution of {complexity_metric.replace('_', ' ').title()} by Agency",
            labels={
                "agency_group": "Agency",
                complexity_metric: complexity_metric.replace("_", " ").title(),
            },
        )
        fig_complexity.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_complexity, use_container_width=True)

        # Summary statistics
        st.write(f"Summary Statistics - {complexity_metric.replace('_', ' ').title()}")
        complexity_stats = (
            filtered_df.groupby("agency_group")[complexity_metric]
            .agg(["mean", "median", "std", "count"])
            .round(2)
        )
        st.dataframe(complexity_stats)

    # Additional Analysis Section
    st.header("Additional Analysis")

    st.write(
        """
    **Correlation Matrix**  
    This heatmap shows relationships between different metrics. Strong positive correlations appear in red,
    negative in blue. For example, we can see how different readability scores relate to each other and
    to basic text statistics like word count and sentence length.
    """
    )

    # Correlation heatmap
    metrics = [
        "flesch_reading_ease",
        "gunning_fog",
        "smog_index",
        "automated_readability_index",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "type_token_ratio",
    ]

    correlation = filtered_df[metrics].corr()

    fig_correlation = go.Figure(
        data=go.Heatmap(
            z=correlation,
            x=metrics,
            y=metrics,
            text=correlation.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorscale="RdBu",
        )
    )

    fig_correlation.update_layout(
        title="Correlation Matrix of Metrics", xaxis_tickangle=-45, height=600
    )

    st.plotly_chart(fig_correlation, use_container_width=True)

    # Time series analysis if date is available
    if "date" in df.columns:
        st.subheader("Temporal Analysis")
        st.write(
            """
        This graph shows how regulation complexity has changed over time for each agency.
        Trends might indicate whether agencies are making their regulations more or less complex
        over time.
        """
        )

        # Convert date to datetime if it's not already
        filtered_df["date"] = pd.to_datetime(filtered_df["date"])

        # Group by date and agency, calculate mean metrics
        temporal_data = (
            filtered_df.groupby(["date", "agency_group"])[complexity_metric]
            .mean()
            .reset_index()
        )

        fig_temporal = px.line(
            temporal_data,
            x="date",
            y=complexity_metric,
            color="agency_group",
            title=f"Trend of {complexity_metric.replace('_', ' ').title()} Over Time",
            labels={
                "date": "Date",
                complexity_metric: complexity_metric.replace("_", " ").title(),
                "agency_group": "Agency",
            },
        )

        st.plotly_chart(fig_temporal, use_container_width=True)


if __name__ == "__main__":
    main()
