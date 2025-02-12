import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

# -------------------------------------
# Page configuration
# -------------------------------------
st.set_page_config(
    page_title="Federal Regulations Analysis",
    page_icon="🚀",
    layout="wide",
    menu_items={},
)


# -------------------------------------
# Custom CSS for refined styling
# -------------------------------------
def local_css(css_string: str):
    st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)

custom_css = """
/* Center the header */
h1 {
    text-align: center;
    color: #333333;
}

/* Sidebar background */
[data-testid="stSidebar"] .css-1d391kg {  
    background-color: #f0f2f6;
}

/* Increase spacing in the sidebar */
[data-testid="stSidebar"] .css-1d391kg {
    padding: 2rem;
}
"""
local_css(custom_css)


# -------------------------------------
# Data Loading and Preparation
# -------------------------------------
@st.cache_data
def load_data():
    """
    Load regulation metrics data from the SQLite database.
    Map agencies to their major groups using an external JSON file.
    """
    try:
        engine = create_engine("sqlite:///data/db/regulations.db")
        df = pd.read_sql_table("regulation_metrics", engine)

        with Path("data/agencies.json").open("r", encoding="utf-8") as f:
            agencies_data = json.load(f)

        # Define major agencies and associated slugs.
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

        # Build mapping from agency slug to major agency.
        mapping = {}
        for agency in agencies_data.get("agencies", []):
            slug = agency.get("slug")
            assigned = False
            for major, slugs in major_agencies.items():
                if slug in slugs:
                    mapping[slug] = major
                    mapping.update(
                        {
                            child.get("slug"): major
                            for child in agency.get("children", [])
                            if child.get("slug")
                        }
                    )
                    assigned = True
                    break
            if not assigned:
                mapping[slug] = "Other Agencies"
                mapping.update(
                    {
                        child.get("slug"): "Other Agencies"
                        for child in agency.get("children", [])
                        if child.get("slug")
                    }
                )

        df["agency_group"] = df["agency"].map(mapping)
        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# -------------------------------------
# Utility: Download Link for Data
# -------------------------------------
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    If the object is a DataFrame, it is converted to CSV.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# -------------------------------------
# Plotting Functions
# -------------------------------------
def plot_word_count(filtered_df: pd.DataFrame):
    """Display a bar chart and summary statistics for total word count by agency."""
    agency_wordcounts = (
        filtered_df.groupby("agency_group")["word_count"]
        .sum()
        .sort_values(ascending=True)
    )
    fig = px.bar(
        x=agency_wordcounts.values,
        y=agency_wordcounts.index,
        orientation="h",
        title="Total Word Count by Agency",
        labels={"x": "Total Word Count", "y": "Agency"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Word Count Summary"):
        stats = (
            filtered_df.groupby("agency_group")["word_count"]
            .agg(["mean", "median", "std", "count"])
            .round(2)
        )
        st.dataframe(stats)


def plot_complexity(filtered_df: pd.DataFrame, metric: str):
    """Display a box plot and summary statistics for a selected complexity metric."""
    fig = px.box(
        filtered_df,
        x="agency_group",
        y=metric,
        title=f"Distribution of {metric.replace('_', ' ').title()} by Agency",
        labels={"agency_group": "Agency", metric: metric.replace("_", " ").title()},
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Complexity Summary"):
        stats = (
            filtered_df.groupby("agency_group")[metric]
            .agg(["mean", "median", "std", "count"])
            .round(2)
        )
        st.dataframe(stats)


def plot_correlation(filtered_df: pd.DataFrame):
    """Display a correlation heatmap for selected metrics."""
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
    fig = go.Figure(
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
    fig.update_layout(
        title="Correlation Matrix of Metrics", xaxis_tickangle=-45, height=600
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_temporal(filtered_df: pd.DataFrame, metric: str):
    """Display a time-series trend for the selected complexity metric by agency."""
    df_copy = filtered_df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    temporal_data = (
        df_copy.groupby(["date", "agency_group"])[metric].mean().reset_index()
    )
    fig = px.line(
        temporal_data,
        x="date",
        y=metric,
        color="agency_group",
        title=f"Trend of {metric.replace('_', ' ').title()} Over Time",
        labels={
            "date": "Date",
            metric: metric.replace("_", " ").title(),
            "agency_group": "Agency",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------
# Main Application
# -------------------------------------
def main():
    # Custom header
    st.markdown(
        "<h1>Federal Regulations Analysis Dashboard</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "Use the sidebar to filter data and navigate between analysis sections."
    )

    # Load data with a spinner for a smoother experience.
    with st.spinner("Loading data..."):
        df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the database connection.")
        return

    # Sidebar: Filter Options
    st.sidebar.header("Filters")
    all_agencies = sorted(df["agency_group"].dropna().unique())
    selected_agencies = st.sidebar.multiselect(
        "Select Agencies", options=all_agencies, default=all_agencies
    )
    filtered_df = df[df["agency_group"].isin(selected_agencies)].copy()
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])

    # Date range filter
    if "date" in filtered_df.columns:
        min_date = filtered_df["date"].min()
        max_date = filtered_df["date"].max()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df["date"].dt.date >= start_date)
                & (filtered_df["date"].dt.date <= end_date)
            ]

    # Sidebar: Data download link
    st.sidebar.markdown("### Download Data")
    tmp_download_link = download_link(
        filtered_df, "filtered_data.csv", "Download Filtered Data (CSV)"
    )
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

    # Sidebar Navigation: Select the analysis section
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Section",
        ("Word Count Analysis", "Complexity Analysis", "Additional Analysis"),
    )

    # -------------------------------------
    # Analysis Sections
    # -------------------------------------
    if analysis_mode == "Word Count Analysis":
        st.header("Word Count Analysis")
        st.markdown(
            "Explore the total word count by agency and view summary statistics."
        )
        plot_word_count(filtered_df)

    elif analysis_mode == "Complexity Analysis":
        st.header("Complexity Analysis")
        st.markdown(
            "Select a complexity metric below to examine its distribution across agencies."
        )
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
        plot_complexity(filtered_df, complexity_metric)

    elif analysis_mode == "Additional Analysis":
        st.header("Additional Analysis")
        st.markdown(
            "Dive deeper into the data by exploring correlations and temporal trends."
        )
        with st.expander("View Correlation Matrix"):
            plot_correlation(filtered_df)
        if "date" in df.columns:
            with st.expander("View Temporal Analysis"):
                temporal_metric = st.selectbox(
                    "Select Complexity Metric for Temporal Analysis",
                    options=[
                        "flesch_reading_ease",
                        "gunning_fog",
                        "smog_index",
                        "automated_readability_index",
                    ],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="temporal_metric",
                )
                plot_temporal(filtered_df, temporal_metric)


if __name__ == "__main__":
    main()
