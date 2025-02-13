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

/* Metric styling */
[data-testid="stMetricLabel"] {
    font-size: 1.0rem !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.95rem !important;
}

/* Sidebar multiselect options styling */
.stMultiSelect div div div div {
    font-size: 0.7rem !important;
}

/* Sidebar multiselect dropdown styling */
.stMultiSelect div[data-baseweb="select"] span {
    font-size: 0.7rem !important;
}

/* Additional sidebar text styling */
.stSidebar .block-container {
    font-size: 0.7rem !important;
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
            # Add new major financial regulators
            "Federal Communications Commission": ["federal-communications-commission"],
            "Federal Reserve System": ["federal-reserve-system"],
            "Commodity Futures Trading Commission": [
                "commodity-futures-trading-commission"
            ],
            "Securities and Exchange Commission": [
                "securities-and-exchange-commission"
            ],
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
    """Display summary statistics and a bar chart for total word count by agency."""
    # Calculate summary metrics
    total_word_count = filtered_df["word_count"].sum()
    avg_word_count_per_agency = (
        filtered_df.groupby("agency_group")["word_count"].mean().mean()
    )

    # Exclude "Other Agencies" when finding the largest agency
    agency_totals = (
        filtered_df[filtered_df["agency_group"] != "Other Agencies"]
        .groupby("agency_group")["word_count"]
        .sum()
    )
    max_agency = agency_totals.idxmax()
    max_agency_words = agency_totals.max()

    # Calculate average complexity
    avg_complexity = filtered_df["gunning_fog"].mean()
    most_complex_agency = (
        filtered_df.groupby("agency_group")["gunning_fog"].mean().idxmax()
    )
    max_complexity = filtered_df.groupby("agency_group")["gunning_fog"].mean().max()

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Word Count",
            f"{total_word_count:,.0f}",
        )
    with col2:
        st.metric(
            "Average Words per Agency",
            f"{avg_word_count_per_agency:,.0f}",
        )
    with col3:
        st.metric(
            "Largest Agency by Words",
            f"{max_agency}",
            delta=f"{max_agency_words:,.0f} words",
        )
    with col4:
        st.metric(
            "Average Complexity (Gunning Fog)",
            f"{avg_complexity:.1f}",
            help="Gunning Fog Index indicates the years of formal education needed to understand the text. Scores: 6=sixth grade, 12=high school, 17=college graduate",
        )

    # Add second row of metrics focusing on complexity
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(
            "Most Complex Agency",
            f"{most_complex_agency}",
            delta=f"Fog Index: {max_complexity:.1f}",
            delta_color="off",
        )

    # Create horizontal bar chart with improved sorting
    agency_wordcounts = (
        filtered_df.groupby("agency_group")["word_count"]
        .sum()
        .sort_values(ascending=True)
    )

    # Move "Other Agencies" to the bottom (which is now the top) of the chart
    if "Other Agencies" in agency_wordcounts.index:
        other_value = agency_wordcounts["Other Agencies"]
        agency_wordcounts = agency_wordcounts[
            agency_wordcounts.index != "Other Agencies"
        ]
        agency_wordcounts["Other Agencies"] = other_value

    fig = px.bar(
        x=agency_wordcounts.values,
        y=agency_wordcounts.index,
        orientation="h",
        title="Total Word Count by Agency",
        labels={"x": "Total Word Count", "y": "Agency"},
    )

    # Improve readability
    fig.update_layout(
        height=max(
            400, len(agency_wordcounts) * 25
        ),  # Dynamic height based on number of agencies
        margin=dict(l=10, r=10, t=30, b=10),
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
    # Sort agencies by total word count
    word_counts = (
        filtered_df.groupby("agency_group")["word_count"]
        .sum()
        .sort_values(ascending=False)
    )

    # Move "Other Agencies" to the bottom
    if "Other Agencies" in word_counts.index:
        other_value = word_counts["Other Agencies"]
        word_counts = word_counts[word_counts.index != "Other Agencies"]
        word_counts["Other Agencies"] = other_value

    # Create box plot with agencies sorted by word count
    fig = px.box(
        filtered_df,
        x=metric,
        y="agency_group",
        title=f"Distribution of {metric.replace('_', ' ').title()} by Agency",
        labels={"agency_group": "Agency", metric: metric.replace("_", " ").title()},
        category_orders={"agency_group": word_counts.index.tolist()},
    )

    # Improve readability
    fig.update_layout(
        height=max(
            400, len(word_counts) * 25
        ),  # Dynamic height based on number of agencies
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis_title_standoff=10,
    )

    # Add median markers
    fig.update_traces(
        boxmean=True, orientation="h"  # Show mean as dashed line  # Horizontal boxes
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Complexity Summary"):
        stats = (
            filtered_df.groupby("agency_group")
            .agg({metric: ["mean", "median", "std", "count"], "word_count": "sum"})
            .round(2)
            .sort_values(
                ("word_count", "sum"), ascending=False
            )  # Sort by total word count
        )
        # Clean up the column names
        stats.columns = [
            f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in stats.columns
        ]
        st.dataframe(stats)


def plot_correlation(filtered_df: pd.DataFrame):
    """Display correlation analysis for readability metrics."""
    # Select and rename metrics for better clarity
    metric_mapping = {
        "flesch_reading_ease": "Flesch Reading Ease",
        "gunning_fog": "Gunning Fog Index",
        "smog_index": "SMOG Index",
        "automated_readability_index": "Auto. Readability Index",
        "word_count": "Word Count",
        "sentence_count": "Sentence Count",
        "avg_sentence_length": "Avg Sentence Length",
        "type_token_ratio": "Type-Token Ratio",
    }

    metrics = list(metric_mapping.keys())
    correlation = filtered_df[metrics].corr()

    # Rename for display
    correlation.index = [metric_mapping[m] for m in correlation.index]
    correlation.columns = [metric_mapping[m] for m in correlation.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation,
            x=correlation.columns,
            y=correlation.columns,
            text=correlation.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorscale="RdBu",
            zmid=0,  # Center the color scale at 0
        )
    )

    fig.update_layout(
        title={
            "text": "Correlation Between Readability Metrics",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_tickangle=-45,
        height=600,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis={"side": "left"},
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add explanatory text
    st.markdown(
        """
    ### Understanding the Metrics
    - **Flesch Reading Ease**: Higher scores (0-100) indicate easier reading
    - **Gunning Fog Index**: Indicates years of formal education needed (6: sixth grade, 12: high school, 17: college graduate)
    - **SMOG Index**: Estimates years of education needed to understand text
    - **Automated Readability Index**: Another grade-level metric
    - **Type-Token Ratio**: Vocabulary diversity (higher means more diverse vocabulary)
    """
    )

    # Add metric statistics by agency
    with st.expander("View Detailed Metric Statistics by Agency"):
        # Calculate mean metrics for each agency
        agency_metrics = filtered_df.groupby("agency_group")[metrics].mean().round(2)
        agency_metrics.columns = [metric_mapping[col] for col in agency_metrics.columns]
        agency_metrics = agency_metrics.sort_values("Word Count", ascending=False)
        st.dataframe(agency_metrics)


def plot_temporal(filtered_df: pd.DataFrame, metric: str):
    """Display a time-series trend of selected metric by agency."""
    df_copy = filtered_df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])

    # Group by date and agency
    if metric == "word_count":
        # Sum for word count
        temporal_data = (
            df_copy.groupby(["date", "agency_group"])[metric].sum().reset_index()
        )
        y_label = "Total Word Count"
    else:
        # Average for complexity metrics
        temporal_data = (
            df_copy.groupby(["date", "agency_group"])[metric].mean().reset_index()
        )
        y_label = metric.replace("_", " ").title()

    # Sort agencies by their most recent value
    latest_date = temporal_data["date"].max()
    latest_values = temporal_data[temporal_data["date"] == latest_date].sort_values(
        metric, ascending=False
    )
    agency_order = latest_values["agency_group"].tolist()

    # Move "Other Agencies" to bottom of legend
    if "Other Agencies" in agency_order:
        agency_order.remove("Other Agencies")
        agency_order.append("Other Agencies")

    fig = px.line(
        temporal_data,
        x="date",
        y=metric,
        color="agency_group",
        title=f"{y_label} Over Time by Agency",
        labels={
            "date": "Date",
            metric: y_label,
            "agency_group": "Agency",
        },
        category_orders={"agency_group": agency_order},
    )

    # Improve readability
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0, orientation="h"),
        xaxis_title="Year",
        hovermode="x unified",
    )

    # Add hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            + "Date: %{x|%Y-%m-%d}<br>"
            + f"{y_label}: %{{y:,.0f}}<extra></extra>"
        )
    )

    # Add custom data for hover
    fig.update_traces(customdata=temporal_data["agency_group"])

    st.plotly_chart(fig, use_container_width=True)

    # Add summary statistics
    with st.expander("View Temporal Analysis Summary"):
        # Calculate changes between first and last date for each agency
        changes = []
        for agency in agency_order:
            agency_data = temporal_data[temporal_data["agency_group"] == agency]
            if len(agency_data) >= 2:
                first = agency_data.iloc[0][metric]
                last = agency_data.iloc[-1][metric]
                change = last - first
                pct_change = (change / first * 100) if first != 0 else float("inf")
                changes.append(
                    {
                        "Agency": agency,
                        "Initial Value": first,
                        "Final Value": last,
                        "Absolute Change": change,
                        "Percent Change": pct_change,
                    }
                )

        changes_df = pd.DataFrame(changes).round(2)
        changes_df = changes_df.sort_values("Absolute Change", ascending=False)
        st.dataframe(changes_df)


def plot_agency_comparison(filtered_df: pd.DataFrame):
    """Display direct comparison between two selected agencies."""
    # Metric definitions and display names
    metrics = {
        "word_count": "Word Count",
        "flesch_reading_ease": "Reading Ease",
        "gunning_fog": "Complexity Score",
        "avg_sentence_length": "Avg Sentence Length",
        "type_token_ratio": "Vocabulary Diversity",
    }

    # Agency selection
    col1, col2 = st.columns(2)
    with col1:
        agency1 = st.selectbox(
            "Select First Agency",
            options=filtered_df["agency_group"].unique(),
            key="agency1",
        )
    with col2:
        remaining_agencies = [
            a for a in filtered_df["agency_group"].unique() if a != agency1
        ]
        agency2 = st.selectbox(
            "Select Second Agency", options=remaining_agencies, key="agency2"
        )

    # Calculate metrics for both agencies
    agency1_data = filtered_df[filtered_df["agency_group"] == agency1]
    agency2_data = filtered_df[filtered_df["agency_group"] == agency2]

    # Create comparison data
    comparison_data = []
    for metric, display_name in metrics.items():
        val1 = agency1_data[metric].mean()
        val2 = agency2_data[metric].mean()
        diff_pct = ((val2 - val1) / val1) * 100

        comparison_data.append(
            {
                "Metric": display_name,
                f"{agency1}": val1,
                f"{agency2}": val2,
                "Difference (%)": diff_pct,
            }
        )

    # Create comparison chart
    df_comp = pd.DataFrame(comparison_data)

    # Bar chart comparison with separate axes
    fig_bar = go.Figure()

    # Calculate positions for each metric
    num_metrics = len(metrics)
    positions = list(range(num_metrics))

    # Add bars for each agency with separate axes
    for i, (metric, display_name) in enumerate(metrics.items()):
        # Create a separate axis for each metric
        axis_name = f"y{i+1}" if i > 0 else "y"

        # Add bars for both agencies
        for j, agency in enumerate([agency1, agency2]):
            value = df_comp[df_comp["Metric"] == display_name][agency].iloc[0]

            fig_bar.add_trace(
                go.Bar(
                    name=agency,
                    x=[display_name],
                    y=[value],
                    text=[f"{value:,.1f}"],
                    textposition="outside",
                    legendgroup=agency,
                    showlegend=i == 0,
                    marker_color="rgb(55, 83, 109)" if j == 0 else "rgb(26, 118, 255)",
                    yaxis=axis_name,
                    textangle=0,
                )
            )

        # Configure the axis
        if i > 0:
            fig_bar.update_layout(
                **{
                    f"yaxis{i+1}": dict(
                        overlaying="y", visible=False, showgrid=False, zeroline=False
                    )
                }
            )

    # Update layout for bar chart
    fig_bar.update_layout(
        title="Direct Comparison of Metrics",
        barmode="group",
        height=400,
        margin=dict(l=10, r=10, t=30, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        bargap=0.15,
        bargroupgap=0.05,
        uniformtext=dict(mode="hide", minsize=8),
        xaxis=dict(tickangle=0),
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # Display detailed comparison table
    st.markdown("### Detailed Comparison")
    formatted_df = df_comp.copy()
    for col in [agency1, agency2]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.2f}")
    formatted_df["Difference (%)"] = formatted_df["Difference (%)"].apply(
        lambda x: f"{x:+.1f}%"
    )

    st.dataframe(formatted_df.set_index("Metric"))


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

    # Convert date column to datetime first
    df["date"] = pd.to_datetime(df["date"])

    # Year slider first
    if "date" in df.columns:
        min_year = df["date"].dt.year.min()
        max_year = df["date"].dt.year.max()
        selected_year = st.sidebar.slider(
            "Select Year", min_value=2017, max_value=2025, value=2025, step=1
        )

    # Agency selection second
    all_agencies = sorted(df["agency_group"].dropna().unique())
    selected_agencies = st.sidebar.multiselect(
        "Select Agencies", options=all_agencies, default=all_agencies
    )

    # Apply filters in correct order
    filtered_df = df[df["agency_group"].isin(selected_agencies)].copy()
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    filtered_df = filtered_df[filtered_df["date"].dt.year == selected_year]

    # Sidebar: Data download link
    st.sidebar.markdown("### Download Data")
    tmp_download_link = download_link(
        filtered_df, "filtered_data.csv", "Download Filtered Data (CSV)"
    )
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)

    # Sidebar Navigation: Select the analysis section
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Section",
        (
            "Word Count Analysis",
            "Complexity Analysis",
            "Temporal Analysis",
            "Readability Metrics Analysis",
            "Agency Comparison Analysis",
        ),
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

    elif analysis_mode == "Temporal Analysis":
        st.header("Metrics Over Time")
        st.markdown("Explore how regulation metrics change over time for each agency.")
        st.markdown(
            "*Note: This view shows trends across all years, regardless of year selection.*"
        )

        temporal_metric = st.selectbox(
            "Select Metric",
            options=[
                "word_count",
                "flesch_reading_ease",
                "gunning_fog",
                "smog_index",
                "automated_readability_index",
            ],
            format_func=lambda x: (
                "Total Word Count" if x == "word_count" else x.replace("_", " ").title()
            ),
        )

        # Create a separate dataframe for temporal analysis that isn't filtered by year
        temporal_df = df[df["agency_group"].isin(selected_agencies)].copy()
        temporal_df["date"] = pd.to_datetime(temporal_df["date"])
        plot_temporal(temporal_df, temporal_metric)

    elif analysis_mode == "Readability Metrics Analysis":
        st.header("Readability Metrics Analysis")
        st.markdown(
            """
            Explore the relationships between different readability metrics and text characteristics.
            The heatmap shows correlations between metrics, where:
            - Red indicates positive correlation (metrics increase together)
            - Blue indicates negative correlation (one increases as other decreases)
            - Darker colors indicate stronger correlations
            """
        )
        plot_correlation(filtered_df)

    elif analysis_mode == "Agency Comparison Analysis":
        st.header("Agency Comparison Analysis")
        st.markdown(
            """
            Compare any two agencies across multiple metrics. The radar chart shows relative 
            performance across metrics, while the table provides exact values and differences.
            
            - Select two agencies to compare
            - View their relative performance across key metrics
            - See exact values and percentage differences
            """
        )
        plot_agency_comparison(filtered_df)


if __name__ == "__main__":
    main()
