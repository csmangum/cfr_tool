import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine


# Create necessary directories first
def create_directories():
    """Create required directories if they don't exist"""
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    Path("data/stats").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)


# Create directories before setting up logging
create_directories()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/visualize_metrics.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_data():
    """Load data from SQLite database into a pandas DataFrame"""
    logger.info("Loading data from database")
    try:
        engine = create_engine("sqlite:///data/db/regulations.db")
        df = pd.read_sql_table("regulation_metrics", engine)
        logger.info(f"Successfully loaded {len(df)} records from DB")

        # Load agency mapping from agencies.json to group smaller agencies under appropriate parents
        def load_agency_mapping():
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
                "Other Agencies": [],  # Will contain all other agencies
            }

            agencies_file = Path("data/agencies.json")
            if agencies_file.exists():
                with agencies_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                    # First pass: map known major departments and their children
                    for agency in data.get("agencies", []):
                        slug = agency.get("slug")
                        name = agency.get("display_name")

                        # Find which major agency this belongs to
                        assigned = False
                        for major_name, major_slugs in major_agencies.items():
                            if slug in major_slugs:
                                mapping[slug] = major_name
                                # Map children to parent
                                for child in agency.get("children", []):
                                    child_slug = child.get("slug")
                                    if child_slug:
                                        mapping[child_slug] = major_name
                                assigned = True
                                break

                        # If not assigned to a major department, put in Other Agencies
                        if not assigned:
                            mapping[slug] = "Other Agencies"
                            for child in agency.get("children", []):
                                child_slug = child.get("slug")
                                if child_slug:
                                    mapping[child_slug] = "Other Agencies"

            return mapping

        agency_mapping = load_agency_mapping()
        # Create a new column to hold the group name using the mapping
        df["agency_group"] = df["agency"].apply(
            lambda x: agency_mapping.get(x, "Other Agencies")
        )

        logger.info(f"Successfully loaded and grouped {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def plot_readability_by_agency(df):
    """Create a box plot of readability scores by agency"""
    logger.info("Creating readability by agency plot")
    try:
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, x="agency_group", y="flesch_reading_ease")
        plt.xticks(rotation=45, ha="right")
        plt.title("Flesch Reading Ease Scores by Agency")
        plt.tight_layout()
        plt.savefig("data/plots/readability_by_agency.png")
        plt.close()
        logger.info("Successfully saved readability_by_agency.png")
    except Exception as e:
        logger.error(f"Error creating readability plot: {str(e)}")


def plot_readability_metrics_distribution(df):
    """Create histograms of different readability metrics"""
    metrics = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        "smog_index",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Distribution of Readability Metrics")

    for ax, metric in zip(axes.flat, metrics):
        sns.histplot(data=df, x=metric, ax=ax)
        ax.set_title(metric.replace("_", " ").title())

    plt.tight_layout()
    plt.savefig("data/plots/readability_distributions.png")
    plt.close()


def plot_word_count_by_agency(df):
    """Create a bar plot of average word counts by agency"""
    plt.figure(figsize=(12, 6))

    # Calculate average words by agency group
    avg_words = (
        df.groupby("agency_group")["word_count"]
        .agg(mean="mean", count="count")
        .sort_values("mean", ascending=False)
    )

    # Only show top 13 agencies plus "Other Agencies"
    top_agencies = avg_words.head(13)  # Changed from 14 to 13
    other_agencies = avg_words[13:]  # Changed from 14 to 13

    # Create final plotting data
    plot_data = pd.concat(
        [
            top_agencies,
            pd.DataFrame(
                {
                    "mean": [other_agencies["mean"].mean()],
                    "count": [other_agencies["count"].sum()],
                },
                index=["Other Agencies"],
            ),
        ]
    )

    # Create plot data frame
    plot_df = plot_data.reset_index()
    plot_df.columns = ["agency", "mean", "count"]  # Rename columns for clarity

    # Create bar plot with explicit hue parameter
    sns.barplot(
        data=plot_df,
        x="agency",
        y="mean",
        hue="agency",
        palette=[
            "#ff7f0e" if x == "Other Agencies" else "#1f77b4" for x in plot_df["agency"]
        ],
        legend=False,
    )

    plt.xticks(rotation=45, ha="right")
    plt.title("Average Word Count by Agency")
    plt.ylabel("Average Word Count")
    plt.xlabel("")

    # Add count annotations
    for i, row in enumerate(plot_df.itertuples()):
        plt.text(i, row.mean, f"n={int(row.count)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("data/plots/word_count_by_agency.png", bbox_inches="tight")
    plt.close()


def plot_readability_correlation(df):
    """Create a correlation heatmap of readability metrics"""
    metrics = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        "smog_index",
        "automated_readability_index",
        "coleman_liau_index",
        "linsear_write",
        "dale_chall",
    ]

    plt.figure(figsize=(12, 10))
    sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation between Readability Metrics")
    plt.tight_layout()
    plt.savefig("data/plots/readability_correlation.png")
    plt.close()


def generate_summary_stats(df):
    """Generate and save summary statistics"""
    logger.info("Generating summary statistics")
    try:
        # Overall statistics
        overall_stats = df.describe()
        overall_stats.to_csv("data/stats/overall_statistics.csv")
        logger.info("Saved overall statistics to CSV")

        # Agency-level statistics
        agency_stats = (
            df.groupby("agency_group")
            .agg(
                {
                    "word_count": ["mean", "std"],
                    "flesch_reading_ease": ["mean", "std"],
                    "difficult_words": ["mean", "std"],
                }
            )
            .round(2)
        )
        agency_stats.to_csv("data/stats/agency_statistics.csv")
        logger.info("Saved agency statistics to CSV")

        return overall_stats, agency_stats
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise


def plot_complexity_metrics_distribution(df):
    """Create histograms of different complexity metrics"""
    logger.info("Creating complexity metrics distribution plots")

    metrics = [
        "avg_sentence_length",
        "avg_syllables_per_word",
        "type_token_ratio",
        "polysyllabic_words",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Distribution of Text Complexity Metrics")

    for ax, metric in zip(axes.flat, metrics):
        sns.histplot(data=df, x=metric, ax=ax)
        ax.set_title(metric.replace("_", " ").title())

    plt.tight_layout()
    plt.savefig("data/plots/complexity_distributions.png")
    plt.close()
    logger.info("Saved complexity distributions plot")


def plot_complexity_by_agency(df):
    """Create box plots for complexity metrics by agency"""
    logger.info("Creating complexity by agency plots")

    metrics = [
        ("avg_sentence_length", "Average Sentence Length"),
        ("type_token_ratio", "Vocabulary Diversity (Type-Token Ratio)"),
        ("avg_syllables_per_word", "Average Syllables per Word"),
        ("polysyllabic_words", "Number of Complex Words (3+ syllables)"),
    ]

    for metric, title in metrics:
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, x="agency_group", y=metric)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{title} by Agency")
        plt.tight_layout()
        plt.savefig(f"data/plots/{metric}_by_agency.png")
        plt.close()

    logger.info("Saved complexity by agency plots")


def plot_complexity_correlation_matrix(df):
    """Create a correlation heatmap of complexity metrics"""
    logger.info("Creating complexity correlation matrix")

    metrics = [
        "avg_sentence_length",
        "avg_syllables_per_word",
        "type_token_ratio",
        "polysyllabic_words",
        "word_count",
        "sentence_count",
        "syllable_count",
        "difficult_words",
    ]

    plt.figure(figsize=(12, 10))
    sns.heatmap(df[metrics].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation between Complexity Metrics")
    plt.tight_layout()
    plt.savefig("data/plots/complexity_correlation.png")
    plt.close()
    logger.info("Saved complexity correlation matrix")


def plot_complexity_vs_readability(df):
    """Create scatter plots comparing complexity metrics with readability scores"""
    logger.info("Creating complexity vs readability plots")

    complexity_metrics = [
        "avg_sentence_length",
        "type_token_ratio",
        "avg_syllables_per_word",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Relationship between Complexity Metrics and Readability")

    for ax, metric in zip(axes, complexity_metrics):
        sns.scatterplot(data=df, x=metric, y="flesch_reading_ease", alpha=0.5, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} vs\nFlesch Reading Ease')

    plt.tight_layout()
    plt.savefig("data/plots/complexity_vs_readability.png")
    plt.close()
    logger.info("Saved complexity vs readability plots")


def main():
    logger.info("Starting visualization process")
    try:
        # Set style for all plots
        plt.style.use("seaborn-v0_8")

        # Load data
        df = load_data()

        # Create visualizations
        plot_readability_by_agency(df)
        plot_readability_metrics_distribution(df)
        plot_word_count_by_agency(df)
        plot_readability_correlation(df)

        # Add new complexity visualizations
        plot_complexity_metrics_distribution(df)
        plot_complexity_by_agency(df)
        plot_complexity_correlation_matrix(df)
        plot_complexity_vs_readability(df)

        # Generate statistics
        overall_stats, agency_stats = generate_summary_stats(df)

        # Print some key findings
        logger.info("\nKey Findings:")
        logger.info("-" * 50)
        logger.info(f"Total number of documents analyzed: {len(df)}")
        logger.info(f"Number of agencies: {df['agency_group'].nunique()}")

        # Add complexity metrics to key findings
        logger.info("\nComplexity Metrics Averages:")
        logger.info(
            f"Average sentence length: {df['avg_sentence_length'].mean():.2f} words"
        )
        logger.info(
            f"Average syllables per word: {df['avg_syllables_per_word'].mean():.2f}"
        )
        logger.info(f"Average type-token ratio: {df['type_token_ratio'].mean():.2f}")
        logger.info(
            f"Average number of complex words: {df['polysyllabic_words'].mean():.2f}"
        )

        # Most readable and least readable agencies
        most_readable = (
            df.groupby("agency_group")["flesch_reading_ease"].mean().idxmax()
        )
        least_readable = (
            df.groupby("agency_group")["flesch_reading_ease"].mean().idxmin()
        )

        logger.info(f"\nMost readable agency: {most_readable}")
        logger.info(f"Least readable agency: {least_readable}")

        logger.info("Visualization process completed successfully")

    except Exception as e:
        logger.error(f"Error in main visualization process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
