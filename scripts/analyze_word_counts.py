"""
Word Count Analysis Script

This script analyzes the word counts of regulation text files and creates plots
showing how the regulations have changed over time.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def count_words(text_file: Path) -> int:
    """Count words in a text file."""
    with text_file.open("r", encoding="utf-8") as f:
        text = f.read()
        # Split on whitespace and filter out empty strings
        words = [w for w in re.split(r"\s+", text) if w]
        return len(words)


def extract_date_from_filename(filename: str) -> str:
    """Extract date from filename format title_X_chapter_Y_YYYY-MM-DD.txt"""
    match = re.search(r"(\d{4}-\d{2}-\d{2})\.txt$", filename)
    if match:
        return match.group(1)
    return None


def analyze_agency_word_counts(agency_dir: Path) -> pd.DataFrame:
    """
    Analyze word counts for all text files in an agency directory.
    Returns a DataFrame with dates and word counts, aggregated by year.
    """
    text_dir = agency_dir / "text"
    if not text_dir.exists():
        return pd.DataFrame()

    data = []
    for text_file in text_dir.glob("*.txt"):
        date = extract_date_from_filename(text_file.name)
        if date:
            word_count = count_words(text_file)
            data.append(
                {
                    "date": pd.to_datetime(date),
                    "word_count": word_count,
                    "agency": agency_dir.name,
                }
            )

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Aggregate word counts by year
    df["year"] = df["date"].dt.year
    yearly_counts = (
        df.groupby("year")
        .agg(
            {
                "word_count": "sum",
                "agency": "first",
                "date": lambda x: x.iloc[0],  # Keep first date for plotting
            }
        )
        .reset_index()
    )

    return yearly_counts


def plot_word_counts(df: pd.DataFrame, output_dir: Path):
    """Create plots of word count trends."""
    # Set style
    plt.style.use("seaborn-v0_8")

    # Create the plots directory if it doesn't exist
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Sort data chronologically
    df_sorted = df.sort_values("date")

    # Line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_sorted, x="year", y="word_count", marker="o")
    plt.title("Total Regulation Word Count by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Word Count")
    plt.grid(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "word_count_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_sorted, x="year", y="word_count")
    plt.title("Total Regulation Word Count by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Word Count")
    plt.grid(True, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "word_count_bars.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save the data
    df_sorted.to_csv(plots_dir / "word_counts.csv", index=False)

    # Print summary statistics
    print("\nYearly Word Count Summary Statistics:")
    print(df_sorted["word_count"].describe().round(2))

    # Print total word counts and changes between years
    print("\nYearly Word Counts and Changes:")
    for i in range(len(df_sorted)):
        curr = df_sorted.iloc[i]
        print(f"{curr['year']}: {curr['word_count']:,d} words", end="")

        if i > 0:
            prev = df_sorted.iloc[i - 1]
            change = curr["word_count"] - prev["word_count"]
            pct_change = (change / prev["word_count"]) * 100
            print(f" (Change: {change:+,d} words, {pct_change:+.1f}%)")
        else:
            print()  # Just add newline for first year


def main():
    # Base directory
    base_dir = Path("data")
    agency_slug = "treasury-department"
    agency_dir = base_dir / "agencies" / agency_slug

    # Analyze word counts
    df = analyze_agency_word_counts(agency_dir)

    if df.empty:
        print(f"No text files found for agency: {agency_slug}")
        return

    # Create plots
    plot_word_counts(df, base_dir)
    print(f"Analysis complete. Plots saved in {base_dir}/plots/")


if __name__ == "__main__":
    main()
