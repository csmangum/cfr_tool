"""
Other Agencies Analysis Script

This script analyzes which agencies are being grouped into the "Other Agencies" category
by examining the agency mapping logic in the data processing pipeline.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def setup_logging():
    """Configure logging to write to both file and console."""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "other_agencies_analysis.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_agency_data():
    """Load the agencies data from the JSON file."""
    try:
        with Path("data/agencies.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: data/agencies.json not found")
        return None


def get_major_agencies_mapping():
    """Return the mapping of major agencies and their slugs."""
    return {
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
    }


def analyze_other_agencies():
    """Analyze which agencies are being grouped into 'Other Agencies'."""
    setup_logging()
    agencies_data = load_agency_data()
    if not agencies_data:
        return

    major_agencies = get_major_agencies_mapping()

    # Create reverse mapping from slug to major agency
    slug_to_major = {}
    for major, slugs in major_agencies.items():
        for slug in slugs:
            slug_to_major[slug] = major

    # Track agencies that end up in "Other Agencies"
    other_agencies = defaultdict(list)
    total_word_counts = defaultdict(int)

    # Load regulation metrics from database
    try:
        engine = create_engine("sqlite:///data/db/regulations.db")
        metrics_df = pd.read_sql_table("regulation_metrics", engine)
    except Exception as e:
        print(f"Error loading database: {e}")
        metrics_df = None

    # Analyze each agency
    for agency in agencies_data.get("agencies", []):
        slug = agency.get("slug")
        name = agency.get("name")

        # Check if this agency maps to a major department
        major_dept = slug_to_major.get(slug)

        if not major_dept:
            # This agency will be grouped into "Other Agencies"
            agency_info = {
                "name": name,
                "slug": slug,
                "cfr_references": agency.get("cfr_references", []),
                "children": [
                    {"name": child.get("name"), "slug": child.get("slug")}
                    for child in agency.get("children", [])
                ],
            }

            # Get word count metrics if available
            if metrics_df is not None:
                word_count = metrics_df[metrics_df["agency"] == slug][
                    "word_count"
                ].sum()
                total_word_counts[slug] = word_count
                agency_info["total_word_count"] = word_count

            other_agencies[name].append(agency_info)

    # Print analysis results
    logging.info("\nAGENCIES GROUPED AS 'OTHER AGENCIES'")
    logging.info("=" * 80)

    # Sort by word count if available, otherwise by name
    if total_word_counts:
        sorted_agencies = sorted(
            other_agencies.items(),
            key=lambda x: sum(total_word_counts[info["slug"]] for info in x[1]),
            reverse=True,
        )
    else:
        sorted_agencies = sorted(other_agencies.items())

    for agency_name, info_list in sorted_agencies:
        logging.info(f"\n{agency_name}")
        logging.info("-" * len(agency_name))

        for info in info_list:
            word_count = info.get("total_word_count", "N/A")
            if word_count != "N/A":
                word_count = f"{word_count:,}"
            logging.info(f"Slug: {info['slug']}")
            logging.info(f"Total Word Count: {word_count}")

            if info["cfr_references"]:
                logging.info("CFR References:")
                for ref in info["cfr_references"]:
                    logging.info(
                        f"  - Title {ref.get('title')}, Chapter {ref.get('chapter')}"
                    )

            if info["children"]:
                logging.info("Sub-agencies:")
                for child in info["children"]:
                    logging.info(f"  - {child['name']} (slug: {child['slug']})")
            logging.info("")

    # Print summary statistics
    logging.info("\nSUMMARY STATISTICS")
    logging.info("=" * 80)
    logging.info(f"Total number of 'Other Agencies': {len(other_agencies)}")

    if total_word_counts:
        total_words = sum(total_word_counts.values())
        logging.info(f"Total word count across all 'Other Agencies': {total_words:,}")

        # Calculate percentage of total regulations
        total_all_words = metrics_df["word_count"].sum()
        percentage = (total_words / total_all_words) * 100 if total_all_words > 0 else 0
        logging.info(f"Percentage of total regulations: {percentage:.1f}%")


if __name__ == "__main__":
    analyze_other_agencies()
