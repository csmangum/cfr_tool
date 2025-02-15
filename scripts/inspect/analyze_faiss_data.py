#!/usr/bin/env python3
"""
Script to analyze the Faiss index and metadata, showing statistics like vector counts by agency.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import faiss
import numpy as np
from tabulate import tabulate


def load_data(index_path: str, metadata_path: str):
    """Load Faiss index and metadata."""
    # Load Faiss index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    return index, metadata


def analyze_vectors(index: faiss.Index, metadata: Dict[str, Any]):
    """Analyze vectors and metadata to generate statistics."""
    total_vectors = index.ntotal
    dimension = index.d  # Should now be 1536
    
    # Add embedding dimension analysis
    embedding_stats = {
        "total_dimension": dimension,
        "base_dimension": 384,
        "metadata_dimensions": {
            "cross_references": 384,
            "definitions": 384,
            "authority": 384
        }
    }
    
    # Add metadata field analysis
    metadata_stats = {
        "cross_references_present": sum(1 for m in metadata.values() 
                                      if m.get("cross_references")),
        "definitions_present": sum(1 for m in metadata.values() 
                                 if m.get("definitions")),
        "authority_present": sum(1 for m in metadata.values() 
                               if m.get("enforcement_agencies"))
    }
    
    # Count vectors by agency
    agency_counts = Counter(item["agency"] for item in metadata.values())

    # Get vectors by title within each agency
    agency_title_counts = defaultdict(Counter)
    for item in metadata.values():
        agency_title_counts[item["agency"]][item["title"]] += 1

    # Count unique values for different fields
    unique_titles = len(set(item["title"] for item in metadata.values()))
    unique_chapters = len(set(item["chapter"] for item in metadata.values()))
    unique_dates = len(set(item["date"] for item in metadata.values()))

    # Add summary stats
    total_count = sum(agency_counts.values())
    summary = {
        "total_count": total_count,
        "agency_summary": [
            {
                "agency": agency,
                "count": count,
                "percentage": (count / total_count * 100),
            }
            for agency, count in agency_counts.most_common()
        ],
    }

    return {
        "total_vectors": total_vectors,
        "dimension": dimension,
        "embedding_stats": embedding_stats,
        "metadata_stats": metadata_stats,
        "agency_counts": agency_counts,
        "agency_title_counts": agency_title_counts,
        "unique_titles": unique_titles,
        "unique_chapters": unique_chapters,
        "unique_dates": unique_dates,
        "summary": summary,
    }


def print_analysis(stats: Dict[str, Any]):
    """Print analysis results in a formatted way."""
    print("\n=== FAISS Index Statistics ===")
    print(f"Total Vectors: {stats['total_vectors']:,}")
    print(f"Vector Dimension: {stats['dimension']}")
    print(f"Unique Titles: {stats['unique_titles']}")
    print(f"Unique Chapters: {stats['unique_chapters']}")
    print(f"Unique Dates: {stats['unique_dates']}")

    print("\n=== Vector Counts by Agency ===")
    # Create table data for agency counts
    agency_table = [
        [agency, count, f"{count/stats['total_vectors']*100:.1f}%"]
        for agency, count in stats["agency_counts"].most_common()
    ]
    print(
        tabulate(
            agency_table,
            headers=["Agency", "Vector Count", "Percentage"],
            tablefmt="grid",
        )
    )

    print("\n=== Detailed Breakdown by Agency and Title ===")
    for agency, title_counts in stats["agency_title_counts"].items():
        print(f"\n{agency}:")
        title_table = [
            [f"Title {title}", count] for title, count in title_counts.most_common()
        ]
        print(tabulate(title_table, headers=["Title", "Count"], tablefmt="simple"))

    print("\n=== Summary of Vector Counts ===")
    print(f"Total Vectors Across All Agencies: {stats['summary']['total_count']:,}")
    print("\nBreakdown by Agency:")
    summary_table = [
        [item["agency"], f"{item['count']:,}", f"{item['percentage']:.1f}%"]
        for item in stats["summary"]["agency_summary"]
    ]
    print(
        tabulate(
            summary_table,
            headers=["Agency", "Vector Count", "Percentage of Total"],
            tablefmt="simple",
            numalign="right",
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Faiss index and metadata statistics."
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/faiss/regulation_index.faiss",
        help="Path to the Faiss index file",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/faiss/regulation_metadata.json",
        help="Path to the metadata JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save analysis results as JSON",
    )

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading index from {args.index}")
        print(f"Loading metadata from {args.metadata}")
        index, metadata = load_data(args.index, args.metadata)

        # Analyze data
        print("Analyzing vectors...")
        stats = analyze_vectors(index, metadata)

        # Print results
        print_analysis(stats)

        # Optionally save results
        if args.output:
            output_path = Path(args.output)
            # Convert Counter objects to dictionaries for JSON serialization
            stats["agency_counts"] = dict(stats["agency_counts"])
            stats["agency_title_counts"] = {
                agency: dict(title_counts)
                for agency, title_counts in stats["agency_title_counts"].items()
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nAnalysis results saved to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
