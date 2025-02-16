"""
Federal Regulations Analysis Pipeline

This module serves as the main entry point for processing and analyzing federal regulations.
It orchestrates a multi-step pipeline that:

1. Downloads regulations from the Electronic Code of Federal Regulations (eCFR)
2. Processes and analyzes text metrics for readability
3. Generates embeddings for semantic search capabilities
4. Exports embeddings to a FAISS index for efficient similarity search
5. Generates visualizations and statistical analyses

The pipeline includes checkpoints at each stage to avoid reprocessing existing data,
making it efficient for incremental updates and reruns.

Output Directories:
    - data/plots/: Visualization plots and charts
    - data/stats/: Statistical summaries and metrics
    - data/logs/: Processing logs and debug information
    - data/faiss/: FAISS search index and metadata
    - data/db/: SQLite databases containing processed data

Usage:
    Run this script directly to execute the full pipeline:
    $ python main.py

Dependencies:
    - rich: For console output and progress tracking
    - faiss: For similarity search indexing
    - Various custom pipeline modules in the pipelines/ directory
"""

import json
from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from pipelines.embed_regulations import process_regulations
from pipelines.export_to_faiss import export_to_faiss
from pipelines.get_data import ECFRDownloader
from pipelines.process_data import process_agencies
from pipelines.visualize_metrics import main as visualize_main

from scripts.search_regulations import BooleanQueryProcessor, TemporalSearchEnhancer, AgencyRelationshipMapper


def load_agencies():
    """Load the agencies from the JSON file."""
    with open("data/agencies.json") as f:
        return json.load(f)["agencies"]


def check_agencies_exist():
    """Check if all agency folders exist in data/agencies."""
    if not Path("data/agencies.json").exists():
        return False

    agencies = load_agencies()
    agencies_dir = Path("data/agencies")

    # Get all expected agency slugs (including children)
    expected_slugs = set()
    for agency in agencies:
        expected_slugs.add(agency["slug"])
        for child in agency.get("children", []):
            expected_slugs.add(child["slug"])

    # Check if all agency folders exist
    existing_folders = {p.name for p in agencies_dir.iterdir() if p.is_dir()}
    return expected_slugs.issubset(existing_folders)


def display_intro():
    """Display a fancy intro message"""
    console = Console()

    title = """
[bold blue]Federal Regulations Analysis Pipeline[/bold blue]
[dim]A tool for analyzing the readability of federal regulations[/dim]
    """

    steps = """
[green]1.[/green] Download regulations from eCFR
[green]2.[/green] Process and analyze text metrics
[green]3.[/green] Generate embeddings for search
[green]4.[/green] Export embeddings to FAISS index
[green]5.[/green] Generate visualizations and statistics
    """

    console.print(Panel(title, expand=False))
    console.print(Panel(steps, title="Steps", expand=False))

    # Ensure data directories exist
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/db").mkdir(parents=True, exist_ok=True)
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    Path("data/stats").mkdir(parents=True, exist_ok=True)
    Path("data/faiss").mkdir(parents=True, exist_ok=True)


def check_db_exists(db_path: str) -> bool:
    """Check if a database file exists and has content."""
    db_file = Path(db_path)
    return db_file.exists() and db_file.stat().st_size > 0


def check_faiss_exists() -> bool:
    """Check if FAISS index and metadata files exist."""
    index_file = Path("data/faiss/regulation_index.faiss")
    metadata_file = Path("data/faiss/regulation_metadata.json")
    return (
        index_file.exists()
        and metadata_file.exists()
        and index_file.stat().st_size > 0
        and metadata_file.stat().st_size > 0
    )


def main():
    display_intro()
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Download data
        download_task = progress.add_task("[cyan]Downloading regulations...", total=100)
        if check_agencies_exist():
            rprint(
                "[yellow]Agency folders already exist - skipping download step[/yellow]"
            )
            progress.update(download_task, completed=100)
        else:
            downloader = ECFRDownloader()
            downloader.download_all_agencies(progress=progress, task_id=download_task)
            progress.update(download_task, completed=100)

        # Process data
        process_task = progress.add_task(
            "[magenta]Processing regulations...", total=100
        )
        if check_db_exists("data/db/regulations.db"):
            rprint(
                "[yellow]Regulations database already exists - skipping processing step[/yellow]"
            )
            progress.update(process_task, completed=100)
        else:
            process_agencies()
            progress.update(process_task, completed=100)

        # Generate embeddings
        embed_task = progress.add_task("[yellow]Generating embeddings...", total=100)
        if check_db_exists("data/db/regulation_embeddings.db"):
            rprint(
                "[yellow]Embeddings database already exists - skipping embedding generation[/yellow]"
            )
            progress.update(embed_task, completed=100)
        else:
            process_regulations(
                config_path=Path("config/default.yml"), data_dir=Path("data/agencies")
            )
            progress.update(embed_task, completed=100)

        # Export to FAISS
        faiss_task = progress.add_task("[blue]Exporting to FAISS index...", total=100)
        if check_faiss_exists():
            rprint("[yellow]FAISS index already exists - skipping export step[/yellow]")
            progress.update(faiss_task, completed=100)
        else:
            export_to_faiss(
                db_path="data/db/regulation_embeddings.db",
                index_out_path="data/faiss/regulation_index.faiss",
                metadata_out_path="data/faiss/regulation_metadata.json",
            )
            progress.update(faiss_task, completed=100)

        # Visualize data
        viz_task = progress.add_task("[green]Generating visualizations...", total=100)
        visualize_main()
        progress.update(viz_task, completed=100)

    # Final success message
    rprint("\n[bold green]✨ Analysis complete! ✨[/bold green]")
    rprint(
        """
[dim]Generated files can be found in:[/dim]
  • [blue]data/plots/[/blue] - Visualization plots
  • [blue]data/stats/[/blue] - Statistical summaries
  • [blue]data/logs/[/blue]  - Processing logs
  • [blue]data/faiss/[/blue] - Search index files
  • [blue]data/db/[/blue]    - Databases
    """
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[bold red]Process interrupted by user[/bold red]")
    except Exception as e:
        rprint(f"\n[bold red]Error: {str(e)}[/bold red]")
