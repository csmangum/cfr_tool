import time
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

from get_data import ECFRDownloader
from process_data import process_agencies
from visualize_metrics import main as visualize_main


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
[green]3.[/green] Generate visualizations and statistics
    """

    console.print(Panel(title, expand=False))
    console.print(Panel(steps, title="Steps", expand=False))

    # Ensure data directories exist
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/db").mkdir(parents=True, exist_ok=True)
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    Path("data/stats").mkdir(parents=True, exist_ok=True)


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
        downloader = ECFRDownloader()
        downloader.download_all_agencies()
        progress.update(download_task, completed=100)

        # Process data
        process_task = progress.add_task(
            "[magenta]Processing regulations...", total=100
        )
        process_agencies()
        progress.update(process_task, completed=100)

        # Visualize data
        viz_task = progress.add_task("[yellow]Generating visualizations...", total=100)
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
    """
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[bold red]Process interrupted by user[/bold red]")
    except Exception as e:
        rprint(f"\n[bold red]Error: {str(e)}[/bold red]")
