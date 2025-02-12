# Federal Regulations Analysis Pipeline

A Python-based tool for downloading, analyzing, and visualizing the readability of federal regulations from the Electronic Code of Federal Regulations (eCFR).

## Overview

This tool provides an automated pipeline to:
1. Download regulations from the eCFR API
2. Process and analyze text metrics (readability, complexity, etc.)
3. Generate visualizations and statistical summaries

## Features

- Downloads regulations from multiple federal agencies
- Calculates various readability metrics:
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - SMOG Index
  - And more...
- Generates visualizations and statistical reports
- Progress tracking with rich console interface
- Automated data organization and storage

## Installation

```bash
# Clone the repository
git clone https://github.com/csmangum/cfr_tool.git
cd cfr_tool

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main script to start the analysis pipeline:

```bash
python main.py
```

The tool will:
1. Download regulations from eCFR
2. Process and analyze text metrics
3. Generate visualizations and statistics

## Output

Generated files will be organized in the following directories:
- `data/plots/` - Visualization plots
- `data/stats/` - Statistical summaries
- `data/logs/` - Processing logs
- `data/db/` - Database files

## Project Structure

```
cfr_tool/
├── main.py              # Main entry point and pipeline orchestration
├── get_data.py         # eCFR data downloading functionality
├── process_data.py     # Text processing and metrics calculation
├── visualize_metrics.py # Data visualization and reporting
└── data/               # Generated data and output files
    ├── logs/           # Processing logs
    ├── db/            # SQLite database
    ├── plots/         # Generated visualizations
    └── stats/         # Statistical summaries
```

## Dependencies

- rich - For console interface and progress tracking
- requests - For API communication
- textstat - For text analysis metrics
- pandas - For data processing
- matplotlib/seaborn - For visualization
- SQLAlchemy - For database operations