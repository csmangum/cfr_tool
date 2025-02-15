# Application Interface

This document describes the interactive web applications for analyzing and searching federal regulations.

![Analysis Dashboard](../images/app.png)
*Federal Regulations Analysis Dashboard showing word count analysis and agency comparisons*

## Overview

The application suite consists of two Streamlit interfaces:
1. Analysis Dashboard (`streamlit_app.py`)
2. Semantic Search Interface (`streamlit_search.py`)

## Analysis Dashboard

### Running the Dashboard
```bash
streamlit run streamlit_app.py
```

### Features

#### 1. Data Filtering
- **Year Selection**: Filter regulations by year (2017-2025)
- **Agency Selection**: Multi-select agencies to include
- **Data Download**: Export filtered data as CSV

#### 2. Analysis Sections

##### Word Count Analysis
- Total word count by agency
- Average words per agency
- Largest agency identification
- Complexity metrics overview
- Interactive bar charts

##### Complexity Analysis
- Distribution of complexity metrics:
  - Flesch Reading Ease
  - Gunning Fog Index
  - SMOG Index
  - Automated Readability Index
- Box plots with statistical summaries

##### Temporal Analysis
- Metric trends over time
- Agency-specific temporal patterns
- Interactive line charts
- Change analysis summaries

##### Readability Metrics Analysis
- Correlation heatmap
- Metric relationships
- Statistical summaries
- Agency-level comparisons

##### Agency Comparison Analysis
- Direct comparison between two agencies
- Multiple metrics visualization
- Percentage differences
- Detailed comparison tables

### Dashboard Layout
```
├── Header
├── Sidebar
│   ├── Year Filter
│   ├── Agency Selection
│   ├── Data Download
│   └── Analysis Section Selection
└── Main Content
    ├── Visualizations
    ├── Statistics
    ├── Explanatory Text
    └── Interactive Elements
```

## Semantic Search Interface

![Search Interface](../images/search.png)
*Semantic Search Interface with filtering options and results display*

### Running the Search Interface
```bash
streamlit run streamlit_search.py
```

### Features

#### 1. Search Options
- **Results Count**: Adjust number of results (1-20)
- **Agency Filters**: Filter by specific agencies
- **Date Range**: Filter by regulation date
- **Similarity Threshold**: Set minimum similarity score
- **Sample Questions**: Try random example queries

#### 2. Search Results
- Relevance scores
- Agency metadata
- Regulation text
- Interactive expandable sections

#### 3. Export Options
- CSV format
- JSON format
- TXT format
- Custom formatting options

#### 4. Visualizations
- Score distribution histograms
- Agency breakdown pie charts
- Interactive plots

#### 5. Search History
- Recent queries list
- One-click rerun
- History clearing

### Interface Layout
```
├── Header
├── Sidebar
│   ├── Search Options
│   ├── Filters
│   ├── Sample Questions
│   └── Search History
└── Main Content
    ├── Search Bar
    ├── Results Display
    │   ├── Metadata
    │   ├── Text Content
    │   └── Similarity Score
    ├── Export Options
    └── Visualizations
```

## Usage Examples

### Analysis Dashboard
```python
# Example: Filtering and analyzing specific agencies
1. Select year range: 2023
2. Choose agencies: "Department of Defense", "Department of Energy"
3. Navigate to "Complexity Analysis"
4. Select "Gunning Fog Index" metric
5. Export filtered data
```

### Search Interface
```python
# Example: Searching for specific regulations
1. Enter query: "What are the requirements for cybersecurity compliance?"
2. Set filters:
   - Minimum similarity: 0.7
   - Agencies: ["Department of Defense", "Department of Homeland Security"]
   - Date range: Last 2 years
3. Export results as JSON
```

## Performance Considerations

### Dashboard
- Caches data loading
- Optimizes plot rendering
- Handles large datasets efficiently
- Responsive interface design

### Search Interface
- Fast similarity search using FAISS
- Efficient metadata retrieval
- Optimized result formatting
- Responsive query handling

## Customization

### Dashboard Customization
```python
# Add new visualization
def plot_custom_metric(filtered_df):
    # Custom visualization code
    pass

# Add to analysis sections
if analysis_mode == "Custom Analysis":
    plot_custom_metric(filtered_df)
```

### Search Interface Customization
```python
# Add custom filter
with st.sidebar:
    custom_filter = st.selectbox(
        "Custom Filter",
        options=["Option 1", "Option 2"]
    )
```

## Troubleshooting

Common issues and solutions:
1. **Slow Loading**
   - Reduce date range
   - Filter agencies
   - Clear cache

2. **Search Issues**
   - Check minimum similarity score
   - Verify database connection
   - Refresh FAISS index

3. **Visualization Errors**
   - Check data filtering
   - Verify metric selection
   - Reset to defaults

## References

For additional information:
- [Analysis Documentation](Analysis.md)
- [Pipeline Documentation](Pipeline.md)
- [Search Documentation](Search.md)
- [Streamlit Documentation](https://docs.streamlit.io) 