# Readability Metrics

This document explains the various readability metrics used to analyze federal regulations. These metrics help assess how easily regulations can be understood by different audiences.

## Overview

The system calculates multiple readability scores to provide a comprehensive view of text complexity. Each metric uses different factors and formulas to estimate reading difficulty.

## Core Metrics

### 1. Flesch Reading Ease
- **Scale**: 0-100 (higher scores = easier to read)
- **Formula**: 206.835 - 1.015(total words/total sentences) - 84.6(total syllables/total words)
- **Interpretation**:
  - 90-100: Very easy (5th grade)
  - 80-89: Easy (6th grade)
  - 70-79: Fairly easy (7th grade)
  - 60-69: Standard (8th-9th grade)
  - 50-59: Fairly difficult (10th-12th grade)
  - 30-49: Difficult (College)
  - 0-29: Very difficult (College graduate)

### 2. Flesch-Kincaid Grade Level
- **Scale**: Grade level (e.g., 8.2 = 8th grade, 2nd month)
- **Formula**: 0.39(total words/total sentences) + 11.8(total syllables/total words) - 15.59
- **Use Case**: Widely used in government documents and technical writing
- **Target**: Most regulations aim for grades 10-12

### 3. Gunning Fog Index
- **Scale**: Grade level
- **Formula**: 0.4[(words/sentences) + 100(complex words/words)]
- **Features**:
  - Emphasizes sentence length
  - Counts words with 3+ syllables
  - Used for technical and business documents
- **Target**: Aim for scores 12 or lower

### 4. SMOG Index (Simple Measure of Gobbledygook)
- **Scale**: Grade level
- **Formula**: 1.0430 × sqrt(number of polysyllables × (30/number of sentences)) + 3.1291
- **Features**:
  - Focuses on polysyllabic words
  - More accurate for higher reading levels
  - Used in healthcare and academic documents

## Additional Metrics

### 5. Automated Readability Index (ARI)
- **Scale**: Grade level
- **Formula**: 4.71(characters/words) + 0.5(words/sentences) - 21.43
- **Feature**: Uses character count instead of syllables

### 6. Coleman-Liau Index
- **Scale**: Grade level
- **Formula**: 0.0588L - 0.296S - 15.8
  - L = average number of letters per 100 words
  - S = average number of sentences per 100 words

### 7. Dale-Chall Readability Score
- **Scale**: Grade level
- **Features**:
  - Uses a list of 3,000 common words
  - Counts "difficult" words not on the list
  - Particularly relevant for regulatory text

## Implementation

The metrics are calculated in `scripts/process_data.py`:

```python
def calculate_metrics(text):
    """Calculate comprehensive readability metrics for regulation text."""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "dale_chall": textstat.dale_chall_readability_score(text)
    }
```

## Usage in Analysis

The system uses these metrics to:
1. Compare readability across different agencies
2. Track changes in regulation complexity over time
3. Identify regulations that may need simplification
4. Generate readability reports and visualizations

## Visualization Examples

The system generates several visualizations:
- Distribution of readability scores
- Agency comparisons
- Correlation between different metrics
- Historical trends

Example visualization code from `scripts/visualize_metrics.py`:

```python
def plot_readability_by_agency(df):
    """Create box plot of readability scores across agencies."""
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df, x="agency_group", y="flesch_reading_ease")
    plt.xticks(rotation=45, ha="right")
    plt.title("Flesch Reading Ease Scores by Agency")
    plt.tight_layout()
    plt.savefig("data/plots/readability_by_agency.png")
```

## Best Practices

When analyzing regulation readability:
1. Consider multiple metrics for a comprehensive view
2. Account for technical terminology requirements
3. Focus on trends and relative comparisons
4. Use visualizations to communicate findings
5. Consider the target audience for each regulation

## Limitations

Important considerations when using these metrics:
1. May not fully account for necessary technical language
2. Don't measure clarity of organization or logic
3. Can't assess accuracy or completeness
4. May not reflect domain expertise of target readers

## Future Improvements

Potential enhancements to readability analysis:
1. Domain-specific readability metrics
2. Machine learning-based complexity measures
3. Context-aware scoring adjustments
4. Interactive visualization tools
5. Automated improvement suggestions 