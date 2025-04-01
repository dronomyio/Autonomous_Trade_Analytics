# Bank Regulatory Data Analysis

This directory contains bank regulatory data files and utilities for analysis, with a focus on FR Y-9C filings.

## Y-9C Data Format

The Federal Reserve Form FR Y-9C is filed quarterly by bank holding companies with assets of $3 billion or more. This data is valuable for detecting anomalies and trends in bank financial health.

### Key Metrics in Y-9C Filings

| Metric | Description | Typical Range | Potential Red Flags |
|--------|-------------|---------------|---------------------|
| total_assets | Total assets on balance sheet | Varies by bank size | Sudden large decrease |
| tier1_capital_ratio | Core capital / Risk-weighted assets | 7-15% | Below 8% or rapid decline |
| total_capital_ratio | Total capital / Risk-weighted assets | 10-18% | Below 10.5% or rapid decline |
| leverage_ratio | Tier 1 capital / Total assets | 5-10% | Below 5% or rapid decline |
| npl_ratio | Non-performing loans / Total loans | 0.5-2% | Above 2% or rapid increase |
| roa | Return on assets (annualized) | 0.5-1.5% | Negative or rapid decline |
| roe | Return on equity (annualized) | 8-15% | Below 5% or rapid decline |
| net_interest_margin | Net interest income / Avg earning assets | 2.5-4% | Below 2% or rapid decline |
| efficiency_ratio | Non-interest expense / Revenue | 50-70% | Above 75% (lower is better) |
| liquid_assets_ratio | Liquid assets / Total assets | 15-30% | Below 15% or rapid decline |
| loan_to_deposit_ratio | Total loans / Total deposits | 70-90% | Above 100% or rapid increase |
| trading_assets_ratio | Trading assets / Total assets | Varies by bank type | Rapid change in either direction |

## Anomaly Detection

The Y-9C anomaly detector identifies unusual patterns in bank filing data using:

1. **Statistical Anomalies**: Values that deviate significantly from historical patterns
2. **Sequential Anomalies**: Quarter-to-quarter changes that exceed normal thresholds
3. **LLM Analysis**: Claude-powered analysis of detected anomalies with banking expertise

### Anomaly Detection Prompt Structure

The anomaly detection prompt for Claude follows this structure:

```
I'm analyzing anomalies in FR Y-9C bank regulatory filings. I've detected the following unusual patterns in the data:

ANOMALY SUMMARY:
Bank ID: [ID] (Total Assets: $[AMOUNT])
- [metric]: [value] (Z-score: [score], unusual [direction])
- [metric]: [pct_change]% [direction] from [previous] to [current]

METRICS EXPLANATION:
[Detailed explanation of banking metrics]

QUESTIONS:
1. What are the most concerning anomalies in this data and why?
2. What potential explanations could there be for these anomalies?
3. What additional information would be helpful to investigate these anomalies further?
4. What recommendations would you make to risk managers reviewing these results?
5. What regulatory implications of these anomalies should be considered?

Please analyze these anomalies in detail, considering banking industry trends, regulatory concerns, and potential business implications.
```

## Usage Examples

### Basic Anomaly Detection

```python
from tools.y9c_anomaly_detector import Y9CAnomalyDetector

# Initialize the detector
detector = Y9CAnomalyDetector()

# Load Y-9C data
data = detector.load_y9c_data()

# Detect statistical anomalies
anomalies = detector.detect_statistical_anomalies(data)
print(f"Found {len(anomalies)} statistical anomalies")

# Detect sequential anomalies (quarter-to-quarter changes)
seq_anomalies = detector.detect_sequential_anomalies(data)
print(f"Found {len(seq_anomalies)} sequential anomalies")
```

### Using Claude for Analysis

```python
from tools.y9c_anomaly_detector import Y9CAnomalyDetector
import os

# Get API key
api_key = os.environ.get('ANTHROPIC_API_KEY')

# Initialize with API key
detector = Y9CAnomalyDetector(api_key=api_key)

# Run full analysis workflow
results = detector.full_anomaly_analysis(use_claude=True)

# View Claude's analysis
print(results['claude_analysis'])
```

### Visual Analysis

```python
from tools.y9c_anomaly_detector import Y9CAnomalyDetector

# Initialize detector
detector = Y9CAnomalyDetector()

# Load data
data = detector.load_y9c_data()

# Plot trends for a specific bank with anomaly highlighting
detector.plot_metric_trends(data, bank_id=9012)
```