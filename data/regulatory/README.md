# Regulatory Filing Analysis

This directory contains regulatory filing data and utilities for analyzing financial institutions and public companies, with a focus on FR Y-9C and SEC 10-K filings.

## Data Structure

The regulatory data is organized into subdirectories:

- `/y9c/` - FR Y-9C data for bank holding companies
- `/10k/` - SEC 10-K annual report data for public companies

## Y-9C Bank Filing Analysis

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

### Y-9C Anomaly Detection Prompt

The Y-9C anomaly detection prompt follows this detailed structure:

```
# Bank Regulatory Anomaly Analysis: FR Y-9C Filing Review

## BANK CONTEXT
Bank ID: [Bank Identifier]
Bank Name: [Bank Name if available]
Asset Size: $[Total Assets] billion
Business Model: [Commercial/Investment/Universal/Regional] 
Peer Group: [Relevant peer group details]

## DETECTED ANOMALIES

### Statistical Anomalies (Z-Score Analysis)
The following metrics show significant deviation from this bank's historical patterns:

1. [Metric Name]: [Current Value] (Z-score: [Z-Score Value])
   - Historical average: [Mean Value]
   - Standard deviation: [Std Dev Value]
   - Deviation direction: [Increase/Decrease]
   - Percentile in peer group: [Percentile]%

### Sequential Anomalies (Quarter-over-Quarter Changes)
The following metrics show unusual changes compared to the previous quarter:

1. [Metric Name]: [Current Value]
   - Previous quarter: [Previous Value]
   - Change: [Percentage]% [Increase/Decrease]
   - Typical quarterly change range: [Range]%
   - Significance: [X] times larger than typical change

### Related Metrics Comparison
Unusual relationships between traditionally correlated metrics:

1. Metric Pair: [Metric A] vs [Metric B]
   - Historical correlation: [Correlation Value]
   - Current relationship: [Description of unusual pattern]
   - Potential significance: [Brief explanation]

## RECENT BANK ACTIVITIES
Notable activities disclosed in recent filings or public records:
- [Activity 1 - e.g., Major acquisition]
- [Activity 2 - e.g., Change in business strategy]

## ECONOMIC CONTEXT
Current macroeconomic conditions that may be relevant:
- Interest Rate Environment: [Current conditions]
- Credit Cycle Phase: [Expansion/Contraction/Stress]

## REGULATORY CONSIDERATIONS
- Applicable regulatory thresholds: [Relevant capital or other requirements]
- Recent regulatory changes affecting this metric: [If applicable]

## ANALYSIS QUESTIONS
1. Critical Assessment of Anomalies:
   - Which of these anomalies represent the most significant potential risks?
   - Are there any anomalies that are likely benign given the context?

2. Potential Explanations:
   - What are the most likely explanations for the observed anomalies?
   - Are these anomalies likely interconnected or representing separate issues?

3. Hidden Concerns:
   - What potentially concerning issues might these anomalies be signaling?
   - Are there any "leading indicator" anomalies that could presage more significant problems?

4. Investigation Recommendations:
   - What specific additional data should examiners request to better understand these anomalies?
   - Which areas of the bank's operations would you prioritize for deeper examination?

5. Reporting Recommendations:
   - How should these findings be characterized in regulatory reporting?
   - What level of regulatory concern is warranted based on these anomalies?
```

## 10-K SEC Filing Analysis

### Key Metrics in 10-K Filings

| Category | Metrics | Description |
|----------|---------|-------------|
| Income Statement | revenue, revenue_growth, gross_margin, operating_margin, net_margin, eps, eps_growth, tax_rate | Profitability and growth measures |
| Balance Sheet | debt_to_equity, asset_turnover, inventory_turnover, accounts_receivable_days, goodwill_to_assets, pe_ratio | Efficiency and leverage measures |
| Cash Flow | free_cash_flow, capex_to_revenue, interest_coverage | Liquidity and cash management |
| Non-Financial | auditor_changed, material_weakness, textual_sentiment, textual_complexity, new_risk_language, ceo_changed, cfo_changed, major_acquisition, etc. | Governance and disclosure indicators |

### 10-K Anomaly Detection Prompt

The 10-K anomaly detection prompt follows this structured format:

```
# Corporate Financial Anomaly Analysis: 10-K Filing Review

## COMPANY CONTEXT
Company: [Company Name] ([Ticker Symbol])
Industry: [Industry Classification]
Market Cap: $[Market Capitalization] billion
Business Model: [Brief description of core business]
Fiscal Year End: [Date]
Filing Date: [10-K submission date]
Peer Group: [Key competitors or industry peers]

## DETECTED ANOMALIES

### Financial Statement Anomalies

#### Income Statement Anomalies
The following metrics show significant deviation from historical patterns or industry norms:

1. [Metric Name]: [Current Value] (Z-score: [Z-Score Value])
   - Historical average: [Mean Value]
   - YoY change: [Percentage]% [Increase/Decrease]
   - Industry average: [Value]
   - 5-year trend: [Brief description]
   - Unusual because: [Brief explanation]

#### Balance Sheet Anomalies
The following balance sheet items show unusual patterns:
...

#### Cash Flow Statement Anomalies
The following cash flow items show unusual patterns:
...

### MD&A and Footnote Anomalies
The following disclosures in Management Discussion & Analysis or footnotes warrant attention:
...

### Auditor and Internal Control Anomalies
Unusual patterns related to auditors or internal controls:
...

### Textual Analysis Anomalies
The following textual changes in the 10-K suggest potential areas of concern:
...

## CORPORATE EVENT CONTEXT
Recent significant corporate events:
- [Event 1 - e.g., CEO/CFO change]
- [Event 2 - e.g., Major acquisition/divestiture]

## INDUSTRY/ECONOMIC CONTEXT
- Industry performance period: [Growth/Contraction/Disruption]
- Competitive landscape changes: [Major developments]
- Regulatory environment: [New regulations or enforcement trends]

## ANALYSIS QUESTIONS

1. Significance Assessment:
   - Which anomalies represent the most significant potential risks to the company's financial health or valuation?
   - Which anomalies might indicate potential accounting irregularities versus legitimate business changes?

2. Narrative vs. Numbers Analysis:
   - Are there inconsistencies between the company's narrative explanation in MD&A and the quantitative anomalies detected?
   - Has the tone or specific language changed regarding areas showing financial anomalies?

3. Potential Explanations:
   - What legitimate business explanations could account for the observed anomalies?
   - How might recent corporate events explain the unusual patterns?

4. Forward-Looking Implications:
   - How might these anomalies affect future financial performance?
   - What key performance indicators should investors monitor in coming quarters based on these findings?

5. Investigation Recommendations:
   - What specific additional disclosures or information would help clarify these anomalies?
   - Which financial statement areas warrant deeper forensic analysis?
```

## Usage Examples

### Using the Regulatory Analyzer

```python
from tools.regulatory_analysis import RegulatoryAnalyzer

# Initialize the analyzer with your Claude API key
analyzer = RegulatoryAnalyzer(api_key="your_anthropic_api_key")

# Analyze Y-9C bank filings
y9c_results = analyzer.analyze_y9c_filing(
    bank_id=9012,  # Optional: focus on a specific bank
    use_claude=True
)

# Analyze 10-K SEC filings
k10_results = analyzer.analyze_10k_filing(
    ticker="META",  # Optional: focus on a specific company
    use_claude=True
)

# Generate prompts without running Claude
y9c_prompt = analyzer.get_y9c_prompt(bank_id=9012)
k10_prompt = analyzer.get_10k_prompt(ticker="META")

# Run custom regulatory analysis
custom_prompt = """
[Your custom regulatory filing analysis prompt here]
"""
analysis = analyzer.run_claude_analysis(custom_prompt, max_tokens=4000)
```

### Visualization

Both the Y-9C and 10-K analyzers include built-in plotting capabilities:

```python
# Visualize bank metrics with anomaly highlighting
analyzer.y9c_detector.plot_metric_trends(
    data=analyzer.y9c_detector.load_y9c_data(),
    bank_id=9012
)

# Visualize company metrics with anomaly highlighting
analyzer.sec_detector.plot_company_metrics(
    data=analyzer.sec_detector.load_10k_data(),
    ticker="META"
)
```