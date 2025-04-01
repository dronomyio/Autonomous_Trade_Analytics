"""
SEC Filing Anomaly Detector

This module provides tools for detecting anomalies in SEC filings,
particularly 10-K annual reports. It supports both statistical anomaly detection
and Claude-powered analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

# Import Claude if available
try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: Anthropic Claude SDK not available. LLM-based analysis will be disabled.")

class TenKAnomalyDetector:
    """
    Detects anomalies in 10-K SEC filings using statistical methods
    and optional LLM-powered analysis.
    """
    
    def __init__(self, data_dir: str = None, api_key: str = None):
        """
        Initialize the 10-K anomaly detector.
        
        Args:
            data_dir: Path to the regulatory data directory
            api_key: Anthropic API key for Claude (optional)
        """
        if data_dir is None:
            # Use absolute path calculation
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(root_dir, "data", "regulatory", "10k")
        else:
            self.data_dir = data_dir
            
        # Optional Claude client for LLM-based analysis
        self.claude = None
        if CLAUDE_AVAILABLE and api_key:
            self.claude = Anthropic(api_key=api_key)
    
    def load_10k_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load 10-K filing data from CSV.
        
        Args:
            file_path: Path to CSV file (defaults to sample data)
            
        Returns:
            DataFrame with 10-K data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "sample_10k_data.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"10-K data file not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert date columns
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df['fiscal_year_end'] = pd.to_datetime(df['fiscal_year_end'])
        
        return df
    
    def detect_financial_statement_anomalies(self, 
                                            data: pd.DataFrame, 
                                            ticker: Optional[str] = None,
                                            z_threshold: float = 2.5) -> pd.DataFrame:
        """
        Detect anomalies in financial statement metrics.
        
        Args:
            data: DataFrame with 10-K data
            ticker: Filter for specific company ticker (optional)
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        # Filter for specific company if requested
        if ticker is not None:
            data = data[data['ticker'] == ticker].copy()
        
        # Financial metrics to analyze (excluding identifiers, dates, and non-financial indicators)
        non_financial_cols = ['ticker', 'company_name', 'industry', 'fiscal_year_end', 'filing_date',
                             'auditor', 'auditor_changed', 'material_weakness', 'textual_sentiment', 
                             'textual_complexity', 'new_risk_language', 'ceo_changed', 'cfo_changed',
                             'major_acquisition', 'major_divestiture', 'restructuring', 
                             'regulatory_investigation']
        
        metrics = [col for col in data.columns if col not in non_financial_cols]
        
        # Create anomaly detection results
        results = []
        
        # Group by ticker 
        for company, company_data in data.groupby('ticker'):
            # Sort by fiscal year end
            company_data = company_data.sort_values('fiscal_year_end')
            
            # For each metric
            for metric in metrics:
                # Skip metrics that might not apply to all companies (like inventory for services)
                if company_data[metric].isnull().all() or metric == 'market_cap':
                    continue
                
                # Get the values
                values = company_data[metric].values
                
                # Need at least 2 values for z-score
                if len(values) < 2:
                    continue
                
                # Calculate z-scores
                mean = np.mean(values[:-1])  # Mean of all but the latest
                std = np.std(values[:-1])    # Std dev of all but the latest
                
                # Avoid division by zero
                if std == 0 or np.isnan(std):
                    continue
                
                # Calculate z-score for the latest value
                latest_value = values[-1]
                latest_date = company_data['fiscal_year_end'].iloc[-1]
                z_score = (latest_value - mean) / std
                
                # Flag as anomaly if beyond threshold
                is_anomaly = abs(z_score) > z_threshold
                
                if is_anomaly:
                    results.append({
                        'ticker': company,
                        'company_name': company_data['company_name'].iloc[-1],
                        'metric': metric,
                        'fiscal_year_end': latest_date,
                        'value': latest_value,
                        'mean': mean,
                        'std': std,
                        'z_score': z_score,
                        'direction': 'increase' if z_score > 0 else 'decrease',
                        'anomaly_type': 'financial_statement',
                        'is_anomaly': True
                    })
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['ticker', 'company_name', 'metric', 'fiscal_year_end', 
                                      'value', 'mean', 'std', 'z_score', 'direction', 
                                      'anomaly_type', 'is_anomaly'])
    
    def detect_yoy_anomalies(self, 
                           data: pd.DataFrame, 
                           ticker: Optional[str] = None,
                           threshold_pct: Dict[str, float] = None) -> pd.DataFrame:
        """
        Detect anomalies based on year-over-year changes.
        
        Args:
            data: DataFrame with 10-K data
            ticker: Filter for specific company ticker (optional)
            threshold_pct: Dictionary of metric -> percentage change threshold
            
        Returns:
            DataFrame with YoY anomaly flags
        """
        # Default thresholds if not provided
        if threshold_pct is None:
            threshold_pct = {
                'revenue': 20.0,
                'eps': 30.0,
                'operating_margin': 5.0,
                'net_margin': 5.0,
                'debt_to_equity': 30.0,
                'free_cash_flow': 30.0,
                'inventory_turnover': 25.0,
                'accounts_receivable_days': 25.0,
                'interest_coverage': 30.0,
                'default': 25.0  # Default for other metrics
            }
        
        # Filter for specific company if requested
        if ticker is not None:
            data = data[data['ticker'] == ticker].copy()
        
        # Financial metrics to analyze (excluding identifiers, dates, and non-financial indicators)
        non_financial_cols = ['ticker', 'company_name', 'industry', 'fiscal_year_end', 'filing_date',
                             'auditor', 'auditor_changed', 'material_weakness', 'textual_sentiment', 
                             'textual_complexity', 'new_risk_language', 'ceo_changed', 'cfo_changed',
                             'major_acquisition', 'major_divestiture', 'restructuring', 
                             'regulatory_investigation']
        
        metrics = [col for col in data.columns if col not in non_financial_cols]
        
        # Create anomaly detection results
        results = []
        
        # Group by ticker 
        for company, company_data in data.groupby('ticker'):
            # Sort by fiscal year end
            company_data = company_data.sort_values('fiscal_year_end')
            
            # Need at least 2 years for YoY analysis
            if len(company_data) < 2:
                continue
            
            # For each metric
            for metric in metrics:
                # Skip metrics that might not apply to all companies
                if company_data[metric].isnull().all() or metric == 'market_cap':
                    continue
                
                # Create a series with the metric values
                series = company_data[metric]
                
                # Calculate percentage changes
                yoy_changes = series.pct_change() * 100
                
                # Get thresholds
                threshold = threshold_pct.get(metric, threshold_pct['default'])
                
                # Find large changes (loop through all years, not just the latest)
                for i, pct_change in enumerate(yoy_changes):
                    if i == 0 or pd.isna(pct_change):
                        continue
                    
                    if abs(pct_change) >= threshold:
                        results.append({
                            'ticker': company,
                            'company_name': company_data['company_name'].iloc[i],
                            'metric': metric,
                            'fiscal_year_end': company_data['fiscal_year_end'].iloc[i],
                            'previous_year_end': company_data['fiscal_year_end'].iloc[i-1],
                            'current_value': series.iloc[i],
                            'previous_value': series.iloc[i-1],
                            'pct_change': pct_change,
                            'direction': 'increase' if pct_change > 0 else 'decrease',
                            'anomaly_type': 'year_over_year',
                            'is_anomaly': True
                        })
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['ticker', 'company_name', 'metric', 'fiscal_year_end', 
                                      'previous_year_end', 'current_value', 'previous_value', 
                                      'pct_change', 'direction', 'anomaly_type', 'is_anomaly'])
    
    def detect_non_financial_anomalies(self, data: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Detect anomalies in non-financial indicators.
        
        Args:
            data: DataFrame with 10-K data
            ticker: Filter for specific company ticker (optional)
            
        Returns:
            DataFrame with non-financial anomaly flags
        """
        # Filter for specific company if requested
        if ticker is not None:
            data = data[data['ticker'] == ticker].copy()
        
        # Non-financial indicators to analyze
        indicators = ['auditor_changed', 'material_weakness', 'textual_sentiment', 
                     'textual_complexity', 'new_risk_language', 'ceo_changed', 'cfo_changed',
                     'major_acquisition', 'major_divestiture', 'restructuring', 
                     'regulatory_investigation']
        
        # Create anomaly detection results
        results = []
        
        # Group by ticker 
        for company, company_data in data.groupby('ticker'):
            # Sort by fiscal year end
            company_data = company_data.sort_values('fiscal_year_end')
            
            # For each indicator
            for indicator in indicators:
                # Skip indicators that don't exist
                if indicator not in company_data.columns:
                    continue
                
                # Get the latest value
                latest_value = company_data[indicator].iloc[-1]
                
                # Check different types of anomalies based on indicator type
                is_anomaly = False
                reason = ""
                
                if indicator in ['auditor_changed', 'material_weakness', 'new_risk_language', 
                               'ceo_changed', 'cfo_changed', 'regulatory_investigation']:
                    # Binary indicators - anomaly if True (1)
                    is_anomaly = bool(latest_value)
                    if is_anomaly:
                        reason = f"{indicator.replace('_', ' ').title()} detected"
                        
                elif indicator in ['major_acquisition', 'major_divestiture', 'restructuring']:
                    # Business change indicators - anomaly if True (1)
                    is_anomaly = bool(latest_value)
                    if is_anomaly:
                        reason = f"{indicator.replace('_', ' ').title()} occurred"
                        
                elif indicator == 'textual_sentiment':
                    # Sentiment - anomaly if significant decrease
                    sentiment_values = company_data[indicator].values
                    if len(sentiment_values) > 1:
                        change = sentiment_values[-1] - np.mean(sentiment_values[:-1])
                        if change < -0.1:  # Sentiment decreased by more than 0.1
                            is_anomaly = True
                            reason = f"Sentiment decreased significantly by {abs(change):.2f}"
                            
                elif indicator == 'textual_complexity':
                    # Complexity - anomaly if significant increase
                    complexity_values = company_data[indicator].values
                    if len(complexity_values) > 1:
                        change = complexity_values[-1] - np.mean(complexity_values[:-1])
                        if change > 0.05:  # Complexity increased by more than 0.05
                            is_anomaly = True
                            reason = f"Textual complexity increased significantly by {change:.2f}"
                
                if is_anomaly:
                    results.append({
                        'ticker': company,
                        'company_name': company_data['company_name'].iloc[-1],
                        'indicator': indicator,
                        'fiscal_year_end': company_data['fiscal_year_end'].iloc[-1],
                        'value': latest_value,
                        'reason': reason,
                        'anomaly_type': 'non_financial',
                        'is_anomaly': True
                    })
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['ticker', 'company_name', 'indicator', 'fiscal_year_end', 
                                      'value', 'reason', 'anomaly_type', 'is_anomaly'])
    
    def generate_10k_anomaly_prompt(self, 
                                  anomalies_df: pd.DataFrame, 
                                  company_data: pd.DataFrame,
                                  ticker: Optional[str] = None) -> str:
        """
        Generate a prompt for Claude to analyze 10-K anomalies.
        
        Args:
            anomalies_df: DataFrame with detected anomalies
            company_data: Complete 10-K data
            ticker: Specific company ticker to focus on (optional)
            
        Returns:
            String prompt for Claude
        """
        # Filter data for specific company if requested
        if ticker is not None:
            company_data = company_data[company_data['ticker'] == ticker].copy()
            anomalies_df = anomalies_df[anomalies_df['ticker'] == ticker].copy()
        
        # Check if we have any anomalies
        if len(anomalies_df) == 0:
            return "No anomalies detected in the provided data."
        
        # Sort anomalies by company and fiscal year end
        if 'fiscal_year_end' in anomalies_df.columns:
            anomalies_df = anomalies_df.sort_values(['ticker', 'fiscal_year_end'])
        
        # Create a dictionary of ticker -> list of anomalies
        anomalies_by_company = {}
        
        # Group by ticker and anomaly type
        for (ticker, anom_type), group in anomalies_df.groupby(['ticker', 'anomaly_type']):
            if ticker not in anomalies_by_company:
                anomalies_by_company[ticker] = {}
            
            anomalies_by_company[ticker][anom_type] = group.to_dict('records')
        
        # Build the prompt using the detailed 10-K prompt structure
        prompt = """# Corporate Financial Anomaly Analysis: 10-K Filing Review

"""
        
        # Add company-specific sections
        for ticker, anomaly_types in anomalies_by_company.items():
            # Get company info
            company_info = company_data[company_data['ticker'] == ticker].sort_values('fiscal_year_end').iloc[-1]
            
            prompt += f"""## COMPANY CONTEXT
Company: {company_info['company_name']} ({ticker})
Industry: {company_info['industry']}
Market Cap: ${company_info['market_cap']:,.0f}
Fiscal Year End: {company_info['fiscal_year_end'].strftime('%Y-%m-%d')}
Filing Date: {company_info['filing_date'].strftime('%Y-%m-%d')}

## DETECTED ANOMALIES

"""
            
            # Add financial statement anomalies if present
            if 'financial_statement' in anomaly_types:
                prompt += "### Financial Statement Anomalies\n\n"
                
                # Group by statement type
                income_statement_metrics = ['revenue', 'revenue_growth', 'gross_margin', 'operating_margin', 
                                         'net_margin', 'eps', 'eps_growth', 'tax_rate']
                
                balance_sheet_metrics = ['debt_to_equity', 'asset_turnover', 'inventory_turnover', 
                                      'accounts_receivable_days', 'goodwill_to_assets', 'pe_ratio']
                
                cash_flow_metrics = ['free_cash_flow', 'capex_to_revenue', 'interest_coverage']
                
                # Add income statement anomalies
                has_income_anomalies = False
                for anomaly in anomaly_types['financial_statement']:
                    if anomaly['metric'] in income_statement_metrics:
                        if not has_income_anomalies:
                            prompt += "#### Income Statement Anomalies\n"
                            has_income_anomalies = True
                        
                        prompt += f"- {anomaly['metric']}: {anomaly['value']:.2f} (Z-score: {anomaly['z_score']:.2f})\n"
                        prompt += f"  - Historical average: {anomaly['mean']:.2f}\n"
                        prompt += f"  - Direction: {anomaly['direction']}\n"
                        
                        # Add YoY change if available
                        if 'year_over_year' in anomaly_types:
                            for yoy in anomaly_types['year_over_year']:
                                if yoy['metric'] == anomaly['metric'] and yoy['fiscal_year_end'] == anomaly['fiscal_year_end']:
                                    prompt += f"  - YoY change: {yoy['pct_change']:.2f}% {yoy['direction']}\n"
                        
                        prompt += "\n"
                
                # Add balance sheet anomalies
                has_balance_anomalies = False
                for anomaly in anomaly_types['financial_statement']:
                    if anomaly['metric'] in balance_sheet_metrics:
                        if not has_balance_anomalies:
                            prompt += "\n#### Balance Sheet Anomalies\n"
                            has_balance_anomalies = True
                        
                        prompt += f"- {anomaly['metric']}: {anomaly['value']:.2f} (Z-score: {anomaly['z_score']:.2f})\n"
                        prompt += f"  - Historical average: {anomaly['mean']:.2f}\n"
                        prompt += f"  - Direction: {anomaly['direction']}\n"
                        
                        # Add YoY change if available
                        if 'year_over_year' in anomaly_types:
                            for yoy in anomaly_types['year_over_year']:
                                if yoy['metric'] == anomaly['metric'] and yoy['fiscal_year_end'] == anomaly['fiscal_year_end']:
                                    prompt += f"  - YoY change: {yoy['pct_change']:.2f}% {yoy['direction']}\n"
                        
                        prompt += "\n"
                
                # Add cash flow anomalies
                has_cash_flow_anomalies = False
                for anomaly in anomaly_types['financial_statement']:
                    if anomaly['metric'] in cash_flow_metrics:
                        if not has_cash_flow_anomalies:
                            prompt += "\n#### Cash Flow Statement Anomalies\n"
                            has_cash_flow_anomalies = True
                        
                        prompt += f"- {anomaly['metric']}: {anomaly['value']:.2f} (Z-score: {anomaly['z_score']:.2f})\n"
                        prompt += f"  - Historical average: {anomaly['mean']:.2f}\n"
                        prompt += f"  - Direction: {anomaly['direction']}\n"
                        
                        # Add YoY change if available
                        if 'year_over_year' in anomaly_types:
                            for yoy in anomaly_types['year_over_year']:
                                if yoy['metric'] == anomaly['metric'] and yoy['fiscal_year_end'] == anomaly['fiscal_year_end']:
                                    prompt += f"  - YoY change: {yoy['pct_change']:.2f}% {yoy['direction']}\n"
                        
                        prompt += "\n"
            
            # Add YoY anomalies that weren't already covered
            if 'year_over_year' in anomaly_types:
                yoy_already_shown = set()
                
                if 'financial_statement' in anomaly_types:
                    for fs_anomaly in anomaly_types['financial_statement']:
                        for yoy_anomaly in anomaly_types['year_over_year']:
                            if (fs_anomaly['metric'] == yoy_anomaly['metric'] and 
                                fs_anomaly['fiscal_year_end'] == yoy_anomaly['fiscal_year_end']):
                                yoy_already_shown.add((yoy_anomaly['metric'], yoy_anomaly['fiscal_year_end']))
                
                remaining_yoy = [a for a in anomaly_types['year_over_year'] 
                               if (a['metric'], a['fiscal_year_end']) not in yoy_already_shown]
                
                if remaining_yoy:
                    prompt += "\n### Additional Year-over-Year Changes\n\n"
                    
                    for anomaly in remaining_yoy:
                        prompt += f"- {anomaly['metric']}: {anomaly['pct_change']:.2f}% {anomaly['direction']}\n"
                        prompt += f"  - Changed from {anomaly['previous_value']:.2f} to {anomaly['current_value']:.2f}\n"
                        prompt += f"  - Period: {anomaly['previous_year_end'].strftime('%Y-%m-%d')} to {anomaly['fiscal_year_end'].strftime('%Y-%m-%d')}\n\n"
            
            # Add non-financial anomalies
            if 'non_financial' in anomaly_types:
                prompt += "\n### Governance & Disclosure Anomalies\n\n"
                
                # Group by categories
                auditing = []
                management = []
                business = []
                textual = []
                
                for anomaly in anomaly_types['non_financial']:
                    if anomaly['indicator'] in ['auditor_changed', 'material_weakness']:
                        auditing.append(anomaly)
                    elif anomaly['indicator'] in ['ceo_changed', 'cfo_changed']:
                        management.append(anomaly)
                    elif anomaly['indicator'] in ['major_acquisition', 'major_divestiture', 
                                                'restructuring', 'regulatory_investigation']:
                        business.append(anomaly)
                    elif anomaly['indicator'] in ['textual_sentiment', 'textual_complexity', 'new_risk_language']:
                        textual.append(anomaly)
                
                if auditing:
                    prompt += "#### Audit & Control Issues\n"
                    for a in auditing:
                        prompt += f"- {a['reason']}\n"
                    prompt += "\n"
                
                if management:
                    prompt += "#### Management Changes\n"
                    for a in management:
                        prompt += f"- {a['reason']}\n"
                    prompt += "\n"
                
                if business:
                    prompt += "#### Business Events\n"
                    for a in business:
                        prompt += f"- {a['reason']}\n"
                    prompt += "\n"
                
                if textual:
                    prompt += "#### Disclosure Analysis\n"
                    for a in textual:
                        prompt += f"- {a['reason']}\n"
                    prompt += "\n"
            
            # Add context
            prompt += f"""
## CORPORATE EVENT CONTEXT
Recent significant corporate events:
"""
            
            # Add events from company data
            events = []
            if company_info['ceo_changed']:
                events.append("- CEO change")
            if company_info['cfo_changed']:
                events.append("- CFO change")
            if company_info['major_acquisition']:
                events.append("- Major acquisition")
            if company_info['major_divestiture']:
                events.append("- Major divestiture")
            if company_info['restructuring']:
                events.append("- Corporate restructuring")
            if company_info['regulatory_investigation']:
                events.append("- Regulatory investigation")
            if company_info['auditor_changed']:
                events.append("- Auditor change")
            
            if events:
                prompt += "\n".join(events) + "\n"
            else:
                prompt += "- No major corporate events reported in the filing\n"
            
            # Industry context (this would need to be enhanced with actual industry data)
            prompt += f"""
## INDUSTRY/ECONOMIC CONTEXT
- Industry: {company_info['industry']}
- Relevant economic factors for this industry and reporting period
"""
            
            # Break after one company if we're looking at multiple
            if len(anomalies_by_company) > 1 and ticker != list(anomalies_by_company.keys())[-1]:
                prompt += "\n---\n\n"
        
        # Add analysis questions
        prompt += """
## ANALYSIS QUESTIONS

1. Significance Assessment:
   - Which anomalies represent the most significant potential risks to the company's financial health or valuation?
   - Which anomalies might indicate potential accounting irregularities versus legitimate business changes?
   - Are there any red flags suggesting potential revenue recognition issues, expense manipulations, or other accounting concerns?

2. Narrative vs. Numbers Analysis:
   - Are there inconsistencies between the company's narrative explanation (if available) and the quantitative anomalies detected?
   - Has the tone or specific language changed regarding areas showing financial anomalies?
   - Are risk disclosures adequate for the anomalies identified, or are potential risks being understated?

3. Potential Explanations:
   - What legitimate business explanations could account for the observed anomalies?
   - How might recent corporate events explain the unusual patterns?
   - Are industry-wide factors likely contributing to these anomalies?

4. Forward-Looking Implications:
   - How might these anomalies affect future financial performance?
   - What key performance indicators should investors monitor in coming quarters based on these findings?
   - Are there early warning signs of potential goodwill impairments, asset write-downs, or other future negative events?

5. Investigation Recommendations:
   - What specific additional disclosures or information would help clarify these anomalies?
   - Which financial statement areas warrant deeper forensic analysis?
   - What questions should analysts or auditors ask management regarding these findings?

Please provide a comprehensive analysis of these anomalies, drawing on your understanding of financial reporting, accounting principles, securities regulations, and industry dynamics. Focus on substantive issues that could affect investment decisions, regulatory compliance, or audit procedures. Distinguish between technical accounting anomalies and those that may signal fundamental business challenges.
"""
        
        return prompt
    
    def analyze_with_claude(self, 
                          prompt: str, 
                          max_tokens: int = 4000,
                          temperature: float = 0.0) -> str:
        """
        Analyze anomalies using Claude LLM.
        
        Args:
            prompt: Prompt with anomaly data
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            
        Returns:
            Claude's analysis
        """
        if not CLAUDE_AVAILABLE or self.claude is None:
            return "Claude analysis not available. Please ensure the Anthropic Python SDK is installed and API key is provided."
        
        try:
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error using Claude API: {str(e)}"
    
    def plot_company_metrics(self, 
                           data: pd.DataFrame, 
                           ticker: str,
                           metrics: Optional[List[str]] = None) -> None:
        """
        Plot trends for key metrics with anomaly highlighting.
        
        Args:
            data: DataFrame with 10-K data
            ticker: Company ticker to analyze
            metrics: List of metrics to plot (defaults to a standard set)
        """
        # Filter data for the specific company
        company_data = data[data['ticker'] == ticker].copy()
        
        if len(company_data) == 0:
            print(f"No data found for {ticker}")
            return
        
        # Sort by date
        company_data = company_data.sort_values('fiscal_year_end')
        
        # Define metrics to plot if not provided
        if metrics is None:
            metrics = [
                'revenue_growth', 'operating_margin', 'net_margin', 
                'eps_growth', 'debt_to_equity', 'free_cash_flow',
                'inventory_turnover', 'accounts_receivable_days'
            ]
            
            # Remove metrics that don't exist for this company
            metrics = [m for m in metrics if m in company_data.columns and not company_data[m].isnull().all()]
        
        # Detect anomalies for highlighting
        financial_anomalies = self.detect_financial_statement_anomalies(data, ticker)
        yoy_anomalies = self.detect_yoy_anomalies(data, ticker)
        
        # Create a dictionary of metric -> anomaly points
        anomaly_points = {}
        for metric in metrics:
            anomaly_points[metric] = {
                'financial': {
                    'dates': financial_anomalies[financial_anomalies['metric'] == metric]['fiscal_year_end'].tolist(),
                    'values': financial_anomalies[financial_anomalies['metric'] == metric]['value'].tolist()
                },
                'yoy': {
                    'dates': yoy_anomalies[yoy_anomalies['metric'] == metric]['fiscal_year_end'].tolist(),
                    'values': yoy_anomalies[yoy_anomalies['metric'] == metric]['current_value'].tolist()
                }
            }
        
        # Create plots
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // n_cols
        
        plt.figure(figsize=(15, n_rows * 4))
        
        for i, metric in enumerate(metrics):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Plot trend line
            plt.plot(company_data['fiscal_year_end'], company_data[metric], 'b-', label=metric)
            
            # Plot anomaly points
            for anom_type, points in anomaly_points[metric].items():
                if points['dates']:
                    marker = 'ro' if anom_type == 'financial' else 'mo'
                    plt.plot(points['dates'], points['values'], marker, label=f"{anom_type} anomaly")
            
            plt.title(f"{metric} for {ticker}")
            plt.xlabel("Fiscal Year End")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def full_10k_analysis(self, 
                        file_path: Optional[str] = None,
                        ticker: Optional[str] = None,
                        use_claude: bool = False,
                        save_prompt: bool = False,
                        save_plot: bool = False) -> Dict:
        """
        Perform a full 10-K anomaly analysis workflow.
        
        Args:
            file_path: Path to 10-K data file
            ticker: Specific company ticker to analyze (optional)
            use_claude: Whether to use Claude for analysis
            save_prompt: Whether to save the prompt to a file
            save_plot: Whether to save plots to files
            
        Returns:
            Dictionary with analysis results
        """
        # Load data
        data = self.load_10k_data(file_path)
        
        # If ticker is not specified but there are multiple companies, provide a summary
        if ticker is None and len(data['ticker'].unique()) > 1:
            print(f"Data contains {len(data['ticker'].unique())} companies:")
            for t in data['ticker'].unique():
                company_name = data[data['ticker'] == t]['company_name'].iloc[0]
                print(f"- {t}: {company_name}")
            
            if not use_claude:
                # If not using Claude, pick the first company with anomalies
                financial_anomalies = self.detect_financial_statement_anomalies(data)
                if len(financial_anomalies) > 0:
                    ticker = financial_anomalies['ticker'].iloc[0]
                    print(f"Analyzing {ticker} which has detected anomalies")
                else:
                    ticker = data['ticker'].iloc[0]
                    print(f"No anomalies detected. Analyzing first company ({ticker})")
        
        # Detect anomalies
        financial_anomalies = self.detect_financial_statement_anomalies(data, ticker)
        yoy_anomalies = self.detect_yoy_anomalies(data, ticker)
        non_financial_anomalies = self.detect_non_financial_anomalies(data, ticker)
        
        # Combine all anomalies
        all_anomalies = pd.concat([financial_anomalies, yoy_anomalies, non_financial_anomalies], 
                                ignore_index=True)
        
        # Generate prompt
        prompt = self.generate_10k_anomaly_prompt(all_anomalies, data, ticker)
        
        # Save prompt if requested
        if save_prompt:
            prompt_file = os.path.join(self.data_dir, f"10k_anomaly_prompt{'_' + ticker if ticker else ''}.txt")
            with open(prompt_file, 'w') as f:
                f.write(prompt)
            print(f"Prompt saved to {prompt_file}")
        
        # Use Claude if requested
        claude_analysis = None
        if use_claude and self.claude is not None:
            print("Analyzing anomalies with Claude...")
            claude_analysis = self.analyze_with_claude(prompt)
        
        # Generate plots if a specific company is selected
        if ticker is not None:
            self.plot_company_metrics(data, ticker)
        
        # Return results
        return {
            'financial_anomalies': financial_anomalies,
            'yoy_anomalies': yoy_anomalies,
            'non_financial_anomalies': non_financial_anomalies,
            'prompt': prompt,
            'claude_analysis': claude_analysis
        }


# Example usage
if __name__ == "__main__":
    # Get API key from environment variable if available
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    # Initialize detector
    detector = TenKAnomalyDetector(api_key=api_key)
    
    try:
        # Load sample data
        print("Loading 10-K sample data...")
        data = detector.load_10k_data()
        print(f"Loaded data for {len(data['ticker'].unique())} companies")
        
        # Simple analysis example - run financial anomaly detection
        print("\nRunning financial statement anomaly detection...")
        anomalies = detector.detect_financial_statement_anomalies(data)
        
        if len(anomalies) > 0:
            print(f"Found {len(anomalies)} financial statement anomalies")
            print("\nTop 5 financial anomalies:")
            print(anomalies.sort_values('z_score', ascending=False).head())
        else:
            print("No financial statement anomalies detected")
        
        # Run YoY anomaly detection
        print("\nRunning year-over-year anomaly detection...")
        yoy_anomalies = detector.detect_yoy_anomalies(data)
        
        if len(yoy_anomalies) > 0:
            print(f"Found {len(yoy_anomalies)} year-over-year anomalies")
            print("\nTop 5 YoY anomalies:")
            print(yoy_anomalies.sort_values('pct_change', ascending=False).head())
        else:
            print("No year-over-year anomalies detected")
        
        # Run non-financial anomaly detection
        print("\nRunning non-financial anomaly detection...")
        non_financial_anomalies = detector.detect_non_financial_anomalies(data)
        
        if len(non_financial_anomalies) > 0:
            print(f"Found {len(non_financial_anomalies)} non-financial anomalies")
            print("\nNon-financial anomalies:")
            print(non_financial_anomalies[['ticker', 'indicator', 'reason']].head())
        else:
            print("No non-financial anomalies detected")
        
        # Pick an interesting company for demo
        example_ticker = 'META'  # Meta had significant anomalies in our sample data
        
        # Generate prompt
        print(f"\nGenerating Claude prompt for {example_ticker}...")
        all_anomalies = pd.concat([
            anomalies[anomalies['ticker'] == example_ticker], 
            yoy_anomalies[yoy_anomalies['ticker'] == example_ticker],
            non_financial_anomalies[non_financial_anomalies['ticker'] == example_ticker]
        ], ignore_index=True)
        
        prompt = detector.generate_10k_anomaly_prompt(all_anomalies, data, example_ticker)
        
        print("\nPrompt preview:")
        print(prompt[:500] + "...\n[Truncated]")
        
        # Use Claude if available
        if CLAUDE_AVAILABLE and api_key:
            print("\nAnalyzing with Claude (if API key is provided)...")
            analysis = detector.analyze_with_claude(prompt)
            print("\nClaude Analysis preview:")
            print(analysis[:500] + "...\n[Truncated]")
        else:
            print("\nSkipping Claude analysis (SDK or API key not available)")
        
        # Plot example for a specific company
        print(f"\nGenerating plots for {example_ticker}...")
        detector.plot_company_metrics(data, example_ticker)
        
    except Exception as e:
        print(f"Error in 10-K anomaly detector example: {str(e)}")