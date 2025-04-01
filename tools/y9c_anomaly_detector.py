"""
Y-9C Bank Filing Anomaly Detector

This module provides tools for detecting anomalies in FR Y-9C bank regulatory filings.
It supports both statistical anomaly detection and Claude-powered analysis.
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

class Y9CAnomalyDetector:
    """
    Detects anomalies in Y-9C bank regulatory filings using statistical methods
    and optional LLM-powered analysis.
    """
    
    def __init__(self, data_dir: str = None, api_key: str = None):
        """
        Initialize the Y-9C anomaly detector.
        
        Args:
            data_dir: Path to the regulatory data directory
            api_key: Anthropic API key for Claude (optional)
        """
        if data_dir is None:
            # Use absolute path calculation
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(root_dir, "data", "regulatory", "y9c")
        else:
            self.data_dir = data_dir
            
        # Optional Claude client for LLM-based analysis
        self.claude = None
        if CLAUDE_AVAILABLE and api_key:
            self.claude = Anthropic(api_key=api_key)
    
    def load_y9c_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Y-9C filing data from CSV.
        
        Args:
            file_path: Path to CSV file (defaults to sample data)
            
        Returns:
            DataFrame with Y-9C data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "sample_y9c_data.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Y-9C data file not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert date columns
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        
        return df
    
    def detect_statistical_anomalies(self, 
                                    data: pd.DataFrame, 
                                    bank_id: Optional[int] = None,
                                    z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using z-score method on bank metrics.
        
        Args:
            data: DataFrame with Y-9C data
            bank_id: Filter for specific bank ID (optional)
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        # Filter for specific bank if requested
        if bank_id is not None:
            data = data[data['bank_id'] == bank_id].copy()
        
        # Metrics to analyze (excluding identifiers and dates)
        metrics = [col for col in data.columns if col not in ['bank_id', 'filing_date']]
        
        # Create anomaly detection results
        results = []
        
        # Group by bank_id 
        for bank, bank_data in data.groupby('bank_id'):
            # Sort by filing date
            bank_data = bank_data.sort_values('filing_date')
            
            # For each metric
            for metric in metrics:
                # Get the values
                values = bank_data[metric].values
                
                # Need at least 2 values for z-score
                if len(values) < 2:
                    continue
                
                # Calculate z-scores
                mean = np.mean(values[:-1])  # Mean of all but the latest
                std = np.std(values[:-1])    # Std dev of all but the latest
                
                # Avoid division by zero
                if std == 0:
                    continue
                
                # Calculate z-score for the latest value
                latest_value = values[-1]
                latest_date = bank_data['filing_date'].iloc[-1]
                z_score = (latest_value - mean) / std
                
                # Flag as anomaly if beyond threshold
                is_anomaly = abs(z_score) > z_threshold
                
                if is_anomaly:
                    results.append({
                        'bank_id': bank,
                        'metric': metric,
                        'filing_date': latest_date,
                        'value': latest_value,
                        'mean': mean,
                        'std': std,
                        'z_score': z_score,
                        'direction': 'increase' if z_score > 0 else 'decrease',
                        'is_anomaly': True
                    })
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['bank_id', 'metric', 'filing_date', 'value', 
                                      'mean', 'std', 'z_score', 'direction', 'is_anomaly'])
    
    def detect_sequential_anomalies(self, 
                                  data: pd.DataFrame, 
                                  bank_id: Optional[int] = None,
                                  threshold_pct: float = 25.0) -> pd.DataFrame:
        """
        Detect anomalies based on quarter-to-quarter changes.
        
        Args:
            data: DataFrame with Y-9C data
            bank_id: Filter for specific bank ID (optional)
            threshold_pct: Percentage change threshold for anomaly detection
            
        Returns:
            DataFrame with sequential anomaly flags
        """
        # Filter for specific bank if requested
        if bank_id is not None:
            data = data[data['bank_id'] == bank_id].copy()
        
        # Metrics to analyze (excluding identifiers and dates)
        metrics = [col for col in data.columns if col not in ['bank_id', 'filing_date']]
        
        # Create anomaly detection results
        results = []
        
        # Group by bank_id 
        for bank, bank_data in data.groupby('bank_id'):
            # Sort by filing date
            bank_data = bank_data.sort_values('filing_date')
            
            # Need at least 2 quarters for sequential analysis
            if len(bank_data) < 2:
                continue
            
            # For each metric
            for metric in metrics:
                # Create a series with the metric values
                series = bank_data[metric]
                
                # Calculate percentage changes
                pct_changes = series.pct_change() * 100
                
                # Find large changes
                for i, pct_change in enumerate(pct_changes):
                    if i == 0 or pd.isna(pct_change):
                        continue
                    
                    if abs(pct_change) >= threshold_pct:
                        results.append({
                            'bank_id': bank,
                            'metric': metric,
                            'filing_date': bank_data['filing_date'].iloc[i],
                            'previous_date': bank_data['filing_date'].iloc[i-1],
                            'current_value': series.iloc[i],
                            'previous_value': series.iloc[i-1],
                            'pct_change': pct_change,
                            'direction': 'increase' if pct_change > 0 else 'decrease',
                            'is_anomaly': True
                        })
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['bank_id', 'metric', 'filing_date', 'previous_date',
                                      'current_value', 'previous_value', 'pct_change', 
                                      'direction', 'is_anomaly'])
    
    def generate_anomaly_prompt(self, 
                               anomalies_df: pd.DataFrame, 
                               bank_data: pd.DataFrame,
                               bank_id: Optional[int] = None) -> str:
        """
        Generate a prompt for Claude to analyze anomalies.
        
        Args:
            anomalies_df: DataFrame with detected anomalies
            bank_data: Complete bank Y-9C data
            bank_id: Specific bank ID to focus on (optional)
            
        Returns:
            String prompt for Claude
        """
        # Filter data for specific bank if requested
        if bank_id is not None:
            bank_data = bank_data[bank_data['bank_id'] == bank_id].copy()
            anomalies_df = anomalies_df[anomalies_df['bank_id'] == bank_id].copy()
        
        # Check if we have any anomalies
        if len(anomalies_df) == 0:
            return "No anomalies detected in the provided data."
        
        # Sort anomalies by bank and filing date
        anomalies_df = anomalies_df.sort_values(['bank_id', 'filing_date'])
        
        # Create a dictionary of bank_id -> list of anomalies
        anomalies_by_bank = {}
        for bank_id, group in anomalies_df.groupby('bank_id'):
            anomalies_by_bank[bank_id] = group.to_dict('records')
        
        # Build the prompt
        prompt = """I'm analyzing anomalies in FR Y-9C bank regulatory filings. I've detected the following unusual patterns in the data:

ANOMALY SUMMARY:
"""
        
        # Add anomalies by bank
        for bank_id, anomalies in anomalies_by_bank.items():
            bank_filings = bank_data[bank_data['bank_id'] == bank_id].sort_values('filing_date')
            latest_assets = bank_filings['total_assets'].iloc[-1]
            
            prompt += f"\nBank ID: {bank_id} (Total Assets: ${latest_assets:,.0f})\n"
            
            for anomaly in anomalies:
                if 'z_score' in anomaly:
                    # Statistical anomaly
                    prompt += f"- {anomaly['metric']}: {anomaly['value']:.2f} "
                    prompt += f"(Z-score: {anomaly['z_score']:.2f}, unusual {anomaly['direction']})\n"
                elif 'pct_change' in anomaly:
                    # Sequential anomaly
                    prompt += f"- {anomaly['metric']}: {anomaly['pct_change']:.2f}% "
                    prompt += f"{anomaly['direction']} from {anomaly['previous_value']:.2f} to {anomaly['current_value']:.2f}\n"
        
        # Add context about metrics
        prompt += """
METRICS EXPLANATION:
- total_assets: Total assets on the bank's balance sheet
- tier1_capital_ratio: Tier 1 capital as a percentage of risk-weighted assets
- total_capital_ratio: Total capital as a percentage of risk-weighted assets
- leverage_ratio: Tier 1 capital as a percentage of total assets
- npl_ratio: Non-performing loans as a percentage of total loans
- roa: Return on assets (annualized)
- roe: Return on equity (annualized)
- net_interest_margin: Net interest income as a percentage of average earning assets
- efficiency_ratio: Non-interest expenses divided by revenue (lower is better)
- liquid_assets_ratio: Liquid assets as a percentage of total assets
- loan_to_deposit_ratio: Total loans divided by total deposits
- trading_assets_ratio: Trading assets as a percentage of total assets

QUESTIONS:
1. What are the most concerning anomalies in this data and why?
2. What potential explanations could there be for these anomalies?
3. What additional information would be helpful to investigate these anomalies further?
4. What recommendations would you make to risk managers reviewing these results?
5. Are there any regulatory implications of these anomalies that should be considered?

Please analyze these anomalies in detail, considering banking industry trends, regulatory concerns, and potential business implications.
"""
        
        return prompt
    
    def analyze_with_claude(self, 
                          prompt: str, 
                          max_tokens: int = 2000,
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
    
    def plot_metric_trends(self, 
                         data: pd.DataFrame, 
                         bank_id: int,
                         metrics: Optional[List[str]] = None) -> None:
        """
        Plot trends for key metrics with anomaly highlighting.
        
        Args:
            data: DataFrame with Y-9C data
            bank_id: Bank ID to analyze
            metrics: List of metrics to plot (defaults to a standard set)
        """
        # Filter data for the specific bank
        bank_data = data[data['bank_id'] == bank_id].copy()
        
        if len(bank_data) == 0:
            print(f"No data found for bank ID {bank_id}")
            return
        
        # Sort by date
        bank_data = bank_data.sort_values('filing_date')
        
        # Define metrics to plot if not provided
        if metrics is None:
            metrics = [
                'tier1_capital_ratio', 'leverage_ratio', 'npl_ratio',
                'roa', 'net_interest_margin', 'efficiency_ratio',
                'loan_to_deposit_ratio'
            ]
        
        # Detect anomalies for highlighting
        statistical_anomalies = self.detect_statistical_anomalies(data, bank_id)
        sequential_anomalies = self.detect_sequential_anomalies(data, bank_id)
        
        # Create a dictionary of metric -> anomaly points
        anomaly_points = {}
        for metric in metrics:
            anomaly_points[metric] = {
                'statistical': {
                    'dates': statistical_anomalies[statistical_anomalies['metric'] == metric]['filing_date'].tolist(),
                    'values': statistical_anomalies[statistical_anomalies['metric'] == metric]['value'].tolist()
                },
                'sequential': {
                    'dates': sequential_anomalies[sequential_anomalies['metric'] == metric]['filing_date'].tolist(),
                    'values': sequential_anomalies[sequential_anomalies['metric'] == metric]['current_value'].tolist()
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
            plt.plot(bank_data['filing_date'], bank_data[metric], 'b-', label=metric)
            
            # Plot anomaly points
            for anom_type, points in anomaly_points[metric].items():
                if points['dates']:
                    marker = 'ro' if anom_type == 'statistical' else 'mo'
                    plt.plot(points['dates'], points['values'], marker, label=f"{anom_type} anomaly")
            
            plt.title(f"{metric} for Bank {bank_id}")
            plt.xlabel("Filing Date")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def full_anomaly_analysis(self, 
                           file_path: Optional[str] = None,
                           bank_id: Optional[int] = None,
                           use_claude: bool = False,
                           save_prompt: bool = False,
                           save_plot: bool = False) -> Dict:
        """
        Perform a full anomaly analysis workflow.
        
        Args:
            file_path: Path to Y-9C data file
            bank_id: Specific bank ID to analyze (optional)
            use_claude: Whether to use Claude for analysis
            save_prompt: Whether to save the prompt to a file
            save_plot: Whether to save plots to files
            
        Returns:
            Dictionary with analysis results
        """
        # Load data
        data = self.load_y9c_data(file_path)
        
        # If bank_id is not specified but there are multiple banks, provide a summary
        if bank_id is None and len(data['bank_id'].unique()) > 1:
            print(f"Data contains {len(data['bank_id'].unique())} banks. Bank IDs: {data['bank_id'].unique().tolist()}")
            
            if not use_claude:
                # If not using Claude, pick the first bank with anomalies
                statistical_anomalies = self.detect_statistical_anomalies(data)
                if len(statistical_anomalies) > 0:
                    bank_id = statistical_anomalies['bank_id'].iloc[0]
                    print(f"Analyzing bank ID {bank_id} which has detected anomalies")
                else:
                    bank_id = data['bank_id'].iloc[0]
                    print(f"No anomalies detected. Analyzing first bank (ID: {bank_id})")
        
        # Detect anomalies
        statistical_anomalies = self.detect_statistical_anomalies(data, bank_id)
        sequential_anomalies = self.detect_sequential_anomalies(data, bank_id)
        
        # Generate prompt
        all_anomalies = pd.concat([statistical_anomalies, sequential_anomalies], ignore_index=True)
        prompt = self.generate_anomaly_prompt(all_anomalies, data, bank_id)
        
        # Save prompt if requested
        if save_prompt:
            prompt_file = os.path.join(self.data_dir, 'anomaly_prompt.txt')
            with open(prompt_file, 'w') as f:
                f.write(prompt)
            print(f"Prompt saved to {prompt_file}")
        
        # Use Claude if requested
        claude_analysis = None
        if use_claude and self.claude is not None:
            print("Analyzing anomalies with Claude...")
            claude_analysis = self.analyze_with_claude(prompt)
        
        # Generate plots if a specific bank is selected
        if bank_id is not None:
            self.plot_metric_trends(data, bank_id)
        
        # Return results
        return {
            'statistical_anomalies': statistical_anomalies,
            'sequential_anomalies': sequential_anomalies,
            'prompt': prompt,
            'claude_analysis': claude_analysis
        }


# Example usage
if __name__ == "__main__":
    # Get API key from environment variable if available
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    # Initialize detector
    detector = Y9CAnomalyDetector(api_key=api_key)
    
    try:
        # Load sample data
        print("Loading Y-9C sample data...")
        data = detector.load_y9c_data()
        print(f"Loaded data for {len(data['bank_id'].unique())} banks")
        
        # Simple analysis example - run statistical anomaly detection
        print("\nRunning statistical anomaly detection...")
        anomalies = detector.detect_statistical_anomalies(data)
        
        if len(anomalies) > 0:
            print(f"Found {len(anomalies)} statistical anomalies")
            print("\nTop 5 statistical anomalies:")
            print(anomalies.sort_values('z_score', ascending=False).head())
        else:
            print("No statistical anomalies detected")
        
        # Run sequential anomaly detection
        print("\nRunning sequential anomaly detection...")
        seq_anomalies = detector.detect_sequential_anomalies(data)
        
        if len(seq_anomalies) > 0:
            print(f"Found {len(seq_anomalies)} sequential anomalies")
            print("\nTop 5 sequential anomalies:")
            print(seq_anomalies.sort_values('pct_change', ascending=False).head())
        else:
            print("No sequential anomalies detected")
        
        # Generate prompt
        print("\nGenerating Claude prompt...")
        all_anomalies = pd.concat([anomalies, seq_anomalies], ignore_index=True)
        prompt = detector.generate_anomaly_prompt(all_anomalies, data)
        
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
        
        # Plot example for a specific bank
        if len(data['bank_id'].unique()) > 0:
            bank_id = data['bank_id'].iloc[0]
            print(f"\nGenerating plots for Bank ID {bank_id}...")
            detector.plot_metric_trends(data, bank_id)
        
    except Exception as e:
        print(f"Error in Y-9C anomaly detector example: {str(e)}")