"""
Regulatory Filing Analysis Module

This module provides a unified interface for analyzing financial regulatory filings,
including FR Y-9C bank reports and SEC 10-K annual reports. It integrates the specialized
anomaly detectors and provides standardized access to their functionality.
"""

import os
import json
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd

# Import specialized detectors
from tools.y9c_anomaly_detector import Y9CAnomalyDetector
from tools.sec_anomaly_detector import TenKAnomalyDetector

class RegulatoryAnalyzer:
    """
    Unified interface for analyzing various regulatory filings.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the regulatory analyzer.
        
        Args:
            api_key: Anthropic API key for Claude (optional)
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        
        # Initialize the specialized detectors
        self.y9c_detector = Y9CAnomalyDetector(api_key=self.api_key)
        self.sec_detector = TenKAnomalyDetector(api_key=self.api_key)
    
    def analyze_y9c_filing(self, 
                         file_path: Optional[str] = None,
                         bank_id: Optional[int] = None,
                         use_claude: bool = True,
                         save_output: bool = False) -> Dict:
        """
        Analyze a Y-9C bank regulatory filing for anomalies.
        
        Args:
            file_path: Path to Y-9C data file (optional)
            bank_id: Specific bank ID to focus on (optional)
            use_claude: Whether to use Claude for analysis
            save_output: Whether to save output to files
            
        Returns:
            Dictionary with analysis results
        """
        return self.y9c_detector.full_anomaly_analysis(
            file_path=file_path,
            bank_id=bank_id,
            use_claude=use_claude,
            save_prompt=save_output,
            save_plot=save_output
        )
    
    def analyze_10k_filing(self, 
                         file_path: Optional[str] = None,
                         ticker: Optional[str] = None,
                         use_claude: bool = True,
                         save_output: bool = False) -> Dict:
        """
        Analyze a 10-K SEC filing for anomalies.
        
        Args:
            file_path: Path to 10-K data file (optional)
            ticker: Specific company ticker to focus on (optional)
            use_claude: Whether to use Claude for analysis
            save_output: Whether to save output to files
            
        Returns:
            Dictionary with analysis results
        """
        return self.sec_detector.full_10k_analysis(
            file_path=file_path,
            ticker=ticker,
            use_claude=use_claude,
            save_prompt=save_output,
            save_plot=save_output
        )
    
    def get_y9c_prompt(self,
                     file_path: Optional[str] = None,
                     bank_id: Optional[int] = None) -> str:
        """
        Generate a Y-9C bank anomaly detection prompt without running analysis.
        
        Args:
            file_path: Path to Y-9C data (optional)
            bank_id: Specific bank ID (optional)
            
        Returns:
            Formatted prompt for anomaly detection
        """
        # Load data
        data = self.y9c_detector.load_y9c_data(file_path)
        
        # Detect anomalies
        statistical_anomalies = self.y9c_detector.detect_statistical_anomalies(data, bank_id)
        sequential_anomalies = self.y9c_detector.detect_sequential_anomalies(data, bank_id)
        
        # Generate and return prompt
        all_anomalies = pd.concat([statistical_anomalies, sequential_anomalies], ignore_index=True)
        return self.y9c_detector.generate_anomaly_prompt(all_anomalies, data, bank_id)
    
    def get_10k_prompt(self,
                     file_path: Optional[str] = None,
                     ticker: Optional[str] = None) -> str:
        """
        Generate a 10-K filing anomaly detection prompt without running analysis.
        
        Args:
            file_path: Path to 10-K data (optional)
            ticker: Specific company ticker (optional)
            
        Returns:
            Formatted prompt for anomaly detection
        """
        # Load data
        data = self.sec_detector.load_10k_data(file_path)
        
        # Detect anomalies
        financial_anomalies = self.sec_detector.detect_financial_statement_anomalies(data, ticker)
        yoy_anomalies = self.sec_detector.detect_yoy_anomalies(data, ticker)
        non_financial_anomalies = self.sec_detector.detect_non_financial_anomalies(data, ticker)
        
        # Generate and return prompt
        all_anomalies = pd.concat([financial_anomalies, yoy_anomalies, non_financial_anomalies], 
                                ignore_index=True)
        return self.sec_detector.generate_10k_anomaly_prompt(all_anomalies, data, ticker)
    
    def run_claude_analysis(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Run a Claude analysis on a provided prompt.
        
        Args:
            prompt: Analysis prompt to send to Claude
            max_tokens: Maximum tokens for response
            
        Returns:
            Claude's analysis text
        """
        # Use either detector's Claude client
        return self.y9c_detector.analyze_with_claude(prompt, max_tokens=max_tokens)


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    print("Initializing regulatory analyzer...")
    analyzer = RegulatoryAnalyzer()
    
    # Y-9C example
    print("\n--- Y-9C BANK FILING ANALYSIS ---")
    try:
        # Generate prompt for bank data
        print("Generating Y-9C anomaly detection prompt...")
        y9c_prompt = analyzer.get_y9c_prompt()
        
        # Show preview
        print("\nY-9C Prompt Preview:")
        print(y9c_prompt[:500] + "...\n[Truncated]")
        
    except Exception as e:
        print(f"Error in Y-9C analysis: {str(e)}")
    
    # 10-K example
    print("\n--- 10-K SEC FILING ANALYSIS ---")
    try:
        # Generate prompt for META (example)
        print("Generating 10-K anomaly detection prompt for META...")
        k10_prompt = analyzer.get_10k_prompt(ticker="META")
        
        # Show preview
        print("\n10-K Prompt Preview:")
        print(k10_prompt[:500] + "...\n[Truncated]")
        
    except Exception as e:
        print(f"Error in 10-K analysis: {str(e)}")
    
    # Run Claude analysis if API key is available
    if analyzer.api_key:
        print("\n--- CLAUDE ANALYSIS EXAMPLE ---")
        print("Running Claude analysis on a sample prompt...")
        
        # Use a smaller test prompt for the example
        test_prompt = """
        Analyze the following financial anomaly:
        
        Company: Meta Platforms Inc. (META)
        Metric: operating_margin
        Current Value: 34.20
        Historical Average: 40.60
        Z-score: -2.80
        
        The company also reported a recent restructuring effort.
        
        What might explain this anomaly and what questions should analysts ask?
        """
        
        analysis = analyzer.run_claude_analysis(test_prompt, max_tokens=1000)
        print("\nClaude Analysis:")
        print(analysis)