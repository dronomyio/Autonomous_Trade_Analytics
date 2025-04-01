"""
Interface to access financial engineering tools from the main application.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import datetime
import logging

# Import tools modules
from tools import returns, risk, statistics, indicators, portfolio

logger = logging.getLogger(__name__)

class FinancialToolkit:
    """
    Interface class to provide access to financial engineering tools.
    """
    
    def __init__(self):
        """Initialize the financial toolkit."""
        logger.info("Financial toolkit initialized")
        
    def log_return(self, prices: Union[List[float], np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate logarithmic returns directly using the returns module.
        
        Args:
            prices: Series of asset prices
            
        Returns:
            Series of log returns
        """
        try:
            return returns.log_return(prices)
        except Exception as e:
            logger.error(f"Error calculating log returns: {str(e)}")
            if isinstance(prices, pd.Series):
                return pd.Series(dtype=float)
            else:
                return np.array([])
    
    def analyze_returns(self, prices: pd.Series, 
                       periods_per_year: int = 252) -> Dict[str, float]:
        """
        Analyze returns for a given price series.
        
        Args:
            prices: Series of asset prices
            periods_per_year: Number of periods in a year
            
        Returns:
            Dictionary of return metrics
        """
        # Calculate returns
        simple_rets = returns.simple_return(prices)
        log_rets = returns.log_return(prices)
        
        # Calculate return metrics
        cumulative_ret = returns.cumulative_return(simple_rets)
        annual_ret = returns.annualized_return(simple_rets, periods_per_year)
        
        # Calculate risk metrics
        vol = risk.volatility(simple_rets, periods_per_year)
        sharpe = risk.sharpe_ratio(simple_rets, periods_per_year=periods_per_year)
        max_dd, max_dd_start, max_dd_end = risk.max_drawdown(prices)
        var95 = risk.value_at_risk(simple_rets, confidence_level=0.95)
        es95 = risk.expected_shortfall(simple_rets, confidence_level=0.95)
        
        # Return dictionary of metrics
        return {
            'total_return': cumulative_ret,
            'annualized_return': annual_ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'var_95': var95,
            'expected_shortfall_95': es95,
        }
    
    def generate_technical_indicators(self, 
                                    prices: pd.Series,
                                    high: Optional[pd.Series] = None, 
                                    low: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Generate common technical indicators for a price series.
        
        Args:
            prices: Series of closing prices
            high: Series of high prices (optional)
            low: Series of low prices (optional)
            
        Returns:
            Dictionary of technical indicators
        """
        indicators_dict = {}
        
        # Moving averages
        indicators_dict['sma_20'] = pd.Series(
            indicators.simple_moving_average(prices, 20),
            index=prices.index
        )
        indicators_dict['sma_50'] = pd.Series(
            indicators.simple_moving_average(prices, 50),
            index=prices.index
        )
        indicators_dict['ema_20'] = pd.Series(
            indicators.exponential_moving_average(prices, 20),
            index=prices.index
        )
        
        # Bollinger Bands
        bb = indicators.bollinger_bands(prices, 20, 2.0)
        indicators_dict['bb_upper'] = pd.Series(bb['upper'], index=prices.index)
        indicators_dict['bb_middle'] = pd.Series(bb['middle'], index=prices.index)
        indicators_dict['bb_lower'] = pd.Series(bb['lower'], index=prices.index)
        
        # RSI
        indicators_dict['rsi'] = pd.Series(
            indicators.relative_strength_index(prices),
            index=prices.index
        )
        
        # MACD
        macd_result = indicators.macd(prices)
        indicators_dict['macd'] = pd.Series(macd_result['macd'], index=prices.index)
        indicators_dict['macd_signal'] = pd.Series(macd_result['signal'], index=prices.index)
        indicators_dict['macd_histogram'] = pd.Series(macd_result['histogram'], index=prices.index)
        
        # Add ATR if high and low prices are available
        if high is not None and low is not None:
            indicators_dict['atr'] = pd.Series(
                indicators.average_true_range(high, low, prices),
                index=prices.index
            )
        
        return indicators_dict
    
    def optimize_portfolio(self, 
                         returns_df: pd.DataFrame,
                         risk_free_rate: float = 0.0,
                         target_return: Optional[float] = None) -> Dict:
        """
        Optimize a portfolio of assets based on historical returns.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            risk_free_rate: Risk-free rate
            target_return: Target portfolio return (optional)
            
        Returns:
            Dictionary containing optimal portfolio information
        """
        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # If target return is specified, optimize for minimum volatility at that return
        if target_return is not None:
            target_returns = np.array([target_return])
            volatilities, weights = portfolio.efficient_frontier(
                expected_returns.values, 
                cov_matrix.values,
                target_returns
            )
            
            optimal_weights = pd.Series(
                weights[0], 
                index=returns_df.columns
            )
            
            # Calculate portfolio metrics
            portfolio_return = portfolio.portfolio_return(weights[0], expected_returns.values)
            portfolio_vol = volatilities[0]
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_vol,
                'sharpe_ratio': sharpe
            }
        
        # Otherwise, find the portfolio with the highest Sharpe ratio
        else:
            stats, optimal_weights = portfolio.optimal_portfolio(
                expected_returns.values,
                cov_matrix.values,
                risk_free_rate
            )
            
            # Assign column names to weights
            optimal_weights.index = returns_df.columns
            
            return {
                'weights': optimal_weights,
                'expected_return': stats['return'],
                'expected_volatility': stats['volatility'],
                'sharpe_ratio': stats['sharpe_ratio']
            }
    
    def analyze_portfolio_performance(self,
                                   returns_df: pd.DataFrame,
                                   weights: pd.Series,
                                   risk_free_rate: float = 0.0) -> Dict:
        """
        Analyze the historical performance of a portfolio.
        
        Args:
            returns_df: DataFrame of asset returns
            weights: Series of asset weights (index should match returns_df columns)
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        # Align weights with returns columns
        aligned_weights = weights.reindex(returns_df.columns, fill_value=0)
        
        # Calculate performance metrics
        performance = portfolio.portfolio_performance(
            returns_df,
            aligned_weights.values,
            risk_free_rate=risk_free_rate
        )
        
        return performance
    
    def test_cointegration(self, 
                         series1: pd.Series, 
                         series2: pd.Series) -> Dict:
        """
        Test for cointegration between two price series.
        
        Args:
            series1: First price series
            series2: Second price series
            
        Returns:
            Dictionary with cointegration test results
        """
        is_cointegrated, p_value, test_stats = statistics.cointegration_test(
            series1.values, 
            series2.values
        )
        
        return {
            'is_cointegrated': is_cointegrated,
            'p_value': p_value,
            'hedge_ratio': test_stats['beta'],
            'test_statistic': test_stats['adf_statistic'],
            'critical_values': test_stats['critical_values']
        }