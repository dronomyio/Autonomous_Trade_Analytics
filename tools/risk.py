"""
Tools for calculating risk metrics for financial assets.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional


def volatility(returns: Union[List[float], np.ndarray, pd.Series], 
               periods_per_year: int = 252,
               annualized: bool = True) -> float:
    """
    Calculate the volatility (standard deviation) of returns.
    
    Args:
        returns: Array-like of asset returns
        periods_per_year: Number of periods in a year (e.g., 252 for daily trading days)
        annualized: If True, returns the annualized volatility
        
    Returns:
        Volatility of returns
    """
    returns = np.array(returns)
    vol = np.std(returns, ddof=1)
    
    if annualized:
        vol = vol * np.sqrt(periods_per_year)
        
    return vol


def sharpe_ratio(returns: Union[List[float], np.ndarray, pd.Series], 
                risk_free_rate: float = 0.0, 
                periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: Array-like of asset returns
        risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        periods_per_year: Number of periods in a year (e.g., 252 for daily trading days)
        
    Returns:
        Sharpe ratio
    """
    returns = np.array(returns)
    
    # Convert annual risk-free rate to period risk-free rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - rf_per_period
    ann_excess_return = np.mean(excess_returns) * periods_per_year
    ann_volatility = volatility(returns, periods_per_year, True)
    
    return ann_excess_return / ann_volatility


def max_drawdown(prices: Union[List[float], np.ndarray, pd.Series]) -> Tuple[float, int, int]:
    """
    Calculate the maximum drawdown along with its start and end indices.
    
    Args:
        prices: Array-like of asset prices
        
    Returns:
        Tuple containing (maximum drawdown, start index, end index)
    """
    prices = np.array(prices)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdowns
    drawdowns = (running_max - prices) / running_max
    
    # Find maximum drawdown and its index
    max_dd = np.max(drawdowns)
    max_dd_end_idx = np.argmax(drawdowns)
    
    # Find the start of the drawdown period (the peak before the drawdown)
    max_dd_start_idx = np.argmax(prices[:max_dd_end_idx+1])
    
    return max_dd, max_dd_start_idx, max_dd_end_idx


def value_at_risk(returns: Union[List[float], np.ndarray, pd.Series], 
                  confidence_level: float = 0.95,
                  method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR) for a series of returns.
    
    Args:
        returns: Array-like of asset returns
        confidence_level: Confidence level for VaR calculation (e.g., 0.95 for 95%)
        method: 'historical' or 'parametric'
        
    Returns:
        Value at Risk (VaR)
    """
    returns = np.array(returns)
    
    if method == 'historical':
        # Historical VaR is simply the relevant percentile of the returns
        return -np.percentile(returns, 100 * (1 - confidence_level))
    
    elif method == 'parametric':
        # Parametric VaR assumes normally distributed returns
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        # Find the z-score for the given confidence level
        z_score = -np.percentile(np.random.normal(0, 1, 100000), 100 * (1 - confidence_level))
        return -(mean + z_score * std)
    
    else:
        raise ValueError("Method must be 'historical' or 'parametric'")


def expected_shortfall(returns: Union[List[float], np.ndarray, pd.Series], 
                      confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR) for a series of returns.
    
    Args:
        returns: Array-like of asset returns
        confidence_level: Confidence level for ES calculation (e.g., 0.95 for 95%)
        
    Returns:
        Expected Shortfall
    """
    returns = np.array(returns)
    
    # Sort returns in ascending order
    sorted_returns = np.sort(returns)
    
    # Find the cutoff index
    cutoff_index = int(len(sorted_returns) * (1 - confidence_level))
    
    # Calculate the average of the worst (1-confidence_level)% returns
    return -np.mean(sorted_returns[:cutoff_index])