"""
Tools for calculating various types of returns on financial assets.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple


def simple_return(prices: Union[List[float], np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Calculate simple returns from a series of prices.
    
    Args:
        prices: Array-like of asset prices
        
    Returns:
        Array-like of simple returns (same type as input)
    """
    if isinstance(prices, pd.Series):
        return prices.pct_change().dropna()
    elif isinstance(prices, (list, np.ndarray)):
        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]
        return returns
    else:
        raise TypeError("Input must be a list, numpy array, or pandas Series")


def log_return(prices: Union[List[float], np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Calculate logarithmic returns from a series of prices.
    
    Args:
        prices: Array-like of asset prices
        
    Returns:
        Array-like of logarithmic returns (same type as input)
    """
    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1)).dropna()
    elif isinstance(prices, (list, np.ndarray)):
        prices = np.array(prices)
        returns = np.diff(np.log(prices))
        return returns
    else:
        raise TypeError("Input must be a list, numpy array, or pandas Series")


def cumulative_return(returns: Union[List[float], np.ndarray, pd.Series], log_returns: bool = False) -> float:
    """
    Calculate the cumulative return from a series of returns.
    
    Args:
        returns: Array-like of asset returns
        log_returns: If True, input is treated as logarithmic returns
        
    Returns:
        Cumulative return as a percentage
    """
    if log_returns:
        return np.exp(np.sum(returns)) - 1
    else:
        returns = np.array(returns)
        return np.prod(1 + returns) - 1


def annualized_return(returns: Union[List[float], np.ndarray, pd.Series], 
                      periods_per_year: int, 
                      log_returns: bool = False) -> float:
    """
    Calculate the annualized return from a series of returns.
    
    Args:
        returns: Array-like of asset returns
        periods_per_year: Number of periods in a year (e.g., 252 for daily trading days)
        log_returns: If True, input is treated as logarithmic returns
        
    Returns:
        Annualized return as a percentage
    """
    cumul_return = cumulative_return(returns, log_returns)
    n_periods = len(returns)
    
    return (1 + cumul_return) ** (periods_per_year / n_periods) - 1