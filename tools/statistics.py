"""
Statistical tools for analyzing financial data.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional
from scipy import stats


def rolling_statistics(returns: Union[pd.Series, np.ndarray], 
                       window: int = 21) -> Dict[str, pd.Series]:
    """
    Calculate rolling statistics for returns.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        
    Returns:
        Dictionary of pandas Series for various rolling statistics
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    roll_stats = {
        'mean': returns.rolling(window=window).mean(),
        'std': returns.rolling(window=window).std(),
        'skew': returns.rolling(window=window).skew(),
        'kurt': returns.rolling(window=window).kurt(),
        'sharpe': returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(252),
        'min': returns.rolling(window=window).min(),
        'max': returns.rolling(window=window).max(),
    }
    
    return roll_stats


def is_normal(returns: Union[List[float], np.ndarray, pd.Series], 
              test_method: str = 'shapiro',
              alpha: float = 0.05) -> Tuple[bool, float, Dict]:
    """
    Test if returns follow a normal distribution.
    
    Args:
        returns: Array-like of asset returns
        test_method: Statistical test to use ('shapiro', 'ks', 'jarque_bera')
        alpha: Significance level
        
    Returns:
        Tuple of (is_normal, p_value, test_statistics)
    """
    returns = np.array(returns)
    
    if test_method == 'shapiro':
        stat, p_value = stats.shapiro(returns)
        test_stats = {'statistic': stat}
    
    elif test_method == 'ks':
        # Fit normal distribution to the data
        mean, std = np.mean(returns), np.std(returns)
        # Perform Kolmogorov-Smirnov test against the fitted normal distribution
        stat, p_value = stats.kstest(returns, 'norm', args=(mean, std))
        test_stats = {'statistic': stat}
    
    elif test_method == 'jarque_bera':
        stat, p_value = stats.jarque_bera(returns)
        test_stats = {
            'statistic': stat,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns, fisher=False)
        }
    
    else:
        raise ValueError("Method must be 'shapiro', 'ks', or 'jarque_bera'")
    
    # If p-value > alpha, we fail to reject the null hypothesis (normality)
    is_normal = p_value > alpha
    
    return is_normal, p_value, test_stats


def autocorrelation(returns: Union[List[float], np.ndarray, pd.Series], 
                    lags: int = 20) -> pd.Series:
    """
    Calculate autocorrelation of returns for different lags.
    
    Args:
        returns: Array-like of asset returns
        lags: Number of lags to calculate
        
    Returns:
        Series of autocorrelation coefficients for each lag
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    return pd.Series([returns.autocorr(lag=i) for i in range(1, lags+1)],
                     index=[f'Lag {i}' for i in range(1, lags+1)])


def cointegration_test(series1: Union[List[float], np.ndarray, pd.Series],
                     series2: Union[List[float], np.ndarray, pd.Series],
                     alpha: float = 0.05) -> Tuple[bool, float, Dict]:
    """
    Test for cointegration between two time series using the Engle-Granger method.
    
    Args:
        series1: First time series
        series2: Second time series
        alpha: Significance level
        
    Returns:
        Tuple of (is_cointegrated, p_value, test_statistics)
    """
    if not isinstance(series1, np.ndarray):
        series1 = np.array(series1)
    if not isinstance(series2, np.ndarray):
        series2 = np.array(series2)
    
    # Ensure both series have the same length
    min_length = min(len(series1), len(series2))
    series1 = series1[-min_length:]
    series2 = series2[-min_length:]
    
    # Step 1: Run OLS regression
    X = np.column_stack((np.ones(min_length), series1))
    beta = np.linalg.lstsq(X, series2, rcond=None)[0]
    
    # Step 2: Get residuals (deviations from the long-run equilibrium)
    residuals = series2 - X @ beta
    
    # Step 3: Test for unit root in the residuals
    # If residuals are stationary, the series are cointegrated
    result = stats.adfuller(residuals)
    
    p_value = result[1]
    is_cointegrated = p_value < alpha
    
    test_stats = {
        'adf_statistic': result[0],
        'critical_values': result[4],
        'beta': beta[1],  # The cointegration coefficient
        'alpha': beta[0]  # The intercept
    }
    
    return is_cointegrated, p_value, test_stats