"""
Technical indicators for financial market analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional


def simple_moving_average(data: Union[List[float], np.ndarray, pd.Series], 
                         window: int = 20) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data: Array-like of price data
        window: Window size for moving average
        
    Returns:
        Array of SMA values (with NaN for the first window-1 positions)
    """
    data = np.array(data)
    weights = np.ones(window)
    sma = np.convolve(data, weights/weights.sum(), mode='valid')
    
    # Pad the beginning with NaNs to maintain the original length
    return np.append(np.array([np.nan] * (window-1)), sma)


def exponential_moving_average(data: Union[List[float], np.ndarray, pd.Series], 
                              span: int = 20) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data: Array-like of price data
        span: Span parameter for EMA calculation
        
    Returns:
        Array of EMA values
    """
    if isinstance(data, pd.Series):
        return data.ewm(span=span, adjust=False).mean().values
    else:
        data = pd.Series(data)
        return data.ewm(span=span, adjust=False).mean().values


def bollinger_bands(data: Union[List[float], np.ndarray, pd.Series], 
                  window: int = 20, 
                  num_std: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Array-like of price data
        window: Window size for moving average
        num_std: Number of standard deviations for the bands
        
    Returns:
        Dictionary containing 'upper', 'middle', and 'lower' bands
    """
    data = np.array(data)
    
    # Calculate middle band (SMA)
    middle_band = simple_moving_average(data, window)
    
    # Calculate standard deviation within the rolling window
    if isinstance(data, pd.Series):
        rolling_std = data.rolling(window=window).std().values
    else:
        rolling_std = pd.Series(data).rolling(window=window).std().values
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }


def relative_strength_index(prices: Union[List[float], np.ndarray, pd.Series], 
                           window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Array-like of price data
        window: Window size for RSI calculation
        
    Returns:
        Array of RSI values
    """
    prices = np.array(prices)
    
    # Calculate price changes
    deltas = np.diff(prices)
    seed = deltas[:window]
    
    # Calculate initial gains and losses
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    
    if down == 0:
        # First value is set to 100 if no down days
        rs = np.inf
        rsi = np.array([100.0])
    else:
        rs = up / down
        rsi = np.array([100.0 - (100.0 / (1.0 + rs))])
    
    # Calculate RSI for the remaining data points
    for i in range(window, len(deltas)):
        delta = deltas[i]
        
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta
            
        # Use exponential moving average to update up and down
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        rs = up / down if down != 0 else np.inf
        rsi = np.append(rsi, 100.0 - (100.0 / (1.0 + rs)))
    
    # Pad with NaNs to maintain original length
    return np.append(np.array([np.nan] * window), rsi)


def macd(prices: Union[List[float], np.ndarray, pd.Series], 
        fast_span: int = 12, 
        slow_span: int = 26, 
        signal_span: int = 9) -> Dict[str, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Array-like of price data
        fast_span: Span for the fast EMA
        slow_span: Span for the slow EMA
        signal_span: Span for the signal line
        
    Returns:
        Dictionary containing 'macd', 'signal', and 'histogram'
    """
    prices = np.array(prices)
    
    # Calculate fast and slow EMAs
    fast_ema = exponential_moving_average(prices, fast_span)
    slow_ema = exponential_moving_average(prices, slow_span)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = exponential_moving_average(macd_line, signal_span)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def average_true_range(high: Union[List[float], np.ndarray, pd.Series],
                     low: Union[List[float], np.ndarray, pd.Series],
                     close: Union[List[float], np.ndarray, pd.Series],
                     window: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: Array-like of high prices
        low: Array-like of low prices
        close: Array-like of closing prices
        window: Window size for ATR calculation
        
    Returns:
        Array of ATR values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    # Calculate true range
    prev_close = np.insert(close[:-1], 0, close[0])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR using a simple moving average of true range
    atr = np.zeros_like(true_range)
    atr[:window] = np.nan
    atr[window - 1] = np.mean(true_range[:window])
    
    # Calculate the remaining ATR values
    for i in range(window, len(true_range)):
        atr[i] = (atr[i-1] * (window-1) + true_range[i]) / window
    
    return atr