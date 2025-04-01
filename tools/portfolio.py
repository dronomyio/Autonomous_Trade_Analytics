"""
Portfolio optimization and analysis tools.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional
from scipy.optimize import minimize


def portfolio_return(weights: np.ndarray, 
                    expected_returns: np.ndarray) -> float:
    """
    Calculate the expected return of a portfolio.
    
    Args:
        weights: Array of asset weights in the portfolio
        expected_returns: Array of expected returns for each asset
        
    Returns:
        Expected portfolio return
    """
    return np.sum(weights * expected_returns)


def portfolio_volatility(weights: np.ndarray, 
                       cov_matrix: np.ndarray) -> float:
    """
    Calculate the volatility (standard deviation) of a portfolio.
    
    Args:
        weights: Array of asset weights
        cov_matrix: Covariance matrix of asset returns
        
    Returns:
        Portfolio volatility
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def efficient_frontier(expected_returns: np.ndarray,
                     cov_matrix: np.ndarray,
                     target_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the efficient frontier for a portfolio.
    
    Args:
        expected_returns: Array of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        target_returns: Array of target returns for which to find optimal portfolios
        
    Returns:
        Tuple of (volatilities, optimal_weights) where optimal_weights is a 2D array
    """
    n_assets = len(expected_returns)
    
    def portfolio_volatility_func(weights):
        return portfolio_volatility(weights, cov_matrix)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    ]
    
    # Bounds for weights (0% to 100%)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess
    init_guess = np.ones(n_assets) / n_assets
    
    # Storage for results
    volatilities = np.zeros(len(target_returns))
    optimal_weights = np.zeros((len(target_returns), n_assets))
    
    for i, target_return in enumerate(target_returns):
        # Add return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda x: portfolio_return(x, expected_returns) - target_return
        }
        specific_constraints = constraints + [return_constraint]
        
        # Optimize
        result = minimize(
            portfolio_volatility_func,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=specific_constraints
        )
        
        # Store results
        volatilities[i] = result['fun']
        optimal_weights[i, :] = result['x']
    
    return volatilities, optimal_weights


def optimal_portfolio(expected_returns: np.ndarray,
                     cov_matrix: np.ndarray,
                     risk_free_rate: float = 0.0) -> Tuple[Dict, pd.Series]:
    """
    Find the optimal portfolio based on the Sharpe ratio.
    
    Args:
        expected_returns: Array of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Tuple of (portfolio_stats, optimal_weights)
    """
    n_assets = len(expected_returns)
    
    def negative_sharpe_ratio(weights):
        portfolio_ret = portfolio_return(weights, expected_returns)
        portfolio_vol = portfolio_volatility(weights, cov_matrix)
        sharpe = (portfolio_ret - risk_free_rate) / portfolio_vol
        return -sharpe
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    ]
    
    # Bounds for weights (0% to 100%)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess (equal weight)
    init_guess = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        negative_sharpe_ratio,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = pd.Series(result['x'], index=list(range(n_assets)))
    
    # Calculate portfolio statistics
    portfolio_ret = portfolio_return(result['x'], expected_returns)
    portfolio_vol = portfolio_volatility(result['x'], cov_matrix)
    sharpe = (portfolio_ret - risk_free_rate) / portfolio_vol
    
    portfolio_stats = {
        'return': portfolio_ret,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe
    }
    
    return portfolio_stats, optimal_weights


def portfolio_performance(returns: pd.DataFrame,
                        weights: np.ndarray,
                        risk_free_rate: float = 0.0,
                        periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate performance metrics for a portfolio.
    
    Args:
        returns: DataFrame of asset returns
        weights: Array of asset weights
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert weights to array
    weights = np.array(weights)
    
    # Calculate portfolio returns
    portfolio_rets = returns.dot(weights)
    
    # Calculate mean return and volatility
    mean_return = portfolio_rets.mean() * periods_per_year
    volatility = portfolio_rets.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    sharpe = (mean_return - risk_free_rate) / volatility
    
    # Max drawdown
    cum_returns = (1 + portfolio_rets).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    max_dd = drawdown.min()
    
    # Sortino ratio: penalizes only downside volatility
    downside_returns = portfolio_rets[portfolio_rets < 0]
    downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)
    sortino = (mean_return - risk_free_rate) / downside_volatility if len(downside_returns) > 0 else np.nan
    
    # Calmar ratio: return / max drawdown
    calmar = -mean_return / max_dd if max_dd != 0 else np.nan
    
    return {
        'return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'cumulative_return': cum_returns.iloc[-1] - 1
    }