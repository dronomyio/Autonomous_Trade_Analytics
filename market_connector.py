"""
Market Connector for Trade Analytics

This module handles connections to market data APIs and brokerages.
Replaces the Pokemon MCP Handler in the original example.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from eva import log, LogLevel, MarketInstance

class MarketDataProvider:
    """Provides market data from various sources."""
    
    def __init__(self):
        """Initialize the market data provider."""
        self.cache = {}
        self.cache_expiry = {}
        # Default cache time of 5 minutes for most data
        self.default_cache_time = 300  
    
    async def get_price_data(self, ticker: str, period: str = "1d", interval: str = "1m"):
        """Get price data for a ticker."""
        cache_key = f"price_{ticker}_{period}_{interval}"
        
        # Check cache first
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            log(LogLevel.DEBUG, f"Using cached price data for {ticker}")
            return self.cache[cache_key]
        
        try:
            # Use yfinance to get data
            log(LogLevel.INFO, f"Fetching price data for {ticker}", 
                extra={"period": period, "interval": interval})
            
            # This would be async in a real implementation
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, lambda: yf.download(
                ticker, period=period, interval=interval, progress=False
            ))
            
            # Convert to dict for easier JSON handling
            result = {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "last_price": float(data['Close'].iloc[-1]) if not data.empty else None,
                "change_pct": float(data['Close'].pct_change().iloc[-1]) if not data.empty else None,
                "high": float(data['High'].max()) if not data.empty else None,
                "low": float(data['Low'].min()) if not data.empty else None,
                "volume": int(data['Volume'].sum()) if not data.empty else None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in cache
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + self.default_cache_time
            
            return result
        except Exception as e:
            log(LogLevel.ERROR, f"Error fetching price data for {ticker}", extra={"error": str(e)})
            return {"error": str(e)}
    
    async def get_market_summary(self):
        """Get overall market summary."""
        cache_key = "market_summary"
        
        # Check cache first
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Get data for major indices
            indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
            index_names = ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000"]
            
            all_data = {}
            for idx, name in zip(indices, index_names):
                data = await self.get_price_data(idx, period="1d", interval="1h")
                if "error" not in data:
                    all_data[name] = data
            
            # Get sector performance
            sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
            sector_names = ["Technology", "Financial", "Energy", "Healthcare", "Industrial", 
                           "Consumer Staples", "Consumer Discretionary", "Utilities", 
                           "Materials", "Real Estate", "Communication"]
            
            sector_data = {}
            for sector, name in zip(sectors, sector_names):
                data = await self.get_price_data(sector, period="1d", interval="1h")
                if "error" not in data:
                    sector_data[name] = {"change_pct": data["change_pct"]}
            
            result = {
                "indices": all_data,
                "sectors": sector_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in cache (shorter expiry for market summary)
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + 60  # 1 minute cache
            
            return result
        except Exception as e:
            log(LogLevel.ERROR, f"Error fetching market summary", extra={"error": str(e)})
            return {"error": str(e)}
    
    async def get_company_info(self, ticker: str):
        """Get company information."""
        cache_key = f"company_info_{ticker}"
        
        # Check cache first (longer cache time for company info)
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cache[cache_key]
        
        try:
            # Get company info from yfinance
            log(LogLevel.INFO, f"Fetching company info for {ticker}")
            
            # This would be async in a real implementation
            loop = asyncio.get_event_loop()
            stock = await loop.run_in_executor(None, lambda: yf.Ticker(ticker))
            
            # Get basic info and financials
            info = stock.info if hasattr(stock, 'info') else {}
            
            # Create simplified result
            result = {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in cache (longer cache time for company info)
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + 86400  # 24 hour cache
            
            return result
        except Exception as e:
            log(LogLevel.ERROR, f"Error fetching company info for {ticker}", extra={"error": str(e)})
            return {"error": str(e)}

class PortfolioManager:
    """Manages portfolio operations and tracking."""
    
    def __init__(self, mode="paper"):
        """Initialize the portfolio manager."""
        self.mode = mode  # 'paper' or 'live'
        self.portfolio = self._initialize_portfolio()
        self.market_data = MarketDataProvider()
        self.transaction_history = []
        log(LogLevel.INFO, f"Initialized Portfolio Manager in {mode} mode")
    
    def _initialize_portfolio(self):
        """Initialize a test portfolio."""
        # In a real app, this would load from a database or API
        return {
            "cash": 100000.0,
            "positions": {
                "AAPL": {"shares": 50, "avg_price": 180.0, "current_price": 180.0},
                "MSFT": {"shares": 30, "avg_price": 350.0, "current_price": 350.0},
                "GOOGL": {"shares": 20, "avg_price": 140.0, "current_price": 140.0},
                "AMZN": {"shares": 15, "avg_price": 160.0, "current_price": 160.0},
            },
            "initial_value": 100000.0,  # Starting cash
            "current_value": 100000.0,  # Will be updated
            "last_updated": datetime.now().isoformat(),
            "target_allocation": {
                "Technology": 0.40,
                "Consumer Discretionary": 0.15,
                "Healthcare": 0.15,
                "Financial": 0.15,
                "Energy": 0.05,
                "Cash": 0.10
            },
            "current_allocation": {
                "Technology": 0.60,  # Higher than target
                "Consumer Discretionary": 0.15,
                "Healthcare": 0.05,  # Lower than target
                "Financial": 0.10,  # Lower than target
                "Energy": 0.0,  # No exposure
                "Cash": 0.10
            },
            "risk_metrics": {
                "portfolio_volatility": 0.20,
                "beta": 1.2,
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.15
            },
            "initial_risk_metrics": {
                "portfolio_volatility": 0.25,  # Higher initial risk
                "beta": 1.3,
                "sharpe_ratio": 0.7,
                "max_drawdown": 0.18
            },
            "sector_exposure": {
                "Technology": 0.60,
                "Consumer Discretionary": 0.15,
                "Healthcare": 0.05,
                "Financial": 0.10,
                "Energy": 0.0,
                "Cash": 0.10
            },
            "target_return": 0.05  # 5% target return
        }
    
    async def update_portfolio_prices(self):
        """Update current prices for all positions."""
        for ticker, position in self.portfolio["positions"].items():
            price_data = await self.market_data.get_price_data(ticker)
            if "error" not in price_data and price_data["last_price"]:
                position["current_price"] = price_data["last_price"]
        
        # Update portfolio value
        self._calculate_portfolio_value()
        
        log(LogLevel.INFO, "Updated portfolio prices")
        return self.portfolio
    
    def _calculate_portfolio_value(self):
        """Calculate the current portfolio value."""
        total_value = self.portfolio["cash"]
        
        for ticker, position in self.portfolio["positions"].items():
            position_value = position["shares"] * position["current_price"]
            total_value += position_value
        
        self.portfolio["current_value"] = total_value
        self.portfolio["last_updated"] = datetime.now().isoformat()
        
        return total_value
    
    async def execute_order(self, order: Dict[str, Any]):
        """Execute a trading order."""
        order_type = order.get("type", "").lower()
        ticker = order.get("ticker", "")
        quantity = order.get("quantity", 0)
        price = order.get("price", 0)
        
        if not ticker or quantity <= 0:
            return {"error": "Invalid order parameters"}
        
        # Get current price if not specified
        if price <= 0:
            price_data = await self.market_data.get_price_data(ticker)
            if "error" in price_data or not price_data["last_price"]:
                return {"error": f"Could not get price for {ticker}"}
            price = price_data["last_price"]
        
        transaction = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "type": order_type,
            "status": "pending"
        }
        
        try:
            # Process based on order type
            if order_type == "buy":
                transaction = await self._execute_buy(ticker, quantity, price, transaction)
            elif order_type == "sell":
                transaction = await self._execute_sell(ticker, quantity, price, transaction)
            else:
                transaction["status"] = "failed"
                transaction["message"] = f"Unsupported order type: {order_type}"
            
            # Add to transaction history
            self.transaction_history.append(transaction)
            
            # Update portfolio value
            self._calculate_portfolio_value()
            
            return transaction
        except Exception as e:
            log(LogLevel.ERROR, f"Error executing order", extra={"error": str(e), "order": order})
            transaction["status"] = "failed"
            transaction["message"] = str(e)
            self.transaction_history.append(transaction)
            return transaction
    
    async def _execute_buy(self, ticker, quantity, price, transaction):
        """Execute a buy order."""
        # Calculate total cost
        total_cost = quantity * price
        
        # Check if we have enough cash
        if self.portfolio["cash"] < total_cost:
            transaction["status"] = "failed"
            transaction["message"] = "Insufficient cash"
            return transaction
        
        # Update cash
        self.portfolio["cash"] -= total_cost
        
        # Update position
        if ticker in self.portfolio["positions"]:
            # Update existing position
            position = self.portfolio["positions"][ticker]
            total_shares = position["shares"] + quantity
            total_cost = (position["shares"] * position["avg_price"]) + (quantity * price)
            position["shares"] = total_shares
            position["avg_price"] = total_cost / total_shares
            position["current_price"] = price
        else:
            # Create new position
            self.portfolio["positions"][ticker] = {
                "shares": quantity,
                "avg_price": price,
                "current_price": price
            }
        
        transaction["status"] = "completed"
        transaction["message"] = f"Bought {quantity} shares of {ticker} at ${price:.2f}"
        
        # Update sector allocation data
        await self._update_sector_allocation()
        
        return transaction
    
    async def _execute_sell(self, ticker, quantity, price, transaction):
        """Execute a sell order."""
        # Check if we have the position
        if ticker not in self.portfolio["positions"]:
            transaction["status"] = "failed"
            transaction["message"] = f"No position in {ticker}"
            return transaction
        
        position = self.portfolio["positions"][ticker]
        
        # Check if we have enough shares
        if position["shares"] < quantity:
            transaction["status"] = "failed"
            transaction["message"] = f"Insufficient shares: have {position['shares']}, trying to sell {quantity}"
            return transaction
        
        # Calculate proceeds
        proceeds = quantity * price
        
        # Update cash
        self.portfolio["cash"] += proceeds
        
        # Update position
        position["shares"] -= quantity
        if position["shares"] == 0:
            # Remove position if no shares left
            del self.portfolio["positions"][ticker]
        else:
            # Update current price
            position["current_price"] = price
        
        transaction["status"] = "completed"
        transaction["message"] = f"Sold {quantity} shares of {ticker} at ${price:.2f}"
        
        # Update sector allocation data
        await self._update_sector_allocation()
        
        return transaction
    
    async def _update_sector_allocation(self):
        """Update sector allocation data after transactions."""
        # Get total portfolio value
        total_value = self._calculate_portfolio_value()
        
        # Reset sector exposure
        self.portfolio["sector_exposure"] = {
            "Technology": 0.0,
            "Consumer Discretionary": 0.0,
            "Healthcare": 0.0,
            "Financial": 0.0,
            "Energy": 0.0,
            "Cash": self.portfolio["cash"] / total_value
        }
        
        # Update current allocation
        for ticker, position in self.portfolio["positions"].items():
            # Get company info for sector
            company_info = await self.market_data.get_company_info(ticker)
            sector = company_info.get("sector", "Other")
            
            # Calculate position value and percentage
            position_value = position["shares"] * position["current_price"]
            position_pct = position_value / total_value
            
            # Add to sector exposure
            if sector in self.portfolio["sector_exposure"]:
                self.portfolio["sector_exposure"][sector] += position_pct
            else:
                self.portfolio["sector_exposure"][sector] = position_pct
        
        # Update current allocation to match sector exposure
        self.portfolio["current_allocation"] = self.portfolio["sector_exposure"]
        
        return self.portfolio["sector_exposure"]
    
    async def get_order_book(self):
        """Get the order book (simulated in paper trading)."""
        # In a real implementation, this would connect to a broker API
        # For paper trading, we'll simulate a simple order book
        
        # Get recent transactions
        recent_transactions = sorted(
            self.transaction_history[-10:] if len(self.transaction_history) > 10 else self.transaction_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return {
            "recent_transactions": recent_transactions,
            "open_orders": [],  # No open orders in this simple implementation
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_portfolio_state(self):
        """Get the current portfolio state."""
        # Update prices first
        await self.update_portfolio_prices()
        
        return self.portfolio
    
    async def rebalance_portfolio(self, target_allocation: Optional[Dict[str, float]] = None):
        """Rebalance portfolio to target allocation."""
        if target_allocation:
            self.portfolio["target_allocation"] = target_allocation
        
        target = self.portfolio["target_allocation"]
        current = self.portfolio["current_allocation"]
        total_value = self.portfolio["current_value"]
        
        # Calculate needed adjustments
        adjustments = []
        
        for sector, target_pct in target.items():
            current_pct = current.get(sector, 0)
            diff = target_pct - current_pct
            
            # Only make significant adjustments
            if abs(diff) > 0.02:  # 2% threshold
                direction = "increase" if diff > 0 else "decrease"
                adjustments.append({
                    "sector": sector,
                    "current_pct": current_pct,
                    "target_pct": target_pct,
                    "diff": diff,
                    "amount": abs(diff * total_value),
                    "direction": direction
                })
        
        # In a real implementation, this would generate actual trades
        # For this example, we'll just update the allocations directly
        
        # Simulate portfolio adjustment
        if adjustments:
            # Update current allocation to match target (simplified)
            self.portfolio["current_allocation"] = target_allocation or self.portfolio["target_allocation"]
            
            # Update sector exposure to match
            self.portfolio["sector_exposure"] = self.portfolio["current_allocation"]
            
            log(LogLevel.INFO, "Portfolio rebalanced", extra={"adjustments": adjustments})
        
        return {
            "adjustments": adjustments,
            "new_allocation": self.portfolio["current_allocation"]
        }

class MarketConnector:
    """Main connector for market data and trading APIs."""
    
    def __init__(self, mode="paper"):
        """Initialize the market connector."""
        self.mode = mode
        self.portfolio_manager = PortfolioManager(mode=mode)
        self.market_data = MarketDataProvider()
        log(LogLevel.INFO, f"Initialized MarketConnector in {mode} mode")
    
    async def connect(self):
        """Connect to market APIs."""
        # In a real implementation, this would establish connections
        # For this example, we'll just simulate connection
        log(LogLevel.INFO, "Connected to market APIs")
        return True
    
    async def get_market_state(self):
        """Get the current market state."""
        # Get market summary
        market_summary = await self.market_data.get_market_summary()
        
        # Get portfolio state
        portfolio = await self.portfolio_manager.get_portfolio_state()
        
        # Combined state
        state = {
            "portfolio": portfolio,
            "market_data": market_summary,
            "timestamp": datetime.now().isoformat()
        }
        
        return state
    
    async def execute_trade(self, action: str):
        """Execute a trade action."""
        log(LogLevel.INFO, f"Executing trade action: {action}")
        
        try:
            # Parse action string
            parts = action.split(":", 1)
            
            if len(parts) != 2:
                raise ValueError("Invalid action format. Expected 'action_type:json_data'")
            
            action_type = parts[0]
            action_data_json = parts[1]
            
            try:
                action_data = json.loads(action_data_json)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in the action string")
            
            # Execute based on action type
            if action_type == "buy" or action_type == "sell":
                order = {
                    "type": action_type,
                    "ticker": action_data.get("ticker"),
                    "quantity": action_data.get("quantity"),
                    "price": action_data.get("price", 0)
                }
                result = await self.portfolio_manager.execute_order(order)
            elif action_type == "rebalance":
                result = await self.portfolio_manager.rebalance_portfolio(
                    action_data.get("target_allocation")
                )
            elif action_type == "get_price":
                result = await self.market_data.get_price_data(
                    action_data.get("ticker"),
                    action_data.get("period", "1d"),
                    action_data.get("interval", "1m")
                )
            elif action_type == "get_company_info":
                result = await self.market_data.get_company_info(
                    action_data.get("ticker")
                )
            elif action_type == "get_portfolio":
                result = await self.portfolio_manager.get_portfolio_state()
            elif action_type == "get_market_summary":
                result = await self.market_data.get_market_summary()
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            # Get updated state
            state = await self.get_market_state()
            
            # Add the specific result to the state
            state["action_result"] = result
            
            return state
        except Exception as e:
            log(LogLevel.ERROR, f"Error executing trade action", extra={"error": str(e), "action": action})
            return {
                "error": str(e),
                "action": action
            }
    
    async def get_trading_tools(self):
        """Get available trading tools in a format suitable for Claude."""
        tools = [
            {
                "name": "buy_stock",
                "description": "Buy shares of a specific stock",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL)"},
                        "quantity": {"type": "number", "description": "Number of shares to buy"},
                        "price": {"type": "number", "description": "Optional limit price (omit for market order)"}
                    },
                    "required": ["ticker", "quantity"]
                }
            },
            {
                "name": "sell_stock",
                "description": "Sell shares of a specific stock",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL)"},
                        "quantity": {"type": "number", "description": "Number of shares to sell"},
                        "price": {"type": "number", "description": "Optional limit price (omit for market order)"}
                    },
                    "required": ["ticker", "quantity"]
                }
            },
            {
                "name": "get_price",
                "description": "Get current price and data for a stock",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL)"},
                        "period": {"type": "string", "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)"},
                        "interval": {"type": "string", "description": "Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d)"}
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "get_portfolio",
                "description": "Get current portfolio state including positions, values, and allocations",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_market_summary",
                "description": "Get overall market data including major indices and sector performance",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_company_info",
                "description": "Get detailed company information for a ticker",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL)"}
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "rebalance_portfolio",
                "description": "Rebalance portfolio to target allocation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_allocation": {
                            "type": "object",
                            "description": "Target allocation percentages by sector"
                        }
                    }
                }
            }
        ]
        
        return tools
    
    async def cleanup(self):
        """Clean up resources."""
        # In a real implementation, this would close connections
        log(LogLevel.INFO, "Market connector resources cleaned up")
        return True
    
    @staticmethod
    async def create_instance(mode="paper"):
        """Factory method to create a MarketInstance."""
        connector = MarketConnector(mode=mode)
        await connector.connect()
        
        # Get initial state
        state = await connector.get_market_state()
        
        # Create instance with reference to connector
        instance = MarketInstance(state, connector)
        
        return instance, connector