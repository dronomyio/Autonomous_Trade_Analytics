"""
Chat Interface for Trade Analytics
Provides a direct conversational interface to interact with the trading agent
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from anthropic import Anthropic
from market_connector import MarketConnector
from eva import log, LogLevel

class TradingChatInterface:
    """
    A chat interface for interacting with the trading agent directly
    through natural language conversations
    """
    
    def __init__(self, market_connector: MarketConnector, model_name="claude-3-7-sonnet-latest"):
        """Initialize the chat interface."""
        self.market_connector = market_connector
        self.anthropic = Anthropic()
        self.model_name = model_name
        self.chat_history = []
        self.system_prompt = """You are a financial advisor and trading assistant helping a user manage their investment portfolio.

You can:
1. Analyze market trends and provide investment insights
2. Explain technical concepts in simple terms
3. Make recommendations based on the portfolio and market data
4. Execute trades on behalf of the user when explicitly requested
5. Answer questions about portfolio performance

When the user asks you to make a trade:
- Only execute trades when there's a clear instruction (e.g., "buy 10 shares of AAPL")
- Always confirm the trade details before execution
- Provide reasoning for why the trade fits their strategy
- Report the results after execution

To execute trades, you'll use special trading tools. Do not invent trades that weren't executed.

Remember:
- You are dealing with real financial information, so be accurate and responsible
- Inform the user of both potential upside and downside risks
- Avoid giving overly speculative advice
- Always prioritize the user's stated investment goals and risk tolerance"""
        
        log(LogLevel.INFO, f"Initialized TradingChatInterface")
    
    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: The message from the user
            
        Returns:
            A dictionary with the assistant's response and any actions taken
        """
        log(LogLevel.INFO, f"Processing user message", extra={"message_length": len(user_message)})
        
        # Get current market and portfolio state
        market_state = await self.market_connector.get_market_state()
        
        # Format portfolio data for context
        portfolio = market_state.get("portfolio", {})
        portfolio_text = f"""
## Current Portfolio Information
- Portfolio Value: ${portfolio.get('current_value', 0):,.2f}
- Cash Balance: ${portfolio.get('cash', 0):,.2f}
- Current Positions: {len(portfolio.get('positions', {}))} assets
"""
        
        # Add current portfolio positions if available
        positions = portfolio.get("positions", {})
        if positions:
            portfolio_text += "\n### Positions\n"
            for ticker, position in positions.items():
                shares = position.get("shares", 0)
                avg_price = position.get("avg_price", 0)
                current_price = position.get("current_price", 0)
                value = shares * current_price
                pnl = (current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
                portfolio_text += f"- {ticker}: {shares} shares @ ${avg_price:.2f}, now ${current_price:.2f} ({pnl:+.2f}%), value: ${value:.2f}\n"
        
        # Add sector allocation if available
        if "sector_exposure" in portfolio:
            portfolio_text += "\n### Sector Allocation\n"
            for sector, allocation in portfolio.get("sector_exposure", {}).items():
                portfolio_text += f"- {sector}: {allocation*100:.1f}%\n"
        
        # Format market data
        market_data = market_state.get("market_data", {})
        market_text = """
## Current Market Summary
"""
        # Add major indices
        if "indices" in market_data:
            market_text += "\n### Major Indices\n"
            for index_name, index_data in market_data.get("indices", {}).items():
                change = index_data.get("change_pct", 0) * 100
                market_text += f"- {index_name}: {change:+.2f}%\n"
        
        # Add sector performance
        if "sectors" in market_data:
            market_text += "\n### Sector Performance\n"
            for sector_name, sector_data in market_data.get("sectors", {}).items():
                change = sector_data.get("change_pct", 0) * 100
                market_text += f"- {sector_name}: {change:+.2f}%\n"
        
        # Add this context to the first message if it's a new conversation
        if not self.chat_history:
            context_message = {
                "role": "user", 
                "content": f"Here is your current portfolio and market information for reference:\n{portfolio_text}\n{market_text}\n\nPlease help me manage my investments."
            }
            self.chat_history.append(context_message)
            
            # Add assistant acknowledgment
            intro_response = {
                "role": "assistant",
                "content": "I'll help you manage your investments. I have access to your portfolio data and current market information. What would you like to do today? I can analyze your portfolio, provide market insights, or help you execute trades."
            }
            self.chat_history.append(intro_response)
        
        # Add user message to history
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Get response from Claude
        try:
            response = await self._get_claude_response(market_state)
            
            # Check if the response includes a trade action
            trade_action = self._extract_trade_action(response)
            
            # If there's a trade action, execute it
            trade_result = None
            if trade_action:
                trade_result = await self._execute_trade(trade_action)
                
                # Add the trade result to the chat history
                result_message = {
                    "role": "user",
                    "content": f"Trade executed: {trade_result.get('message', 'No details available')}\n\nUpdated portfolio value: ${portfolio.get('current_value', 0):,.2f}"
                }
                self.chat_history.append(result_message)
                
                # Get a follow-up response with the trade result
                response = await self._get_claude_response(market_state)
            
            return {
                "response": response,
                "trade_executed": trade_action is not None,
                "trade_result": trade_result
            }
            
        except Exception as e:
            log(LogLevel.ERROR, f"Error getting response", extra={"error": str(e)})
            error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            self.chat_history.append({"role": "assistant", "content": error_response})
            return {"response": error_response, "error": str(e)}
    
    async def _get_claude_response(self, market_state: Dict[str, Any]) -> str:
        """Get a response from Claude based on the conversation history."""
        tools = await self.market_connector.get_trading_tools()
        
        response = self.anthropic.messages.create(
            model=self.model_name,
            max_tokens=2000,
            system=self.system_prompt,
            messages=self.chat_history,
            tools=tools,
            temperature=0.7
        )
        
        # Extract text response
        text_content = " ".join([block.text for block in response.content if block.type == "text"])
        
        # Add response to history
        self.chat_history.append({"role": "assistant", "content": text_content})
        
        return text_content
    
    def _extract_trade_action(self, response: str) -> Optional[str]:
        """Extract trade action from Claude's response if one exists."""
        # Look for tool use in the last assistant message
        if not self.chat_history:
            return None
            
        last_message = self.chat_history[-1]
        if last_message["role"] != "assistant" or not isinstance(last_message["content"], list):
            return None
        
        # Find tool use blocks
        for content_item in last_message["content"]:
            if isinstance(content_item, dict) and content_item.get("type") == "tool_use":
                tool_name = content_item.get("name")
                tool_input = content_item.get("input")
                
                # Convert to action string
                if tool_name in ["buy_stock", "sell_stock", "rebalance_portfolio"]:
                    if tool_name == "buy_stock":
                        return f"buy:{json.dumps(tool_input)}"
                    elif tool_name == "sell_stock":
                        return f"sell:{json.dumps(tool_input)}"
                    else:
                        return f"rebalance:{json.dumps(tool_input)}"
        
        return None
    
    async def _execute_trade(self, action: str) -> Dict[str, Any]:
        """Execute a trade action and return the result."""
        log(LogLevel.INFO, f"Executing trade", extra={"action": action})
        
        try:
            # Execute the trade through the market connector
            result = await self.market_connector.execute_trade(action)
            
            # Check for errors
            if "error" in result:
                log(LogLevel.ERROR, f"Trade execution failed", 
                    extra={"error": result["error"]})
                return {
                    "success": False,
                    "message": f"Trade failed: {result['error']}"
                }
            
            # Extract action result
            action_result = result.get("action_result", {})
            
            # Format a success message
            if isinstance(action_result, dict):
                if "status" in action_result and action_result["status"] == "completed":
                    message = action_result.get("message", "Trade completed successfully")
                    return {
                        "success": True,
                        "message": message,
                        "details": action_result
                    }
            
            # Generic success
            return {
                "success": True,
                "message": "Trade executed successfully",
                "details": action_result
            }
            
        except Exception as e:
            log(LogLevel.ERROR, f"Error executing trade", extra={"error": str(e)})
            return {
                "success": False,
                "message": f"Error executing trade: {str(e)}"
            }
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
        log(LogLevel.INFO, "Chat history cleared")