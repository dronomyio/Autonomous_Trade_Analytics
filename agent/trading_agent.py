"""
Trading Agent Implementation

This module implements a Claude-powered trading agent that can make decisions
and execute trades based on market data and portfolio state.
"""

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Sequence, Union, TypeVar, Generic, Tuple

from anthropic import Anthropic
import pandas as pd
import numpy as np

# Import from the main E.V.A. framework
from eva import (
    Instance, VerificationResult, VerifiedTask, Agent, Trajectory, TrajectoryStep,
    MarketInstance, log, LogLevel
)

# Import local modules
from market_connector import MarketConnector

# Define standard event types
EVENT_SNAPSHOT_CREATED = "snapshot_created"
EVENT_ACTION_STARTED = "action_started"
EVENT_ACTION_COMPLETED = "action_completed"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_CLAUDE_TEXT = "claude_text"
EVENT_TRAJECTORY_UPDATED = "trajectory_updated"

@dataclass
class TradingTrajectoryStep(TrajectoryStep[Dict[str, Any], str, bool, str]):
    """A specialized trajectory step for trading with extended data."""
    
    # Add trading-specific fields
    portfolio_value: Optional[float] = None  # Portfolio value at this step
    cash_balance: Optional[float] = None  # Cash balance at this step
    action_type: Optional[str] = None  # Type of action (buy, sell, etc.)
    ticker: Optional[str] = None  # Ticker symbol if applicable
    quantity: Optional[int] = None  # Quantity if applicable
    price: Optional[float] = None  # Price if applicable
    reasoning: Optional[str] = None  # Claude's reasoning before the action
    
    def __post_init__(self):
        """Process after initialization."""
        # Extract portfolio value and cash balance from state if not provided
        if not self.portfolio_value and self.state and hasattr(self.state, 'state'):
            portfolio = self.state.state.get('portfolio', {})
            self.portfolio_value = portfolio.get('current_value')
            self.cash_balance = portfolio.get('cash')
        
        # Parse action string if provided
        if self.action and not self.action_type:
            try:
                parts = self.action.split(":", 1)
                if len(parts) == 2:
                    self.action_type = parts[0]
                    action_data = json.loads(parts[1])
                    
                    if self.action_type in ['buy', 'sell']:
                        self.ticker = action_data.get('ticker')
                        self.quantity = action_data.get('quantity')
                        self.price = action_data.get('price')
            except:
                pass  # Ignore parsing errors
        
        # Log step creation
        action_str = f" -> {self.action}" if self.action else " (initial)"
        log(LogLevel.DEBUG, f"Trading trajectory step{action_str}", 
            extra={
                "event_type": "trading_step_created",
                "step_type": "initial" if self.action is None else "action",
                "portfolio_value": self.portfolio_value,
                "action_type": self.action_type
            })

class TradingTrajectory(Trajectory[Dict[str, Any], str, bool, str]):
    """A specialized trajectory for trading with additional metadata."""
    
    def __init__(self):
        """Initialize a trading trajectory."""
        super().__init__()
        # Override steps with trading-specific step type
        self.steps: List[TradingTrajectoryStep] = []
    
    def add_step(self, state: Instance[Dict[str, Any], str], 
                 action: Optional[str] = None,
                 result: Optional[VerificationResult[bool]] = None,
                 reasoning: Optional[str] = None) -> None:
        """Add a trading-specific step to the trajectory."""
        # Extract portfolio data from state
        portfolio = state.state.get('portfolio', {})
        portfolio_value = portfolio.get('current_value')
        cash_balance = portfolio.get('cash')
        
        # Parse action details if present
        action_type = None
        ticker = None
        quantity = None
        price = None
        
        if action:
            try:
                parts = action.split(":", 1)
                if len(parts) == 2:
                    action_type = parts[0]
                    action_data = json.loads(parts[1])
                    
                    if action_type in ['buy', 'sell']:
                        ticker = action_data.get('ticker')
                        quantity = action_data.get('quantity')
                        price = action_data.get('price')
            except:
                pass  # Ignore parsing errors
        
        # Get the current step index
        current_step_index = len(self.steps)
        
        # Create snapshot ID (in a real app, this might be a database ID)
        snapshot_id = state.snapshot()
        
        # Create specialized step
        step = TradingTrajectoryStep(
            state=state,
            snapshot=snapshot_id,
            action=action,
            result=result,
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            cash_balance=cash_balance,
            action_type=action_type,
            ticker=ticker,
            quantity=quantity,
            price=price,
            reasoning=reasoning
        )
        
        # Add step to trajectory
        self.steps.append(step)
        
        # Log with standardized event
        if len(self.steps) == 1:
            log(LogLevel.INFO, "Started new trading trajectory", 
                extra={
                    "event_type": "trajectory_started", 
                    "step_index": 0
                })
        else:
            action_str = f"{action}" if action else "None"
            log(LogLevel.INFO, f"Step {len(self.steps)-1}: Action={action_str}", 
                extra={
                    "event_type": "step_added", 
                    "step_index": len(self.steps)-1, 
                    "action": action_str,
                    "portfolio_value": portfolio_value,
                    "snapshot_id": snapshot_id
                })
        
        # Emit trajectory updated event
        log(LogLevel.INFO, "Trajectory updated", 
            extra={
                "event_type": EVENT_TRAJECTORY_UPDATED,
                "step_count": len(self.steps),
                "latest_step_index": len(self.steps) - 1
            })
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the trajectory to a dictionary."""
        serialized = {
            "steps": [],
            "portfolio_history": [],
            "actions": []
        }
        
        for i, step in enumerate(self.steps):
            # Basic step data
            step_data = {
                "index": i,
                "timestamp": step.timestamp.isoformat(),
                "action": step.action if step.action else None,
                "portfolio_value": step.portfolio_value,
                "cash_balance": step.cash_balance,
                "snapshot_id": step.snapshot,
            }
            
            # Add action details if available
            if step.action_type:
                step_data["action_details"] = {
                    "type": step.action_type,
                    "ticker": step.ticker,
                    "quantity": step.quantity,
                    "price": step.price
                }
            
            # Add Claude's reasoning if available
            if step.reasoning:
                step_data["reasoning"] = step.reasoning
            
            # Add result information if available
            if step.result:
                step_data["result"] = {
                    "success": step.result.success,
                    "message": step.result.message
                }
            
            serialized["steps"].append(step_data)
            
            # Add to portfolio history for charting
            if step.portfolio_value:
                serialized["portfolio_history"].append({
                    "timestamp": step.timestamp.isoformat(),
                    "value": step.portfolio_value
                })
            
            # Add to actions list for timeline view
            if step.action:
                action_entry = {
                    "index": i,
                    "timestamp": step.timestamp.isoformat(),
                    "action": step.action,
                    "action_type": step.action_type,
                    "details": {}
                }
                
                if step.ticker:
                    action_entry["details"]["ticker"] = step.ticker
                if step.quantity:
                    action_entry["details"]["quantity"] = step.quantity
                if step.price:
                    action_entry["details"]["price"] = step.price
                
                serialized["actions"].append(action_entry)
        
        return serialized

class TradingAgent(Agent[Dict[str, Any], str, bool, str]):
    """An agent that makes trading decisions using the Claude API and tracks its trajectory."""
    
    def __init__(self, market_connector: MarketConnector, model_name="claude-3-7-sonnet-latest", max_tokens=4000, max_history=30):
        """Initialize the trading agent."""
        super().__init__()
        self.market_connector = market_connector
        self.anthropic = Anthropic()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = 0.7
        self.message_history = []
        self.max_history = max_history
        
        # Specialized trajectory
        self.trajectory = TradingTrajectory()
        
        # Current reasoning for trajectory
        self.current_reasoning = None
        
        # For pause and resume functionality
        self.paused = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused
        
        # System prompt for Claude
        self.system_prompt = """You are an intelligent trading agent that analyzes market data and makes trading decisions. Your goal is to optimize portfolio performance according to the given objective.

Key Trading Principles:
1. Risk Management: Never risk more than 2% of the portfolio on a single trade
2. Diversification: Maintain exposure across multiple sectors
3. Cost Awareness: Consider transaction costs in your decisions
4. Trend Analysis: Consider both short and long-term trends
5. Fundamental Analysis: Consider company fundamentals when available

Before each action, explain your reasoning briefly, then use the available trading tools to execute your strategy. Always justify your decisions with data-driven analysis."""
        
        log(LogLevel.INFO, f"Initialized TradingAgent", 
            extra={"model": model_name, "max_tokens": max_tokens})
    
    def set_objective(self, objective: str):
        """Set the agent's current objective."""
        self.objective = objective
        log(LogLevel.INFO, f"Setting agent objective", extra={"objective": objective})
    
    async def initialize_state(self, snapshot_id: str = None) -> MarketInstance:
        """Initialize the state using the market connector."""
        log(LogLevel.INFO, "Initializing trading agent state")
        
        # Get initial state from the market connector
        instance, _ = await MarketConnector.create_instance(mode="paper")
        
        # Add a starting message to the history
        initial_message = f"Your current objective is: {self.objective}\n\nYou may now begin analyzing the market and making trading decisions."
        self.message_history = [{"role": "user", "content": initial_message}]
        
        # Initialize the trajectory with the first step
        self.trajectory.add_step(instance)
        
        log(LogLevel.INFO, "Initial state and trajectory created")
        return instance
    
    async def wait_if_paused(self):
        """Wait if the agent is paused."""
        await self.pause_event.wait()
    
    def pause(self):
        """Pause the agent's execution."""
        self.paused = True
        self.pause_event.clear()
        log(LogLevel.INFO, "Agent paused", extra={"event_type": "agent_paused"})
    
    def resume(self):
        """Resume the agent's execution."""
        self.paused = False
        self.pause_event.set()
        log(LogLevel.INFO, "Agent resumed", extra={"event_type": "agent_resumed"})
    
    async def run_step(self, state: Instance[Dict[str, Any], str]) -> str:
        """Determine the next action using Claude."""
        log(LogLevel.INFO, "Determining next trading action with Claude")
        
        # Wait if paused
        await self.wait_if_paused()
        
        # Create user message with market data and portfolio state
        user_content = []
        
        # Add text description
        user_content.append({
            "type": "text",
            "text": "Here is the current market and portfolio state. Please analyze and decide your next action."
        })
        
        # Format portfolio data
        portfolio = state.state.get("portfolio", {})
        portfolio_text = f"""
## Portfolio Summary
- Current Value: ${portfolio.get('current_value', 0):,.2f}
- Cash Balance: ${portfolio.get('cash', 0):,.2f}

### Current Positions
"""
        positions = portfolio.get("positions", {})
        for ticker, position in positions.items():
            current_price = position.get("current_price", 0)
            avg_price = position.get("avg_price", 0)
            shares = position.get("shares", 0)
            position_value = current_price * shares
            pnl_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
            
            portfolio_text += f"- {ticker}: {shares} shares @ avg ${avg_price:.2f}, current: ${current_price:.2f}, value: ${position_value:.2f}, P&L: {pnl_pct:.2f}%\n"
        
        portfolio_text += """
### Sector Allocation
"""
        for sector, allocation in portfolio.get("current_allocation", {}).items():
            target = portfolio.get("target_allocation", {}).get(sector, 0)
            diff = allocation - target
            portfolio_text += f"- {sector}: {allocation*100:.1f}% (target: {target*100:.1f}%, diff: {diff*100:+.1f}%)\n"
        
        portfolio_text += """
### Risk Metrics
"""
        risk_metrics = portfolio.get("risk_metrics", {})
        for metric, value in risk_metrics.items():
            portfolio_text += f"- {metric}: {value}\n"
        
        user_content.append({"type": "text", "text": portfolio_text})
        
        # Format market data
        market_data = state.state.get("market_data", {})
        market_text = """
## Market Summary
"""
        # Add indices
        market_text += "### Major Indices\n"
        for index_name, index_data in market_data.get("indices", {}).items():
            change = index_data.get("change_pct", 0) * 100
            market_text += f"- {index_name}: {change:+.2f}%\n"
        
        # Add sectors
        market_text += "\n### Sector Performance\n"
        for sector, data in market_data.get("sectors", {}).items():
            change = data.get("change_pct", 0) * 100
            market_text += f"- {sector}: {change:+.2f}%\n"
        
        user_content.append({"type": "text", "text": market_text})
        
        # Add the message to history
        self.message_history.append({"role": "user", "content": user_content})
        
        # Get action with retries if needed
        action, success, reasoning = await self._retry_with_nudge(max_retries=3)
        
        if not success:
            log(LogLevel.ERROR, "Failed to get valid tool call after retries, using fallback action")
        
        # Store reasoning for trajectory
        self.current_reasoning = reasoning
        
        return action
    
    async def _retry_with_nudge(self, max_retries=3):
        """Retry getting a tool call from Claude with nudges."""
        attempts = 0
        tool_calls = []
        reasoning = ""
        
        while attempts < max_retries and not tool_calls:
            attempts += 1
            
            # If this is a retry, add a nudge message
            if attempts > 1:
                nudge_message = {
                    "role": "user", 
                    "content": f"Please make a decision and use one of the available trading tools. This is attempt {attempts} of {max_retries}."
                }
                self.message_history.append(nudge_message)
                log(LogLevel.WARNING, f"No tool calls found, adding nudge (attempt {attempts}/{max_retries})")
            
            # Create a copy of message history for cache control
            messages = copy.deepcopy(self.message_history)
            
            # Wait if paused before making Claude API call
            await self.wait_if_paused()
            
            # Get Claude's response
            response = self.anthropic.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages,
                tools=await self.market_connector.get_trading_tools(),
                temperature=self.temperature
            )
            
            # Extract Claude's text reasoning
            reasoning = " ".join([block.text for block in response.content if block.type == "text"])
            
            # Log Claude's text with standardized event
            if reasoning:
                log(LogLevel.INFO, f"Claude reasoning text received", 
                    extra={"event_type": EVENT_CLAUDE_TEXT, "text_length": len(reasoning)})
            
            # Extract tool calls
            tool_calls = [
                block for block in response.content if block.type == "tool_use"
            ]
            
            # Add Claude's response to history
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({"type": "tool_use", **dict(block)})
                    log(LogLevel.DEBUG, f"Found tool call", 
                        extra={"tool": block.name, "input": json.dumps(block.input)})
            
            self.message_history.append({"role": "assistant", "content": assistant_content})
            
            # If we found tool calls, break out of the loop
            if tool_calls:
                break
        
        # After max retries or successful tool call
        if tool_calls:
            # Extract the first tool call for action
            tool_call = tool_calls[0]
            tool_name = tool_call.name
            tool_input = tool_call.input
            
            # Log tool call with standardized event
            log(LogLevel.INFO, f"Claude tool use", 
                extra={"event_type": EVENT_TOOL_CALL, "tool_name": tool_name})
            
            # Convert to action string format
            if tool_name == "buy_stock":
                action = f"buy:{json.dumps(tool_input)}"
            elif tool_name == "sell_stock":
                action = f"sell:{json.dumps(tool_input)}"
            elif tool_name == "get_price":
                action = f"get_price:{json.dumps(tool_input)}"
            elif tool_name == "get_portfolio":
                action = f"get_portfolio:{json.dumps({})}"
            elif tool_name == "get_market_summary":
                action = f"get_market_summary:{json.dumps({})}"
            elif tool_name == "get_company_info":
                action = f"get_company_info:{json.dumps(tool_input)}"
            elif tool_name == "rebalance_portfolio":
                action = f"rebalance:{json.dumps(tool_input)}"
            else:
                action = f"{tool_name}:{json.dumps(tool_input)}"
            
            return action, True, reasoning
        else:
            # If we still don't have tool calls after max retries
            log(LogLevel.ERROR, f"No tool calls after {max_retries} attempts")
            return "get_portfolio:{}", False, reasoning  # Return fallback and False for success
    
    async def apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply an action and return the new state."""
        log(LogLevel.INFO, f"Applying action", 
            extra={"event_type": EVENT_ACTION_STARTED, "action": action})
        
        # Wait if paused
        await self.wait_if_paused()
        
        # Execute the action
        action_result = await self.market_connector.execute_trade(action)
        
        # Check for errors
        if "error" in action_result:
            log(LogLevel.ERROR, f"Error executing action", 
                extra={"error": action_result["error"], "action": action})
        
        # Create a new state (in a real app, this would get the updated market state)
        new_state = action_result
        
        # Log tool result with standardized event
        log(LogLevel.INFO, f"Trade executed", 
            extra={
                "event_type": EVENT_TOOL_RESULT, 
                "action": action,
                "portfolio_value": new_state.get("portfolio", {}).get("current_value")
            })
        
        # Create tool results from the action for Claude history
        tool_results = []
        
        # Extract most recent assistant message to get tool ID
        if self.message_history and self.message_history[-1]["role"] == "assistant":
            assistant_content = self.message_history[-1]["content"]
            tool_use_items = [item for item in assistant_content if isinstance(item, dict) and item.get("type") == "tool_use"]
            
            if tool_use_items:
                tool_use_id = tool_use_items[0].get("id")
                
                if tool_use_id:
                    # Create result content
                    result_content = []
                    
                    # Format the result based on action type
                    action_type = action.split(":", 1)[0] if ":" in action else action
                    
                    if action_type == "buy" or action_type == "sell":
                        result = action_result.get("action_result", {})
                        status = result.get("status", "unknown")
                        message = result.get("message", "No message")
                        
                        result_text = f"Trade {status}: {message}"
                        
                        # Add portfolio summary
                        portfolio = new_state.get("portfolio", {})
                        result_text += f"\n\nUpdated Portfolio Value: ${portfolio.get('current_value', 0):,.2f}"
                        result_text += f"\nCash Balance: ${portfolio.get('cash', 0):,.2f}"
                        
                    elif action_type == "get_price":
                        result = action_result.get("action_result", {})
                        ticker = result.get("ticker", "unknown")
                        price = result.get("last_price", 0)
                        change = result.get("change_pct", 0) * 100
                        
                        result_text = f"{ticker} Price: ${price:.2f} ({change:+.2f}%)"
                        result_text += f"\nPeriod: {result.get('period', '1d')}, Interval: {result.get('interval', '1m')}"
                        result_text += f"\nHigh: ${result.get('high', 0):.2f}, Low: ${result.get('low', 0):.2f}"
                        result_text += f"\nVolume: {result.get('volume', 0):,}"
                        
                    elif action_type == "get_company_info":
                        result = action_result.get("action_result", {})
                        ticker = result.get("ticker", "unknown")
                        name = result.get("name", "Unknown Company")
                        
                        result_text = f"{name} ({ticker})"
                        result_text += f"\nSector: {result.get('sector', 'Unknown')}"
                        result_text += f"\nIndustry: {result.get('industry', 'Unknown')}"
                        result_text += f"\nMarket Cap: ${result.get('market_cap', 0):,.0f}"
                        result_text += f"\nP/E Ratio: {result.get('pe_ratio', 'N/A')}"
                        result_text += f"\nDividend Yield: {result.get('dividend_yield', 0):.2f}%"
                        result_text += f"\n52-Week Range: ${result.get('52w_low', 0):.2f} - ${result.get('52w_high', 0):.2f}"
                        
                    elif action_type == "rebalance":
                        result = action_result.get("action_result", {})
                        adjustments = result.get("adjustments", [])
                        
                        result_text = "Portfolio Rebalancing Result:\n"
                        if adjustments:
                            for adj in adjustments:
                                sector = adj.get("sector", "unknown")
                                direction = adj.get("direction", "unknown")
                                amount = adj.get("amount", 0)
                                result_text += f"\n- {sector}: {direction} by ${amount:,.2f}"
                        else:
                            result_text += "\nNo significant adjustments needed."
                            
                        result_text += "\n\nNew Allocation:"
                        for sector, alloc in result.get("new_allocation", {}).items():
                            result_text += f"\n- {sector}: {alloc*100:.1f}%"
                    
                    else:
                        # Generic result formatting
                        result_text = f"Action '{action}' executed."
                        if "action_result" in action_result:
                            result_text += f"\nResult: {json.dumps(action_result['action_result'], indent=2)}"
                    
                    result_content.append({"type": "text", "text": result_text})
                    
                    # Create a proper tool result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_content
                    })
                    
                    # Add the tool results to message history
                    self.message_history.append({"role": "user", "content": tool_results})
        
        # Create MarketInstance from the new state
        instance = MarketInstance(new_state, self.market_connector)
        
        # Add step to trajectory with the current reasoning
        self.trajectory.add_step(
            state=instance,
            action=action,
            reasoning=self.current_reasoning
        )
        
        # Reset current reasoning
        self.current_reasoning = None
        
        # Log action completion
        log(LogLevel.INFO, f"Action completed", 
            extra={"event_type": EVENT_ACTION_COMPLETED, "action": action})
        
        return new_state
    
    async def summarize_history(self):
        """Summarize the conversation history to save context space."""
        log(LogLevel.INFO, "Summarizing conversation history")
        
        # Create summary prompt
        summary_prompt = """I need you to create a detailed summary of our trading conversation history up to this point. This summary will replace the full conversation history to manage the context window.

        Please include:
        1. Key trading decisions you've made
        2. Current portfolio status and positions
        3. Market insights you've identified
        4. Your current strategy and objectives
        5. Any important market trends or data points

        The summary should be comprehensive enough that you can continue trading without losing important context about what has happened so far."""
        
        # Add the summary prompt to message history
        messages = copy.deepcopy(self.message_history)
        messages.append({
            "role": "user",
            "content": summary_prompt,
        })
        
        # Get summary from Claude
        response = self.anthropic.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=messages,
            temperature=0.7
        )
        
        # Extract the summary text
        summary_text = " ".join([block.text for block in response.content if block.type == "text"])
        
        if not summary_text:
            log(LogLevel.WARNING, "Failed to generate summary, keeping message history")
            return
            
        log(LogLevel.INFO, f"Trading History Summary Generated", 
            extra={"summary_length": len(summary_text)})
        
        # Create summary content
        summary_content = []
        summary_content.append({
            "type": "text",
            "text": f"CONVERSATION HISTORY SUMMARY (representing {len(self.message_history)} previous messages): {summary_text}"
        })
        
        # Add continuation prompt
        summary_content.append({
            "type": "text",
            "text": "You may now continue trading by selecting your next action."
        })
        
        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": summary_content
            }
        ]
        log(LogLevel.INFO, "Message history condensed into summary")