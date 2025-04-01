#!/usr/bin/env python3
"""
Trade Agent Runner
Main entry point for the trading agent system
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Load environment variables
import load_env

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import framework components
from eva import log, LogLevel, run
from market_connector import MarketConnector
from agent.trading_agent import TradingAgent
from agent.chat_interface import TradingChatInterface
from tasks import create_trading_verified_task, get_task_by_id, REGISTERED_TASKS


# === API Models ===

class TaskRequest(BaseModel):
    """Request to start a task."""
    task_id: str
    steps: int = 100
    objective: Optional[str] = None

class RollbackRequest(BaseModel):
    """Request to roll back to a specific step."""
    step_index: int
    
class ChatMessage(BaseModel):
    """A chat message from the user."""
    message: str


# === Trading API ===

class TradingAPI:
    """API for the trading agent system."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """Initialize the API."""
        self.host = host
        self.port = port
        self.app = FastAPI(title="Trading Agent API")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.agent = None
        self.task = None
        self.market_connector = None
        self.chat_interface = None
        self.running_task = None
        
        # Register routes
        self.register_routes()
    
    def register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {"message": "Trading Agent API"}
        
        @self.app.get("/tasks")
        async def list_tasks():
            """Return a list of all registered tasks."""
            return [
                {
                    "id": task.id,
                    "instruction": task.instruction,
                    "verification_fn_name": task.verification_fn_name,
                    "verification_message": task.verification_message,
                    "metadata": task.metadata,
                }
                for task in REGISTERED_TASKS
            ]

        @self.app.get("/status")
        async def get_status():
            """Get agent status."""
            if not self.agent:
                return {"status": "not_initialized"}
            
            return {
                "status": "paused" if self.agent.paused else "running",
                "step_count": len(self.agent.trajectory.steps) if self.agent.trajectory else 0,
                "objective": getattr(self.agent, 'objective', None)
            }
        
        @self.app.get("/trajectory")
        async def get_trajectory():
            """Get the current trajectory."""
            if not self.agent or not self.agent.trajectory:
                return {"steps": [], "portfolio_history": [], "actions": []}
            
            return self.agent.trajectory.serialize()
        
        @self.app.post("/start")
        async def start_agent(request: TaskRequest, background_tasks: BackgroundTasks):
            """Start the agent with a task."""
            if self.running_task:
                return {"success": False, "error": "Agent is already running"}
            
            try:
                task_def = get_task_by_id(request.task_id)
                if not task_def:
                    return {"success": False, "error": f"Unknown task ID: {request.task_id}"}
                
                # Set custom objective if provided
                objective = request.objective or task_def.instruction
                
                background_tasks.add_task(
                    self.run_agent_task, 
                    request.task_id,
                    request.steps,
                    objective
                )

                return {"success": True, "message": "Agent started"}
            except Exception as e:
                log(LogLevel.ERROR, f"Failed to start agent", extra={"error": str(e)})
                return {"success": False, "error": str(e)}
        
        @self.app.post("/pause")
        async def pause_agent():
            """Pause the agent."""
            if not self.agent:
                return {"success": False, "error": "Agent not initialized"}
            
            if self.agent.paused:
                return {"success": True, "message": "Agent already paused"}
            
            self.agent.pause()
            return {"success": True, "message": "Agent paused"}
        
        @self.app.post("/resume")
        async def resume_agent():
            """Resume the agent."""
            if not self.agent:
                return {"success": False, "error": "Agent not initialized"}
            
            if not self.agent.paused:
                return {"success": True, "message": "Agent already running"}
            
            self.agent.resume()
            return {"success": True, "message": "Agent resumed"}
        
        @self.app.post("/rollback")
        async def rollback(request: RollbackRequest):
            """Roll back to a specific step."""
            if not self.agent or not self.agent.trajectory:
                return {"success": False, "error": "Agent not initialized"}
            
            if request.step_index < 0 or request.step_index >= len(self.agent.trajectory.steps):
                return {"success": False, "error": "Invalid step index"}
            
            # Not implemented in this example
            return {"success": False, "error": "Rollback not implemented in this example"}
        
        @self.app.post("/stop")
        async def stop_agent():
            """Stop the agent."""
            if not self.agent:
                return {"success": False, "error": "Agent not initialized"}
            
            try:
                # Pause first
                self.agent.pause()
                
                # Clean up resources
                if self.market_connector:
                    await self.market_connector.cleanup()
                    self.market_connector = None
                
                self.agent = None
                self.task = None
                self.running_task = None
                
                return {"success": True, "message": "Agent stopped"}
            except Exception as e:
                log(LogLevel.ERROR, f"Failed to stop agent", extra={"error": str(e)})
                return {"success": False, "error": str(e)}
                
        # === Chat Interface Routes ===
        
        @self.app.get("/chat")
        async def chat_ui(request: Request):
            """Render the chat UI."""
            from fastapi.templating import Jinja2Templates
            templates = Jinja2Templates(directory="templates")
            
            return templates.TemplateResponse(
                "chat.html", 
                {"request": request, "api_url": f"http://{self.host}:{self.port}"}
            )
            
        @self.app.get("/chat/history")
        async def get_chat_history():
            """Get chat history."""
            if not self.chat_interface:
                await self._init_chat_interface()
                
            return {
                "messages": self.chat_interface.chat_history
            }
            
        @self.app.post("/chat/message")
        async def send_chat_message(message: ChatMessage):
            """Send a message to the chat interface."""
            if not self.chat_interface:
                await self._init_chat_interface()
                
            try:
                result = await self.chat_interface.process_message(message.message)
                return result
            except Exception as e:
                log(LogLevel.ERROR, f"Error processing chat message", extra={"error": str(e)})
                return {"error": str(e)}
                
        @self.app.post("/chat/clear")
        async def clear_chat_history():
            """Clear chat history."""
            if self.chat_interface:
                self.chat_interface.clear_history()
                
            return {"success": True}
    
    async def _init_chat_interface(self):
        """Initialize the chat interface if not already initialized."""
        if not self.market_connector:
            # Create market connector
            self.market_connector = MarketConnector(mode="paper")
            await self.market_connector.connect()
            log(LogLevel.INFO, "Initialized market connector for chat interface")
        
        if not self.chat_interface:
            # Create chat interface
            self.chat_interface = TradingChatInterface(market_connector=self.market_connector)
            log(LogLevel.INFO, "Initialized chat interface")
    
    async def run_agent_task(self, task_id: str, steps: int, objective: str):
        """Run an agent task."""
        try:
            # Create the task
            self.task = create_trading_verified_task(task_id)
            
            # Create market connector if not already initialized
            if not self.market_connector:
                self.market_connector = MarketConnector(mode="paper")
                await self.market_connector.connect()
            
            # Create the agent
            self.agent = TradingAgent(market_connector=self.market_connector)
            self.agent.set_objective(objective)
            
            # Mark task as running
            self.running_task = task_id
            
            # Run the agent using E.V.A.
            log(LogLevel.INFO, f"Starting E.V.A. run for task: {objective}")
            result, trajectory = await run(
                task=self.task,
                agent=self.agent,
                max_steps=steps,
                verify_every_step=True,
                ttl_seconds=3600  # 1 hour TTL
            )
            
            log(LogLevel.INFO, f"Agent run complete", 
                extra={"success": result.success, "message": result.message})
            
            # Clear running task
            self.running_task = None
            
        except Exception as e:
            log(LogLevel.ERROR, f"Error in run_agent_task", extra={"error": str(e)})
            import traceback
            log(LogLevel.ERROR, f"Traceback", extra={"traceback": traceback.format_exc()})
            self.running_task = None
    
    def start(self):
        """Start the API server."""
        uvicorn.run(self.app, host=self.host, port=self.port)


async def run_trading_agent(task_id: str, steps: int = 100, objective: Optional[str] = None):
    """Run the trading agent directly without API."""
    log(LogLevel.INFO, f"Starting trading agent directly", 
        extra={"task_id": task_id, "steps": steps})
    
    try:
        # Create the task
        task_def = get_task_by_id(task_id)
        if not task_def:
            log(LogLevel.ERROR, f"Unknown task ID: {task_id}")
            return
        
        log(LogLevel.INFO, f"Running task: {task_def.instruction}")
        task = create_trading_verified_task(task_id)
        
        # Create market connector
        market_connector = MarketConnector(mode="paper")
        await market_connector.connect()
        
        # Create the agent
        agent = TradingAgent(market_connector=market_connector)
        agent.set_objective(objective or task_def.instruction)
        
        # Run the agent
        log(LogLevel.INFO, f"Starting run for task: {task_def.instruction}")
        result, trajectory = await run(
            task=task,
            agent=agent,
            max_steps=steps,
            verify_every_step=True,
            ttl_seconds=3600  # 1 hour TTL
        )
        
        log(LogLevel.INFO, f"Agent run complete", 
            extra={"success": result.success, "message": result.message})
        
        # Clean up
        await market_connector.cleanup()
        
    except Exception as e:
        log(LogLevel.ERROR, f"Error running agent: {e}")
        import traceback
        log(LogLevel.ERROR, f"Error traceback", 
            extra={"traceback": traceback.format_exc()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trading agent')
    parser.add_argument('--task', type=str, default="rebalance-portfolio", 
                        help='ID of the task to run')
    parser.add_argument('--steps', type=int, default=10, 
                        help='Maximum number of steps to run')
    parser.add_argument('--mode', type=str, choices=['api', 'direct'], default='api',
                        help='Run mode: api or direct')
    parser.add_argument('--port', type=int, default=8000, 
                        help='API server port (if running in API mode)')
    parser.add_argument('--objective', type=str, 
                        help='Custom objective (overrides task instruction)')
    
    args = parser.parse_args()
    
    if args.mode == 'direct':
        # Run directly
        asyncio.run(run_trading_agent(args.task, args.steps, args.objective))
    else:
        # Start API server
        api = TradingAPI(port=args.port)
        print(f"Starting Trading Agent API on port {args.port}...")
        api.start()