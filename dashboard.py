#!/usr/bin/env python3
"""
Trading Dashboard
A simple web UI for monitoring and controlling the trading agent
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
import json
import requests
from typing import Dict, Any, List, Optional

# Load environment variables
import load_env

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# === Dashboard App ===

class TradingDashboard:
    """Web dashboard for the trading agent."""
    
    def __init__(self, api_url: str, host: str = "127.0.0.1", port: int = 8080):
        """Initialize the dashboard."""
        self.api_url = api_url
        self.host = host
        self.port = port
        self.app = FastAPI(title="Trading Dashboard")
        
        # Create templates directory if it doesn't exist
        os.makedirs("templates", exist_ok=True)
        
        # Create the dashboard HTML template
        self._create_templates()
        
        # Set up templates
        self.templates = Jinja2Templates(directory="templates")
        
        # Register routes
        self.register_routes()
    
    def _create_templates(self):
        """Create the dashboard HTML template."""
        dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Analytics Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; }
        .card { @apply bg-white p-6 rounded-lg shadow-md border border-gray-200; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800">Trading Analytics Dashboard</h1>
            <p class="text-gray-600">AI-powered portfolio management</p>
        </header>
        
        <!-- Status Bar -->
        <div class="flex justify-between items-center mb-6 bg-white p-4 rounded-lg shadow">
            <div>
                <span class="font-semibold">Status:</span>
                <span id="status-badge" class="ml-2 px-2 py-1 rounded text-xs font-semibold">Not Connected</span>
            </div>
            <div>
                <span class="font-semibold">Steps:</span>
                <span id="step-count" class="ml-2">0</span>
            </div>
            <div>
                <span class="font-semibold">Objective:</span>
                <span id="objective" class="ml-2">None</span>
            </div>
            <div class="flex space-x-2">
                <button id="startBtn" class="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded">Start</button>
                <button id="pauseBtn" class="bg-yellow-500 hover:bg-yellow-600 text-white px-3 py-1 rounded">Pause</button>
                <button id="resumeBtn" class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded">Resume</button>
                <button id="stopBtn" class="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded">Stop</button>
                <a href="/chat" class="bg-purple-500 hover:bg-purple-600 text-white px-3 py-1 rounded">
                    <i class="fas fa-comments mr-1"></i> Chat
                </a>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="grid grid-cols-1 md:grid-cols-12 gap-6">
            <!-- Portfolio Overview -->
            <div class="md:col-span-8">
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Portfolio Performance</h2>
                    <canvas id="portfolioChart" height="250"></canvas>
                </div>
            </div>
            
            <!-- Key Metrics -->
            <div class="md:col-span-4">
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Key Metrics</h2>
                    <div class="space-y-4">
                        <div>
                            <p class="text-gray-600">Portfolio Value</p>
                            <p id="portfolio-value" class="text-2xl font-bold">$0.00</p>
                        </div>
                        <div>
                            <p class="text-gray-600">Cash Balance</p>
                            <p id="cash-balance" class="text-2xl font-bold">$0.00</p>
                        </div>
                        <div>
                            <p class="text-gray-600">Total Positions</p>
                            <p id="position-count" class="text-2xl font-bold">0</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Trajectory Table -->
            <div class="md:col-span-12">
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Trade Trajectory</h2>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead>
                                <tr>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Step</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Portfolio Value</th>
                                </tr>
                            </thead>
                            <tbody id="trajectory-body" class="bg-white divide-y divide-gray-200">
                                <tr>
                                    <td colspan="5" class="px-6 py-4 text-center text-gray-500">No data available</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Agent Reasoning -->
            <div class="md:col-span-12">
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Latest Agent Reasoning</h2>
                    <div id="agent-reasoning" class="bg-gray-50 p-4 rounded border border-gray-200 max-h-64 overflow-y-auto">
                        <p class="text-gray-500">No reasoning available</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Task Selection Modal -->
        <div id="taskModal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-white rounded-lg p-6 max-w-md w-full">
                <h2 class="text-xl font-semibold mb-4">Select Task</h2>
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2" for="taskSelect">Available Tasks:</label>
                    <select id="taskSelect" class="block w-full px-3 py-2 border border-gray-300 rounded">
                        <option value="" disabled selected>Choose a task</option>
                    </select>
                </div>
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2" for="stepsInput">Max Steps:</label>
                    <input type="number" id="stepsInput" class="block w-full px-3 py-2 border border-gray-300 rounded" value="10" min="1" max="100">
                </div>
                <div class="flex justify-end space-x-3">
                    <button id="cancelTaskBtn" class="px-4 py-2 border border-gray-300 rounded text-gray-700 hover:bg-gray-100">Cancel</button>
                    <button id="confirmTaskBtn" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Start Task</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Constants
        const API_URL = '{{ api_url }}';
        
        // Chart instances
        let portfolioChart;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            loadTasks();
            updateStatus();
            setInterval(updateStatus, 2000);
            setInterval(updateTrajectory, 2000);
            
            // Connect buttons
            document.getElementById('startBtn').addEventListener('click', showTaskModal);
            document.getElementById('pauseBtn').addEventListener('click', pauseAgent);
            document.getElementById('resumeBtn').addEventListener('click', resumeAgent);
            document.getElementById('stopBtn').addEventListener('click', stopAgent);
            document.getElementById('confirmTaskBtn').addEventListener('click', startTask);
            document.getElementById('cancelTaskBtn').addEventListener('click', hideTaskModal);
        });
        
        function initCharts() {
            // Initialize portfolio chart
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            portfolioChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: 'rgba(59, 130, 246, 1)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: value => '$' + value.toLocaleString()
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: context => '$' + context.raw.toLocaleString()
                            }
                        }
                    }
                }
            });
        }
        
        async function loadTasks() {
            try {
                const response = await fetch(`${API_URL}/tasks`);
                const tasks = await response.json();
                
                const taskSelect = document.getElementById('taskSelect');
                taskSelect.innerHTML = '<option value="" disabled selected>Choose a task</option>';
                
                tasks.forEach(task => {
                    const option = document.createElement('option');
                    option.value = task.id;
                    option.textContent = task.instruction;
                    taskSelect.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading tasks:', error);
            }
        }
        
        async function updateStatus() {
            try {
                const response = await fetch(`${API_URL}/status`);
                const status = await response.json();
                
                const statusBadge = document.getElementById('status-badge');
                const stepCount = document.getElementById('step-count');
                const objective = document.getElementById('objective');
                
                // Update status badge
                statusBadge.textContent = status.status;
                statusBadge.className = 'ml-2 px-2 py-1 rounded text-xs font-semibold';
                
                if (status.status === 'running') {
                    statusBadge.classList.add('bg-green-100', 'text-green-800');
                } else if (status.status === 'paused') {
                    statusBadge.classList.add('bg-yellow-100', 'text-yellow-800');
                } else {
                    statusBadge.classList.add('bg-gray-100', 'text-gray-800');
                }
                
                // Update step count
                stepCount.textContent = status.step_count || 0;
                
                // Update objective
                objective.textContent = status.objective || 'None';
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }
        
        async function updateTrajectory() {
            try {
                const response = await fetch(`${API_URL}/trajectory`);
                const trajectory = await response.json();
                
                // Update portfolio chart
                updatePortfolioChart(trajectory.portfolio_history || []);
                
                // Update trajectory table
                updateTrajectoryTable(trajectory.steps || []);
                
                // Update latest reasoning
                updateLatestReasoning(trajectory.steps || []);
                
                // Update key metrics
                updateKeyMetrics(trajectory.steps || []);
            } catch (error) {
                console.error('Error updating trajectory:', error);
            }
        }
        
        function updatePortfolioChart(portfolioHistory) {
            if (!portfolioHistory.length) return;
            
            const labels = portfolioHistory.map(point => {
                const date = new Date(point.timestamp);
                return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            });
            
            const values = portfolioHistory.map(point => point.value);
            
            portfolioChart.data.labels = labels;
            portfolioChart.data.datasets[0].data = values;
            portfolioChart.update();
        }
        
        function updateTrajectoryTable(steps) {
            const tbody = document.getElementById('trajectory-body');
            
            if (!steps.length) {
                tbody.innerHTML = '<tr><td colspan="5" class="px-6 py-4 text-center text-gray-500">No data available</td></tr>';
                return;
            }
            
            tbody.innerHTML = '';
            
            // Sort steps in reverse order (newest first)
            const sortedSteps = [...steps].reverse();
            
            sortedSteps.forEach(step => {
                const tr = document.createElement('tr');
                
                // Format time
                const timestamp = new Date(step.timestamp);
                const timeStr = timestamp.toLocaleTimeString();
                
                // Format action
                let actionStr = step.action || 'Initial State';
                let detailsStr = '';
                
                if (step.action_details) {
                    const {type, ticker, quantity, price} = step.action_details;
                    actionStr = type ? type.toUpperCase() : actionStr;
                    
                    if (ticker && quantity) {
                        detailsStr = `${ticker} x ${quantity}`;
                        if (price) detailsStr += ` @ $${price.toFixed(2)}`;
                    }
                }
                
                // Format portfolio value
                const valueStr = step.portfolio_value ? `$${step.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '';
                
                tr.innerHTML = `
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${step.index}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${timeStr}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium ${actionStr === 'Initial State' ? 'text-gray-500' : ''}">${actionStr}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${detailsStr}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${valueStr}</td>
                `;
                
                tbody.appendChild(tr);
            });
        }
        
        function updateLatestReasoning(steps) {
            const reasoningDiv = document.getElementById('agent-reasoning');
            
            // Filter steps with reasoning
            const stepsWithReasoning = steps.filter(step => step.reasoning);
            
            if (!stepsWithReasoning.length) {
                reasoningDiv.innerHTML = '<p class="text-gray-500">No reasoning available</p>';
                return;
            }
            
            // Get the latest reasoning
            const latestStep = stepsWithReasoning.reduce((latest, current) => 
                current.index > latest.index ? current : latest, stepsWithReasoning[0]);
            
            // Format the reasoning with simple markdown-like formatting
            let formattedReasoning = latestStep.reasoning
                .replace(/\n\n/g, '<br><br>')
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
            
            reasoningDiv.innerHTML = `<p>${formattedReasoning}</p>`;
        }
        
        function updateKeyMetrics(steps) {
            // Find the latest step with portfolio data
            const latestStep = steps.length ? steps[steps.length - 1] : null;
            
            if (!latestStep) return;
            
            // Update portfolio value
            const portfolioValue = document.getElementById('portfolio-value');
            if (latestStep.portfolio_value) {
                portfolioValue.textContent = `$${latestStep.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }
            
            // Update cash balance
            const cashBalance = document.getElementById('cash-balance');
            if (latestStep.cash_balance) {
                cashBalance.textContent = `$${latestStep.cash_balance.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }
            
            // Get count of positions by looking at latest state
            const positionCount = document.getElementById('position-count');
            let count = 0;
            
            if (latestStep.state && latestStep.state.portfolio && latestStep.state.portfolio.positions) {
                count = Object.keys(latestStep.state.portfolio.positions).length;
            }
            
            positionCount.textContent = count;
        }
        
        function showTaskModal() {
            document.getElementById('taskModal').classList.remove('hidden');
        }
        
        function hideTaskModal() {
            document.getElementById('taskModal').classList.add('hidden');
        }
        
        async function startTask() {
            const taskId = document.getElementById('taskSelect').value;
            const steps = document.getElementById('stepsInput').value;
            
            if (!taskId) {
                alert('Please select a task');
                return;
            }
            
            try {
                const response = await fetch(`${API_URL}/start`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        task_id: taskId,
                        steps: parseInt(steps)
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    hideTaskModal();
                    updateStatus();
                    updateTrajectory();
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                console.error('Error starting task:', error);
                alert('Error starting task');
            }
        }
        
        async function pauseAgent() {
            try {
                await fetch(`${API_URL}/pause`, { method: 'POST' });
                updateStatus();
            } catch (error) {
                console.error('Error pausing agent:', error);
            }
        }
        
        async function resumeAgent() {
            try {
                await fetch(`${API_URL}/resume`, { method: 'POST' });
                updateStatus();
            } catch (error) {
                console.error('Error resuming agent:', error);
            }
        }
        
        async function stopAgent() {
            if (!confirm('Are you sure you want to stop the agent?')) return;
            
            try {
                await fetch(`${API_URL}/stop`, { method: 'POST' });
                updateStatus();
                updateTrajectory();
            } catch (error) {
                console.error('Error stopping agent:', error);
            }
        }
    </script>
</body>
</html>
"""
        
        with open("templates/dashboard.html", "w") as f:
            f.write(dashboard_html)
    
    def register_routes(self):
        """Register dashboard routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Render the dashboard."""
            return self.templates.TemplateResponse(
                "dashboard.html", 
                {"request": request, "api_url": self.api_url}
            )
    
    def start(self):
        """Start the dashboard server."""
        uvicorn.run(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Dashboard')
    parser.add_argument('--api-url', type=str, default="http://localhost:8000",
                        help='URL of the Trading Agent API')
    parser.add_argument('--port', type=int, default=8080, 
                        help='Dashboard server port')
    
    args = parser.parse_args()
    
    dashboard = TradingDashboard(api_url=args.api_url, port=args.port)
    print(f"Starting Trading Dashboard on port {args.port}...")
    dashboard.start()