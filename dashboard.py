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
            <div class="mt-4">
                <a href="/" class="text-blue-600 hover:text-blue-800 mr-4">Dashboard</a>
                <a href="/tools" class="text-blue-600 hover:text-blue-800 mr-4">Financial Tools</a>
                <a href="/chat" class="text-blue-600 hover:text-blue-800">Chat</a>
            </div>
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
            
    def _create_financial_tools_template(self):
        """Create the financial tools HTML template."""
        financial_tools_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Tools | Trading Analytics</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; }
        .card { @apply bg-white p-6 rounded-lg shadow-md border border-gray-200; }
        textarea { resize: vertical; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800">Financial Tools</h1>
            <p class="text-gray-600">Advanced financial engineering and analytics</p>
            <div class="mt-4">
                <a href="/" class="text-blue-600 hover:text-blue-800 mr-4">Dashboard</a>
                <a href="/tools" class="text-blue-600 hover:text-blue-800 mr-4 font-bold">Financial Tools</a>
                <a href="/chat" class="text-blue-600 hover:text-blue-800">Chat</a>
            </div>
        </header>
        
        <!-- Tool Navigation -->
        <div class="mb-6">
            <div class="border-b border-gray-200">
                <nav class="flex -mb-px">
                    <button id="navReturns" class="tab-btn text-blue-600 border-b-2 border-blue-600 py-2 px-4 font-medium">
                        Returns Analysis
                    </button>
                    <button id="navIndicators" class="tab-btn text-gray-500 hover:text-gray-700 py-2 px-4 font-medium">
                        Technical Indicators
                    </button>
                    <button id="navPortfolio" class="tab-btn text-gray-500 hover:text-gray-700 py-2 px-4 font-medium">
                        Portfolio Optimization
                    </button>
                </nav>
            </div>
        </div>
        
        <!-- Returns Analysis Tool -->
        <div id="returnsAnalysisTool" class="tool-section">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Input Price Data</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2" for="priceData">Price Series (comma-separated or one per line)</label>
                        <textarea id="priceData" rows="6" class="w-full px-3 py-2 border border-gray-300 rounded" placeholder="100,101,99,102,105,104..."></textarea>
                    </div>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2" for="periodsPerYear">Periods Per Year</label>
                        <select id="periodsPerYear" class="w-full px-3 py-2 border border-gray-300 rounded">
                            <option value="252">252 (Daily Trading)</option>
                            <option value="52">52 (Weekly)</option>
                            <option value="12">12 (Monthly)</option>
                            <option value="4">4 (Quarterly)</option>
                            <option value="1">1 (Yearly)</option>
                        </select>
                    </div>
                    <button id="analyzeReturnsBtn" class="w-full bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
                        Analyze Returns
                    </button>
                </div>
                
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Returns Analysis Results</h2>
                    <div id="returnsResults" class="bg-gray-50 p-4 rounded border border-gray-200 min-h-64">
                        <p class="text-gray-500">Results will appear here</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Technical Indicators Tool -->
        <div id="technicalIndicatorsTool" class="tool-section hidden">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Input Price Data</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2" for="indicatorsPrice">Close Prices (comma-separated or one per line)</label>
                        <textarea id="indicatorsPrice" rows="6" class="w-full px-3 py-2 border border-gray-300 rounded" placeholder="100,101,99,102,105,104..."></textarea>
                    </div>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2" for="indicatorsHigh">High Prices (optional)</label>
                        <textarea id="indicatorsHigh" rows="3" class="w-full px-3 py-2 border border-gray-300 rounded" placeholder="101,103,100,104,107,106..."></textarea>
                    </div>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2" for="indicatorsLow">Low Prices (optional)</label>
                        <textarea id="indicatorsLow" rows="3" class="w-full px-3 py-2 border border-gray-300 rounded" placeholder="98,100,97,101,103,102..."></textarea>
                    </div>
                    <button id="calculateIndicatorsBtn" class="w-full bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
                        Calculate Indicators
                    </button>
                </div>
                
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Technical Indicators</h2>
                    <div class="mb-4">
                        <canvas id="indicatorsChart" height="200"></canvas>
                    </div>
                    <div id="indicatorsSummary" class="bg-gray-50 p-4 rounded border border-gray-200">
                        <p class="text-gray-500">Indicators summary will appear here</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Portfolio Optimization Tool -->
        <div id="portfolioOptimizationTool" class="tool-section hidden">
            <div class="grid grid-cols-1 gap-6">
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Portfolio Optimization</h2>
                    <div class="mb-4">
                        <p class="text-gray-700 mb-2">Enter return data for different assets. Each column represents an asset, each row represents a period.</p>
                        <div id="portfolioAssets" class="mb-4">
                            <div class="flex mb-2">
                                <input type="text" placeholder="Asset 1" class="asset-name w-1/3 px-3 py-2 border border-gray-300 rounded-l">
                                <textarea placeholder="Returns data (comma-separated)" class="asset-returns w-2/3 px-3 py-2 border border-gray-300 rounded-r" rows="2"></textarea>
                            </div>
                            <div class="flex mb-2">
                                <input type="text" placeholder="Asset 2" class="asset-name w-1/3 px-3 py-2 border border-gray-300 rounded-l">
                                <textarea placeholder="Returns data (comma-separated)" class="asset-returns w-2/3 px-3 py-2 border border-gray-300 rounded-r" rows="2"></textarea>
                            </div>
                        </div>
                        <button id="addAssetBtn" class="px-4 py-2 border border-gray-300 rounded text-gray-700 hover:bg-gray-100 mb-4">
                            + Add Asset
                        </button>
                    </div>
                    <div class="mb-4">
                        <label class="block text-gray-700 mb-2" for="riskFreeRate">Risk-Free Rate (annual)</label>
                        <input type="number" id="riskFreeRate" step="0.001" min="0" max="0.2" value="0.03" class="w-full px-3 py-2 border border-gray-300 rounded">
                    </div>
                    <button id="optimizePortfolioBtn" class="w-full bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
                        Optimize Portfolio
                    </button>
                </div>
                
                <div class="card">
                    <h2 class="text-xl font-semibold mb-4">Optimization Results</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h3 class="font-semibold mb-2">Optimal Weights</h3>
                            <div id="optimalWeights" class="bg-gray-50 p-4 rounded border border-gray-200 min-h-40">
                                <p class="text-gray-500">Results will appear here</p>
                            </div>
                        </div>
                        <div>
                            <h3 class="font-semibold mb-2">Portfolio Metrics</h3>
                            <div id="portfolioMetrics" class="bg-gray-50 p-4 rounded border border-gray-200 min-h-40">
                                <p class="text-gray-500">Results will appear here</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        const API_URL = '{{ api_url }}';
        let indicatorsChart = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Set up tab navigation
            setupTabs();
            
            // Set up event listeners
            document.getElementById('analyzeReturnsBtn').addEventListener('click', analyzeReturns);
            document.getElementById('calculateIndicatorsBtn').addEventListener('click', calculateIndicators);
            document.getElementById('optimizePortfolioBtn').addEventListener('click', optimizePortfolio);
            document.getElementById('addAssetBtn').addEventListener('click', addAssetField);
            
            // Initialize charts
            initializeCharts();
        });
        
        function setupTabs() {
            const tabButtons = document.querySelectorAll('.tab-btn');
            const toolSections = document.querySelectorAll('.tool-section');
            
            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    // Deactivate all tabs
                    tabButtons.forEach(btn => {
                        btn.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
                        btn.classList.add('text-gray-500', 'hover:text-gray-700');
                    });
                    
                    // Hide all sections
                    toolSections.forEach(section => {
                        section.classList.add('hidden');
                    });
                    
                    // Activate clicked tab
                    button.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');
                    button.classList.remove('text-gray-500', 'hover:text-gray-700');
                    
                    // Show corresponding section
                    const targetId = button.id.replace('nav', '') + 'Tool';
                    document.getElementById(targetId).classList.remove('hidden');
                });
            });
        }
        
        function initializeCharts() {
            // Will initialize charts when needed
        }
        
        function parseInput(inputStr) {
            // Handle comma-separated or newline-separated inputs
            if (!inputStr.trim()) return [];
            
            // Try to split by comma
            if (inputStr.includes(',')) {
                return inputStr.split(',').map(val => parseFloat(val.trim())).filter(val => !isNaN(val));
            }
            
            // Try to split by newline
            return inputStr.split('\\n').map(val => parseFloat(val.trim())).filter(val => !isNaN(val));
        }
        
        async function analyzeReturns() {
            const priceData = parseInput(document.getElementById('priceData').value);
            const periodsPerYear = parseInt(document.getElementById('periodsPerYear').value);
            
            if (priceData.length < 2) {
                alert('Please enter at least 2 price points');
                return;
            }
            
            try {
                const response = await fetch('/proxy/analyze/returns', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prices: priceData,
                        periods_per_year: periodsPerYear
                    })
                });
                
                const results = await response.json();
                displayReturnsResults(results);
            } catch (error) {
                console.error('Error analyzing returns:', error);
                document.getElementById('returnsResults').innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
            }
        }
        
        function displayReturnsResults(results) {
            const resultsDiv = document.getElementById('returnsResults');
            
            const formattedResults = `
                <div class="space-y-3">
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Total Return:</span>
                        <span>${(results.total_return * 100).toFixed(2)}%</span>
                    </div>
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Annualized Return:</span>
                        <span>${(results.annualized_return * 100).toFixed(2)}%</span>
                    </div>
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Volatility:</span>
                        <span>${(results.volatility * 100).toFixed(2)}%</span>
                    </div>
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Sharpe Ratio:</span>
                        <span>${results.sharpe_ratio.toFixed(2)}</span>
                    </div>
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Max Drawdown:</span>
                        <span>${(results.max_drawdown * 100).toFixed(2)}%</span>
                    </div>
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Value at Risk (95%):</span>
                        <span>${(results.var_95 * 100).toFixed(2)}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="font-semibold">Expected Shortfall (95%):</span>
                        <span>${(results.expected_shortfall_95 * 100).toFixed(2)}%</span>
                    </div>
                </div>
            `;
            
            resultsDiv.innerHTML = formattedResults;
        }
        
        async function calculateIndicators() {
            const priceData = parseInput(document.getElementById('indicatorsPrice').value);
            const highData = parseInput(document.getElementById('indicatorsHigh').value);
            const lowData = parseInput(document.getElementById('indicatorsLow').value);
            
            if (priceData.length < 20) {
                alert('Please enter at least 20 price points for meaningful indicators');
                return;
            }
            
            try {
                const response = await fetch('/proxy/analyze/indicators', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prices: priceData,
                        high: highData.length > 0 ? highData : null,
                        low: lowData.length > 0 ? lowData : null
                    })
                });
                
                const results = await response.json();
                displayIndicatorsResults(priceData, results);
            } catch (error) {
                console.error('Error calculating indicators:', error);
                document.getElementById('indicatorsSummary').innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
            }
        }
        
        function displayIndicatorsResults(priceData, indicators) {
            const summaryDiv = document.getElementById('indicatorsSummary');
            
            // Create time labels (just indices for simplicity)
            const labels = Array.from({length: priceData.length}, (_, i) => i+1);
            
            // Update the chart
            if (indicatorsChart) {
                indicatorsChart.destroy();
            }
            
            const ctx = document.getElementById('indicatorsChart').getContext('2d');
            indicatorsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Price',
                            data: priceData,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'SMA 20',
                            data: indicators.sma_20,
                            borderColor: 'rgba(153, 102, 255, 1)',
                            backgroundColor: 'rgba(153, 102, 255, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'EMA 20',
                            data: indicators.ema_20,
                            borderColor: 'rgba(255, 159, 64, 1)',
                            backgroundColor: 'rgba(255, 159, 64, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'RSI',
                            data: indicators.rsi,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.1,
                            hidden: true,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Price'
                            }
                        },
                        y1: {
                            position: 'right',
                            min: 0,
                            max: 100,
                            title: {
                                display: true,
                                text: 'RSI'
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                }
            });
            
            // Create summary text
            const lastIndex = priceData.length - 1;
            const sma20 = indicators.sma_20[lastIndex];
            const ema20 = indicators.ema_20[lastIndex];
            const rsi = indicators.rsi[lastIndex];
            const price = priceData[lastIndex];
            
            const rsiStatus = rsi < 30 ? 'Oversold' : rsi > 70 ? 'Overbought' : 'Neutral';
            const maStatus = price > sma20 ? 'Above SMA20 (Bullish)' : 'Below SMA20 (Bearish)';
            
            let macdStatus = 'Neutral';
            if (indicators.macd && indicators.macd_signal) {
                const macd = indicators.macd[lastIndex];
                const macdSignal = indicators.macd_signal[lastIndex];
                macdStatus = macd > macdSignal ? 'Bullish (MACD > Signal)' : 'Bearish (MACD < Signal)';
            }
            
            const summaryText = `
                <div class="space-y-2">
                    <p class="font-semibold">Current Price: $${price.toFixed(2)}</p>
                    <p>SMA20: $${sma20.toFixed(2)}</p>
                    <p>EMA20: $${ema20.toFixed(2)}</p>
                    <p>RSI: ${rsi.toFixed(2)} (${rsiStatus})</p>
                    <p>Trend: ${maStatus}</p>
                    <p>MACD Signal: ${macdStatus}</p>
                </div>
            `;
            
            summaryDiv.innerHTML = summaryText;
        }
        
        function addAssetField() {
            const assetsContainer = document.getElementById('portfolioAssets');
            const assetCount = assetsContainer.children.length + 1;
            
            const assetDiv = document.createElement('div');
            assetDiv.className = 'flex mb-2';
            assetDiv.innerHTML = `
                <input type="text" placeholder="Asset ${assetCount}" class="asset-name w-1/3 px-3 py-2 border border-gray-300 rounded-l">
                <textarea placeholder="Returns data (comma-separated)" class="asset-returns w-2/3 px-3 py-2 border border-gray-300 rounded-r" rows="2"></textarea>
            `;
            
            assetsContainer.appendChild(assetDiv);
        }
        
        async function optimizePortfolio() {
            const assetNames = Array.from(document.querySelectorAll('.asset-name')).map(input => input.value.trim() || input.placeholder);
            const assetReturns = Array.from(document.querySelectorAll('.asset-returns')).map(textarea => parseInput(textarea.value));
            const riskFreeRate = parseFloat(document.getElementById('riskFreeRate').value);
            
            // Validate inputs
            let valid = true;
            assetReturns.forEach((returns, index) => {
                if (returns.length < 2) {
                    alert(`Please enter at least 2 return values for ${assetNames[index]}`);
                    valid = false;
                }
            });
            
            if (!valid) return;
            
            // Prepare returns data
            const returnsLength = Math.min(...assetReturns.map(arr => arr.length));
            const returnsData = {};
            
            assetNames.forEach((name, index) => {
                returnsData[name] = assetReturns[index].slice(0, returnsLength);
            });
            
            try {
                const response = await fetch('/proxy/analyze/optimize-portfolio', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        returns: returnsData,
                        risk_free_rate: riskFreeRate
                    })
                });
                
                const results = await response.json();
                displayOptimizationResults(results);
            } catch (error) {
                console.error('Error optimizing portfolio:', error);
                document.getElementById('optimalWeights').innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
                document.getElementById('portfolioMetrics').innerHTML = '';
            }
        }
        
        function displayOptimizationResults(results) {
            const weightsDiv = document.getElementById('optimalWeights');
            const metricsDiv = document.getElementById('portfolioMetrics');
            
            // Format weights
            let weightsHtml = '<div class="space-y-2">';
            for (const [asset, weight] of Object.entries(results.weights)) {
                weightsHtml += `
                    <div class="flex justify-between">
                        <span>${asset}:</span>
                        <span>${(weight * 100).toFixed(2)}%</span>
                    </div>
                `;
            }
            weightsHtml += '</div>';
            
            // Format metrics
            const metricsHtml = `
                <div class="space-y-3">
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Expected Return:</span>
                        <span>${(results.expected_return * 100).toFixed(2)}%</span>
                    </div>
                    <div class="flex justify-between border-b pb-2">
                        <span class="font-semibold">Expected Volatility:</span>
                        <span>${(results.expected_volatility * 100).toFixed(2)}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="font-semibold">Sharpe Ratio:</span>
                        <span>${results.sharpe_ratio.toFixed(2)}</span>
                    </div>
                </div>
            `;
            
            weightsDiv.innerHTML = weightsHtml;
            metricsDiv.innerHTML = metricsHtml;
        }
    </script>
</body>
</html>
"""
        
        with open("templates/financial_tools.html", "w") as f:
            f.write(financial_tools_html)
    
    def register_routes(self):
        """Register dashboard routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Render the dashboard."""
            return self.templates.TemplateResponse(
                "dashboard.html", 
                {"request": request, "api_url": self.api_url}
            )
            
        @self.app.get("/tools", response_class=HTMLResponse)
        async def financial_tools(request: Request):
            """Render the financial tools page."""
            # Create the financial tools template if it doesn't exist
            self._create_financial_tools_template()
            
            return self.templates.TemplateResponse(
                "financial_tools.html",
                {"request": request, "api_url": self.api_url}
            )
        
        @self.app.post("/proxy/analyze/returns")
        async def proxy_analyze_returns(request: Request):
            """Proxy endpoint for analyzing returns."""
            data = await request.json()
            
            response = requests.post(
                f"{self.api_url}/analyze/returns",
                json=data
            )
            
            return response.json()
            
        @self.app.post("/proxy/analyze/indicators")
        async def proxy_analyze_indicators(request: Request):
            """Proxy endpoint for generating indicators."""
            data = await request.json()
            
            response = requests.post(
                f"{self.api_url}/analyze/indicators",
                json=data
            )
            
            return response.json()
            
        @self.app.post("/proxy/analyze/optimize-portfolio")
        async def proxy_optimize_portfolio(request: Request):
            """Proxy endpoint for portfolio optimization."""
            data = await request.json()
            
            response = requests.post(
                f"{self.api_url}/analyze/optimize-portfolio",
                json=data
            )
            
            return response.json()
    
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