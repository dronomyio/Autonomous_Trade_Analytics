# Trade Analytics Platform

An AI-powered stock trading analytics platform that leverages Claude to make trading decisions, analyze markets, and manage investment portfolios. This project adapts the architecture from the Pokemon example to stock trading.

## Features

- Claude-powered trading agent with market reasoning capabilities
- Multiple trading strategies and verification systems
- Portfolio tracking and performance metrics
- Trade trajectory system for decision auditing
- Market data integration and analysis tools
- Advanced financial engineering tools for quantitative analysis:
  - Returns analysis (simple, log, cumulative)
  - Risk metrics (volatility, Sharpe ratio, VaR, drawdown)
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  - Portfolio optimization using modern portfolio theory
  - Statistical analysis for financial time series
- Regulatory filing anomaly detection:
  - FR Y-9C bank filing analysis
  - SEC 10-K annual report analysis
  - Claude-powered analysis of detected anomalies

## Architecture Overview

This platform is built on the EVA (Executions with Verified Agents) framework, adapted from the Pokemon example to handle stock trading. Key components include:

1. **Trade Agent**: Using Claude LLM to analyze market data and make trading decisions
2. **Market Connector**: Interface with trading platforms and market data providers
3. **Trajectory System**: Recording and tracking every action, state, and result
4. **Task Verification**: Validating if trading objectives have been accomplished
5. **Web Dashboard**: Visualizing portfolio performance and agent decisions
6. **Financial Tools**: Suite of quantitative analysis tools for financial engineering
7. **Regulatory Analysis**: Tools for detecting anomalies in financial regulatory filings

## Setup Instructions

You can run this application either directly on your host system or using Docker.

### Option 1: Local Installation

#### Prerequisites

- Python 3.11 or higher
- Anthropic API key

#### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trade-analytics.git
cd trade-analytics

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Unix/MacOS
# OR
venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

#### Configuration

Create a `.env` file with the following variables:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Option 2: Docker Installation

#### Prerequisites

- Docker and Docker Compose
- Anthropic API key

#### Quick Start (Recommended)

We provide a convenient startup script that handles the setup process:

```bash
# Clone the repository
git clone https://github.com/yourusername/trade-analytics.git
cd trade-analytics

# Run the quick start script
./run.sh
```

The script will:
1. Create the .env file if it doesn't exist
2. Prompt you to add your API key
3. Check for Docker and Docker Compose
4. Build and start the containers
5. Show you the URLs to access the application

#### Manual Installation

If you prefer to set up manually:

```bash
# Clone the repository
git clone https://github.com/yourusername/trade-analytics.git
cd trade-analytics

# Create .env file from example
cp .env.example .env

# Edit the .env file to add your Anthropic API key
# ANTHROPIC_API_KEY=your_anthropic_api_key

# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f
```

This will start two containers:
- `trade-agent` on port 8000 (API)
- `dashboard` on port 8080 (Web UI)

Access the dashboard at [http://localhost:8080](http://localhost:8080)

#### Stopping the Application

```bash
docker-compose down
```

## Usage

The platform can be run in two modes:

### API Server Mode

This mode runs both the trading agent API and dashboard UI:

```bash
# Start the trading agent API (default: port 8000)
python trade_agent.py --mode api --port 8000

# In a separate terminal, start the dashboard
python dashboard.py --api-url http://localhost:8000 --port 8080
```

Then access the dashboard at [http://localhost:8080](http://localhost:8080)

#### Financial Tools Interface

The platform includes a suite of financial engineering tools:
- Access the financial tools UI at [http://localhost:8080/tools](http://localhost:8080/tools)
- Analyze returns and calculate risk metrics
- Generate and visualize technical indicators
- Optimize portfolio allocation using modern portfolio theory
- API endpoints available at `/analyze/returns`, `/analyze/indicators`, and `/analyze/optimize-portfolio`

#### Regulatory Analysis Interface

The platform includes advanced regulatory filing analysis:
- Analyze FR Y-9C bank regulatory filings for anomalies
- Analyze SEC 10-K annual reports for financial statement anomalies
- Generate detailed analysis prompts for Claude
- Visualize financial metrics with anomaly highlighting
- Python API for programmatic regulatory analysis

##### Prompt Integration Architecture

The regulatory analysis prompts are fully integrated into the application architecture:

1. **Anomaly Detection → Prompt Generation → Claude Analysis Pipeline**:
   ```
   Data Loading → Statistical Analysis → Prompt Assembly → Claude LLM Analysis
   ```

2. **Dynamic Prompt Generation**:
   - Prompts are dynamically generated from detected anomalies
   - Context-specific details (bank/company info, metrics, thresholds) are automatically included
   - Statistical data (z-scores, YoY changes, peer comparisons) is integrated into the prompt
   - Analysis questions are tailored to the filing type (Y-9C or 10-K)

3. **Implementation Locations**:
   - Y-9C prompts: `tools/y9c_anomaly_detector.py` → `generate_anomaly_prompt()`
   - 10-K prompts: `tools/sec_anomaly_detector.py` → `generate_10k_anomaly_prompt()`
   - Unified access: `tools/regulatory_analysis.py` → `get_y9c_prompt()` and `get_10k_prompt()`

4. **Claude Integration**:
   - Prompts are sent to Claude via Anthropic API
   - Responses are captured and integrated with visualization tools
   - Analysis can be saved to files or returned programmatically

5. **Example Prompt Generation**:
   ```python
   from tools.regulatory_analysis import RegulatoryAnalyzer
   
   analyzer = RegulatoryAnalyzer()
   
   # Generate a Y-9C prompt without running full analysis
   y9c_prompt = analyzer.get_y9c_prompt(bank_id=9012)
   
   # Or get a 10-K prompt
   k10_prompt = analyzer.get_10k_prompt(ticker="META")
   
   # Run a custom prompt through Claude
   analysis = analyzer.run_claude_analysis(
       "Your custom prompt here...",
       max_tokens=4000
   )
   ```

6. **Complete Analysis Workflow Example**:
   ```python
   from tools.regulatory_analysis import RegulatoryAnalyzer
   import os
   
   # Get API key from environment (or provide directly)
   api_key = os.environ.get("ANTHROPIC_API_KEY")
   
   # Initialize analyzer
   analyzer = RegulatoryAnalyzer(api_key=api_key)
   
   # Run a full 10-K analysis on Meta's filing
   results = analyzer.analyze_10k_filing(
       ticker="META",
       use_claude=True,
       save_output=True
   )
   
   # Extract key components
   anomalies = results['financial_anomalies']
   yoy_changes = results['yoy_anomalies']
   prompt = results['prompt']
   claude_analysis = results['claude_analysis']
   
   # Display summary of findings
   print(f"Found {len(anomalies)} financial statement anomalies")
   print(f"Found {len(yoy_changes)} significant year-over-year changes")
   print("\nTop 3 anomalies by z-score:")
   print(anomalies.sort_values('z_score', ascending=False).head(3)[
       ['metric', 'value', 'z_score', 'direction']
   ])
   
   # Display Claude's key insights (first 500 chars)
   print("\nClaude's Analysis Highlights:")
   print(claude_analysis[:500] + "...")
   
   # Visualize the anomalies
   analyzer.sec_detector.plot_company_metrics(
       analyzer.sec_detector.load_10k_data(),
       ticker="META"
   )
   ```

#### Data Integration

The platform supports various market data formats:
- Stock data in CSV format in the `/data/stocks/` directory
- Options chains in JSON format in the `/data/options/` directory
- Financial news and sentiment in JSON format in the `/data/news/` directory
- Regulatory filing data in the `/data/regulatory/` directory

A data loader utility is provided to integrate this data with the financial tools:
```python
from tools.data_loader import DataLoader

# Initialize the data loader
loader = DataLoader()

# List available stocks
tickers = loader.list_available_stocks()

# Load stock data for analysis
aapl_data = loader.load_stock_data("AAPL")

# Load options data
aapl_options = loader.load_options_data("AAPL")

# Load news data
aapl_news = loader.load_news_data(ticker="AAPL")

# Run full analysis using the financial toolkit
analysis = loader.analyze_stock("AAPL")
```

#### Conversational Interface

The platform includes a chat interface where you can directly interact with the trading agent:
- Access the chat UI at [http://localhost:8000/chat](http://localhost:8000/chat)
- Discuss your portfolio and market trends
- Ask for analysis and recommendations
- Execute trades through natural language

### Direct Mode

This mode runs the trading agent directly without API or dashboard:

```bash
# Run with specific task and step count
python trade_agent.py --mode direct --task rebalance-portfolio --steps 10
```

## Available Tasks

The system comes with several predefined trading tasks:

- **rebalance-portfolio**: Rebalance to match target allocation
- **reduce-risk**: Reduce portfolio volatility by at least 10%
- **reach-profit-target**: Achieve 5% portfolio return
- **implement-dca**: Implement dollar cost averaging strategy
- **sector-diversification**: Diversify across at least 4 sectors

## Project Structure

- `agent/` - Trading agent implementation
- `eva.py` - Execution with Verified Agents framework
- `tasks.py` - Trading tasks and verification functions
- `market_connector.py` - Market data and broker API connections
- `trade_agent.py` - Main implementation with trajectory system
- `dashboard.py` - Web UI for monitoring and control
- `tools/` - Financial engineering and quantitative analysis toolset:
  - `returns.py`, `risk.py`, `statistics.py`, etc. - Financial computation modules
  - `interface.py` - Unified interface for financial tools
  - `data_loader.py` - Utilities for loading market data
  - `y9c_anomaly_detector.py` - Bank regulatory filing analysis
  - `sec_anomaly_detector.py` - SEC 10-K filing analysis
  - `regulatory_analysis.py` - Unified interface for regulatory analysis
- `data/` - Market data directory structure:
  - `stocks/` - Historical stock price data in CSV format
  - `options/` - Options chain data in JSON format
  - `news/` - Financial news and sentiment data in JSON format
  - `regulatory/` - Regulatory filing data:
    - `y9c/` - FR Y-9C bank filing data
    - `10k/` - SEC 10-K filing data

## Extending the System

The platform can be extended in several ways:

1. Add new trading strategies in `tasks.py`
2. Connect to real brokerage APIs by extending `market_connector.py`
3. Implement additional verification metrics
4. Enhance the dashboard with more visualizations
5. Expand the financial tools module with additional analytics
6. Integrate machine learning models for prediction

## License

MIT