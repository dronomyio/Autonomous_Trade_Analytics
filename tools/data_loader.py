"""
Data loader for the Trade Analytics platform.
Utilities for loading and processing stock, options, and news data.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

# Try to import pandas, but don't fail if it's not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Some functionality will be limited.")
    
    # Create a minimal pandas replacement for basic operations
    class DummyPandas:
        class Series:
            def __init__(self, data=None):
                self.data = data or []
                self.empty = len(self.data) == 0
                
            def __len__(self):
                return len(self.data)
                
        class DataFrame:
            def __init__(self):
                self.empty = True
                self.columns = []
                
            def __len__(self):
                return 0
    
    if not PANDAS_AVAILABLE:
        pd = DummyPandas()

class DataLoader:
    """
    Loads and processes data from the data directory for use with the financial tools.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        if data_dir is None:
            # Use absolute path calculation
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(root_dir, "data")
        else:
            self.data_dir = data_dir
            
        self.stocks_dir = os.path.join(self.data_dir, "stocks")
        self.options_dir = os.path.join(self.data_dir, "options")
        self.news_dir = os.path.join(self.data_dir, "news")
    
    def list_available_stocks(self) -> List[str]:
        """
        List all available stock tickers in the data directory.
        
        Returns:
            List of stock tickers
        """
        if not os.path.exists(self.stocks_dir):
            return []
            
        tickers = set()
        for filename in os.listdir(self.stocks_dir):
            if filename.endswith(".csv"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    tickers.add(parts[0])
        
        return list(tickers)
    
    def load_stock_data(self, ticker: str, timeframe: str = "daily") -> Union[pd.DataFrame, Dict]:
        """
        Load stock price data for a given ticker and timeframe.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe of the data (daily, hourly, etc.)
            
        Returns:
            DataFrame with price data if pandas is available, otherwise a dictionary
        """
        filepath = os.path.join(self.stocks_dir, f"{ticker}_{timeframe}.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No {timeframe} data found for {ticker}")
        
        if not PANDAS_AVAILABLE:
            # Manual CSV parsing for when pandas is not available
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            header = lines[0].strip().split(',')
            
            # Parse data
            data = []
            for line in lines[1:]:
                values = line.strip().split(',')
                row = {header[i]: values[i] for i in range(len(header))}
                data.append(row)
            
            return {
                'data': data,
                'columns': header,
                'ticker': ticker,
                'timeframe': timeframe
            }
        else:
            # Use pandas for more efficient data handling
            df = pd.read_csv(filepath)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
    
    def load_options_data(self, ticker: str, date: Optional[str] = None) -> Dict:
        """
        Load options data for a given ticker and date.
        
        Args:
            ticker: Stock ticker symbol
            date: Date string in YYYY-MM-DD format (uses most recent if None)
            
        Returns:
            Dictionary with options data
        """
        if date:
            filepath = os.path.join(self.options_dir, f"{ticker}_options_{date}.json")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No options data found for {ticker} on {date}")
        else:
            # Find the most recent file
            files = [f for f in os.listdir(self.options_dir) 
                    if f.startswith(f"{ticker}_options_") and f.endswith(".json")]
            
            if not files:
                raise FileNotFoundError(f"No options data found for {ticker}")
            
            # Sort by date (newest first)
            files.sort(reverse=True)
            filepath = os.path.join(self.options_dir, files[0])
        
        # Load the data
        with open(filepath, 'r') as f:
            options_data = json.load(f)
        
        return options_data
    
    def load_news_data(self, date: Optional[str] = None, ticker: Optional[str] = None) -> List[Dict]:
        """
        Load news data for a given date and/or ticker.
        
        Args:
            date: Date string in YYYY-MM-DD format (uses most recent if None)
            ticker: Filter news for a specific ticker
            
        Returns:
            List of news articles
        """
        if date and ticker:
            # Try ticker-specific news first
            filepath = os.path.join(self.news_dir, f"{ticker}_news_{date}.json")
            if not os.path.exists(filepath):
                # Fall back to general news
                filepath = os.path.join(self.news_dir, f"news_{date}.json")
        elif date:
            filepath = os.path.join(self.news_dir, f"news_{date}.json")
        elif ticker:
            # Find the most recent ticker-specific news
            files = [f for f in os.listdir(self.news_dir) 
                    if f.startswith(f"{ticker}_news_") and f.endswith(".json")]
            
            if not files:
                # Find the most recent general news
                files = [f for f in os.listdir(self.news_dir) 
                        if f.startswith("news_") and f.endswith(".json")]
                
                if not files:
                    raise FileNotFoundError(f"No news data found")
            
            # Sort by date (newest first)
            files.sort(reverse=True)
            filepath = os.path.join(self.news_dir, files[0])
        else:
            # Find the most recent general news
            files = [f for f in os.listdir(self.news_dir) 
                    if f.startswith("news_") and f.endswith(".json")]
            
            if not files:
                raise FileNotFoundError(f"No news data found")
            
            # Sort by date (newest first)
            files.sort(reverse=True)
            filepath = os.path.join(self.news_dir, files[0])
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No news data found for the specified criteria")
        
        # Load the data
        with open(filepath, 'r') as f:
            news_data = json.load(f)
        
        # Filter for ticker if specified
        if ticker and 'articles' in news_data:
            articles = []
            for article in news_data['articles']:
                if 'tickers' in article and ticker in article['tickers']:
                    articles.append(article)
            
            return articles
        elif 'articles' in news_data:
            return news_data['articles']
        else:
            return []
    
    def get_stock_returns(self, ticker: str, timeframe: str = "daily") -> pd.Series:
        """
        Get log returns for a stock.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe of the data (daily, hourly, etc.)
            
        Returns:
            Series of log returns
        """
        from tools.returns import log_return
        
        # Load price data
        prices = self.load_stock_data(ticker, timeframe)
        
        # Calculate log returns
        returns = log_return(prices['close'])
        
        return returns
    
    def analyze_stock(self, ticker: str, timeframe: str = "daily") -> Dict:
        """
        Run full analysis on a stock using the financial toolkit.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe of the data (daily, hourly, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        from tools.interface import FinancialToolkit
        
        # Initialize the financial toolkit
        toolkit = FinancialToolkit()
        
        # Load price data
        prices = self.load_stock_data(ticker, timeframe)
        
        # Calculate returns metrics
        returns_analysis = toolkit.analyze_returns(prices['close'])
        
        # Generate technical indicators
        indicators = toolkit.generate_technical_indicators(
            prices['close'],
            prices['high'] if 'high' in prices.columns else None,
            prices['low'] if 'low' in prices.columns else None
        )
        
        # Get the most recent values for key indicators
        latest_indicators = {}
        for name, series in indicators.items():
            if not series.empty:
                latest_indicators[name] = series.iloc[-1]
        
        # Return combined analysis
        return {
            'ticker': ticker,
            'timeframe': timeframe,
            'current_price': prices['close'].iloc[-1],
            'returns_analysis': returns_analysis,
            'latest_indicators': latest_indicators
        }


# Example usage
if __name__ == "__main__":
    def simple_example():
        """Run a simple example that doesn't require financial tools"""
        loader = DataLoader()
        
        print(f"Data directory: {loader.data_dir}")
        print(f"Stocks directory: {loader.stocks_dir}")
        print(f"Options directory: {loader.options_dir}")
        print(f"News directory: {loader.news_dir}")
        
        # List available stocks
        try:
            tickers = loader.list_available_stocks()
            print(f"\nAvailable tickers: {tickers}")
            
            if tickers:
                ticker = tickers[0]
                
                # Try loading stock data
                try:
                    stock_data = loader.load_stock_data(ticker)
                    print(f"\nStock data for {ticker}:")
                    
                    if PANDAS_AVAILABLE:
                        print(f"First date: {stock_data.index[0].strftime('%Y-%m-%d')}")
                        print(f"Last date: {stock_data.index[-1].strftime('%Y-%m-%d')}")
                        print(f"Latest price: ${stock_data['close'].iloc[-1]:.2f}")
                        print(f"Available columns: {stock_data.columns.tolist()}")
                    else:
                        print(f"First date: {stock_data['data'][0]['date']}")
                        print(f"Last date: {stock_data['data'][-1]['date']}")
                        print(f"Latest price: ${float(stock_data['data'][-1]['close']):.2f}")
                        print(f"Available columns: {stock_data['columns']}")
                except FileNotFoundError as e:
                    print(f"\nCould not load stock data: {str(e)}")
                
                # Try loading options data
                try:
                    options = loader.load_options_data(ticker)
                    print(f"\nOptions data for {ticker}:")
                    print(f"Date: {options['date']}")
                    print(f"Underlying price: ${options['underlying_price']:.2f}")
                    print(f"Available expiry dates: {options['expiration_dates']}")
                except FileNotFoundError as e:
                    print(f"\nCould not load options data: {str(e)}")
                
                # Try loading news data
                try:
                    news = loader.load_news_data(ticker=ticker)
                    print(f"\nNews for {ticker}:")
                    print(f"Found {len(news)} articles")
                    if news:
                        article = news[0]
                        print(f"Latest headline: {article['headline']}")
                        print(f"Source: {article['source']}")
                        print(f"Date: {article['date']}")
                except FileNotFoundError as e:
                    print(f"\nCould not load news data: {str(e)}")
        
        except Exception as e:
            print(f"Error during simple example: {str(e)}")
    
    def full_example():
        """Run a full example that uses financial tools"""
        try:
            # Import required modules for analysis
            from tools.interface import FinancialToolkit
            import pandas as pd
            
            loader = DataLoader()
            
            # List available stocks
            tickers = loader.list_available_stocks()
            print(f"Available tickers: {tickers}")
            
            if tickers:
                # Analyze the first available stock
                ticker = tickers[0]
                analysis = loader.analyze_stock(ticker)
                
                print(f"\nAnalysis for {ticker}:")
                print(f"Current price: ${analysis['current_price']:.2f}")
                print("\nReturns Analysis:")
                for metric, value in analysis['returns_analysis'].items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value*100:.2f}%")
                    else:
                        print(f"  {metric}: {value}")
                
                print("\nTechnical Indicators:")
                for indicator, value in analysis['latest_indicators'].items():
                    if not pd.isna(value):
                        print(f"  {indicator}: {value:.4f}")
                
                # Load options data if available
                try:
                    options = loader.load_options_data(ticker)
                    expiry = options['expiration_dates'][0]
                    calls = options['chains'][expiry]['calls']
                    puts = options['chains'][expiry]['puts']
                    
                    atm_call = next((c for c in calls if c['strike'] >= analysis['current_price']), calls[0])
                    atm_put = next((p for p in puts if p['strike'] <= analysis['current_price']), puts[0])
                    
                    print(f"\nOptions (Expiry: {expiry}):")
                    print(f"  ATM Call (Strike: ${atm_call['strike']}): ${atm_call['price']:.2f} (IV: {atm_call['iv']*100:.1f}%)")
                    print(f"  ATM Put (Strike: ${atm_put['strike']}): ${atm_put['price']:.2f} (IV: {atm_put['iv']*100:.1f}%)")
                except FileNotFoundError:
                    print("\nNo options data available")
                
                # Load news data if available
                try:
                    news = loader.load_news_data(ticker=ticker)
                    if news:
                        print("\nRecent News:")
                        for article in news[:3]:  # Show first 3 articles
                            print(f"  {article['headline']}")
                            print(f"  Source: {article['source']} | Sentiment: {article['sentiment']:.2f}")
                            print(f"  {article['summary'][:100]}...\n")
                except FileNotFoundError:
                    print("\nNo news data available")
        except ImportError:
            print("Could not run full example - required modules not available")
            print("Make sure pandas and other dependencies are installed")
        except Exception as e:
            print(f"Error during full example: {str(e)}")
    
    # Run the simple example by default
    print("Running simple example (no financial tools required)...")
    simple_example()
    
    # Try running the full example
    print("\n\nAttempting to run full example (requires financial tools)...")
    full_example()