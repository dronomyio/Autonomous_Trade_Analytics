# Data Directory

This directory contains market data for the Trade Analytics platform.

## Directory Structure

- `/stocks` - Historical and real-time stock price data
- `/options` - Options chain data and derivatives information
- `/news` - Financial news data and sentiment analysis

## Data Formats

### Stocks Data

Stock data should be stored in CSV format with the following structure:

```
date,open,high,low,close,volume,adj_close
2023-01-01,150.23,153.45,149.01,152.35,5000000,152.35
```

Recommended filename format: `{TICKER}_{TIMEFRAME}.csv`
Example: `AAPL_daily.csv`, `MSFT_hourly.csv`

### Options Data

Options data should be stored in JSON format with the following structure:

```json
{
  "underlying": "AAPL",
  "date": "2023-01-01",
  "expiration_dates": ["2023-02-15", "2023-03-15"],
  "chains": {
    "2023-02-15": {
      "calls": [
        {"strike": 150, "price": 5.65, "volume": 1200, "open_interest": 5000, "iv": 0.35},
        {"strike": 155, "price": 3.25, "volume": 800, "open_interest": 3500, "iv": 0.32}
      ],
      "puts": [
        {"strike": 150, "price": 4.20, "volume": 900, "open_interest": 4200, "iv": 0.33},
        {"strike": 145, "price": 2.10, "volume": 600, "open_interest": 2800, "iv": 0.30}
      ]
    }
  }
}
```

Recommended filename format: `{TICKER}_options_{DATE}.json`
Example: `AAPL_options_2023-01-01.json`

### News Data

News data should be stored in JSON format with the following structure:

```json
{
  "articles": [
    {
      "headline": "Apple Reports Record Q4 Earnings",
      "source": "Financial Times",
      "date": "2023-01-01T14:30:00Z",
      "url": "https://example.com/article1",
      "summary": "Apple Inc. reported record quarterly earnings...",
      "sentiment": 0.75,
      "tickers": ["AAPL"]
    }
  ]
}
```

Recommended filename format: `news_{DATE}.json` or `{TICKER}_news_{DATE}.json`
Example: `news_2023-01-01.json`, `AAPL_news_2023-01-01.json`

## Using Data in the Platform

The Trade Analytics platform can load and analyze data from these directories. To use custom data:

1. Place your data files in the appropriate subdirectory
2. The platform will automatically detect and use the most recent data files
3. For backtesting, specify the date range in the trading agent parameters

## Adding New Data

You can add new data through:

1. Manual uploads to these directories
2. API integrations (see `market_connector.py` for implementation)
3. Data scrapers (custom implementation required)

When adding new data, maintain the file naming conventions to ensure the platform can properly identify and use the data.