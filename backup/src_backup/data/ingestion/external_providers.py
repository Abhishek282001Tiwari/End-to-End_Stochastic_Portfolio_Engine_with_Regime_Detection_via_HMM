import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import time
from abc import ABC, abstractmethod
import requests
from functools import wraps
import sqlite3
import os
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)


def rate_limited(calls_per_second: float = 1.0):
    """Decorator to rate limit API calls"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            ret = await func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def get_historical_prices(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        pass
    
    @abstractmethod
    async def get_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        pass


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage API provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://www.alphavantage.co/query")
        
    @rate_limited(5.0)  # 5 calls per second limit
    async def get_historical_prices(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        logger.info(f"Fetching historical prices for {len(symbols)} symbols from Alpha Vantage")
        
        all_data = {}
        
        for symbol in symbols:
            try:
                params = {
                    'function': 'TIME_SERIES_DAILY_ADJUSTED',
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
                
                async with self.session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    if 'Time Series (Daily)' in data:
                        time_series = data['Time Series (Daily)']
                        
                        df = pd.DataFrame.from_dict(time_series, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df = df.astype(float)
                        
                        # Rename columns
                        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
                        
                        # Filter by date range
                        df = df[(df.index >= start_date) & (df.index <= end_date)]
                        df = df.sort_index()
                        
                        # Create multi-level columns
                        for col in df.columns:
                            all_data[(col, symbol)] = df[col]
                    
                    elif 'Note' in data:
                        logger.warning(f"API rate limit reached for {symbol}")
                        await asyncio.sleep(60)  # Wait 1 minute
                    else:
                        logger.error(f"Error fetching data for {symbol}: {data}")
                        
            except Exception as e:
                logger.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df.columns = pd.MultiIndex.from_tuples(result_df.columns)
            return result_df
        else:
            return pd.DataFrame()
    
    @rate_limited(5.0)
    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        logger.info(f"Fetching real-time quotes for {len(symbols)} symbols")
        
        quotes = {}
        
        for symbol in symbols:
            try:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                async with self.session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    if 'Global Quote' in data:
                        quote_data = data['Global Quote']
                        quotes[symbol] = {
                            'price': float(quote_data.get('05. price', 0)),
                            'change': float(quote_data.get('09. change', 0)),
                            'change_percent': float(quote_data.get('10. change percent', '0%').rstrip('%')),
                            'volume': int(quote_data.get('06. volume', 0))
                        }
                        
            except Exception as e:
                logger.error(f"Error fetching real-time quote for {symbol}: {e}")
        
        return quotes
    
    @rate_limited(5.0)
    async def get_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        logger.info(f"Fetching fundamental data for {len(symbols)} symbols")
        
        fundamentals = {}
        
        for symbol in symbols:
            try:
                params = {
                    'function': 'OVERVIEW',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                async with self.session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    if 'Symbol' in data:
                        fundamentals[symbol] = {
                            'market_cap': self._parse_number(data.get('MarketCapitalization', '0')),
                            'pe_ratio': self._parse_number(data.get('PERatio', '0')),
                            'dividend_yield': self._parse_number(data.get('DividendYield', '0')),
                            'beta': self._parse_number(data.get('Beta', '0')),
                            'eps': self._parse_number(data.get('EPS', '0')),
                            'book_value': self._parse_number(data.get('BookValue', '0')),
                            'sector': data.get('Sector', ''),
                            'industry': data.get('Industry', '')
                        }
                        
            except Exception as e:
                logger.error(f"Error fetching fundamental data for {symbol}: {e}")
        
        return fundamentals
    
    def _parse_number(self, value: str) -> float:
        """Parse number strings that may contain 'None' or be empty"""
        try:
            if value == 'None' or value == '' or value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0


class QuandlProvider(MarketDataProvider):
    """Quandl API provider for economic and alternative data"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://www.quandl.com/api/v3")
        
    async def get_economic_indicators(
        self, 
        indicators: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch economic indicators from Quandl"""
        logger.info(f"Fetching economic indicators: {indicators}")
        
        all_data = {}
        
        indicator_mapping = {
            'gdp': 'FRED/GDP',
            'inflation': 'FRED/CPIAUCSL',
            'unemployment': 'FRED/UNRATE',
            'fed_funds': 'FRED/FEDFUNDS',
            'vix': 'CBOE/VIX',
            'yield_10y': 'FRED/GS10',
            'yield_2y': 'FRED/GS2'
        }
        
        for indicator in indicators:
            quandl_code = indicator_mapping.get(indicator)
            if not quandl_code:
                logger.warning(f"Unknown indicator: {indicator}")
                continue
                
            try:
                params = {
                    'api_key': self.api_key,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                }
                
                url = f"{self.base_url}/datasets/{quandl_code}/data.json"
                
                async with self.session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if 'dataset_data' in data:
                        dataset = data['dataset_data']
                        df = pd.DataFrame(dataset['data'], columns=dataset['column_names'])
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Use the value column (typically the second column)
                        if len(df.columns) > 0:
                            all_data[indicator] = df.iloc[:, 0]
                            
            except Exception as e:
                logger.error(f"Error fetching {indicator} from Quandl: {e}")
        
        if all_data:
            return pd.DataFrame(all_data)
        else:
            return pd.DataFrame()
    
    async def get_historical_prices(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        # Quandl doesn't provide equity prices in the same way
        return pd.DataFrame()
    
    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        return {}
    
    async def get_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        return {}


class AlternativeDataProvider(MarketDataProvider):
    """Provider for alternative data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    async def get_sentiment_data(
        self, 
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get sentiment data for symbols"""
        logger.info(f"Fetching sentiment data for {len(symbols)} symbols")
        
        # Simulate sentiment data - in production, this would connect to real sentiment APIs
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        sentiment_data = {}
        for symbol in symbols:
            # Generate realistic sentiment scores (-1 to 1)
            np.random.seed(hash(symbol) % 2**32)
            sentiment_scores = np.random.normal(0, 0.3, len(date_range))
            sentiment_scores = np.clip(sentiment_scores, -1, 1)
            
            sentiment_data[f"{symbol}_sentiment"] = sentiment_scores
        
        return pd.DataFrame(sentiment_data, index=date_range)
    
    async def get_news_analytics(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get news analytics data"""
        logger.info(f"Fetching news analytics for {len(symbols)} symbols")
        
        # Simulate news analytics - in production, connect to news APIs
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        news_data = {}
        for symbol in symbols:
            np.random.seed(hash(symbol + "news") % 2**32)
            
            # News volume (0-100)
            news_volume = np.random.poisson(10, len(date_range))
            
            # News sentiment (-1 to 1)
            news_sentiment = np.random.normal(0, 0.2, len(date_range))
            news_sentiment = np.clip(news_sentiment, -1, 1)
            
            news_data[f"{symbol}_news_volume"] = news_volume
            news_data[f"{symbol}_news_sentiment"] = news_sentiment
        
        return pd.DataFrame(news_data, index=date_range)
    
    async def get_satellite_data(
        self,
        regions: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get satellite-based economic indicators"""
        logger.info(f"Fetching satellite data for regions: {regions}")
        
        # Simulate satellite data - economic activity indicators
        date_range = pd.date_range(start_date, end_date, freq='W')
        
        satellite_data = {}
        for region in regions:
            np.random.seed(hash(region + "satellite") % 2**32)
            
            # Economic activity index
            activity_index = np.random.normal(100, 10, len(date_range))
            activity_index = np.maximum(activity_index, 0)
            
            satellite_data[f"{region}_economic_activity"] = activity_index
        
        return pd.DataFrame(satellite_data, index=date_range)
    
    async def get_historical_prices(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        return pd.DataFrame()
    
    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        return {}
    
    async def get_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        return {}


class DataCache:
    """Simple SQLite-based cache for market data"""
    
    def __init__(self, cache_file: str = "data/cache/market_data.db"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database"""
        conn = sqlite3.connect(str(self.cache_file))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_cache (
                symbol TEXT,
                date TEXT,
                data_type TEXT,
                data TEXT,
                timestamp TEXT,
                PRIMARY KEY (symbol, date, data_type)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_cached_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        data_type: str = "prices"
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data"""
        try:
            conn = sqlite3.connect(str(self.cache_file))
            
            query = '''
                SELECT date, data FROM price_cache 
                WHERE symbol = ? AND data_type = ? 
                AND date >= ? AND date <= ?
                ORDER BY date
            '''
            
            result = conn.execute(
                query, 
                (symbol, data_type, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            ).fetchall()
            
            conn.close()
            
            if result:
                data_dict = {}
                for date_str, data_json in result:
                    data_dict[date_str] = json.loads(data_json)
                
                if data_dict:
                    df = pd.DataFrame.from_dict(data_dict, orient='index')
                    df.index = pd.to_datetime(df.index)
                    return df
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
        
        return None
    
    def cache_data(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        data_type: str = "prices"
    ):
        """Cache data to database"""
        try:
            conn = sqlite3.connect(str(self.cache_file))
            cursor = conn.cursor()
            
            for date, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO price_cache 
                    (symbol, date, data_type, data, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    data_type,
                    json.dumps(row.to_dict()),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")


class IntegratedDataManager:
    """Integrated manager for all data providers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.cache = DataCache()
        
        # Initialize providers based on configuration
        self.providers = {}
        
        if self.config.data.sources.get('alpha_vantage') and os.getenv('ALPHA_VANTAGE_API_KEY'):
            self.providers['alpha_vantage'] = AlphaVantageProvider(os.getenv('ALPHA_VANTAGE_API_KEY'))
        
        if self.config.data.sources.get('quandl') and os.getenv('QUANDL_API_KEY'):
            self.providers['quandl'] = QuandlProvider(os.getenv('QUANDL_API_KEY'))
        
        # Always include alternative data provider
        self.providers['alternative'] = AlternativeDataProvider(self.config.data)
        
    async def get_comprehensive_dataset(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        include_alternative: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Get comprehensive dataset from all available sources"""
        logger.info("Fetching comprehensive dataset from all providers")
        
        datasets = {}
        
        # Get price data from primary provider
        primary_provider = None
        for provider_name in ['alpha_vantage', 'yahoo_finance']:
            if provider_name in self.providers:
                primary_provider = self.providers[provider_name]
                break
        
        if primary_provider:
            async with primary_provider:
                price_data = await primary_provider.get_historical_prices(symbols, start_date, end_date)
                if not price_data.empty:
                    datasets['prices'] = price_data
        
        # Get economic indicators
        if 'quandl' in self.providers:
            async with self.providers['quandl']:
                economic_data = await self.providers['quandl'].get_economic_indicators(
                    ['gdp', 'inflation', 'unemployment', 'fed_funds', 'vix'],
                    start_date, end_date
                )
                if not economic_data.empty:
                    datasets['economic'] = economic_data
        
        # Get alternative data
        if include_alternative and 'alternative' in self.providers:
            async with self.providers['alternative']:
                sentiment_data = await self.providers['alternative'].get_sentiment_data(
                    symbols, start_date, end_date
                )
                if not sentiment_data.empty:
                    datasets['sentiment'] = sentiment_data
                
                news_data = await self.providers['alternative'].get_news_analytics(
                    symbols, start_date, end_date
                )
                if not news_data.empty:
                    datasets['news'] = news_data
        
        return datasets
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from all available sources"""
        logger.info("Fetching real-time data")
        
        real_time_data = {
            'quotes': {},
            'fundamentals': {},
            'timestamp': datetime.now()
        }
        
        # Get real-time quotes
        for provider_name, provider in self.providers.items():
            if hasattr(provider, 'get_real_time_quotes'):
                try:
                    async with provider:
                        quotes = await provider.get_real_time_quotes(symbols)
                        real_time_data['quotes'].update(quotes)
                except Exception as e:
                    logger.error(f"Error getting real-time quotes from {provider_name}: {e}")
        
        return real_time_data
    
    def create_unified_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create unified feature set from multiple data sources"""
        logger.info("Creating unified feature set")
        
        # Start with price-based features
        if 'prices' in datasets:
            price_data = datasets['prices']
            
            # Calculate returns
            adj_close = price_data.xs('Adj Close', level=0, axis=1, drop_level=False)
            returns = adj_close.pct_change()
            
            # Calculate volatility
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Market-wide features
            market_return = returns.mean(axis=1)
            market_volatility = volatility.mean(axis=1)
            
            features = pd.DataFrame({
                'market_return': market_return,
                'market_volatility': market_volatility
            })
        else:
            features = pd.DataFrame()
        
        # Add economic features
        if 'economic' in datasets:
            economic_data = datasets['economic']
            
            # Resample to daily frequency and forward fill
            economic_daily = economic_data.resample('D').ffill()
            
            # Add to features
            for col in economic_daily.columns:
                if not features.empty:
                    features = features.join(economic_daily[col], how='outer')
                else:
                    features = economic_daily[[col]].copy()
        
        # Add sentiment features
        if 'sentiment' in datasets:
            sentiment_data = datasets['sentiment']
            avg_sentiment = sentiment_data.mean(axis=1)
            
            if not features.empty:
                features = features.join(avg_sentiment.rename('avg_sentiment'), how='outer')
            else:
                features = pd.DataFrame({'avg_sentiment': avg_sentiment})
        
        # Add news features
        if 'news' in datasets:
            news_data = datasets['news']
            
            # Average news sentiment and volume
            news_sentiment_cols = [col for col in news_data.columns if 'sentiment' in col]
            news_volume_cols = [col for col in news_data.columns if 'volume' in col]
            
            if news_sentiment_cols:
                avg_news_sentiment = news_data[news_sentiment_cols].mean(axis=1)
                if not features.empty:
                    features = features.join(avg_news_sentiment.rename('avg_news_sentiment'), how='outer')
                else:
                    features = pd.DataFrame({'avg_news_sentiment': avg_news_sentiment})
            
            if news_volume_cols:
                avg_news_volume = news_data[news_volume_cols].mean(axis=1)
                if not features.empty:
                    features = features.join(avg_news_volume.rename('avg_news_volume'), how='outer')
                elif 'avg_news_sentiment' in locals():
                    features = pd.DataFrame({
                        'avg_news_sentiment': avg_news_sentiment,
                        'avg_news_volume': avg_news_volume
                    })
                else:
                    features = pd.DataFrame({'avg_news_volume': avg_news_volume})
        
        # Forward fill missing values and drop NaN
        features = features.fillna(method='ffill').dropna()
        
        logger.info(f"Created unified feature set with {len(features.columns)} features and {len(features)} observations")
        
        return features


async def create_integrated_data_manager(config: Optional[Dict[str, Any]] = None) -> IntegratedDataManager:
    """Factory function to create integrated data manager"""
    return IntegratedDataManager(config)


# Example usage and testing functions
async def test_data_providers():
    """Test all data providers"""
    config = get_config()
    manager = IntegratedDataManager(config)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # Test comprehensive dataset
    datasets = await manager.get_comprehensive_dataset(symbols, start_date, end_date)
    
    print("Available datasets:")
    for name, data in datasets.items():
        print(f"  {name}: {data.shape}")
    
    # Test unified features
    features = manager.create_unified_features(datasets)
    print(f"\nUnified features: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Test real-time data
    real_time = await manager.get_real_time_data(symbols)
    print(f"\nReal-time quotes: {len(real_time['quotes'])} symbols")


if __name__ == "__main__":
    asyncio.run(test_data_providers())