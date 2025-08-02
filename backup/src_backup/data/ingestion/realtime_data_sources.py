#!/usr/bin/env python3

import asyncio
import aiohttp
import websockets
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from queue import Queue, Empty
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class DataSourceType(Enum):
    """Types of data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON_IO = "polygon_io"
    BLOOMBERG_API = "bloomberg_api"
    FINNHUB = "finnhub"
    TWELVE_DATA = "twelve_data"
    IEX_CLOUD = "iex_cloud"
    WEBSOCKET_STREAM = "websocket_stream"


class DataFrequency(Enum):
    """Data update frequencies"""
    REAL_TIME = "real_time"
    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    FIFTEEN_MINUTE = "15min"
    HOURLY = "1hour"
    DAILY = "1day"
    WEEKLY = "1week"
    MONTHLY = "1month"


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    name: str
    source_type: DataSourceType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3
    websocket_url: Optional[str] = None
    enable_streaming: bool = False
    subscription_symbols: List[str] = field(default_factory=list)


@dataclass
class MarketDataPoint:
    """Individual market data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str
    additional_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingDataCallback:
    """Callback configuration for streaming data"""
    callback_function: Callable[[MarketDataPoint], None]
    symbols: List[str]
    frequency: DataFrequency


class BaseRealTimeDataSource(ABC):
    """Abstract base class for real-time data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.is_connected = False
        self.last_request_time = 0
        self.request_count = 0
        self.data_queue = Queue()
        self.callbacks = []
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        pass
    
    @abstractmethod
    async def fetch_latest_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch latest data for symbols"""
        pass
    
    @abstractmethod
    async def fetch_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Fetch historical data"""
        pass
    
    @abstractmethod
    async def start_streaming(self, symbols: List[str], callback: Callable[[MarketDataPoint], None]):
        """Start streaming real-time data"""
        pass
    
    @abstractmethod
    async def stop_streaming(self):
        """Stop streaming data"""
        pass
    
    def add_callback(self, callback: StreamingDataCallback):
        """Add streaming data callback"""
        self.callbacks.append(callback)
    
    def _check_rate_limit(self):
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset counter if minute has passed
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        if self.request_count >= self.config.rate_limit:
            sleep_time = 60 - (current_time - self.last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1


class YahooFinanceRealTimeSource(BaseRealTimeDataSource):
    """Enhanced Yahoo Finance data source with real-time capabilities"""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        if config is None:
            config = DataSourceConfig(
                name="yahoo_finance_rt",
                source_type=DataSourceType.YAHOO_FINANCE,
                rate_limit=200,
                enable_streaming=False
            )
        super().__init__(config)
        
    async def connect(self) -> bool:
        """Yahoo Finance doesn't require explicit connection"""
        self.is_connected = True
        logger.info("Connected to Yahoo Finance data source")
        return True
    
    async def disconnect(self):
        """Disconnect from Yahoo Finance"""
        self.is_connected = False
        logger.info("Disconnected from Yahoo Finance")
    
    async def fetch_latest_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch latest data using yfinance"""
        self._check_rate_limit()
        
        try:
            data_points = []
            
            # Use yfinance to get real-time data
            tickers = yf.Tickers(" ".join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.history(period="1d", interval="1m").tail(1)
                    
                    if not info.empty:
                        row = info.iloc[0]
                        data_point = MarketDataPoint(
                            symbol=symbol,
                            timestamp=info.index[0].to_pydatetime(),
                            open=row['Open'],
                            high=row['High'],
                            low=row['Low'],
                            close=row['Close'],
                            volume=int(row['Volume']),
                            source="yahoo_finance"
                        )
                        data_points.append(data_point)
                        
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol}: {e}")
                    continue
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching latest data from Yahoo Finance: {e}")
            return []
    
    async def fetch_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        
        interval_map = {
            DataFrequency.MINUTE: "1m",
            DataFrequency.FIVE_MINUTE: "5m", 
            DataFrequency.FIFTEEN_MINUTE: "15m",
            DataFrequency.HOURLY: "1h",
            DataFrequency.DAILY: "1d",
            DataFrequency.WEEKLY: "1wk",
            DataFrequency.MONTHLY: "1mo"
        }
        
        interval = interval_map.get(frequency, "1d")
        
        try:
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                threads=True
            )
            
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, symbols])
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    async def start_streaming(self, symbols: List[str], callback: Callable[[MarketDataPoint], None]):
        """Start pseudo-streaming by polling at regular intervals"""
        logger.info(f"Starting pseudo-streaming for {len(symbols)} symbols")
        
        async def stream_loop():
            while self.is_connected:
                try:
                    data_points = await self.fetch_latest_data(symbols)
                    for point in data_points:
                        callback(point)
                    
                    await asyncio.sleep(60)  # Update every minute
                    
                except Exception as e:
                    logger.error(f"Error in streaming loop: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(stream_loop())
    
    async def stop_streaming(self):
        """Stop streaming"""
        self.is_connected = False
        logger.info("Stopped streaming from Yahoo Finance")


class AlphaVantageRealTimeSource(BaseRealTimeDataSource):
    """Alpha Vantage data source with real-time capabilities"""
    
    def __init__(self, api_key: str):
        config = DataSourceConfig(
            name="alpha_vantage_rt",
            source_type=DataSourceType.ALPHA_VANTAGE,
            api_key=api_key,
            base_url="https://www.alphavantage.co/query",
            rate_limit=5,  # 5 requests per minute for free tier
            enable_streaming=False
        )
        super().__init__(config)
        
    async def connect(self) -> bool:
        """Test connection to Alpha Vantage"""
        try:
            # Test with a simple API call
            url = f"{self.config.base_url}?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&apikey={self.config.api_key}&outputsize=compact"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "Error Message" not in data and "Note" not in data:
                            self.is_connected = True
                            logger.info("Successfully connected to Alpha Vantage")
                            return True
            
            logger.error("Failed to connect to Alpha Vantage")
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to Alpha Vantage: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpha Vantage"""
        self.is_connected = False
        logger.info("Disconnected from Alpha Vantage")
    
    async def fetch_latest_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch latest data from Alpha Vantage"""
        self._check_rate_limit()
        
        data_points = []
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    url = f"{self.config.base_url}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.config.api_key}&outputsize=compact"
                    
                    async with session.get(url, timeout=self.config.timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if "Time Series (1min)" in data:
                                time_series = data["Time Series (1min)"]
                                latest_time = max(time_series.keys())
                                latest_data = time_series[latest_time]
                                
                                data_point = MarketDataPoint(
                                    symbol=symbol,
                                    timestamp=datetime.strptime(latest_time, "%Y-%m-%d %H:%M:%S"),
                                    open=float(latest_data["1. open"]),
                                    high=float(latest_data["2. high"]),
                                    low=float(latest_data["3. low"]),
                                    close=float(latest_data["4. close"]),
                                    volume=int(latest_data["5. volume"]),
                                    source="alpha_vantage"
                                )
                                data_points.append(data_point)
                    
                    # Rate limiting
                    await asyncio.sleep(12)  # 5 requests per minute = 12 seconds between requests
                    
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol} from Alpha Vantage: {e}")
                    continue
        
        return data_points
    
    async def fetch_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Fetch historical data from Alpha Vantage"""
        
        function_map = {
            DataFrequency.MINUTE: "TIME_SERIES_INTRADAY",
            DataFrequency.DAILY: "TIME_SERIES_DAILY_ADJUSTED",
            DataFrequency.WEEKLY: "TIME_SERIES_WEEKLY_ADJUSTED",
            DataFrequency.MONTHLY: "TIME_SERIES_MONTHLY_ADJUSTED"
        }
        
        function = function_map.get(frequency, "TIME_SERIES_DAILY_ADJUSTED")
        all_data = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    url = f"{self.config.base_url}?function={function}&symbol={symbol}&apikey={self.config.api_key}&outputsize=full"
                    
                    if frequency == DataFrequency.MINUTE:
                        url += "&interval=1min"
                    
                    async with session.get(url, timeout=self.config.timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Parse based on function type
                            if "Time Series" in str(data):
                                for key, value in data.items():
                                    if "Time Series" in key:
                                        time_series = value
                                        break
                                
                                df_data = []
                                for date_str, values in time_series.items():
                                    date = datetime.strptime(date_str, "%Y-%m-%d" if frequency != DataFrequency.MINUTE else "%Y-%m-%d %H:%M:%S")
                                    
                                    if start_date <= date <= end_date:
                                        df_data.append({
                                            'Date': date,
                                            'Open': float(values.get("1. open", values.get("2. open", 0))),
                                            'High': float(values.get("2. high", values.get("3. high", 0))),
                                            'Low': float(values.get("3. low", values.get("4. low", 0))),
                                            'Close': float(values.get("4. close", values.get("5. adjusted close", values.get("6. close", 0)))),
                                            'Volume': int(values.get("5. volume", values.get("6. volume", 0)))
                                        })
                                
                                if df_data:
                                    symbol_df = pd.DataFrame(df_data).set_index('Date').sort_index()
                                    all_data[symbol] = symbol_df
                    
                    await asyncio.sleep(12)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error fetching historical data for {symbol}: {e}")
                    continue
        
        if all_data:
            # Combine all symbols into multi-index DataFrame
            combined_df = pd.concat(all_data, axis=1)
            return combined_df
        
        return pd.DataFrame()
    
    async def start_streaming(self, symbols: List[str], callback: Callable[[MarketDataPoint], None]):
        """Start pseudo-streaming with Alpha Vantage"""
        logger.info(f"Starting pseudo-streaming for {len(symbols)} symbols via Alpha Vantage")
        
        async def stream_loop():
            while self.is_connected:
                try:
                    data_points = await self.fetch_latest_data(symbols)
                    for point in data_points:
                        callback(point)
                    
                    await asyncio.sleep(300)  # Update every 5 minutes due to rate limits
                    
                except Exception as e:
                    logger.error(f"Error in Alpha Vantage streaming loop: {e}")
                    await asyncio.sleep(300)
        
        asyncio.create_task(stream_loop())
    
    async def stop_streaming(self):
        """Stop streaming"""
        self.is_connected = False
        logger.info("Stopped streaming from Alpha Vantage")


class PolygonIOSource(BaseRealTimeDataSource):
    """Polygon.io data source with WebSocket streaming"""
    
    def __init__(self, api_key: str):
        config = DataSourceConfig(
            name="polygon_io",
            source_type=DataSourceType.POLYGON_IO,
            api_key=api_key,
            base_url="https://api.polygon.io",
            websocket_url="wss://socket.polygon.io/stocks",
            rate_limit=300,  # Depends on plan
            enable_streaming=True
        )
        super().__init__(config)
        self.websocket = None
        
    async def connect(self) -> bool:
        """Connect to Polygon.io"""
        try:
            # Test REST API connection
            url = f"{self.config.base_url}/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02?apikey={self.config.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.config.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "OK":
                            self.is_connected = True
                            logger.info("Successfully connected to Polygon.io")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to Polygon.io: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Polygon.io"""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("Disconnected from Polygon.io")
    
    async def fetch_latest_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch latest data from Polygon.io"""
        self._check_rate_limit()
        
        data_points = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    url = f"{self.config.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=desc&limit=1&apikey={self.config.api_key}"
                    
                    async with session.get(url, timeout=self.config.timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data.get("status") == "OK" and data.get("results"):
                                result = data["results"][0]
                                
                                data_point = MarketDataPoint(
                                    symbol=symbol,
                                    timestamp=datetime.fromtimestamp(result["t"] / 1000),
                                    open=result["o"],
                                    high=result["h"],
                                    low=result["l"],
                                    close=result["c"],
                                    volume=result["v"],
                                    source="polygon_io"
                                )
                                data_points.append(data_point)
                
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol} from Polygon.io: {e}")
                    continue
        
        return data_points
    
    async def fetch_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Fetch historical data from Polygon.io"""
        
        timespan_map = {
            DataFrequency.MINUTE: "minute",
            DataFrequency.FIVE_MINUTE: "minute",
            DataFrequency.HOURLY: "hour",
            DataFrequency.DAILY: "day",
            DataFrequency.WEEKLY: "week",
            DataFrequency.MONTHLY: "month"
        }
        
        multiplier_map = {
            DataFrequency.MINUTE: 1,
            DataFrequency.FIVE_MINUTE: 5,
            DataFrequency.HOURLY: 1,
            DataFrequency.DAILY: 1,
            DataFrequency.WEEKLY: 1,
            DataFrequency.MONTHLY: 1
        }
        
        timespan = timespan_map.get(frequency, "day")
        multiplier = multiplier_map.get(frequency, 1)
        
        all_data = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    url = f"{self.config.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apikey={self.config.api_key}"
                    
                    async with session.get(url, timeout=self.config.timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data.get("status") == "OK" and data.get("results"):
                                df_data = []
                                for result in data["results"]:
                                    df_data.append({
                                        'Date': datetime.fromtimestamp(result["t"] / 1000),
                                        'Open': result["o"],
                                        'High': result["h"],
                                        'Low': result["l"],
                                        'Close': result["c"],
                                        'Volume': result["v"]
                                    })
                                
                                if df_data:
                                    symbol_df = pd.DataFrame(df_data).set_index('Date').sort_index()
                                    all_data[symbol] = symbol_df
                
                except Exception as e:
                    logger.warning(f"Error fetching historical data for {symbol}: {e}")
                    continue
        
        if all_data:
            combined_df = pd.concat(all_data, axis=1)
            return combined_df
        
        return pd.DataFrame()
    
    async def start_streaming(self, symbols: List[str], callback: Callable[[MarketDataPoint], None]):
        """Start WebSocket streaming from Polygon.io"""
        logger.info(f"Starting WebSocket streaming for {len(symbols)} symbols via Polygon.io")
        
        try:
            self.websocket = await websockets.connect(
                f"{self.config.websocket_url}",
                extra_headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
            
            # Subscribe to symbols
            auth_message = {"action": "auth", "params": self.config.api_key}
            await self.websocket.send(json.dumps(auth_message))
            
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join([f"T.{symbol}" for symbol in symbols])
            }
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Listen for messages
            async def listen_loop():
                while self.is_connected and self.websocket:
                    try:
                        message = await self.websocket.recv()
                        data = json.loads(message)
                        
                        for item in data:
                            if item.get("ev") == "T":  # Trade event
                                data_point = MarketDataPoint(
                                    symbol=item["sym"],
                                    timestamp=datetime.fromtimestamp(item["t"] / 1000),
                                    open=item.get("o", item["p"]),
                                    high=item.get("h", item["p"]),
                                    low=item.get("l", item["p"]),
                                    close=item["p"],
                                    volume=item.get("s", 0),
                                    source="polygon_io",
                                    additional_fields={"price": item["p"], "size": item.get("s", 0)}
                                )
                                callback(data_point)
                    
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break
                    except Exception as e:
                        logger.error(f"Error in streaming loop: {e}")
                        await asyncio.sleep(1)
            
            asyncio.create_task(listen_loop())
            
        except Exception as e:
            logger.error(f"Error starting WebSocket streaming: {e}")
    
    async def stop_streaming(self):
        """Stop WebSocket streaming"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
        logger.info("Stopped WebSocket streaming from Polygon.io")


class RealTimeDataManager:
    """Manager for multiple real-time data sources with failover and aggregation"""
    
    def __init__(self):
        self.data_sources: Dict[str, BaseRealTimeDataSource] = {}
        self.primary_source: Optional[str] = None
        self.fallback_sources: List[str] = []
        self.data_cache: Dict[str, MarketDataPoint] = {}
        self.streaming_active = False
        self.callbacks: List[StreamingDataCallback] = []
        
    def add_data_source(self, name: str, source: BaseRealTimeDataSource, is_primary: bool = False):
        """Add a data source"""
        self.data_sources[name] = source
        
        if is_primary:
            self.primary_source = name
        else:
            self.fallback_sources.append(name)
        
        logger.info(f"Added data source: {name} (primary: {is_primary})")
    
    async def connect_all_sources(self) -> Dict[str, bool]:
        """Connect to all data sources"""
        connection_results = {}
        
        for name, source in self.data_sources.items():
            try:
                result = await source.connect()
                connection_results[name] = result
                logger.info(f"Connection to {name}: {'Success' if result else 'Failed'}")
            except Exception as e:
                logger.error(f"Error connecting to {name}: {e}")
                connection_results[name] = False
        
        return connection_results
    
    async def disconnect_all_sources(self):
        """Disconnect from all data sources"""
        for name, source in self.data_sources.items():
            try:
                await source.disconnect()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
    
    async def fetch_latest_data_with_failover(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch latest data with automatic failover"""
        
        # Try primary source first
        if self.primary_source and self.primary_source in self.data_sources:
            try:
                source = self.data_sources[self.primary_source]
                if source.is_connected:
                    data = await source.fetch_latest_data(symbols)
                    if data:
                        # Update cache
                        for point in data:
                            self.data_cache[point.symbol] = point
                        return data
            except Exception as e:
                logger.warning(f"Primary source {self.primary_source} failed: {e}")
        
        # Try fallback sources
        for source_name in self.fallback_sources:
            if source_name in self.data_sources:
                try:
                    source = self.data_sources[source_name]
                    if source.is_connected:
                        data = await source.fetch_latest_data(symbols)
                        if data:
                            logger.info(f"Using fallback source: {source_name}")
                            # Update cache
                            for point in data:
                                self.data_cache[point.symbol] = point
                            return data
                except Exception as e:
                    logger.warning(f"Fallback source {source_name} failed: {e}")
                    continue
        
        # Return cached data if available
        cached_data = []
        for symbol in symbols:
            if symbol in self.data_cache:
                cached_data.append(self.data_cache[symbol])
        
        if cached_data:
            logger.warning("Returning cached data due to source failures")
        
        return cached_data
    
    async def fetch_historical_data_aggregated(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Fetch historical data from best available source"""
        
        # Try sources in order of preference
        source_order = []
        if self.primary_source:
            source_order.append(self.primary_source)
        source_order.extend(self.fallback_sources)
        
        for source_name in source_order:
            if source_name in self.data_sources:
                try:
                    source = self.data_sources[source_name]
                    if source.is_connected:
                        data = await source.fetch_historical_data(symbols, start_date, end_date, frequency)
                        if not data.empty:
                            logger.info(f"Historical data fetched from: {source_name}")
                            return data
                except Exception as e:
                    logger.warning(f"Error fetching historical data from {source_name}: {e}")
                    continue
        
        logger.error("Failed to fetch historical data from any source")
        return pd.DataFrame()
    
    def add_streaming_callback(self, callback: StreamingDataCallback):
        """Add streaming callback"""
        self.callbacks.append(callback)
        logger.info(f"Added streaming callback for {len(callback.symbols)} symbols")
    
    async def start_streaming_all_sources(self):
        """Start streaming from all capable sources"""
        if self.streaming_active:
            logger.warning("Streaming already active")
            return
        
        self.streaming_active = True
        
        def consolidated_callback(data_point: MarketDataPoint):
            """Consolidated callback that routes to registered callbacks"""
            self.data_cache[data_point.symbol] = data_point
            
            for callback in self.callbacks:
                if data_point.symbol in callback.symbols:
                    try:
                        callback.callback_function(data_point)
                    except Exception as e:
                        logger.error(f"Error in callback function: {e}")
        
        # Start streaming from all sources that support it
        for name, source in self.data_sources.items():
            if source.config.enable_streaming and source.is_connected:
                try:
                    # Get all symbols from callbacks
                    all_symbols = set()
                    for callback in self.callbacks:
                        all_symbols.update(callback.symbols)
                    
                    if all_symbols:
                        await source.start_streaming(list(all_symbols), consolidated_callback)
                        logger.info(f"Started streaming from {name}")
                
                except Exception as e:
                    logger.error(f"Error starting streaming from {name}: {e}")
    
    async def stop_streaming_all_sources(self):
        """Stop streaming from all sources"""
        self.streaming_active = False
        
        for name, source in self.data_sources.items():
            if source.config.enable_streaming:
                try:
                    await source.stop_streaming()
                    logger.info(f"Stopped streaming from {name}")
                except Exception as e:
                    logger.error(f"Error stopping streaming from {name}: {e}")
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""
        metrics = {
            'total_sources': len(self.data_sources),
            'connected_sources': sum(1 for source in self.data_sources.values() if source.is_connected),
            'cached_symbols': len(self.data_cache),
            'streaming_active': self.streaming_active,
            'last_update_times': {
                symbol: point.timestamp.isoformat() 
                for symbol, point in self.data_cache.items()
            }
        }
        
        return metrics
    
    def export_data_cache(self) -> pd.DataFrame:
        """Export cached data as DataFrame"""
        if not self.data_cache:
            return pd.DataFrame()
        
        data_list = []
        for symbol, point in self.data_cache.items():
            data_list.append({
                'symbol': point.symbol,
                'timestamp': point.timestamp,
                'open': point.open,
                'high': point.high,
                'low': point.low,
                'close': point.close,
                'volume': point.volume,
                'source': point.source
            })
        
        return pd.DataFrame(data_list)


# Factory function to create configured data manager
def create_realtime_data_manager(
    yahoo_finance: bool = True,
    alpha_vantage_key: Optional[str] = None,
    polygon_key: Optional[str] = None
) -> RealTimeDataManager:
    """Factory function to create a configured real-time data manager"""
    
    manager = RealTimeDataManager()
    
    # Add Yahoo Finance (always available, set as primary by default)
    if yahoo_finance:
        yahoo_source = YahooFinanceRealTimeSource()
        manager.add_data_source("yahoo_finance", yahoo_source, is_primary=True)
    
    # Add Alpha Vantage if API key provided
    if alpha_vantage_key:
        alpha_source = AlphaVantageRealTimeSource(alpha_vantage_key)
        manager.add_data_source("alpha_vantage", alpha_source, is_primary=not yahoo_finance)
    
    # Add Polygon.io if API key provided
    if polygon_key:
        polygon_source = PolygonIOSource(polygon_key)
        manager.add_data_source("polygon_io", polygon_source, is_primary=not (yahoo_finance or alpha_vantage_key))
    
    logger.info(f"Created real-time data manager with {len(manager.data_sources)} sources")
    return manager