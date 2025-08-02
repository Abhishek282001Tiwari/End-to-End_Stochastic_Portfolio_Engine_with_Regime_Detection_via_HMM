import yfinance as yf
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataSource(ABC):
    @abstractmethod
    async def fetch_price_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def fetch_market_data(
        self, 
        indicators: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        pass


class YahooFinanceSource(DataSource):
    def __init__(self):
        self.name = "yahoo_finance"
        
    async def fetch_price_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        logger.info(f"Fetching price data for {len(symbols)} symbols from Yahoo Finance")
        
        try:
            data = yf.download(
                symbols, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True,
                threads=True
            )
            
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, symbols])
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching price data from Yahoo Finance: {e}")
            raise
    
    async def fetch_market_data(
        self, 
        indicators: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        logger.info(f"Fetching market indicators: {indicators}")
        
        indicator_mapping = {
            "vix": "^VIX",
            "treasury_10y": "^TNX",
            "treasury_2y": "^IRX",
            "dxy": "DX-Y.NYB",
            "gold": "GC=F",
            "oil": "CL=F"
        }
        
        symbols = [indicator_mapping.get(ind, ind) for ind in indicators]
        return await self.fetch_price_data(symbols, start_date, end_date)


class DataIngestionPipeline:
    def __init__(self, data_sources: Dict[str, DataSource]):
        self.data_sources = data_sources
        self.logger = get_logger(__name__)
        
    async def fetch_asset_universe(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        source: str = "yahoo_finance"
    ) -> pd.DataFrame:
        if source not in self.data_sources:
            raise ValueError(f"Data source {source} not available")
        
        data_source = self.data_sources[source]
        return await data_source.fetch_price_data(symbols, start_date, end_date)
    
    async def fetch_market_indicators(
        self, 
        indicators: List[str], 
        start_date: datetime, 
        end_date: datetime,
        source: str = "yahoo_finance"
    ) -> pd.DataFrame:
        if source not in self.data_sources:
            raise ValueError(f"Data source {source} not available")
        
        data_source = self.data_sources[source]
        return await data_source.fetch_market_data(indicators, start_date, end_date)
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().dropna()
    
    def calculate_volatility(
        self, 
        returns: pd.DataFrame, 
        window: int = 20
    ) -> pd.DataFrame:
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        volume_data = data.xs('Volume', level=0, axis=1)
        
        volume_ma = volume_data.rolling(window=20).mean()
        volume_ratio = volume_data / volume_ma
        
        return pd.DataFrame({
            'volume_ma': volume_ma.mean(axis=1),
            'volume_ratio': volume_ratio.mean(axis=1)
        })
    
    async def create_features_dataset(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        price_data = await self.fetch_asset_universe(symbols, start_date, end_date)
        market_data = await self.fetch_market_indicators(
            ["vix", "treasury_10y", "treasury_2y"], 
            start_date, 
            end_date
        )
        
        adj_close = price_data.xs('Adj Close', level=0, axis=1)
        returns = self.calculate_returns(adj_close)
        volatility = self.calculate_volatility(returns)
        volume_indicators = self.calculate_volume_indicators(price_data)
        
        vix = market_data.xs('Adj Close', level=0, axis=1).iloc[:, 0]
        treasury_10y = market_data.xs('Adj Close', level=0, axis=1).iloc[:, 1]
        treasury_2y = market_data.xs('Adj Close', level=0, axis=1).iloc[:, 2]
        
        yield_curve_slope = treasury_10y - treasury_2y
        
        features = pd.DataFrame({
            'market_return': returns.mean(axis=1),
            'market_volatility': volatility.mean(axis=1),
            'vix': vix,
            'yield_curve_slope': yield_curve_slope,
            'volume_ratio': volume_indicators['volume_ratio']
        })
        
        return features.dropna()


def create_data_pipeline() -> DataIngestionPipeline:
    data_sources = {
        "yahoo_finance": YahooFinanceSource()
    }
    
    return DataIngestionPipeline(data_sources)


def create_enhanced_data_pipeline_integration():
    """
    Factory function to create enhanced data pipeline integration
    
    This function provides backwards compatibility while enabling
    access to the new enhanced real-time data integration capabilities.
    """
    try:
        from src.data.ingestion.enhanced_data_pipeline import create_enhanced_data_pipeline
        from src.data.ingestion.enhanced_data_pipeline import DataPipelineConfig
        
        # Create enhanced pipeline with default configuration
        config = DataPipelineConfig(
            enable_realtime=True,
            enable_caching=True,
            cache_duration_minutes=5,
            enable_quality_monitoring=True,
            enable_cross_validation=True,
            fallback_to_cache=True
        )
        
        return create_enhanced_data_pipeline(config)
        
    except ImportError:
        logger.warning("Enhanced data pipeline not available, using legacy pipeline")
        return create_data_pipeline()