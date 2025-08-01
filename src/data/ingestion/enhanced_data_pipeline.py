#!/usr/bin/env python3

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from src.data.ingestion.realtime_data_sources import (
    RealTimeDataManager, 
    MarketDataPoint, 
    DataFrequency,
    StreamingDataCallback,
    create_realtime_data_manager
)
from src.data.ingestion.data_sources import DataIngestionPipeline, create_data_pipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class DataQualityStatus(Enum):
    """Data quality status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness: float  # Percentage of expected data points
    timeliness: float    # How recent the data is (minutes old)
    accuracy: float      # Cross-source validation score
    consistency: float   # Internal consistency score
    availability: float  # Source availability score
    overall_status: DataQualityStatus
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataPipelineConfig:
    """Configuration for enhanced data pipeline"""
    enable_realtime: bool = True
    enable_caching: bool = True
    cache_duration_minutes: int = 5
    enable_quality_monitoring: bool = True
    enable_cross_validation: bool = True
    fallback_to_cache: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    data_validation_rules: Dict[str, Any] = field(default_factory=dict)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class EnhancedDataPipeline:
    """Enhanced data pipeline with real-time capabilities and quality monitoring"""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.legacy_pipeline = create_data_pipeline()
        self.realtime_manager = None
        self.data_cache = {}
        self.quality_cache = {}
        self.cache_timestamps = {}
        self.db_connection = None
        self.streaming_callbacks = []
        
        # Initialize database for caching if enabled
        if config.enable_caching:
            self._init_cache_database()
    
    def _init_cache_database(self):
        """Initialize SQLite database for data caching"""
        try:
            self.db_connection = sqlite3.connect(':memory:', check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create tables for different data types
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    timestamp TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT,
                    created_at TEXT,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_indicators (
                    indicator TEXT,
                    timestamp TEXT,
                    value REAL,
                    source TEXT,
                    created_at TEXT,
                    PRIMARY KEY (indicator, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    symbol TEXT,
                    timestamp TEXT,
                    completeness REAL,
                    timeliness REAL,
                    accuracy REAL,
                    consistency REAL,
                    availability REAL,
                    status TEXT,
                    issues TEXT,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Initialized cache database")
            
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")
    
    async def initialize_realtime_sources(
        self,
        alpha_vantage_key: Optional[str] = None,
        polygon_key: Optional[str] = None,
        enable_yahoo: bool = True
    ):
        """Initialize real-time data sources"""
        if not self.config.enable_realtime:
            logger.info("Real-time data disabled in configuration")
            return
        
        try:
            self.realtime_manager = create_realtime_data_manager(
                yahoo_finance=enable_yahoo,
                alpha_vantage_key=alpha_vantage_key,
                polygon_key=polygon_key
            )
            
            # Connect to all sources
            connection_results = await self.realtime_manager.connect_all_sources()
            
            connected_sources = [name for name, connected in connection_results.items() if connected]
            logger.info(f"Connected to {len(connected_sources)} real-time data sources: {connected_sources}")
            
        except Exception as e:
            logger.error(f"Error initializing real-time sources: {e}")
    
    async def fetch_enhanced_asset_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
        include_quality_metrics: bool = True
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, DataQualityMetrics]]]:
        """Fetch asset data with enhanced capabilities"""
        
        logger.info(f"Fetching enhanced data for {len(symbols)} symbols")
        
        # Check cache first
        cache_key = f"{'_'.join(symbols)}_{start_date}_{end_date}_{frequency.value}"
        
        if self.config.enable_caching and self._is_cache_valid(cache_key):
            logger.info("Returning cached data")
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                quality_metrics = self._get_cached_quality_metrics(symbols) if include_quality_metrics else None
                return cached_data, quality_metrics
        
        # Fetch from multiple sources
        data_sources = []
        
        # Try real-time sources first for recent data
        if self.realtime_manager and self.config.enable_realtime:
            try:
                realtime_data = await self.realtime_manager.fetch_historical_data_aggregated(
                    symbols, start_date, end_date, frequency
                )
                if not realtime_data.empty:
                    data_sources.append(("realtime", realtime_data))
            except Exception as e:
                logger.warning(f"Error fetching from real-time sources: {e}")
        
        # Fallback to legacy pipeline
        try:
            legacy_data = await self.legacy_pipeline.fetch_asset_universe(
                symbols, start_date, end_date
            )
            if not legacy_data.empty:
                data_sources.append(("legacy", legacy_data))
        except Exception as e:
            logger.warning(f"Error fetching from legacy pipeline: {e}")
        
        # Combine and validate data
        if not data_sources:
            logger.error("No data sources available")
            return pd.DataFrame(), None
        
        # Use the best available data source
        final_data = data_sources[0][1]
        data_source_name = data_sources[0][0]
        
        # Cross-validate if multiple sources available
        quality_metrics = None
        if include_quality_metrics:
            quality_metrics = await self._assess_data_quality(
                symbols, final_data, data_sources
            )
        
        # Cache the results
        if self.config.enable_caching:
            self._cache_data(cache_key, final_data, quality_metrics)
        
        logger.info(f"Fetched data from {data_source_name} source with {len(final_data)} records")
        return final_data, quality_metrics
    
    async def fetch_enhanced_features_dataset(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        custom_features: Optional[Dict[str, Callable]] = None
    ) -> pd.DataFrame:
        """Create enhanced features dataset with additional indicators"""
        
        logger.info("Creating enhanced features dataset")
        
        # Get base features from legacy pipeline
        base_features = await self.legacy_pipeline.create_features_dataset(
            symbols, start_date, end_date
        )
        
        # Get additional market data from real-time sources
        additional_indicators = ["vix", "treasury_10y", "treasury_2y", "dxy", "gold", "oil"]
        
        try:
            if self.realtime_manager:
                market_data = await self.realtime_manager.fetch_historical_data_aggregated(
                    additional_indicators, start_date, end_date, DataFrequency.DAILY
                )
                
                if not market_data.empty:
                    # Add market regime indicators
                    base_features = self._add_market_regime_features(base_features, market_data)
        
        except Exception as e:
            logger.warning(f"Error fetching additional market indicators: {e}")
        
        # Add custom features if provided
        if custom_features:
            base_features = self._add_custom_features(base_features, custom_features, symbols)
        
        # Add technical indicators
        base_features = self._add_technical_indicators(base_features)
        
        return base_features.dropna()
    
    def _add_market_regime_features(self, features: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators to features"""
        
        try:
            # Extract market indicators
            if 'dxy' in market_data.columns.get_level_values(1):
                dxy = market_data.xs('Close', level=0, axis=1)['dxy']
                features['dollar_strength'] = dxy.pct_change(20)  # 20-day change
            
            if 'gold' in market_data.columns.get_level_values(1):
                gold = market_data.xs('Close', level=0, axis=1)['gold']
                features['gold_returns'] = gold.pct_change()
                features['gold_volatility'] = gold.pct_change().rolling(20).std()
            
            if 'oil' in market_data.columns.get_level_values(1):
                oil = market_data.xs('Close', level=0, axis=1)['oil']
                features['oil_returns'] = oil.pct_change()
                features['oil_volatility'] = oil.pct_change().rolling(20).std()
            
            # Risk-on/Risk-off indicator
            if 'vix' in features.columns:
                features['risk_sentiment'] = np.where(
                    features['vix'] > features['vix'].rolling(60).mean(),
                    -1,  # Risk-off
                    1    # Risk-on
                )
            
            # Term structure indicators
            if 'yield_curve_slope' in features.columns:
                features['yield_curve_regime'] = np.where(
                    features['yield_curve_slope'] > 0,
                    1,   # Normal curve
                    -1   # Inverted curve
                )
        
        except Exception as e:
            logger.warning(f"Error adding market regime features: {e}")
        
        return features
    
    def _add_custom_features(
        self, 
        features: pd.DataFrame, 
        custom_features: Dict[str, Callable],
        symbols: List[str]
    ) -> pd.DataFrame:
        """Add custom user-defined features"""
        
        for feature_name, feature_func in custom_features.items():
            try:
                features[feature_name] = feature_func(features, symbols)
                logger.info(f"Added custom feature: {feature_name}")
            except Exception as e:
                logger.warning(f"Error adding custom feature {feature_name}: {e}")
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        
        try:
            if 'market_return' in features.columns:
                returns = features['market_return']
                
                # Moving averages
                features['sma_10'] = returns.rolling(10).mean()
                features['sma_20'] = returns.rolling(20).mean()
                features['sma_50'] = returns.rolling(50).mean()
                
                # Momentum indicators
                features['rsi'] = self._calculate_rsi(returns, 14)
                features['momentum_10'] = returns.rolling(10).sum()
                features['momentum_20'] = returns.rolling(20).sum()
                
                # Volatility indicators
                if 'market_volatility' in features.columns:
                    vol = features['market_volatility']
                    features['vol_sma_10'] = vol.rolling(10).mean()
                    features['vol_regime'] = np.where(
                        vol > vol.rolling(60).mean(),
                        1,   # High vol regime
                        -1   # Low vol regime
                    )
            
            # Cross-correlations
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                features['correlation_regime'] = self._calculate_rolling_correlation(
                    features[numeric_columns[:2]]
                )
        
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {e}")
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rolling_correlation(self, data: pd.DataFrame, window: int = 30) -> pd.Series:
        """Calculate rolling correlation between first two columns"""
        if data.shape[1] < 2:
            return pd.Series(index=data.index, dtype=float)
        
        return data.iloc[:, 0].rolling(window).corr(data.iloc[:, 1])
    
    async def _assess_data_quality(
        self,
        symbols: List[str],
        data: pd.DataFrame,
        data_sources: List[Tuple[str, pd.DataFrame]]
    ) -> Dict[str, DataQualityMetrics]:
        """Assess data quality across multiple dimensions"""
        
        quality_metrics = {}
        
        for symbol in symbols:
            try:
                # Completeness check
                expected_records = len(pd.date_range(
                    start=data.index.min(), 
                    end=data.index.max(), 
                    freq='D'
                ))
                actual_records = len(data.dropna())
                completeness = (actual_records / expected_records) * 100
                
                # Timeliness check
                if not data.empty:
                    latest_timestamp = data.index.max()
                    timeliness = (datetime.now() - latest_timestamp).total_seconds() / 60  # minutes old
                else:
                    timeliness = float('inf')
                
                # Accuracy check (cross-source validation)
                accuracy = 100.0  # Default if only one source
                if len(data_sources) > 1:
                    accuracy = self._cross_validate_sources(symbol, data_sources)
                
                # Consistency check
                consistency = self._check_internal_consistency(data, symbol)
                
                # Availability check
                availability = 100.0 if not data.empty else 0.0
                
                # Determine overall status
                overall_scores = [completeness, min(100, max(0, 100 - timeliness)), accuracy, consistency, availability]
                avg_score = np.mean(overall_scores)
                
                if avg_score >= 90:
                    status = DataQualityStatus.EXCELLENT
                elif avg_score >= 75:
                    status = DataQualityStatus.GOOD
                elif avg_score >= 60:
                    status = DataQualityStatus.FAIR
                elif avg_score >= 30:
                    status = DataQualityStatus.POOR
                else:
                    status = DataQualityStatus.UNAVAILABLE
                
                # Generate issues and recommendations
                issues = []
                recommendations = []
                
                if completeness < 95:
                    issues.append(f"Data completeness only {completeness:.1f}%")
                    recommendations.append("Consider using additional data sources")
                
                if timeliness > 60:
                    issues.append(f"Data is {timeliness:.0f} minutes old")
                    recommendations.append("Enable real-time data sources")
                
                if accuracy < 90:
                    issues.append(f"Cross-source accuracy only {accuracy:.1f}%")
                    recommendations.append("Investigate data source discrepancies")
                
                quality_metrics[symbol] = DataQualityMetrics(
                    completeness=completeness,
                    timeliness=timeliness,
                    accuracy=accuracy,
                    consistency=consistency,
                    availability=availability,
                    overall_status=status,
                    issues=issues,
                    recommendations=recommendations
                )
                
            except Exception as e:
                logger.warning(f"Error assessing quality for {symbol}: {e}")
                
                quality_metrics[symbol] = DataQualityMetrics(
                    completeness=0,
                    timeliness=float('inf'),
                    accuracy=0,
                    consistency=0,
                    availability=0,
                    overall_status=DataQualityStatus.UNAVAILABLE,
                    issues=[f"Quality assessment failed: {str(e)}"],
                    recommendations=["Check data source connectivity"]
                )
        
        return quality_metrics
    
    def _cross_validate_sources(self, symbol: str, data_sources: List[Tuple[str, pd.DataFrame]]) -> float:
        """Cross-validate data between sources"""
        
        if len(data_sources) < 2:
            return 100.0
        
        try:
            # Compare close prices between sources
            source1_data = data_sources[0][1]
            source2_data = data_sources[1][1] 
            
            # Extract close prices for the symbol
            if symbol in source1_data.columns.get_level_values(1) and symbol in source2_data.columns.get_level_values(1):
                close1 = source1_data.xs('Close', level=0, axis=1)[symbol]
                close2 = source2_data.xs('Close', level=0, axis=1)[symbol]
                
                # Align data by index
                common_dates = close1.index.intersection(close2.index)
                
                if len(common_dates) > 10:
                    aligned_close1 = close1.loc[common_dates]
                    aligned_close2 = close2.loc[common_dates]
                    
                    # Calculate percentage differences
                    pct_diff = abs((aligned_close1 - aligned_close2) / aligned_close1) * 100
                    avg_diff = pct_diff.mean()
                    
                    # Convert to accuracy score (100% - average difference)
                    accuracy = max(0, 100 - avg_diff)
                    return accuracy
            
            return 50.0  # Default if can't compare
            
        except Exception as e:
            logger.warning(f"Error in cross-validation for {symbol}: {e}")
            return 50.0
    
    def _check_internal_consistency(self, data: pd.DataFrame, symbol: str) -> float:
        """Check internal data consistency"""
        
        try:
            consistency_score = 100.0
            
            if symbol in data.columns.get_level_values(1):
                # Extract OHLC data
                open_prices = data.xs('Open', level=0, axis=1).get(symbol)
                high_prices = data.xs('High', level=0, axis=1).get(symbol)
                low_prices = data.xs('Low', level=0, axis=1).get(symbol)
                close_prices = data.xs('Close', level=0, axis=1).get(symbol)
                
                if all(x is not None for x in [open_prices, high_prices, low_prices, close_prices]):
                    # Check OHLC relationships
                    violations = 0
                    total_checks = len(close_prices)
                    
                    for i in range(len(close_prices)):
                        if pd.notna(open_prices.iloc[i]) and pd.notna(high_prices.iloc[i]) and pd.notna(low_prices.iloc[i]) and pd.notna(close_prices.iloc[i]):
                            o, h, l, c = open_prices.iloc[i], high_prices.iloc[i], low_prices.iloc[i], close_prices.iloc[i]
                            
                            # High should be >= max(open, close)
                            # Low should be <= min(open, close)
                            if h < max(o, c) or l > min(o, c):
                                violations += 1
                    
                    if total_checks > 0:
                        consistency_score = ((total_checks - violations) / total_checks) * 100
            
            return consistency_score
            
        except Exception as e:
            logger.warning(f"Error checking consistency for {symbol}: {e}")
            return 50.0
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        expiry_time = cache_time + timedelta(minutes=self.config.cache_duration_minutes)
        
        return datetime.now() < expiry_time
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache"""
        return self.data_cache.get(cache_key)
    
    def _get_cached_quality_metrics(self, symbols: List[str]) -> Optional[Dict[str, DataQualityMetrics]]:
        """Retrieve quality metrics from cache"""
        cached_metrics = {}
        
        for symbol in symbols:
            if symbol in self.quality_cache:
                cached_metrics[symbol] = self.quality_cache[symbol]
        
        return cached_metrics if cached_metrics else None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame, quality_metrics: Optional[Dict[str, DataQualityMetrics]]):
        """Cache data and quality metrics"""
        self.data_cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
        
        if quality_metrics:
            self.quality_cache.update(quality_metrics)
    
    def add_streaming_callback(self, symbols: List[str], callback_function: Callable[[MarketDataPoint], None]):
        """Add callback for streaming data"""
        if self.realtime_manager:
            streaming_callback = StreamingDataCallback(
                callback_function=callback_function,
                symbols=symbols,
                frequency=DataFrequency.REAL_TIME
            )
            
            self.realtime_manager.add_streaming_callback(streaming_callback)
            self.streaming_callbacks.append(streaming_callback)
            
            logger.info(f"Added streaming callback for {len(symbols)} symbols")
    
    async def start_streaming(self):
        """Start streaming data from all sources"""
        if self.realtime_manager and self.config.enable_realtime:
            await self.realtime_manager.start_streaming_all_sources()
            logger.info("Started streaming from all sources")
    
    async def stop_streaming(self):
        """Stop streaming data"""
        if self.realtime_manager:
            await self.realtime_manager.stop_streaming_all_sources()
            logger.info("Stopped streaming from all sources")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        status = {
            'config': {
                'realtime_enabled': self.config.enable_realtime,
                'caching_enabled': self.config.enable_caching,
                'quality_monitoring_enabled': self.config.enable_quality_monitoring,
                'cache_duration_minutes': self.config.cache_duration_minutes
            },
            'cache': {
                'entries': len(self.data_cache),
                'quality_entries': len(self.quality_cache),
                'oldest_entry': min(self.cache_timestamps.values()) if self.cache_timestamps else None
            },
            'streaming': {
                'active': len(self.streaming_callbacks) > 0,
                'callbacks': len(self.streaming_callbacks)
            }
        }
        
        if self.realtime_manager:
            status['realtime'] = self.realtime_manager.get_data_quality_metrics()
        
        return status
    
    def generate_quality_report(self, quality_metrics: Dict[str, DataQualityMetrics]) -> str:
        """Generate a comprehensive data quality report"""
        
        report = "DATA QUALITY ASSESSMENT REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Assessment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Assets Analyzed: {len(quality_metrics)}\n\n"
        
        # Overall summary
        statuses = [metrics.overall_status for metrics in quality_metrics.values()]
        status_counts = {status: statuses.count(status) for status in DataQualityStatus}
        
        report += "OVERALL QUALITY DISTRIBUTION:\n"
        report += "-" * 30 + "\n"
        for status, count in status_counts.items():
            if count > 0:
                report += f"{status.value.upper()}: {count}\n"
        
        report += "\n"
        
        # Detailed analysis
        report += "DETAILED ANALYSIS:\n"
        report += "-" * 20 + "\n"
        
        for symbol, metrics in quality_metrics.items():
            report += f"\n{symbol}:\n"
            report += f"  Status: {metrics.overall_status.value.upper()}\n"
            report += f"  Completeness: {metrics.completeness:.1f}%\n"
            report += f"  Timeliness: {metrics.timeliness:.0f} minutes old\n"
            report += f"  Accuracy: {metrics.accuracy:.1f}%\n"
            report += f"  Consistency: {metrics.consistency:.1f}%\n"
            report += f"  Availability: {metrics.availability:.1f}%\n"
            
            if metrics.issues:
                report += "  Issues:\n"
                for issue in metrics.issues:
                    report += f"    - {issue}\n"
            
            if metrics.recommendations:
                report += "  Recommendations:\n"
                for rec in metrics.recommendations:
                    report += f"    - {rec}\n"
        
        report += "\n" + "=" * 50 + "\n"
        report += f"Report generated by Enhanced Data Pipeline v1.0\n"
        
        return report
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.realtime_manager:
                await self.realtime_manager.disconnect_all_sources()
            
            if self.db_connection:
                self.db_connection.close()
            
            logger.info("Enhanced data pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory function to create configured enhanced pipeline
def create_enhanced_data_pipeline(
    config: Optional[DataPipelineConfig] = None,
    alpha_vantage_key: Optional[str] = None,
    polygon_key: Optional[str] = None
) -> EnhancedDataPipeline:
    """Factory function to create configured enhanced data pipeline"""
    
    if config is None:
        config = DataPipelineConfig(
            enable_realtime=True,
            enable_caching=True,
            cache_duration_minutes=5,
            enable_quality_monitoring=True,
            enable_cross_validation=True,
            fallback_to_cache=True,
            parallel_processing=True,
            max_workers=4
        )
    
    pipeline = EnhancedDataPipeline(config)
    
    # Initialize real-time sources asynchronously
    async def init_sources():
        await pipeline.initialize_realtime_sources(
            alpha_vantage_key=alpha_vantage_key,
            polygon_key=polygon_key,
            enable_yahoo=True
        )
    
    # Run initialization
    asyncio.create_task(init_sources())
    
    logger.info("Created enhanced data pipeline with real-time capabilities")
    return pipeline