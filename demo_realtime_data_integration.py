#!/usr/bin/env python3
"""
Real-Time Data Integration System Demonstration

This script demonstrates the comprehensive real-time data integration capabilities
of the Enhanced Stochastic Portfolio Engine, including:

1. Multi-source data aggregation with failover
2. Real-time streaming and WebSocket services  
3. Data quality monitoring and assessment
4. Live regime detection and risk monitoring
5. Cross-source validation and caching

Usage:
    python demo_realtime_data_integration.py [--alpha-vantage-key KEY] [--polygon-key KEY]
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
import json
import time
import threading
import signal
import sys

# Enhanced data integration imports
from src.data.ingestion.enhanced_data_pipeline import (
    EnhancedDataPipeline, 
    DataPipelineConfig,
    create_enhanced_data_pipeline
)
from src.data.ingestion.realtime_data_sources import (
    DataFrequency,
    MarketDataPoint,
    create_realtime_data_manager
)
from src.data.streaming.realtime_streaming_service import (
    create_streaming_service,
    StreamingEventType
)

from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_file="realtime_demo.log")
logger = get_logger(__name__)


class RealTimeDataDemo:
    """Demonstration of real-time data integration capabilities"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, polygon_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.polygon_key = polygon_key
        self.pipeline = None
        self.streaming_service = None
        self.demo_running = False
        
        # Demo portfolio
        self.demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ']
        self.demo_weights = {
            'AAPL': 0.15, 'GOOGL': 0.15, 'MSFT': 0.15, 'AMZN': 0.10,
            'TSLA': 0.10, 'NVDA': 0.10, 'SPY': 0.15, 'QQQ': 0.10
        }
        
        # Metrics tracking
        self.data_points_received = 0
        self.regime_changes_detected = 0
        self.quality_assessments = 0
        self.start_time = None
    
    async def run_comprehensive_demo(self):
        """Run comprehensive real-time data integration demonstration"""
        
        logger.info("🚀 Starting Real-Time Data Integration System Demo")
        logger.info("=" * 60)
        
        self.demo_running = True
        self.start_time = datetime.now()
        
        try:
            # Phase 1: Initialize Enhanced Data Pipeline
            await self._demo_pipeline_initialization()
            
            # Phase 2: Multi-Source Data Fetching
            await self._demo_multi_source_data_fetching()
            
            # Phase 3: Data Quality Assessment
            await self._demo_data_quality_assessment()
            
            # Phase 4: Real-Time Streaming Setup
            await self._demo_streaming_setup()
            
            # Phase 5: Live Monitoring (run for demonstration period)
            await self._demo_live_monitoring()
            
            # Phase 6: Performance Analytics
            await self._demo_performance_analytics()
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            await self._cleanup_demo()
    
    async def _demo_pipeline_initialization(self):
        """Demo Phase 1: Pipeline Initialization"""
        
        logger.info("\n📊 PHASE 1: Enhanced Data Pipeline Initialization")
        logger.info("-" * 50)
        
        # Create enhanced pipeline configuration
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
        
        logger.info("🔧 Creating enhanced data pipeline with configuration:")
        logger.info(f"   - Real-time enabled: {config.enable_realtime}")
        logger.info(f"   - Caching enabled: {config.enable_caching}")
        logger.info(f"   - Quality monitoring: {config.enable_quality_monitoring}")
        logger.info(f"   - Cross-validation: {config.enable_cross_validation}")
        
        # Initialize pipeline
        self.pipeline = create_enhanced_data_pipeline(
            config=config,
            alpha_vantage_key=self.alpha_vantage_key,
            polygon_key=self.polygon_key
        )
        
        # Initialize real-time sources
        await self.pipeline.initialize_realtime_sources(
            alpha_vantage_key=self.alpha_vantage_key,
            polygon_key=self.polygon_key,
            enable_yahoo=True
        )
        
        # Get pipeline status
        status = self.pipeline.get_pipeline_status()
        logger.info("✅ Pipeline initialized successfully")
        logger.info(f"   - Real-time sources: {status.get('realtime', {}).get('total_sources', 0)}")
        logger.info(f"   - Connected sources: {status.get('realtime', {}).get('connected_sources', 0)}")
        
        await asyncio.sleep(2)
    
    async def _demo_multi_source_data_fetching(self):
        """Demo Phase 2: Multi-Source Data Fetching"""
        
        logger.info("\n📈 PHASE 2: Multi-Source Data Fetching with Failover")
        logger.info("-" * 50)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        logger.info(f"🔍 Fetching data for {len(self.demo_symbols)} symbols")
        logger.info(f"   - Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   - Symbols: {', '.join(self.demo_symbols)}")
        
        # Fetch enhanced asset data with quality metrics
        data, quality_metrics = await self.pipeline.fetch_enhanced_asset_data(
            symbols=self.demo_symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAILY,
            include_quality_metrics=True
        )
        
        if not data.empty:
            logger.info("✅ Data fetching successful")
            logger.info(f"   - Records retrieved: {len(data)}")
            logger.info(f"   - Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"   - Symbols with data: {len(data.columns.get_level_values(1).unique())}")
        else:
            logger.warning("⚠️  No data retrieved")
        
        # Display sample data
        if not data.empty:
            logger.info("\n📋 Sample Data (Last 3 Days):")
            sample_data = data.tail(3)
            for date in sample_data.index:
                logger.info(f"   {date.strftime('%Y-%m-%d')}: {len(sample_data.loc[date].dropna())} price points")
        
        await asyncio.sleep(2)
    
    async def _demo_data_quality_assessment(self):
        """Demo Phase 3: Data Quality Assessment"""
        
        logger.info("\n🔍 PHASE 3: Data Quality Assessment")
        logger.info("-" * 50)
        
        # Fetch data with quality metrics
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Shorter period for quality demo
        
        data, quality_metrics = await self.pipeline.fetch_enhanced_asset_data(
            symbols=self.demo_symbols[:4],  # Use fewer symbols for detailed analysis
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAILY,
            include_quality_metrics=True
        )
        
        if quality_metrics:
            logger.info("📊 Data Quality Assessment Results:")
            
            status_summary = {}
            for symbol, metrics in quality_metrics.items():
                status = metrics.overall_status.value
                status_summary[status] = status_summary.get(status, 0) + 1
                
                logger.info(f"\n   {symbol}:")
                logger.info(f"     - Status: {status.upper()}")
                logger.info(f"     - Completeness: {metrics.completeness:.1f}%")
                logger.info(f"     - Timeliness: {metrics.timeliness:.0f} minutes old")
                logger.info(f"     - Accuracy: {metrics.accuracy:.1f}%")
                logger.info(f"     - Consistency: {metrics.consistency:.1f}%")
                
                if metrics.issues:
                    logger.info(f"     - Issues: {', '.join(metrics.issues[:2])}")
                if metrics.recommendations:
                    logger.info(f"     - Recommendations: {', '.join(metrics.recommendations[:2])}")
            
            logger.info(f"\n📈 Quality Summary:")
            for status, count in status_summary.items():
                logger.info(f"   - {status.upper()}: {count} symbols")
            
            # Generate quality report
            quality_report = self.pipeline.generate_quality_report(quality_metrics)
            
            # Save quality report
            with open('data_quality_report.txt', 'w') as f:
                f.write(quality_report)
            logger.info("📄 Detailed quality report saved to 'data_quality_report.txt'")
            
            self.quality_assessments += 1
        
        await asyncio.sleep(2)
    
    async def _demo_streaming_setup(self):
        """Demo Phase 4: Real-Time Streaming Setup"""
        
        logger.info("\n🌊 PHASE 4: Real-Time Streaming Service Setup")
        logger.info("-" * 50)
        
        # Create streaming service
        self.streaming_service = create_streaming_service(self.pipeline)
        
        # Initialize streaming service
        await self.streaming_service.initialize(hmm_components=3)
        
        logger.info("🔧 Streaming service initialized with:")
        logger.info(f"   - HMM regime detection (3 components)")
        logger.info(f"   - Real-time risk monitoring")
        logger.info(f"   - WebSocket server capability")
        
        # Setup custom data callback
        def market_data_callback(data_point: MarketDataPoint):
            """Custom callback for market data updates"""
            self.data_points_received += 1
            if self.data_points_received % 10 == 0:  # Log every 10th update
                logger.info(f"📊 Received data: {data_point.symbol} @ {data_point.close:.2f} "
                          f"({data_point.timestamp.strftime('%H:%M:%S')})")
        
        # Add streaming callback
        self.pipeline.add_streaming_callback(
            symbols=self.demo_symbols,
            callback_function=market_data_callback
        )
        
        logger.info("✅ Streaming callbacks configured")
        logger.info(f"   - Monitoring {len(self.demo_symbols)} symbols")
        
        await asyncio.sleep(2)
    
    async def _demo_live_monitoring(self):
        """Demo Phase 5: Live Monitoring Simulation"""
        
        logger.info("\n📡 PHASE 5: Live Monitoring Demonstration")
        logger.info("-" * 50)
        logger.info("🚀 Starting live data monitoring (60 seconds)...")
        
        # Start data streaming
        await self.pipeline.start_streaming()
        
        # Monitor for demonstration period
        monitoring_duration = 60  # seconds
        start_time = time.time()
        last_status_update = 0
        
        while (time.time() - start_time) < monitoring_duration and self.demo_running:
            current_time = time.time() - start_time
            
            # Status update every 15 seconds
            if current_time - last_status_update >= 15:
                logger.info(f"⏱️  Monitoring Status ({current_time:.0f}s):")
                logger.info(f"   - Data points received: {self.data_points_received}")
                logger.info(f"   - Quality assessments: {self.quality_assessments}")
                
                # Get pipeline status
                status = self.pipeline.get_pipeline_status()
                logger.info(f"   - Cache entries: {status.get('cache', {}).get('entries', 0)}")
                
                last_status_update = current_time
            
            await asyncio.sleep(5)
        
        # Stop streaming
        await self.pipeline.stop_streaming()
        
        logger.info("✅ Live monitoring demonstration completed")
        logger.info(f"   - Total data points processed: {self.data_points_received}")
        
        await asyncio.sleep(2)
    
    async def _demo_performance_analytics(self):
        """Demo Phase 6: Performance Analytics"""
        
        logger.info("\n📊 PHASE 6: Performance Analytics and Reporting")
        logger.info("-" * 50)
        
        # Calculate demonstration metrics
        demo_duration = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("📈 Demonstration Performance Metrics:")
        logger.info(f"   - Total runtime: {demo_duration:.1f} seconds")
        logger.info(f"   - Data points processed: {self.data_points_received}")
        logger.info(f"   - Quality assessments: {self.quality_assessments}")
        logger.info(f"   - Average data rate: {self.data_points_received / demo_duration:.2f} points/sec")
        
        # Get final pipeline status
        final_status = self.pipeline.get_pipeline_status()
        logger.info("\n🔧 Final Pipeline Status:")
        logger.info(f"   - Real-time enabled: {final_status['config']['realtime_enabled']}")
        logger.info(f"   - Caching enabled: {final_status['config']['caching_enabled']}")
        logger.info(f"   - Cache entries: {final_status['cache']['entries']}")
        logger.info(f"   - Quality entries: {final_status['cache']['quality_entries']}")
        
        if 'realtime' in final_status:
            rt_status = final_status['realtime']
            logger.info(f"   - Connected sources: {rt_status.get('connected_sources', 0)}")
            logger.info(f"   - Cached symbols: {rt_status.get('cached_symbols', 0)}")
        
        # Export cached data
        try:
            if hasattr(self.pipeline, 'realtime_manager') and self.pipeline.realtime_manager:
                cached_data = self.pipeline.realtime_manager.export_data_cache()
                if not cached_data.empty:
                    cached_data.to_csv('realtime_data_cache.csv', index=False)
                    logger.info("💾 Exported cached data to 'realtime_data_cache.csv'")
        except Exception as e:
            logger.warning(f"Could not export cached data: {e}")
        
        # Performance summary
        logger.info("\n🎯 Demonstration Summary:")
        logger.info("   ✅ Multi-source data integration with failover")
        logger.info("   ✅ Real-time streaming capabilities")
        logger.info("   ✅ Data quality monitoring and assessment")
        logger.info("   ✅ Caching and performance optimization")
        logger.info("   ✅ Cross-source validation")
        
        await asyncio.sleep(2)
    
    async def _cleanup_demo(self):
        """Cleanup demo resources"""
        
        logger.info("\n🧹 Cleaning up demo resources...")
        
        try:
            if self.pipeline:
                await self.pipeline.cleanup()
            
            if self.streaming_service:
                if hasattr(self.streaming_service, 'stop_server'):
                    await self.streaming_service.stop_server()
            
            logger.info("✅ Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def run_websocket_server_demo(pipeline: EnhancedDataPipeline, port: int = 8765):
    """Run WebSocket server demonstration"""
    
    logger.info(f"\n🌐 WebSocket Server Demo (Port {port})")
    logger.info("-" * 40)
    
    try:
        # Create and initialize streaming service
        streaming_service = create_streaming_service(pipeline)
        await streaming_service.initialize()
        
        logger.info(f"🚀 Starting WebSocket server on port {port}")
        logger.info("📱 Connect with WebSocket client to:")
        logger.info(f"   ws://localhost:{port}")
        logger.info("\n📋 Available message types:")
        logger.info("   - {'type': 'subscribe', 'symbols': ['AAPL', 'GOOGL'], 'events': ['price_update']}")
        logger.info("   - {'type': 'get_status'}")
        logger.info("   - {'type': 'heartbeat'}")
        
        # Start server
        await streaming_service.start_server(host="localhost", port=port)
        
    except Exception as e:
        logger.error(f"WebSocket server error: {e}")


def create_sample_websocket_client():
    """Create a sample WebSocket client for testing"""
    
    client_code = '''
import asyncio
import websockets
import json

async def sample_client():
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to streaming server")
            
            # Subscribe to updates
            subscribe_msg = {
                "type": "subscribe",
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "events": ["price_update", "regime_change", "system_status"]
            }
            await websocket.send(json.dumps(subscribe_msg))
            
            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                print(f"Received: {data['type']} - {data.get('data', '')}")
                
    except Exception as e:
        print(f"Client error: {e}")

if __name__ == "__main__":
    asyncio.run(sample_client())
'''
    
    with open('sample_websocket_client.py', 'w') as f:
        f.write(client_code)
    
    logger.info("📝 Created sample WebSocket client: 'sample_websocket_client.py'")


def setup_signal_handlers(demo: RealTimeDataDemo):
    """Setup signal handlers for graceful shutdown"""
    
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, shutting down gracefully...")
        demo.demo_running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main demonstration function"""
    
    parser = argparse.ArgumentParser(description="Real-Time Data Integration System Demo")
    parser.add_argument('--alpha-vantage-key', type=str, help='Alpha Vantage API key')
    parser.add_argument('--polygon-key', type=str, help='Polygon.io API key')
    parser.add_argument('--websocket-server', action='store_true', 
                       help='Run WebSocket server demo instead')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port')
    parser.add_argument('--create-client', action='store_true',
                       help='Create sample WebSocket client script')
    
    args = parser.parse_args()
    
    logger.info("🚀 Real-Time Data Integration System Demonstration")
    logger.info("=" * 60)
    
    if args.create_client:
        create_sample_websocket_client()
        return
    
    # Create demo instance
    demo = RealTimeDataDemo(
        alpha_vantage_key=args.alpha_vantage_key,
        polygon_key=args.polygon_key
    )
    
    # Setup signal handlers
    setup_signal_handlers(demo)
    
    if args.websocket_server:
        # Run WebSocket server demo
        config = DataPipelineConfig(enable_realtime=True, enable_caching=True)
        pipeline = create_enhanced_data_pipeline(
            config=config,
            alpha_vantage_key=args.alpha_vantage_key,
            polygon_key=args.polygon_key
        )
        
        await run_websocket_server_demo(pipeline, args.port)
    else:
        # Run comprehensive demo
        await demo.run_comprehensive_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)