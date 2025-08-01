#!/usr/bin/env python3
"""
Startup script for the End-to-End Stochastic Portfolio Engine
with Regime Detection via HMM
"""

import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import get_config
from src.utils.logging_config import get_logger
from src.data.ingestion.data_sources import YahooFinanceDataSource
from src.data.processing.data_processor import MarketDataProcessor
from src.models.hmm.hmm_engine import RegimeDetectionHMM
from src.models.ml_enhancements.ensemble_methods import create_advanced_ensemble
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.backtesting.engine.backtesting_engine import BacktestingEngine, BacktestConfig
from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor
from src.dashboard.web_dashboard import PortfolioDashboard
from src.reporting.automated_reports import AutomatedReportingSystem
from src.monitoring.metrics import get_metrics_instance, MetricsServer
from src.validation.model_validation import create_validation_framework

logger = get_logger(__name__)


class PortfolioEngineOrchestrator:
    """Main orchestrator for the portfolio engine"""
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.components = {}
        self.data_cache = {}
        self.metrics = get_metrics_instance()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Starting Portfolio Engine Initialization...")
        
        try:
            # 1. Initialize data sources
            logger.info("📊 Initializing data sources...")
            self.components['data_source'] = YahooFinanceDataSource()
            self.components['data_processor'] = MarketDataProcessor()
            
            # 2. Load initial data
            logger.info("📈 Loading market data...")
            await self._load_initial_data()
            
            # 3. Initialize regime detection models
            logger.info("🔄 Setting up regime detection...")
            self.components['regime_detector'] = create_advanced_ensemble(
                self.data_cache['processed_data'],
                n_components=3
            )
            
            # 4. Initialize portfolio optimization
            logger.info("🎯 Setting up portfolio optimization...")
            self.components['optimizer'] = PortfolioOptimizationEngine()
            
            # 5. Initialize risk monitoring
            logger.info("⚠️ Setting up risk monitoring...")
            self.components['risk_monitor'] = RealTimeRiskMonitor()
            
            # 6. Initialize validation framework
            logger.info("✅ Setting up model validation...")
            self.components['validator'] = create_validation_framework()
            
            # 7. Initialize reporting system
            logger.info("📋 Setting up automated reporting...")
            self.components['reporting'] = AutomatedReportingSystem()
            
            # 8. Initialize web dashboard
            logger.info("🌐 Setting up web dashboard...")
            self.components['dashboard'] = PortfolioDashboard()
            
            # 9. Start monitoring
            logger.info("📊 Starting metrics collection...")
            self.metrics.start_monitoring()
            
            # 10. Run initial model training and validation
            logger.info("🧠 Training and validating models...")
            await self._train_and_validate_models()
            
            logger.info("✅ Portfolio Engine initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Portfolio Engine: {e}")
            raise
    
    async def _load_initial_data(self):
        """Load initial market data"""
        # Define symbols to track
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        
        # Load historical data (2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        # Fetch data
        raw_data = await self.components['data_source'].fetch_data(
            symbols, start_date, end_date
        )
        
        if raw_data.empty:
            raise ValueError("No data loaded - check symbols and date range")
        
        # Process data
        processed_data = self.components['data_processor'].process_market_data(raw_data)
        
        # Store in cache
        self.data_cache['raw_data'] = raw_data
        self.data_cache['processed_data'] = processed_data
        self.data_cache['symbols'] = symbols
        
        logger.info(f"✅ Loaded data: {processed_data.shape[0]} days, {processed_data.shape[1]} features")
    
    async def _train_and_validate_models(self):
        """Train and validate all models"""
        processed_data = self.data_cache['processed_data']
        
        # Train regime detection ensemble
        logger.info("Training regime detection ensemble...")
        self.components['regime_detector'].fit(processed_data)
        
        # Get regime predictions
        regime_predictions = self.components['regime_detector'].predict_regimes(processed_data)
        regime_probabilities = self.components['regime_detector'].predict_regime_probabilities(processed_data)
        
        # Store regime results
        self.data_cache['regime_predictions'] = regime_predictions
        self.data_cache['regime_probabilities'] = regime_probabilities
        
        # Validate regime detection model
        logger.info("Validating regime detection model...")
        validation_results = self.components['validator'].validate_regime_detection_model(
            self.components['regime_detector'],
            processed_data,
            test_size=0.3,
            n_splits=5
        )
        
        # Log validation results
        logger.info(f"Regime detection validation - CV Score: {validation_results['cross_validation']['mean_cv_score']:.4f}")
        
        # Update metrics
        self.metrics.update_model_metrics(
            'regime_ensemble',
            validation_results['cross_validation']['mean_cv_score']
        )
        
        # Generate validation report
        validation_report = self.components['validator'].generate_validation_report(validation_results)
        
        # Save validation report
        report_path = Path("reports") / f"model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(validation_report)
        
        logger.info(f"✅ Model validation complete - report saved to {report_path}")
    
    async def run_portfolio_engine(self):
        """Main portfolio engine loop"""
        logger.info("🔄 Starting portfolio engine main loop...")
        self.running = True
        
        # Start web dashboard in background
        dashboard_thread = threading.Thread(
            target=self.components['dashboard'].run_server,
            kwargs={'debug': False, 'host': '0.0.0.0', 'port': 8050},
            daemon=True
        )
        dashboard_thread.start()
        logger.info("🌐 Web dashboard started at http://localhost:8050")
        
        # Start metrics server
        metrics_thread = threading.Thread(
            target=MetricsServer(self.metrics, port=8080).start,
            daemon=True
        )
        metrics_thread.start()
        logger.info("📊 Metrics server started at http://localhost:8080/metrics")
        
        # Main processing loop
        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"🔄 Processing iteration {iteration}")
                
                # Update data (in production, this would fetch new data)
                await self._update_data()
                
                # Update regime detection
                self._update_regime_detection()
                
                # Optimize portfolio
                portfolio_weights = self._optimize_portfolio()
                
                # Monitor risk
                risk_metrics = self._monitor_risk(portfolio_weights)
                
                # Update metrics
                self._update_system_metrics(portfolio_weights, risk_metrics)
                
                # Generate reports (periodically)
                if iteration % 10 == 0:  # Every 10 iterations
                    self._generate_reports(portfolio_weights, risk_metrics)
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"❌ Error in main loop iteration {iteration}: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _update_data(self):
        """Update market data (placeholder for real-time updates)"""
        # In production, this would fetch latest market data
        # For demo, we'll simulate data updates
        processed_data = self.data_cache['processed_data']
        
        # Update data quality metrics
        self.metrics.update_data_quality_metrics(
            'yahoo_finance',
            freshness_minutes=5.0,  # Simulated 5 minutes fresh
            quality_score=0.95,     # 95% quality score
            missing_points=0        # No missing points
        )
    
    def _update_regime_detection(self):
        """Update regime detection"""
        processed_data = self.data_cache['processed_data']
        
        # Get latest regime predictions
        regime_predictions = self.components['regime_detector'].predict_regimes(processed_data.tail(1))
        regime_probabilities = self.components['regime_detector'].predict_regime_probabilities(processed_data.tail(1))
        
        current_regime = regime_predictions[-1]
        regime_confidence = np.max(regime_probabilities[-1])
        
        # Check for regime switch
        if hasattr(self, '_last_regime') and self._last_regime != current_regime:
            logger.info(f"🔄 Regime switch detected: {self._last_regime} → {current_regime}")
            self.metrics.record_regime_switch()
        
        self._last_regime = current_regime
        
        # Update regime metrics
        self.metrics.update_regime_metrics({
            'current_regime': current_regime,
            'regime_confidence': regime_confidence
        })
        
        logger.info(f"Current regime: {current_regime} (confidence: {regime_confidence:.2%})")
    
    def _optimize_portfolio(self):
        """Optimize portfolio allocation"""
        processed_data = self.data_cache['processed_data']
        symbols = self.data_cache['symbols']
        
        # Calculate expected returns (simple historical mean)
        returns_data = processed_data[symbols] if all(s in processed_data.columns for s in symbols) else processed_data.iloc[:, :len(symbols)]
        expected_returns = returns_data.pct_change().mean() * 252  # Annualized
        
        # Calculate covariance matrix
        cov_matrix = returns_data.pct_change().cov() * 252  # Annualized
        
        # Optimize portfolio using mean-variance
        try:
            optimal_weights = self.components['optimizer'].optimize_mean_variance(
                expected_returns.values,
                cov_matrix.values,
                risk_aversion=1.0
            )
            
            # Create weights series
            portfolio_weights = pd.Series(optimal_weights, index=expected_returns.index)
            
            logger.info(f"Portfolio optimized - Top 3 weights: {portfolio_weights.nlargest(3).to_dict()}")
            
            return portfolio_weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return equal weights as fallback
            n_assets = len(expected_returns)
            equal_weights = np.ones(n_assets) / n_assets
            return pd.Series(equal_weights, index=expected_returns.index)
    
    def _monitor_risk(self, portfolio_weights):
        """Monitor portfolio risk"""
        processed_data = self.data_cache['processed_data']
        
        # Calculate portfolio returns
        symbols = self.data_cache['symbols']
        returns_data = processed_data[symbols] if all(s in processed_data.columns for s in symbols) else processed_data.iloc[:, :len(symbols)]
        returns = returns_data.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        
        # Calculate risk metrics
        risk_metrics = {
            'portfolio_volatility': portfolio_returns.std() * np.sqrt(252),
            'var_95': np.percentile(portfolio_returns, 5),
            'var_99': np.percentile(portfolio_returns, 1),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'sharpe_ratio': (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        }
        
        # Check for risk alerts
        self._check_risk_alerts(risk_metrics)
        
        return risk_metrics
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return abs(drawdown.min())
    
    def _check_risk_alerts(self, risk_metrics):
        """Check for risk limit breaches"""
        # Define risk limits
        risk_limits = {
            'portfolio_volatility': 0.20,  # 20% annual volatility
            'var_95': 0.05,                # 5% daily VaR
            'max_drawdown': 0.15           # 15% max drawdown
        }
        
        for metric, value in risk_metrics.items():
            if metric in risk_limits:
                limit = risk_limits[metric]
                if abs(value) > limit:
                    severity = 'HIGH' if abs(value) > limit * 1.5 else 'MEDIUM'
                    logger.warning(f"⚠️ Risk alert: {metric} = {value:.2%} exceeds limit {limit:.2%}")
                    self.metrics.fire_alert(f'risk_{metric}', severity)
    
    def _update_system_metrics(self, portfolio_weights, risk_metrics):
        """Update system metrics"""
        # Calculate portfolio value (assume $1M initial)
        portfolio_value = 1000000  # Placeholder
        
        # Create portfolio data for metrics
        portfolio_data = {
            'portfolio_value': portfolio_value,
            'portfolio_returns': pd.Series([0.001]),  # Placeholder daily return
            'risk_metrics': risk_metrics
        }
        
        # Update metrics
        self.metrics.update_portfolio_metrics(portfolio_data)
    
    def _generate_reports(self, portfolio_weights, risk_metrics):
        """Generate periodic reports"""
        logger.info("📋 Generating reports...")
        
        # Prepare portfolio data for reporting
        portfolio_data = {
            'portfolio_value': 1000000,  # Placeholder
            'portfolio_returns': pd.Series([0.001]),  # Placeholder
            'portfolio_weights': portfolio_weights,
            'risk_metrics': risk_metrics,
            'regime_data': {
                'current_regime_name': f'Regime {getattr(self, "_last_regime", 0)}',
                'regime_confidence': 0.85  # Placeholder
            },
            'risk_alerts': []
        }
        
        # Generate daily report
        try:
            report_path = self.components['reporting'].report_generator.generate_daily_report(portfolio_data)
            logger.info(f"📄 Daily report generated: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
    
    def shutdown(self):
        """Shutdown the portfolio engine"""
        logger.info("🛑 Shutting down Portfolio Engine...")
        self.running = False
        
        # Stop monitoring
        if hasattr(self.metrics, 'stop_monitoring'):
            self.metrics.stop_monitoring()
        
        # Stop reporting scheduler
        if 'reporting' in self.components:
            self.components['reporting'].stop_scheduler()
        
        logger.info("✅ Portfolio Engine shutdown complete")


async def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║      🚀 End-to-End Stochastic Portfolio Engine 🚀          ║
    ║             with Regime Detection via HMM                   ║
    ║                                                              ║
    ║  • Real-time regime detection and portfolio optimization    ║
    ║  • Machine learning ensemble methods                        ║
    ║  • Comprehensive risk monitoring and reporting              ║
    ║  • Interactive web dashboard and automated alerts           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create and initialize orchestrator
    orchestrator = PortfolioEngineOrchestrator()
    
    try:
        # Initialize all components
        await orchestrator.initialize()
        
        print("\n🎉 Portfolio Engine is ready!")
        print("🌐 Web Dashboard: http://localhost:8050")
        print("📊 Metrics: http://localhost:8080/metrics")
        print("📋 Reports: ./reports/")
        print("\nPress Ctrl+C to stop...\n")
        
        # Run main engine
        await orchestrator.run_portfolio_engine()
        
    except KeyboardInterrupt:
        logger.info("👋 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    # Run the portfolio engine
    asyncio.run(main())