#!/usr/bin/env python3
"""
Main entry point for the Stochastic Portfolio Engine with HMM Regime Detection

This comprehensive portfolio management system combines:
- Hidden Markov Model regime detection
- Advanced portfolio optimization 
- Monte Carlo simulation
- Comprehensive risk management
- Performance attribution analysis
- Real-time monitoring and alerting

Usage:
    python main.py                    # Run with default configuration
    python -c config/my_config.yaml  # Run with custom configuration
    
For CLI usage:
    portfolio --help                  # Show all CLI commands
    portfolio optimize -s "AAPL,GOOGL,MSFT" --monte-carlo
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
import sys
import json

from src.utils.config import get_config
from src.utils.logging_config import setup_logging, get_logger
from src.data.ingestion.data_sources import create_data_pipeline
from src.data.processing.data_processor import DataProcessor
from src.data.validation.data_validator import DataValidator
from src.models.hmm.hmm_engine import RegimeDetectionHMM, AdvancedBaumWelchHMM
from src.models.hmm.regime_analyzer import RegimeAnalyzer
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.backtesting.engine.backtesting_engine import BacktestingEngine, BacktestConfig
from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor, RiskLimits
from src.risk.dashboard.risk_dashboard import ComprehensiveRiskDashboard, DashboardConfig
from src.utils.performance_analytics import PerformanceAnalytics, PerformanceVisualizer
from src.simulation.monte_carlo_engine import MonteCarloEngine, SimulationConfig
from src.models.factors.advanced_factor_models import create_default_factor_models, FactorModelManager
from src.analytics.performance_attribution import AdvancedPerformanceAttribution


async def enhanced_main(config_file: Optional[str] = None, demo_mode: bool = False):
    """Enhanced main function showcasing all system capabilities"""
    
    # Load configuration
    if config_file:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_config()
    
    setup_logging(
        log_level=config.get('logging', {}).get('level', 'INFO'),
        log_file='portfolio_engine.log'
    )
    
    logger = get_logger(__name__)
    logger.info("🚀 Starting Enhanced Stochastic Portfolio Engine with HMM Regime Detection")
    
    try:
        # Initialize data pipeline
        data_pipeline = create_data_pipeline()
        
        # Enhanced asset universe for demonstration
        symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',  # Tech
            'JPM', 'BAC', 'GS', 'V', 'MA',            # Finance
            'JNJ', 'PFE', 'UNH', 'ABBV',              # Healthcare
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT'         # ETFs
        ]
        
        if demo_mode:
            symbols = symbols[:10]  # Reduce for demo
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.get('data', {}).get('lookback_days', 252) * 3)
        
        logger.info(f"📊 Fetching data for {len(symbols)} assets from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch comprehensive data
        price_data = await data_pipeline.fetch_asset_universe(symbols, start_date, end_date)
        features_data = await data_pipeline.create_features_dataset(symbols, start_date, end_date)
        
        # Data processing and validation
        logger.info("🔍 Processing and validating data")
        data_processor = DataProcessor(scaling_method="robust")
        processed_features = data_processor.process_data(
            features_data, handle_missing=True, detect_outliers_flag=True, scale_data=True
        )
        
        validator = DataValidator()
        validation_results = validator.validate_dataset(processed_features, "features")
        
        if validation_results['errors']:
            logger.error(f"❌ Data validation failed: {validation_results['errors']}")
            return
        
        # 1. ENHANCED HMM REGIME DETECTION
        logger.info("🧠 Training Advanced HMM Regime Detection Model")
        
        hmm_model = AdvancedBaumWelchHMM(
            n_components=config.get('hmm', {}).get('n_components', 3),
            covariance_type=config.get('hmm', {}).get('covariance_type', 'full'),
            n_iter=config.get('hmm', {}).get('n_iter', 100),
            random_state=42
        )
        
        hmm_model.fit(processed_features)
        regime_stats = hmm_model.get_regime_statistics(processed_features)
        regime_summary = hmm_model.get_regime_summary(processed_features)
        
        logger.info("📈 Regime Detection Results:")
        for regime_id, regime_name in regime_stats['regime_names'].items():
            freq = regime_stats['regime_distribution'][int(regime_id)]
            duration = regime_stats['average_regime_duration'][int(regime_id)]
            logger.info(f"  {regime_name}: {freq:.1%} frequency, {duration:.1f} avg duration")
        
        # 2. COMPREHENSIVE RISK DASHBOARD
        logger.info("⚠️  Setting up Comprehensive Risk Dashboard")
        
        dashboard_config = DashboardConfig(
            confidence_levels=[0.01, 0.05, 0.10],
            risk_free_rate=0.02
        )
        
        risk_dashboard = ComprehensiveRiskDashboard(dashboard_config)
        
        # 3. FACTOR MODEL ANALYSIS
        logger.info("📊 Initializing Factor Models")
        
        factor_manager = FactorModelManager()
        default_models = create_default_factor_models()
        
        for name, model in default_models.items():
            factor_manager.add_model(name, model)
        
        # Prepare returns for factor analysis
        adj_close = price_data.xs('Adj Close', level=0, axis=1)
        returns = adj_close.pct_change().dropna()
        
        # Fit factor models (simplified for demo)
        try:
            logger.info("🔍 Fitting Factor Models...")
            fitted_models = factor_manager.fit_all_models(returns)
            logger.info(f"✅ Successfully fitted {len(fitted_models)} factor models")
        except Exception as e:
            logger.warning(f"⚠️  Factor model fitting encountered issues: {e}")
        
        # 4. MONTE CARLO SIMULATION ENGINE
        logger.info("🎲 Setting up Monte Carlo Simulation")
        
        expected_returns = returns.mean() * 252
        covariance_matrix = returns.cov() * 252
        
        # Equal weight portfolio for demonstration
        portfolio_weights = pd.Series(
            [1.0 / len(symbols)] * len(symbols),
            index=symbols,
            name='equal_weight'
        )
        
        simulation_config = SimulationConfig(
            n_simulations=5000 if demo_mode else 10000,
            time_horizon=252,
            confidence_levels=[0.01, 0.05, 0.10],
            random_seed=42
        )
        
        mc_engine = MonteCarloEngine(simulation_config)
        
        logger.info(f"🎯 Running Monte Carlo simulation with {simulation_config.n_simulations:,} paths")
        mc_results = mc_engine.simulate_portfolio_paths(
            portfolio_weights.values,
            expected_returns.values,
            covariance_matrix.values
        )
        
        logger.info("📊 Monte Carlo Results:")
        logger.info(f"  Expected Final Value: {mc_results.statistics['mean_final_value']:.3f}")
        logger.info(f"  Probability of Loss: {mc_results.statistics['probability_of_loss']:.1%}")
        logger.info(f"  95% VaR: {abs(mc_results.risk_metrics['var_5%']):.2%}")
        logger.info(f"  95% CVaR: {abs(mc_results.risk_metrics['cvar_5%']):.2%}")
        
        # 5. PERFORMANCE ATTRIBUTION ANALYSIS
        logger.info("📈 Running Performance Attribution Analysis")
        
        attribution_analyzer = AdvancedPerformanceAttribution()
        
        # Calculate portfolio returns
        portfolio_returns = (returns @ portfolio_weights).dropna()
        
        # Get benchmark data
        benchmark_data = await data_pipeline.fetch_asset_universe(['SPY'], start_date, end_date)
        benchmark_returns = benchmark_data.xs('Adj Close', level=0, axis=1).iloc[:, 0].pct_change().dropna()
        
        # Regime-based attribution
        try:
            regime_labels = hmm_model.predict_regimes(processed_features.tail(len(portfolio_returns)))
            
            regime_attribution = attribution_analyzer.regime_based_attribution(
                portfolio_returns, benchmark_returns, regime_labels, 
                regime_stats['regime_names']
            )
            
            attribution_report = attribution_analyzer.generate_attribution_report(
                regime_attribution, "Regime-Based Analysis"
            )
            
            logger.info("📊 Performance Attribution Summary:")
            logger.info(f"  Total Active Return: {regime_attribution.total_active_return:.4f}")
            logger.info(f"  Allocation Effect: {regime_attribution.allocation_effect:.4f}")
            logger.info(f"  Selection Effect: {regime_attribution.selection_effect:.4f}")
            
        except Exception as e:
            logger.warning(f"⚠️  Performance attribution analysis encountered issues: {e}")
        
        # 6. COMPREHENSIVE BACKTESTING
        logger.info("🔄 Running Enhanced Backtesting")
        
        backtest_config = BacktestConfig(
            start_date=start_date + timedelta(days=252),
            end_date=end_date,
            initial_capital=1000000,
            rebalance_frequency='M',
            benchmark_symbol='SPY',
            lookback_window=126,
            optimization_method='mean_variance'
        )
        
        portfolio_optimizer = PortfolioOptimizationEngine()
        backtesting_engine = BacktestingEngine(backtest_config, portfolio_optimizer, hmm_model)
        
        backtest_results = backtesting_engine.run_backtest(
            returns, benchmark_returns, processed_features
        )
        
        logger.info("📊 Backtesting Results:")
        for metric, value in backtest_results.performance_metrics.items():
            if isinstance(value, (int, float)):
                if 'ratio' in metric.lower() or 'return' in metric.lower():
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
        
        # 7. RISK MONITORING AND ALERTS
        logger.info("⚠️  Setting up Risk Monitoring")
        
        risk_limits = RiskLimits(
            max_portfolio_volatility=0.25,
            max_individual_weight=0.20,
            max_drawdown=0.15,
            max_var_95=0.05
        )
        
        risk_monitor = RealTimeRiskMonitor(risk_limits)
        
        if len(backtest_results.portfolio_weights) > 0:
            latest_weights = backtest_results.portfolio_weights.iloc[-1]
            recent_returns = returns.tail(60)
            
            risk_alerts = risk_monitor.monitor_portfolio_risk(
                portfolio_weights=latest_weights,
                returns_data=recent_returns
            )
            
            logger.info(f"🚨 Risk Monitoring: {len(risk_alerts)} alerts generated")
            for alert in risk_alerts[:5]:  # Show first 5 alerts
                logger.info(f"  {alert.level.value}: {alert.message}")
        
        # 8. COMPREHENSIVE REPORTING
        logger.info("📝 Generating Comprehensive Reports")
        
        # Save enhanced results
        enhanced_results = {
            'system_info': {
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'assets_analyzed': len(symbols),
                'analysis_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days': (end_date - start_date).days
                }
            },
            'regime_analysis': {
                'model_type': 'AdvancedBaumWelchHMM',
                'n_regimes': hmm_model.n_components,
                'regime_names': regime_stats['regime_names'],
                'regime_distribution': regime_stats['regime_distribution'].tolist(),
                'converged': hmm_model.converged_,
                'log_likelihood': hmm_model.log_likelihood_history_[-1] if hmm_model.log_likelihood_history_ else None
            },
            'portfolio_optimization': {
                'method': 'mean_variance',
                'weights': dict(portfolio_weights),
                'expected_return': float(expected_returns.mean()),
                'expected_volatility': float(np.sqrt(np.diag(covariance_matrix)).mean())
            },
            'monte_carlo_simulation': {
                'n_simulations': mc_results.config.n_simulations,
                'statistics': mc_results.statistics,
                'risk_metrics': mc_results.risk_metrics,
                'execution_time': mc_results.execution_time
            },
            'backtesting_results': {
                'performance_metrics': backtest_results.performance_metrics,
                'total_trades': len(backtest_results.transactions) if backtest_results.transactions is not None else 0
            },
            'risk_monitoring': {
                'total_alerts': len(risk_alerts) if 'risk_alerts' in locals() else 0,
                'alert_levels': {
                    alert.level.value: 1 for alert in (risk_alerts if 'risk_alerts' in locals() else [])
                }
            }
        }
        
        # Save to JSON for detailed analysis
        with open('enhanced_portfolio_results.json', 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        # Save traditional CSV outputs
        if len(backtest_results.portfolio_returns) > 0:
            results_df = pd.DataFrame({
                'date': backtest_results.portfolio_returns.index,
                'portfolio_return': backtest_results.portfolio_returns.values,
                'benchmark_return': backtest_results.benchmark_returns.values,
                'cumulative_portfolio': (1 + backtest_results.portfolio_returns).cumprod().values,
                'cumulative_benchmark': (1 + backtest_results.benchmark_returns).cumprod().values
            })
            results_df.to_csv('enhanced_portfolio_timeseries.csv', index=False)
        
        if len(backtest_results.portfolio_weights) > 0:
            backtest_results.portfolio_weights.to_csv('enhanced_portfolio_weights.csv')
        
        # Save regime analysis
        regime_summary.to_csv('regime_analysis_summary.csv')
        
        logger.info("✅ Enhanced Portfolio Engine Analysis Complete!")
        logger.info("📁 Results saved to:")
        logger.info("  - enhanced_portfolio_results.json")
        logger.info("  - enhanced_portfolio_timeseries.csv")
        logger.info("  - enhanced_portfolio_weights.csv")
        logger.info("  - regime_analysis_summary.csv")
        
        # Display summary statistics
        logger.info("\n" + "="*60)
        logger.info("📊 ENHANCED PORTFOLIO ENGINE SUMMARY")
        logger.info("="*60)
        logger.info(f"🎯 Assets Analyzed: {len(symbols)}")
        logger.info(f"📅 Analysis Period: {(end_date - start_date).days} days")
        logger.info(f"🧠 Regime Detection: {hmm_model.n_components} regimes identified")
        logger.info(f"🎲 Monte Carlo: {mc_results.config.n_simulations:,} simulations")
        logger.info(f"📈 Expected Return: {expected_returns.mean():.2%}")
        logger.info(f"⚠️  95% VaR: {abs(mc_results.risk_metrics['var_5%']):.2%}")
        logger.info(f"🚨 Risk Alerts: {len(risk_alerts) if 'risk_alerts' in locals() else 0}")
        logger.info("="*60)
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced main execution: {e}", exc_info=True)
        raise


async def main():
    """Original main function for backwards compatibility"""
    return await enhanced_main()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Stochastic Portfolio Engine with HMM Regime Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default settings
  python main.py --config config.yaml     # Run with custom config
  python main.py --demo                    # Run in demo mode (faster)
  
For interactive CLI usage:
  portfolio --help                         # Show CLI commands
  portfolio optimize -s "AAPL,GOOGL,MSFT" # Optimize portfolio
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run in demo mode with reduced dataset and simulations'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Stochastic Portfolio Engine v1.0.0'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())