#!/usr/bin/env python3
"""
Advanced Backtesting Framework Demonstration

This script demonstrates the comprehensive advanced backtesting capabilities
of the Enhanced Stochastic Portfolio Engine, including:

1. Realistic trading simulation with market impact and slippage
2. Multiple backtesting modes (vectorized, event-driven, walk-forward, Monte Carlo)
3. Comprehensive performance and risk analytics
4. Transaction cost modeling and execution simulation
5. Regime-aware portfolio optimization

Usage:
    python demo_advanced_backtesting.py [--mode MODE] [--symbols SYMBOLS] [--years YEARS]
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
import json
import time
import warnings

# Advanced backtesting imports
from src.backtesting.framework.advanced_backtesting import (
    AdvancedBacktestingFramework,
    AdvancedBacktestConfig,
    BacktestMode,
    RebalanceFrequency,
    create_advanced_backtesting_framework
)
from src.backtesting.execution.trading_simulator import (
    create_trading_simulator,
    TradingCosts,
    AdvancedMarketImpactModel
)
from src.data.ingestion.enhanced_data_pipeline import (
    create_enhanced_data_pipeline,
    DataPipelineConfig
)

from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_file="backtesting_demo.log")
logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class AdvancedBacktestingDemo:
    """Demonstration of advanced backtesting capabilities"""
    
    def __init__(self):
        self.demo_results = {}
        self.comparison_results = {}
        
        # Default demo configuration
        self.default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'NVDA', 'SPY']
        self.benchmark_symbol = 'SPY'
        
    def run_comprehensive_demo(
        self, 
        mode: BacktestMode = BacktestMode.EVENT_DRIVEN,
        symbols: Optional[List[str]] = None,
        years: int = 3,
        initial_capital: float = 1000000
    ):
        """Run comprehensive advanced backtesting demonstration"""
        
        logger.info("🚀 Starting Advanced Backtesting Framework Demo")
        logger.info("=" * 70)
        
        # Setup
        if symbols is None:
            symbols = self.default_symbols
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        logger.info(f"📊 Demo Configuration:")
        logger.info(f"   - Mode: {mode.value}")
        logger.info(f"   - Symbols: {', '.join(symbols)}")
        logger.info(f"   - Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   - Initial Capital: ${initial_capital:,.0f}")
        
        try:
            # Phase 1: Basic Backtesting Demo
            await self._demo_basic_backtesting(mode, symbols, start_date, end_date, initial_capital)
            
            # Phase 2: Trading Simulation Demo
            await self._demo_trading_simulation(symbols, start_date, end_date, initial_capital)
            
            # Phase 3: Multiple Modes Comparison
            await self._demo_multiple_modes_comparison(symbols, start_date, end_date, initial_capital)
            
            # Phase 4: Walk-Forward Analysis
            if mode in [BacktestMode.WALK_FORWARD, BacktestMode.MONTE_CARLO]:
                await self._demo_advanced_analysis(mode, symbols, start_date, end_date, initial_capital)
            
            # Phase 5: Results Analysis and Reporting
            await self._demo_results_analysis()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        
        logger.info("✅ Advanced Backtesting Demo completed successfully!")
    
    async def _demo_basic_backtesting(
        self, 
        mode: BacktestMode, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime, 
        initial_capital: float
    ):
        """Demo Phase 1: Basic Advanced Backtesting"""
        
        logger.info("\n📈 PHASE 1: Advanced Backtesting Framework")
        logger.info("-" * 50)
        
        # Create enhanced data pipeline
        logger.info("🔧 Initializing enhanced data pipeline...")
        pipeline_config = DataPipelineConfig(
            enable_realtime=False,  # Disable for backtesting
            enable_caching=True,
            enable_quality_monitoring=True
        )
        data_pipeline = create_enhanced_data_pipeline(pipeline_config)
        
        # Create advanced backtesting configuration
        backtest_config = AdvancedBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            symbols=symbols,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            commission_rate=0.001,  # 0.1% commission
            bid_ask_spread=0.0005,  # 0.05% spread
            market_impact_model="advanced",
            optimization_method="mean_variance",
            lookback_window=252,
            max_weight=0.25,
            use_regime_detection=True,
            regime_components=3,
            benchmark_symbol=self.benchmark_symbol,
            verbose=True
        )
        
        # Create and run backtest
        logger.info(f"🎯 Running {mode.value} backtest...")
        framework = AdvancedBacktestingFramework(backtest_config, data_pipeline)
        
        start_time = time.time()
        results = framework.run_backtest(mode)
        execution_time = time.time() - start_time
        
        # Store results for analysis
        self.demo_results['basic_backtest'] = {
            'mode': mode.value,
            'results': results,
            'execution_time': execution_time
        }
        
        # Display key results
        if not results.portfolio_returns.empty:
            logger.info("✅ Backtesting completed successfully")
            logger.info(f"   - Execution time: {execution_time:.2f} seconds")
            logger.info(f"   - Total return: {results.performance_metrics.get('total_return', 0):.2%}")
            logger.info(f"   - Annualized return: {results.performance_metrics.get('annualized_return', 0):.2%}")
            logger.info(f"   - Sharpe ratio: {results.performance_metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"   - Max drawdown: {results.performance_metrics.get('max_drawdown', 0):.2%}")
            
            if results.trading_costs:
                logger.info(f"   - Total trading costs: ${results.trading_costs.get('total_trading_costs', 0):,.2f}")
                logger.info(f"   - Number of trades: {results.trading_costs.get('total_trades', 0)}")
        else:
            logger.warning("⚠️  Backtest returned empty results")
    
    async def _demo_trading_simulation(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime, 
        initial_capital: float
    ):
        """Demo Phase 2: Realistic Trading Simulation"""
        
        logger.info("\n🎮 PHASE 2: Realistic Trading Simulation")
        logger.info("-" * 50)
        
        # Create trading simulator with different cost structures
        cost_scenarios = {
            'low_cost': TradingCosts(
                commission_rate=0.0005,
                bid_ask_spread=0.0002,
                market_impact_linear=0.00005,
                slippage_std=0.0001
            ),
            'moderate_cost': TradingCosts(
                commission_rate=0.001,
                bid_ask_spread=0.0005,
                market_impact_linear=0.0001,
                slippage_std=0.0002
            ),
            'high_cost': TradingCosts(
                commission_rate=0.002,
                bid_ask_spread=0.001,
                market_impact_linear=0.0002,
                slippage_std=0.0003
            )
        }
        
        simulation_results = {}
        
        for scenario_name, trading_costs in cost_scenarios.items():
            logger.info(f"📊 Running {scenario_name} trading simulation...")
            
            # Create configuration with specific trading costs
            config = AdvancedBacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                symbols=symbols[:6],  # Use fewer symbols for detailed simulation
                rebalance_frequency=RebalanceFrequency.MONTHLY,
                commission_rate=trading_costs.commission_rate,
                bid_ask_spread=trading_costs.bid_ask_spread,
                market_impact_model="advanced",
                optimization_method="mean_variance",
                use_regime_detection=False,  # Disable for cleaner comparison
                verbose=False
            )
            
            # Run event-driven backtest
            framework = AdvancedBacktestingFramework(config)
            results = framework.run_backtest(BacktestMode.EVENT_DRIVEN)
            
            simulation_results[scenario_name] = results
            
            if not results.portfolio_returns.empty:
                logger.info(f"   ✅ {scenario_name}: Return = {results.performance_metrics.get('total_return', 0):.2%}, "
                          f"Costs = ${results.trading_costs.get('total_trading_costs', 0):,.0f}")
        
        # Store results and analyze impact of trading costs
        self.demo_results['trading_simulation'] = simulation_results
        
        logger.info("\n📈 Trading Cost Impact Analysis:")
        if simulation_results:
            for scenario, results in simulation_results.items():
                if not results.portfolio_returns.empty:
                    cost_impact = results.trading_costs.get('total_cost_rate', 0) * 100
                    logger.info(f"   - {scenario}: {cost_impact:.3f}% of portfolio value in costs")
    
    async def _demo_multiple_modes_comparison(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime, 
        initial_capital: float
    ):
        """Demo Phase 3: Multiple Backtesting Modes Comparison"""
        
        logger.info("\n🔄 PHASE 3: Multiple Backtesting Modes Comparison")
        logger.info("-" * 50)
        
        # Test different backtesting modes
        modes_to_test = [BacktestMode.VECTORIZED, BacktestMode.EVENT_DRIVEN]
        mode_results = {}
        
        base_config = AdvancedBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            symbols=symbols[:8],  # Use subset for comparison
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            commission_rate=0.001,
            optimization_method="mean_variance",
            use_regime_detection=True,
            verbose=False
        )
        
        for mode in modes_to_test:
            logger.info(f"⚡ Running {mode.value} backtest...")
            
            framework = AdvancedBacktestingFramework(base_config)
            
            start_time = time.time()
            results = framework.run_backtest(mode)
            execution_time = time.time() - start_time
            
            mode_results[mode.value] = {
                'results': results,
                'execution_time': execution_time
            }
            
            if not results.portfolio_returns.empty:
                logger.info(f"   ✅ {mode.value}: {execution_time:.2f}s, "
                          f"Return = {results.performance_metrics.get('total_return', 0):.2%}")
        
        # Store comparison results
        self.comparison_results['mode_comparison'] = mode_results
        
        # Performance comparison
        logger.info("\n📊 Mode Performance Comparison:")
        for mode, data in mode_results.items():
            if not data['results'].portfolio_returns.empty:
                metrics = data['results'].performance_metrics
                logger.info(f"   {mode}:")
                logger.info(f"     - Execution time: {data['execution_time']:.2f}s")
                logger.info(f"     - Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                logger.info(f"     - Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    async def _demo_advanced_analysis(
        self, 
        mode: BacktestMode, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime, 
        initial_capital: float
    ):
        """Demo Phase 4: Advanced Analysis (Walk-Forward or Monte Carlo)"""
        
        logger.info(f"\n🔬 PHASE 4: {mode.value.upper()} Analysis")
        logger.info("-" * 50)
        
        if mode == BacktestMode.WALK_FORWARD:
            logger.info("📊 Running Walk-Forward Analysis...")
            
            config = AdvancedBacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                symbols=symbols[:6],  # Fewer symbols for intensive analysis
                rebalance_frequency=RebalanceFrequency.MONTHLY,
                training_window=252,  # 1 year training
                test_window=63,      # 3 months testing
                step_size=21,        # 1 month step
                optimization_method="mean_variance",
                use_regime_detection=True,
                verbose=False
            )
            
        elif mode == BacktestMode.MONTE_CARLO:
            logger.info("🎲 Running Monte Carlo Analysis...")
            
            config = AdvancedBacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                symbols=symbols[:5],  # Even fewer symbols for MC
                n_simulations=100,   # Reduced for demo
                bootstrap_method="circular",
                parallel_processing=True,
                optimization_method="mean_variance",
                verbose=False
            )
        
        framework = AdvancedBacktestingFramework(config)
        
        start_time = time.time()
        results = framework.run_backtest(mode)
        execution_time = time.time() - start_time
        
        # Store results
        self.demo_results['advanced_analysis'] = {
            'mode': mode.value,
            'results': results,
            'execution_time': execution_time
        }
        
        logger.info(f"✅ {mode.value} analysis completed in {execution_time:.2f} seconds")
        
        if mode == BacktestMode.WALK_FORWARD and results.walk_forward_results:
            logger.info(f"   - Number of walk-forward windows: {len(results.walk_forward_results)}")
            avg_return = np.mean([w['performance'].get('total_return', 0) for w in results.walk_forward_results])
            logger.info(f"   - Average window return: {avg_return:.2%}")
        
        elif mode == BacktestMode.MONTE_CARLO and results.monte_carlo_results:
            mc_results = results.monte_carlo_results
            logger.info(f"   - Simulations completed: {mc_results.get('n_simulations', 0)}")
            if 'performance_distribution' in mc_results:
                return_dist = mc_results['performance_distribution'].get('total_return', {})
                logger.info(f"   - Return distribution: {return_dist.get('mean', 0):.2%} ± {return_dist.get('std', 0):.2%}")
                logger.info(f"   - 5th percentile: {return_dist.get('percentiles', {}).get('5%', 0):.2%}")
                logger.info(f"   - 95th percentile: {return_dist.get('percentiles', {}).get('95%', 0):.2%}")
    
    async def _demo_results_analysis(self):
        """Demo Phase 5: Comprehensive Results Analysis"""
        
        logger.info("\n📋 PHASE 5: Results Analysis and Reporting")
        logger.info("-" * 50)
        
        # Generate comprehensive analysis report
        report = self._generate_comprehensive_report()
        
        # Save detailed results
        self._save_detailed_results()
        
        # Display summary
        logger.info("📊 Demo Summary:")
        logger.info(f"   - Backtesting modes tested: {len(self.comparison_results.get('mode_comparison', {}))}")
        logger.info(f"   - Trading cost scenarios: {len(self.demo_results.get('trading_simulation', {}))}")
        logger.info(f"   - Advanced analyses: {1 if 'advanced_analysis' in self.demo_results else 0}")
        
        logger.info("\n💾 Generated Files:")
        logger.info("   - backtesting_demo_results.json (Detailed results)")
        logger.info("   - backtesting_analysis_report.txt (Comprehensive report)")
        
        # Performance comparison across all tests
        self._display_performance_comparison()
    
    def _generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report"""
        
        report = "ADVANCED BACKTESTING FRAMEWORK ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Basic backtest results
        if 'basic_backtest' in self.demo_results:
            basic = self.demo_results['basic_backtest']
            results = basic['results']
            
            report += "BASIC BACKTESTING RESULTS\n"
            report += "-" * 30 + "\n"
            report += f"Mode: {basic['mode']}\n"
            report += f"Execution Time: {basic['execution_time']:.2f} seconds\n"
            
            if not results.portfolio_returns.empty:
                metrics = results.performance_metrics
                report += f"Total Return: {metrics.get('total_return', 0):.2%}\n"
                report += f"Annualized Return: {metrics.get('annualized_return', 0):.2%}\n"
                report += f"Volatility: {metrics.get('annualized_volatility', 0):.2%}\n"
                report += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
                report += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
                report += f"Alpha: {metrics.get('alpha', 0):.2%}\n"
                report += f"Information Ratio: {metrics.get('information_ratio', 0):.3f}\n"
            
            report += "\n"
        
        # Trading simulation analysis
        if 'trading_simulation' in self.demo_results:
            report += "TRADING COST IMPACT ANALYSIS\n"
            report += "-" * 30 + "\n"
            
            for scenario, results in self.demo_results['trading_simulation'].items():
                if not results.portfolio_returns.empty:
                    metrics = results.performance_metrics
                    costs = results.trading_costs
                    
                    report += f"{scenario.upper()}:\n"
                    report += f"  Return: {metrics.get('total_return', 0):.2%}\n"
                    report += f"  Trading Costs: ${costs.get('total_trading_costs', 0):,.0f}\n"
                    report += f"  Cost Rate: {costs.get('total_cost_rate', 0):.3%}\n"
                    report += f"  Number of Trades: {costs.get('total_trades', 0)}\n\n"
        
        # Mode comparison
        if 'mode_comparison' in self.comparison_results:
            report += "BACKTESTING MODE COMPARISON\n"
            report += "-" * 30 + "\n"
            
            for mode, data in self.comparison_results['mode_comparison'].items():
                if not data['results'].portfolio_returns.empty:
                    metrics = data['results'].performance_metrics
                    
                    report += f"{mode.upper()}:\n"
                    report += f"  Execution Time: {data['execution_time']:.2f}s\n"
                    report += f"  Return: {metrics.get('total_return', 0):.2%}\n"
                    report += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
                    report += f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n\n"
        
        # Advanced analysis
        if 'advanced_analysis' in self.demo_results:
            advanced = self.demo_results['advanced_analysis']
            
            report += f"{advanced['mode'].upper()} ANALYSIS\n"
            report += "-" * 30 + "\n"
            report += f"Execution Time: {advanced['execution_time']:.2f} seconds\n"
            
            results = advanced['results']
            if results.walk_forward_results:
                report += f"Walk-Forward Windows: {len(results.walk_forward_results)}\n"
                avg_return = np.mean([w['performance'].get('total_return', 0) for w in results.walk_forward_results])
                report += f"Average Window Return: {avg_return:.2%}\n"
            
            elif results.monte_carlo_results:
                mc = results.monte_carlo_results
                report += f"Monte Carlo Simulations: {mc.get('n_simulations', 0)}\n"
                if 'performance_distribution' in mc:
                    return_dist = mc['performance_distribution'].get('total_return', {})
                    report += f"Return Mean: {return_dist.get('mean', 0):.2%}\n"
                    report += f"Return Std: {return_dist.get('std', 0):.2%}\n"
            
            report += "\n"
        
        report += "=" * 60 + "\n"
        report += "Report completed successfully.\n"
        
        # Save report
        with open('backtesting_analysis_report.txt', 'w') as f:
            f.write(report)
        
        return report
    
    def _save_detailed_results(self):
        """Save detailed results to JSON"""
        
        # Prepare results for JSON serialization
        serializable_results = {}
        
        for key, value in self.demo_results.items():
            if key == 'basic_backtest':
                serializable_results[key] = {
                    'mode': value['mode'],
                    'execution_time': value['execution_time'],
                    'performance_metrics': value['results'].performance_metrics,
                    'risk_metrics': value['results'].risk_metrics,
                    'trading_costs': value['results'].trading_costs
                }
            
            elif key == 'trading_simulation':
                serializable_results[key] = {}
                for scenario, results in value.items():
                    serializable_results[key][scenario] = {
                        'performance_metrics': results.performance_metrics,
                        'trading_costs': results.trading_costs
                    }
            
            elif key == 'advanced_analysis':
                serializable_results[key] = {
                    'mode': value['mode'],
                    'execution_time': value['execution_time'],
                    'performance_metrics': value['results'].performance_metrics
                }
                
                if value['results'].walk_forward_results:
                    serializable_results[key]['walk_forward_summary'] = {
                        'n_windows': len(value['results'].walk_forward_results),
                        'avg_return': np.mean([w['performance'].get('total_return', 0) 
                                             for w in value['results'].walk_forward_results])
                    }
                
                if value['results'].monte_carlo_results:
                    serializable_results[key]['monte_carlo_summary'] = value['results'].monte_carlo_results
        
        # Add comparison results
        if self.comparison_results:
            serializable_results['comparisons'] = {}
            for key, value in self.comparison_results.items():
                if key == 'mode_comparison':
                    serializable_results['comparisons'][key] = {}
                    for mode, data in value.items():
                        serializable_results['comparisons'][key][mode] = {
                            'execution_time': data['execution_time'],
                            'performance_metrics': data['results'].performance_metrics
                        }
        
        # Save to JSON
        with open('backtesting_demo_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    def _display_performance_comparison(self):
        """Display performance comparison across all tests"""
        
        logger.info("\n🏆 PERFORMANCE COMPARISON SUMMARY")
        logger.info("-" * 40)
        
        all_results = []
        
        # Collect all results
        if 'basic_backtest' in self.demo_results:
            basic = self.demo_results['basic_backtest']
            if not basic['results'].portfolio_returns.empty:
                all_results.append({
                    'name': f"Basic ({basic['mode']})",
                    'return': basic['results'].performance_metrics.get('total_return', 0),
                    'sharpe': basic['results'].performance_metrics.get('sharpe_ratio', 0),
                    'drawdown': basic['results'].performance_metrics.get('max_drawdown', 0)
                })
        
        if 'mode_comparison' in self.comparison_results:
            for mode, data in self.comparison_results['mode_comparison'].items():
                if not data['results'].portfolio_returns.empty:
                    all_results.append({
                        'name': mode,
                        'return': data['results'].performance_metrics.get('total_return', 0),
                        'sharpe': data['results'].performance_metrics.get('sharpe_ratio', 0),
                        'drawdown': data['results'].performance_metrics.get('max_drawdown', 0)
                    })
        
        # Display comparison table
        if all_results:
            logger.info(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10}")
            logger.info("-" * 50)
            
            for result in all_results:
                logger.info(f"{result['name']:<20} {result['return']:>8.2%} {result['sharpe']:>7.2f} {result['drawdown']:>8.2%}")


def main():
    """Main demonstration function"""
    
    parser = argparse.ArgumentParser(description="Advanced Backtesting Framework Demo")
    parser.add_argument('--mode', type=str, default='event_driven',
                       choices=['vectorized', 'event_driven', 'walk_forward', 'monte_carlo'],
                       help='Backtesting mode to demonstrate')
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V'],
                       help='Symbols to include in backtest')
    parser.add_argument('--years', type=int, default=3,
                       help='Number of years of historical data')
    parser.add_argument('--capital', type=float, default=1000000,
                       help='Initial capital')
    
    args = parser.parse_args()
    
    # Convert mode string to enum
    mode_map = {
        'vectorized': BacktestMode.VECTORIZED,
        'event_driven': BacktestMode.EVENT_DRIVEN,
        'walk_forward': BacktestMode.WALK_FORWARD,
        'monte_carlo': BacktestMode.MONTE_CARLO
    }
    
    mode = mode_map[args.mode]
    
    logger.info("🚀 Advanced Backtesting Framework Demonstration")
    logger.info("=" * 60)
    
    # Create and run demo
    demo = AdvancedBacktestingDemo()
    
    try:
        asyncio.run(demo.run_comprehensive_demo(
            mode=mode,
            symbols=args.symbols,
            years=args.years,
            initial_capital=args.capital
        ))
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()