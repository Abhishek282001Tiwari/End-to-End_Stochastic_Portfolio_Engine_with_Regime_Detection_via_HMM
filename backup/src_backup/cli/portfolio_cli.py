#!/usr/bin/env python3

import click
import asyncio
import pandas as pd
import numpy as np
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.utils.logging_config import setup_logging, get_logger
from src.data.ingestion.data_sources import create_data_pipeline
from src.models.hmm.hmm_engine import RegimeDetectionHMM, AdvancedBaumWelchHMM
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.risk.dashboard.risk_dashboard import ComprehensiveRiskDashboard, DashboardConfig
from src.simulation.monte_carlo_engine import MonteCarloEngine, SimulationConfig, create_default_scenarios
from src.models.factors.advanced_factor_models import create_default_factor_models, FactorModelManager
from src.analytics.performance_attribution import AdvancedPerformanceAttribution
from src.backtesting.engine.backtesting_engine import BacktestingEngine, BacktestConfig

warnings.filterwarnings("ignore", category=FutureWarning)


class PortfolioCLI:
    """Command Line Interface for Portfolio Engine"""
    
    def __init__(self):
        self.config = None
        self.logger = None
        self.data_pipeline = None
        self.setup_complete = False
        
    def setup(self, config_file: Optional[str] = None, log_level: str = "INFO"):
        """Initialize CLI with configuration"""
        try:
            # Load configuration
            if config_file and os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        self.config = yaml.safe_load(f)
                    else:
                        self.config = json.load(f)
            else:
                self.config = get_config()
            
            # Setup logging
            setup_logging(log_level=log_level, log_file="portfolio_cli.log")
            self.logger = get_logger(__name__)
            
            # Initialize data pipeline
            self.data_pipeline = create_data_pipeline()
            
            self.setup_complete = True
            self.logger.info("Portfolio CLI initialized successfully")
            
        except Exception as e:
            print(f"Error initializing CLI: {e}")
            sys.exit(1)


# Create CLI instance
cli_instance = PortfolioCLI()


@click.group()
@click.option('--config', '-c', type=str, help='Configuration file path')
@click.option('--log-level', '-l', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Logging level')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(config: Optional[str], log_level: str, verbose: bool):
    """
    Stochastic Portfolio Engine with HMM Regime Detection
    
    A comprehensive portfolio optimization system using Hidden Markov Models
    for market regime detection and dynamic portfolio allocation.
    """
    if verbose:
        log_level = 'DEBUG'
    
    cli_instance.setup(config, log_level)
    
    if not cli_instance.setup_complete:
        click.echo("Failed to initialize CLI", err=True)
        sys.exit(1)


@main.command()
@click.option('--symbols', '-s', type=str, required=True,
              help='Comma-separated list of symbols (e.g., AAPL,GOOGL,MSFT)')
@click.option('--start-date', type=str, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=str, help='End date (YYYY-MM-DD)')
@click.option('--regimes', '-r', type=int, default=3, help='Number of market regimes')
@click.option('--method', '-m', type=click.Choice(['mean_variance', 'black_litterman', 'risk_parity']),
              default='mean_variance', help='Optimization method')
@click.option('--output', '-o', type=str, help='Output file for results')
@click.option('--monte-carlo', is_flag=True, help='Run Monte Carlo simulation')
@click.option('--simulations', type=int, default=10000, help='Number of Monte Carlo simulations')
def optimize(symbols: str, start_date: Optional[str], end_date: Optional[str], 
             regimes: int, method: str, output: Optional[str], 
             monte_carlo: bool, simulations: int):
    """
    Optimize portfolio allocation with regime detection
    
    Example:
        portfolio optimize -s "AAPL,GOOGL,MSFT,AMZN" -r 3 -m mean_variance --monte-carlo
    """
    try:
        click.echo("🚀 Starting Portfolio Optimization...")
        
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        click.echo(f"📊 Assets: {', '.join(symbol_list)}")
        
        # Set date range
        if not end_date:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if not start_date:
            start_date = end_date - timedelta(days=cli_instance.config.get('data', {}).get('lookback_days', 252) * 2)
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        click.echo(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Run optimization
        result = asyncio.run(_run_optimization(
            symbol_list, start_date, end_date, regimes, method, 
            monte_carlo, simulations
        ))
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"💾 Results saved to {output}")
        else:
            _display_optimization_results(result)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


async def _run_optimization(symbol_list: List[str], start_date: datetime, 
                           end_date: datetime, regimes: int, method: str,
                           monte_carlo: bool, simulations: int) -> Dict[str, Any]:
    """Run the portfolio optimization process"""
    
    logger = cli_instance.logger
    data_pipeline = cli_instance.data_pipeline
    
    # Fetch data
    click.echo("📥 Fetching market data...")
    price_data = await data_pipeline.fetch_asset_universe(symbol_list, start_date, end_date)
    features_data = await data_pipeline.create_features_dataset(symbol_list, start_date, end_date)
    
    # Train HMM model
    click.echo(f"🧠 Training HMM with {regimes} regimes...")
    hmm_model = AdvancedBaumWelchHMM(n_components=regimes, random_state=42)
    hmm_model.fit(features_data)
    
    # Get regime statistics
    regime_stats = hmm_model.get_regime_statistics(features_data)
    
    # Prepare return data
    adj_close = price_data.xs('Adj Close', level=0, axis=1)
    returns = adj_close.pct_change().dropna()
    
    # Portfolio optimization
    click.echo(f"🎯 Optimizing portfolio using {method}...")
    optimizer = PortfolioOptimizationEngine()
    
    expected_returns = returns.mean() * 252
    covariance_matrix = returns.cov() * 252
    
    optimization_result = optimizer.optimize_portfolio(
        method, expected_returns.values, covariance_matrix.values
    )
    
    # Create portfolio weights series
    portfolio_weights = pd.Series(
        optimization_result['weights'], 
        index=symbol_list, 
        name='weights'
    )
    
    result = {
        'optimization': {
            'method': method,
            'weights': dict(portfolio_weights),
            'expected_return': optimization_result['expected_return'],
            'expected_volatility': optimization_result['expected_volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio']
        },
        'regime_analysis': {
            'n_regimes': regimes,
            'regime_distribution': regime_stats['regime_distribution'].tolist(),
            'regime_names': regime_stats['regime_names']
        }
    }
    
    # Monte Carlo simulation
    if monte_carlo:
        click.echo(f"🎲 Running Monte Carlo simulation ({simulations:,} paths)...")
        
        config = SimulationConfig(n_simulations=simulations, time_horizon=252)
        mc_engine = MonteCarloEngine(config)
        
        mc_result = mc_engine.simulate_portfolio_paths(
            portfolio_weights.values,
            expected_returns.values,
            covariance_matrix.values
        )
        
        result['monte_carlo'] = {
            'statistics': mc_result.statistics,
            'risk_metrics': mc_result.risk_metrics,
            'confidence_intervals': mc_result.confidence_intervals,
            'execution_time': mc_result.execution_time
        }
    
    return result


def _display_optimization_results(result: Dict[str, Any]):
    """Display optimization results in a formatted way"""
    
    click.echo("\n" + "="*60)
    click.echo("📈 PORTFOLIO OPTIMIZATION RESULTS")
    click.echo("="*60)
    
    # Optimization results
    opt = result['optimization']
    click.echo(f"\n🎯 Optimization Method: {opt['method'].upper()}")
    click.echo(f"📊 Expected Return: {opt['expected_return']:.2%}")
    click.echo(f"📉 Expected Volatility: {opt['expected_volatility']:.2%}")
    click.echo(f"📈 Sharpe Ratio: {opt['sharpe_ratio']:.2f}")
    
    click.echo("\n💼 Portfolio Weights:")
    for asset, weight in opt['weights'].items():
        click.echo(f"  {asset}: {weight:.1%}")
    
    # Regime analysis
    regime = result['regime_analysis']
    click.echo(f"\n🧠 Regime Analysis ({regime['n_regimes']} regimes):")
    
    for i, (regime_idx, freq) in enumerate(zip(regime['regime_names'].keys(), regime['regime_distribution'])):
        regime_name = regime['regime_names'][str(regime_idx)]
        click.echo(f"  {regime_name}: {freq:.1%}")
    
    # Monte Carlo results
    if 'monte_carlo' in result:
        mc = result['monte_carlo']
        click.echo(f"\n🎲 Monte Carlo Simulation:")
        click.echo(f"  Mean Final Value: {mc['statistics']['mean_final_value']:.3f}")
        click.echo(f"  Probability of Loss: {mc['statistics']['probability_of_loss']:.1%}")
        click.echo(f"  95% VaR: {abs(mc['risk_metrics']['var_5%']):.2%}")
        click.echo(f"  95% CVaR: {abs(mc['risk_metrics']['cvar_5%']):.2%}")
        click.echo(f"  Execution Time: {mc['execution_time']:.1f}s")


@main.command()
@click.option('--symbols', '-s', type=str, required=True, help='Comma-separated list of symbols')
@click.option('--start-date', type=str, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=str, help='End date (YYYY-MM-DD)')
@click.option('--initial-capital', type=float, default=1000000, help='Initial capital')
@click.option('--rebalance-freq', type=str, default='M', help='Rebalancing frequency')
@click.option('--benchmark', type=str, default='SPY', help='Benchmark symbol')
@click.option('--output', '-o', type=str, help='Output directory for results')
def backtest(symbols: str, start_date: Optional[str], end_date: Optional[str],
             initial_capital: float, rebalance_freq: str, benchmark: str,
             output: Optional[str]):
    """
    Run comprehensive backtesting with regime detection
    
    Example:
        portfolio backtest -s "AAPL,GOOGL,MSFT" --initial-capital 1000000
    """
    try:
        click.echo("📊 Starting Portfolio Backtesting...")
        
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Set date range
        if not end_date:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if not start_date:
            start_date = end_date - timedelta(days=756)  # ~3 years
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Run backtest
        result = asyncio.run(_run_backtest(
            symbol_list, start_date, end_date, initial_capital,
            rebalance_freq, benchmark
        ))
        
        # Save results
        if output:
            os.makedirs(output, exist_ok=True)
            
            # Save performance metrics
            with open(os.path.join(output, 'backtest_results.json'), 'w') as f:
                json.dump(result['metrics'], f, indent=2, default=str)
            
            # Save returns
            if 'returns' in result:
                result['returns'].to_csv(os.path.join(output, 'returns.csv'))
            
            # Save weights
            if 'weights' in result:
                result['weights'].to_csv(os.path.join(output, 'weights.csv'))
            
            click.echo(f"💾 Results saved to {output}/")
        
        _display_backtest_results(result)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


async def _run_backtest(symbol_list: List[str], start_date: datetime, 
                       end_date: datetime, initial_capital: float,
                       rebalance_freq: str, benchmark: str) -> Dict[str, Any]:
    """Run backtesting process"""
    
    data_pipeline = cli_instance.data_pipeline
    
    # Fetch data
    click.echo("📥 Fetching data for backtesting...")
    price_data = await data_pipeline.fetch_asset_universe(symbol_list, start_date, end_date)
    features_data = await data_pipeline.create_features_dataset(symbol_list, start_date, end_date)
    
    # Get benchmark data
    benchmark_data = await data_pipeline.fetch_asset_universe([benchmark], start_date, end_date)
    
    # Prepare data
    adj_close = price_data.xs('Adj Close', level=0, axis=1)
    returns = adj_close.pct_change().dropna()
    
    benchmark_returns = benchmark_data.xs('Adj Close', level=0, axis=1).iloc[:, 0].pct_change().dropna()
    
    # Setup backtesting
    click.echo("🔄 Setting up backtesting engine...")
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_freq,
        benchmark_symbol=benchmark,
        lookback_window=126
    )
    
    # Initialize HMM for regime detection
    hmm_model = RegimeDetectionHMM(n_components=3, random_state=42)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizationEngine()
    
    # Run backtest
    click.echo("⚡ Running backtest...")
    backtesting_engine = BacktestingEngine(backtest_config, optimizer, hmm_model)
    
    backtest_results = backtesting_engine.run_backtest(returns, benchmark_returns, features_data)
    
    return {
        'metrics': backtest_results.performance_metrics,
        'returns': pd.DataFrame({
            'portfolio': backtest_results.portfolio_returns,
            'benchmark': backtest_results.benchmark_returns
        }),
        'weights': backtest_results.portfolio_weights,
        'config': {
            'symbols': symbol_list,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'rebalance_frequency': rebalance_freq,
            'benchmark': benchmark
        }
    }


def _display_backtest_results(result: Dict[str, Any]):
    """Display backtest results"""
    
    click.echo("\n" + "="*60)
    click.echo("📊 BACKTESTING RESULTS")
    click.echo("="*60)
    
    metrics = result['metrics']
    config = result['config']
    
    click.echo(f"\n📅 Period: {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}")
    click.echo(f"💰 Initial Capital: ${config['initial_capital']:,.0f}")
    click.echo(f"📊 Benchmark: {config['benchmark']}")
    
    click.echo(f"\n📈 Performance Metrics:")
    click.echo(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    click.echo(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    click.echo(f"  Volatility: {metrics.get('volatility', 0):.2%}")
    click.echo(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    click.echo(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    if 'alpha' in metrics:
        click.echo(f"  Alpha: {metrics['alpha']:.2%}")
        click.echo(f"  Beta: {metrics.get('beta', 0):.2f}")
        click.echo(f"  Information Ratio: {metrics.get('information_ratio', 0):.2f}")


@main.command()
@click.option('--symbols', '-s', type=str, required=True, help='Comma-separated list of symbols')
@click.option('--start-date', type=str, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=str, help='End date (YYYY-MM-DD)')
@click.option('--regimes', '-r', type=int, default=3, help='Number of regimes')
def analyze_regimes(symbols: str, start_date: Optional[str], end_date: Optional[str], regimes: int):
    """
    Analyze market regimes using HMM
    
    Example:
        portfolio analyze-regimes -s "SPY,QQQ,IWM" -r 4
    """
    try:
        click.echo("🧠 Analyzing Market Regimes...")
        
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Set date range
        if not end_date:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if not start_date:
            start_date = end_date - timedelta(days=756)
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Run regime analysis
        result = asyncio.run(_analyze_regimes(symbol_list, start_date, end_date, regimes))
        
        _display_regime_analysis(result)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


async def _analyze_regimes(symbol_list: List[str], start_date: datetime, 
                          end_date: datetime, regimes: int) -> Dict[str, Any]:
    """Run regime analysis"""
    
    data_pipeline = cli_instance.data_pipeline
    
    # Fetch data
    click.echo("📥 Fetching market data...")
    price_data = await data_pipeline.fetch_asset_universe(symbol_list, start_date, end_date)
    features_data = await data_pipeline.create_features_dataset(symbol_list, start_date, end_date)
    
    # Train HMM
    click.echo(f"🔍 Training HMM with {regimes} regimes...")
    hmm_model = AdvancedBaumWelchHMM(n_components=regimes, random_state=42)
    hmm_model.fit(features_data)
    
    # Get predictions
    regime_sequence = hmm_model.predict_regimes(features_data)
    regime_probabilities = hmm_model.predict_regime_probabilities(features_data)
    
    # Get statistics
    stats = hmm_model.get_regime_statistics(features_data)
    
    # Get summary
    summary = hmm_model.get_regime_summary(features_data)
    
    return {
        'regime_sequence': regime_sequence,
        'regime_probabilities': regime_probabilities,
        'statistics': stats,
        'summary': summary,
        'model_parameters': hmm_model.get_parameters().__dict__
    }


def _display_regime_analysis(result: Dict[str, Any]):
    """Display regime analysis results"""
    
    click.echo("\n" + "="*60)
    click.echo("🧠 REGIME ANALYSIS RESULTS")
    click.echo("="*60)
    
    stats = result['statistics']
    
    click.echo(f"\n📊 Regime Distribution:")
    for i, (regime_name, frequency) in enumerate(zip(stats['regime_names'].values(), stats['regime_distribution'])):
        click.echo(f"  {regime_name}: {frequency:.1%}")
    
    click.echo(f"\n⏱️  Average Regime Duration (periods):")
    for regime, duration in stats['average_regime_duration'].items():
        regime_name = stats['regime_names'][regime]
        click.echo(f"  {regime_name}: {duration:.1f}")
    
    click.echo(f"\n🔄 Transition Matrix:")
    transition_matrix = stats['transition_probabilities']
    
    # Print header
    regime_names = list(stats['regime_names'].values())
    header = "     " + "  ".join(f"{name[:8]:>8}" for name in regime_names)
    click.echo(header)
    
    # Print matrix
    for i, from_regime in enumerate(regime_names):
        row_values = "  ".join(f"{transition_matrix[i, j]:8.2%}" for j in range(len(regime_names)))
        click.echo(f"{from_regime[:8]:<8} {row_values}")


@main.command()
@click.option('--symbols', '-s', type=str, required=True, help='Comma-separated list of symbols')
@click.option('--weights', '-w', type=str, help='Comma-separated portfolio weights (default: equal weight)')
@click.option('--simulations', '-n', type=int, default=10000, help='Number of simulations')
@click.option('--horizon', '-h', type=int, default=252, help='Time horizon in days')
@click.option('--scenarios', is_flag=True, help='Run scenario analysis')
def simulate(symbols: str, weights: Optional[str], simulations: int, horizon: int, scenarios: bool):
    """
    Run Monte Carlo portfolio simulation
    
    Example:
        portfolio simulate -s "AAPL,GOOGL,MSFT" -w "0.4,0.3,0.3" -n 50000 --scenarios
    """
    try:
        click.echo("🎲 Running Monte Carlo Simulation...")
        
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Parse weights
        if weights:
            weight_list = [float(w.strip()) for w in weights.split(',')]
            if len(weight_list) != len(symbol_list):
                raise ValueError("Number of weights must match number of symbols")
            if abs(sum(weight_list) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")
        else:
            weight_list = [1.0 / len(symbol_list)] * len(symbol_list)
        
        # Run simulation
        result = asyncio.run(_run_simulation(
            symbol_list, weight_list, simulations, horizon, scenarios
        ))
        
        _display_simulation_results(result)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


async def _run_simulation(symbol_list: List[str], weight_list: List[float],
                         simulations: int, horizon: int, scenarios: bool) -> Dict[str, Any]:
    """Run Monte Carlo simulation"""
    
    data_pipeline = cli_instance.data_pipeline
    
    # Fetch recent data for parameter estimation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=504)  # ~2 years
    
    click.echo("📥 Fetching data for simulation...")
    price_data = await data_pipeline.fetch_asset_universe(symbol_list, start_date, end_date)
    
    # Prepare returns
    adj_close = price_data.xs('Adj Close', level=0, axis=1)
    returns = adj_close.pct_change().dropna()
    
    expected_returns = returns.mean() * 252
    covariance_matrix = returns.cov() * 252
    
    # Setup simulation
    config = SimulationConfig(
        n_simulations=simulations,
        time_horizon=horizon,
        random_seed=42
    )
    
    mc_engine = MonteCarloEngine(config)
    
    # Run base simulation
    click.echo(f"🎯 Running {simulations:,} simulations...")
    base_result = mc_engine.simulate_portfolio_paths(
        np.array(weight_list),
        expected_returns.values,
        covariance_matrix.values
    )
    
    result = {
        'base_simulation': {
            'statistics': base_result.statistics,
            'risk_metrics': base_result.risk_metrics,
            'confidence_intervals': base_result.confidence_intervals,
            'execution_time': base_result.execution_time
        },
        'portfolio_weights': dict(zip(symbol_list, weight_list))
    }
    
    # Run scenario analysis
    if scenarios:
        click.echo("📊 Running scenario analysis...")
        scenario_configs = create_default_scenarios()
        
        scenario_results = mc_engine.scenario_analysis(
            np.array(weight_list),
            expected_returns.values,
            covariance_matrix.values,
            scenario_configs
        )
        
        result['scenarios'] = {
            name: {
                'statistics': sr.statistics,
                'risk_metrics': sr.risk_metrics
            }
            for name, sr in scenario_results.items()
        }
    
    return result


def _display_simulation_results(result: Dict[str, Any]):
    """Display simulation results"""
    
    click.echo("\n" + "="*60)
    click.echo("🎲 MONTE CARLO SIMULATION RESULTS")
    click.echo("="*60)
    
    # Portfolio weights
    click.echo("\n💼 Portfolio Weights:")
    for symbol, weight in result['portfolio_weights'].items():
        click.echo(f"  {symbol}: {weight:.1%}")
    
    # Base simulation results
    base = result['base_simulation']
    stats = base['statistics']
    risk = base['risk_metrics']
    
    click.echo(f"\n📊 Simulation Statistics:")
    click.echo(f"  Expected Final Value: {stats['mean_final_value']:.3f}")
    click.echo(f"  Annualized Return: {stats['annualized_return']:.2%}")
    click.echo(f"  Annualized Volatility: {stats['annualized_volatility']:.2%}")
    click.echo(f"  Probability of Loss: {stats['probability_of_loss']:.1%}")
    click.echo(f"  Expected Profit: {stats['expected_profit']:.3f}")
    
    click.echo(f"\n⚠️  Risk Metrics:")
    click.echo(f"  95% VaR: {abs(risk['var_5%']):.2%}")
    click.echo(f"  95% CVaR: {abs(risk['cvar_5%']):.2%}")
    click.echo(f"  Maximum Drawdown: {abs(risk['maximum_drawdown']):.2%}")
    
    # Scenario results
    if 'scenarios' in result:
        click.echo(f"\n🎭 Scenario Analysis:")
        for scenario_name, scenario_result in result['scenarios'].items():
            scenario_stats = scenario_result['statistics']
            click.echo(f"  {scenario_name}:")
            click.echo(f"    Final Value: {scenario_stats['mean_final_value']:.3f}")
            click.echo(f"    Loss Probability: {scenario_stats['probability_of_loss']:.1%}")


@main.command()
@click.option('--config-file', '-c', type=str, help='Configuration file to create')
@click.option('--template', '-t', type=click.Choice(['basic', 'advanced', 'production']),
              default='basic', help='Configuration template')
def init_config(config_file: Optional[str], template: str):
    """
    Initialize configuration file
    
    Example:
        portfolio init-config -c my_config.yaml -t advanced
    """
    try:
        if not config_file:
            config_file = f"portfolio_config_{template}.yaml"
        
        config = _create_config_template(template)
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, indent=2, default_flow_style=False)
        
        click.echo(f"✅ Configuration file created: {config_file}")
        click.echo(f"📝 Template: {template}")
        click.echo(f"🔧 Edit the file to customize your settings")
        
    except Exception as e:
        click.echo(f"❌ Error creating config: {e}", err=True)
        sys.exit(1)


def _create_config_template(template: str) -> Dict[str, Any]:
    """Create configuration template"""
    
    base_config = {
        'data': {
            'sources': {
                'yahoo': True,
                'alpha_vantage': False
            },
            'refresh_frequency': '1H',
            'lookback_days': 252
        },
        'hmm': {
            'n_components': 3,
            'covariance_type': 'full',
            'n_iter': 100,
            'random_state': 42,
            'features': ['returns', 'volatility', 'volume']
        },
        'portfolio': {
            'optimization': {
                'method': 'mean_variance',
                'rebalance_frequency': '1W',
                'transaction_costs': 0.001
            },
            'constraints': {
                'max_weight': 0.2,
                'min_weight': 0.0,
                'leverage': 1.0
            },
            'risk': {
                'target_volatility': 0.15,
                'max_drawdown': 0.2,
                'var_confidence': 0.05
            }
        },
        'backtesting': {
            'initial_capital': 1000000,
            'benchmark': 'SPY'
        },
        'logging': {
            'level': 'INFO',
            'file': 'portfolio_engine.log'
        }
    }
    
    if template == 'advanced':
        base_config['monte_carlo'] = {
            'n_simulations': 10000,
            'time_horizon': 252,
            'confidence_levels': [0.01, 0.05, 0.10],
            'parallel_processing': True
        }
        
        base_config['factor_models'] = {
            'enabled': ['fama_french_3', 'statistical'],
            'statistical': {
                'n_factors': 5,
                'method': 'pca'
            }
        }
    
    elif template == 'production':
        base_config.update({
            'database': {
                'type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'database': 'portfolio_db'
            },
            'alerts': {
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'port': 587
                },
                'thresholds': {
                    'max_drawdown_alert': 0.10,
                    'var_breach_alert': 0.05
                }
            }
        })
    
    return base_config


@main.command()
def version():
    """Show version information"""
    click.echo("🚀 Stochastic Portfolio Engine v1.0.0")
    click.echo("📊 Hidden Markov Model Regime Detection")
    click.echo("🎯 Advanced Portfolio Optimization")
    click.echo("📈 Comprehensive Risk Management")


if __name__ == '__main__':
    main()