#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from src.backtesting.execution.trading_simulator import (
    RealisticTradingSimulator, 
    create_trading_simulator, 
    OrderSide, 
    OrderType
)
from src.backtesting.engine.backtesting_engine import BacktestConfig, BacktestResults
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.models.hmm.hmm_engine import AdvancedBaumWelchHMM
from src.data.ingestion.enhanced_data_pipeline import EnhancedDataPipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class BacktestMode(Enum):
    """Backtesting modes"""
    VECTORIZED = "vectorized"  # Fast vectorized calculation
    EVENT_DRIVEN = "event_driven"  # Realistic event-driven simulation
    WALK_FORWARD = "walk_forward"  # Walk-forward analysis
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation


class RebalanceFrequency(Enum):
    """Rebalancing frequencies"""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M" 
    QUARTERLY = "Q"
    SEMI_ANNUAL = "6M"
    ANNUAL = "A"


@dataclass
class AdvancedBacktestConfig:
    """Advanced backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    
    # Rebalancing
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    rebalance_threshold: float = 0.05  # Rebalance if weight deviation > 5%
    
    # Trading costs
    commission_rate: float = 0.001
    bid_ask_spread: float = 0.0005
    market_impact_model: str = "advanced"
    
    # Portfolio construction
    optimization_method: str = "mean_variance"
    lookback_window: int = 252
    min_history_required: int = 60
    max_weight: float = 0.25
    min_weight: float = 0.0
    
    # Risk management
    stop_loss_threshold: Optional[float] = None  # e.g., -0.10 for 10% stop loss
    take_profit_threshold: Optional[float] = None  # e.g., 0.20 for 20% take profit
    max_drawdown_threshold: Optional[float] = None  # e.g., -0.15 for 15% max drawdown
    
    # Regime detection
    use_regime_detection: bool = True
    regime_components: int = 3
    regime_lookback: int = 504  # 2 years
    
    # Walk-forward analysis
    training_window: int = 252  # 1 year training
    test_window: int = 63  # 3 months testing
    step_size: int = 21  # 1 month step
    
    # Monte Carlo
    n_simulations: int = 1000
    bootstrap_method: str = "circular"  # "circular" or "stationary"
    
    # Performance
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
    
    # Execution
    execution_delay: int = 1  # Days delay between signal and execution
    partial_fill_probability: float = 0.0  # Probability of partial fills
    
    # Other
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all available cores
    save_intermediate_results: bool = True
    verbose: bool = True


@dataclass
class AdvancedBacktestResults:
    """Advanced backtesting results"""
    # Core results
    portfolio_returns: pd.Series
    portfolio_weights: pd.DataFrame
    portfolio_values: pd.Series
    benchmark_returns: pd.Series
    
    # Trading details
    trades: pd.DataFrame
    positions: pd.DataFrame
    trading_costs: Dict[str, float]
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    drawdown_analysis: Dict[str, Any]
    
    # Regime analysis
    regime_history: Optional[pd.DataFrame] = None
    regime_performance: Optional[Dict[str, Any]] = None
    
    # Walk-forward results
    walk_forward_results: Optional[List[Dict[str, Any]]] = None
    
    # Monte Carlo results
    monte_carlo_results: Optional[Dict[str, Any]] = None
    
    # Metadata
    config: AdvancedBacktestConfig = None
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)


class AdvancedBacktestingFramework:
    """Advanced backtesting framework with realistic trading simulation"""
    
    def __init__(self, config: AdvancedBacktestConfig, data_pipeline: Optional[EnhancedDataPipeline] = None):
        self.config = config
        self.data_pipeline = data_pipeline
        
        # Initialize components
        self.optimizer = PortfolioOptimizationEngine()
        self.trading_simulator = create_trading_simulator(
            commission_rate=config.commission_rate,
            bid_ask_spread=config.bid_ask_spread,
            market_impact_model=config.market_impact_model
        )
        
        # HMM for regime detection
        self.hmm_model = None
        if config.use_regime_detection:
            self.hmm_model = AdvancedBaumWelchHMM(
                n_components=config.regime_components,
                random_state=42
            )
        
        # Data storage
        self.price_data = None
        self.returns_data = None
        self.features_data = None
        self.benchmark_data = None
        
        # Results tracking
        self.results_history = []
    
    def run_backtest(self, mode: BacktestMode = BacktestMode.EVENT_DRIVEN) -> AdvancedBacktestResults:
        """Run advanced backtest with specified mode"""
        
        logger.info(f"Starting advanced backtest in {mode.value} mode")
        start_time = datetime.now()
        
        try:
            # Load and prepare data
            self._prepare_data()
            
            # Run backtest based on mode
            if mode == BacktestMode.VECTORIZED:
                results = self._run_vectorized_backtest()
            elif mode == BacktestMode.EVENT_DRIVEN:
                results = self._run_event_driven_backtest()
            elif mode == BacktestMode.WALK_FORWARD:
                results = self._run_walk_forward_backtest()
            elif mode == BacktestMode.MONTE_CARLO:
                results = self._run_monte_carlo_backtest()
            else:
                raise ValueError(f"Unknown backtest mode: {mode}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            results.execution_time = execution_time
            results.config = self.config
            
            logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _prepare_data(self):
        """Load and prepare data for backtesting"""
        
        logger.info("Preparing data for backtesting")
        
        # Extend date range to include lookback period
        extended_start = self.config.start_date - timedelta(days=self.config.lookback_window + 50)
        
        if self.data_pipeline:
            # Use enhanced data pipeline
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Get price data
                price_data, _ = loop.run_until_complete(
                    self.data_pipeline.fetch_enhanced_asset_data(
                        symbols=self.config.symbols,
                        start_date=extended_start,
                        end_date=self.config.end_date,
                        include_quality_metrics=False
                    )
                )
                
                # Get benchmark data
                benchmark_data, _ = loop.run_until_complete(
                    self.data_pipeline.fetch_enhanced_asset_data(
                        symbols=[self.config.benchmark_symbol],
                        start_date=extended_start,
                        end_date=self.config.end_date,
                        include_quality_metrics=False
                    )
                )
                
                # Get features data
                features_data = loop.run_until_complete(
                    self.data_pipeline.fetch_enhanced_features_dataset(
                        symbols=self.config.symbols,
                        start_date=extended_start,
                        end_date=self.config.end_date
                    )
                )
                
            finally:
                loop.close()
        else:
            # Use basic data loading (placeholder - would implement yfinance loading)
            import yfinance as yf
            
            # Download price data
            tickers = self.config.symbols + [self.config.benchmark_symbol]
            price_data = yf.download(tickers, start=extended_start, end=self.config.end_date, auto_adjust=True)
            benchmark_data = price_data[[self.config.benchmark_symbol]]
            price_data = price_data[self.config.symbols]
            
            # Create simple features
            returns = price_data.pct_change().dropna()
            features_data = pd.DataFrame({
                'market_return': returns.mean(axis=1),
                'market_volatility': returns.rolling(20).std().mean(axis=1),
                'momentum': returns.rolling(10).mean().mean(axis=1)
            }).dropna()
        
        # Store data
        self.price_data = price_data
        self.returns_data = price_data.pct_change().dropna()
        self.features_data = features_data
        self.benchmark_data = benchmark_data
        
        # Extract benchmark returns
        if isinstance(benchmark_data.columns, pd.MultiIndex):
            benchmark_prices = benchmark_data.xs('Close', level=0, axis=1).iloc[:, 0]
        else:
            benchmark_prices = benchmark_data.iloc[:, 0]
            
        self.benchmark_returns = benchmark_prices.pct_change().dropna()
        
        logger.info(f"Data prepared: {len(self.price_data)} days, {len(self.config.symbols)} symbols")
    
    def _run_event_driven_backtest(self) -> AdvancedBacktestResults:
        """Run realistic event-driven backtest"""
        
        logger.info("Running event-driven backtest with realistic trading simulation")
        
        # Initialize simulator
        self.trading_simulator.reset(self.config.initial_capital)
        
        # Create rebalance dates
        rebalance_dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.rebalance_frequency.value
        )
        
        # Results tracking
        portfolio_values = []
        portfolio_returns = []
        portfolio_weights_history = []
        regime_history = []
        trade_details = []
        
        current_weights = None
        last_portfolio_value = self.config.initial_capital
        
        # Daily simulation loop
        simulation_dates = pd.date_range(self.config.start_date, self.config.end_date, freq='D')
        
        for current_date in simulation_dates:
            if current_date not in self.price_data.index:
                continue
            
            # Prepare market data for simulator
            market_data = {}
            for symbol in self.config.symbols:
                if symbol in self.price_data.columns:
                    symbol_data = {
                        'close': self.price_data.loc[current_date, symbol],
                        'price': self.price_data.loc[current_date, symbol],
                        'volume': 1000000,  # Default volume
                        'volatility': self.returns_data[symbol].rolling(20).std().loc[current_date] if current_date in self.returns_data.index else 0.02
                    }
                    market_data[symbol] = symbol_data
            
            # Process any pending orders
            fills = self.trading_simulator.process_orders(market_data, current_date)
            if fills:
                trade_details.extend([{
                    'date': fill.timestamp,
                    'symbol': fill.symbol,
                    'side': fill.side.value,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'commission': fill.commission,
                    'slippage': fill.slippage,
                    'market_impact': fill.market_impact
                } for fill in fills])
            
            # Check if it's a rebalance date
            if current_date in rebalance_dates:
                new_weights = self._generate_portfolio_weights(current_date)
                if new_weights is not None:
                    # Execute rebalancing trades
                    self._execute_rebalancing(new_weights, market_data, current_date)
                    current_weights = new_weights
            
            # Calculate portfolio value and return
            portfolio_value = self.trading_simulator.get_portfolio_value(market_data)
            portfolio_return = (portfolio_value / last_portfolio_value - 1) if last_portfolio_value > 0 else 0
            
            # Get current weights
            current_position_weights = self.trading_simulator.get_portfolio_weights(market_data)
            
            # Store results
            portfolio_values.append(portfolio_value)
            portfolio_returns.append(portfolio_return)
            portfolio_weights_history.append(current_position_weights.to_dict() if not current_position_weights.empty else {})
            
            # Regime detection
            if self.hmm_model and current_date in self.features_data.index:
                try:
                    lookback_features = self.features_data.loc[:current_date].tail(self.config.regime_lookback)
                    if len(lookback_features) >= 30:
                        if not self.hmm_model.is_fitted:
                            self.hmm_model.fit(lookback_features)
                        
                        latest_features = lookback_features.tail(1)
                        regime = self.hmm_model.predict_regimes(latest_features)[0]
                        regime_probs = self.hmm_model.predict_regime_probabilities(latest_features)[0]
                    else:
                        regime = None
                        regime_probs = None
                except:
                    regime = None
                    regime_probs = None
            else:
                regime = None
                regime_probs = None
            
            regime_history.append({
                'date': current_date,
                'regime': regime,
                'regime_probabilities': regime_probs
            })
            
            last_portfolio_value = portfolio_value
        
        # Compile results
        results_index = pd.DatetimeIndex([entry['date'] for entry in regime_history])
        
        return AdvancedBacktestResults(
            portfolio_returns=pd.Series(portfolio_returns, index=results_index),
            portfolio_weights=pd.DataFrame(portfolio_weights_history, index=results_index),
            portfolio_values=pd.Series(portfolio_values, index=results_index),
            benchmark_returns=self.benchmark_returns.reindex(results_index).fillna(0),
            trades=pd.DataFrame(trade_details),
            positions=self.trading_simulator.get_positions_summary(),
            trading_costs=self.trading_simulator.get_trading_statistics(),
            performance_metrics=self._calculate_performance_metrics(
                pd.Series(portfolio_returns, index=results_index),
                self.benchmark_returns.reindex(results_index).fillna(0)
            ),
            risk_metrics=self._calculate_risk_metrics(pd.Series(portfolio_returns, index=results_index)),
            drawdown_analysis=self._analyze_drawdowns(pd.Series(portfolio_values, index=results_index)),
            regime_history=pd.DataFrame(regime_history).set_index('date') if regime_history else None
        )
    
    def _run_vectorized_backtest(self) -> AdvancedBacktestResults:
        """Run fast vectorized backtest (less realistic but faster)"""
        
        logger.info("Running vectorized backtest")
        
        # Create rebalance dates
        rebalance_dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.rebalance_frequency.value
        )
        
        # Initialize tracking variables
        portfolio_weights_history = []
        portfolio_returns = []
        dates = []
        
        current_weights = None
        
        for rebalance_date in rebalance_dates:
            if rebalance_date not in self.returns_data.index:
                continue
            
            # Generate new weights
            new_weights = self._generate_portfolio_weights(rebalance_date)
            if new_weights is None:
                continue
            
            current_weights = pd.Series(new_weights, index=self.config.symbols)
            
            # Calculate period returns until next rebalance
            next_rebalance_idx = rebalance_dates.get_loc(rebalance_date) + 1
            if next_rebalance_idx < len(rebalance_dates):
                period_end = rebalance_dates[next_rebalance_idx]
            else:
                period_end = self.config.end_date
            
            period_returns = self.returns_data.loc[rebalance_date:period_end]
            
            if len(period_returns) > 0:
                # Calculate portfolio returns for the period
                period_portfolio_returns = (period_returns * current_weights).sum(axis=1)
                
                # Apply transaction costs (simplified)
                if len(portfolio_returns) > 0:  # Not first rebalance
                    transaction_cost = sum(abs(new_weights[symbol] - (portfolio_weights_history[-1].get(symbol, 0) if portfolio_weights_history else 0)) 
                                         for symbol in self.config.symbols) * self.config.commission_rate
                    period_portfolio_returns.iloc[0] -= transaction_cost
                
                portfolio_returns.extend(period_portfolio_returns.tolist())
                dates.extend(period_returns.index.tolist())
                
                # Store weights for this period
                weights_dict = dict(current_weights)
                for date in period_returns.index:
                    portfolio_weights_history.append(weights_dict)
        
        # Create series
        portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
        portfolio_weights_df = pd.DataFrame(portfolio_weights_history, index=dates)
        
        # Calculate portfolio values
        portfolio_values = (1 + portfolio_returns_series).cumprod() * self.config.initial_capital
        
        # Get benchmark returns
        benchmark_returns_aligned = self.benchmark_returns.reindex(portfolio_returns_series.index).fillna(0)
        
        return AdvancedBacktestResults(
            portfolio_returns=portfolio_returns_series,
            portfolio_weights=portfolio_weights_df,
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_returns_aligned,
            trades=pd.DataFrame(),  # No individual trades in vectorized mode
            positions=pd.DataFrame(),
            trading_costs={'total_commission': len(portfolio_weights_history) * self.config.commission_rate * 0.01},
            performance_metrics=self._calculate_performance_metrics(portfolio_returns_series, benchmark_returns_aligned),
            risk_metrics=self._calculate_risk_metrics(portfolio_returns_series),
            drawdown_analysis=self._analyze_drawdowns(portfolio_values)
        )
    
    def _run_walk_forward_backtest(self) -> AdvancedBacktestResults:
        """Run walk-forward analysis"""
        
        logger.info("Running walk-forward analysis")
        
        walk_forward_results = []
        
        # Create walk-forward windows
        start_date = self.config.start_date
        
        while start_date < self.config.end_date:
            # Define training and test periods
            training_start = start_date - timedelta(days=self.config.training_window)
            training_end = start_date
            test_start = start_date
            test_end = min(start_date + timedelta(days=self.config.test_window), self.config.end_date)
            
            if training_start < self.returns_data.index[0] or test_end > self.returns_data.index[-1]:
                start_date += timedelta(days=self.config.step_size)
                continue
            
            logger.info(f"Walk-forward window: train {training_start.date()} to {training_end.date()}, "
                       f"test {test_start.date()} to {test_end.date()}")
            
            # Create temporary config for this window
            temp_config = AdvancedBacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=self.config.initial_capital,
                symbols=self.config.symbols,
                **{k: v for k, v in self.config.__dict__.items() 
                   if k not in ['start_date', 'end_date']}
            )
            
            # Run backtest for this window
            temp_framework = AdvancedBacktestingFramework(temp_config, self.data_pipeline)
            temp_framework.price_data = self.price_data
            temp_framework.returns_data = self.returns_data
            temp_framework.features_data = self.features_data
            temp_framework.benchmark_returns = self.benchmark_returns
            
            try:
                window_results = temp_framework._run_vectorized_backtest()
                
                walk_forward_results.append({
                    'training_start': training_start,
                    'training_end': training_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'performance': window_results.performance_metrics,
                    'returns': window_results.portfolio_returns
                })
                
            except Exception as e:
                logger.warning(f"Walk-forward window failed: {e}")
            
            start_date += timedelta(days=self.config.step_size)
        
        # Combine all walk-forward results
        all_returns = []
        all_dates = []
        all_weights = []
        
        for result in walk_forward_results:
            if 'returns' in result:
                all_returns.extend(result['returns'].tolist())
                all_dates.extend(result['returns'].index.tolist())
                # Use equal weights for simplicity in walk-forward
                equal_weights = {symbol: 1.0/len(self.config.symbols) for symbol in self.config.symbols}
                all_weights.extend([equal_weights] * len(result['returns']))
        
        if all_returns:
            combined_returns = pd.Series(all_returns, index=all_dates)
            combined_weights = pd.DataFrame(all_weights, index=all_dates)
            combined_values = (1 + combined_returns).cumprod() * self.config.initial_capital
            benchmark_aligned = self.benchmark_returns.reindex(combined_returns.index).fillna(0)
            
            return AdvancedBacktestResults(
                portfolio_returns=combined_returns,
                portfolio_weights=combined_weights,
                portfolio_values=combined_values,
                benchmark_returns=benchmark_aligned,
                trades=pd.DataFrame(),
                positions=pd.DataFrame(),
                trading_costs={},
                performance_metrics=self._calculate_performance_metrics(combined_returns, benchmark_aligned),
                risk_metrics=self._calculate_risk_metrics(combined_returns),
                drawdown_analysis=self._analyze_drawdowns(combined_values),
                walk_forward_results=walk_forward_results
            )
        else:
            # Return empty results if no successful windows
            return AdvancedBacktestResults(
                portfolio_returns=pd.Series(dtype=float),
                portfolio_weights=pd.DataFrame(),
                portfolio_values=pd.Series(dtype=float),
                benchmark_returns=pd.Series(dtype=float),
                trades=pd.DataFrame(),
                positions=pd.DataFrame(),
                trading_costs={},
                performance_metrics={},
                risk_metrics={},
                drawdown_analysis={},
                walk_forward_results=walk_forward_results
            )
    
    def _run_monte_carlo_backtest(self) -> AdvancedBacktestResults:
        """Run Monte Carlo simulation backtest"""
        
        logger.info(f"Running Monte Carlo backtest with {self.config.n_simulations} simulations")
        
        # Get historical returns for bootstrap
        historical_returns = self.returns_data.dropna()
        
        monte_carlo_results = []
        
        # Run multiple simulations
        if self.config.parallel_processing and self.config.n_jobs != 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), self.config.n_simulations)) as executor:
                futures = [
                    executor.submit(self._run_single_monte_carlo_simulation, historical_returns, sim_id)
                    for sim_id in range(self.config.n_simulations)
                ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            monte_carlo_results.append(result)
                    except Exception as e:
                        logger.warning(f"Monte Carlo simulation failed: {e}")
        else:
            # Sequential execution
            for sim_id in range(self.config.n_simulations):
                try:
                    result = self._run_single_monte_carlo_simulation(historical_returns, sim_id)
                    if result is not None:
                        monte_carlo_results.append(result)
                except Exception as e:
                    logger.warning(f"Monte Carlo simulation {sim_id} failed: {e}")
        
        if not monte_carlo_results:
            logger.error("All Monte Carlo simulations failed")
            return AdvancedBacktestResults(
                portfolio_returns=pd.Series(dtype=float),
                portfolio_weights=pd.DataFrame(),
                portfolio_values=pd.Series(dtype=float),
                benchmark_returns=pd.Series(dtype=float),
                trades=pd.DataFrame(),
                positions=pd.DataFrame(),
                trading_costs={},
                performance_metrics={},
                risk_metrics={},
                drawdown_analysis={},
                monte_carlo_results={'simulations': []}
            )
        
        # Aggregate Monte Carlo results
        mc_summary = self._aggregate_monte_carlo_results(monte_carlo_results)
        
        # Use the median simulation as the main result
        median_idx = len(monte_carlo_results) // 2
        main_result = monte_carlo_results[median_idx]
        
        return AdvancedBacktestResults(
            portfolio_returns=main_result['returns'],
            portfolio_weights=main_result['weights'],
            portfolio_values=main_result['values'],
            benchmark_returns=main_result['benchmark_returns'],
            trades=pd.DataFrame(),
            positions=pd.DataFrame(),
            trading_costs={},
            performance_metrics=main_result['performance'],
            risk_metrics=main_result['risk_metrics'],
            drawdown_analysis=main_result['drawdown_analysis'],
            monte_carlo_results=mc_summary
        )
    
    def _run_single_monte_carlo_simulation(self, historical_returns: pd.DataFrame, sim_id: int) -> Optional[Dict[str, Any]]:
        """Run single Monte Carlo simulation"""
        
        try:
            # Bootstrap returns
            n_days = (self.config.end_date - self.config.start_date).days
            
            if self.config.bootstrap_method == "circular":
                # Circular block bootstrap
                block_size = 20  # 20-day blocks
                n_blocks = n_days // block_size + 1
                
                bootstrap_returns = []
                for _ in range(n_blocks):
                    start_idx = np.random.randint(0, len(historical_returns) - block_size)
                    block = historical_returns.iloc[start_idx:start_idx + block_size]
                    bootstrap_returns.append(block)
                
                simulated_returns = pd.concat(bootstrap_returns, ignore_index=True).iloc[:n_days]
            else:
                # Stationary bootstrap
                indices = np.random.choice(len(historical_returns), size=n_days, replace=True)
                simulated_returns = historical_returns.iloc[indices].reset_index(drop=True)
            
            # Create date index
            sim_dates = pd.date_range(self.config.start_date, periods=len(simulated_returns), freq='D')
            simulated_returns.index = sim_dates
            
            # Run backtest on simulated data
            temp_framework = AdvancedBacktestingFramework(self.config, None)
            temp_framework.returns_data = simulated_returns
            temp_framework.price_data = (1 + simulated_returns).cumprod() * 100  # Synthetic prices
            temp_framework.benchmark_returns = self.benchmark_returns.reindex(sim_dates).fillna(0)
            
            # Create simple features
            temp_framework.features_data = pd.DataFrame({
                'market_return': simulated_returns.mean(axis=1),
                'market_volatility': simulated_returns.rolling(20).std().mean(axis=1),
                'momentum': simulated_returns.rolling(10).mean().mean(axis=1)
            }).dropna()
            
            result = temp_framework._run_vectorized_backtest()
            
            return {
                'sim_id': sim_id,
                'returns': result.portfolio_returns,
                'weights': result.portfolio_weights,
                'values': result.portfolio_values,
                'benchmark_returns': result.benchmark_returns,
                'performance': result.performance_metrics,
                'risk_metrics': result.risk_metrics,
                'drawdown_analysis': result.drawdown_analysis
            }
            
        except Exception as e:
            logger.debug(f"Monte Carlo simulation {sim_id} failed: {e}")
            return None
    
    def _aggregate_monte_carlo_results(self, monte_carlo_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate Monte Carlo simulation results"""
        
        # Extract performance metrics
        performance_metrics = [result['performance'] for result in monte_carlo_results]
        
        aggregated = {
            'n_simulations': len(monte_carlo_results),
            'performance_distribution': {},
            'risk_distribution': {},
            'final_values': [],
            'max_drawdowns': [],
            'sharpe_ratios': []
        }
        
        # Aggregate key metrics
        for metric in ['total_return', 'annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']:
            values = [perf.get(metric, 0) for perf in performance_metrics]
            aggregated['performance_distribution'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentiles': {
                    '5%': np.percentile(values, 5),
                    '25%': np.percentile(values, 25),
                    '50%': np.percentile(values, 50),
                    '75%': np.percentile(values, 75),
                    '95%': np.percentile(values, 95)
                }
            }
        
        # Extract final values and other metrics
        for result in monte_carlo_results:
            if not result['values'].empty:
                aggregated['final_values'].append(result['values'].iloc[-1])
            aggregated['max_drawdowns'].append(result['performance'].get('max_drawdown', 0))
            aggregated['sharpe_ratios'].append(result['performance'].get('sharpe_ratio', 0))
        
        return aggregated
    
    def _generate_portfolio_weights(self, rebalance_date: datetime) -> Optional[Dict[str, float]]:
        """Generate portfolio weights for rebalancing"""
        
        # Get lookback data
        lookback_start = rebalance_date - timedelta(days=self.config.lookback_window)
        lookback_returns = self.returns_data.loc[lookback_start:rebalance_date].iloc[:-1]
        
        if len(lookback_returns) < self.config.min_history_required:
            logger.warning(f"Insufficient data for {rebalance_date}, skipping rebalance")
            return None
        
        # Calculate expected returns and covariance
        expected_returns = lookback_returns.mean() * 252
        covariance_matrix = lookback_returns.cov() * 252
        
        try:
            # Optimize portfolio
            optimization_result = self.optimizer.optimize_portfolio(
                method=self.config.optimization_method,
                expected_returns=expected_returns.values,
                covariance_matrix=covariance_matrix.values,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight
            )
            
            weights = dict(zip(self.config.symbols, optimization_result['weights']))
            return weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed for {rebalance_date}: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / len(self.config.symbols)
            return {symbol: equal_weight for symbol in self.config.symbols}
    
    def _execute_rebalancing(self, target_weights: Dict[str, float], market_data: Dict[str, Any], trade_date: datetime):
        """Execute rebalancing trades"""
        
        current_weights = self.trading_simulator.get_portfolio_weights(market_data)
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > self.config.rebalance_threshold:
                # Calculate target quantity
                portfolio_value = self.trading_simulator.get_portfolio_value(market_data)
                target_value = target_weight * portfolio_value
                current_price = market_data[symbol]['price']
                target_quantity = target_value / current_price
                
                current_position = self.trading_simulator.positions.get(symbol)
                current_quantity = current_position.quantity if current_position else 0.0
                
                quantity_diff = target_quantity - current_quantity
                
                if abs(quantity_diff) > 0.001:  # Minimum trade size
                    side = OrderSide.BUY if quantity_diff > 0 else OrderSide.SELL
                    self.trading_simulator.submit_order(
                        symbol=symbol,
                        side=side,
                        quantity=abs(quantity_diff),
                        order_type=OrderType.MARKET
                    )
    
    def _calculate_performance_metrics(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if len(portfolio_returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / rolling_max - 1)
        max_drawdown = drawdowns.min()
        
        # Benchmark comparison
        if len(benchmark_returns) > 0:
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
            alpha = annualized_return - benchmark_annualized
            
            # Beta calculation
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            beta = covariance / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
            
            # Information ratio
            excess_returns = portfolio_returns - benchmark_returns
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        else:
            alpha = annualized_return
            beta = 0
            information_ratio = 0
        
        # Additional metrics
        downside_returns = portfolio_returns[portfolio_returns < self.config.risk_free_rate / 252]
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win/Loss statistics
        win_rate = (portfolio_returns > 0).mean()
        avg_win = portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).any() else 0
        avg_loss = portfolio_returns[portfolio_returns < 0].mean() if (portfolio_returns < 0).any() else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def _calculate_risk_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        if len(portfolio_returns) == 0:
            return {}
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        
        # Maximum consecutive losses
        losses = (portfolio_returns < 0).astype(int)
        max_consecutive_losses = losses.groupby((losses != losses.shift()).cumsum()).sum().max()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _analyze_drawdowns(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """Analyze drawdown periods"""
        
        if len(portfolio_values) == 0:
            return {}
        
        cumulative = portfolio_values / portfolio_values.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative / rolling_max - 1)
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        current_period = {}
        
        for date, dd in drawdowns.items():
            if dd < -0.005 and not in_drawdown:  # Start of drawdown (0.5% threshold)
                in_drawdown = True
                current_period = {
                    'start_date': date,
                    'peak_value': portfolio_values[date],
                    'trough_value': portfolio_values[date],
                    'trough_date': date
                }
            elif in_drawdown:
                if portfolio_values[date] < current_period['trough_value']:
                    current_period['trough_value'] = portfolio_values[date]
                    current_period['trough_date'] = date
                
                if abs(dd) < 0.001:  # End of drawdown
                    current_period['end_date'] = date
                    current_period['recovery_value'] = portfolio_values[date]
                    current_period['max_drawdown'] = (current_period['trough_value'] / current_period['peak_value'] - 1)
                    current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
                    current_period['recovery_days'] = (current_period['end_date'] - current_period['trough_date']).days
                    
                    drawdown_periods.append(current_period)
                    in_drawdown = False
        
        if not drawdown_periods:
            return {'max_drawdown': drawdowns.min(), 'avg_drawdown_duration': 0, 'drawdown_periods': []}
        
        avg_duration = np.mean([period['duration_days'] for period in drawdown_periods])
        avg_recovery = np.mean([period.get('recovery_days', 0) for period in drawdown_periods])
        max_duration = max([period['duration_days'] for period in drawdown_periods])
        
        return {
            'max_drawdown': drawdowns.min(),
            'avg_drawdown_duration': avg_duration,
            'avg_recovery_time': avg_recovery,
            'max_drawdown_duration': max_duration,
            'num_drawdown_periods': len(drawdown_periods),
            'drawdown_periods': drawdown_periods
        }


def create_advanced_backtesting_framework(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 1000000,
    **kwargs
) -> AdvancedBacktestingFramework:
    """Factory function to create advanced backtesting framework"""
    
    config = AdvancedBacktestConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        **kwargs
    )
    
    framework = AdvancedBacktestingFramework(config)
    logger.info(f"Created advanced backtesting framework for {len(symbols)} symbols")
    
    return framework