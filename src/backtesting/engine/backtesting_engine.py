import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.models.hmm.hmm_engine import RegimeDetectionHMM
from src.optimization.objectives.risk_measures import PortfolioRiskCalculator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float
    rebalance_frequency: str  # 'D', 'W', 'M', 'Q'
    transaction_costs: float
    benchmark_symbol: str
    lookback_window: int
    optimization_method: str
    risk_target: Optional[float] = None
    max_weight: float = 0.2
    min_weight: float = 0.0


@dataclass
class BacktestResults:
    portfolio_returns: pd.Series
    portfolio_weights: pd.DataFrame
    portfolio_values: pd.Series
    benchmark_returns: pd.Series
    transactions: pd.DataFrame
    regime_history: pd.DataFrame
    performance_metrics: Dict[str, float]
    drawdown_periods: List[Dict[str, Any]]
    trade_analysis: Dict[str, Any]


class BacktestingEngine:
    def __init__(
        self,
        config: BacktestConfig,
        portfolio_optimizer: PortfolioOptimizationEngine,
        regime_detector: Optional[RegimeDetectionHMM] = None
    ):
        self.config = config
        self.portfolio_optimizer = portfolio_optimizer
        self.regime_detector = regime_detector
        self.risk_calculator = PortfolioRiskCalculator()
        
        self.current_weights = None
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}
        
        self.results = {
            'dates': [],
            'portfolio_values': [],
            'portfolio_returns': [],
            'portfolio_weights': [],
            'benchmark_returns': [],
            'regimes': [],
            'regime_probabilities': [],
            'transactions': [],
            'cash_flows': []
        }
    
    def run_backtest(
        self,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        market_features: Optional[pd.DataFrame] = None,
        custom_signals: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        backtest_dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self._get_rebalance_frequency()
        )
        
        asset_returns_aligned = asset_returns.reindex(
            pd.date_range(self.config.start_date, self.config.end_date, freq='D')
        ).fillna(0)
        
        benchmark_aligned = benchmark_returns.reindex(asset_returns_aligned.index).fillna(0)
        
        for i, rebalance_date in enumerate(backtest_dates):
            try:
                self._execute_rebalance_step(
                    rebalance_date,
                    asset_returns_aligned,
                    benchmark_aligned,
                    market_features,
                    custom_signals,
                    i
                )
            except Exception as e:
                logger.error(f"Error during rebalance on {rebalance_date}: {e}")
                continue
        
        return self._compile_results(benchmark_aligned)
    
    def _get_rebalance_frequency(self) -> str:
        freq_map = {
            'D': 'D',
            'W': 'W-FRI',
            'M': 'M',
            'Q': 'Q'
        }
        return freq_map.get(self.config.rebalance_frequency, 'M')
    
    def _execute_rebalance_step(
        self,
        rebalance_date: datetime,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        market_features: Optional[pd.DataFrame],
        custom_signals: Optional[pd.DataFrame],
        step_index: int
    ):
        if rebalance_date not in asset_returns.index:
            return
        
        lookback_start = rebalance_date - timedelta(days=self.config.lookback_window)
        lookback_returns = asset_returns.loc[lookback_start:rebalance_date].iloc[:-1]
        
        if len(lookback_returns) < 30:
            logger.warning(f"Insufficient data for {rebalance_date}, skipping rebalance")
            return
        
        current_regime = None
        regime_probabilities = None
        
        if self.regime_detector is not None and market_features is not None:
            try:
                regime_features = market_features.loc[lookback_start:rebalance_date].iloc[:-1]
                if len(regime_features) >= 30:
                    if not self.regime_detector.is_fitted:
                        self.regime_detector.fit(regime_features)
                    
                    latest_features = regime_features.tail(1)
                    current_regime = self.regime_detector.predict_regimes(latest_features)[0]
                    regime_probabilities = self.regime_detector.predict_regime_probabilities(latest_features)[0]
                    
            except Exception as e:
                logger.warning(f"Regime detection failed for {rebalance_date}: {e}")
        
        expected_returns = lookback_returns.mean() * 252
        covariance_matrix = lookback_returns.cov() * 252
        
        optimization_kwargs = {
            'current_weights': self.current_weights,
            'max_weight': self.config.max_weight,
            'min_weight': self.config.min_weight
        }
        
        if custom_signals is not None and rebalance_date in custom_signals.index:
            signals = custom_signals.loc[rebalance_date]
            optimization_kwargs['custom_signals'] = signals
        
        try:
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                method=self.config.optimization_method,
                expected_returns=expected_returns.values,
                covariance_matrix=covariance_matrix.values,
                **optimization_kwargs
            )
            
            new_weights = pd.Series(
                optimization_result['weights'],
                index=expected_returns.index
            )
            
        except Exception as e:
            logger.error(f"Optimization failed for {rebalance_date}: {e}")
            new_weights = self.current_weights if self.current_weights is not None else pd.Series(
                1.0 / len(expected_returns), index=expected_returns.index
            )
        
        portfolio_return, transactions = self._execute_trades(
            rebalance_date, new_weights, asset_returns
        )
        
        self.current_weights = new_weights
        
        self.results['dates'].append(rebalance_date)
        self.results['portfolio_values'].append(self.portfolio_value)
        self.results['portfolio_returns'].append(portfolio_return)
        self.results['portfolio_weights'].append(new_weights.to_dict())
        self.results['benchmark_returns'].append(benchmark_returns.loc[rebalance_date])
        self.results['regimes'].append(current_regime)
        self.results['regime_probabilities'].append(regime_probabilities)
        self.results['transactions'].extend(transactions)
        
        logger.debug(f"Rebalance completed for {rebalance_date}: Portfolio value = ${self.portfolio_value:,.2f}")
    
    def _execute_trades(
        self,
        trade_date: datetime,
        target_weights: pd.Series,
        asset_returns: pd.DataFrame
    ) -> Tuple[float, List[Dict[str, Any]]]:
        if self.current_weights is None:
            self.current_weights = pd.Series(0, index=target_weights.index)
        
        current_weights_aligned = self.current_weights.reindex(target_weights.index).fillna(0)
        
        weight_changes = target_weights - current_weights_aligned
        
        transactions = []
        total_transaction_costs = 0
        
        for asset in target_weights.index:
            weight_change = weight_changes[asset]
            
            if abs(weight_change) > 0.001:
                trade_value = abs(weight_change) * self.portfolio_value
                transaction_cost = trade_value * self.config.transaction_costs
                total_transaction_costs += transaction_cost
                
                transactions.append({
                    'date': trade_date,
                    'asset': asset,
                    'weight_change': weight_change,
                    'trade_value': trade_value,
                    'transaction_cost': transaction_cost,
                    'trade_type': 'buy' if weight_change > 0 else 'sell'
                })
        
        if trade_date in asset_returns.index:
            daily_returns = asset_returns.loc[trade_date]
            portfolio_return = (current_weights_aligned * daily_returns).sum()
            
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_value -= total_transaction_costs
        else:
            portfolio_return = 0
        
        return portfolio_return, transactions
    
    def _compile_results(self, benchmark_returns: pd.Series) -> BacktestResults:
        logger.info("Compiling backtest results")
        
        portfolio_returns = pd.Series(
            self.results['portfolio_returns'],
            index=self.results['dates']
        )
        
        portfolio_values = pd.Series(
            self.results['portfolio_values'],
            index=self.results['dates']
        )
        
        weights_df = pd.DataFrame(
            self.results['portfolio_weights'],
            index=self.results['dates']
        ).fillna(0)
        
        transactions_df = pd.DataFrame(self.results['transactions'])
        
        regime_df = pd.DataFrame({
            'regime': self.results['regimes'],
            'regime_probabilities': self.results['regime_probabilities']
        }, index=self.results['dates'])
        
        benchmark_aligned = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
        
        performance_metrics = self._calculate_performance_metrics(
            portfolio_returns, benchmark_aligned
        )
        
        drawdown_periods = self._identify_drawdown_periods(portfolio_values)
        
        trade_analysis = self._analyze_trades(transactions_df)
        
        return BacktestResults(
            portfolio_returns=portfolio_returns,
            portfolio_weights=weights_df,
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_aligned,
            transactions=transactions_df,
            regime_history=regime_df,
            performance_metrics=performance_metrics,
            drawdown_periods=drawdown_periods,
            trade_analysis=trade_analysis
        )
    
    def _calculate_performance_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / rolling_max - 1)
        max_drawdown = drawdowns.min()
        
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
        alpha = annualized_return - benchmark_annualized
        
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino_ratio = annualized_return / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': (portfolio_returns > 0).mean(),
            'avg_win': portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).any() else 0,
            'avg_loss': portfolio_returns[portfolio_returns < 0].mean() if (portfolio_returns < 0).any() else 0
        }
    
    def _identify_drawdown_periods(self, portfolio_values: pd.Series) -> List[Dict[str, Any]]:
        cumulative = portfolio_values / portfolio_values.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative / rolling_max - 1)
        
        drawdown_periods = []
        in_drawdown = False
        current_period = {}
        
        for date, dd in drawdowns.items():
            if dd < -0.01 and not in_drawdown:
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
                
                if abs(dd) < 0.001:
                    current_period['end_date'] = date
                    current_period['recovery_value'] = portfolio_values[date]
                    current_period['max_drawdown'] = (current_period['trough_value'] / current_period['peak_value'] - 1)
                    current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
                    
                    drawdown_periods.append(current_period)
                    in_drawdown = False
        
        return drawdown_periods
    
    def _analyze_trades(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        if len(transactions) == 0:
            return {}
        
        total_trades = len(transactions)
        total_volume = transactions['trade_value'].sum()
        total_costs = transactions['transaction_cost'].sum()
        
        avg_trade_size = transactions['trade_value'].mean()
        
        buy_trades = transactions[transactions['trade_type'] == 'buy']
        sell_trades = transactions[transactions['trade_type'] == 'sell']
        
        return {
            'total_trades': total_trades,
            'total_volume': total_volume,
            'total_transaction_costs': total_costs,
            'average_trade_size': avg_trade_size,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'cost_ratio': total_costs / total_volume if total_volume > 0 else 0,
            'trades_per_rebalance': total_trades / len(transactions['date'].unique()) if len(transactions['date'].unique()) > 0 else 0
        }