#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

from src.backtesting.framework.advanced_backtesting import AdvancedBacktestResults
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('seaborn-v0_8')


@dataclass
class PerformanceAnalysisConfig:
    """Configuration for performance analysis"""
    benchmark_name: str = "Benchmark"
    risk_free_rate: float = 0.02
    confidence_levels: List[float] = None
    rolling_window: int = 252
    create_plots: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


class AdvancedPerformanceAnalyzer:
    """Advanced performance analyzer for backtesting results"""
    
    def __init__(self, config: PerformanceAnalysisConfig = None):
        self.config = config or PerformanceAnalysisConfig()
        self.analysis_results = {}
        
    def analyze_backtest_results(
        self, 
        results: AdvancedBacktestResults, 
        strategy_name: str = "Strategy"
    ) -> Dict[str, Any]:
        """Comprehensive analysis of backtesting results"""
        
        logger.info(f"Analyzing performance for {strategy_name}")
        
        if results.portfolio_returns.empty:
            logger.warning("Empty portfolio returns, skipping analysis")
            return {}
        
        analysis = {
            'strategy_name': strategy_name,
            'analysis_date': datetime.now().isoformat(),
            'basic_metrics': self._calculate_basic_metrics(results),
            'risk_adjusted_metrics': self._calculate_risk_adjusted_metrics(results),
            'drawdown_analysis': self._analyze_drawdowns(results),
            'rolling_analysis': self._calculate_rolling_metrics(results),
            'regime_analysis': self._analyze_regime_performance(results),
            'trade_analysis': self._analyze_trading_performance(results),
            'benchmark_comparison': self._compare_to_benchmark(results)
        }
        
        # Add volatility analysis
        analysis['volatility_analysis'] = self._analyze_volatility_patterns(results)
        
        # Add correlation analysis
        analysis['correlation_analysis'] = self._analyze_correlations(results)
        
        # Store results
        self.analysis_results[strategy_name] = analysis
        
        # Generate plots if enabled
        if self.config.create_plots:
            self._create_performance_plots(results, strategy_name)
        
        logger.info(f"Performance analysis completed for {strategy_name}")
        return analysis
    
    def compare_strategies(self, results_dict: Dict[str, AdvancedBacktestResults]) -> Dict[str, Any]:
        """Compare multiple strategies"""
        
        logger.info(f"Comparing {len(results_dict)} strategies")
        
        comparison = {
            'strategies': list(results_dict.keys()),
            'comparison_date': datetime.now().isoformat(),
            'performance_summary': {},
            'risk_comparison': {},
            'drawdown_comparison': {},
            'correlation_matrix': {},
            'ranking': {}
        }
        
        # Analyze each strategy
        for name, results in results_dict.items():
            self.analyze_backtest_results(results, name)
        
        # Create comparison metrics
        comparison['performance_summary'] = self._create_performance_summary()
        comparison['risk_comparison'] = self._create_risk_comparison()
        comparison['ranking'] = self._rank_strategies()
        
        # Create comparison plots
        if self.config.create_plots:
            self._create_comparison_plots(results_dict)
        
        return comparison
    
    def _calculate_basic_metrics(self, results: AdvancedBacktestResults) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        
        returns = results.portfolio_returns
        
        if returns.empty:
            return {}
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Win/Loss statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Streak analysis
        streaks = self._calculate_win_loss_streaks(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': volatility,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'max_win_streak': streaks['max_win_streak'],
            'max_loss_streak': streaks['max_loss_streak'],
            'current_streak': streaks['current_streak']
        }
    
    def _calculate_risk_adjusted_metrics(self, results: AdvancedBacktestResults) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        
        returns = results.portfolio_returns
        
        if returns.empty:
            return {}
        
        # Basic risk metrics
        annualized_return = (1 + (1 + returns).prod() - 1) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < self.config.risk_free_rate / 252]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        max_drawdown = self._calculate_max_drawdown(results.portfolio_values)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_metrics = self._calculate_var_cvar(returns)
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio
        tail_ratio = self._calculate_tail_ratio(returns)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            **var_metrics
        }
    
    def _analyze_drawdowns(self, results: AdvancedBacktestResults) -> Dict[str, Any]:
        """Detailed drawdown analysis"""
        
        portfolio_values = results.portfolio_values
        
        if portfolio_values.empty:
            return {}
        
        # Calculate drawdowns
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values / peak - 1)
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        current_period = {}
        
        threshold = -0.01  # 1% drawdown threshold
        
        for date, dd in drawdown.items():
            if dd <= threshold and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                current_period = {
                    'start': date,
                    'peak_value': portfolio_values[date],
                    'trough_value': portfolio_values[date],
                    'trough_date': date,
                    'max_drawdown': dd
                }
            elif in_drawdown:
                # Update trough if deeper
                if dd < current_period['max_drawdown']:
                    current_period['max_drawdown'] = dd
                    current_period['trough_value'] = portfolio_values[date]
                    current_period['trough_date'] = date
                
                # Check for recovery
                if dd > -0.001:  # Recovery threshold
                    current_period['end'] = date
                    current_period['recovery_value'] = portfolio_values[date]
                    current_period['duration'] = (current_period['end'] - current_period['start']).days
                    current_period['recovery_time'] = (current_period['end'] - current_period['trough_date']).days
                    
                    drawdown_periods.append(current_period)
                    in_drawdown = False
        
        # Calculate statistics
        if drawdown_periods:
            avg_drawdown = np.mean([period['max_drawdown'] for period in drawdown_periods])
            avg_duration = np.mean([period['duration'] for period in drawdown_periods])
            avg_recovery = np.mean([period.get('recovery_time', 0) for period in drawdown_periods])
            max_duration = max([period['duration'] for period in drawdown_periods])
        else:
            avg_drawdown = 0
            avg_duration = 0
            avg_recovery = 0
            max_duration = 0
        
        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': avg_drawdown,
            'num_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': avg_duration,
            'max_drawdown_duration': max_duration,
            'avg_recovery_time': avg_recovery,
            'drawdown_periods': drawdown_periods,
            'current_drawdown': drawdown.iloc[-1] if not drawdown.empty else 0
        }
    
    def _calculate_rolling_metrics(self, results: AdvancedBacktestResults) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics"""
        
        returns = results.portfolio_returns
        
        if returns.empty or len(returns) < self.config.rolling_window:
            return {}
        
        window = min(self.config.rolling_window, len(returns) // 2)
        
        # Rolling metrics
        rolling_return = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        rolling_volatility = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return * 252 - self.config.risk_free_rate) / rolling_volatility
        
        # Rolling drawdown
        portfolio_values = results.portfolio_values
        rolling_peak = portfolio_values.rolling(window).max()
        rolling_drawdown = (portfolio_values / rolling_peak - 1)
        
        return {
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe,
            'rolling_drawdown': rolling_drawdown
        }
    
    def _analyze_regime_performance(self, results: AdvancedBacktestResults) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        
        if results.regime_history is None or results.regime_history.empty:
            return {'regime_analysis_available': False}
        
        regime_data = results.regime_history
        returns = results.portfolio_returns
        
        # Align data
        common_dates = returns.index.intersection(regime_data.index)
        if len(common_dates) == 0:
            return {'regime_analysis_available': False}
        
        aligned_returns = returns.loc[common_dates]
        aligned_regimes = regime_data.loc[common_dates, 'regime']
        
        # Analyze performance by regime
        regime_performance = {}
        
        for regime in aligned_regimes.unique():
            if pd.notna(regime):
                regime_returns = aligned_returns[aligned_regimes == regime]
                
                if len(regime_returns) > 0:
                    regime_performance[f'regime_{int(regime)}'] = {
                        'count': len(regime_returns),
                        'total_return': (1 + regime_returns).prod() - 1,
                        'avg_return': regime_returns.mean(),
                        'volatility': regime_returns.std(),
                        'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'win_rate': (regime_returns > 0).mean(),
                        'max_return': regime_returns.max(),
                        'min_return': regime_returns.min()
                    }
        
        return {
            'regime_analysis_available': True,
            'regime_performance': regime_performance,
            'regime_transitions': len(aligned_regimes.diff().dropna().nonzero()[0])
        }
    
    def _analyze_trading_performance(self, results: AdvancedBacktestResults) -> Dict[str, Any]:
        """Analyze trading execution performance"""
        
        if results.trades.empty:
            return {'trading_analysis_available': False}
        
        trades = results.trades
        
        # Basic trade statistics
        total_trades = len(trades)
        total_volume = trades.get('trade_value', pd.Series()).sum()
        
        # Trade size analysis
        if 'trade_value' in trades.columns:
            avg_trade_size = trades['trade_value'].mean()
            median_trade_size = trades['trade_value'].median()
            trade_size_std = trades['trade_value'].std()
        else:
            avg_trade_size = median_trade_size = trade_size_std = 0
        
        # Cost analysis
        if 'commission' in trades.columns:
            total_commission = trades['commission'].sum()
            avg_commission = trades['commission'].mean()
        else:
            total_commission = avg_commission = 0
        
        if 'slippage' in trades.columns:
            total_slippage = trades['slippage'].sum()
            avg_slippage = trades['slippage'].mean()
        else:
            total_slippage = avg_slippage = 0
        
        if 'market_impact' in trades.columns:
            total_market_impact = trades['market_impact'].sum()
            avg_market_impact = trades['market_impact'].mean()
        else:
            total_market_impact = avg_market_impact = 0
        
        # Buy/Sell analysis
        if 'side' in trades.columns:
            buy_trades = trades[trades['side'] == 'buy']
            sell_trades = trades[trades['side'] == 'sell']
            
            buy_volume = buy_trades.get('trade_value', pd.Series()).sum()
            sell_volume = sell_trades.get('trade_value', pd.Series()).sum()
        else:
            buy_volume = sell_volume = 0
            buy_trades = sell_trades = pd.DataFrame()
        
        return {
            'trading_analysis_available': True,
            'total_trades': total_trades,
            'total_volume': total_volume,
            'avg_trade_size': avg_trade_size,
            'median_trade_size': median_trade_size,
            'trade_size_volatility': trade_size_std,
            'total_commission': total_commission,
            'avg_commission': avg_commission,
            'total_slippage': total_slippage,
            'avg_slippage': avg_slippage,
            'total_market_impact': total_market_impact,
            'avg_market_impact': avg_market_impact,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_costs': total_commission + total_slippage + total_market_impact
        }
    
    def _compare_to_benchmark(self, results: AdvancedBacktestResults) -> Dict[str, float]:
        """Compare strategy to benchmark"""
        
        portfolio_returns = results.portfolio_returns
        benchmark_returns = results.benchmark_returns
        
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}
        
        # Align returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}
        
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]
        
        # Calculate metrics
        portfolio_total = (1 + port_ret).prod() - 1
        benchmark_total = (1 + bench_ret).prod() - 1
        
        alpha = portfolio_total - benchmark_total
        
        # Beta calculation
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        benchmark_variance = np.var(bench_ret)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Tracking error and information ratio
        excess_returns = port_ret - bench_ret
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Up/Down capture ratios
        up_market = bench_ret > 0
        down_market = bench_ret < 0
        
        if up_market.any():
            up_capture = (port_ret[up_market].mean() / bench_ret[up_market].mean()) if bench_ret[up_market].mean() != 0 else 0
        else:
            up_capture = 0
            
        if down_market.any():
            down_capture = (port_ret[down_market].mean() / bench_ret[down_market].mean()) if bench_ret[down_market].mean() != 0 else 0
        else:
            down_capture = 0
        
        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'correlation': np.corrcoef(port_ret, bench_ret)[0, 1] if len(port_ret) > 1 else 0
        }
    
    def _analyze_volatility_patterns(self, results: AdvancedBacktestResults) -> Dict[str, Any]:
        """Analyze volatility patterns and clustering"""
        
        returns = results.portfolio_returns
        
        if returns.empty:
            return {}
        
        # Rolling volatility
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Volatility clustering (ARCH test)
        squared_returns = returns ** 2
        vol_autocorr = squared_returns.autocorr(lag=1)
        
        # Volatility regimes
        high_vol_threshold = rolling_vol.quantile(0.75)
        low_vol_threshold = rolling_vol.quantile(0.25)
        
        high_vol_periods = rolling_vol > high_vol_threshold
        low_vol_periods = rolling_vol < low_vol_threshold
        
        return {
            'avg_volatility': rolling_vol.mean(),
            'volatility_std': rolling_vol.std(),
            'max_volatility': rolling_vol.max(),
            'min_volatility': rolling_vol.min(),
            'vol_autocorrelation': vol_autocorr,
            'high_vol_periods': high_vol_periods.sum(),
            'low_vol_periods': low_vol_periods.sum(),
            'vol_regime_persistence': self._calculate_regime_persistence(rolling_vol > rolling_vol.median())
        }
    
    def _analyze_correlations(self, results: AdvancedBacktestResults) -> Dict[str, Any]:
        """Analyze correlation patterns"""
        
        portfolio_returns = results.portfolio_returns
        benchmark_returns = results.benchmark_returns
        
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}
        
        # Align returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 2:
            return {}
        
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]
        
        # Static correlation
        static_corr = np.corrcoef(port_ret, bench_ret)[0, 1]
        
        # Rolling correlation
        window = min(60, len(common_dates) // 4)  # 60-day window or 1/4 of data
        if window >= 10:
            rolling_corr = port_ret.rolling(window).corr(bench_ret)
            
            return {
                'static_correlation': static_corr,
                'avg_rolling_correlation': rolling_corr.mean(),
                'correlation_volatility': rolling_corr.std(),
                'max_correlation': rolling_corr.max(),
                'min_correlation': rolling_corr.min()
            }
        else:
            return {'static_correlation': static_corr}
    
    def _calculate_win_loss_streaks(self, returns: pd.Series) -> Dict[str, int]:
        """Calculate win/loss streaks"""
        
        wins = (returns > 0).astype(int)
        losses = (returns < 0).astype(int)
        
        # Calculate streaks
        win_streaks = []
        loss_streaks = []
        current_win_streak = 0
        current_loss_streak = 0
        
        for ret in returns:
            if ret > 0:
                current_win_streak += 1
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
            elif ret < 0:
                current_loss_streak += 1
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
            else:  # ret == 0
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
        
        # Add final streaks
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        return {
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'current_streak': current_win_streak if current_win_streak > 0 else -current_loss_streak
        }
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        if portfolio_values.empty:
            return 0
        
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values / peak - 1)
        return drawdown.min()
    
    def _calculate_var_cvar(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate VaR and CVaR at different confidence levels"""
        
        if returns.empty:
            return {}
        
        var_cvar = {}
        
        for confidence in self.config.confidence_levels:
            alpha = 1 - confidence
            var = np.percentile(returns, alpha * 100)
            cvar = returns[returns <= var].mean() if (returns <= var).any() else var
            
            var_cvar[f'var_{int(confidence*100)}'] = var
            var_cvar[f'cvar_{int(confidence*100)}'] = cvar
        
        return var_cvar
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        
        if returns.empty:
            return 0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        return abs(p95 / p5) if p5 != 0 else 0
    
    def _calculate_regime_persistence(self, regime_series: pd.Series) -> float:
        """Calculate regime persistence (probability of staying in same regime)"""
        
        if len(regime_series) < 2:
            return 0
        
        transitions = regime_series.diff().fillna(False) == 0
        return transitions.mean()
    
    def _create_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Create performance summary for all analyzed strategies"""
        
        summary = {}
        
        for strategy_name, analysis in self.analysis_results.items():
            basic_metrics = analysis.get('basic_metrics', {})
            risk_metrics = analysis.get('risk_adjusted_metrics', {})
            
            summary[strategy_name] = {
                'total_return': basic_metrics.get('total_return', 0),
                'annualized_return': basic_metrics.get('annualized_return', 0),
                'volatility': basic_metrics.get('annualized_volatility', 0),
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'win_rate': basic_metrics.get('win_rate', 0)
            }
        
        return summary
    
    def _create_risk_comparison(self) -> Dict[str, Dict[str, float]]:
        """Create risk comparison for all strategies"""
        
        risk_comparison = {}
        
        for strategy_name, analysis in self.analysis_results.items():
            risk_metrics = analysis.get('risk_adjusted_metrics', {})
            
            risk_comparison[strategy_name] = {
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': risk_metrics.get('sortino_ratio', 0),
                'calmar_ratio': risk_metrics.get('calmar_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'var_95': risk_metrics.get('var_95', 0),
                'skewness': risk_metrics.get('skewness', 0),
                'kurtosis': risk_metrics.get('kurtosis', 0)
            }
        
        return risk_comparison
    
    def _rank_strategies(self) -> Dict[str, Dict[str, int]]:
        """Rank strategies by different metrics"""
        
        if not self.analysis_results:
            return {}
        
        metrics_to_rank = [
            'total_return', 'sharpe_ratio', 'sortino_ratio', 
            'calmar_ratio', 'win_rate'
        ]
        
        rankings = {}
        
        for metric in metrics_to_rank:
            strategy_values = []
            
            for strategy_name, analysis in self.analysis_results.items():
                if metric in ['total_return', 'win_rate']:
                    value = analysis.get('basic_metrics', {}).get(metric, 0)
                else:
                    value = analysis.get('risk_adjusted_metrics', {}).get(metric, 0)
                
                strategy_values.append((strategy_name, value))
            
            # Sort by value (descending for positive metrics)
            if metric == 'max_drawdown':
                strategy_values.sort(key=lambda x: x[1])  # Ascending for drawdown
            else:
                strategy_values.sort(key=lambda x: x[1], reverse=True)  # Descending for others
            
            # Create rankings
            rankings[metric] = {
                strategy: rank + 1 
                for rank, (strategy, _) in enumerate(strategy_values)
            }
        
        return rankings
    
    def _create_performance_plots(self, results: AdvancedBacktestResults, strategy_name: str):
        """Create comprehensive performance plots"""
        
        if results.portfolio_returns.empty:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Performance Analysis: {strategy_name}', fontsize=16)
        
        # 1. Cumulative returns
        cumulative_returns = (1 + results.portfolio_returns).cumprod()
        if not results.benchmark_returns.empty:
            benchmark_cumulative = (1 + results.benchmark_returns).cumprod()
            axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                           label=self.config.benchmark_name, alpha=0.7)
        
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                       label=strategy_name, linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        portfolio_values = results.portfolio_values
        if not portfolio_values.empty:
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values / peak - 1) * 100
            
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                                   alpha=0.3, color='red', label='Drawdown')
            axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling volatility
        if len(results.portfolio_returns) > 60:
            rolling_vol = results.portfolio_returns.rolling(60).std() * np.sqrt(252) * 100
            axes[1, 0].plot(rolling_vol.index, rolling_vol.values, linewidth=2)
            axes[1, 0].set_title('Rolling Volatility (60-day)')
            axes[1, 0].set_ylabel('Volatility (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Return distribution
        axes[1, 1].hist(results.portfolio_returns * 100, bins=50, alpha=0.7, density=True)
        axes[1, 1].axvline(results.portfolio_returns.mean() * 100, color='red', 
                          linestyle='--', label='Mean')
        axes[1, 1].set_title('Return Distribution')
        axes[1, 1].set_xlabel('Daily Return (%)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f'performance_analysis_{strategy_name}.{self.config.plot_format}'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance plot: {filename}")
        
        plt.show()
    
    def _create_comparison_plots(self, results_dict: Dict[str, AdvancedBacktestResults]):
        """Create comparison plots for multiple strategies"""
        
        if not results_dict:
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Comparison', fontsize=16)
        
        # 1. Cumulative returns comparison
        for strategy_name, results in results_dict.items():
            if not results.portfolio_returns.empty:
                cumulative_returns = (1 + results.portfolio_returns).cumprod()
                axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                               label=strategy_name, linewidth=2)
        
        axes[0, 0].set_title('Cumulative Returns Comparison')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Risk-Return scatter
        returns = []
        volatilities = []
        strategy_names = []
        
        for strategy_name, analysis in self.analysis_results.items():
            if analysis:
                basic_metrics = analysis.get('basic_metrics', {})
                ret = basic_metrics.get('annualized_return', 0) * 100
                vol = basic_metrics.get('annualized_volatility', 0) * 100
                
                returns.append(ret)
                volatilities.append(vol)
                strategy_names.append(strategy_name)
        
        if returns and volatilities:
            scatter = axes[0, 1].scatter(volatilities, returns, s=100, alpha=0.7)
            
            for i, name in enumerate(strategy_names):
                axes[0, 1].annotate(name, (volatilities[i], returns[i]), 
                                   xytext=(5, 5), textcoords='offset points')
            
            axes[0, 1].set_title('Risk-Return Profile')
            axes[0, 1].set_xlabel('Volatility (%)')
            axes[0, 1].set_ylabel('Annualized Return (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sharpe ratio comparison
        sharpe_ratios = []
        for strategy_name, analysis in self.analysis_results.items():
            if analysis:
                sharpe = analysis.get('risk_adjusted_metrics', {}).get('sharpe_ratio', 0)
                sharpe_ratios.append(sharpe)
        
        if sharpe_ratios:
            axes[1, 0].bar(range(len(strategy_names)), sharpe_ratios, alpha=0.7)
            axes[1, 0].set_title('Sharpe Ratio Comparison')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].set_xticks(range(len(strategy_names)))
            axes[1, 0].set_xticklabels(strategy_names, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Maximum drawdown comparison
        max_drawdowns = []
        for strategy_name, analysis in self.analysis_results.items():
            if analysis:
                dd = analysis.get('risk_adjusted_metrics', {}).get('max_drawdown', 0) * 100
                max_drawdowns.append(abs(dd))
        
        if max_drawdowns:
            axes[1, 1].bar(range(len(strategy_names)), max_drawdowns, alpha=0.7, color='red')
            axes[1, 1].set_title('Maximum Drawdown Comparison')
            axes[1, 1].set_ylabel('Max Drawdown (%)')
            axes[1, 1].set_xticks(range(len(strategy_names)))
            axes[1, 1].set_xticklabels(strategy_names, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f'strategy_comparison.{self.config.plot_format}'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot: {filename}")
        
        plt.show()
    
    def generate_analysis_report(self, strategy_name: str) -> str:
        """Generate comprehensive text report for a strategy"""
        
        if strategy_name not in self.analysis_results:
            return f"No analysis available for {strategy_name}"
        
        analysis = self.analysis_results[strategy_name]
        
        report = f"PERFORMANCE ANALYSIS REPORT: {strategy_name.upper()}\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Analysis Date: {analysis['analysis_date']}\n\n"
        
        # Basic metrics
        basic = analysis.get('basic_metrics', {})
        if basic:
            report += "BASIC PERFORMANCE METRICS\n"
            report += "-" * 30 + "\n"
            report += f"Total Return: {basic.get('total_return', 0):.2%}\n"
            report += f"Annualized Return: {basic.get('annualized_return', 0):.2%}\n"
            report += f"Annualized Volatility: {basic.get('annualized_volatility', 0):.2%}\n"
            report += f"Win Rate: {basic.get('win_rate', 0):.1%}\n"
            report += f"Win/Loss Ratio: {basic.get('win_loss_ratio', 0):.2f}\n"
            report += f"Max Win Streak: {basic.get('max_win_streak', 0)}\n"
            report += f"Max Loss Streak: {basic.get('max_loss_streak', 0)}\n\n"
        
        # Risk-adjusted metrics
        risk = analysis.get('risk_adjusted_metrics', {})
        if risk:
            report += "RISK-ADJUSTED METRICS\n"
            report += "-" * 25 + "\n"
            report += f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.3f}\n"
            report += f"Sortino Ratio: {risk.get('sortino_ratio', 0):.3f}\n"
            report += f"Calmar Ratio: {risk.get('calmar_ratio', 0):.3f}\n"
            report += f"Maximum Drawdown: {risk.get('max_drawdown', 0):.2%}\n"
            report += f"VaR (95%): {risk.get('var_95', 0):.2%}\n"
            report += f"CVaR (95%): {risk.get('cvar_95', 0):.2%}\n"
            report += f"Skewness: {risk.get('skewness', 0):.3f}\n"
            report += f"Kurtosis: {risk.get('kurtosis', 0):.3f}\n\n"
        
        # Benchmark comparison
        benchmark = analysis.get('benchmark_comparison', {})
        if benchmark:
            report += "BENCHMARK COMPARISON\n"
            report += "-" * 20 + "\n"
            report += f"Alpha: {benchmark.get('alpha', 0):.2%}\n"
            report += f"Beta: {benchmark.get('beta', 0):.3f}\n"
            report += f"Information Ratio: {benchmark.get('information_ratio', 0):.3f}\n"
            report += f"Tracking Error: {benchmark.get('tracking_error', 0):.2%}\n"
            report += f"Correlation: {benchmark.get('correlation', 0):.3f}\n\n"
        
        # Trading analysis
        trading = analysis.get('trade_analysis', {})
        if trading.get('trading_analysis_available', False):
            report += "TRADING PERFORMANCE\n"
            report += "-" * 20 + "\n"
            report += f"Total Trades: {trading.get('total_trades', 0)}\n"
            report += f"Total Volume: ${trading.get('total_volume', 0):,.0f}\n"
            report += f"Total Costs: ${trading.get('total_costs', 0):,.0f}\n"
            report += f"Average Trade Size: ${trading.get('avg_trade_size', 0):,.0f}\n"
            report += f"Commission: ${trading.get('total_commission', 0):,.0f}\n"
            report += f"Slippage: ${trading.get('total_slippage', 0):,.0f}\n"
            report += f"Market Impact: ${trading.get('total_market_impact', 0):,.0f}\n\n"
        
        report += "=" * 60 + "\n"
        report += f"Report generated by Advanced Performance Analyzer\n"
        
        return report


def create_performance_analyzer(
    benchmark_name: str = "Benchmark",
    risk_free_rate: float = 0.02,
    create_plots: bool = True
) -> AdvancedPerformanceAnalyzer:
    """Factory function to create performance analyzer"""
    
    config = PerformanceAnalysisConfig(
        benchmark_name=benchmark_name,
        risk_free_rate=risk_free_rate,
        create_plots=create_plots
    )
    
    analyzer = AdvancedPerformanceAnalyzer(config)
    logger.info("Created advanced performance analyzer")
    
    return analyzer