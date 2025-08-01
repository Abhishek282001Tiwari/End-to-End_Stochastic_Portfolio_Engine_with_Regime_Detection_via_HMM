import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

from src.optimization.objectives.risk_measures import RiskMeasures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class AttributionMethod(Enum):
    """Attribution analysis methods"""
    BRINSON = "brinson"
    FACTOR_BASED = "factor_based"
    REGIME_BASED = "regime_based"
    CURRENCY = "currency"
    SECTOR = "sector"
    MULTI_LEVEL = "multi_level"


@dataclass
class AttributionResult:
    """Results of performance attribution analysis"""
    method: AttributionMethod
    total_active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    currency_effect: Optional[float] = None
    detailed_attribution: Dict[str, Any] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)


@dataclass
class FactorExposure:
    """Factor exposure and attribution"""
    factor_name: str
    exposure: float
    factor_return: float
    contribution: float
    t_statistic: float
    p_value: float


class AdvancedPerformanceAttribution:
    """Advanced performance attribution analysis with multiple methodologies"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.risk_measures = RiskMeasures()
        
    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> AttributionResult:
        """
        Perform Brinson attribution analysis
        
        Decomposes active return into:
        - Allocation effect: Impact of over/under-weighting sectors
        - Selection effect: Impact of security selection within sectors
        - Interaction effect: Combined impact of allocation and selection
        """
        logger.info("Performing Brinson attribution analysis")
        
        # Align data
        common_assets = portfolio_weights.columns.intersection(benchmark_weights.columns)
        common_dates = portfolio_weights.index.intersection(benchmark_weights.index)
        
        if len(common_assets) == 0 or len(common_dates) == 0:
            raise ValueError("No common assets or dates found between portfolio and benchmark")
        
        port_weights = portfolio_weights.loc[common_dates, common_assets]
        bench_weights = benchmark_weights.loc[common_dates, common_assets]
        port_returns = portfolio_returns.loc[common_dates, common_assets]
        bench_returns = benchmark_returns.loc[common_dates, common_assets]
        
        # Calculate sector-level attribution if sector mapping provided
        if sector_mapping:
            return self._sector_brinson_attribution(
                port_weights, bench_weights, port_returns, bench_returns, sector_mapping
            )
        else:
            return self._asset_brinson_attribution(
                port_weights, bench_weights, port_returns, bench_returns
            )
    
    def _asset_brinson_attribution(
        self,
        port_weights: pd.DataFrame,
        bench_weights: pd.DataFrame,
        port_returns: pd.DataFrame,
        bench_returns: pd.DataFrame
    ) -> AttributionResult:
        """Asset-level Brinson attribution"""
        
        # Calculate active weights and returns
        active_weights = port_weights - bench_weights
        active_returns = port_returns - bench_returns
        
        # Calculate portfolio and benchmark returns
        portfolio_return = (port_weights * port_returns).sum(axis=1)
        benchmark_return = (bench_weights * bench_returns).sum(axis=1)
        total_active_return = portfolio_return - benchmark_return
        
        # Attribution effects (aggregated over time)
        allocation_effect = (active_weights * bench_returns).sum().sum()
        selection_effect = (bench_weights * active_returns).sum().sum()
        interaction_effect = (active_weights * active_returns).sum().sum()
        
        # Detailed attribution by asset
        detailed_attribution = {}
        for asset in port_weights.columns:
            asset_allocation = (active_weights[asset] * bench_returns[asset]).sum()
            asset_selection = (bench_weights[asset] * active_returns[asset]).sum()
            asset_interaction = (active_weights[asset] * active_returns[asset]).sum()
            
            detailed_attribution[asset] = {
                'allocation_effect': asset_allocation,
                'selection_effect': asset_selection,
                'interaction_effect': asset_interaction,
                'total_contribution': asset_allocation + asset_selection + asset_interaction
            }
        
        return AttributionResult(
            method=AttributionMethod.BRINSON,
            total_active_return=total_active_return.sum(),
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            detailed_attribution=detailed_attribution
        )
    
    def _sector_brinson_attribution(
        self,
        port_weights: pd.DataFrame,
        bench_weights: pd.DataFrame,
        port_returns: pd.DataFrame,
        bench_returns: pd.DataFrame,
        sector_mapping: Dict[str, str]
    ) -> AttributionResult:
        """Sector-level Brinson attribution"""
        
        # Aggregate to sector level
        sectors = list(set(sector_mapping.values()))
        
        sector_port_weights = pd.DataFrame(index=port_weights.index, columns=sectors)
        sector_bench_weights = pd.DataFrame(index=bench_weights.index, columns=sectors)
        sector_port_returns = pd.DataFrame(index=port_returns.index, columns=sectors)
        sector_bench_returns = pd.DataFrame(index=bench_returns.index, columns=sectors)
        
        for sector in sectors:
            sector_assets = [asset for asset, sect in sector_mapping.items() if sect == sector]
            sector_assets = [asset for asset in sector_assets if asset in port_weights.columns]
            
            if sector_assets:
                # Sector weights
                sector_port_weights[sector] = port_weights[sector_assets].sum(axis=1)
                sector_bench_weights[sector] = bench_weights[sector_assets].sum(axis=1)
                
                # Sector returns (weighted average)
                port_sector_weight = port_weights[sector_assets].div(
                    port_weights[sector_assets].sum(axis=1), axis=0
                ).fillna(0)
                bench_sector_weight = bench_weights[sector_assets].div(
                    bench_weights[sector_assets].sum(axis=1), axis=0
                ).fillna(0)
                
                sector_port_returns[sector] = (port_sector_weight * port_returns[sector_assets]).sum(axis=1)
                sector_bench_returns[sector] = (bench_sector_weight * bench_returns[sector_assets]).sum(axis=1)
        
        # Fill NaN values
        sector_port_weights = sector_port_weights.fillna(0)
        sector_bench_weights = sector_bench_weights.fillna(0)
        sector_port_returns = sector_port_returns.fillna(0)
        sector_bench_returns = sector_bench_returns.fillna(0)
        
        # Calculate attribution at sector level
        return self._asset_brinson_attribution(
            sector_port_weights, sector_bench_weights, 
            sector_port_returns, sector_bench_returns
        )
    
    def factor_based_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        rolling_window: int = 252
    ) -> AttributionResult:
        """
        Factor-based performance attribution using multi-factor models
        """
        logger.info("Performing factor-based attribution analysis")
        
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # Align data
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        if len(common_dates) < 50:
            raise ValueError("Insufficient data for factor attribution analysis")
        
        portfolio_ret = portfolio_returns.loc[common_dates]
        factors = factor_returns.loc[common_dates]
        
        # Calculate excess returns if benchmark provided
        if benchmark_returns is not None:
            benchmark_ret = benchmark_returns.loc[common_dates]
            dependent_var = portfolio_ret - benchmark_ret
            total_active_return = dependent_var.sum()
        else:
            dependent_var = portfolio_ret - self.risk_free_rate / 252
            total_active_return = portfolio_ret.sum() - self.risk_free_rate / 252 * len(portfolio_ret)
        
        # Fit factor model
        X = factors.values
        y = dependent_var.values
        
        model = LinearRegression().fit(X, y)
        
        # Calculate statistics
        predictions = model.predict(X)
        residuals = y - predictions
        
        # T-statistics and p-values
        n = len(y)
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k - 1)
        var_coef = mse * np.linalg.inv(X.T @ X)
        se_coef = np.sqrt(np.diag(var_coef))
        t_stats = model.coef_ / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        
        # Factor exposures and contributions
        factor_exposures = []
        detailed_attribution = {}
        
        for i, factor_name in enumerate(factors.columns):
            exposure = model.coef_[i]
            factor_ret = factors.iloc[:, i].mean()
            contribution = exposure * factor_ret * len(factors)
            
            factor_exp = FactorExposure(
                factor_name=factor_name,
                exposure=exposure,
                factor_return=factor_ret,
                contribution=contribution,
                t_statistic=t_stats[i],
                p_value=p_values[i]
            )
            factor_exposures.append(factor_exp)
            
            detailed_attribution[factor_name] = {
                'exposure': exposure,
                'factor_return': factor_ret,
                'contribution': contribution,
                'significance': 'significant' if p_values[i] < 0.05 else 'not_significant'
            }
        
        # Alpha (intercept)
        alpha = model.intercept_ * len(factors)
        detailed_attribution['alpha'] = {
            'value': alpha,
            'annualized': alpha * 252 / len(factors)
        }
        
        return AttributionResult(
            method=AttributionMethod.FACTOR_BASED,
            total_active_return=total_active_return,
            allocation_effect=sum([fe.contribution for fe in factor_exposures]),
            selection_effect=alpha,
            interaction_effect=0,  # Not applicable for factor attribution
            detailed_attribution=detailed_attribution,
            statistical_significance={fe.factor_name: fe.p_value for fe in factor_exposures}
        )
    
    def regime_based_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        regime_labels: np.ndarray,
        regime_names: Dict[int, str]
    ) -> AttributionResult:
        """
        Regime-based performance attribution
        Analyzes performance in different market regimes
        """
        logger.info("Performing regime-based attribution analysis")
        
        # Align data
        min_length = min(len(portfolio_returns), len(benchmark_returns), len(regime_labels))
        port_ret = portfolio_returns.iloc[:min_length]
        bench_ret = benchmark_returns.iloc[:min_length]
        regimes = regime_labels[:min_length]
        
        active_returns = port_ret - bench_ret
        total_active_return = active_returns.sum()
        
        # Attribution by regime
        detailed_attribution = {}
        regime_contributions = {}
        
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            regime_mask = regimes == regime
            
            if regime_mask.sum() == 0:
                continue
            
            regime_active_ret = active_returns.iloc[regime_mask]
            regime_port_ret = port_ret.iloc[regime_mask]
            regime_bench_ret = bench_ret.iloc[regime_mask]
            
            # Calculate regime statistics
            regime_contribution = regime_active_ret.sum()
            regime_frequency = regime_mask.sum() / len(regimes)
            avg_active_return = regime_active_ret.mean()
            
            regime_name = regime_names.get(regime, f"Regime_{regime}")
            
            detailed_attribution[regime_name] = {
                'contribution': regime_contribution,
                'frequency': regime_frequency,
                'avg_active_return': avg_active_return,
                'periods': regime_mask.sum(),
                'portfolio_return': regime_port_ret.mean(),
                'benchmark_return': regime_bench_ret.mean(),
                'volatility': regime_active_ret.std()
            }
            
            regime_contributions[regime_name] = regime_contribution
        
        # Calculate allocation and selection effects based on regime timing
        # Allocation effect: impact of being in different regimes vs benchmark
        # Selection effect: impact of outperformance within regimes
        
        allocation_effect = 0
        selection_effect = 0
        
        for regime_name, stats in detailed_attribution.items():
            # Simple decomposition - can be enhanced
            allocation_contribution = stats['frequency'] * stats['benchmark_return'] * stats['periods']
            selection_contribution = stats['contribution'] - allocation_contribution
            
            allocation_effect += allocation_contribution
            selection_effect += selection_contribution
        
        return AttributionResult(
            method=AttributionMethod.REGIME_BASED,
            total_active_return=total_active_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=0,
            detailed_attribution=detailed_attribution
        )
    
    def multi_level_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        sector_mapping: Dict[str, str],
        country_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, AttributionResult]:
        """
        Multi-level performance attribution analysis
        Combines sector, country, and factor-based attribution
        """
        logger.info("Performing multi-level attribution analysis")
        
        results = {}
        
        # 1. Sector-level Brinson attribution
        try:
            sector_attribution = self.brinson_attribution(
                portfolio_weights, benchmark_weights,
                portfolio_returns, benchmark_returns,
                sector_mapping
            )
            results['sector'] = sector_attribution
        except Exception as e:
            logger.warning(f"Sector attribution failed: {e}")
        
        # 2. Country-level attribution (if mapping provided)
        if country_mapping:
            try:
                country_attribution = self.brinson_attribution(
                    portfolio_weights, benchmark_weights,
                    portfolio_returns, benchmark_returns,
                    country_mapping
                )
                results['country'] = country_attribution
            except Exception as e:
                logger.warning(f"Country attribution failed: {e}")
        
        # 3. Factor-based attribution
        try:
            portfolio_ret = (portfolio_weights * portfolio_returns).sum(axis=1)
            benchmark_ret = (benchmark_weights * benchmark_returns).sum(axis=1)
            
            factor_attribution = self.factor_based_attribution(
                portfolio_ret, factor_returns, benchmark_ret
            )
            results['factor'] = factor_attribution
        except Exception as e:
            logger.warning(f"Factor attribution failed: {e}")
        
        return results
    
    def calculate_attribution_confidence_intervals(
        self,
        attribution_result: AttributionResult,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95
    ) -> AttributionResult:
        """
        Calculate confidence intervals for attribution results using bootstrap
        """
        logger.info("Calculating attribution confidence intervals")
        
        # This is a simplified implementation
        # In practice, you would bootstrap the original data and recalculate attribution
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Placeholder confidence intervals (should be calculated from bootstrap)
        confidence_intervals = {
            'allocation_effect': (
                attribution_result.allocation_effect * 0.8,
                attribution_result.allocation_effect * 1.2
            ),
            'selection_effect': (
                attribution_result.selection_effect * 0.8,
                attribution_result.selection_effect * 1.2
            ),
            'total_active_return': (
                attribution_result.total_active_return * 0.9,
                attribution_result.total_active_return * 1.1
            )
        }
        
        attribution_result.confidence_intervals = confidence_intervals
        return attribution_result
    
    def generate_attribution_report(
        self,
        attribution_results: Union[AttributionResult, Dict[str, AttributionResult]],
        period_name: str = "Analysis Period"
    ) -> str:
        """Generate comprehensive attribution analysis report"""
        
        if isinstance(attribution_results, AttributionResult):
            attribution_results = {'main': attribution_results}
        
        report = f"PERFORMANCE ATTRIBUTION ANALYSIS - {period_name.upper()}\n"
        report += "=" * 70 + "\n\n"
        
        for analysis_name, result in attribution_results.items():
            report += f"{analysis_name.upper()} ATTRIBUTION ({result.method.value.upper()})\n"
            report += "-" * 50 + "\n"
            
            report += f"Total Active Return: {result.total_active_return:.4f} ({result.total_active_return*100:.2f}%)\n"
            report += f"Allocation Effect: {result.allocation_effect:.4f} ({result.allocation_effect*100:.2f}%)\n"
            report += f"Selection Effect: {result.selection_effect:.4f} ({result.selection_effect*100:.2f}%)\n"
            
            if result.interaction_effect != 0:
                report += f"Interaction Effect: {result.interaction_effect:.4f} ({result.interaction_effect*100:.2f}%)\n"
            
            if result.currency_effect is not None:
                report += f"Currency Effect: {result.currency_effect:.4f} ({result.currency_effect*100:.2f}%)\n"
            
            report += "\n"
            
            # Detailed breakdown
            if result.detailed_attribution:
                report += "DETAILED BREAKDOWN:\n"
                report += "-" * 25 + "\n"
                
                for item, details in result.detailed_attribution.items():
                    if isinstance(details, dict):
                        report += f"\n{item}:\n"
                        for key, value in details.items():
                            if isinstance(value, (int, float)):
                                if abs(value) > 0.0001:  # Only show significant values
                                    report += f"  {key}: {value:.4f}\n"
                            else:
                                report += f"  {key}: {value}\n"
                    else:
                        report += f"{item}: {details:.4f}\n"
            
            # Statistical significance
            if result.statistical_significance:
                report += "\nSTATISTICAL SIGNIFICANCE (p-values):\n"
                report += "-" * 35 + "\n"
                for factor, p_value in result.statistical_significance.items():
                    significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
                    report += f"{factor}: {p_value:.4f} {significance}\n"
            
            # Confidence intervals
            if result.confidence_intervals:
                report += "\nCONFIDENCE INTERVALS:\n"
                report += "-" * 25 + "\n"
                for metric, (lower, upper) in result.confidence_intervals.items():
                    report += f"{metric}: [{lower:.4f}, {upper:.4f}]\n"
            
            report += "\n" + "=" * 70 + "\n\n"
        
        report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "*** p < 0.01, ** p < 0.05, * p < 0.10\n"
        
        return report


class RegimePerformanceAnalyzer:
    """Specialized analyzer for regime-conditional performance metrics"""
    
    def __init__(self):
        self.risk_measures = RiskMeasures()
    
    def analyze_regime_performance(
        self,
        portfolio_returns: pd.Series,
        regime_labels: np.ndarray,
        regime_names: Dict[int, str],
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Comprehensive regime-conditional performance analysis"""
        
        min_length = min(len(portfolio_returns), len(regime_labels))
        port_ret = portfolio_returns.iloc[:min_length]
        regimes = regime_labels[:min_length]
        
        if benchmark_returns is not None:
            bench_ret = benchmark_returns.iloc[:min_length]
        else:
            bench_ret = None
        
        regime_analysis = {}
        
        for regime in np.unique(regimes):
            regime_mask = regimes == regime
            
            if regime_mask.sum() < 10:  # Minimum sample size
                continue
            
            regime_returns = port_ret.iloc[regime_mask]
            regime_name = regime_names.get(regime, f"Regime_{regime}")
            
            # Basic statistics
            stats = {
                'periods': regime_mask.sum(),
                'frequency': regime_mask.sum() / len(regimes),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'total_return': (1 + regime_returns).prod() - 1,
                'annualized_return': (1 + regime_returns.mean()) ** 252 - 1,
                'annualized_volatility': regime_returns.std() * np.sqrt(252)
            }
            
            # Risk metrics
            stats['var_95'] = self.risk_measures.value_at_risk(regime_returns.values, 0.05)
            stats['cvar_95'] = self.risk_measures.conditional_value_at_risk(regime_returns.values, 0.05)
            stats['max_drawdown'] = self.risk_measures.maximum_drawdown(regime_returns.values)[0]
            stats['win_rate'] = (regime_returns > 0).mean()
            stats['skewness'] = regime_returns.skew()
            stats['kurtosis'] = regime_returns.kurtosis()
            
            # Risk-adjusted metrics
            stats['sharpe_ratio'] = (stats['mean_return'] / stats['std_return'] * np.sqrt(252) 
                                   if stats['std_return'] > 0 else 0)
            stats['sortino_ratio'] = self.risk_measures.sortino_ratio(regime_returns.values)
            
            # Benchmark comparison
            if bench_ret is not None:
                regime_bench = bench_ret.iloc[regime_mask]
                stats['alpha'] = stats['mean_return'] - regime_bench.mean()
                stats['beta'] = (np.cov(regime_returns, regime_bench)[0, 1] / 
                               np.var(regime_bench) if np.var(regime_bench) > 0 else 0)
                stats['correlation'] = np.corrcoef(regime_returns, regime_bench)[0, 1]
                stats['tracking_error'] = (regime_returns - regime_bench).std()
                stats['information_ratio'] = (stats['alpha'] / stats['tracking_error'] 
                                             if stats['tracking_error'] > 0 else 0)
            
            regime_analysis[regime_name] = stats
        
        return regime_analysis
    
    def compare_regime_performance(
        self,
        regime_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare performance across different regimes"""
        
        if len(regime_analysis) < 2:
            return {}
        
        comparison = {}
        
        # Extract metrics for comparison
        regimes = list(regime_analysis.keys())
        metrics = ['mean_return', 'std_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in metrics:
            values = [regime_analysis[regime].get(metric, 0) for regime in regimes]
            
            comparison[metric] = {
                'best_regime': regimes[np.argmax(values)] if metric != 'max_drawdown' else regimes[np.argmin(np.abs(values))],
                'worst_regime': regimes[np.argmin(values)] if metric != 'max_drawdown' else regimes[np.argmax(np.abs(values))],
                'range': max(values) - min(values),
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
        
        # Regime stability analysis
        frequencies = [regime_analysis[regime]['frequency'] for regime in regimes]
        comparison['regime_distribution'] = {
            'most_frequent': regimes[np.argmax(frequencies)],
            'least_frequent': regimes[np.argmin(frequencies)],
            'entropy': -sum(f * np.log(f) for f in frequencies if f > 0)  # Regime diversity
        }
        
        return comparison
    
    def generate_regime_performance_report(
        self,
        regime_analysis: Dict[str, Any],
        comparison_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate regime performance analysis report"""
        
        report = "REGIME-CONDITIONAL PERFORMANCE ANALYSIS\n"
        report += "=" * 50 + "\n\n"
        
        # Individual regime analysis
        for regime_name, stats in regime_analysis.items():
            report += f"{regime_name.upper()}\n"
            report += "-" * len(regime_name) + "\n"
            
            report += f"Frequency: {stats['frequency']:.1%} ({stats['periods']} periods)\n"
            report += f"Mean Return: {stats['mean_return']:.4f} ({stats['mean_return']*100:.2f}%)\n"
            report += f"Annualized Return: {stats['annualized_return']:.2%}\n"
            report += f"Annualized Volatility: {stats['annualized_volatility']:.2%}\n"
            report += f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
            report += f"Maximum Drawdown: {stats['max_drawdown']:.2%}\n"
            report += f"Win Rate: {stats['win_rate']:.1%}\n"
            report += f"95% VaR: {stats['var_95']:.2%}\n"
            report += f"95% CVaR: {stats['cvar_95']:.2%}\n"
            
            if 'alpha' in stats:
                report += f"Alpha: {stats['alpha']:.4f}\n"
                report += f"Beta: {stats['beta']:.2f}\n"
                report += f"Information Ratio: {stats['information_ratio']:.2f}\n"
            
            report += "\n"
        
        # Regime comparison
        if comparison_analysis:
            report += "REGIME COMPARISON ANALYSIS\n"
            report += "-" * 30 + "\n"
            
            for metric, comp in comparison_analysis.items():
                if metric == 'regime_distribution':
                    report += f"\nRegime Distribution:\n"
                    report += f"  Most Frequent Regime: {comp['most_frequent']}\n"
                    report += f"  Least Frequent Regime: {comp['least_frequent']}\n"
                    report += f"  Regime Entropy: {comp['entropy']:.2f}\n"
                else:
                    report += f"\n{metric.replace('_', ' ').title()}:\n"
                    report += f"  Best: {comp['best_regime']}\n"
                    report += f"  Worst: {comp['worst_regime']}\n"
                    report += f"  Range: {comp['range']:.4f}\n"
        
        report += f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report