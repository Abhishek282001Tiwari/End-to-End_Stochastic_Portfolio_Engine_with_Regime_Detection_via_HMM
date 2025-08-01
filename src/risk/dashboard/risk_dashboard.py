import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings

from src.optimization.objectives.risk_measures import (
    RiskMeasures, PortfolioRiskCalculator, RiskMetric, 
    StressTestScenario, RiskDecomposition, RiskMetricType
)
from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor, RiskLimits, RiskAlert
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class DashboardConfig:
    """Configuration for risk dashboard"""
    update_frequency: str = "1H"  # Update frequency
    lookback_periods: int = 252  # Lookback period for calculations
    confidence_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    risk_free_rate: float = 0.02
    benchmark_symbol: str = "SPY"
    stress_test_scenarios: List[StressTestScenario] = field(default_factory=list)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class ComprehensiveRiskDashboard:
    """Comprehensive risk management dashboard with real-time analytics"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.risk_calculator = PortfolioRiskCalculator()
        self.risk_monitor = None
        
        # Dashboard state
        self.current_metrics = {}
        self.metrics_history = []
        self.stress_test_results = {}
        self.scenario_analysis = {}
        
        # Performance tracking
        self.performance_attribution = {}
        self.risk_decomposition = None
        
        logger.info("Comprehensive Risk Dashboard initialized")
    
    def initialize_risk_monitor(self, risk_limits: RiskLimits):
        """Initialize real-time risk monitoring"""
        self.risk_monitor = RealTimeRiskMonitor(risk_limits)
        logger.info("Risk monitoring system initialized")
    
    def update_portfolio_metrics(
        self,
        portfolio_weights: pd.Series,
        returns_data: pd.DataFrame,
        prices_data: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        regime_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Update comprehensive portfolio risk metrics"""
        
        logger.info("Updating portfolio risk metrics")
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data @ portfolio_weights).dropna()
        covariance_matrix = returns_data.cov().values
        
        # Basic risk metrics
        basic_metrics = self._calculate_basic_risk_metrics(
            portfolio_returns, portfolio_weights.values, covariance_matrix
        )
        
        # Advanced risk metrics
        advanced_metrics = self._calculate_advanced_risk_metrics(
            portfolio_returns, benchmark_returns
        )
        
        # Regime-conditional metrics
        if regime_labels is not None:
            regime_metrics = self._calculate_regime_conditional_metrics(
                portfolio_returns, regime_labels
            )
            advanced_metrics.update(regime_metrics)
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **advanced_metrics}
        
        # Calculate risk decomposition
        self.risk_decomposition = self._calculate_risk_decomposition(
            portfolio_weights.values, covariance_matrix, returns_data.columns
        )
        
        # Store metrics with timestamp
        timestamp = datetime.now()
        metric_record = {
            'timestamp': timestamp,
            'metrics': all_metrics,
            'risk_decomposition': self.risk_decomposition
        }
        
        self.metrics_history.append(metric_record)
        self.current_metrics = all_metrics
        
        # Keep only recent history
        cutoff_date = timestamp - timedelta(days=30)
        self.metrics_history = [
            record for record in self.metrics_history 
            if record['timestamp'] > cutoff_date
        ]
        
        return all_metrics
    
    def _calculate_basic_risk_metrics(
        self, 
        portfolio_returns: pd.Series, 
        weights: np.ndarray, 
        covariance_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic risk metrics"""
        
        metrics = {}
        
        # Volatility metrics
        metrics['daily_volatility'] = np.std(portfolio_returns)
        metrics['annualized_volatility'] = metrics['daily_volatility'] * np.sqrt(252)
        
        # VaR and CVaR at multiple confidence levels
        for confidence_level in self.config.confidence_levels:
            var_hist = RiskMeasures.value_at_risk(
                portfolio_returns.values, confidence_level, "historical"
            )
            var_param = RiskMeasures.value_at_risk(
                portfolio_returns.values, confidence_level, "parametric"
            )
            var_cf = RiskMeasures.value_at_risk(
                portfolio_returns.values, confidence_level, "cornish_fisher"
            )
            cvar = RiskMeasures.conditional_value_at_risk(
                portfolio_returns.values, confidence_level
            )
            
            level_pct = int(confidence_level * 100)
            metrics[f'var_{level_pct}%_historical'] = var_hist
            metrics[f'var_{level_pct}%_parametric'] = var_param
            metrics[f'var_{level_pct}%_cornish_fisher'] = var_cf
            metrics[f'cvar_{level_pct}%'] = cvar
        
        # Drawdown metrics
        max_dd, start_idx, end_idx = RiskMeasures.maximum_drawdown(portfolio_returns.values)
        metrics['maximum_drawdown'] = max_dd
        metrics['pain_index'] = RiskMeasures.pain_index(portfolio_returns.values)
        metrics['ulcer_index'] = RiskMeasures.ulcer_index(portfolio_returns.values)
        metrics['conditional_drawdown_5%'] = RiskMeasures.conditional_drawdown_at_risk(
            portfolio_returns.values, 0.05
        )
        
        # Portfolio composition metrics
        metrics['diversification_ratio'] = RiskMeasures.diversification_ratio(
            weights, covariance_matrix
        )
        metrics['effective_number_of_assets'] = RiskMeasures.effective_number_of_assets(weights)
        metrics['concentration_index'] = np.sum(weights ** 2)  # Herfindahl index
        
        return metrics
    
    def _calculate_advanced_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate advanced risk metrics"""
        
        metrics = {}
        returns_array = portfolio_returns.values
        
        # Performance ratios
        metrics['calmar_ratio'] = RiskMeasures.calmar_ratio(returns_array)
        metrics['sortino_ratio'] = RiskMeasures.sortino_ratio(
            returns_array, self.config.risk_free_rate / 252
        )
        metrics['omega_ratio'] = RiskMeasures.omega_ratio(returns_array)
        metrics['tail_ratio'] = RiskMeasures.tail_ratio(returns_array)
        metrics['burke_ratio'] = RiskMeasures.burke_ratio(
            returns_array, self.config.risk_free_rate
        )
        
        # Return distribution metrics
        metrics['skewness'] = float(portfolio_returns.skew())
        metrics['kurtosis'] = float(portfolio_returns.kurtosis())
        metrics['win_rate'] = RiskMeasures.win_rate(returns_array)
        metrics['gain_loss_ratio'] = RiskMeasures.gain_loss_ratio(returns_array)
        
        # Upside/downside metrics
        metrics['upside_deviation'] = RiskMeasures.upside_deviation(returns_array)
        metrics['downside_deviation'] = RiskMeasures.semi_deviation(returns_array)
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).dropna()
            aligned_portfolio = portfolio_returns.reindex(aligned_benchmark.index)
            
            if len(aligned_portfolio) > 0:
                excess_returns = aligned_portfolio - aligned_benchmark
                
                metrics['tracking_error'] = np.std(excess_returns)
                metrics['information_ratio'] = (
                    np.mean(excess_returns) / np.std(excess_returns) 
                    if np.std(excess_returns) > 0 else 0
                )
                
                # Beta calculation
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                # Alpha calculation
                risk_free_daily = self.config.risk_free_rate / 252
                portfolio_alpha = (
                    np.mean(aligned_portfolio) - risk_free_daily - 
                    metrics['beta'] * (np.mean(aligned_benchmark) - risk_free_daily)
                )
                metrics['alpha'] = portfolio_alpha * 252  # Annualized alpha
        
        return metrics
    
    def _calculate_regime_conditional_metrics(
        self,
        portfolio_returns: pd.Series,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate risk metrics conditional on market regimes"""
        
        regime_metrics = {}
        returns_array = portfolio_returns.values
        
        # Align regime labels with returns
        min_length = min(len(returns_array), len(regime_labels))
        aligned_returns = returns_array[:min_length]
        aligned_regimes = regime_labels[:min_length]
        
        # Calculate regime-conditional VaR
        regime_vars = RiskMeasures.regime_conditional_var(
            aligned_returns, aligned_regimes, 0.05
        )
        regime_metrics['regime_conditional_var_5%'] = regime_vars
        
        # Calculate metrics for each regime
        unique_regimes = np.unique(aligned_regimes)
        regime_stats = {}
        
        for regime in unique_regimes:
            regime_returns = aligned_returns[aligned_regimes == regime]
            
            if len(regime_returns) > 10:  # Minimum sample size
                regime_stats[f'regime_{regime}'] = {
                    'mean_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'var_5%': RiskMeasures.value_at_risk(regime_returns, 0.05),
                    'max_drawdown': RiskMeasures.maximum_drawdown(regime_returns)[0],
                    'win_rate': RiskMeasures.win_rate(regime_returns),
                    'sample_size': len(regime_returns)
                }
        
        regime_metrics['regime_statistics'] = regime_stats
        
        return regime_metrics
    
    def _calculate_risk_decomposition(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        asset_names: List[str]
    ) -> RiskDecomposition:
        """Calculate comprehensive risk decomposition"""
        
        # Portfolio variance
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Risk contributions
        marginal_contrib = covariance_matrix @ weights / portfolio_vol
        component_contrib = weights * marginal_contrib
        percentage_contrib = component_contrib / portfolio_vol
        
        # Risk budget utilization (assuming equal risk budget)
        n_assets = len(weights)
        target_risk_contrib = np.ones(n_assets) / n_assets
        risk_budget_util = percentage_contrib / target_risk_contrib
        
        # Concentration metrics
        concentration_metrics = {
            'herfindahl_index': np.sum(percentage_contrib ** 2),
            'effective_number_of_bets': 1 / np.sum(percentage_contrib ** 2),
            'max_component_contribution': np.max(percentage_contrib),
            'top_3_concentration': np.sum(np.sort(percentage_contrib)[-3:])
        }
        
        return RiskDecomposition(
            total_risk=portfolio_vol,
            marginal_contributions=marginal_contrib,
            component_contributions=component_contrib,
            percentage_contributions=percentage_contrib,
            risk_budget_utilization=risk_budget_util,
            concentration_metrics=concentration_metrics
        )
    
    def run_stress_tests(
        self,
        portfolio_weights: pd.Series,
        returns_data: pd.DataFrame,
        scenarios: Optional[List[StressTestScenario]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive stress tests"""
        
        logger.info("Running portfolio stress tests")
        
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        stress_results = {}
        
        for scenario in scenarios:
            # Apply shocks to returns
            shocked_returns = returns_data.copy()
            
            for asset, shock in scenario.shocks.items():
                if asset in shocked_returns.columns:
                    shocked_returns[asset] = shocked_returns[asset] + shock
            
            # Calculate stressed portfolio metrics
            stressed_portfolio_returns = (shocked_returns @ portfolio_weights).dropna()
            
            scenario_results = {
                'portfolio_return': np.sum(shocked_returns.iloc[-1] * portfolio_weights),
                'portfolio_var_5%': RiskMeasures.value_at_risk(
                    stressed_portfolio_returns.values, 0.05
                ),
                'portfolio_volatility': np.std(stressed_portfolio_returns),
                'max_drawdown': RiskMeasures.maximum_drawdown(
                    stressed_portfolio_returns.values
                )[0]
            }
            
            stress_results[scenario.name] = scenario_results
        
        self.stress_test_results = stress_results
        return stress_results
    
    def _get_default_stress_scenarios(self) -> List[StressTestScenario]:
        """Get default stress test scenarios"""
        
        scenarios = [
            StressTestScenario(
                name="Market Crash",
                description="2008-style market crash scenario",
                shocks={"AAPL": -0.30, "GOOGL": -0.35, "MSFT": -0.25, "SPY": -0.40}
            ),
            StressTestScenario(
                name="Tech Selloff",
                description="Technology sector selloff",
                shocks={"AAPL": -0.20, "GOOGL": -0.25, "MSFT": -0.22, "TSLA": -0.30}
            ),
            StressTestScenario(
                name="Interest Rate Shock",
                description="Sharp rise in interest rates",
                shocks={"TLT": -0.15, "JPM": 0.05, "V": -0.10}
            ),
            StressTestScenario(
                name="Inflation Spike",
                description="Unexpected inflation surge",
                shocks={"GLD": 0.10, "TLT": -0.20, "SPY": -0.15}
            ),
            StressTestScenario(
                name="Flight to Quality",
                description="Risk-off market environment",
                shocks={"SPY": -0.25, "QQQ": -0.30, "TLT": 0.15, "GLD": 0.08}
            )
        ]
        
        return scenarios
    
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk report"""
        
        if not self.current_metrics:
            return "No risk metrics available. Please update portfolio metrics first."
        
        report = "COMPREHENSIVE PORTFOLIO RISK REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Executive Summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 20 + "\n"
        report += f"Portfolio Volatility (Annualized): {self.current_metrics.get('annualized_volatility', 0):.2%}\n"
        report += f"95% VaR (Daily): {abs(self.current_metrics.get('var_5%_historical', 0)):.2%}\n"
        report += f"95% CVaR (Daily): {abs(self.current_metrics.get('cvar_5%', 0)):.2%}\n"
        report += f"Maximum Drawdown: {abs(self.current_metrics.get('maximum_drawdown', 0)):.2%}\n"
        report += f"Diversification Ratio: {self.current_metrics.get('diversification_ratio', 0):.2f}\n"
        report += f"Effective Number of Assets: {self.current_metrics.get('effective_number_of_assets', 0):.1f}\n\n"
        
        # Risk Metrics Detail
        report += "DETAILED RISK METRICS\n"
        report += "-" * 25 + "\n"
        
        # VaR Analysis
        report += "Value at Risk Analysis:\n"
        for confidence in [1, 5, 10]:
            hist_var = self.current_metrics.get(f'var_{confidence}%_historical', 0)
            param_var = self.current_metrics.get(f'var_{confidence}%_parametric', 0)
            cvar = self.current_metrics.get(f'cvar_{confidence}%', 0)
            
            report += f"  {confidence}% VaR (Historical): {abs(hist_var):.2%}\n"
            report += f"  {confidence}% VaR (Parametric): {abs(param_var):.2%}\n"
            report += f"  {confidence}% CVaR: {abs(cvar):.2%}\n"
        report += "\n"
        
        # Performance Ratios
        report += "Risk-Adjusted Performance:\n"
        report += f"  Sortino Ratio: {self.current_metrics.get('sortino_ratio', 0):.2f}\n"
        report += f"  Calmar Ratio: {self.current_metrics.get('calmar_ratio', 0):.2f}\n"
        report += f"  Omega Ratio: {self.current_metrics.get('omega_ratio', 0):.2f}\n"
        report += f"  Burke Ratio: {self.current_metrics.get('burke_ratio', 0):.2f}\n\n"
        
        # Distribution Analysis
        report += "Return Distribution:\n"
        report += f"  Skewness: {self.current_metrics.get('skewness', 0):.2f}\n"
        report += f"  Kurtosis: {self.current_metrics.get('kurtosis', 0):.2f}\n"
        report += f"  Win Rate: {self.current_metrics.get('win_rate', 0):.2%}\n"
        report += f"  Gain/Loss Ratio: {self.current_metrics.get('gain_loss_ratio', 0):.2f}\n\n"
        
        # Risk Decomposition
        if self.risk_decomposition:
            report += "RISK DECOMPOSITION\n"
            report += "-" * 20 + "\n"
            report += f"Total Portfolio Risk: {self.risk_decomposition.total_risk:.2%}\n"
            report += f"Concentration Index: {self.risk_decomposition.concentration_metrics['herfindahl_index']:.3f}\n"
            report += f"Effective Number of Bets: {self.risk_decomposition.concentration_metrics['effective_number_of_bets']:.1f}\n"
            report += f"Max Component Contribution: {self.risk_decomposition.concentration_metrics['max_component_contribution']:.2%}\n\n"
        
        # Stress Test Results
        if self.stress_test_results:
            report += "STRESS TEST RESULTS\n"
            report += "-" * 22 + "\n"
            for scenario, results in self.stress_test_results.items():
                report += f"{scenario}:\n"
                report += f"  Portfolio Return: {results['portfolio_return']:.2%}\n"
                report += f"  95% VaR: {abs(results['portfolio_var_5%']):.2%}\n"
                report += f"  Volatility: {results['portfolio_volatility']:.2%}\n"
                report += f"  Max Drawdown: {abs(results['max_drawdown']):.2%}\n\n"
        
        report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    def create_risk_visualization(self) -> Dict[str, go.Figure]:
        """Create comprehensive risk visualizations"""
        
        if not self.metrics_history:
            logger.warning("No metrics history available for visualization")
            return {}
        
        figures = {}
        
        # Extract time series data
        timestamps = [record['timestamp'] for record in self.metrics_history]
        var_5_series = [record['metrics'].get('var_5%_historical', 0) for record in self.metrics_history]
        volatility_series = [record['metrics'].get('annualized_volatility', 0) for record in self.metrics_history]
        drawdown_series = [record['metrics'].get('maximum_drawdown', 0) for record in self.metrics_history]
        
        # 1. Risk Metrics Time Series
        fig_timeseries = make_subplots(
            rows=3, cols=1,
            subplot_titles=['95% VaR', 'Annualized Volatility', 'Maximum Drawdown'],
            vertical_spacing=0.1
        )
        
        fig_timeseries.add_trace(
            go.Scatter(x=timestamps, y=[abs(v) for v in var_5_series], name='95% VaR'),
            row=1, col=1
        )
        fig_timeseries.add_trace(
            go.Scatter(x=timestamps, y=volatility_series, name='Volatility'),
            row=2, col=1
        )
        fig_timeseries.add_trace(
            go.Scatter(x=timestamps, y=[abs(v) for v in drawdown_series], name='Max Drawdown'),
            row=3, col=1
        )
        
        fig_timeseries.update_layout(title="Risk Metrics Time Series", height=800)
        figures['risk_timeseries'] = fig_timeseries
        
        # 2. Risk Decomposition Chart
        if self.risk_decomposition:
            fig_decomp = go.Figure(data=[
                go.Bar(
                    x=list(range(len(self.risk_decomposition.percentage_contributions))),
                    y=self.risk_decomposition.percentage_contributions,
                    name='Risk Contribution'
                )
            ])
            fig_decomp.update_layout(
                title="Risk Contribution by Asset",
                xaxis_title="Asset Index",
                yaxis_title="Risk Contribution (%)"
            )
            figures['risk_decomposition'] = fig_decomp
        
        # 3. Stress Test Results
        if self.stress_test_results:
            scenarios = list(self.stress_test_results.keys())
            returns = [self.stress_test_results[s]['portfolio_return'] for s in scenarios]
            vars = [abs(self.stress_test_results[s]['portfolio_var_5%']) for s in scenarios]
            
            fig_stress = go.Figure(data=[
                go.Bar(x=scenarios, y=returns, name='Stressed Returns'),
                go.Bar(x=scenarios, y=vars, name='Stressed 95% VaR')
            ])
            fig_stress.update_layout(
                title="Stress Test Results",
                xaxis_title="Scenario",
                yaxis_title="Impact (%)"
            )
            figures['stress_tests'] = fig_stress
        
        return figures
    
    def export_metrics_to_excel(self, filename: str = "risk_dashboard.xlsx"):
        """Export risk metrics to Excel file"""
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Current metrics
            if self.current_metrics:
                current_df = pd.DataFrame([self.current_metrics])
                current_df.to_excel(writer, sheet_name='Current_Metrics', index=False)
            
            # Historical metrics
            if self.metrics_history:
                history_data = []
                for record in self.metrics_history:
                    row = {'timestamp': record['timestamp']}
                    row.update(record['metrics'])
                    history_data.append(row)
                
                history_df = pd.DataFrame(history_data)
                history_df.to_excel(writer, sheet_name='Metrics_History', index=False)
            
            # Risk decomposition
            if self.risk_decomposition:
                decomp_data = {
                    'Asset_Index': list(range(len(self.risk_decomposition.percentage_contributions))),
                    'Marginal_Contribution': self.risk_decomposition.marginal_contributions,
                    'Component_Contribution': self.risk_decomposition.component_contributions,
                    'Percentage_Contribution': self.risk_decomposition.percentage_contributions,
                    'Risk_Budget_Utilization': self.risk_decomposition.risk_budget_utilization
                }
                decomp_df = pd.DataFrame(decomp_data)
                decomp_df.to_excel(writer, sheet_name='Risk_Decomposition', index=False)
            
            # Stress test results
            if self.stress_test_results:
                stress_df = pd.DataFrame(self.stress_test_results).T
                stress_df.to_excel(writer, sheet_name='Stress_Tests')
        
        logger.info(f"Risk metrics exported to {filename}")


class RiskDashboardManager:
    """Manager for multiple risk dashboards"""
    
    def __init__(self):
        self.dashboards: Dict[str, ComprehensiveRiskDashboard] = {}
        self.active_dashboard: Optional[str] = None
    
    def create_dashboard(self, name: str, config: DashboardConfig) -> ComprehensiveRiskDashboard:
        """Create a new risk dashboard"""
        dashboard = ComprehensiveRiskDashboard(config)
        self.dashboards[name] = dashboard
        
        if self.active_dashboard is None:
            self.active_dashboard = name
        
        logger.info(f"Created risk dashboard: {name}")
        return dashboard
    
    def get_dashboard(self, name: str) -> Optional[ComprehensiveRiskDashboard]:
        """Get a specific dashboard"""
        return self.dashboards.get(name)
    
    def set_active_dashboard(self, name: str):
        """Set the active dashboard"""
        if name in self.dashboards:
            self.active_dashboard = name
            logger.info(f"Set active dashboard to: {name}")
        else:
            logger.warning(f"Dashboard {name} not found")
    
    def get_active_dashboard(self) -> Optional[ComprehensiveRiskDashboard]:
        """Get the active dashboard"""
        if self.active_dashboard:
            return self.dashboards.get(self.active_dashboard)
        return None
    
    def generate_consolidated_report(self) -> str:
        """Generate consolidated report across all dashboards"""
        if not self.dashboards:
            return "No dashboards available"
        
        report = "CONSOLIDATED RISK DASHBOARD REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for name, dashboard in self.dashboards.items():
            report += f"DASHBOARD: {name.upper()}\n"
            report += "-" * 30 + "\n"
            report += dashboard.generate_risk_report()
            report += "\n" + "=" * 50 + "\n\n"
        
        return report