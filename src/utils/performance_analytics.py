import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from src.optimization.objectives.risk_measures import RiskMeasures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PerformanceAnalytics:
    def __init__(self):
        self.risk_measures = RiskMeasures()
        
    def calculate_comprehensive_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        logger.info("Calculating comprehensive performance metrics")
        
        metrics = {}
        
        metrics['total_return'] = (1 + portfolio_returns).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (periods_per_year / len(portfolio_returns)) - 1
        
        metrics['volatility'] = portfolio_returns.std()
        metrics['annualized_volatility'] = metrics['volatility'] * np.sqrt(periods_per_year)
        
        excess_returns = portfolio_returns - risk_free_rate / periods_per_year
        metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year) if excess_returns.std() > 0 else 0
        
        metrics['sortino_ratio'] = self.risk_measures.sortino_ratio(portfolio_returns, risk_free_rate, periods_per_year)
        
        max_dd, start_idx, end_idx = self.risk_measures.maximum_drawdown(portfolio_returns)
        metrics['max_drawdown'] = max_dd
        
        metrics['calmar_ratio'] = self.risk_measures.calmar_ratio(portfolio_returns, periods_per_year)
        
        metrics['var_95'] = self.risk_measures.value_at_risk(portfolio_returns, 0.05)
        metrics['cvar_95'] = self.risk_measures.conditional_value_at_risk(portfolio_returns, 0.05)
        
        metrics['skewness'] = portfolio_returns.skew()
        metrics['kurtosis'] = portfolio_returns.kurtosis()
        
        metrics['omega_ratio'] = self.risk_measures.omega_ratio(portfolio_returns)
        metrics['tail_ratio'] = self.risk_measures.tail_ratio(portfolio_returns)
        
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
            
            metrics['alpha'] = metrics['annualized_return'] - ((1 + aligned_benchmark).prod() ** (periods_per_year / len(aligned_benchmark)) - 1)
            
            excess_vs_benchmark = portfolio_returns - aligned_benchmark
            metrics['information_ratio'] = excess_vs_benchmark.mean() / excess_vs_benchmark.std() * np.sqrt(periods_per_year) if excess_vs_benchmark.std() > 0 else 0
            
            metrics['tracking_error'] = excess_vs_benchmark.std() * np.sqrt(periods_per_year)
            
            metrics['beta'] = np.cov(portfolio_returns, aligned_benchmark)[0, 1] / np.var(aligned_benchmark) if np.var(aligned_benchmark) > 0 else 0
            
            metrics['correlation'] = np.corrcoef(portfolio_returns, aligned_benchmark)[0, 1]
            
            up_periods = aligned_benchmark > 0
            down_periods = aligned_benchmark < 0
            
            if up_periods.sum() > 0:
                up_capture_port = (1 + portfolio_returns[up_periods]).prod() - 1
                up_capture_bench = (1 + aligned_benchmark[up_periods]).prod() - 1
                metrics['up_capture'] = up_capture_port / up_capture_bench if up_capture_bench != 0 else 0
            else:
                metrics['up_capture'] = 0
                
            if down_periods.sum() > 0:
                down_capture_port = (1 + portfolio_returns[down_periods]).prod() - 1
                down_capture_bench = (1 + aligned_benchmark[down_periods]).prod() - 1
                metrics['down_capture'] = down_capture_port / down_capture_bench if down_capture_bench != 0 else 0
            else:
                metrics['down_capture'] = 0
        
        metrics['win_rate'] = (portfolio_returns > 0).mean()
        metrics['average_win'] = portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).any() else 0
        metrics['average_loss'] = portfolio_returns[portfolio_returns < 0].mean() if (portfolio_returns < 0).any() else 0
        metrics['win_loss_ratio'] = abs(metrics['average_win'] / metrics['average_loss']) if metrics['average_loss'] != 0 else np.inf
        
        return metrics
    
    def analyze_rolling_performance(
        self,
        portfolio_returns: pd.Series,
        window: int = 252,
        metrics: List[str] = ['return', 'volatility', 'sharpe']
    ) -> pd.DataFrame:
        logger.info(f"Calculating rolling performance metrics with {window}-day window")
        
        rolling_metrics = pd.DataFrame(index=portfolio_returns.index)
        
        if 'return' in metrics:
            rolling_metrics['rolling_return'] = portfolio_returns.rolling(window).apply(
                lambda x: (1 + x).prod() - 1
            )
            rolling_metrics['annualized_return'] = (1 + rolling_metrics['rolling_return']) ** (252 / window) - 1
        
        if 'volatility' in metrics:
            rolling_metrics['rolling_volatility'] = portfolio_returns.rolling(window).std() * np.sqrt(252)
        
        if 'sharpe' in metrics:
            rolling_metrics['rolling_sharpe'] = (
                portfolio_returns.rolling(window).mean() / portfolio_returns.rolling(window).std() * np.sqrt(252)
            )
        
        if 'max_drawdown' in metrics:
            rolling_metrics['rolling_max_drawdown'] = portfolio_returns.rolling(window).apply(
                lambda x: self.risk_measures.maximum_drawdown(x)[0]
            )
        
        if 'var' in metrics:
            rolling_metrics['rolling_var_95'] = portfolio_returns.rolling(window).apply(
                lambda x: self.risk_measures.value_at_risk(x, 0.05)
            )
        
        return rolling_metrics.dropna()
    
    def calculate_factor_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        logger.info("Calculating factor exposures and attribution")
        
        from sklearn.linear_model import LinearRegression
        
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        
        if len(common_dates) < 30:
            logger.warning("Insufficient data for factor analysis")
            return {}
        
        y = portfolio_returns.loc[common_dates].values.reshape(-1, 1)
        X = factor_returns.loc[common_dates].values
        
        model = LinearRegression().fit(X, y.ravel())
        
        factor_exposures = dict(zip(factor_returns.columns, model.coef_))
        
        predictions = model.predict(X)
        residuals = y.ravel() - predictions
        
        r_squared = model.score(X, y.ravel())
        
        factor_contributions = {}
        for i, factor in enumerate(factor_returns.columns):
            factor_contribution = model.coef_[i] * factor_returns.loc[common_dates, factor]
            factor_contributions[factor] = factor_contribution.sum()
        
        return {
            'factor_exposures': factor_exposures,
            'alpha': model.intercept_,
            'r_squared': r_squared,
            'factor_contributions': factor_contributions,
            'residual_risk': np.std(residuals),
            'tracking_error_explained': np.sqrt(r_squared) if r_squared > 0 else 0
        }
    
    def analyze_drawdown_periods(
        self,
        portfolio_returns: pd.Series,
        threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        logger.info("Analyzing drawdown periods")
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / rolling_max - 1)
        
        drawdown_periods = []
        in_drawdown = False
        current_period = {}
        
        for date, dd in drawdowns.items():
            if dd <= -threshold and not in_drawdown:
                in_drawdown = True
                current_period = {
                    'start_date': date,
                    'peak_value': cumulative_returns[date],
                    'trough_value': cumulative_returns[date],
                    'trough_date': date
                }
            
            elif in_drawdown:
                if cumulative_returns[date] < current_period['trough_value']:
                    current_period['trough_value'] = cumulative_returns[date]
                    current_period['trough_date'] = date
                
                if abs(dd) < 0.001:
                    current_period['end_date'] = date
                    current_period['recovery_value'] = cumulative_returns[date]
                    current_period['max_drawdown'] = (current_period['trough_value'] / current_period['peak_value'] - 1)
                    current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
                    current_period['recovery_days'] = (current_period['end_date'] - current_period['trough_date']).days
                    
                    drawdown_periods.append(current_period)
                    in_drawdown = False
        
        if in_drawdown:
            current_period['end_date'] = drawdowns.index[-1]
            current_period['recovery_value'] = cumulative_returns.iloc[-1]
            current_period['max_drawdown'] = (current_period['trough_value'] / current_period['peak_value'] - 1)
            current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
            current_period['recovery_days'] = None
            drawdown_periods.append(current_period)
        
        return drawdown_periods
    
    def generate_performance_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02
    ) -> str:
        logger.info("Generating comprehensive performance report")
        
        report = "PORTFOLIO PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Analysis Period: {portfolio_returns.index[0].strftime('%Y-%m-%d')} to {portfolio_returns.index[-1].strftime('%Y-%m-%d')}\n"
        report += f"Total Observations: {len(portfolio_returns)}\n\n"
        
        metrics = self.calculate_comprehensive_metrics(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        
        report += "RETURN METRICS:\n"
        report += "-" * 20 + "\n"
        report += f"Total Return: {metrics['total_return']:.2%}\n"
        report += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
        report += f"Average Win: {metrics['average_win']:.2%}\n"
        report += f"Average Loss: {metrics['average_loss']:.2%}\n"
        report += f"Win Rate: {metrics['win_rate']:.1%}\n"
        report += f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}\n\n"
        
        report += "RISK METRICS:\n"
        report += "-" * 20 + "\n"
        report += f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n"
        report += f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
        report += f"95% VaR: {metrics['var_95']:.2%}\n"
        report += f"95% CVaR: {metrics['cvar_95']:.2%}\n"
        report += f"Skewness: {metrics['skewness']:.2f}\n"
        report += f"Kurtosis: {metrics['kurtosis']:.2f}\n\n"
        
        report += "RISK-ADJUSTED METRICS:\n"
        report += "-" * 25 + "\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        report += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
        report += f"Calmar Ratio: {metrics['calmar_ratio']:.2f}\n"
        report += f"Omega Ratio: {metrics['omega_ratio']:.2f}\n"
        report += f"Tail Ratio: {metrics['tail_ratio']:.2f}\n\n"
        
        if benchmark_returns is not None:
            report += "BENCHMARK COMPARISON:\n"
            report += "-" * 25 + "\n"
            report += f"Alpha: {metrics['alpha']:.2%}\n"
            report += f"Beta: {metrics['beta']:.2f}\n"
            report += f"Information Ratio: {metrics['information_ratio']:.2f}\n"
            report += f"Tracking Error: {metrics['tracking_error']:.2%}\n"
            report += f"Correlation: {metrics['correlation']:.2f}\n"
            report += f"Up Capture: {metrics['up_capture']:.1%}\n"
            report += f"Down Capture: {metrics['down_capture']:.1%}\n\n"
        
        drawdown_periods = self.analyze_drawdown_periods(portfolio_returns)
        if drawdown_periods:
            report += "MAJOR DRAWDOWN PERIODS:\n"
            report += "-" * 25 + "\n"
            for i, dd in enumerate(drawdown_periods[:5]):
                report += f"Drawdown {i+1}:\n"
                report += f"  Period: {dd['start_date'].strftime('%Y-%m-%d')} to {dd['end_date'].strftime('%Y-%m-%d')}\n"
                report += f"  Max Drawdown: {dd['max_drawdown']:.2%}\n"
                report += f"  Duration: {dd['duration_days']} days\n"
                if dd.get('recovery_days'):
                    report += f"  Recovery: {dd['recovery_days']} days\n"
                report += "\n"
        
        if factor_returns is not None:
            factor_analysis = self.calculate_factor_exposures(portfolio_returns, factor_returns)
            if factor_analysis:
                report += "FACTOR EXPOSURE ANALYSIS:\n"
                report += "-" * 25 + "\n"
                report += f"Alpha: {factor_analysis['alpha']:.4f}\n"
                report += f"R-squared: {factor_analysis['r_squared']:.2%}\n"
                report += f"Residual Risk: {factor_analysis['residual_risk']:.4f}\n\n"
                
                report += "Factor Exposures:\n"
                for factor, exposure in factor_analysis['factor_exposures'].items():
                    report += f"  {factor}: {exposure:.3f}\n"
        
        return report


class PerformanceVisualizer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_cumulative_returns(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Cumulative Returns"
    ) -> go.Figure:
        fig = go.Figure()
        
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_portfolio.index,
                y=cumulative_portfolio.values,
                mode='lines',
                name='Portfolio',
                line=dict(color=self.colors[0], width=2)
            )
        )
        
        if benchmark_returns is not None:
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=cumulative_benchmark.index,
                    y=cumulative_benchmark.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.colors[1], width=2)
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_drawdown(
        self,
        portfolio_returns: pd.Series,
        title: str = "Portfolio Drawdown"
    ) -> go.Figure:
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / rolling_max - 1) * 100
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)'
            )
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_rolling_metrics(
        self,
        rolling_metrics: pd.DataFrame,
        title: str = "Rolling Performance Metrics"
    ) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Rolling Return', 'Rolling Volatility', 'Rolling Sharpe', 'Rolling Max Drawdown'],
            vertical_spacing=0.08
        )
        
        metrics = ['annualized_return', 'rolling_volatility', 'rolling_sharpe', 'rolling_max_drawdown']
        
        for i, metric in enumerate(metrics):
            if metric in rolling_metrics.columns:
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_metrics.index,
                        y=rolling_metrics[metric],
                        mode='lines',
                        name=metric,
                        line=dict(color=self.colors[i % len(self.colors)])
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=600,
            title_text=title,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_risk_return_scatter(
        self,
        returns_data: Dict[str, pd.Series],
        title: str = "Risk-Return Scatter"
    ) -> go.Figure:
        fig = go.Figure()
        
        for name, returns in returns_data.items():
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(
                    x=[annual_vol],
                    y=[annual_return],
                    mode='markers',
                    name=name,
                    marker=dict(size=10)
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            template='plotly_white'
        )
        
        return fig
    
    def create_performance_dashboard(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        rolling_window: int = 252
    ) -> Dict[str, go.Figure]:
        dashboard = {}
        
        dashboard['cumulative_returns'] = self.plot_cumulative_returns(
            portfolio_returns, benchmark_returns
        )
        
        dashboard['drawdown'] = self.plot_drawdown(portfolio_returns)
        
        analytics = PerformanceAnalytics()
        rolling_metrics = analytics.analyze_rolling_performance(
            portfolio_returns, rolling_window
        )
        
        dashboard['rolling_metrics'] = self.plot_rolling_metrics(rolling_metrics)
        
        if portfolio_weights is not None and len(portfolio_weights.columns) > 1:
            latest_weights = portfolio_weights.iloc[-1]
            top_positions = latest_weights.nlargest(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_positions.index,
                    y=top_positions.values,
                    marker_color=self.colors[0]
                )
            ])
            
            fig.update_layout(
                title="Top 10 Portfolio Positions",
                xaxis_title="Asset",
                yaxis_title="Weight",
                template='plotly_white'
            )
            
            dashboard['positions'] = fig
        
        return dashboard