import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.performance_analytics import PerformanceAnalytics
from src.optimization.objectives.risk_measures import RiskMeasures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class InteractivePortfolioAnalytics:
    def __init__(self):
        self.performance_analytics = PerformanceAnalytics()
        self.risk_measures = RiskMeasures()
        self.colors = px.colors.qualitative.Set1
        
    def create_portfolio_heatmap(
        self,
        returns_data: pd.DataFrame,
        correlation_window: int = 60
    ) -> go.Figure:
        """Create interactive correlation heatmap"""
        logger.info("Creating portfolio correlation heatmap")
        
        if len(returns_data) < correlation_window:
            correlation_window = len(returns_data)
        
        recent_returns = returns_data.tail(correlation_window)
        correlation_matrix = recent_returns.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        correlation_matrix_masked = correlation_matrix.mask(mask)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix_masked.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdYlBu_r',
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix_masked.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Asset Correlation Matrix ({correlation_window}-day window)",
            xaxis_title="Assets",
            yaxis_title="Assets",
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def create_rolling_metrics_dashboard(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        window: int = 60
    ) -> go.Figure:
        """Create comprehensive rolling metrics dashboard"""
        logger.info("Creating rolling metrics dashboard")
        
        rolling_metrics = self.performance_analytics.analyze_rolling_performance(
            portfolio_returns, window, ['return', 'volatility', 'sharpe', 'max_drawdown', 'var']
        )
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f'Rolling {window}D Return', f'Rolling {window}D Volatility',
                f'Rolling {window}D Sharpe Ratio', f'Rolling {window}D Max Drawdown',
                f'Rolling {window}D VaR (95%)', 'Excess Return vs Benchmark'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Rolling Return
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['annualized_return'] * 100,
                mode='lines',
                name='Rolling Return',
                line=dict(color=self.colors[0], width=2)
            ),
            row=1, col=1
        )
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_volatility'] * 100,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color=self.colors[1], width=2)
            ),
            row=1, col=2
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_sharpe'],
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.colors[2], width=2)
            ),
            row=2, col=1
        )
        
        # Add zero line for Sharpe ratio
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        # Rolling Max Drawdown
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_max_drawdown'] * 100,
                mode='lines',
                name='Rolling Max DD',
                line=dict(color=self.colors[3], width=2),
                fill='tonexty'
            ),
            row=2, col=2
        )
        
        # Rolling VaR
        fig.add_trace(
            go.Scatter(
                x=rolling_metrics.index,
                y=rolling_metrics['rolling_var_95'] * 100,
                mode='lines',
                name='Rolling VaR 95%',
                line=dict(color=self.colors[4], width=2),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        # Excess Return vs Benchmark
        if benchmark_returns is not None:
            excess_returns = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index).fillna(0)
            cumulative_excess = excess_returns.cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_excess.index,
                    y=cumulative_excess * 100,
                    mode='lines',
                    name='Cumulative Excess Return',
                    line=dict(color=self.colors[5], width=2)
                ),
                row=3, col=2
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=3, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Rolling Performance Metrics Dashboard",
            showlegend=False,
            template='plotly_white'
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=2)
        fig.update_yaxes(title_text="VaR (%)", row=3, col=1)
        fig.update_yaxes(title_text="Excess Return (%)", row=3, col=2)
        
        return fig
    
    def create_regime_performance_comparison(
        self,
        portfolio_returns: pd.Series,
        regime_history: pd.Series,
        regime_names: Dict[int, str]
    ) -> go.Figure:
        """Create regime-based performance comparison"""
        logger.info("Creating regime performance comparison")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Returns by Regime', 'Volatility by Regime',
                'Sharpe Ratio by Regime', 'Win Rate by Regime'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        regime_stats = {}
        
        for regime_id in regime_history.unique():
            if pd.isna(regime_id):
                continue
                
            regime_mask = regime_history == regime_id
            regime_returns = portfolio_returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_stats[regime_id] = {
                    'name': regime_names.get(regime_id, f'Regime {regime_id}'),
                    'annualized_return': regime_returns.mean() * 252,
                    'annualized_volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'win_rate': (regime_returns > 0).mean(),
                    'frequency': len(regime_returns) / len(portfolio_returns)
                }
        
        if not regime_stats:
            return go.Figure().add_annotation(text="No regime data available", showarrow=False)
        
        regime_ids = list(regime_stats.keys())
        names = [regime_stats[rid]['name'] for rid in regime_ids]
        colors = [self.colors[i % len(self.colors)] for i in range(len(regime_ids))]
        
        # Annualized Returns
        returns = [regime_stats[rid]['annualized_return'] * 100 for rid in regime_ids]
        fig.add_trace(
            go.Bar(x=names, y=returns, name='Returns', marker_color=colors),
            row=1, col=1
        )
        
        # Volatility
        volatilities = [regime_stats[rid]['annualized_volatility'] * 100 for rid in regime_ids]
        fig.add_trace(
            go.Bar(x=names, y=volatilities, name='Volatility', marker_color=colors),
            row=1, col=2
        )
        
        # Sharpe Ratio
        sharpe_ratios = [regime_stats[rid]['sharpe_ratio'] for rid in regime_ids]
        fig.add_trace(
            go.Bar(x=names, y=sharpe_ratios, name='Sharpe', marker_color=colors),
            row=2, col=1
        )
        
        # Win Rate
        win_rates = [regime_stats[rid]['win_rate'] * 100 for rid in regime_ids]
        fig.add_trace(
            go.Bar(x=names, y=win_rates, name='Win Rate', marker_color=colors),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Performance Analysis by Market Regime",
            showlegend=False,
            template='plotly_white'
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxis(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)
        
        return fig
    
    def create_portfolio_composition_timeline(
        self,
        weights_history: pd.DataFrame,
        top_n: int = 10
    ) -> go.Figure:
        """Create portfolio composition over time"""
        logger.info("Creating portfolio composition timeline")
        
        if weights_history.empty:
            return go.Figure().add_annotation(text="No portfolio weights history available", showarrow=False)
        
        # Get top N assets by average weight
        avg_weights = weights_history.mean().sort_values(ascending=False)
        top_assets = avg_weights.head(top_n).index
        
        # Group remaining assets as "Others"
        other_assets = avg_weights.tail(-top_n).index
        weights_history_display = weights_history[top_assets].copy()
        
        if len(other_assets) > 0:
            weights_history_display['Others'] = weights_history[other_assets].sum(axis=1)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(weights_history_display.columns)]
        
        for i, asset in enumerate(weights_history_display.columns):
            fig.add_trace(
                go.Scatter(
                    x=weights_history_display.index,
                    y=weights_history_display[asset] * 100,
                    mode='lines',
                    name=asset,
                    stackgroup='one',
                    line=dict(width=0.5, color=colors[i]),
                    fillcolor=colors[i]
                )
            )
        
        fig.update_layout(
            title="Portfolio Composition Over Time",
            xaxis_title="Date",
            yaxis_title="Weight (%)",
            hovermode='x unified',
            template='plotly_white',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_risk_contribution_analysis(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> go.Figure:
        """Create risk contribution analysis"""
        logger.info("Creating risk contribution analysis")
        
        portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
        marginal_contrib = (covariance_matrix @ weights) / portfolio_vol
        contrib = weights * marginal_contrib
        contrib_pct = contrib / contrib.sum()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Risk Contribution ($)', 'Risk Contribution (%)'],
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart of absolute risk contributions
        fig.add_trace(
            go.Bar(
                x=contrib.index,
                y=contrib.values,
                name='Risk Contribution',
                marker_color=self.colors[0]
            ),
            row=1, col=1
        )
        
        # Pie chart of percentage risk contributions
        fig.add_trace(
            go.Pie(
                labels=contrib_pct.index,
                values=contrib_pct.values,
                name='Risk %',
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3[:len(contrib_pct)])
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Portfolio Risk Contribution Analysis",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_performance_attribution_waterfall(
        self,
        attribution_data: Dict[str, float],
        title: str = "Performance Attribution"
    ) -> go.Figure:
        """Create waterfall chart for performance attribution"""
        logger.info("Creating performance attribution waterfall")
        
        if not attribution_data:
            return go.Figure().add_annotation(text="No attribution data available", showarrow=False)
        
        categories = list(attribution_data.keys())
        values = [attribution_data[cat] * 100 for cat in categories]  # Convert to percentage
        
        # Calculate cumulative values for waterfall
        cumulative = [0]
        for value in values:
            cumulative.append(cumulative[-1] + value)
        
        fig = go.Figure()
        
        # Add bars for each attribution component
        for i, (category, value) in enumerate(zip(categories, values)):
            color = 'green' if value >= 0 else 'red'
            
            fig.add_trace(
                go.Bar(
                    x=[category],
                    y=[value],
                    name=category,
                    marker_color=color,
                    text=f"{value:.2f}%",
                    textposition='outside'
                )
            )
        
        # Add cumulative line
        fig.add_trace(
            go.Scatter(
                x=categories + ['Total'],
                y=cumulative,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='blue', width=2, dash='dash'),
                marker=dict(size=8)
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Attribution Component",
            yaxis_title="Contribution (%)",
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_drawdown_periods_analysis(
        self,
        portfolio_returns: pd.Series,
        threshold: float = 0.05
    ) -> go.Figure:
        """Create detailed drawdown periods analysis"""
        logger.info("Creating drawdown periods analysis")
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / rolling_max - 1)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Cumulative Returns & Drawdowns', 'Drawdown Distribution'],
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdowns
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values * 100,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)',
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Drawdown distribution histogram
        fig.add_trace(
            go.Histogram(
                x=drawdowns.values * 100,
                nbinsx=50,
                name='Drawdown Distribution',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_hline(
            y=-threshold * 100,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold: {threshold:.1%}",
            row=1, col=1
        )
        
        fig.update_layout(
            title_text="Portfolio Drawdown Analysis",
            template='plotly_white',
            height=600
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_monte_carlo_simulation_results(
        self,
        simulation_results: Dict[str, Any],
        n_paths_display: int = 100
    ) -> go.Figure:
        """Create Monte Carlo simulation visualization"""
        logger.info("Creating Monte Carlo simulation results")
        
        if 'paths' not in simulation_results:
            return go.Figure().add_annotation(text="No simulation data available", showarrow=False)
        
        paths = simulation_results['paths']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Simulated Price Paths', 'Final Return Distribution',
                'Value at Risk Analysis', 'Return Statistics'
            ],
            specs=[[{"secondary_y": False}, {"type": "histogram"}],
                   [{"type": "box"}, {"type": "table"}]]
        )
        
        # Display subset of paths for performance
        n_display = min(n_paths_display, paths.shape[0])
        selected_paths = paths[:n_display]
        
        for i in range(n_display):
            fig.add_trace(
                go.Scatter(
                    y=selected_paths[i],
                    mode='lines',
                    line=dict(width=1, color='blue'),
                    opacity=0.3,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Mean path
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(
            go.Scatter(
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(width=3, color='red')
            ),
            row=1, col=1
        )
        
        # Final return distribution
        final_returns = paths[:, -1]
        fig.add_trace(
            go.Histogram(
                x=final_returns * 100,
                nbinsx=50,
                name='Final Returns',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # VaR analysis
        var_levels = [0.01, 0.05, 0.10]
        var_values = [np.percentile(final_returns, level * 100) for level in var_levels]
        
        fig.add_trace(
            go.Box(
                y=final_returns * 100,
                name='Return Distribution',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Statistics table
        stats = [
            ['Mean Return', f"{simulation_results.get('mean_final_return', 0):.2%}"],
            ['Std Deviation', f"{simulation_results.get('std_final_return', 0):.2%}"],
            ['VaR (5%)', f"{simulation_results.get('var_5%', 0):.2%}"],
            ['VaR (1%)', f"{simulation_results.get('var_1%', 0):.2%}"],
            ['Prob of Loss', f"{simulation_results.get('probability_of_loss', 0):.1%}"],
            ['Max Loss', f"{simulation_results.get('max_simulated_loss', 0):.2%}"],
            ['Max Gain', f"{simulation_results.get('max_simulated_gain', 0):.2%}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*stats)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Monte Carlo Simulation Results",
            template='plotly_white',
            height=800
        )
        
        return fig


def create_interactive_analytics() -> InteractivePortfolioAnalytics:
    """Factory function to create analytics instance"""
    return InteractivePortfolioAnalytics()