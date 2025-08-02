import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
import time
import json
from typing import Dict, List, Optional, Any

from src.data.ingestion.data_sources import create_data_pipeline
from src.models.hmm.hmm_engine import RegimeDetectionHMM
from src.models.hmm.regime_analyzer import RegimeAnalyzer
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor, RiskLimits
from src.utils.performance_analytics import PerformanceAnalytics, PerformanceVisualizer
from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PortfolioDashboard:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = get_config()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        self.data_pipeline = create_data_pipeline()
        self.hmm_model = None
        self.portfolio_optimizer = PortfolioOptimizationEngine()
        self.risk_monitor = None
        self.performance_analytics = PerformanceAnalytics()
        self.visualizer = PerformanceVisualizer()
        
        self.current_data = {}
        self.portfolio_state = {}
        self.regime_history = pd.DataFrame()
        self.portfolio_returns = pd.Series()
        self.benchmark_returns = pd.Series()
        
        self._setup_layout()
        self._setup_callbacks()
        
        self.data_update_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        self.data_update_thread.start()
    
    def _setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Stochastic Portfolio Engine", className="text-center mb-4"),
                    html.H4("HMM Regime Detection & Dynamic Allocation", className="text-center text-muted mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Status"),
                        dbc.CardBody([
                            html.Div(id="system-status"),
                            html.Div(id="last-update", className="text-muted small")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Regime"),
                        dbc.CardBody([
                            html.Div(id="current-regime", className="text-center"),
                            html.Div(id="regime-confidence", className="text-center text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Value"),
                        dbc.CardBody([
                            html.Div(id="portfolio-value", className="text-center"),
                            html.Div(id="daily-return", className="text-center")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Alerts"),
                        dbc.CardBody([
                            html.Div(id="risk-alerts")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col("Regime Probabilities", width=8),
                                dbc.Col([
                                    dbc.ButtonGroup([
                                        dbc.Button("1D", id="regime-1d", size="sm", outline=True, color="primary"),
                                        dbc.Button("1W", id="regime-1w", size="sm", outline=True, color="primary"),
                                        dbc.Button("1M", id="regime-1m", size="sm", outline=True, color="primary", active=True),
                                        dbc.Button("3M", id="regime-3m", size="sm", outline=True, color="primary"),
                                    ])
                                ], width=4)
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="regime-probabilities-chart")
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Regime Statistics"),
                        dbc.CardBody([
                            html.Div(id="regime-stats")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col("Portfolio Composition", width=8),
                                dbc.Col([
                                    dbc.Switch(
                                        id="weight-type-switch",
                                        label="Show %",
                                        value=True,
                                        className="float-end"
                                    )
                                ], width=4)
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="portfolio-composition-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Analytics"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col("Risk Dashboard", width=6),
                                dbc.Col([
                                    dbc.Button("Update Risk Limits", id="update-risk-btn", size="sm", color="warning", outline=True)
                                ], width=6)
                            ])
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="risk-metrics-chart")
                                ], width=8),
                                dbc.Col([
                                    html.Div(id="risk-metrics-table")
                                ], width=4)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Attribution"),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab(label="Regime Attribution", tab_id="regime-attribution"),
                                dbc.Tab(label="Factor Attribution", tab_id="factor-attribution"),
                                dbc.Tab(label="Sector Attribution", tab_id="sector-attribution")
                            ], id="attribution-tabs", active_tab="regime-attribution"),
                            html.Div(id="attribution-content")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            ),
            
            dcc.Store(id='portfolio-data-store'),
            dcc.Store(id='regime-data-store'),
            dcc.Store(id='risk-data-store')
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('last-update', 'children'),
             Output('current-regime', 'children'),
             Output('regime-confidence', 'children'),
             Output('portfolio-value', 'children'),
             Output('daily-return', 'children'),
             Output('risk-alerts', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_status_cards(n):
            try:
                status_color = "success" if self.current_data.get('system_healthy', True) else "danger"
                status_text = "Operational" if self.current_data.get('system_healthy', True) else "Issues Detected"
                
                system_status = dbc.Badge(status_text, color=status_color, className="fs-6")
                
                last_update = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
                
                current_regime = self.current_data.get('current_regime_name', 'Unknown')
                regime_confidence = f"Confidence: {self.current_data.get('regime_confidence', 0):.1%}"
                
                portfolio_value = f"${self.current_data.get('portfolio_value', 0):,.2f}"
                daily_return_val = self.current_data.get('daily_return', 0)
                daily_return_color = "success" if daily_return_val >= 0 else "danger"
                daily_return = dbc.Badge(f"{daily_return_val:+.2%}", color=daily_return_color)
                
                alerts = self.current_data.get('risk_alerts', [])
                risk_alerts = []
                for alert in alerts[-3:]:  # Show last 3 alerts
                    color = {
                        'LOW': 'info',
                        'MEDIUM': 'warning', 
                        'HIGH': 'danger',
                        'CRITICAL': 'danger'
                    }.get(alert.get('level', 'LOW'), 'info')
                    
                    risk_alerts.append(
                        dbc.Alert(
                            alert.get('message', ''),
                            color=color,
                            className="small mb-1",
                            dismissable=True
                        )
                    )
                
                return system_status, last_update, current_regime, regime_confidence, portfolio_value, daily_return, risk_alerts
                
            except Exception as e:
                logger.error(f"Error updating status cards: {e}")
                return "Error", "", "", "", "$0.00", "0.00%", []
        
        @self.app.callback(
            Output('regime-probabilities-chart', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('regime-1d', 'n_clicks'),
             Input('regime-1w', 'n_clicks'),
             Input('regime-1m', 'n_clicks'),
             Input('regime-3m', 'n_clicks')]
        )
        def update_regime_chart(n, btn1d, btn1w, btn1m, btn3m):
            try:
                ctx = callback_context
                period = '1M'  # Default
                
                if ctx.triggered:
                    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                    if 'regime-1d' in button_id:
                        period = '1D'
                    elif 'regime-1w' in button_id:
                        period = '1W'
                    elif 'regime-1m' in button_id:
                        period = '1M'
                    elif 'regime-3m' in button_id:
                        period = '3M'
                
                return self._create_regime_probabilities_chart(period)
                
            except Exception as e:
                logger.error(f"Error updating regime chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('portfolio-composition-chart', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('weight-type-switch', 'value')]
        )
        def update_portfolio_composition(n, show_percentage):
            try:
                return self._create_portfolio_composition_chart(show_percentage)
            except Exception as e:
                logger.error(f"Error updating portfolio composition: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(n):
            try:
                return self._create_performance_chart()
            except Exception as e:
                logger.error(f"Error updating performance chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('risk-metrics-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_risk_chart(n):
            try:
                return self._create_risk_metrics_chart()
            except Exception as e:
                logger.error(f"Error updating risk chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('attribution-content', 'children'),
            [Input('attribution-tabs', 'active_tab'),
             Input('interval-component', 'n_intervals')]
        )
        def update_attribution_content(active_tab, n):
            try:
                if active_tab == "regime-attribution":
                    return self._create_regime_attribution_content()
                elif active_tab == "factor-attribution":
                    return self._create_factor_attribution_content()
                else:
                    return self._create_sector_attribution_content()
            except Exception as e:
                logger.error(f"Error updating attribution content: {e}")
                return html.Div("Error loading attribution data")
    
    def _create_regime_probabilities_chart(self, period: str) -> go.Figure:
        if self.regime_history.empty:
            return go.Figure().add_annotation(text="No regime data available", showarrow=False)
        
        # Filter data based on period
        end_date = self.regime_history.index[-1]
        if period == '1D':
            start_date = end_date - timedelta(days=1)
        elif period == '1W':
            start_date = end_date - timedelta(weeks=1)
        elif period == '1M':
            start_date = end_date - timedelta(days=30)
        else:  # 3M
            start_date = end_date - timedelta(days=90)
        
        filtered_data = self.regime_history[self.regime_history.index >= start_date]
        
        fig = go.Figure()
        
        colors = ['#ff6b6b', '#ffa726', '#66bb6a']  # Red, Orange, Green
        regime_names = ['Bear Market', 'Sideways Market', 'Bull Market']
        
        for i, (color, name) in enumerate(zip(colors, regime_names)):
            if f'regime_prob_{i}' in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[f'regime_prob_{i}'],
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2),
                        fill='tonexty' if i > 0 else 'tozeroy'
                    )
                )
        
        fig.update_layout(
            title="Market Regime Probabilities",
            xaxis_title="Date",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_portfolio_composition_chart(self, show_percentage: bool) -> go.Figure:
        weights = self.current_data.get('portfolio_weights', {})
        
        if not weights:
            return go.Figure().add_annotation(text="No portfolio data available", showarrow=False)
        
        assets = list(weights.keys())
        values = list(weights.values())
        
        if show_percentage:
            values = [v * 100 for v in values]
            text = [f"{asset}<br>{value:.1f}%" for asset, value in zip(assets, values)]
        else:
            portfolio_value = self.current_data.get('portfolio_value', 1000000)
            values = [v * portfolio_value for v in values]
            text = [f"{asset}<br>${value:,.0f}" for asset, value in zip(assets, values)]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=assets,
                values=values,
                text=text,
                textinfo='text',
                hole=0.4,
                marker=dict(
                    colors=px.colors.qualitative.Set3[:len(assets)]
                )
            )
        ])
        
        fig.update_layout(
            title="Current Portfolio Allocation",
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def _create_performance_chart(self) -> go.Figure:
        if self.portfolio_returns.empty:
            return go.Figure().add_annotation(text="No performance data available", showarrow=False)
        
        cumulative_portfolio = (1 + self.portfolio_returns).cumprod()
        cumulative_benchmark = (1 + self.benchmark_returns).cumprod() if not self.benchmark_returns.empty else None
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_portfolio.index,
                y=cumulative_portfolio.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            )
        )
        
        if cumulative_benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=cumulative_benchmark.index,
                    y=cumulative_benchmark.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=2, dash='dash')
                )
            )
        
        fig.update_layout(
            title="Cumulative Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_risk_metrics_chart(self) -> go.Figure:
        risk_metrics = self.current_data.get('risk_metrics', {})
        
        if not risk_metrics:
            return go.Figure().add_annotation(text="No risk data available", showarrow=False)
        
        # Create gauge charts for key risk metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Portfolio Volatility', 'VaR (95%)', 'Max Drawdown', 'Sharpe Ratio'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Portfolio Volatility
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics.get('annualized_volatility', 0) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Volatility (%)"},
                gauge={
                    'axis': {'range': [None, 30]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 20], 'color': "yellow"},
                        {'range': [20, 30], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ),
            row=1, col=1
        )
        
        # VaR
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=abs(risk_metrics.get('var_95', 0)) * 100,
                title={'text': "VaR 95% (%)"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgreen"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "red"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Max Drawdown
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=abs(risk_metrics.get('maximum_drawdown', 0)) * 100,
                title={'text': "Max DD (%)"},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 25], 'color': "yellow"},
                        {'range': [25, 50], 'color': "red"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # Sharpe Ratio
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics.get('sharpe_ratio', 0),
                title={'text': "Sharpe Ratio"},
                gauge={
                    'axis': {'range': [-2, 3]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [-2, 0], 'color': "red"},
                        {'range': [0, 1], 'color': "yellow"},
                        {'range': [1, 3], 'color': "lightgreen"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=400, template='plotly_white')
        
        return fig
    
    def _create_regime_attribution_content(self):
        regime_attribution = self.current_data.get('regime_attribution', {})
        
        if not regime_attribution:
            return html.Div("No regime attribution data available")
        
        cards = []
        for regime_id, performance in regime_attribution.items():
            regime_name = performance.get('regime_name', f'Regime {regime_id}')
            
            card = dbc.Card([
                dbc.CardHeader(regime_name),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Frequency: {performance.get('frequency', 0):.1%}", className="mb-1"),
                            html.P(f"Avg Return: {performance.get('mean_return', 0):.2%}", className="mb-1"),
                            html.P(f"Volatility: {performance.get('volatility', 0):.2%}", className="mb-1")
                        ], width=6),
                        dbc.Col([
                            html.P(f"Sharpe: {performance.get('sharpe_ratio', 0):.2f}", className="mb-1"),
                            html.P(f"Max DD: {performance.get('max_drawdown', 0):.2%}", className="mb-1"),
                            html.P(f"Win Rate: {performance.get('win_rate', 0):.1%}", className="mb-1")
                        ], width=6)
                    ])
                ])
            ], className="mb-2")
            
            cards.append(card)
        
        return html.Div(cards)
    
    def _create_factor_attribution_content(self):
        return html.Div("Factor attribution analysis coming soon...")
    
    def _create_sector_attribution_content(self):
        return html.Div("Sector attribution analysis coming soon...")
    
    def _data_update_loop(self):
        """Background thread to update data periodically"""
        while True:
            try:
                asyncio.run(self._update_data())
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
    async def _update_data(self):
        """Update all dashboard data"""
        try:
            # Simulate data updates - in production, this would fetch real data
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'SPY']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)
            
            # Update current data with simulated values
            self.current_data.update({
                'system_healthy': True,
                'current_regime_name': np.random.choice(['Bull Market', 'Bear Market', 'Sideways Market']),
                'regime_confidence': np.random.uniform(0.6, 0.95),
                'portfolio_value': 1000000 * (1 + np.random.normal(0, 0.01)),
                'daily_return': np.random.normal(0.001, 0.02),
                'portfolio_weights': {
                    symbol: np.random.uniform(0.1, 0.3) for symbol in symbols[:5]
                },
                'risk_metrics': {
                    'annualized_volatility': np.random.uniform(0.12, 0.25),
                    'var_95': -np.random.uniform(0.02, 0.06),
                    'maximum_drawdown': -np.random.uniform(0.05, 0.20),
                    'sharpe_ratio': np.random.uniform(0.5, 2.0)
                },
                'risk_alerts': [],
                'regime_attribution': {
                    0: {
                        'regime_name': 'Bear Market',
                        'frequency': 0.25,
                        'mean_return': -0.05,
                        'volatility': 0.25,
                        'sharpe_ratio': -0.2,
                        'max_drawdown': -0.15,
                        'win_rate': 0.4
                    },
                    1: {
                        'regime_name': 'Sideways Market',
                        'frequency': 0.45,
                        'mean_return': 0.02,
                        'volatility': 0.12,
                        'sharpe_ratio': 0.17,
                        'max_drawdown': -0.08,
                        'win_rate': 0.52
                    },
                    2: {
                        'regime_name': 'Bull Market',
                        'frequency': 0.30,
                        'mean_return': 0.15,
                        'volatility': 0.18,
                        'sharpe_ratio': 0.83,
                        'max_drawdown': -0.05,
                        'win_rate': 0.65
                    }
                }
            })
            
            # Generate simulated regime history
            dates = pd.date_range(start_date, end_date, freq='D')
            regime_probs = np.random.dirichlet([2, 3, 2], len(dates))
            
            self.regime_history = pd.DataFrame({
                'regime_prob_0': regime_probs[:, 0],
                'regime_prob_1': regime_probs[:, 1],
                'regime_prob_2': regime_probs[:, 2]
            }, index=dates)
            
            # Generate simulated returns
            self.portfolio_returns = pd.Series(
                np.random.normal(0.001, 0.015, len(dates)),
                index=dates
            )
            
            self.benchmark_returns = pd.Series(
                np.random.normal(0.0008, 0.012, len(dates)),
                index=dates
            )
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    def run(self, host='0.0.0.0', port=8050, debug=False):
        """Run the dashboard server"""
        logger.info(f"Starting dashboard server on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def create_dashboard(config_path: str = "config/config.yaml") -> PortfolioDashboard:
    """Factory function to create dashboard instance"""
    return PortfolioDashboard(config_path)


if __name__ == "__main__":
    dashboard = create_dashboard()
    dashboard.run(debug=True)