#!/usr/bin/env python3
"""
Portfolio Overview Page

Advanced portfolio dashboard showing:
- Real-time portfolio allocation
- Performance metrics and visualizations
- Asset price movements
- Risk indicators
- Rebalancing history
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from src.utils.performance_analytics import PerformanceAnalytics

def render_portfolio_overview():
    """Render the main portfolio overview page"""
    
    st.title("📊 Portfolio Overview")
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        render_welcome_screen()
        return
    
    # Get portfolio data
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data is None or portfolio_data.empty:
        st.warning("No portfolio data available. Please load data using the sidebar.")
        return
    
    # Render main dashboard
    render_portfolio_metrics(portfolio_data)
    render_portfolio_charts(portfolio_data)
    render_asset_analysis(portfolio_data)
    render_recent_activity()

def render_welcome_screen():
    """Render welcome screen when no data is loaded"""
    
    st.markdown("""
    ## Welcome to the Stochastic Portfolio Engine! 🚀
    
    This advanced portfolio management system provides comprehensive tools for:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 Regime Detection
        - Hidden Markov Models
        - Market state identification
        - Regime-aware optimization
        - Dynamic allocation strategies
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Advanced Backtesting
        - Realistic trading simulation
        - Multiple backtesting modes
        - Transaction cost modeling
        - Walk-forward analysis
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Risk Analytics
        - Real-time risk monitoring
        - VaR and CVaR calculations
        - Stress testing
        - Performance attribution
        """)
    
    st.markdown("""
    ---
    
    ### 🎯 Getting Started
    
    1. **Select Assets**: Use the sidebar to choose assets for your portfolio
    2. **Set Date Range**: Choose the analysis period
    3. **Load Data**: Click "Refresh Data" to begin analysis
    4. **Explore**: Navigate through different pages to explore features
    
    👈 **Start by selecting assets and loading data in the sidebar!**
    """)
    
    # Quick demo charts
    st.markdown("### 📈 Sample Analysis")
    
    # Generate sample data for demo
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    sample_data = pd.DataFrame({
        'Portfolio': np.cumsum(np.random.normal(0.001, 0.015, len(dates))),
        'Benchmark': np.cumsum(np.random.normal(0.0008, 0.012, len(dates)))
    }, index=dates)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sample_data.index,
        y=sample_data['Portfolio'],
        mode='lines',
        name='Sample Portfolio',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=sample_data.index,
        y=sample_data['Benchmark'],
        mode='lines',
        name='Sample Benchmark',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title="Sample Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_portfolio_metrics(portfolio_data):
    """Render key portfolio metrics"""
    
    st.subheader("📊 Portfolio Metrics")
    
    # Extract price data
    if 'Close' in portfolio_data.columns.get_level_values(0):
        close_prices = portfolio_data.xs('Close', level=0, axis=1)
    else:
        close_prices = portfolio_data
    
    # Calculate returns
    returns = close_prices.pct_change().dropna()
    
    # Portfolio returns (equal weight for demo)
    portfolio_returns = returns.mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate metrics
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / rolling_max - 1)
    max_drawdown = drawdown.min()
    
    # Current drawdown
    current_drawdown = drawdown.iloc[-1]
    
    # Portfolio value
    initial_value = 1000000  # $1M default
    current_value = initial_value * cumulative_returns.iloc[-1]
    daily_pnl = (portfolio_returns.iloc[-1] * current_value) if len(portfolio_returns) > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${current_value:,.0f}",
            f"{daily_pnl:+,.0f}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Total Return",
            f"{total_return:.2%}",
            f"{annualized_return:.2%} annualized"
        )
    
    with col3:
        st.metric(
            "Volatility",
            f"{volatility:.2%}",
            "Annualized"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{sharpe_ratio:.2f}",
            "Risk-adjusted return"
        )
    
    with col5:
        st.metric(
            "Max Drawdown",
            f"{max_drawdown:.2%}",
            f"{current_drawdown:.2%} current"
        )
    
    # Additional metrics in expandable section
    with st.expander("📈 Additional Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            win_rate = (portfolio_returns > 0).mean()
            st.metric("Win Rate", f"{win_rate:.1%}")
        
        with col2:
            avg_win = portfolio_returns[portfolio_returns > 0].mean()
            st.metric("Avg Win", f"{avg_win:.3%}")
        
        with col3:
            avg_loss = portfolio_returns[portfolio_returns < 0].mean()
            st.metric("Avg Loss", f"{avg_loss:.3%}")
        
        with col4:
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.metric("Profit Factor", f"{profit_factor:.2f}")

def render_portfolio_charts(portfolio_data):
    """Render portfolio performance charts"""
    
    # Extract price data
    if 'Close' in portfolio_data.columns.get_level_values(0):
        close_prices = portfolio_data.xs('Close', level=0, axis=1)
    else:
        close_prices = portfolio_data
    
    returns = close_prices.pct_change().dropna()
    portfolio_returns = returns.mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Create subplots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Cumulative Performance")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x}</b><br>Return: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Current Allocation")
        
        # Equal weight allocation for demo
        symbols = st.session_state.get('selected_symbols', [])
        weights = [1/len(symbols)] * len(symbols) if symbols else []
        
        if weights:
            fig = go.Figure(data=[go.Pie(
                labels=symbols,
                values=weights,
                hole=0.3,
                hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No assets selected for allocation display.")
    
    # Drawdown chart
    st.subheader("📉 Drawdown Analysis")
    
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / rolling_max - 1) * 100  # Convert to percentage
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=1),
        name='Drawdown',
        hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_asset_analysis(portfolio_data):
    """Render individual asset analysis"""
    
    st.subheader("📊 Asset Analysis")
    
    # Extract price data
    if 'Close' in portfolio_data.columns.get_level_values(0):
        close_prices = portfolio_data.xs('Close', level=0, axis=1)
    else:
        close_prices = portfolio_data
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Price Charts", "📊 Returns Analysis", "🔄 Correlation Matrix"])
    
    with tab1:
        # Normalize prices to 100 for comparison
        normalized_prices = close_prices / close_prices.iloc[0] * 100
        
        # Price chart options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"])
            show_volume = st.checkbox("Show Volume", False)
        
        with col1:
            fig = go.Figure()
            
            for symbol in normalized_prices.columns:
                fig.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[symbol],
                    mode='lines',
                    name=symbol,
                    hovertemplate=f'<b>{symbol}</b><br>Date: %{{x}}<br>Price: %{{y:.2f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Normalized Asset Prices (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Returns analysis
        returns = close_prices.pct_change().dropna()
        
        # Returns statistics
        returns_stats = pd.DataFrame({
            'Mean': returns.mean() * 252,
            'Volatility': returns.std() * np.sqrt(252),
            'Sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'Min': returns.min(),
            'Max': returns.max(),
            'Skew': returns.skew(),
            'Kurtosis': returns.kurtosis()
        })
        
        st.dataframe(
            returns_stats.style.format({
                'Mean': '{:.2%}',
                'Volatility': '{:.2%}',
                'Sharpe': '{:.2f}',
                'Min': '{:.3%}',
                'Max': '{:.3%}',
                'Skew': '{:.2f}',
                'Kurtosis': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Returns distribution
        selected_asset = st.selectbox("Select Asset for Distribution", returns.columns)
        
        if selected_asset:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=returns[selected_asset] * 100,
                nbinsx=50,
                name=f'{selected_asset} Returns',
                opacity=0.7,
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{selected_asset} Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation matrix
        returns = close_prices.pct_change().dropna()
        correlation_matrix = returns.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_recent_activity():
    """Render recent portfolio activity"""
    
    st.subheader("📋 Recent Activity")
    
    # Mock recent activity data
    recent_activity = pd.DataFrame({
        'Date': pd.date_range(end=datetime.now(), periods=5, freq='D'),
        'Action': ['Rebalance', 'Buy', 'Sell', 'Dividend', 'Rebalance'],
        'Asset': ['Portfolio', 'AAPL', 'GOOGL', 'MSFT', 'Portfolio'],
        'Amount': [0, 1000, -1500, 25, 0],
        'Description': [
            'Monthly rebalancing executed',
            'Purchased 5 shares of AAPL',
            'Sold 2 shares of GOOGL',
            'Dividend payment received',
            'Weekly rebalancing executed'
        ]
    })
    
    # Display in a nice format
    for idx, row in recent_activity.iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
        
        with col1:
            st.write(f"**{row['Date'].strftime('%Y-%m-%d')}**")
        
        with col2:
            if row['Action'] == 'Buy':
                st.success(f"🟢 {row['Action']}")
            elif row['Action'] == 'Sell':
                st.error(f"🔴 {row['Action']}")
            elif row['Action'] == 'Rebalance':
                st.info(f"🔄 {row['Action']}")
            else:
                st.write(f"💰 {row['Action']}")
        
        with col3:
            st.write(f"**{row['Asset']}**")
        
        with col4:
            st.write(row['Description'])
    
    # Load more button
    if st.button("📜 Load More Activity"):
        st.info("More activity history would be loaded here.")

# Main execution
if __name__ == "__main__":
    render_portfolio_overview()