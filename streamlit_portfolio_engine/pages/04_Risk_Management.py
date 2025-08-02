#!/usr/bin/env python3
"""
Risk Management Dashboard Page

Comprehensive risk monitoring and analysis featuring:
- Real-time VaR and CVaR calculations
- Position-level risk monitoring
- Risk limit alerts and notifications
- Stress testing scenarios
- Correlation and exposure analysis
- Risk attribution breakdown
- Historical risk metrics
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
from scipy import stats

# Add src to path
sys.path.append('src')

from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor, RiskLimits
from src.risk.metrics.risk_analytics import RiskAnalytics

def render_risk_management():
    """Render the risk management dashboard"""
    
    st.title("Risk Management Dashboard")
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        render_risk_welcome_screen()
        return
    
    # Get portfolio data
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data is None or portfolio_data.empty:
        st.warning("No portfolio data available. Please load data using the sidebar.")
        return
    
    # Render main dashboard
    render_risk_controls()
    render_current_risk_status()
    render_var_analysis()
    render_stress_testing()
    render_position_risk()
    render_correlation_analysis()
    render_risk_attribution()

def render_risk_welcome_screen():
    """Render welcome screen for risk management"""
    
    st.markdown("""
    ## ⚠️ Comprehensive Risk Management System
    
    Advanced risk monitoring and analysis tools for portfolio protection and optimization.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 📊 VaR & CVaR
        - Value at Risk calculations
        - Conditional VaR (Expected Shortfall)
        - Multiple confidence levels
        - Historical & parametric methods
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Position Risk
        - Individual asset risk
        - Concentration limits
        - Exposure monitoring
        - Risk contribution analysis
        """)
    
    with col3:
        st.markdown("""
        ### 🧪 Stress Testing
        - Historical scenarios
        - Monte Carlo simulations
        - Custom shock tests
        - Tail risk analysis
        """)
    
    with col4:
        st.markdown("""
        ### 🔗 Correlation Analysis
        - Asset correlation matrices
        - Regime-dependent correlations
        - Risk clustering
        - Diversification metrics
        """)
    
    st.markdown("""
    ---
    
    ### 🎯 Key Features
    
    - **Real-time Monitoring**: Live risk metrics with automatic alerts
    - **Multi-method VaR**: Historical simulation, parametric, and Monte Carlo VaR
    - **Stress Testing**: Pre-built and custom scenarios for extreme event analysis
    - **Risk Attribution**: Breakdown of portfolio risk by asset, sector, and factor
    - **Limit Management**: Configurable risk limits with breach notifications
    - **Regulatory Reporting**: Standard risk reports for compliance
    
    👈 **Load portfolio data in the sidebar to begin risk analysis!**
    """)

def render_risk_controls():
    """Render risk management controls and settings"""
    
    st.subheader("⚙️ Risk Parameters")
    
    with st.expander("🎛️ VaR Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence_levels = st.multiselect(
                "Confidence Levels (%)",
                [90, 95, 97.5, 99, 99.5],
                default=[95, 99],
                help="VaR confidence intervals"
            )
            st.session_state.var_confidence_levels = confidence_levels
        
        with col2:
            var_methods = st.multiselect(
                "VaR Methods",
                ["Historical", "Parametric", "Monte Carlo", "Cornish-Fisher"],
                default=["Historical", "Parametric"],
                help="Risk calculation methodologies"
            )
            st.session_state.var_methods = var_methods
        
        with col3:
            lookback_window = st.slider(
                "Lookback Window (days)",
                min_value=30,
                max_value=1000,
                value=st.session_state.get('risk_lookback_window', 252),
                help="Historical data window for risk calculations"
            )
            st.session_state.risk_lookback_window = lookback_window
        
        with col4:
            decay_factor = st.slider(
                "Decay Factor",
                min_value=0.90,
                max_value=0.99,
                value=st.session_state.get('decay_factor', 0.94),
                step=0.01,
                help="Exponential weighting for recent observations"
            )
            st.session_state.decay_factor = decay_factor
    
    # Risk limits configuration
    with st.expander("🚨 Risk Limits"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Portfolio Limits:**")
            
            max_portfolio_var = st.slider(
                "Max Portfolio VaR (%)",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.get('max_portfolio_var', 3.0),
                step=0.1,
                help="Maximum daily VaR as percentage of portfolio value"
            )
            st.session_state.max_portfolio_var = max_portfolio_var
            
            max_drawdown_limit = st.slider(
                "Max Drawdown Limit (%)",
                min_value=5.0,
                max_value=50.0,
                value=st.session_state.get('max_drawdown_limit', 15.0),
                step=1.0,
                help="Maximum acceptable drawdown"
            )
            st.session_state.max_drawdown_limit = max_drawdown_limit
        
        with col2:
            st.markdown("**Position Limits:**")
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=5.0,
                max_value=50.0,
                value=st.session_state.get('max_position_size', 20.0),
                step=1.0,
                help="Maximum allocation to single position"
            )
            st.session_state.max_position_size = max_position_size
            
            concentration_limit = st.slider(
                "Concentration Limit (%)",
                min_value=10.0,
                max_value=80.0,
                value=st.session_state.get('concentration_limit', 40.0),
                step=5.0,
                help="Maximum concentration in top N positions"
            )
            st.session_state.concentration_limit = concentration_limit
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("📊 Calculate Risk Metrics", use_container_width=True, type="primary"):
            calculate_risk_metrics()
    
    with col2:
        if st.button("🧪 Run Stress Test", use_container_width=True):
            run_stress_tests()
    
    with col3:
        if st.button("📥 Export Report", use_container_width=True):
            export_risk_report()

def calculate_risk_metrics():
    """Calculate comprehensive risk metrics"""
    
    with st.spinner("📊 Calculating Risk Metrics..."):
        try:
            portfolio_data = st.session_state.portfolio_data
            
            # Extract price data
            if 'Close' in portfolio_data.columns.get_level_values(0):
                close_prices = portfolio_data.xs('Close', level=0, axis=1)
            else:
                close_prices = portfolio_data
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Portfolio returns (equal weight for demo)
            portfolio_returns = returns.mean(axis=1)
            
            # Limit to lookback window
            lookback = st.session_state.risk_lookback_window
            if len(portfolio_returns) > lookback:
                portfolio_returns = portfolio_returns.tail(lookback)
                returns = returns.tail(lookback)
            
            # Calculate VaR using different methods
            var_results = {}
            confidence_levels = st.session_state.var_confidence_levels
            
            for method in st.session_state.var_methods:
                var_results[method] = calculate_var_by_method(
                    portfolio_returns, method, confidence_levels
                )
            
            # Calculate position-level risk
            position_risk = calculate_position_risk(returns, close_prices)
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Stress test scenarios
            stress_results = run_basic_stress_tests(portfolio_returns, returns)
            
            # Store results
            st.session_state.risk_metrics = {
                'portfolio_returns': portfolio_returns,
                'asset_returns': returns,
                'var_results': var_results,
                'position_risk': position_risk,
                'correlation_matrix': correlation_matrix,
                'stress_results': stress_results,
                'calculation_time': datetime.now()
            }
            
            st.success("✅ Risk metrics calculated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error calculating risk metrics: {str(e)}")
            st.exception(e)

def calculate_var_by_method(returns, method, confidence_levels):
    """Calculate VaR using specified method"""
    
    var_results = {}
    
    for confidence in confidence_levels:
        alpha = (100 - confidence) / 100
        
        if method == "Historical":
            var_value = np.percentile(returns, alpha * 100)
            cvar_value = returns[returns <= var_value].mean()
        
        elif method == "Parametric":
            mu = returns.mean()
            sigma = returns.std()
            var_value = stats.norm.ppf(alpha, mu, sigma)
            cvar_value = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
        
        elif method == "Monte Carlo":
            # Simple Monte Carlo simulation
            n_sims = 10000
            simulated_returns = np.random.normal(
                returns.mean(), returns.std(), n_sims
            )
            var_value = np.percentile(simulated_returns, alpha * 100)
            cvar_value = simulated_returns[simulated_returns <= var_value].mean()
        
        elif method == "Cornish-Fisher":
            # Cornish-Fisher expansion for non-normal distributions
            mu = returns.mean()
            sigma = returns.std()
            skew = returns.skew()
            kurt = returns.kurtosis()
            
            z_alpha = stats.norm.ppf(alpha)
            cf_adjustment = (z_alpha**2 - 1) * skew / 6 + (z_alpha**3 - 3*z_alpha) * kurt / 24
            var_value = mu + sigma * (z_alpha + cf_adjustment)
            cvar_value = mu - sigma * stats.norm.pdf(z_alpha) / alpha  # Approximation
        
        var_results[f"{confidence}%"] = {
            'var': var_value,
            'cvar': cvar_value
        }
    
    return var_results

def calculate_position_risk(returns, prices):
    """Calculate individual position risk metrics"""
    
    position_risk = {}
    
    for asset in returns.columns:
        asset_returns = returns[asset]
        
        position_risk[asset] = {
            'volatility': asset_returns.std() * np.sqrt(252),
            'var_95': np.percentile(asset_returns, 5),
            'cvar_95': asset_returns[asset_returns <= np.percentile(asset_returns, 5)].mean(),
            'max_drawdown': calculate_max_drawdown(prices[asset]),
            'skewness': asset_returns.skew(),
            'kurtosis': asset_returns.kurtosis(),
            'beta': calculate_beta(asset_returns, returns.mean(axis=1))
        }
    
    return position_risk

def calculate_max_drawdown(price_series):
    """Calculate maximum drawdown for a price series"""
    
    running_max = price_series.expanding().max()
    drawdown = (price_series / running_max - 1)
    return drawdown.min()

def calculate_beta(asset_returns, market_returns):
    """Calculate beta relative to market"""
    
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    return covariance / market_variance if market_variance != 0 else 0

def run_basic_stress_tests(portfolio_returns, asset_returns):
    """Run basic stress testing scenarios"""
    
    stress_scenarios = {
        "2008 Financial Crisis": {"market_shock": -0.05, "volatility_shock": 2.0},
        "COVID-19 Crash": {"market_shock": -0.08, "volatility_shock": 3.0},
        "Flash Crash": {"market_shock": -0.10, "volatility_shock": 1.5},
        "Interest Rate Shock": {"market_shock": -0.03, "volatility_shock": 1.2},
        "Correlation Breakdown": {"market_shock": -0.04, "volatility_shock": 2.5}
    }
    
    stress_results = {}
    
    for scenario_name, shocks in stress_scenarios.items():
        # Apply shock to returns
        shocked_returns = portfolio_returns + shocks["market_shock"]
        shocked_volatility = portfolio_returns.std() * shocks["volatility_shock"]
        
        # Calculate impact
        portfolio_value_impact = (1 + shocked_returns.iloc[-1]) - 1
        var_impact = np.percentile(shocked_returns, 5)
        
        stress_results[scenario_name] = {
            'portfolio_impact': portfolio_value_impact,
            'var_impact': var_impact,
            'volatility_impact': shocked_volatility
        }
    
    return stress_results

def render_current_risk_status():
    """Render current risk status dashboard"""
    
    if 'risk_metrics' not in st.session_state:
        st.info("👆 Calculate risk metrics to view current risk status.")
        return
    
    st.subheader("🎯 Current Risk Status")
    
    risk_data = st.session_state.risk_metrics
    portfolio_returns = risk_data['portfolio_returns']
    var_results = risk_data['var_results']
    
    # Current risk metrics
    current_volatility = portfolio_returns.std() * np.sqrt(252)
    current_var_95 = np.percentile(portfolio_returns, 5)
    
    # Risk limit checks
    var_limit = st.session_state.max_portfolio_var / 100
    drawdown_limit = st.session_state.max_drawdown_limit / 100
    
    var_breach = abs(current_var_95) > var_limit
    
    # Status indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status_color = "normal" if not var_breach else "inverse"
        st.metric(
            "Daily VaR (95%)",
            f"{current_var_95:.3%}",
            delta=f"Limit: {var_limit:.1%}",
            delta_color=status_color
        )
    
    with col2:
        st.metric(
            "Annualized Vol",
            f"{current_volatility:.2%}",
            delta="Current estimate"
        )
    
    with col3:
        # Portfolio value change
        latest_return = portfolio_returns.iloc[-1]
        st.metric(
            "Latest Return",
            f"{latest_return:.3%}",
            delta=f"vs avg: {portfolio_returns.mean():.3%}"
        )
    
    with col4:
        # Max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        current_drawdown = (cumulative_returns.iloc[-1] / running_max.iloc[-1] - 1)
        
        dd_breach = abs(current_drawdown) > drawdown_limit
        dd_color = "normal" if not dd_breach else "inverse"
        
        st.metric(
            "Current Drawdown",
            f"{current_drawdown:.2%}",
            delta=f"Limit: {drawdown_limit:.0%}",
            delta_color=dd_color
        )
    
    with col5:
        # Risk score (composite)
        risk_score = min(100, (abs(current_var_95) / var_limit * 50 + 
                              abs(current_drawdown) / drawdown_limit * 50))
        
        risk_color = "normal" if risk_score < 70 else "inverse"
        st.metric(
            "Risk Score",
            f"{risk_score:.0f}/100",
            delta="Composite risk level",
            delta_color=risk_color
        )
    
    # Alert messages
    if var_breach or dd_breach:
        st.error("🚨 **RISK LIMIT BREACH DETECTED**")
        
        if var_breach:
            st.error(f"VaR limit exceeded: {abs(current_var_95):.3%} > {var_limit:.1%}")
        
        if dd_breach:
            st.error(f"Drawdown limit exceeded: {abs(current_drawdown):.2%} > {drawdown_limit:.0%}")
    else:
        st.success("✅ All risk limits within acceptable ranges")

def render_var_analysis():
    """Render comprehensive VaR analysis"""
    
    if 'risk_metrics' not in st.session_state:
        return
    
    st.subheader("📊 Value at Risk Analysis")
    
    risk_data = st.session_state.risk_metrics
    var_results = risk_data['var_results']
    portfolio_returns = risk_data['portfolio_returns']
    
    # VaR comparison table
    var_comparison = []
    
    for method, results in var_results.items():
        for confidence, values in results.items():
            var_comparison.append({
                'Method': method,
                'Confidence': confidence,
                'VaR': values['var'],
                'CVaR': values['cvar']
            })
    
    if var_comparison:
        var_df = pd.DataFrame(var_comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**VaR Comparison by Method:**")
            
            styled_df = var_df.style.format({
                'VaR': '{:.3%}',
                'CVaR': '{:.3%}'
            })
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.markdown("**VaR Visualization:**")
            
            # Create VaR comparison chart
            fig = go.Figure()
            
            methods = var_df['Method'].unique()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, method in enumerate(methods):
                method_data = var_df[var_df['Method'] == method]
                
                fig.add_trace(go.Scatter(
                    x=method_data['Confidence'],
                    y=method_data['VaR'] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name=f'{method} VaR',
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title="VaR by Method and Confidence Level",
                xaxis_title="Confidence Level",
                yaxis_title="VaR (%)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Historical VaR backtest
    st.markdown("**VaR Backtesting:**")
    
    # Simple VaR backtest
    var_95 = np.percentile(portfolio_returns, 5)
    exceedances = portfolio_returns < var_95
    exceedance_rate = exceedances.mean()
    expected_rate = 0.05  # 5% for 95% VaR
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Exceedance Rate", f"{exceedance_rate:.2%}")
    
    with col2:
        st.metric("Expected Rate", f"{expected_rate:.1%}")
    
    with col3:
        # Traffic light system
        if abs(exceedance_rate - expected_rate) < 0.01:
            st.success("✅ Good")
        elif abs(exceedance_rate - expected_rate) < 0.02:
            st.warning("⚠️ Moderate")
        else:
            st.error("❌ Poor")

def render_stress_testing():
    """Render stress testing results"""
    
    if 'risk_metrics' not in st.session_state:
        return
    
    st.subheader("🧪 Stress Testing")
    
    risk_data = st.session_state.risk_metrics
    stress_results = risk_data['stress_results']
    
    if not stress_results:
        st.info("No stress test results available.")
        return
    
    # Stress test results table
    stress_df = pd.DataFrame(stress_results).T
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Scenario Impact Analysis:**")
        
        styled_stress = stress_df.style.format({
            'portfolio_impact': '{:.2%}',
            'var_impact': '{:.3%}',
            'volatility_impact': '{:.2%}'
        })
        
        st.dataframe(styled_stress, use_container_width=True)
    
    with col2:
        st.markdown("**Stress Test Visualization:**")
        
        # Create stress test impact chart
        fig = go.Figure()
        
        scenarios = list(stress_results.keys())
        impacts = [stress_results[s]['portfolio_impact'] * 100 for s in scenarios]
        
        colors = ['red' if impact < -5 else 'orange' if impact < -2 else 'yellow' 
                 for impact in impacts]
        
        fig.add_trace(go.Bar(
            x=scenarios,
            y=impacts,
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Impact: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Stress Test Portfolio Impact",
            xaxis_title="Scenario",
            yaxis_title="Portfolio Impact (%)",
            height=300,
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Custom stress test
    with st.expander("🔧 Custom Stress Test"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            custom_shock = st.slider(
                "Market Shock (%)",
                min_value=-20,
                max_value=20,
                value=-5,
                help="Applied market return shock"
            )
        
        with col2:
            volatility_multiplier = st.slider(
                "Volatility Multiplier",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Volatility scaling factor"
            )
        
        with col3:
            if st.button("🧪 Run Custom Test"):
                portfolio_returns = risk_data['portfolio_returns']
                shocked_return = custom_shock / 100
                portfolio_impact = shocked_return
                
                st.metric(
                    "Estimated Impact",
                    f"{portfolio_impact:.2%}",
                    "Single-day shock"
                )

def render_position_risk():
    """Render individual position risk analysis"""
    
    if 'risk_metrics' not in st.session_state:
        return
    
    st.subheader("🎯 Position Risk Analysis")
    
    risk_data = st.session_state.risk_metrics
    position_risk = risk_data['position_risk']
    
    if not position_risk:
        st.info("No position risk data available.")
        return
    
    # Convert to DataFrame for easier manipulation
    position_df = pd.DataFrame(position_risk).T
    
    # Position risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Risk Metrics by Asset:**")
        
        styled_positions = position_df.style.format({
            'volatility': '{:.2%}',
            'var_95': '{:.3%}',
            'cvar_95': '{:.3%}',
            'max_drawdown': '{:.2%}',
            'skewness': '{:.2f}',
            'kurtosis': '{:.2f}',
            'beta': '{:.2f}'
        })
        
        st.dataframe(styled_positions, use_container_width=True)
    
    with col2:
        st.markdown("**Risk Visualization:**")
        
        # Risk-return scatter plot
        fig = go.Figure()
        
        # Calculate expected returns (simplified)
        asset_returns = risk_data['asset_returns']
        expected_returns = asset_returns.mean() * 252
        
        fig.add_trace(go.Scatter(
            x=position_df['volatility'],
            y=expected_returns,
            mode='markers+text',
            text=position_df.index,
            textposition="top center",
            marker=dict(
                size=10,
                color=position_df['beta'],
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Beta")
            ),
            hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2%}<br>Expected Return: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk-Return Profile by Asset",
            xaxis_title="Volatility",
            yaxis_title="Expected Return",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Position limits check
    st.markdown("**Position Limit Analysis:**")
    
    # Simulate equal weights for limit checking
    n_assets = len(position_risk)
    equal_weight = 1 / n_assets
    max_weight_limit = st.session_state.max_position_size / 100
    
    limit_breaches = []
    for asset in position_risk.keys():
        if equal_weight > max_weight_limit:
            limit_breaches.append(asset)
    
    if limit_breaches:
        st.warning(f"⚠️ Position size limits would be breached for: {', '.join(limit_breaches)}")
    else:
        st.success("✅ All positions within size limits")

def render_correlation_analysis():
    """Render correlation and diversification analysis"""
    
    if 'risk_metrics' not in st.session_state:
        return
    
    st.subheader("🔗 Correlation Analysis")
    
    risk_data = st.session_state.risk_metrics
    correlation_matrix = risk_data['correlation_matrix']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Correlation Matrix:**")
        
        # Create correlation heatmap
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
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Diversification Metrics:**")
        
        # Calculate diversification metrics
        n_assets = len(correlation_matrix)
        avg_correlation = (correlation_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))
        
        # Effective number of assets (diversification ratio)
        portfolio_weights = np.ones(n_assets) / n_assets
        portfolio_variance = np.dot(portfolio_weights, np.dot(correlation_matrix, portfolio_weights))
        diversification_ratio = 1 / (portfolio_variance * n_assets)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Avg Correlation", f"{avg_correlation:.3f}")
            st.metric("Max Correlation", f"{correlation_matrix.max().max():.3f}")
        
        with col_b:
            st.metric("Min Correlation", f"{correlation_matrix.min().min():.3f}")
            st.metric("Diversification Ratio", f"{diversification_ratio:.2f}")
        
        # Correlation distribution
        fig = go.Figure()
        
        # Get upper triangle correlations (excluding diagonal)
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        correlations = upper_triangle.stack().values
        
        fig.add_trace(go.Histogram(
            x=correlations,
            nbinsx=20,
            name='Correlation Distribution',
            hovertemplate='Correlation: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Correlation Distribution",
            xaxis_title="Correlation",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_risk_attribution():
    """Render risk attribution analysis"""
    
    if 'risk_metrics' not in st.session_state:
        return
    
    st.subheader("📈 Risk Attribution")
    
    risk_data = st.session_state.risk_metrics
    asset_returns = risk_data['asset_returns']
    correlation_matrix = risk_data['correlation_matrix']
    
    # Calculate risk contributions (simplified)
    n_assets = len(asset_returns.columns)
    equal_weights = np.ones(n_assets) / n_assets
    
    # Individual asset risk contributions
    asset_volatilities = asset_returns.std() * np.sqrt(252)
    portfolio_volatility = (asset_returns.mean(axis=1)).std() * np.sqrt(252)
    
    risk_contributions = {}
    for i, asset in enumerate(asset_returns.columns):
        # Marginal contribution to risk
        marginal_contrib = asset_volatilities[asset] * equal_weights[i]
        risk_contributions[asset] = marginal_contrib / portfolio_volatility
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Risk Contribution by Asset:**")
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(risk_contributions.keys()),
            values=list(risk_contributions.values()),
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>Risk Contribution: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Portfolio Risk Attribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Risk Decomposition:**")
        
        # Risk decomposition table
        risk_decomp = pd.DataFrame({
            'Asset': list(risk_contributions.keys()),
            'Risk Contribution': list(risk_contributions.values()),
            'Volatility': [asset_volatilities[asset] for asset in risk_contributions.keys()],
            'Weight': equal_weights
        })
        
        styled_decomp = risk_decomp.style.format({
            'Risk Contribution': '{:.2%}',
            'Volatility': '{:.2%}',
            'Weight': '{:.2%}'
        })
        
        st.dataframe(styled_decomp, use_container_width=True)
        
        # Summary statistics
        st.markdown("**Risk Summary:**")
        st.metric("Portfolio Volatility", f"{portfolio_volatility:.2%}")
        st.metric("Largest Risk Contributor", 
                 f"{max(risk_contributions, key=risk_contributions.get)}")
        st.metric("Risk Concentration", 
                 f"{max(risk_contributions.values()):.1%}")

def run_stress_tests():
    """Run comprehensive stress testing"""
    
    st.info("🧪 Running comprehensive stress tests...")
    
    # This would trigger the full stress testing suite
    calculate_risk_metrics()

def export_risk_report():
    """Export comprehensive risk report"""
    
    if 'risk_metrics' not in st.session_state:
        st.warning("No risk metrics available to export.")
        return
    
    # Create comprehensive risk report
    risk_data = st.session_state.risk_metrics
    
    # Prepare export data
    report_data = {
        'Risk Calculation Time': [risk_data['calculation_time']],
        'Portfolio Volatility': [risk_data['portfolio_returns'].std() * np.sqrt(252)],
        'VaR 95%': [np.percentile(risk_data['portfolio_returns'], 5)],
        'CVaR 95%': [risk_data['portfolio_returns'][
            risk_data['portfolio_returns'] <= np.percentile(risk_data['portfolio_returns'], 5)
        ].mean()]
    }
    
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Risk Report CSV",
        data=csv,
        file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ Risk report prepared for download!")

# Main execution
if __name__ == "__main__":
    render_risk_management()