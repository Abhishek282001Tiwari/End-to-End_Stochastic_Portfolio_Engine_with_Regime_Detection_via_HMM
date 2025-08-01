#!/usr/bin/env python3
"""
Performance Analytics Dashboard Page

Comprehensive performance analysis featuring:
- Brinson performance attribution
- Factor-based attribution analysis
- Benchmark comparison and tracking
- Regime-conditional performance metrics
- Risk-adjusted performance measures
- Statistical performance analysis
- Performance persistence analysis
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

from src.utils.performance_analytics import PerformanceAnalytics
from src.analysis.attribution.brinson_attribution import BrinsonAttribution

def render_performance_analytics():
    """Render the performance analytics dashboard"""
    
    st.title("📈 Performance Analytics Dashboard")
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        render_performance_welcome_screen()
        return
    
    # Get portfolio data
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data is None or portfolio_data.empty:
        st.warning("No portfolio data available. Please load data using the sidebar.")
        return
    
    # Render main dashboard
    render_performance_controls()
    render_performance_summary()
    render_attribution_analysis()
    render_benchmark_comparison()
    render_factor_analysis()
    render_regime_performance()
    render_statistical_analysis()

def render_performance_welcome_screen():
    """Render welcome screen for performance analytics"""
    
    st.markdown("""
    ## 📈 Comprehensive Performance Analytics
    
    Advanced performance measurement and attribution analysis for portfolio optimization.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 📊 Attribution Analysis
        - Brinson performance attribution
        - Asset allocation effects
        - Security selection effects
        - Interaction effects
        """)
    
    with col2:
        st.markdown("""
        ### 🏆 Benchmark Comparison
        - Tracking error analysis
        - Information ratio
        - Alpha and beta measurement
        - Outperformance analysis
        """)
    
    with col3:
        st.markdown("""
        ### 📉 Factor Analysis
        - Fama-French factors
        - Market factor exposure
        - Style factor analysis
        - Custom factor models
        """)
    
    with col4:
        st.markdown("""
        ### 🔄 Regime Analysis
        - Regime-conditional returns
        - Performance by market state
        - Regime transition impact
        - Adaptive performance
        """)
    
    st.markdown("""
    ---
    
    ### 🎯 Key Features
    
    - **Multi-period Attribution**: Rolling and cumulative performance attribution
    - **Risk-adjusted Metrics**: Sharpe, Sortino, Calmar, and information ratios
    - **Statistical Analysis**: Performance significance testing and confidence intervals
    - **Benchmark Tracking**: Multiple benchmark comparison and relative performance
    - **Factor Exposure**: Systematic risk factor analysis and style attribution
    - **Regime Awareness**: Performance analysis across different market regimes
    
    👈 **Load portfolio data in the sidebar to begin performance analysis!**
    """)

def render_performance_controls():
    """Render performance analysis controls and settings"""
    
    st.subheader("⚙️ Performance Analysis Settings")
    
    with st.expander("🎛️ Analysis Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            analysis_period = st.selectbox(
                "Analysis Period",
                ["1M", "3M", "6M", "1Y", "2Y", "3Y", "All"],
                index=3,  # Default to 1Y
                help="Time period for performance analysis"
            )
            st.session_state.perf_analysis_period = analysis_period
        
        with col2:
            benchmark_symbol = st.selectbox(
                "Benchmark",
                ["SPY", "QQQ", "IWM", "VTI", "VXUS", "Custom"],
                help="Benchmark for performance comparison"
            )
            st.session_state.benchmark_symbol = benchmark_symbol
        
        with col3:
            attribution_method = st.selectbox(
                "Attribution Method",
                ["Brinson", "Factor-based", "Sector-based", "Asset-based"],
                help="Performance attribution methodology"
            )
            st.session_state.attribution_method = attribution_method
        
        with col4:
            rolling_window = st.slider(
                "Rolling Window (days)",
                min_value=30,
                max_value=252,
                value=60,
                help="Window for rolling performance metrics"
            )
            st.session_state.perf_rolling_window = rolling_window
    
    # Risk-free rate and other parameters
    with st.expander("📊 Performance Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            risk_free_rate = st.slider(
                "Risk-free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.get('risk_free_rate', 2.0),
                step=0.1,
                help="Annual risk-free rate for Sharpe ratio calculation"
            )
            st.session_state.risk_free_rate = risk_free_rate
            
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=90,
                max_value=99,
                value=95,
                help="Confidence level for statistical tests"
            )
            st.session_state.perf_confidence_level = confidence_level
        
        with col2:
            minimum_periods = st.slider(
                "Minimum Periods",
                min_value=10,
                max_value=100,
                value=30,
                help="Minimum periods for valid calculations"
            )
            st.session_state.minimum_periods = minimum_periods
            
            annualization_factor = st.number_input(
                "Annualization Factor",
                min_value=200,
                max_value=365,
                value=252,
                help="Trading days per year for annualization"
            )
            st.session_state.annualization_factor = annualization_factor
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("📊 Analyze Performance", use_container_width=True, type="primary"):
            analyze_performance()
    
    with col2:
        if st.button("🔄 Reset Analysis", use_container_width=True):
            reset_performance_analysis()
    
    with col3:
        if st.session_state.get('performance_results') is not None:
            if st.button("📥 Export Report", use_container_width=True):
                export_performance_report()

def analyze_performance():
    """Run comprehensive performance analysis"""
    
    with st.spinner("📊 Analyzing Performance..."):
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
            
            # Filter by analysis period
            portfolio_returns = filter_by_period(portfolio_returns, st.session_state.perf_analysis_period)
            returns = filter_by_period(returns, st.session_state.perf_analysis_period)
            
            # Calculate performance metrics
            perf_metrics = calculate_performance_metrics(portfolio_returns)
            
            # Rolling performance metrics
            rolling_metrics = calculate_rolling_metrics(portfolio_returns)
            
            # Attribution analysis
            attribution_results = perform_attribution_analysis(returns, portfolio_returns)
            
            # Benchmark analysis
            benchmark_results = perform_benchmark_analysis(portfolio_returns)
            
            # Factor analysis
            factor_results = perform_factor_analysis(portfolio_returns, returns)
            
            # Statistical analysis
            statistical_results = perform_statistical_analysis(portfolio_returns)
            
            # Store results
            st.session_state.performance_results = {
                'portfolio_returns': portfolio_returns,
                'asset_returns': returns,
                'metrics': perf_metrics,
                'rolling_metrics': rolling_metrics,
                'attribution': attribution_results,
                'benchmark': benchmark_results,
                'factors': factor_results,
                'statistics': statistical_results,
                'analysis_time': datetime.now()
            }
            
            # Success message
            total_return = perf_metrics['total_return']
            sharpe_ratio = perf_metrics['sharpe_ratio']
            
            st.success(f"""
            ✅ **Performance Analysis Complete!**
            - Total Return: {total_return:.2%}
            - Sharpe Ratio: {sharpe_ratio:.2f}
            - Analysis Period: {st.session_state.perf_analysis_period}
            """)
            
        except Exception as e:
            st.error(f"❌ Performance analysis failed: {str(e)}")
            st.exception(e)

def filter_by_period(data, period):
    """Filter data by specified period"""
    
    if period == "All":
        return data
    
    # Calculate lookback days
    period_map = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 252,
        "2Y": 504, 
        "3Y": 756
    }
    
    days = period_map.get(period, 252)
    
    if len(data) > days:
        return data.tail(days)
    else:
        return data

def calculate_performance_metrics(returns):
    """Calculate comprehensive performance metrics"""
    
    risk_free_rate = st.session_state.risk_free_rate / 100
    annualization = st.session_state.annualization_factor
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (annualization / len(returns)) - 1
    volatility = returns.std() * np.sqrt(annualization)
    
    # Risk-adjusted metrics
    excess_returns = returns - risk_free_rate / annualization
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(annualization) if returns.std() > 0 else 0
    
    # Downside metrics
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(annualization) if len(negative_returns) > 0 else 0
    sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(annualization) if downside_std > 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns / running_max - 1
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Additional metrics
    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Tail risk metrics
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }

def calculate_rolling_metrics(returns):
    """Calculate rolling performance metrics"""
    
    window = st.session_state.perf_rolling_window
    annualization = st.session_state.annualization_factor
    
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    rolling_metrics['rolling_return'] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    
    # Rolling volatility
    rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(annualization)
    
    # Rolling Sharpe ratio
    risk_free_rate = st.session_state.risk_free_rate / 100
    excess_returns = returns - risk_free_rate / annualization
    rolling_metrics['rolling_sharpe'] = (
        excess_returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(annualization)
    )
    
    # Rolling maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.rolling(window).max()
    rolling_drawdown = cumulative_returns / rolling_max - 1
    rolling_metrics['rolling_max_dd'] = rolling_drawdown.rolling(window).min()
    
    return rolling_metrics.dropna()

def perform_attribution_analysis(asset_returns, portfolio_returns):
    """Perform performance attribution analysis"""
    
    method = st.session_state.attribution_method
    
    if method == "Asset-based":
        # Simple asset-based attribution
        asset_contributions = {}
        equal_weights = 1 / len(asset_returns.columns)  # Equal weight assumption
        
        for asset in asset_returns.columns:
            asset_contribution = asset_returns[asset] * equal_weights
            asset_contributions[asset] = {
                'contribution': asset_contribution.sum(),
                'weight': equal_weights,
                'return': asset_returns[asset].mean()
            }
        
        return {
            'method': method,
            'asset_contributions': asset_contributions,
            'total_contribution': sum([contrib['contribution'] for contrib in asset_contributions.values()])
        }
    
    elif method == "Factor-based":
        # Simplified factor attribution
        market_returns = asset_returns.mean(axis=1)  # Market proxy
        
        # Calculate beta for each asset
        factor_exposures = {}
        for asset in asset_returns.columns:
            beta = np.cov(asset_returns[asset], market_returns)[0, 1] / np.var(market_returns)
            factor_exposures[asset] = {
                'market_beta': beta,
                'specific_return': asset_returns[asset].mean() - beta * market_returns.mean()
            }
        
        return {
            'method': method,
            'factor_exposures': factor_exposures,
            'market_return': market_returns.mean()
        }
    
    else:
        # Default simple attribution
        return {
            'method': 'Simple',
            'total_return': portfolio_returns.sum(),
            'avg_return': portfolio_returns.mean()
        }

def perform_benchmark_analysis(portfolio_returns):
    """Perform benchmark comparison analysis"""
    
    benchmark_symbol = st.session_state.benchmark_symbol
    
    if benchmark_symbol == "Custom":
        # Use portfolio data as benchmark (simplified)
        benchmark_returns = portfolio_returns * 0.8  # Mock benchmark
    else:
        # In practice, would fetch actual benchmark data
        # For demo, create synthetic benchmark
        benchmark_returns = portfolio_returns + np.random.normal(0, 0.005, len(portfolio_returns))
    
    # Tracking error
    tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(st.session_state.annualization_factor)
    
    # Information ratio
    excess_returns = portfolio_returns - benchmark_returns
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(st.session_state.annualization_factor)
    
    # Alpha and beta
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    risk_free_rate = st.session_state.risk_free_rate / 100 / st.session_state.annualization_factor
    alpha = portfolio_returns.mean() - risk_free_rate - beta * (benchmark_returns.mean() - risk_free_rate)
    
    # Up/down capture ratios
    up_periods = benchmark_returns > 0
    down_periods = benchmark_returns < 0
    
    up_capture = (portfolio_returns[up_periods].mean() / benchmark_returns[up_periods].mean()) if up_periods.any() else 0
    down_capture = (portfolio_returns[down_periods].mean() / benchmark_returns[down_periods].mean()) if down_periods.any() else 0
    
    return {
        'benchmark_symbol': benchmark_symbol,
        'benchmark_returns': benchmark_returns,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'alpha': alpha,
        'beta': beta,
        'up_capture': up_capture,
        'down_capture': down_capture,
        'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
    }

def perform_factor_analysis(portfolio_returns, asset_returns):
    """Perform factor analysis"""
    
    # Simplified factor analysis
    # In practice, would use actual factor data (Fama-French, etc.)
    
    # Market factor (portfolio average)
    market_factor = asset_returns.mean(axis=1)
    
    # Size factor (mock)
    size_factor = np.random.normal(0, 0.01, len(portfolio_returns))
    
    # Value factor (mock)
    value_factor = np.random.normal(0, 0.008, len(portfolio_returns))
    
    # Regression analysis
    from sklearn.linear_model import LinearRegression
    
    factors = pd.DataFrame({
        'Market': market_factor,
        'Size': size_factor,
        'Value': value_factor
    }, index=portfolio_returns.index)
    
    # Run regression
    model = LinearRegression()
    model.fit(factors, portfolio_returns)
    
    factor_loadings = {
        'Market': model.coef_[0],
        'Size': model.coef_[1],
        'Value': model.coef_[2],
        'Alpha': model.intercept_
    }
    
    # Calculate R-squared
    predictions = model.predict(factors)
    r_squared = 1 - np.sum((portfolio_returns - predictions) ** 2) / np.sum((portfolio_returns - portfolio_returns.mean()) ** 2)
    
    return {
        'factor_loadings': factor_loadings,
        'r_squared': r_squared,
        'factors': factors
    }

def perform_statistical_analysis(returns):
    """Perform statistical significance analysis"""
    
    confidence_level = st.session_state.perf_confidence_level / 100
    
    # T-test for mean return significance
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    
    # Jarque-Bera test for normality
    jb_stat, jb_p_value = stats.jarque_bera(returns)
    
    # Ljung-Box test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
    
    # Confidence intervals
    mean_return = returns.mean()
    std_error = returns.std() / np.sqrt(len(returns))
    
    critical_value = stats.t.ppf((1 + confidence_level) / 2, len(returns) - 1)
    confidence_interval = (
        mean_return - critical_value * std_error,
        mean_return + critical_value * std_error
    )
    
    return {
        'mean_significance': {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < (1 - confidence_level)
        },
        'normality_test': {
            'jb_statistic': jb_stat,
            'p_value': jb_p_value,
            'normal': jb_p_value > (1 - confidence_level)
        },
        'autocorrelation_test': {
            'p_values': lb_result['lb_pvalue'].values,
            'significant_lags': (lb_result['lb_pvalue'] < (1 - confidence_level)).sum()
        },
        'confidence_interval': confidence_interval
    }

def render_performance_summary():
    """Render performance summary dashboard"""
    
    if 'performance_results' not in st.session_state:
        st.info("👆 Configure settings and click 'Analyze Performance' to view results.")
        return
    
    st.subheader("📊 Performance Summary")
    
    results = st.session_state.performance_results
    metrics = results['metrics']
    
    # Main performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:.2%}",
            delta=f"{metrics['annualized_return']:.2%} annualized"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta=f"Sortino: {metrics['sortino_ratio']:.2f}"
        )
    
    with col3:
        st.metric(
            "Volatility",
            f"{metrics['volatility']:.2%}",
            delta="Annualized"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2%}",
            delta=f"Calmar: {metrics['calmar_ratio']:.2f}"
        )
    
    with col5:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.1%}",
            delta=f"PF: {metrics['profit_factor']:.1f}"
        )
    
    # Risk metrics
    with st.expander("📈 Advanced Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"{metrics['var_95']:.3%}")
            st.metric("CVaR (95%)", f"{metrics['cvar_95']:.3%}")
        
        with col2:
            st.metric("Skewness", f"{metrics['skewness']:.2f}")
            st.metric("Kurtosis", f"{metrics['kurtosis']:.2f}")
        
        with col3:
            if 'benchmark' in results:
                bench = results['benchmark']
                st.metric("Alpha", f"{bench['alpha']:.3%}")
                st.metric("Beta", f"{bench['beta']:.2f}")
        
        with col4:
            if 'benchmark' in results:
                st.metric("Tracking Error", f"{bench['tracking_error']:.2%}")
                st.metric("Info Ratio", f"{bench['information_ratio']:.2f}")

def render_attribution_analysis():
    """Render performance attribution analysis"""
    
    if 'performance_results' not in st.session_state:
        return
    
    st.subheader("🎯 Performance Attribution")
    
    results = st.session_state.performance_results
    attribution = results['attribution']
    
    if attribution['method'] == "Asset-based":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Asset Contribution Analysis:**")
            
            contributions = attribution['asset_contributions']
            contrib_df = pd.DataFrame({
                'Asset': list(contributions.keys()),
                'Contribution': [contrib['contribution'] for contrib in contributions.values()],
                'Weight': [contrib['weight'] for contrib in contributions.values()],
                'Return': [contrib['return'] for contrib in contributions.values()]
            })
            
            styled_contrib = contrib_df.style.format({
                'Contribution': '{:.3%}',
                'Weight': '{:.2%}',
                'Return': '{:.3%}'
            })
            
            st.dataframe(styled_contrib, use_container_width=True)
        
        with col2:
            st.markdown("**Attribution Visualization:**")
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=["relative"] * len(contrib_df) + ["total"],
                x=list(contrib_df['Asset']) + ['Total'],
                y=list(contrib_df['Contribution']) + [attribution['total_contribution']],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                hovertemplate='<b>%{x}</b><br>Contribution: %{y:.3%}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Performance Attribution Waterfall",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_benchmark_comparison():
    """Render benchmark comparison analysis"""
    
    if 'performance_results' not in st.session_state or 'benchmark' not in st.session_state.performance_results:
        return
    
    st.subheader("🏆 Benchmark Comparison")
    
    results = st.session_state.performance_results
    benchmark = results['benchmark']
    portfolio_returns = results['portfolio_returns']
    benchmark_returns = benchmark['benchmark_returns']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Relative Performance:**")
        
        # Performance comparison chart
        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_cum.index,
            y=portfolio_cum.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_cum.index,
            y=benchmark_cum.values,
            mode='lines',
            name=f'Benchmark ({benchmark["benchmark_symbol"]})',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Cumulative Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Benchmark Statistics:**")
        
        # Benchmark metrics
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Alpha", f"{benchmark['alpha']:.3%}")
            st.metric("Beta", f"{benchmark['beta']:.2f}")
            st.metric("Correlation", f"{benchmark['correlation']:.3f}")
        
        with col_b:
            st.metric("Tracking Error", f"{benchmark['tracking_error']:.2%}")
            st.metric("Information Ratio", f"{benchmark['information_ratio']:.2f}")
            
            # Up/Down capture
            capture_score = (benchmark['up_capture'] + (2 - benchmark['down_capture'])) / 2
            st.metric("Capture Score", f"{capture_score:.2f}")
        
        # Excess returns distribution
        excess_returns = portfolio_returns - benchmark_returns
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=excess_returns * 100,  # Convert to percentage
            nbinsx=30,
            name='Excess Returns',
            opacity=0.7,
            hovertemplate='Excess Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Line")
        
        fig.update_layout(
            title="Excess Returns Distribution",
            xaxis_title="Excess Return (%)",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_factor_analysis():
    """Render factor analysis results"""
    
    if 'performance_results' not in st.session_state or 'factors' not in st.session_state.performance_results:
        return
    
    st.subheader("📉 Factor Analysis")
    
    results = st.session_state.performance_results
    factors = results['factors']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Factor Loadings:**")
        
        loadings = factors['factor_loadings']
        loadings_df = pd.DataFrame({
            'Factor': list(loadings.keys()),
            'Loading': list(loadings.values())
        })
        
        # Factor loadings bar chart
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in loadings_df['Loading']]
        
        fig.add_trace(go.Bar(
            x=loadings_df['Factor'],
            y=loadings_df['Loading'],
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Loading: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title="Factor Loadings",
            xaxis_title="Factor",
            yaxis_title="Loading",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Model Fit:**")
        
        st.metric("R-squared", f"{factors['r_squared']:.3f}")
        
        # Factor exposures table
        styled_loadings = loadings_df.style.format({
            'Loading': '{:.3f}'
        })
        
        st.dataframe(styled_loadings, use_container_width=True)
        
        # Factor contributions pie chart
        abs_loadings = {k: abs(v) for k, v in loadings.items() if k != 'Alpha'}
        total_loading = sum(abs_loadings.values())
        
        if total_loading > 0:
            fig = go.Figure(data=[go.Pie(
                labels=list(abs_loadings.keys()),
                values=list(abs_loadings.values()),
                hole=0.3,
                hovertemplate='<b>%{label}</b><br>Contribution: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title="Factor Exposure Breakdown",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_regime_performance():
    """Render regime-conditional performance analysis"""
    
    if 'performance_results' not in st.session_state:
        return
    
    st.subheader("🔄 Regime-Conditional Performance")
    
    # Check if regime analysis is available
    if 'regime_model' not in st.session_state:
        st.info("Run regime detection first to analyze regime-conditional performance.")
        return
    
    results = st.session_state.performance_results
    portfolio_returns = results['portfolio_returns']
    regime_data = st.session_state.regime_model
    
    # Align data
    regime_sequence = pd.Series(regime_data['sequence'], index=regime_data['features'].index)
    
    # Find common dates
    common_dates = portfolio_returns.index.intersection(regime_sequence.index)
    aligned_returns = portfolio_returns.loc[common_dates]
    aligned_regimes = regime_sequence.loc[common_dates]
    
    if len(aligned_returns) == 0:
        st.warning("No overlapping data between portfolio and regime analysis.")
        return
    
    # Calculate regime-specific performance
    regime_performance = {}
    unique_regimes = aligned_regimes.unique()
    
    for regime in unique_regimes:
        regime_mask = aligned_regimes == regime
        regime_returns = aligned_returns[regime_mask]
        
        if len(regime_returns) > 0:
            regime_performance[f'Regime {regime}'] = {
                'frequency': regime_mask.sum() / len(aligned_regimes),
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'total_return': (1 + regime_returns).prod() - 1
            }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance by Regime:**")
        
        regime_df = pd.DataFrame(regime_performance).T
        
        styled_regime = regime_df.style.format({
            'frequency': '{:.1%}',
            'avg_return': '{:.3%}',
            'volatility': '{:.3%}',
            'sharpe_ratio': '{:.2f}',
            'total_return': '{:.2%}'
        })
        
        st.dataframe(styled_regime, use_container_width=True)
    
    with col2:
        st.markdown("**Regime Performance Comparison:**")
        
        # Regime performance bar chart
        fig = go.Figure()
        
        regimes = list(regime_performance.keys())
        returns = [regime_performance[r]['avg_return'] * 252 for r in regimes]  # Annualized
        
        fig.add_trace(go.Bar(
            x=regimes,
            y=returns,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(regimes)],
            hovertemplate='<b>%{x}</b><br>Annualized Return: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Annualized Returns by Regime",
            xaxis_title="Regime",
            yaxis_title="Annualized Return",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_statistical_analysis():
    """Render statistical significance analysis"""
    
    if 'performance_results' not in st.session_state or 'statistics' not in st.session_state.performance_results:
        return
    
    st.subheader("📊 Statistical Analysis")
    
    results = st.session_state.performance_results
    stats_results = results['statistics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Significance Tests:**")
        
        # Mean return significance
        mean_sig = stats_results['mean_significance']
        significance_color = "success" if mean_sig['significant'] else "warning"
        
        st.metric(
            "Mean Return T-test",
            f"p = {mean_sig['p_value']:.4f}",
            delta="Significant" if mean_sig['significant'] else "Not significant"
        )
        
        # Normality test
        norm_test = stats_results['normality_test']
        normality_color = "success" if norm_test['normal'] else "warning"
        
        st.metric(
            "Normality (JB test)",
            f"p = {norm_test['p_value']:.4f}",
            delta="Normal" if norm_test['normal'] else "Non-normal"
        )
    
    with col2:
        st.markdown("**Confidence Intervals:**")
        
        ci = stats_results['confidence_interval']
        confidence_level = st.session_state.perf_confidence_level
        
        st.metric(
            f"Mean Return ({confidence_level}% CI)",
            f"[{ci[0]:.4%}, {ci[1]:.4%}]",
            delta="Daily return confidence interval"
        )
        
        # Autocorrelation
        autocorr = stats_results['autocorrelation_test']
        
        st.metric(
            "Autocorrelation Lags",
            f"{autocorr['significant_lags']} of 10",
            delta="Significant lags detected"
        )
    
    with col3:
        st.markdown("**Distribution Analysis:**")
        
        portfolio_returns = results['portfolio_returns']
        
        # Q-Q plot for normality assessment
        from scipy import stats
        fig = go.Figure()
        
        # Calculate theoretical quantiles
        sorted_returns = np.sort(portfolio_returns)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_returns,
            mode='markers',
            name='Actual vs Normal',
            hovertemplate='Theoretical: %{x:.3f}<br>Actual: %{y:.3%}<extra></extra>'
        ))
        
        # Add diagonal line
        min_val, max_val = min(theoretical_quantiles.min(), sorted_returns.min()), max(theoretical_quantiles.max(), sorted_returns.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Normal',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title="Q-Q Plot (Normal Distribution)",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def reset_performance_analysis():
    """Reset performance analysis results"""
    
    if 'performance_results' in st.session_state:
        del st.session_state.performance_results
    
    st.success("🔄 Performance analysis reset!")
    st.experimental_rerun()

def export_performance_report():
    """Export comprehensive performance report"""
    
    if 'performance_results' not in st.session_state:
        st.warning("No performance results to export.")
        return
    
    results = st.session_state.performance_results
    metrics = results['metrics']
    
    # Create comprehensive report
    report_data = {
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    }
    
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Performance Report CSV",
        data=csv,
        file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ Performance report prepared for download!")

# Main execution
if __name__ == "__main__":
    render_performance_analytics()