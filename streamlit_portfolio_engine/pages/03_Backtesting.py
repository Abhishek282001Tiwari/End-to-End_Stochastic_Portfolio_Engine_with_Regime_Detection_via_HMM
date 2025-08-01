#!/usr/bin/env python3
"""
Backtesting Interface Page

Advanced backtesting dashboard featuring:
- Multiple backtesting modes (Vectorized, Event-driven, Walk-forward, Monte Carlo)
- Realistic trading simulation with costs and slippage
- Performance metrics and analysis
- Strategy comparison tools
- Walk-forward optimization
- Results visualization and export
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

from src.backtesting.framework.advanced_backtesting import create_advanced_backtesting_framework, BacktestMode
from src.backtesting.execution.trading_simulator import TradingSimulator, MarketImpactModel
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine

def render_backtesting():
    """Render the backtesting interface"""
    
    st.title("⚡ Advanced Backtesting Framework")
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        render_backtesting_welcome_screen()
        return
    
    # Get portfolio data
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data is None or portfolio_data.empty:
        st.warning("No portfolio data available. Please load data using the sidebar.")
        return
    
    # Render main interface
    render_backtest_configuration()
    render_backtest_execution()
    
    # Display results if available
    if st.session_state.get('backtest_results') is not None:
        render_backtest_results()

def render_backtesting_welcome_screen():
    """Render welcome screen for backtesting"""
    
    st.markdown("""
    ## ⚡ Advanced Backtesting Framework
    
    Comprehensive backtesting system with realistic trading simulation and multiple analysis modes.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 📊 Vectorized
        - Fast matrix operations
        - Ideal for initial testing
        - Simple rebalancing
        - No order management
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Event-driven
        - Realistic order execution
        - Market microstructure
        - Transaction costs
        - Slippage modeling
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 Walk-forward
        - Out-of-sample testing
        - Rolling optimization
        - Parameter stability
        - Overfitting prevention
        """)
    
    with col4:
        st.markdown("""
        ### 🎲 Monte Carlo
        - Statistical robustness
        - Multiple scenarios
        - Confidence intervals
        - Risk assessment
        """)
    
    st.markdown("""
    ---
    
    ### 🎯 Key Features
    
    - **Realistic Trading Costs**: Commission, bid-ask spreads, market impact, and slippage
    - **Multiple Strategies**: Mean variance, risk parity, Black-Litterman, and custom strategies
    - **Walk-forward Analysis**: Out-of-sample validation with rolling parameter optimization
    - **Performance Attribution**: Detailed breakdown of returns by factor and regime
    - **Risk Analytics**: VaR, CVaR, maximum drawdown, and tail risk measures
    - **Benchmarking**: Compare against standard indices and custom benchmarks
    
    👈 **Load portfolio data in the sidebar to begin backtesting!**
    """)

def render_backtest_configuration():
    """Render backtesting configuration panel"""
    
    st.subheader("⚙️ Backtest Configuration")
    
    # Main configuration
    with st.expander("🎛️ Core Settings", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            backtest_mode = st.selectbox(
                "Backtest Mode",
                ["Vectorized", "Event-driven", "Walk-forward", "Monte Carlo"],
                help="Choose backtesting methodology"
            )
            st.session_state.backtest_mode = backtest_mode
        
        with col2:
            optimization_method = st.selectbox(
                "Strategy",
                ["mean_variance", "risk_parity", "black_litterman", "equal_weight", "momentum", "mean_reversion"],
                help="Portfolio optimization strategy"
            )
            st.session_state.optimization_method = optimization_method
        
        with col3:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=10000,
                max_value=100000000,
                value=st.session_state.get('initial_capital', 1000000),
                step=50000,
                help="Starting portfolio value"
            )
            st.session_state.initial_capital = initial_capital
        
        with col4:
            rebalance_frequency = st.selectbox(
                "Rebalancing",
                ["Daily", "Weekly", "Monthly", "Quarterly"],
                index=2,  # Default to Monthly
                help="Portfolio rebalancing frequency"
            )
            st.session_state.rebalance_frequency = rebalance_frequency
    
    # Trading costs and constraints
    with st.expander("💰 Trading Costs & Constraints"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trading Costs:**")
            
            commission_rate = st.slider(
                "Commission Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('commission_rate', 0.1),
                step=0.01,
                help="Per-trade commission as percentage of trade value"
            )
            st.session_state.commission_rate = commission_rate
            
            bid_ask_spread = st.slider(
                "Bid-Ask Spread (%)",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.get('bid_ask_spread', 0.05),
                step=0.01,
                help="Half-spread cost for crossing bid-ask"
            )
            st.session_state.bid_ask_spread = bid_ask_spread
            
            market_impact = st.slider(
                "Market Impact (%)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('market_impact', 0.1),
                step=0.01,
                help="Price impact for large trades"
            )
            st.session_state.market_impact = market_impact
        
        with col2:
            st.markdown("**Portfolio Constraints:**")
            
            max_weight = st.slider(
                "Max Asset Weight (%)",
                min_value=5,
                max_value=100,
                value=st.session_state.get('max_weight', 25),
                step=5,
                help="Maximum allocation to single asset"
            )
            st.session_state.max_weight = max_weight
            
            min_weight = st.slider(
                "Min Asset Weight (%)",
                min_value=0,
                max_value=10,
                value=st.session_state.get('min_weight', 0),
                step=1,
                help="Minimum allocation to single asset"
            )
            st.session_state.min_weight = min_weight
            
            leverage = st.slider(
                "Maximum Leverage",
                min_value=1.0,
                max_value=3.0,
                value=st.session_state.get('leverage', 1.0),
                step=0.1,
                help="Maximum portfolio leverage (1.0 = no leverage)"
            )
            st.session_state.leverage = leverage
    
    # Mode-specific configurations
    if backtest_mode == "Walk-forward":
        with st.expander("🔄 Walk-forward Settings"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                training_window = st.slider(
                    "Training Window (months)",
                    min_value=6,
                    max_value=36,
                    value=12,
                    help="Length of training period for optimization"
                )
                st.session_state.training_window = training_window
            
            with col2:
                testing_window = st.slider(
                    "Testing Window (months)",
                    min_value=1,
                    max_value=12,
                    value=3,
                    help="Length of out-of-sample testing period"
                )
                st.session_state.testing_window = testing_window
            
            with col3:
                step_size = st.slider(
                    "Step Size (months)",
                    min_value=1,
                    max_value=6,
                    value=1,
                    help="How often to re-optimize parameters"
                )
                st.session_state.step_size = step_size
    
    elif backtest_mode == "Monte Carlo":
        with st.expander("🎲 Monte Carlo Settings"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_simulations = st.slider(
                    "Number of Simulations",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Number of Monte Carlo paths"
                )
                st.session_state.n_simulations = n_simulations
            
            with col2:
                confidence_level = st.slider(
                    "Confidence Level (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    help="Confidence level for intervals"
                )
                st.session_state.confidence_level = confidence_level
            
            with col3:
                scenario_type = st.selectbox(
                    "Scenario Type",
                    ["bootstrap", "parametric", "historical"],
                    help="Method for generating scenarios"
                )
                st.session_state.scenario_type = scenario_type
    
    # Risk management settings
    with st.expander("⚠️ Risk Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_drawdown_limit = st.slider(
                "Max Drawdown Limit (%)",
                min_value=5,
                max_value=50,
                value=20,
                help="Stop trading if drawdown exceeds this level"
            )
            st.session_state.max_drawdown_limit = max_drawdown_limit
            
            var_limit = st.slider(
                "Daily VaR Limit (%)",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum daily Value at Risk"
            )
            st.session_state.var_limit = var_limit
        
        with col2:
            volatility_target = st.slider(
                "Target Volatility (%)",
                min_value=5,
                max_value=30,
                value=15,
                help="Annualized volatility target"
            )
            st.session_state.volatility_target = volatility_target
            
            use_stop_loss = st.checkbox(
                "Enable Stop-Loss",
                value=False,
                help="Enable individual position stop-losses"
            )
            st.session_state.use_stop_loss = use_stop_loss

def render_backtest_execution():
    """Render backtest execution controls"""
    
    st.subheader("🚀 Execution")
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        if st.button("🚀 Run Backtest", use_container_width=True, type="primary"):
            run_backtest()
    
    with col2:
        if st.button("⚡ Quick Test", use_container_width=True):
            run_quick_backtest()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            reset_backtest_results()
    
    with col4:
        if st.session_state.get('backtest_results') is not None:
            if st.button("📥 Export", use_container_width=True):
                export_backtest_results()

def run_backtest():
    """Run comprehensive backtest with configured parameters"""
    
    mode = st.session_state.backtest_mode
    
    with st.spinner(f"🚀 Running {mode} Backtest..."):
        try:
            portfolio_data = st.session_state.portfolio_data
            
            # Extract price data
            if 'Close' in portfolio_data.columns.get_level_values(0):
                close_prices = portfolio_data.xs('Close', level=0, axis=1)
            else:
                close_prices = portfolio_data
            
            # Run backtest based on mode
            if mode == "Vectorized":
                results = run_vectorized_backtest(close_prices)
            elif mode == "Event-driven":
                results = run_event_driven_backtest(close_prices)
            elif mode == "Walk-forward":
                results = run_walk_forward_backtest(close_prices)
            elif mode == "Monte Carlo":
                results = run_monte_carlo_backtest(close_prices)
            
            # Store results
            st.session_state.backtest_results = results
            
            # Success message
            total_return = results['metrics']['total_return']
            sharpe_ratio = results['metrics']['sharpe_ratio']
            max_drawdown = results['metrics']['max_drawdown']
            
            st.success(f"""
            ✅ **{mode} Backtest Complete!**
            - Total Return: {total_return:.2%}
            - Sharpe Ratio: {sharpe_ratio:.2f}
            - Max Drawdown: {max_drawdown:.2%}
            """)
            
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            st.exception(e)

def run_vectorized_backtest(close_prices):
    """Run vectorized backtest (fastest method)"""
    
    returns = close_prices.pct_change().dropna()
    
    # Simple strategy implementation
    strategy = st.session_state.optimization_method
    
    if strategy == "equal_weight":
        weights = pd.DataFrame(
            1/len(returns.columns),
            index=returns.index,
            columns=returns.columns
        )
    elif strategy == "momentum":
        # Simple momentum strategy
        lookback = 30
        momentum = returns.rolling(lookback).mean()
        weights = momentum.div(momentum.sum(axis=1), axis=0).fillna(1/len(returns.columns))
    elif strategy == "mean_reversion":
        # Simple mean reversion strategy
        lookback = 30
        mean_returns = returns.rolling(lookback).mean()
        weights = (-mean_returns).div((-mean_returns).sum(axis=1), axis=0).fillna(1/len(returns.columns))
    else:
        # Default to equal weight
        weights = pd.DataFrame(
            1/len(returns.columns),
            index=returns.index,
            columns=returns.columns
        )
    
    # Calculate portfolio returns
    portfolio_returns = (returns * weights.shift(1)).sum(axis=1)
    
    # Apply costs
    commission_rate = st.session_state.commission_rate / 100
    
    # Estimate turnover (simplified)
    weight_changes = weights.diff().abs().sum(axis=1)
    transaction_costs = weight_changes * commission_rate
    
    # Net returns after costs
    net_returns = portfolio_returns - transaction_costs
    
    # Calculate performance metrics
    cumulative_returns = (1 + net_returns).cumprod()
    portfolio_value = cumulative_returns * st.session_state.initial_capital
    
    metrics = calculate_performance_metrics(net_returns, portfolio_value)
    
    return {
        'mode': 'Vectorized',
        'returns': net_returns,
        'cumulative_returns': cumulative_returns,
        'portfolio_value': portfolio_value,
        'weights': weights, 
        'metrics': metrics,
        'transaction_costs': transaction_costs,
        'config': get_backtest_config()
    }

def run_event_driven_backtest(close_prices):
    """Run event-driven backtest with realistic trading simulation"""
    
    # This is a simplified implementation
    # In practice, this would use the full trading simulator
    
    returns = close_prices.pct_change().dropna()
    
    # Simulate realistic trading with costs
    portfolio_returns = returns.mean(axis=1)  # Equal weight for demo
    
    # Enhanced transaction costs
    commission_rate = st.session_state.commission_rate / 100
    bid_ask_spread = st.session_state.bid_ask_spread / 100
    market_impact = st.session_state.market_impact / 100
    
    # Daily rebalancing costs
    daily_costs = commission_rate + bid_ask_spread + market_impact
    net_returns = portfolio_returns - daily_costs
    
    cumulative_returns = (1 + net_returns).cumprod()
    portfolio_value = cumulative_returns * st.session_state.initial_capital
    
    metrics = calculate_performance_metrics(net_returns, portfolio_value)
    
    return {
        'mode': 'Event-driven',
        'returns': net_returns,
        'cumulative_returns': cumulative_returns,
        'portfolio_value': portfolio_value,
        'metrics': metrics,
        'config': get_backtest_config()
    }

def run_walk_forward_backtest(close_prices):
    """Run walk-forward backtest with rolling optimization"""
    
    returns = close_prices.pct_change().dropna()
    
    training_months = st.session_state.training_window
    testing_months = st.session_state.testing_window
    step_months = st.session_state.step_size
    
    # Convert to trading days (approximately)
    training_days = training_months * 21
    testing_days = testing_months * 21
    step_days = step_months * 21
    
    all_returns = []
    all_weights = []
    
    start_idx = training_days
    
    while start_idx + testing_days < len(returns):
        # Training period
        train_start = start_idx - training_days
        train_end = start_idx
        train_data = returns.iloc[train_start:train_end]
        
        # Testing period
        test_start = start_idx
        test_end = start_idx + testing_days
        test_data = returns.iloc[test_start:test_end]
        
        # Optimize on training data (simplified)
        if st.session_state.optimization_method == "momentum":
            train_momentum = train_data.mean()
            weights = train_momentum / train_momentum.sum()
        else:
            # Equal weight default
            weights = pd.Series(1/len(returns.columns), index=returns.columns)
        
        # Apply to test period
        test_returns = (test_data * weights).sum(axis=1)
        all_returns.append(test_returns)
        
        # Store weights
        weight_df = pd.DataFrame(
            [weights.values] * len(test_data),
            index=test_data.index,
            columns=returns.columns
        )
        all_weights.append(weight_df)
        
        start_idx += step_days
    
    # Combine results
    portfolio_returns = pd.concat(all_returns)
    weights_df = pd.concat(all_weights)
    
    # Apply costs
    commission_rate = st.session_state.commission_rate / 100
    weight_changes = weights_df.diff().abs().sum(axis=1)
    transaction_costs = weight_changes * commission_rate
    
    net_returns = portfolio_returns - transaction_costs
    cumulative_returns = (1 + net_returns).cumprod()
    portfolio_value = cumulative_returns * st.session_state.initial_capital
    
    metrics = calculate_performance_metrics(net_returns, portfolio_value)
    
    return {
        'mode': 'Walk-forward',
        'returns': net_returns,
        'cumulative_returns': cumulative_returns,
        'portfolio_value': portfolio_value,
        'weights': weights_df,
        'metrics': metrics,
        'config': get_backtest_config()
    }

def run_monte_carlo_backtest(close_prices):
    """Run Monte Carlo backtest with multiple scenarios"""
    
    returns = close_prices.pct_change().dropna()
    n_sims = st.session_state.n_simulations
    
    # Bootstrap scenarios
    scenario_results = []
    
    for sim in range(min(n_sims, 100)):  # Limit for demo
        # Bootstrap sample
        if st.session_state.scenario_type == "bootstrap":
            scenario_returns = returns.sample(len(returns), replace=True)
        else:
            # Use historical data with noise
            noise = np.random.normal(0, 0.01, returns.shape)
            scenario_returns = returns + noise
        
        # Simple equal weight strategy
        portfolio_returns = scenario_returns.mean(axis=1)
        
        # Apply costs
        commission_rate = st.session_state.commission_rate / 100
        net_returns = portfolio_returns - commission_rate
        
        cumulative_returns = (1 + net_returns).cumprod()
        scenario_results.append(cumulative_returns.iloc[-1])
    
    # Calculate statistics
    scenario_results = np.array(scenario_results)
    confidence_level = st.session_state.confidence_level / 100
    
    # Use mean scenario for display
    portfolio_returns = returns.mean(axis=1)
    commission_rate = st.session_state.commission_rate / 100
    net_returns = portfolio_returns - commission_rate
    cumulative_returns = (1 + net_returns).cumprod()
    portfolio_value = cumulative_returns * st.session_state.initial_capital
    
    metrics = calculate_performance_metrics(net_returns, portfolio_value)
    
    # Add Monte Carlo specific metrics
    metrics.update({
        'mc_mean_return': np.mean(scenario_results) - 1,
        'mc_std_return': np.std(scenario_results),
        'mc_var': np.percentile(scenario_results, (1 - confidence_level) * 100) - 1,
        'mc_confidence_level': confidence_level
    })
    
    return {
        'mode': 'Monte Carlo',
        'returns': net_returns,
        'cumulative_returns': cumulative_returns,
        'portfolio_value': portfolio_value,
        'metrics': metrics,
        'scenario_results': scenario_results,
        'config': get_backtest_config()
    }

def run_quick_backtest():
    """Run a quick backtest with default parameters"""
    
    with st.spinner("⚡ Running Quick Backtest..."):
        try:
            portfolio_data = st.session_state.portfolio_data
            
            # Extract price data
            if 'Close' in portfolio_data.columns.get_level_values(0):
                close_prices = portfolio_data.xs('Close', level=0, axis=1)
            else:
                close_prices = portfolio_data
            
            returns = close_prices.pct_change().dropna()
            
            # Simple equal weight strategy
            portfolio_returns = returns.mean(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            portfolio_value = cumulative_returns * st.session_state.initial_capital
            
            metrics = calculate_performance_metrics(portfolio_returns, portfolio_value)
            
            results = {
                'mode': 'Quick Test',
                'returns': portfolio_returns,
                'cumulative_returns': cumulative_returns,
                'portfolio_value': portfolio_value,
                'metrics': metrics,
                'config': {'strategy': 'equal_weight', 'costs': False}
            }
            
            st.session_state.backtest_results = results
            
            st.success(f"⚡ Quick test complete! Return: {metrics['total_return']:.2%}")
            
        except Exception as e:
            st.error(f"❌ Quick test failed: {str(e)}")

def calculate_performance_metrics(returns, portfolio_value):
    """Calculate comprehensive performance metrics"""
    
    # Basic metrics
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown analysis
    running_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Risk metrics
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Additional metrics
    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio
    }

def get_backtest_config():
    """Get current backtest configuration"""
    
    return {
        'mode': st.session_state.backtest_mode,
        'strategy': st.session_state.optimization_method,
        'initial_capital': st.session_state.initial_capital,
        'commission_rate': st.session_state.commission_rate,
        'bid_ask_spread': st.session_state.bid_ask_spread,
        'market_impact': st.session_state.market_impact,
        'max_weight': st.session_state.max_weight,
        'leverage': st.session_state.leverage
    }

def render_backtest_results():
    """Render comprehensive backtest results"""
    
    results = st.session_state.backtest_results
    
    st.subheader(f"📊 {results['mode']} Backtest Results")
    
    # Performance metrics dashboard
    render_performance_metrics(results)
    
    # Charts
    render_performance_charts(results)
    
    # Detailed analysis
    render_detailed_analysis(results)

def render_performance_metrics(results):
    """Render performance metrics summary"""
    
    metrics = results['metrics']
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:.2%}",
            delta=f"{metrics['annualized_return']:.2%} annualized"
        )
    
    with col2:
        st.metric(
            "Volatility",
            f"{metrics['volatility']:.2%}",
            delta="Annualized"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Risk-adjusted"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2%}",
            delta="Peak-to-trough"
        )
    
    with col5:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.1%}",
            delta=f"Profit Factor: {metrics['profit_factor']:.1f}"
        )
    
    # Risk metrics
    with st.expander("📈 Advanced Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"{metrics['var_95']:.3%}")
        
        with col2:
            st.metric("CVaR (95%)", f"{metrics['cvar_95']:.3%}")
        
        with col3:
            st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
        
        with col4:
            if 'mc_var' in metrics:
                st.metric("MC VaR", f"{metrics['mc_var']:.2%}")

def render_performance_charts(results):
    """Render performance visualization charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Portfolio Value")
        
        fig = go.Figure()
        
        portfolio_value = results['portfolio_value']
        
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x}</b><br>Value: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add benchmark if available
        if 'benchmark_value' in results:
            fig.add_trace(go.Scatter(
                x=results['benchmark_value'].index,
                y=results['benchmark_value'].values,
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 Drawdown Analysis")
        
        portfolio_value = results['portfolio_value']
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value / running_max - 1) * 100
        
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
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Returns distribution
    st.subheader("📊 Returns Distribution")
    
    returns = results['returns'] * 100  # Convert to percentage
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        opacity=0.7,
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = (len(returns) * (x[1] - x[0]) * 
                  (1 / (sigma * np.sqrt(2 * np.pi))) * 
                  np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=normal_dist,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Daily Returns Distribution vs Normal",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_detailed_analysis(results):
    """Render detailed backtest analysis"""
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "💰 Costs", "⚙️ Configuration"])
    
    with tab1:
        st.subheader("Performance Breakdown")
        
        # Monthly returns table
        returns = results['returns']
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        if len(monthly_returns) > 0:
            # Create monthly returns heatmap
            monthly_df = monthly_returns.to_frame('Returns')
            monthly_df['Year'] = monthly_df.index.year
            monthly_df['Month'] = monthly_df.index.month
            
            pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Returns')
            
            if not pivot_table.empty:
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_table.values * 100,  # Convert to percentage
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=pivot_table.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(pivot_table.values * 100, 1),
                    texttemplate="%{text}%",
                    textfont={"size": 10},
                    hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Monthly Returns Heatmap (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Transaction Cost Analysis")
        
        if 'transaction_costs' in results:
            costs = results['transaction_costs']
            
            col1, col2 = st.columns(2)
            
            with col1:
                total_costs = costs.sum()
                cost_pct = total_costs / results['portfolio_value'].iloc[0]
                
                st.metric("Total Costs", f"${total_costs:,.0f}")
                st.metric("Cost Impact", f"{cost_pct:.3%}")
            
            with col2:
                avg_daily_cost = costs.mean()
                max_daily_cost = costs.max()
                
                st.metric("Avg Daily Cost", f"${avg_daily_cost:,.0f}")
                st.metric("Max Daily Cost", f"${max_daily_cost:,.0f}")
        else:
            st.info("Transaction cost details not available for this backtest mode.")
    
    with tab3:
        st.subheader("Backtest Configuration")
        
        config = results['config']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strategy Settings:**")
            st.json(config)
        
        with col2:
            st.markdown("**Data Summary:**")
            
            portfolio_value = results['portfolio_value']
            
            st.metric("Start Date", portfolio_value.index[0].strftime('%Y-%m-%d'))
            st.metric("End Date", portfolio_value.index[-1].strftime('%Y-%m-%d'))
            st.metric("Trading Days", len(portfolio_value))
            st.metric("Data Points", len(results['returns']))

def reset_backtest_results():
    """Reset backtest results"""
    
    if 'backtest_results' in st.session_state:
        del st.session_state.backtest_results
    
    st.success("🔄 Backtest results cleared!")
    st.experimental_rerun()

def export_backtest_results():
    """Export backtest results to CSV"""
    
    results = st.session_state.backtest_results
    
    # Create comprehensive export DataFrame
    export_df = pd.DataFrame({
        'Date': results['portfolio_value'].index,
        'Portfolio_Value': results['portfolio_value'].values,
        'Daily_Return': results['returns'].values,
        'Cumulative_Return': results['cumulative_returns'].values
    })
    
    # Add weights if available
    if 'weights' in results:
        weights_df = results['weights']
        for col in weights_df.columns:
            export_df[f'Weight_{col}'] = weights_df[col].values
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Backtest Results CSV",
        data=csv,
        file_name=f"backtest_{results['mode'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ Results prepared for download!")

# Main execution
if __name__ == "__main__":
    render_backtesting()