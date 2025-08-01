#!/usr/bin/env python3
"""
Monte Carlo Simulations Dashboard Page

Comprehensive Monte Carlo analysis featuring:
- Multiple stochastic processes (GBM, Jump Diffusion, Mean Reversion, Heston)
- Scenario analysis and stress testing
- Confidence interval estimation
- Path-dependent option pricing
- Portfolio optimization under uncertainty
- Risk scenario generation
- Statistical robustness testing
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

from src.simulation.monte_carlo_engine import MonteCarloEngine, SimulationConfig
from src.simulation.stochastic_processes import GeometricBrownianMotion, JumpDiffusionProcess, MeanReversionProcess
from streamlit_portfolio_engine.utils import safe_rerun

def render_monte_carlo():
    """Render the Monte Carlo simulations dashboard"""
    
    st.title("🎲 Monte Carlo Simulations Dashboard")
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        render_monte_carlo_welcome_screen()
        return
    
    # Get portfolio data
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data is None or portfolio_data.empty:
        st.warning("No portfolio data available. Please load data using the sidebar.")
        return
    
    # Render main dashboard
    render_simulation_controls()
    render_simulation_execution()
    
    # Display results if available
    if st.session_state.get('monte_carlo_results') is not None:
        render_simulation_results()

def render_monte_carlo_welcome_screen():
    """Render welcome screen for Monte Carlo simulations"""
    
    st.markdown("""
    ## 🎲 Advanced Monte Carlo Simulation Engine
    
    Sophisticated stochastic modeling and scenario analysis for portfolio risk assessment.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 📈 Geometric Brownian Motion
        - Standard equity modeling
        - Constant drift and volatility
        - Log-normal distributions
        - Efficient simulation
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Jump Diffusion
        - Sudden price movements
        - Market crash modeling
        - Mixed continuous-discrete
        - Extreme event analysis
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 Mean Reversion
        - Interest rate modeling
        - Volatility processes
        - Ornstein-Uhlenbeck
        - Commodity pricing
        """)
    
    with col4:
        st.markdown("""
        ### 📊 Heston Model
        - Stochastic volatility
        - Volatility clustering
        - Options pricing
        - Advanced modeling
        """)
    
    st.markdown("""
    ---
    
    ### 🎯 Key Applications
    
    - **Portfolio Optimization**: Robust optimization under uncertainty
    - **Risk Assessment**: VaR, CVaR, and tail risk estimation
    - **Scenario Analysis**: Stress testing and what-if analysis
    - **Options Pricing**: Path-dependent derivatives valuation
    - **Confidence Intervals**: Statistical significance testing
    - **Model Validation**: Backtesting and parameter stability
    
    👈 **Load portfolio data in the sidebar to begin Monte Carlo analysis!**
    """)

def render_simulation_controls():
    """Render Monte Carlo simulation controls and settings"""
    
    st.subheader("⚙️ Simulation Configuration")
    
    # Main simulation parameters
    with st.expander("🎛️ Core Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_simulations = st.slider(
                "Number of Simulations",
                min_value=100,
                max_value=10000,
                value=st.session_state.get('mc_n_simulations', 1000),
                step=100,
                help="Number of Monte Carlo paths to generate"
            )
            st.session_state.mc_n_simulations = n_simulations
        
        with col2:
            time_horizon = st.slider(
                "Time Horizon (days)",
                min_value=30,
                max_value=1000,
                value=st.session_state.get('mc_time_horizon', 252),
                help="Simulation time horizon"
            )
            st.session_state.mc_time_horizon = time_horizon
        
        with col3:
            time_steps = st.slider(
                "Time Steps",
                min_value=50,
                max_value=1000,
                value=st.session_state.get('mc_time_steps', time_horizon),
                help="Number of time steps in simulation"
            )
            st.session_state.mc_time_steps = time_steps
        
        with col4:
            confidence_levels = st.multiselect(
                "Confidence Levels (%)",
                [90, 95, 97.5, 99, 99.5],
                default=st.session_state.get('mc_confidence_levels', [95, 99]),
                help="Confidence intervals to calculate"
            )
            st.session_state.mc_confidence_levels = confidence_levels
    
    # Stochastic process selection
    with st.expander("📊 Stochastic Process Configuration"):
        process_type = st.selectbox(
            "Primary Process",
            ["Geometric Brownian Motion", "Jump Diffusion", "Mean Reversion", "Heston Model", "Mixed Process"],
            help="Choose the stochastic process for simulation"
        )
        st.session_state.mc_process_type = process_type
        
        col1, col2 = st.columns(2)
        
        with col1:
            if process_type in ["Geometric Brownian Motion", "Jump Diffusion", "Mixed Process"]:
                st.markdown("**GBM Parameters:**")
                
                drift = st.slider(
                    "Drift (μ)",
                    min_value=-0.5,
                    max_value=0.5,
                    value=st.session_state.get('mc_drift', 0.08),
                    step=0.01,
                    help="Annual expected return"
                )
                st.session_state.mc_drift = drift
                
                volatility = st.slider(
                    "Volatility (σ)",
                    min_value=0.05,
                    max_value=1.0,
                    value=st.session_state.get('mc_volatility', 0.2),
                    step=0.01,
                    help="Annual volatility"
                )
                st.session_state.mc_volatility = volatility
            
            if process_type in ["Jump Diffusion", "Mixed Process"]:
                st.markdown("**Jump Parameters:**")
                
                jump_intensity = st.slider(
                    "Jump Intensity (λ)",
                    min_value=0.0,
                    max_value=10.0,
                    value=st.session_state.get('mc_jump_intensity', 2.0),
                    step=0.1,
                    help="Average number of jumps per year"
                )
                st.session_state.mc_jump_intensity = jump_intensity
                
                jump_size_mean = st.slider(
                    "Jump Size Mean",
                    min_value=-0.2,
                    max_value=0.2,
                    value=st.session_state.get('mc_jump_size_mean', -0.05),
                    step=0.01,
                    help="Average jump size"
                )
                st.session_state.mc_jump_size_mean = jump_size_mean
        
        with col2:
            if process_type in ["Mean Reversion", "Mixed Process"]:
                st.markdown("**Mean Reversion Parameters:**")
                
                mean_reversion_speed = st.slider(
                    "Reversion Speed (κ)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.get('mc_reversion_speed', 2.0),
                    step=0.1,
                    help="Speed of mean reversion"
                )
                st.session_state.mc_reversion_speed = mean_reversion_speed
                
                long_term_mean = st.slider(
                    "Long-term Mean (θ)",
                    min_value=-0.2,
                    max_value=0.5,
                    value=st.session_state.get('mc_long_term_mean', 0.08),
                    step=0.01,
                    help="Long-term equilibrium level"
                )
                st.session_state.mc_long_term_mean = long_term_mean
            
            if process_type in ["Heston Model", "Mixed Process"]:
                st.markdown("**Heston Parameters:**")
                
                vol_of_vol = st.slider(
                    "Vol of Vol (σ_v)",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.get('mc_vol_of_vol', 0.3),
                    step=0.05,
                    help="Volatility of volatility"
                )
                st.session_state.mc_vol_of_vol = vol_of_vol
                
                correlation = st.slider(
                    "Correlation (ρ)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=st.session_state.get('mc_correlation', -0.5),
                    step=0.05,
                    help="Correlation between price and volatility"
                )
                st.session_state.mc_correlation = correlation
    
    # Scenario analysis settings
    with st.expander("🧪 Scenario Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            enable_scenarios = st.checkbox(
                "Enable Scenario Analysis",
                value=st.session_state.get('mc_enable_scenarios', False),
                help="Run multiple economic scenarios"
            )
            st.session_state.mc_enable_scenarios = enable_scenarios
            
            if enable_scenarios:
                scenario_types = st.multiselect(
                    "Scenario Types",
                    ["Base Case", "Bull Market", "Bear Market", "High Volatility", "Crisis", "Recovery"],
                    default=["Base Case", "Bear Market", "High Volatility"],
                    help="Economic scenarios to simulate"
                )
                st.session_state.mc_scenario_types = scenario_types
        
        with col2:
            enable_stress_tests = st.checkbox(
                "Enable Stress Testing",
                value=st.session_state.get('mc_enable_stress', False),
                help="Add extreme market stress scenarios"
            )
            st.session_state.mc_enable_stress = enable_stress_tests
            
            if enable_stress_tests:
                stress_magnitude = st.slider(
                    "Stress Magnitude",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Multiplier for stress scenario severity"
                )
                st.session_state.mc_stress_magnitude = stress_magnitude
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=st.session_state.get('mc_random_seed', 42),
                help="Seed for reproducible results"
            )
            st.session_state.mc_random_seed = random_seed
            
            antithetic_variates = st.checkbox(
                "Antithetic Variates",
                value=st.session_state.get('mc_antithetic', True),
                help="Use antithetic variates for variance reduction"
            )
            st.session_state.mc_antithetic = antithetic_variates
        
        with col2:
            control_variates = st.checkbox(
                "Control Variates",
                value=st.session_state.get('mc_control_variates', False),
                help="Use control variates for variance reduction"
            )
            st.session_state.mc_control_variates = control_variates
            
            parallel_processing = st.checkbox(
                "Parallel Processing",
                value=st.session_state.get('mc_parallel', True),
                help="Use multiple cores for faster simulation"
            )
            st.session_state.mc_parallel = parallel_processing

def render_simulation_execution():
    """Render simulation execution controls"""
    
    st.subheader("🚀 Simulation Execution")
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        if st.button("🎲 Run Monte Carlo Simulation", use_container_width=True, type="primary"):
            run_monte_carlo_simulation()
    
    with col2:
        if st.button("⚡ Quick Run", use_container_width=True):
            run_quick_simulation()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            reset_simulation_results()
    
    with col4:
        if st.session_state.get('monte_carlo_results') is not None:
            if st.button("📥 Export", use_container_width=True):
                export_simulation_results()

def run_monte_carlo_simulation():
    """Run comprehensive Monte Carlo simulation"""
    
    n_sims = st.session_state.mc_n_simulations
    
    with st.spinner(f"🎲 Running {n_sims:,} Monte Carlo simulations..."):
        try:
            portfolio_data = st.session_state.portfolio_data
            
            # Extract price data
            if 'Close' in portfolio_data.columns.get_level_values(0):
                close_prices = portfolio_data.xs('Close', level=0, axis=1)
            else:
                close_prices = portfolio_data
            
            # Calculate returns for parameter estimation
            returns = close_prices.pct_change().dropna()
            portfolio_returns = returns.mean(axis=1)
            
            # Estimate parameters from historical data
            historical_params = estimate_parameters(portfolio_returns)
            
            # Run simulations based on selected process
            if st.session_state.mc_enable_scenarios:
                simulation_results = run_scenario_simulations(historical_params)
            else:
                simulation_results = run_single_process_simulation(historical_params)
            
            # Calculate statistics and confidence intervals
            statistics = calculate_simulation_statistics(simulation_results)
            
            # Risk metrics
            risk_metrics = calculate_monte_carlo_risk_metrics(simulation_results)
            
            # Store results
            st.session_state.monte_carlo_results = {
                'process_type': st.session_state.mc_process_type,
                'parameters': historical_params,
                'simulations': simulation_results,
                'statistics': statistics,
                'risk_metrics': risk_metrics,
                'config': get_simulation_config(),
                'run_time': datetime.now()
            }
            
            # Success message
            final_values = simulation_results['final_values']
            mean_return = np.mean(final_values) - 1
            std_return = np.std(final_values)
            
            st.success(f"""
            ✅ **Monte Carlo Simulation Complete!**
            - Simulations: {n_sims:,}
            - Mean Return: {mean_return:.2%}
            - Std Deviation: {std_return:.2%}
            - Process: {st.session_state.mc_process_type}
            """)
            
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            st.exception(e)

def estimate_parameters(returns):
    """Estimate model parameters from historical data"""
    
    # Basic statistics
    mu = returns.mean() * 252  # Annualized
    sigma = returns.std() * np.sqrt(252)  # Annualized
    
    # Override with user inputs if provided
    if hasattr(st.session_state, 'mc_drift'):
        mu = st.session_state.mc_drift
    if hasattr(st.session_state, 'mc_volatility'):
        sigma = st.session_state.mc_volatility
    
    params = {
        'mu': mu,
        'sigma': sigma,
        'initial_value': 1.0  # Normalized starting value
    }
    
    # Add process-specific parameters
    process_type = st.session_state.mc_process_type
    
    if "Jump Diffusion" in process_type:
        params.update({
            'jump_intensity': st.session_state.get('mc_jump_intensity', 2.0),
            'jump_size_mean': st.session_state.get('mc_jump_size_mean', -0.05),
            'jump_size_std': 0.1  # Default
        })
    
    if "Mean Reversion" in process_type:
        params.update({
            'kappa': st.session_state.get('mc_reversion_speed', 2.0),
            'theta': st.session_state.get('mc_long_term_mean', mu),
        })
    
    if "Heston" in process_type:
        params.update({
            'v0': sigma**2,  # Initial variance
            'kappa_v': st.session_state.get('mc_reversion_speed', 2.0),
            'theta_v': sigma**2,  # Long-term variance
            'sigma_v': st.session_state.get('mc_vol_of_vol', 0.3),
            'rho': st.session_state.get('mc_correlation', -0.5)
        })
    
    return params

def run_single_process_simulation(params):
    """Run simulation for single stochastic process"""
    
    n_sims = st.session_state.mc_n_simulations
    T = st.session_state.mc_time_horizon / 252  # Convert to years
    n_steps = st.session_state.mc_time_steps
    dt = T / n_steps
    
    process_type = st.session_state.mc_process_type
    
    # Set random seed for reproducibility
    np.random.seed(st.session_state.mc_random_seed)
    
    # Generate random numbers
    if st.session_state.mc_antithetic:
        # Use antithetic variates
        n_base_sims = n_sims // 2
        randn1 = np.random.randn(n_base_sims, n_steps)
        randn2 = -randn1  # Antithetic variates
        random_numbers = np.vstack([randn1, randn2])
        if n_sims % 2 == 1:
            # Add one more simulation if odd number
            random_numbers = np.vstack([random_numbers, np.random.randn(1, n_steps)])
    else:
        random_numbers = np.random.randn(n_sims, n_steps)
    
    # Initialize results
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = params['initial_value']
    
    # Run simulation based on process type
    if process_type == "Geometric Brownian Motion":
        paths = simulate_gbm(paths, params, dt, random_numbers)
    elif process_type == "Jump Diffusion":
        paths = simulate_jump_diffusion(paths, params, dt, random_numbers)
    elif process_type == "Mean Reversion":
        paths = simulate_mean_reversion(paths, params, dt, random_numbers)
    elif process_type == "Heston Model":
        paths = simulate_heston(paths, params, dt, random_numbers)
    else:
        # Default to GBM
        paths = simulate_gbm(paths, params, dt, random_numbers)
    
    return {
        'paths': paths,
        'final_values': paths[:, -1],
        'time_steps': np.linspace(0, T, n_steps + 1) * 252,  # Convert back to days
        'process_type': process_type
    }

def simulate_gbm(paths, params, dt, random_numbers):
    """Simulate Geometric Brownian Motion"""
    
    mu = params['mu']
    sigma = params['sigma']
    
    for t in range(1, paths.shape[1]):
        dW = random_numbers[:, t-1] * np.sqrt(dt)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    return paths

def simulate_jump_diffusion(paths, params, dt, random_numbers):
    """Simulate Jump Diffusion Process (Merton Model)"""
    
    mu = params['mu']
    sigma = params['sigma']
    lambda_j = params['jump_intensity']
    mu_j = params['jump_size_mean']
    sigma_j = params.get('jump_size_std', 0.1)
    
    n_sims, n_steps = random_numbers.shape
    
    for t in range(1, paths.shape[1]):
        # Diffusion component
        dW = random_numbers[:, t-1] * np.sqrt(dt)
        diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW
        
        # Jump component
        jump_prob = lambda_j * dt
        jumps = np.random.poisson(jump_prob, n_sims)
        jump_sizes = np.random.normal(mu_j, sigma_j, n_sims) * jumps
        
        paths[:, t] = paths[:, t-1] * np.exp(diffusion + jump_sizes)
    
    return paths

def simulate_mean_reversion(paths, params, dt, random_numbers):
    """Simulate Mean Reverting Process (Ornstein-Uhlenbeck)"""
    
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    
    # Convert to log returns for mean reversion
    log_paths = np_log(np.maximum(paths, 1e-10))
    
    for t in range(1, log_paths.shape[1]):
        dW = random_numbers[:, t-1] * np.sqrt(dt)
        log_paths[:, t] = (log_paths[:, t-1] + 
                          kappa * (np_log(theta) - log_paths[:, t-1]) * dt + 
                          sigma * dW)
    
    return np.exp(log_paths)

def simulate_heston(paths, params, dt, random_numbers):
    """Simulate Heston Stochastic Volatility Model"""
    
    # This is a simplified implementation
    # In practice, would use more sophisticated numerical schemes
    
    mu = params['mu']
    v0 = params['v0']
    kappa_v = params['kappa_v']
    theta_v = params['theta_v']
    sigma_v = params['sigma_v']
    rho = params['rho']
    
    n_sims, n_steps = random_numbers.shape
    
    # Initialize variance paths
    v_paths = np.zeros((n_sims, n_steps + 1))
    v_paths[:, 0] = v0
    
    # Generate correlated random numbers for volatility
    random_v = (rho * random_numbers + 
                np.sqrt(1 - rho**2) * np.random.randn(n_sims, n_steps))
    
    for t in range(1, paths.shape[1]):
        # Volatility process (CIR)
        dW_v = random_v[:, t-1] * np.sqrt(dt)
        v_paths[:, t] = np.maximum(
            v_paths[:, t-1] + kappa_v * (theta_v - v_paths[:, t-1]) * dt + 
            sigma_v * np.sqrt(np.maximum(v_paths[:, t-1], 0)) * dW_v,
            0  # Ensure non-negative variance
        )
        
        # Price process
        dW_s = random_numbers[:, t-1] * np.sqrt(dt)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * v_paths[:, t-1]) * dt + 
            np.sqrt(np.maximum(v_paths[:, t-1], 0)) * dW_s
        )
    
    return paths

def run_scenario_simulations(base_params):
    """Run simulations across multiple economic scenarios"""
    
    scenarios = st.session_state.mc_scenario_types
    scenario_results = {}
    
    for scenario in scenarios:
        # Adjust parameters for each scenario
        scenario_params = adjust_parameters_for_scenario(base_params.copy(), scenario)
        
        # Run simulation for this scenario
        scenario_result = run_single_process_simulation(scenario_params)
        scenario_result['scenario'] = scenario
        scenario_results[scenario] = scenario_result
    
    # Combine results
    all_paths = []
    all_final_values = []
    scenario_labels = []
    
    for scenario, result in scenario_results.items():
        all_paths.append(result['paths'])
        all_final_values.extend(result['final_values'])
        scenario_labels.extend([scenario] * len(result['final_values']))
    
    return {
        'paths': np.vstack(all_paths) if all_paths else np.array([]),
        'final_values': np.array(all_final_values),
        'scenario_results': scenario_results,
        'scenario_labels': scenario_labels,
        'time_steps': scenario_results[scenarios[0]]['time_steps'],
        'process_type': 'Multi-Scenario'
    }

def adjust_parameters_for_scenario(params, scenario):
    """Adjust model parameters for different economic scenarios"""
    
    if scenario == "Bull Market":
        params['mu'] *= 1.5  # Higher expected return
        params['sigma'] *= 0.8  # Lower volatility
    elif scenario == "Bear Market":
        params['mu'] *= -0.5  # Negative expected return
        params['sigma'] *= 1.5  # Higher volatility
    elif scenario == "High Volatility":
        params['sigma'] *= 2.0  # Much higher volatility
    elif scenario == "Crisis":
        params['mu'] *= -1.0  # Negative returns
        params['sigma'] *= 2.5  # Very high volatility
        if 'jump_intensity' in params:
            params['jump_intensity'] *= 3.0  # More frequent jumps
    elif scenario == "Recovery":
        params['mu'] *= 1.2  # Moderate positive returns
        params['sigma'] *= 1.2  # Slightly higher volatility
    # "Base Case" keeps original parameters
    
    return params

def calculate_simulation_statistics(results):
    """Calculate comprehensive statistics from simulation results"""
    
    final_values = results['final_values']
    paths = results['paths']
    
    # Basic statistics
    mean_final = np.mean(final_values)
    std_final = np.std(final_values)
    median_final = np.median(final_values)
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(final_values, percentiles)
    
    # Returns
    returns = final_values - 1  # Assuming normalized starting value of 1
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Confidence intervals
    confidence_levels = st.session_state.mc_confidence_levels
    confidence_intervals = {}
    
    for conf in confidence_levels:
        alpha = (100 - conf) / 100
        lower = np.percentile(final_values, alpha/2 * 100)
        upper = np.percentile(final_values, (1 - alpha/2) * 100)
        confidence_intervals[f'{conf}%'] = (lower, upper)
    
    # Path statistics
    max_values = np.max(paths, axis=1)
    min_values = np.min(paths, axis=1)
    
    # Drawdown analysis
    drawdowns = []
    for path in paths:
        running_max = np.maximum.accumulate(path)
        drawdown = (path / running_max - 1)
        drawdowns.append(np.min(drawdown))
    
    max_drawdown = np.mean(drawdowns)
    
    return {
        'final_statistics': {
            'mean': mean_final,
            'std': std_final,
            'median': median_final,
            'min': np.min(final_values),
            'max': np.max(final_values)
        },
        'return_statistics': {
            'mean_return': mean_return,
            'std_return': std_return,
            'positive_returns': np.mean(returns > 0),
            'negative_returns': np.mean(returns < 0)
        },
        'percentiles': dict(zip(percentiles, percentile_values)),
        'confidence_intervals': confidence_intervals,
        'path_statistics': {
            'mean_max': np.mean(max_values),
            'mean_min': np.mean(min_values),
            'max_drawdown': max_drawdown
        }
    }

def calculate_monte_carlo_risk_metrics(results):
    """Calculate risk metrics from Monte Carlo results"""
    
    final_values = results['final_values']
    returns = final_values - 1
    
    # VaR calculations
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # CVaR calculations
    cvar_95 = np.mean(returns[returns <= var_95])
    cvar_99 = np.mean(returns[returns <= var_99])
    
    # Tail risk measures
    tail_expectation = np.mean(returns[returns <= np.percentile(returns, 10)])
    
    # Probability of loss
    prob_loss = np.mean(returns < 0)
    prob_large_loss = np.mean(returns < -0.1)  # More than 10% loss
    
    # Maximum loss
    max_loss = np.min(returns)
    
    # Volatility estimate
    volatility = np.std(returns)
    
    return {
        'var': {
            'var_95': var_95,
            'var_99': var_99
        },
        'cvar': {
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        },
        'tail_risk': {
            'tail_expectation': tail_expectation,
            'max_loss': max_loss
        },
        'probabilities': {
            'prob_loss': prob_loss,
            'prob_large_loss': prob_large_loss
        },
        'volatility': volatility
    }

def get_simulation_config():
    """Get current simulation configuration"""
    
    return {
        'n_simulations': st.session_state.mc_n_simulations,
        'time_horizon': st.session_state.mc_time_horizon,
        'time_steps': st.session_state.mc_time_steps,
        'process_type': st.session_state.mc_process_type,
        'random_seed': st.session_state.mc_random_seed,
        'antithetic_variates': st.session_state.mc_antithetic,
        'enable_scenarios': st.session_state.get('mc_enable_scenarios', False)
    }

def run_quick_simulation():
    """Run a quick Monte Carlo simulation with default parameters"""
    
    with st.spinner("⚡ Running Quick Simulation..."):
        try:
            # Temporarily set quick parameters
            original_n_sims = st.session_state.get('mc_n_simulations', 1000)
            st.session_state.mc_n_simulations = 500  # Reduced for speed
            
            run_monte_carlo_simulation()
            
            # Restore original parameters
            st.session_state.mc_n_simulations = original_n_sims
            
        except Exception as e:
            st.error(f"❌ Quick simulation failed: {str(e)}")

def render_simulation_results():
    """Render comprehensive Monte Carlo simulation results"""
    
    results = st.session_state.monte_carlo_results
    
    st.subheader(f"📊 {results['process_type']} Simulation Results")
    
    # Summary statistics
    render_simulation_summary(results)
    
    # Visualization
    render_simulation_charts(results)
    
    # Risk analysis
    render_risk_analysis(results)
    
    # Scenario analysis (if applicable)
    if 'scenario_results' in results['simulations']:
        render_scenario_analysis(results)

def render_simulation_summary(results):
    """Render simulation summary statistics"""
    
    stats = results['statistics']
    risk_metrics = results['risk_metrics']
    
    # Main statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        mean_return = stats['return_statistics']['mean_return']
        st.metric(
            "Mean Return",
            f"{mean_return:.2%}",
            delta=f"{mean_return * 252:.2%} annualized"
        )
    
    with col2:
        volatility = stats['return_statistics']['std_return']
        st.metric(
            "Volatility",
            f"{volatility:.2%}",
            delta=f"{volatility * np.sqrt(252):.2%} annualized"
        )
    
    with col3:
        win_rate = stats['return_statistics']['positive_returns']
        st.metric(
            "Win Rate",
            f"{win_rate:.1%}",
            delta="Positive outcomes"
        )
    
    with col4:
        var_95 = risk_metrics['var']['var_95']
        st.metric(
            "VaR (95%)",
            f"{var_95:.2%}",
            delta="Value at Risk"
        )
    
    with col5:
        max_loss = risk_metrics['tail_risk']['max_loss']
        st.metric(
            "Max Loss",
            f"{max_loss:.2%}",
            delta="Worst case scenario"
        )
    
    # Detailed statistics
    with st.expander("📈 Detailed Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Return Distribution:**")
            percentiles = stats['percentiles']
            for p in [5, 25, 50, 75, 95]:
                st.metric(f"{p}th Percentile", f"{percentiles[p] - 1:.2%}")
        
        with col2:
            st.markdown("**Risk Metrics:**")
            st.metric("CVaR (95%)", f"{risk_metrics['cvar']['cvar_95']:.2%}")
            st.metric("CVaR (99%)", f"{risk_metrics['cvar']['cvar_99']:.2%}")
            st.metric("Tail Expectation", f"{risk_metrics['tail_risk']['tail_expectation']:.2%}")
        
        with col3:
            st.markdown("**Confidence Intervals:**")
            confidence_intervals = stats['confidence_intervals']
            for conf, (lower, upper) in confidence_intervals.items():
                st.metric(f"{conf} CI", f"[{lower-1:.2%}, {upper-1:.2%}]")

def render_simulation_charts(results):
    """Render simulation visualization charts"""
    
    paths = results['simulations']['paths']
    final_values = results['simulations']['final_values']
    time_steps = results['simulations']['time_steps']
    
    # Path visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛤️ Simulation Paths")
        
        # Show subset of paths for performance
        n_paths_to_show = min(100, len(paths))
        path_indices = np.random.choice(len(paths), n_paths_to_show, replace=False)
        
        fig = go.Figure()
        
        for i in path_indices:
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=paths[i],
                mode='lines',
                line=dict(width=1, color='rgba(0,100,200,0.3)'),
                showlegend=False,
                hovertemplate='Day %{x}<br>Value: %{y:.3f}<extra></extra>'
            ))
        
        # Add mean path
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Mean Path'
        ))
        
        fig.update_layout(
            title=f"Monte Carlo Paths (showing {n_paths_to_show} of {len(paths)})",
            xaxis_title="Days",
            yaxis_title="Portfolio Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Final Value Distribution")
        
        # Histogram of final values
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=final_values,
            nbinsx=50,
            name='Final Values',
            opacity=0.7,
            hovertemplate='Value: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add percentile lines
        percentiles = [5, 50, 95]
        colors = ['red', 'blue', 'red']
        
        for p, color in zip(percentiles, colors):
            value = np.percentile(final_values, p)
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{p}th %ile: {value:.3f}"
            )
        
        fig.update_layout(
            title="Distribution of Final Portfolio Values",
            xaxis_title="Final Value",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk visualization
    st.subheader("⚠️ Risk Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Drawdown analysis
        max_drawdowns = []
        for path in paths[:1000]:  # Limit for performance
            running_max = np.maximum.accumulate(path)
            drawdown = (path / running_max - 1)
            max_drawdowns.append(np.min(drawdown))
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=max_drawdowns,
            nbinsx=30,
            name='Max Drawdowns',
            opacity=0.7,
            hovertemplate='Drawdown: %{x:.2%}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Distribution of Maximum Drawdowns",
            xaxis_title="Maximum Drawdown",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Return distribution
        returns = final_values - 1
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns * 100,  # Convert to percentage
            nbinsx=50,
            name='Returns',
            opacity=0.7,
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ))
        
        # Add normal distribution overlay
        mu, sigma = np.mean(returns) * 100, np.std(returns) * 100
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
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
            title="Return Distribution vs Normal",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_risk_analysis(results):
    """Render detailed risk analysis"""
    
    st.subheader("📉 Risk Analysis")
    
    risk_metrics = results['risk_metrics']
    final_values = results['simulations']['final_values']
    
    tab1, tab2, tab3 = st.tabs(["📊 VaR Analysis", "🎯 Tail Risk", "📈 Stress Scenarios"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Value at Risk:**")
            
            var_data = {
                'Confidence Level': ['95%', '99%'],
                'VaR': [risk_metrics['var']['var_95'], risk_metrics['var']['var_99']],
                'CVaR': [risk_metrics['cvar']['cvar_95'], risk_metrics['cvar']['cvar_99']]
            }
            
            var_df = pd.DataFrame(var_data)
            styled_var = var_df.style.format({
                'VaR': '{:.3%}',
                'CVaR': '{:.3%}'
            })
            
            st.dataframe(styled_var, use_container_width=True)
        
        with col2:
            # VaR visualization
            fig = go.Figure()
            
            returns = final_values - 1
            
            # Plot return distribution
            fig.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Return Distribution',
                opacity=0.6
            ))
            
            # Add VaR lines
            var_95 = risk_metrics['var']['var_95'] * 100
            var_99 = risk_metrics['var']['var_99'] * 100
            
            fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                         annotation_text=f"VaR 95%: {var_95:.2f}%")
            fig.add_vline(x=var_99, line_dash="dash", line_color="red",
                         annotation_text=f"VaR 99%: {var_99:.2f}%")
            
            fig.update_layout(
                title="VaR Visualization",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("**Tail Risk Analysis:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tail Expectation", f"{risk_metrics['tail_risk']['tail_expectation']:.3%}")
            st.metric("Maximum Loss", f"{risk_metrics['tail_risk']['max_loss']:.3%}")
            st.metric("Probability of Loss", f"{risk_metrics['probabilities']['prob_loss']:.1%}")
            st.metric("Prob. Large Loss (>10%)", f"{risk_metrics['probabilities']['prob_large_loss']:.1%}")
        
        with col2:
            # Tail analysis chart
            returns = final_values - 1
            tail_threshold = np.percentile(returns, 10)
            tail_returns = returns[returns <= tail_threshold]
            
            if len(tail_returns) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=tail_returns * 100,
                    nbinsx=20,
                    name='Tail Returns',
                    opacity=0.7,
                    marker_color='red'
                ))
                
                fig.update_layout(
                    title="Tail Risk Distribution (Bottom 10%)",
                    xaxis_title="Return (%)",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("**Stress Test Scenarios:**")
        
        # Create stress scenarios
        stress_scenarios = {
            "Market Crash (-20%)": -0.20,
            "Severe Bear Market (-30%)": -0.30,
            "Black Swan Event (-40%)": -0.40,
            "Great Depression (-50%)": -0.50
        }
        
        portfolio_value = 1000000  # Example portfolio value
        
        stress_results = []
        for scenario, shock in stress_scenarios.items():
            shocked_value = portfolio_value * (1 + shock)
            loss_amount = portfolio_value - shocked_value
            
            stress_results.append({
                'Scenario': scenario,
                'Shock': f"{shock:.0%}",
                'Portfolio Value': f"${shocked_value:,.0f}",
                'Loss Amount': f"${loss_amount:,.0f}"
            })
        
        stress_df = pd.DataFrame(stress_results)
        st.dataframe(stress_df, use_container_width=True)

def render_scenario_analysis(results):
    """Render scenario analysis results"""
    
    st.subheader("🎭 Scenario Analysis")
    
    scenario_results = results['simulations']['scenario_results']
    
    # Scenario comparison
    scenario_summary = {}
    
    for scenario, data in scenario_results.items():
        final_values = data['final_values']
        returns = final_values - 1
        
        scenario_summary[scenario] = {
            'Mean Return': np.mean(returns),
            'Volatility': np.std(returns),
            'VaR 95%': np.percentile(returns, 5),
            'Max Loss': np.min(returns),
            'Win Rate': np.mean(returns > 0)
        }
    
    summary_df = pd.DataFrame(scenario_summary).T
    
    styled_summary = summary_df.style.format({
        'Mean Return': '{:.2%}',
        'Volatility': '{:.2%}',
        'VaR 95%': '{:.2%}',
        'Max Loss': '{:.2%}',
        'Win Rate': '{:.1%}'
    })
    
    st.dataframe(styled_summary, use_container_width=True)
    
    # Scenario comparison chart
    fig = go.Figure()
    
    scenarios = list(scenario_results.keys())
    mean_returns = [scenario_summary[s]['Mean Return'] * 100 for s in scenarios]
    volatilities = [scenario_summary[s]['Volatility'] * 100 for s in scenarios]
    
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=mean_returns,
        mode='markers+text',
        text=scenarios,
        textposition="top center",
        marker=dict(size=15),
        hovertemplate='<b>%{text}</b><br>Mean Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Risk-Return by Scenario",
        xaxis_title="Volatility (%)",
        yaxis_title="Mean Return (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def reset_simulation_results():
    """Reset Monte Carlo simulation results"""
    
    if 'monte_carlo_results' in st.session_state:
        del st.session_state.monte_carlo_results
    
    st.success("🔄 Simulation results cleared!")
    safe_rerun()

def export_simulation_results():
    """Export Monte Carlo simulation results"""
    
    results = st.session_state.monte_carlo_results
    
    # Prepare export data
    final_values = results['simulations']['final_values']
    returns = final_values - 1
    
    export_df = pd.DataFrame({
        'Simulation': range(len(final_values)),
        'Final_Value': final_values,
        'Return': returns,
        'Process_Type': results['process_type']
    })
    
    # Add scenario labels if available
    if 'scenario_labels' in results['simulations']:
        export_df['Scenario'] = results['simulations']['scenario_labels']
    
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Simulation Results CSV",
        data=csv,
        file_name=f"monte_carlo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ Simulation results prepared for download!")

# Helper function for log calculations
def np_log(x):
    """Safe logarithm function"""
    return np.log(np.maximum(x, 1e-10))

# Main execution
if __name__ == "__main__":
    render_monte_carlo()