#!/usr/bin/env python3
"""
Stochastic Portfolio Engine - Main Streamlit Application

A comprehensive web dashboard for portfolio management with:
- HMM regime detection
- Advanced backtesting
- Real-time risk monitoring
- Performance analytics
- Monte Carlo simulations

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Optional, Any
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import portfolio engine components with error handling
try:
    from src.utils.config import get_config
    from src.utils.logging_config import setup_logging, get_logger
    from src.data.ingestion.data_sources import create_data_pipeline
    from src.models.hmm.hmm_engine import AdvancedBaumWelchHMM
    from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
    from src.backtesting.framework.advanced_backtesting import create_advanced_backtesting_framework, BacktestMode
    from src.risk.monitoring.risk_monitor import RealTimeRiskMonitor, RiskLimits
    from src.simulation.monte_carlo_engine import MonteCarloEngine, SimulationConfig
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the src directory structure is correct.")
    st.info("Run: pip install -r requirements.txt")
    st.stop()

warnings.filterwarnings("ignore", category=FutureWarning)

# Streamlit version compatibility helpers
def safe_rerun():
    """Version-safe rerun function"""
    try:
        st.rerun()  # New syntax (Streamlit >= 1.27.0)
    except AttributeError:
        st.experimental_rerun()  # Old syntax fallback

def safe_cache_data(func=None, **kwargs):
    """Version-safe cache_data decorator"""
    try:
        return st.cache_data(**kwargs)(func) if func else st.cache_data(**kwargs)
    except AttributeError:
        return st.experimental_memo(**kwargs)(func) if func else st.experimental_memo(**kwargs)

def safe_cache_resource(func=None, **kwargs):
    """Version-safe cache_resource decorator"""
    try:
        return st.cache_resource(**kwargs)(func) if func else st.cache_resource(**kwargs)
    except AttributeError:
        return st.experimental_singleton(**kwargs)(func) if func else st.experimental_singleton(**kwargs)

# Configure Streamlit page
st.set_page_config(
    page_title="Stochastic Portfolio Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/stochastic-portfolio-engine',
        'Report a bug': 'https://github.com/your-username/stochastic-portfolio-engine/issues',
        'About': """
        # Stochastic Portfolio Engine v1.0.0
        
        Advanced portfolio management system featuring:
        - Hidden Markov Model regime detection
        - Realistic backtesting with trading costs
        - Real-time risk monitoring
        - Monte Carlo simulations
        - Performance attribution analysis
        """
    }
)

# Custom CSS for minimalist professional design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cambria:wght@400;600;700&display=swap');
    
    /* Global font family override */
    html, body, [class*="css"] {
        font-family: 'Cambria', serif !important;
    }
    
    /* Main header styling */
    .main-header {
        font-family: 'Cambria', serif !important;
        font-size: 2.2rem;
        font-weight: 600;
        color: #2c2c2c;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 1rem;
    }
    
    /* Remove Streamlit branding and decorative elements */
    .stDeployButton {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        background: none !important;
        height: 0 !important;
    }
    
    .stDecoration {
        display: none !important;
    }
    
    /* Clean metric cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border: 1px solid #e8e8e8;
        margin: 0.5rem 0;
        font-family: 'Cambria', serif !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #fafafa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #ffffff;
        color: #2c2c2c;
        border: 1px solid #d0d0d0;
        font-family: 'Cambria', serif !important;
        font-weight: 500;
        border-radius: 2px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #f5f5f5;
        border-color: #a0a0a0;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background-color: #2c2c2c;
        color: #ffffff;
        border: 1px solid #2c2c2c;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #404040;
        border-color: #404040;
    }
    
    /* Clean selectbox and input styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 2px;
        font-family: 'Cambria', serif !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 2px;
        font-family: 'Cambria', serif !important;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        color: #2c2c2c;
        font-family: 'Cambria', serif !important;
        font-weight: 500;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 1rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #2c2c2c;
        border-bottom: 2px solid #2c2c2c;
    }
    
    /* Clean expander */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        color: #2c2c2c;
        font-family: 'Cambria', serif !important;
        font-weight: 600;
        border: 1px solid #e0e0e0;
        border-radius: 2px;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e8e8e8;
        padding: 1rem;
        border-radius: 2px;
    }
    
    [data-testid="metric-container"] > div {
        color: #2c2c2c;
        font-family: 'Cambria', serif !important;
    }
    
    /* Clean dataframe styling */
    .stDataFrame {
        font-family: 'Cambria', serif !important;
    }
    
    /* Remove colorful elements */
    .stAlert {
        background-color: #f8f9fa;
        color: #2c2c2c;
        border: 1px solid #e0e0e0;
        font-family: 'Cambria', serif !important;
    }
    
    .stSuccess {
        background-color: #f8f9fa;
        color: #2c2c2c;
        border-left: 3px solid #6c757d;
    }
    
    .stWarning {
        background-color: #f8f9fa;
        color: #2c2c2c;
        border-left: 3px solid #6c757d;
    }
    
    .stError {
        background-color: #f8f9fa;
        color: #2c2c2c;
        border-left: 3px solid #6c757d;
    }
    
    .stInfo {
        background-color: #f8f9fa;
        color: #2c2c2c;
        border-left: 3px solid #6c757d;
    }
    
    /* Chart container styling */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    
    /* Clean markdown styling */
    .stMarkdown {
        font-family: 'Cambria', serif !important;
        color: #2c2c2c;
    }
    
    /* Hide hamburger menu */
    .css-14xtw13.e8zbici0 {
        display: none;
    }
    
    /* Clean slider styling */
    .stSlider > div > div {
        background-color: #ffffff;
    }
    
    .stSlider [role="slider"] {
        background-color: #2c2c2c;
    }
</style>
""", unsafe_allow_html=True)


class PortfolioEngineApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = self._load_config()
        self._initialize_session_state()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open('config/config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error("Configuration file not found. Please check config/config.yaml")
            return {}
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        
        # Portfolio data
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
        
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # Analysis results
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        
        if 'regime_model' not in st.session_state:
            st.session_state.regime_model = None
        
        if 'risk_metrics' not in st.session_state:
            st.session_state.risk_metrics = {}
        
        # UI state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Portfolio Overview"
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def _setup_logging(self):
        """Setup logging configuration"""
        setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file='streamlit_app.log'
        )
        self.logger = get_logger(__name__)
    
    def run(self):
        """Main application entry point"""
        
        # App header
        st.markdown('<h1 class="main-header">Stochastic Portfolio Engine</h1>', unsafe_allow_html=True)
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.current_page
        
        if page == "Portfolio Overview":
            self._render_portfolio_overview()
        elif page == "Regime Detection":
            self._render_regime_detection()
        elif page == "Backtesting":
            self._render_backtesting()
        elif page == "Risk Management":
            self._render_risk_management()
        elif page == "Performance Analytics":
            self._render_performance_analytics()
        elif page == "Monte Carlo":
            self._render_monte_carlo()
        elif page == "Settings":
            self._render_settings()
    
    def _render_sidebar(self):
        """Render sidebar navigation and controls"""
        
        with st.sidebar:
            # Logo/Title
            st.markdown("## Navigation")
            
            # Page selection
            pages = [
                "Portfolio Overview",
                "Regime Detection", 
                "Backtesting",
                "Risk Management",
                "Performance Analytics",
                "Monte Carlo",
                "Settings"
            ]
            
            selected_page = st.selectbox(
                "Select Page",
                pages,
                index=pages.index(st.session_state.current_page)
            )
            
            if selected_page != st.session_state.current_page:
                st.session_state.current_page = selected_page
                safe_rerun()
            
            st.markdown("---")
            
            # Global settings
            st.markdown("## Global Settings")
            
            # Asset selection
            available_symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META',
                'JPM', 'BAC', 'GS', 'V', 'MA', 'JNJ', 'PFE', 'UNH',
                'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'VIX'
            ]
            
            selected_symbols = st.multiselect(
                "Select Assets",
                available_symbols,
                default=st.session_state.selected_symbols,
                help="Choose assets for portfolio analysis"
            )
            
            if selected_symbols != st.session_state.selected_symbols:
                st.session_state.selected_symbols = selected_symbols
                st.session_state.data_loaded = False
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365*2),
                    help="Analysis start date"
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    help="Analysis end date"
                )
            
            # Data refresh
            if st.button("Refresh Data", use_container_width=True):
                self._load_portfolio_data(start_date, end_date)
            
            st.markdown("---")
            
            # System status
            st.markdown("## System Status")
            
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                if st.session_state.data_loaded:
                    st.success("Data Loaded")
                else:
                    st.warning("Data Not Loaded")
            
            with status_col2:
                if st.session_state.last_update:
                    st.info(f"Updated: {st.session_state.last_update.strftime('%H:%M')}")
                else:
                    st.info("Never Updated")
            
            # Quick stats
            if st.session_state.portfolio_data is not None:
                st.markdown("### Quick Stats")
                
                portfolio_data = st.session_state.portfolio_data
                if not portfolio_data.empty:
                    latest_value = portfolio_data.iloc[-1].mean()
                    daily_change = (portfolio_data.iloc[-1] / portfolio_data.iloc[-2] - 1).mean()
                    
                    st.metric("Portfolio Value", f"${latest_value:.2f}", f"{daily_change:.2%}")
                    st.metric("Assets", len(st.session_state.selected_symbols))
    
    def _load_portfolio_data(self, start_date: datetime, end_date: datetime):
        """Load portfolio data from data sources"""
        
        if not st.session_state.selected_symbols:
            st.warning("Please select at least one asset.")
            return
        
        with st.spinner("Loading portfolio data..."):
            try:
                # Create data pipeline
                pipeline = create_data_pipeline()
                
                # Fetch data (simplified for demo)
                import yfinance as yf
                
                # Download data
                data = yf.download(
                    st.session_state.selected_symbols,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True
                )
                
                if len(st.session_state.selected_symbols) == 1:
                    # Single symbol - create MultiIndex
                    data.columns = pd.MultiIndex.from_product([data.columns, st.session_state.selected_symbols])
                
                st.session_state.portfolio_data = data
                st.session_state.data_loaded = True
                st.session_state.last_update = datetime.now()
                
                st.success(f"Data loaded for {len(st.session_state.selected_symbols)} assets")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                self.logger.error(f"Data loading error: {e}")
    
    def _render_portfolio_overview(self):
        """Render portfolio overview dashboard"""
        
        st.header("Portfolio Overview")
        
        if not st.session_state.data_loaded:
            st.info("Please load data using the sidebar controls to view portfolio overview.")
            
            # Show demo content
            st.markdown("## Portfolio Engine Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="feature-highlight">
                    <h4>HMM Regime Detection</h4>
                    <p>Advanced market regime identification using Hidden Markov Models</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="feature-highlight">
                    <h4>Realistic Backtesting</h4>
                    <p>Event-driven backtesting with realistic trading costs and slippage</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="feature-highlight">
                    <h4>Monte Carlo Analysis</h4>
                    <p>Statistical robustness testing with thousands of simulations</p>
                </div>
                """, unsafe_allow_html=True)
            
            return
        
        portfolio_data = st.session_state.portfolio_data
        
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Extract close prices
        if 'Close' in portfolio_data.columns.get_level_values(0):
            close_prices = portfolio_data.xs('Close', level=0, axis=1)
        else:
            close_prices = portfolio_data
        
        returns = close_prices.pct_change().dropna()
        
        # Equal weight portfolio for demo
        portfolio_returns = returns.mean(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        total_return = cumulative_returns.iloc[-1] - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
        max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
        
        with col1:
            st.metric("Total Return", f"{total_return:.2%}")
        
        with col2:
            st.metric("Volatility", f"{volatility:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cumulative Returns")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Asset Allocation")
            
            # Equal weight for demo
            weights = [1/len(st.session_state.selected_symbols)] * len(st.session_state.selected_symbols)
            
            fig = go.Figure(data=[go.Pie(
                labels=st.session_state.selected_symbols,
                values=weights,
                hole=0.3
            )])
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Price charts
        st.subheader("Asset Price Charts")
        
        fig = go.Figure()
        
        # Normalize prices to 100
        normalized_prices = close_prices / close_prices.iloc[0] * 100
        
        for symbol in st.session_state.selected_symbols:
            if symbol in normalized_prices.columns:
                fig.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[symbol],
                    mode='lines',
                    name=symbol
                ))
        
        fig.update_layout(
            title="Normalized Asset Prices (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_regime_detection(self):
        """Render regime detection dashboard"""
        
        st.header("Regime Detection Analysis")
        
        if not st.session_state.data_loaded:
            st.info("Please load data using the sidebar controls to perform regime analysis.")
            return
        
        # Regime detection controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_regimes = st.slider("Number of Regimes", 2, 5, 3)
        
        with col2:
            covariance_type = st.selectbox("Covariance Type", ["full", "diag", "spherical", "tied"])
        
        with col3:
            if st.button("Detect Regimes"):
                self._run_regime_detection(n_regimes, covariance_type)
        
        # Display results if available
        if st.session_state.regime_model is not None:
            self._display_regime_results()
        else:
            st.info("Click 'Detect Regimes' to analyze market regimes using HMM.")
    
    def _run_regime_detection(self, n_regimes: int, covariance_type: str):
        """Run HMM regime detection"""
        
        with st.spinner("Training HMM model for regime detection..."):
            try:
                portfolio_data = st.session_state.portfolio_data
                
                # Extract close prices and calculate features
                if 'Close' in portfolio_data.columns.get_level_values(0):
                    close_prices = portfolio_data.xs('Close', level=0, axis=1)
                else:
                    close_prices = portfolio_data
                
                returns = close_prices.pct_change().dropna()
                
                # Create features
                features = pd.DataFrame({
                    'market_return': returns.mean(axis=1),
                    'market_volatility': returns.rolling(20).std().mean(axis=1),
                    'momentum': returns.rolling(10).mean().mean(axis=1)
                }).dropna()
                
                # Train HMM model
                hmm_model = AdvancedBaumWelchHMM(
                    n_components=n_regimes,
                    covariance_type=covariance_type,
                    random_state=42
                )
                
                hmm_model.fit(features)
                
                # Get predictions
                regime_sequence = hmm_model.predict_regimes(features)
                regime_probabilities = hmm_model.predict_regime_probabilities(features)
                
                # Store results
                st.session_state.regime_model = {
                    'model': hmm_model,
                    'features': features,
                    'sequence': regime_sequence,
                    'probabilities': regime_probabilities,
                    'n_regimes': n_regimes
                }
                
                st.success("Regime detection completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error in regime detection: {str(e)}")
                self.logger.error(f"Regime detection error: {e}")
    
    def _display_regime_results(self):
        """Display regime detection results"""
        
        regime_data = st.session_state.regime_model
        
        # Regime summary
        st.subheader("Regime Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Regimes", regime_data['n_regimes'])
        
        with col2:
            convergence = regime_data['model'].converged_
            st.metric("Model Converged", "Yes" if convergence else "No")
        
        with col3:
            log_likelihood = regime_data['model'].log_likelihood_history_[-1]
            st.metric("Log Likelihood", f"{log_likelihood:.2f}")
        
        # Regime sequence plot
        st.subheader("Regime Sequence Over Time")
        
        fig = go.Figure()
        
        features = regime_data['features']
        sequence = regime_data['sequence']
        
        # Color map for regimes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        fig.add_trace(go.Scatter(
            x=features.index,
            y=sequence,
            mode='markers+lines',
            name='Regime',
            marker=dict(
                color=[colors[r] for r in sequence],
                size=6
            ),
            line=dict(width=1)
        ))
        
        fig.update_layout(
            title="Market Regime Sequence",
            xaxis_title="Date",
            yaxis_title="Regime",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime probabilities
        st.subheader("Regime Probabilities")
        
        probabilities = regime_data['probabilities']
        
        fig = go.Figure()
        
        for i in range(regime_data['n_regimes']):
            fig.add_trace(go.Scatter(
                x=features.index,
                y=probabilities[:, i],
                mode='lines',
                name=f'Regime {i}',
                fill='tonexty' if i > 0 else None,
                line=dict(color=colors[i])
            ))
        
        fig.update_layout(
            title="Regime Probabilities Over Time",
            xaxis_title="Date",
            yaxis_title="Probability",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_backtesting(self):
        """Render backtesting interface"""
        
        st.header("Advanced Backtesting")
        
        if not st.session_state.data_loaded:
            st.info("Please load data using the sidebar controls to run backtests.")
            return
        
        # Backtesting parameters
        st.subheader("Backtest Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_mode = st.selectbox(
                "Backtest Mode",
                ["Vectorized", "Event-driven", "Walk-forward", "Monte Carlo"]
            )
        
        with col2:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["mean_variance", "black_litterman", "risk_parity"]
            )
        
        with col3:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=10000,
                max_value=10000000,
                value=1000000,
                step=50000
            )
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                commission_rate = st.slider("Commission Rate (%)", 0.0, 1.0, 0.1, 0.01)
                bid_ask_spread = st.slider("Bid-Ask Spread (%)", 0.0, 0.5, 0.05, 0.01)
            
            with col2:
                rebalance_frequency = st.selectbox("Rebalance Frequency", ["Daily", "Weekly", "Monthly"])
                max_weight = st.slider("Max Asset Weight (%)", 5, 50, 25, 5)
        
        # Run backtest
        if st.button("🚀 Run Backtest", use_container_width=True):
            self._run_backtest(
                backtest_mode, optimization_method, initial_capital,
                commission_rate/100, bid_ask_spread/100, rebalance_frequency, max_weight/100
            )
        
        # Display results
        if st.session_state.backtest_results is not None:
            self._display_backtest_results()
    
    def _run_backtest(self, mode: str, optimization_method: str, initial_capital: float,
                     commission_rate: float, bid_ask_spread: float, rebalance_frequency: str, max_weight: float):
        """Run backtesting analysis"""
        
        with st.spinner(f"Running {mode} backtest..."):
            try:
                # This is a simplified demo implementation
                portfolio_data = st.session_state.portfolio_data
                
                # Extract close prices
                if 'Close' in portfolio_data.columns.get_level_values(0):
                    close_prices = portfolio_data.xs('Close', level=0, axis=1)
                else:
                    close_prices = portfolio_data
                
                returns = close_prices.pct_change().dropna()
                
                # Simple equal-weight backtest for demo
                portfolio_returns = returns.mean(axis=1)
                cumulative_returns = (1 + portfolio_returns).cumprod() * initial_capital
                
                # Calculate metrics
                total_return = cumulative_returns.iloc[-1] / initial_capital - 1
                volatility = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
                max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
                
                # Store results
                st.session_state.backtest_results = {
                    'mode': mode,
                    'returns': portfolio_returns,
                    'cumulative_returns': cumulative_returns,
                    'metrics': {
                        'total_return': total_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown
                    },
                    'config': {
                        'optimization_method': optimization_method,
                        'initial_capital': initial_capital,
                        'commission_rate': commission_rate,
                        'max_weight': max_weight
                    }
                }
                
                st.success(f"{mode} backtest completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Backtest error: {str(e)}")
                self.logger.error(f"Backtest error: {e}")
    
    def _display_backtest_results(self):
        """Display backtesting results"""
        
        results = st.session_state.backtest_results
        
        st.subheader("Backtest Results")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{results['metrics']['total_return']:.2%}")
        
        with col2:
            st.metric("Volatility", f"{results['metrics']['volatility']:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{results['metrics']['max_drawdown']:.2%}")
        
        # Performance chart
        st.subheader("Portfolio Performance")
        
        fig = go.Figure()
        
        cumulative_returns = results['cumulative_returns']
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"{results['mode']} Backtest Results",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        if st.button("Download Results"):
            # Create downloadable data
            results_df = pd.DataFrame({
                'Date': cumulative_returns.index,
                'Portfolio_Value': cumulative_returns.values,
                'Daily_Return': results['returns'].values
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"backtest_results_{results['mode'].lower()}.csv",
                mime="text/csv"
            )
    
    def _render_risk_management(self):
        """Render risk management dashboard"""
        
        st.header("Risk Management Dashboard")
        
        if not st.session_state.data_loaded:
            st.info("Please load data using the sidebar controls to view risk metrics.")
            return
        
        st.info("Risk Management Dashboard - Coming Soon!")
        st.markdown("This will include VaR, CVaR, exposure analysis, and risk limit monitoring.")
    
    def _render_performance_analytics(self):
        """Render performance analytics page"""
        
        st.header("Performance Analytics")
        
        if not st.session_state.data_loaded:
            st.info("Please load data using the sidebar controls to view performance analytics.")
            return
        
        st.info("Performance Analytics - Coming Soon!")
        st.markdown("This will include factor attribution, benchmark comparison, and statistical analysis.")
    
    def _render_monte_carlo(self):
        """Render Monte Carlo simulations page"""
        
        st.header("Monte Carlo Simulations")
        
        if not st.session_state.data_loaded:
            st.info("Please load data using the sidebar controls to run Monte Carlo analysis.")
            return
        
        st.info("Monte Carlo Simulations - Coming Soon!")
        st.markdown("This will include portfolio simulation, scenario analysis, and confidence intervals.")
    
    def _render_settings(self):
        """Render settings and configuration page"""
        
        st.header("Settings & Configuration")
        
        # App configuration
        st.subheader("Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("App Name", value=self.config.get('app', {}).get('name', ''))
            st.text_input("Version", value=self.config.get('app', {}).get('version', ''))
        
        with col2:
            st.selectbox("Theme", ["Light", "Dark"])
            st.selectbox("Layout", ["Wide", "Centered"])
        
        # API Configuration
        st.subheader("API Configuration")
        
        st.text_input("Alpha Vantage API Key", type="password", help="Enter your Alpha Vantage API key")
        st.text_input("Polygon.io API Key", type="password", help="Enter your Polygon.io API key")
        
        # Save settings
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")


def main():
    """Main application entry point"""
    
    try:
        app = PortfolioEngineApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()