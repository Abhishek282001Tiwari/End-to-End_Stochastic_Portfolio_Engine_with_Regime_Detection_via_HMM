#!/usr/bin/env python3
"""
Regime Detection Dashboard Page

Advanced regime detection interface featuring:
- Interactive regime detection controls
- Real-time regime probability gauges
- Historical regime sequence timeline
- Transition matrix heatmaps
- Regime characteristics analysis
- Parameter adjustment sliders
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from src.models.hmm.hmm_engine import AdvancedBaumWelchHMM
from src.utils.performance_analytics import PerformanceAnalytics
from streamlit_portfolio_engine.utils import safe_rerun

def render_regime_detection():
    """Render the regime detection dashboard"""
    
    st.title("Regime Detection Dashboard")
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        render_regime_welcome_screen()
        return
    
    # Get portfolio data
    portfolio_data = st.session_state.portfolio_data
    
    if portfolio_data is None or portfolio_data.empty:
        st.warning("No portfolio data available. Please load data using the sidebar.")
        return
    
    # Render main dashboard
    render_regime_controls()
    
    # Check if regime analysis has been run
    if st.session_state.get('regime_model') is not None:
        render_current_regime_status()
        render_regime_timeline()
        render_transition_matrix()
        render_regime_characteristics()
        render_regime_predictions()
    else:
        st.info("👆 Configure parameters and click 'Detect Regimes' to begin HMM analysis.")

def render_regime_welcome_screen():
    """Render welcome screen for regime detection"""
    
    st.markdown("""
    ## Hidden Markov Model Regime Detection
    
    Advanced market regime identification using statistical Hidden Markov Models to detect:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Bull Markets
        - High returns
        - Low volatility
        - Strong momentum
        - Risk-on sentiment
        """)
    
    with col2:
        st.markdown("""
        ### Bear Markets  
        - Negative returns
        - High volatility
        - Flight to quality
        - Risk-off sentiment
        """)
    
    with col3:
        st.markdown("""
        ### Sideways Markets
        - Range-bound returns
        - Moderate volatility
        - Mean reversion
        - Mixed sentiment
        """)
    
    st.markdown("""
    ---
    
    ### Key Features
    
    - **Real-time Regime Detection**: Live probability gauges for current market state
    - **Historical Analysis**: Color-coded timeline showing regime evolution
    - **Transition Dynamics**: Heatmap visualization of regime transition probabilities
    - **Statistical Validation**: Model convergence metrics and likelihood analysis
    - **Parameter Tuning**: Interactive controls for model optimization
    
    **Load portfolio data in the sidebar to begin regime analysis!**
    """)

def render_regime_controls():
    """Render regime detection parameter controls"""
    
    st.subheader("HMM Configuration")
    
    with st.expander("Model Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_regimes = st.slider(
                "Number of Regimes",
                min_value=2,
                max_value=5,
                value=st.session_state.get('hmm_n_regimes', 3),
                help="Number of hidden market regimes to detect"
            )
            st.session_state.hmm_n_regimes = n_regimes
        
        with col2:
            covariance_type = st.selectbox(
                "Covariance Type",
                ["full", "diag", "spherical", "tied"],
                index=["full", "diag", "spherical", "tied"].index(
                    st.session_state.get('hmm_covariance_type', 'full')
                ),
                help="Covariance matrix structure for regime modeling"
            )
            st.session_state.hmm_covariance_type = covariance_type
        
        with col3:
            n_iterations = st.slider(
                "Max Iterations",
                min_value=50,
                max_value=500,
                value=st.session_state.get('hmm_n_iterations', 100),
                step=50,
                help="Maximum training iterations for convergence"
            )
            st.session_state.hmm_n_iterations = n_iterations
        
        with col4:
            lookback_window = st.slider(
                "Lookback Window (days)",
                min_value=60,
                max_value=1000,
                value=st.session_state.get('hmm_lookback_window', 252),
                step=30,
                help="Historical data window for training"
            )
            st.session_state.hmm_lookback_window = lookback_window
    
    # Feature selection
    with st.expander("Feature Engineering"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Market Features:**")
            use_returns = st.checkbox("Returns", value=True, help="Daily portfolio returns")
            use_volatility = st.checkbox("Volatility", value=True, help="Rolling volatility")
            use_momentum = st.checkbox("Momentum", value=True, help="Short-term momentum")
            use_volume = st.checkbox("Volume", value=False, help="Trading volume (if available)")
        
        with col2:
            st.markdown("**Technical Features:**")
            vol_window = st.slider("Volatility Window", 5, 60, 20, help="Rolling window for volatility calculation")
            mom_window = st.slider("Momentum Window", 3, 30, 10, help="Rolling window for momentum calculation")
            standardize = st.checkbox("Standardize Features", value=True, help="Z-score normalize features")
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("Detect Regimes", use_container_width=True, type="primary"):
            run_regime_detection(
                n_regimes, covariance_type, n_iterations, lookback_window,
                use_returns, use_volatility, use_momentum, use_volume,
                vol_window, mom_window, standardize
            )
    
    with col2:
        if st.button("Reset Model", use_container_width=True):
            reset_regime_model()
    
    with col3:
        if st.session_state.get('regime_model') is not None:
            if st.button("Export Results", use_container_width=True):
                export_regime_results()

def run_regime_detection(n_regimes, covariance_type, n_iterations, lookback_window,
                        use_returns, use_volatility, use_momentum, use_volume,
                        vol_window, mom_window, standardize):
    """Run HMM regime detection with specified parameters"""
    
    with st.spinner("Training Hidden Markov Model..."):
        try:
            portfolio_data = st.session_state.portfolio_data
            
            # Extract price data
            if 'Close' in portfolio_data.columns.get_level_values(0):
                close_prices = portfolio_data.xs('Close', level=0, axis=1)
            else:
                close_prices = portfolio_data
            
            # Prepare features
            features_df = prepare_hmm_features(
                close_prices, use_returns, use_volatility, use_momentum, use_volume,
                vol_window, mom_window, standardize
            )
            
            # Limit data to lookback window
            if len(features_df) > lookback_window:
                features_df = features_df.tail(lookback_window)
            
            # Train HMM model
            hmm_model = AdvancedBaumWelchHMM(
                n_components=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iterations,
                random_state=42,
                tol=1e-4
            )
            
            # Fit model
            hmm_model.fit(features_df.values)
            
            # Get predictions
            regime_sequence = hmm_model.predict(features_df.values)
            regime_probabilities = hmm_model.predict_proba(features_df.values)
            
            # Calculate regime statistics
            regime_stats = calculate_regime_statistics(
                features_df, regime_sequence, regime_probabilities, close_prices
            )
            
            # Store results
            st.session_state.regime_model = {
                'model': hmm_model,
                'features': features_df,
                'sequence': regime_sequence,
                'probabilities': regime_probabilities,
                'statistics': regime_stats,
                'config': {
                    'n_regimes': n_regimes,
                    'covariance_type': covariance_type,
                    'n_iterations': n_iterations,
                    'lookback_window': lookback_window,
                    'features_used': {
                        'returns': use_returns,
                        'volatility': use_volatility,
                        'momentum': use_momentum,
                        'volume': use_volume
                    }
                }
            }
            
            # Success message with model info
            convergence_info = "Converged" if hmm_model.monitor_.converged else "Did not converge"
            final_ll = hmm_model.monitor_.history[-1] if hmm_model.monitor_.history else "N/A"
            
            st.success(f"""
            🎉 **Regime Detection Complete!**
            - {convergence_info}
            - Final Log-Likelihood: {final_ll:.4f}
            - Training Iterations: {len(hmm_model.monitor_.history)}
            - Data Points: {len(features_df)}
            """)
            
        except Exception as e:
            st.error(f"❌ Error in regime detection: {str(e)}")
            st.exception(e)

def prepare_hmm_features(close_prices, use_returns, use_volatility, use_momentum, use_volume,
                        vol_window, mom_window, standardize):
    """Prepare feature matrix for HMM training"""
    
    features = {}
    
    # Calculate returns
    returns = close_prices.pct_change().dropna()
    portfolio_returns = returns.mean(axis=1)  # Equal weight for demo
    
    if use_returns:
        features['returns'] = portfolio_returns
    
    if use_volatility:
        features['volatility'] = portfolio_returns.rolling(vol_window).std()
    
    if use_momentum:
        features['momentum'] = portfolio_returns.rolling(mom_window).mean()
    
    if use_volume:
        # Volume feature would go here if available
        # For now, create synthetic volume proxy
        features['volume_proxy'] = np.abs(portfolio_returns).rolling(10).mean()
    
    # Create DataFrame
    features_df = pd.DataFrame(features).dropna()
    
    # Standardize features if requested
    if standardize:
        features_df = (features_df - features_df.mean()) / features_df.std()
    
    return features_df

def calculate_regime_statistics(features_df, regime_sequence, regime_probabilities, close_prices):
    """Calculate regime-specific statistics"""
    
    returns = close_prices.pct_change().dropna()
    portfolio_returns = returns.mean(axis=1)
    
    # Align with features
    portfolio_returns = portfolio_returns.loc[features_df.index]
    
    regime_stats = {}
    n_regimes = len(np.unique(regime_sequence))
    
    for regime in range(n_regimes):
        regime_mask = regime_sequence == regime
        regime_returns = portfolio_returns[regime_mask]
        
        if len(regime_returns) > 0:
            regime_stats[regime] = {
                'frequency': regime_mask.sum() / len(regime_sequence),
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'avg_duration': calculate_average_duration(regime_sequence, regime),
                'max_duration': calculate_max_duration(regime_sequence, regime),
                'return_distribution': regime_returns.describe()
            }
        else:
            regime_stats[regime] = {
                'frequency': 0,
                'avg_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'avg_duration': 0,
                'max_duration': 0,
                'return_distribution': pd.Series()
            }
    
    return regime_stats

def calculate_average_duration(sequence, regime):
    """Calculate average duration of regime periods"""
    durations = []
    current_duration = 0
    
    for state in sequence:
        if state == regime:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
    
    # Don't forget the last period
    if current_duration > 0:
        durations.append(current_duration)
    
    return np.mean(durations) if durations else 0

def calculate_max_duration(sequence, regime):
    """Calculate maximum duration of regime periods"""
    durations = []
    current_duration = 0
    
    for state in sequence:
        if state == regime:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
    
    if current_duration > 0:
        durations.append(current_duration)
    
    return max(durations) if durations else 0

def render_current_regime_status():
    """Render current regime status with gauges"""
    
    st.subheader("Current Regime Status")
    
    regime_data = st.session_state.regime_model
    current_probabilities = regime_data['probabilities'][-1]  # Latest probabilities
    current_regime = regime_data['sequence'][-1]  # Latest regime
    
    # Regime names
    regime_names = [f"Regime {i}" for i in range(len(current_probabilities))]
    
    # Create gauge charts for each regime
    cols = st.columns(len(current_probabilities))
    
    for i, (col, prob) in enumerate(zip(cols, current_probabilities)):
        with col:
            # Determine regime characteristics
            stats = regime_data['statistics'][i]
            
            # Create gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"{regime_names[i]}"},
                delta = {'reference': 100/len(current_probabilities)},  # Equal probability reference
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue" if i == current_regime else "lightgray"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Show regime characteristics
            if i == current_regime:
                st.success(f"**ACTIVE REGIME**")
            
            st.caption(f"""
            **Avg Return:** {stats['avg_return']:.3%}  
            **Volatility:** {stats['volatility']:.3%}  
            **Frequency:** {stats['frequency']:.1%}
            """)

def render_regime_timeline():
    """Render historical regime sequence timeline"""
    
    st.subheader("📅 Historical Regime Timeline")
    
    regime_data = st.session_state.regime_model
    features_df = regime_data['features']
    sequence = regime_data['sequence']
    probabilities = regime_data['probabilities']
    
    # Create regime timeline
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Regime Sequence', 'Regime Probabilities'],
        vertical_spacing=0.1,
        row_heights=[0.4, 0.6]
    )
    
    # Color mapping
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Regime sequence as colored line
    for regime in np.unique(sequence):
        regime_mask = sequence == regime
        regime_dates = features_df.index[regime_mask]
        regime_values = np.full(regime_mask.sum(), regime)
        
        fig.add_trace(
            go.Scatter(
                x=regime_dates,
                y=regime_values,
                mode='markers',
                marker=dict(color=colors[regime], size=8),
                name=f'Regime {regime}',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot 2: Stacked area chart of probabilities
    for regime in range(probabilities.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=features_df.index,
                y=probabilities[:, regime],
                mode='lines',
                fill='tonexty' if regime > 0 else 'tozeroy',
                name=f'Regime {regime} Probability',
                line=dict(color=colors[regime]),
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        title="Market Regime Evolution Over Time",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Regime", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1, range=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)

def render_transition_matrix():
    """Render regime transition matrix heatmap"""
    
    st.subheader("Regime Transition Matrix")
    
    regime_data = st.session_state.regime_model
    hmm_model = regime_data['model']
    n_regimes = regime_data['config']['n_regimes']
    
    # Get transition matrix
    transition_matrix = hmm_model.transmat_
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=[f'To Regime {i}' for i in range(n_regimes)],
            y=[f'From Regime {i}' for i in range(n_regimes)],
            colorscale='Blues',
            text=np.round(transition_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='From %{y}<br>To %{x}<br>Probability: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Regime Transition Probabilities",
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Transition Analysis:**")
        
        # Find most persistent regime
        persistence = np.diag(transition_matrix)
        most_persistent = np.argmax(persistence)
        
        st.metric(
            "Most Persistent Regime",
            f"Regime {most_persistent}",
            f"{persistence[most_persistent]:.1%} self-transition"
        )
        
        # Find most volatile regime
        volatility_scores = 1 - persistence
        most_volatile = np.argmax(volatility_scores)
        
        st.metric(
            "Most Volatile Regime", 
            f"Regime {most_volatile}",
            f"{volatility_scores[most_volatile]:.1%} transition rate"
        )
        
        # Average transition probability
        off_diagonal = transition_matrix.copy()
        np.fill_diagonal(off_diagonal, 0)
        avg_transition = off_diagonal.sum() / (n_regimes * (n_regimes - 1))
        
        st.metric(
            "Avg Transition Rate",
            f"{avg_transition:.1%}",
            "Between different regimes"
        )

def render_regime_characteristics():
    """Render detailed regime characteristics"""
    
    st.subheader("📈 Regime Characteristics")
    
    regime_data = st.session_state.regime_model
    regime_stats = regime_data['statistics']
    n_regimes = len(regime_stats)
    
    # Create tabs for each regime
    tabs = st.tabs([f"Regime {i}" for i in range(n_regimes)])
    
    for i, tab in enumerate(tabs):
        with tab:
            stats = regime_stats[i]
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Frequency", f"{stats['frequency']:.1%}")
            
            with col2:
                st.metric("Avg Return", f"{stats['avg_return']:.3%}")
            
            with col3:
                st.metric("Volatility", f"{stats['volatility']:.3%}")
            
            with col4:
                st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
            
            # Duration analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Duration", f"{stats['avg_duration']:.1f} days")
            
            with col2:
                st.metric("Max Duration", f"{stats['max_duration']} days")
            
            # Return distribution if available
            if not stats['return_distribution'].empty:
                st.markdown("**Return Distribution:**")
                
                dist_data = stats['return_distribution']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min", f"{dist_data['min']:.3%}")
                with col2:
                    st.metric("Median", f"{dist_data['50%']:.3%}")
                with col3:
                    st.metric("Max", f"{dist_data['max']:.3%}")

def render_regime_predictions():
    """Render regime predictions and model diagnostics"""
    
    st.subheader("🔮 Model Diagnostics & Predictions")
    
    regime_data = st.session_state.regime_model
    hmm_model = regime_data['model']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Performance:**")
        
        # Convergence info
        converged = hmm_model.monitor_.converged
        st.metric("Convergence", "✅ Yes" if converged else "❌ No")
        
        # Log likelihood progression
        if hasattr(hmm_model.monitor_, 'history') and hmm_model.monitor_.history:
            ll_history = hmm_model.monitor_.history
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=ll_history,
                mode='lines+markers',
                name='Log Likelihood',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="Training Convergence",
                xaxis_title="Iteration",
                yaxis_title="Log Likelihood",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Model Configuration:**")
        
        config = regime_data['config']
        
        st.info(f"""
        **Model Settings:**
        - Regimes: {config['n_regimes']}
        - Covariance: {config['covariance_type']}
        - Max Iterations: {config['n_iterations']}
        - Data Window: {config['lookback_window']} days
        
        **Features Used:**
        - Returns: {'✅' if config['features_used']['returns'] else '❌'}
        - Volatility: {'✅' if config['features_used']['volatility'] else '❌'}
        - Momentum: {'✅' if config['features_used']['momentum'] else '❌'}
        - Volume: {'✅' if config['features_used']['volume'] else '❌'}
        """)

def reset_regime_model():
    """Reset the regime model and clear results"""
    
    if 'regime_model' in st.session_state:
        del st.session_state.regime_model
    
    st.success("Regime model reset successfully!")
    safe_rerun()

def export_regime_results():
    """Export regime detection results to CSV"""
    
    regime_data = st.session_state.regime_model
    features_df = regime_data['features']
    sequence = regime_data['sequence']
    probabilities = regime_data['probabilities']
    
    # Create export DataFrame
    export_df = features_df.copy()
    export_df['Regime'] = sequence
    
    # Add probability columns
    for i in range(probabilities.shape[1]):
        export_df[f'Regime_{i}_Probability'] = probabilities[:, i]
    
    # Convert to CSV
    csv = export_df.to_csv()
    
    st.download_button(
        label="Download Regime Results CSV",
        data=csv,
        file_name=f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ Results prepared for download!")

# Main execution
if __name__ == "__main__":
    render_regime_detection()