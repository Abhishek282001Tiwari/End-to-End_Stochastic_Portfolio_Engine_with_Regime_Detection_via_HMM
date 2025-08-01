import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from src.models.hmm.hmm_engine import RegimeDetectionHMM
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RegimeAnalyzer:
    def __init__(self, hmm_model: RegimeDetectionHMM):
        self.hmm_model = hmm_model
        self.analysis_cache = {}
        
    def analyze_regime_characteristics(
        self, 
        X: pd.DataFrame, 
        price_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        logger.info("Analyzing regime characteristics")
        
        if not self.hmm_model.is_fitted:
            raise ValueError("HMM model must be fitted before analysis")
        
        states = self.hmm_model.predict_regimes(X)
        probabilities = self.hmm_model.predict_regime_probabilities(X)
        
        analysis = {
            "regime_periods": self._identify_regime_periods(X.index, states),
            "regime_statistics": self._calculate_regime_statistics(X, states),
            "transition_analysis": self._analyze_transitions(states),
            "persistence_analysis": self._analyze_persistence(states),
        }
        
        if price_data is not None:
            analysis["performance_by_regime"] = self._analyze_regime_performance(
                price_data, states, X.index
            )
        
        return analysis
    
    def _identify_regime_periods(
        self, 
        dates: pd.DatetimeIndex, 
        states: np.ndarray
    ) -> List[Dict[str, Any]]:
        periods = []
        current_regime = states[0]
        start_date = dates[0]
        
        for i in range(1, len(states)):
            if states[i] != current_regime:
                periods.append({
                    "regime": current_regime,
                    "regime_name": self.hmm_model.regime_names[current_regime],
                    "start_date": start_date,
                    "end_date": dates[i-1],
                    "duration_days": (dates[i-1] - start_date).days
                })
                
                current_regime = states[i]
                start_date = dates[i]
        
        periods.append({
            "regime": current_regime,
            "regime_name": self.hmm_model.regime_names[current_regime],
            "start_date": start_date,
            "end_date": dates[-1],
            "duration_days": (dates[-1] - start_date).days
        })
        
        return periods
    
    def _calculate_regime_statistics(
        self, 
        X: pd.DataFrame, 
        states: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        statistics = {}
        
        for regime in range(self.hmm_model.n_components):
            regime_mask = states == regime
            regime_data = X[regime_mask]
            
            if len(regime_data) > 0:
                stats = {
                    "frequency": np.sum(regime_mask) / len(states),
                    "mean_values": regime_data.mean().to_dict(),
                    "std_values": regime_data.std().to_dict(),
                    "skewness": regime_data.skew().to_dict(),
                    "kurtosis": regime_data.kurtosis().to_dict()
                }
            else:
                stats = {
                    "frequency": 0,
                    "mean_values": {},
                    "std_values": {},
                    "skewness": {},
                    "kurtosis": {}
                }
            
            statistics[regime] = stats
        
        return statistics
    
    def _analyze_transitions(self, states: np.ndarray) -> Dict[str, Any]:
        transition_counts = np.zeros((self.hmm_model.n_components, self.hmm_model.n_components))
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state, next_state] += 1
        
        transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        transition_probs = np.nan_to_num(transition_probs)
        
        return {
            "transition_matrix": transition_probs,
            "transition_counts": transition_counts,
            "most_common_transitions": self._find_common_transitions(transition_counts)
        }
    
    def _find_common_transitions(self, transition_counts: np.ndarray) -> List[Dict[str, Any]]:
        transitions = []
        
        for i in range(self.hmm_model.n_components):
            for j in range(self.hmm_model.n_components):
                if i != j and transition_counts[i, j] > 0:
                    transitions.append({
                        "from_regime": i,
                        "from_name": self.hmm_model.regime_names[i],
                        "to_regime": j,
                        "to_name": self.hmm_model.regime_names[j],
                        "count": int(transition_counts[i, j]),
                        "probability": transition_counts[i, j] / transition_counts[i, :].sum()
                    })
        
        return sorted(transitions, key=lambda x: x["count"], reverse=True)[:10]
    
    def _analyze_persistence(self, states: np.ndarray) -> Dict[int, Dict[str, float]]:
        persistence = {}
        
        for regime in range(self.hmm_model.n_components):
            regime_periods = []
            current_length = 0
            
            for state in states:
                if state == regime:
                    current_length += 1
                else:
                    if current_length > 0:
                        regime_periods.append(current_length)
                    current_length = 0
            
            if current_length > 0:
                regime_periods.append(current_length)
            
            if regime_periods:
                persistence[regime] = {
                    "average_duration": np.mean(regime_periods),
                    "median_duration": np.median(regime_periods),
                    "max_duration": np.max(regime_periods),
                    "min_duration": np.min(regime_periods),
                    "total_periods": len(regime_periods)
                }
            else:
                persistence[regime] = {
                    "average_duration": 0,
                    "median_duration": 0,
                    "max_duration": 0,
                    "min_duration": 0,
                    "total_periods": 0
                }
        
        return persistence
    
    def _analyze_regime_performance(
        self, 
        price_data: pd.DataFrame, 
        states: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[int, Dict[str, float]]:
        returns = price_data.pct_change().dropna()
        
        aligned_returns = returns.reindex(dates).dropna()
        aligned_states = states[:len(aligned_returns)]
        
        performance = {}
        
        for regime in range(self.hmm_model.n_components):
            regime_mask = aligned_states == regime
            regime_returns = aligned_returns[regime_mask]
            
            if len(regime_returns) > 0:
                performance[regime] = {
                    "mean_return": regime_returns.mean().mean(),
                    "volatility": regime_returns.std().mean(),
                    "sharpe_ratio": (regime_returns.mean() / regime_returns.std()).mean(),
                    "max_drawdown": self._calculate_max_drawdown(regime_returns),
                    "positive_days_pct": (regime_returns > 0).mean().mean(),
                    "skewness": regime_returns.skew().mean(),
                    "kurtosis": regime_returns.kurtosis().mean()
                }
            else:
                performance[regime] = {
                    "mean_return": 0,
                    "volatility": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "positive_days_pct": 0,
                    "skewness": 0,
                    "kurtosis": 0
                }
        
        return performance
    
    def _calculate_max_drawdown(self, returns: pd.DataFrame) -> float:
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1)
        
        return drawdown.min().min()
    
    def generate_regime_report(
        self, 
        X: pd.DataFrame, 
        price_data: Optional[pd.DataFrame] = None
    ) -> str:
        analysis = self.analyze_regime_characteristics(X, price_data)
        
        report = "Market Regime Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        report += "REGIME DISTRIBUTION:\n"
        regime_stats = analysis["regime_statistics"]
        for regime, stats in regime_stats.items():
            regime_name = self.hmm_model.regime_names[regime]
            frequency = stats["frequency"]
            report += f"{regime_name}: {frequency:.1%} of time\n"
        
        report += "\nREGIME PERSISTENCE:\n"
        persistence = analysis["persistence_analysis"]
        for regime, stats in persistence.items():
            regime_name = self.hmm_model.regime_names[regime]
            avg_duration = stats["average_duration"]
            max_duration = stats["max_duration"]
            report += f"{regime_name}: Average {avg_duration:.1f} days, Max {max_duration} days\n"
        
        report += "\nCOMMON TRANSITIONS:\n"
        transitions = analysis["transition_analysis"]["most_common_transitions"]
        for trans in transitions[:5]:
            report += f"{trans['from_name']} → {trans['to_name']}: "
            report += f"{trans['count']} times ({trans['probability']:.1%})\n"
        
        if "performance_by_regime" in analysis:
            report += "\nPERFORMANCE BY REGIME:\n"
            performance = analysis["performance_by_regime"]
            for regime, perf in performance.items():
                regime_name = self.hmm_model.regime_names[regime]
                mean_return = perf["mean_return"] * 252
                volatility = perf["volatility"] * np.sqrt(252)
                sharpe = perf["sharpe_ratio"]
                report += f"{regime_name}: Return {mean_return:.1%}, "
                report += f"Vol {volatility:.1%}, Sharpe {sharpe:.2f}\n"
        
        return report
    
    def plot_regime_timeline(
        self, 
        X: pd.DataFrame, 
        price_data: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        if not self.hmm_model.is_fitted:
            raise ValueError("HMM model must be fitted before plotting")
        
        states = self.hmm_model.predict_regimes(X)
        probabilities = self.hmm_model.predict_regime_probabilities(X)
        
        if price_data is not None:
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        colors = ['red', 'orange', 'green']
        
        if price_data is not None:
            axes[0].plot(price_data.index, price_data.iloc[:, 0], 'k-', alpha=0.7)
            axes[0].set_ylabel('Price')
            axes[0].set_title('Asset Price and Market Regimes')
            
            for i in range(len(states)):
                regime = states[i]
                if i < len(price_data):
                    axes[0].scatter(
                        price_data.index[i], 
                        price_data.iloc[i, 0], 
                        c=colors[regime], 
                        alpha=0.6, 
                        s=10
                    )
            
            prob_ax = axes[1]
            timeline_ax = axes[2]
        else:
            prob_ax = axes[0]
            timeline_ax = axes[1]
        
        for regime in range(self.hmm_model.n_components):
            regime_name = self.hmm_model.regime_names[regime]
            prob_ax.plot(
                X.index, 
                probabilities[:, regime], 
                label=regime_name,
                color=colors[regime],
                alpha=0.7
            )
        
        prob_ax.set_ylabel('Regime Probability')
        prob_ax.set_title('Regime Probabilities Over Time')
        prob_ax.legend()
        prob_ax.grid(True, alpha=0.3)
        
        regime_colors = [colors[state] for state in states]
        timeline_ax.scatter(X.index, states, c=regime_colors, alpha=0.8, s=20)
        timeline_ax.set_ylabel('Regime')
        timeline_ax.set_xlabel('Date')
        timeline_ax.set_title('Detected Market Regimes')
        timeline_ax.set_yticks(range(self.hmm_model.n_components))
        timeline_ax.set_yticklabels([
            self.hmm_model.regime_names[i] 
            for i in range(self.hmm_model.n_components)
        ])
        timeline_ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_regime_characteristics(self, X: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
        if not self.hmm_model.is_fitted:
            raise ValueError("HMM model must be fitted before plotting")
        
        states = self.hmm_model.predict_regimes(X)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        regime_names = [self.hmm_model.regime_names[i] for i in range(self.hmm_model.n_components)]
        colors = ['red', 'orange', 'green']
        
        for i, feature in enumerate(X.columns[:4]):
            ax = axes[i // 2, i % 2]
            
            for regime in range(self.hmm_model.n_components):
                regime_data = X[states == regime][feature].dropna()
                if len(regime_data) > 0:
                    ax.hist(
                        regime_data, 
                        alpha=0.6, 
                        label=regime_names[regime],
                        color=colors[regime],
                        bins=30
                    )
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature} by Regime')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig