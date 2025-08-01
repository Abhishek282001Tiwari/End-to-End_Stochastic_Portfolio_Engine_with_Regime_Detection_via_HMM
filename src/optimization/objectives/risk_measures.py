import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RiskMetricType(Enum):
    """Enumeration of risk metric types"""
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    OMEGA_RATIO = "omega_ratio"
    TAIL_RATIO = "tail_ratio"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"


@dataclass
class RiskMetric:
    """Container for risk metrics"""
    name: str
    value: float
    confidence_level: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestScenario:
    """Definition of a stress test scenario"""
    name: str
    description: str
    shocks: Dict[str, float]  # asset -> shock magnitude
    probability: Optional[float] = None
    historical_date: Optional[datetime] = None


@dataclass
class RiskDecomposition:
    """Risk decomposition results"""
    total_risk: float
    marginal_contributions: np.ndarray
    component_contributions: np.ndarray
    percentage_contributions: np.ndarray
    risk_budget_utilization: np.ndarray
    concentration_metrics: Dict[str, float]


class RiskMeasures:
    @staticmethod
    def value_at_risk(
        returns: np.ndarray, 
        confidence_level: float = 0.05,
        method: str = "historical"
    ) -> float:
        if method == "historical":
            return np.percentile(returns, confidence_level * 100)
        
        elif method == "parametric":
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(confidence_level)
            return mean_return + z_score * std_return
        
        elif method == "cornish_fisher":
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            z = stats.norm.ppf(confidence_level)
            z_cf = (z + 
                   (z**2 - 1) * skewness / 6 + 
                   (z**3 - 3*z) * kurtosis / 24 - 
                   (2*z**3 - 5*z) * skewness**2 / 36)
            
            return mean_return + z_cf * std_return
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def conditional_value_at_risk(
        returns: np.ndarray, 
        confidence_level: float = 0.05
    ) -> float:
        var = RiskMeasures.value_at_risk(returns, confidence_level, method="historical")
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def expected_shortfall(
        returns: np.ndarray, 
        confidence_level: float = 0.05
    ) -> float:
        return RiskMeasures.conditional_value_at_risk(returns, confidence_level)
    
    @staticmethod
    def maximum_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
        cumulative = np.cumprod(1 + returns) - 1
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - rolling_max
        
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        
        start_idx = np.argmax(cumulative[:end_idx + 1])
        
        return max_dd, start_idx, end_idx
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
        annual_return = np.mean(returns) * periods_per_year
        max_dd, _, _ = RiskMeasures.maximum_drawdown(returns)
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray, 
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0
        
        return np.mean(excess_returns) * np.sqrt(periods_per_year) / downside_deviation
    
    @staticmethod
    def omega_ratio(
        returns: np.ndarray, 
        threshold: float = 0.0
    ) -> float:
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns < 0]
        
        if len(losses) == 0:
            return np.inf if len(gains) > 0 else 1
        
        gain_sum = np.sum(gains) if len(gains) > 0 else 0
        loss_sum = np.abs(np.sum(losses))
        
        return gain_sum / loss_sum if loss_sum > 0 else np.inf
    
    @staticmethod
    def tail_ratio(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        upper_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        lower_percentile = np.percentile(returns, confidence_level * 100)
        
        if lower_percentile == 0:
            return np.inf if upper_percentile > 0 else 0
        
        return abs(upper_percentile / lower_percentile)
    
    @staticmethod
    def semi_variance(returns: np.ndarray, threshold: float = 0.0) -> float:
        downside_returns = returns[returns < threshold] - threshold
        return np.mean(downside_returns ** 2) if len(downside_returns) > 0 else 0
    
    @staticmethod
    def semi_deviation(returns: np.ndarray, threshold: float = 0.0) -> float:
        return np.sqrt(RiskMeasures.semi_variance(returns, threshold))
    
    @staticmethod
    def upside_deviation(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate upside deviation"""
        upside_returns = returns[returns > threshold] - threshold
        return np.sqrt(np.mean(upside_returns ** 2)) if len(upside_returns) > 0 else 0
    
    @staticmethod
    def gain_loss_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate gain-to-loss ratio"""
        gains = returns[returns > threshold]
        losses = returns[returns < threshold]
        
        if len(losses) == 0:
            return np.inf if len(gains) > 0 else 1
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(np.abs(losses))
        
        return avg_gain / avg_loss if avg_loss > 0 else np.inf
    
    @staticmethod
    def win_rate(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate win rate (percentage of returns above threshold)"""
        return np.mean(returns > threshold)
    
    @staticmethod
    def pain_index(returns: np.ndarray) -> float:
        """Calculate pain index (average drawdown)"""
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        return np.mean(np.abs(drawdowns))
    
    @staticmethod
    def ulcer_index(returns: np.ndarray) -> float:
        """Calculate Ulcer Index (RMS of drawdowns)"""
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        return np.sqrt(np.mean(drawdowns ** 2))
    
    @staticmethod
    def burke_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate Burke Ratio"""
        excess_returns = returns - risk_free_rate / periods_per_year
        ulcer_idx = RiskMeasures.ulcer_index(returns)
        
        if ulcer_idx == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0
        
        return np.mean(excess_returns) * np.sqrt(periods_per_year) / ulcer_idx
    
    @staticmethod
    def conditional_drawdown_at_risk(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Drawdown at Risk (CDaR)"""
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (rolling_max - cumulative) / rolling_max
        
        threshold = np.percentile(drawdowns, (1 - confidence_level) * 100)
        tail_drawdowns = drawdowns[drawdowns >= threshold]
        
        return np.mean(tail_drawdowns) if len(tail_drawdowns) > 0 else 0
    
    @staticmethod
    def diversification_ratio(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_vol = np.sum(weights * individual_vols)
        portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
    
    @staticmethod
    def effective_number_of_assets(weights: np.ndarray) -> float:
        """Calculate effective number of assets (inverse Herfindahl index)"""
        return 1 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 1
    
    @staticmethod
    def regime_conditional_var(returns: np.ndarray, regime_labels: np.ndarray, confidence_level: float = 0.05) -> Dict[int, float]:
        """Calculate VaR conditional on market regimes"""
        regime_vars = {}
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_returns = returns[regime_labels == regime]
            if len(regime_returns) > 10:  # Minimum sample size
                regime_vars[regime] = RiskMeasures.value_at_risk(regime_returns, confidence_level)
            else:
                regime_vars[regime] = np.nan
        
        return regime_vars


class PortfolioRiskCalculator:
    def __init__(self):
        self.risk_measures = RiskMeasures()
    
    def calculate_portfolio_risk_metrics(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
        confidence_levels: Optional[list] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Dict[str, Any]:
        if confidence_levels is None:
            confidence_levels = [0.01, 0.05, 0.10]
        
        portfolio_returns = returns @ weights
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        metrics = {
            "portfolio_volatility": portfolio_volatility,
            "annualized_volatility": portfolio_volatility * np.sqrt(periods_per_year),
            "portfolio_variance": portfolio_variance,
            "tracking_error": np.std(portfolio_returns),
            "information_ratio": np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        }
        
        for confidence_level in confidence_levels:
            var_hist = self.risk_measures.value_at_risk(portfolio_returns, confidence_level, "historical")
            var_param = self.risk_measures.value_at_risk(portfolio_returns, confidence_level, "parametric")
            cvar = self.risk_measures.conditional_value_at_risk(portfolio_returns, confidence_level)
            
            metrics[f"var_{int(confidence_level*100)}%_historical"] = var_hist
            metrics[f"var_{int(confidence_level*100)}%_parametric"] = var_param
            metrics[f"cvar_{int(confidence_level*100)}%"] = cvar
        
        max_dd, start_idx, end_idx = self.risk_measures.maximum_drawdown(portfolio_returns)
        metrics["maximum_drawdown"] = max_dd
        metrics["calmar_ratio"] = self.risk_measures.calmar_ratio(portfolio_returns, periods_per_year)
        
        metrics["sortino_ratio"] = self.risk_measures.sortino_ratio(
            portfolio_returns, risk_free_rate / periods_per_year, periods_per_year
        )
        
        metrics["omega_ratio"] = self.risk_measures.omega_ratio(portfolio_returns)
        metrics["tail_ratio"] = self.risk_measures.tail_ratio(portfolio_returns)
        
        metrics["semi_deviation"] = self.risk_measures.semi_deviation(portfolio_returns)
        
        metrics["skewness"] = stats.skew(portfolio_returns)
        metrics["kurtosis"] = stats.kurtosis(portfolio_returns)
        
        return metrics
    
    def calculate_component_contributions(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        portfolio_variance = weights.T @ covariance_matrix @ weights
        marginal_contributions = covariance_matrix @ weights
        
        component_contributions = weights * marginal_contributions
        percentage_contributions = component_contributions / portfolio_variance
        
        return {
            "marginal_contributions": marginal_contributions,
            "component_contributions": component_contributions,
            "percentage_contributions": percentage_contributions
        }
    
    def stress_test_portfolio(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        stress_scenarios: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        results = {}
        
        for scenario_name, shock in stress_scenarios.items():
            shocked_returns = returns + shock
            portfolio_return = np.sum(shocked_returns * weights)
            results[f"{scenario_name}_return"] = portfolio_return
        
        return results
    
    def monte_carlo_risk_simulation(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        n_simulations: int = 10000,
        time_horizon: int = 252
    ) -> Dict[str, Any]:
        logger.info(f"Running Monte Carlo risk simulation with {n_simulations} paths")
        
        simulated_returns = np.random.multivariate_normal(
            expected_returns,
            covariance_matrix,
            (n_simulations, time_horizon)
        )
        
        portfolio_paths = np.zeros((n_simulations, time_horizon))
        for i in range(n_simulations):
            portfolio_returns = simulated_returns[i] @ weights
            portfolio_paths[i] = np.cumprod(1 + portfolio_returns) - 1
        
        final_returns = portfolio_paths[:, -1]
        
        return {
            "mean_final_return": np.mean(final_returns),
            "std_final_return": np.std(final_returns),
            "var_5%": np.percentile(final_returns, 5),
            "var_1%": np.percentile(final_returns, 1),
            "probability_of_loss": np.mean(final_returns < 0),
            "expected_shortfall_5%": np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            "max_simulated_loss": np.min(final_returns),
            "max_simulated_gain": np.max(final_returns),
            "paths": portfolio_paths
        }


class RiskBudgetOptimizer:
    def __init__(self, target_risk_budget: np.ndarray):
        self.target_risk_budget = target_risk_budget / np.sum(target_risk_budget)
    
    def optimize_risk_budget(
        self,
        covariance_matrix: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        n_assets = covariance_matrix.shape[0]
        weights = np.ones(n_assets) / n_assets
        
        for iteration in range(max_iterations):
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            marginal_contrib = (covariance_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            g_risk = risk_contrib - self.target_risk_budget
            
            if np.max(np.abs(g_risk)) < tolerance:
                logger.info(f"Risk budget optimization converged after {iteration} iterations")
                break
            
            A = np.outer(marginal_contrib, marginal_contrib) / (portfolio_vol ** 2)
            A += np.diag(marginal_contrib / portfolio_vol)
            
            try:
                delta_weights = -np.linalg.solve(A, g_risk)
                weights += 0.1 * delta_weights
                weights = np.maximum(weights, 0.001)
                weights /= np.sum(weights)
                
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix encountered in risk budget optimization")
                break
        
        return weights