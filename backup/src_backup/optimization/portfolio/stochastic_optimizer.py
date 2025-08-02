import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod
import warnings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class StochasticOptimizer(ABC):
    @abstractmethod
    def optimize(
        self, 
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        pass


class MeanVarianceOptimizer(StochasticOptimizer):
    def __init__(
        self,
        risk_aversion: float = 1.0,
        transaction_costs: float = 0.001,
        max_weight: float = 0.2,
        min_weight: float = 0.0,
        leverage: float = 1.0
    ):
        self.risk_aversion = risk_aversion
        self.transaction_costs = transaction_costs
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.leverage = leverage
    
    def optimize(
        self, 
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info("Running Mean-Variance optimization")
        
        n_assets = len(expected_returns)
        
        weights = cp.Variable(n_assets)
        
        portfolio_return = weights.T @ expected_returns
        portfolio_risk = cp.quad_form(weights, covariance_matrix)
        
        objective = portfolio_return - 0.5 * self.risk_aversion * portfolio_risk
        
        constraints = [
            cp.sum(weights) == self.leverage,
            weights >= self.min_weight,
            weights <= self.max_weight
        ]
        
        if current_weights is not None:
            turnover = cp.sum(cp.abs(weights - current_weights))
            objective -= self.transaction_costs * turnover
        
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = weights.value
            portfolio_ret = float(expected_returns.T @ optimal_weights)
            portfolio_vol = float(np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights))
            
            return {
                "weights": optimal_weights,
                "expected_return": portfolio_ret,
                "expected_volatility": portfolio_vol,
                "sharpe_ratio": portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0,
                "status": problem.status,
                "objective_value": problem.value
            }
        else:
            logger.error(f"Optimization failed with status: {problem.status}")
            return {
                "weights": np.ones(n_assets) / n_assets,
                "expected_return": 0,
                "expected_volatility": 0,
                "sharpe_ratio": 0,
                "status": problem.status,
                "objective_value": None
            }


class BlackLittermanOptimizer(StochasticOptimizer):
    def __init__(
        self,
        tau: float = 0.025,
        risk_aversion: float = 3.0,
        transaction_costs: float = 0.001,
        max_weight: float = 0.2,
        min_weight: float = 0.0
    ):
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.transaction_costs = transaction_costs
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def optimize(
        self, 
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        market_caps: Optional[np.ndarray] = None,
        views_matrix: Optional[np.ndarray] = None,
        views_returns: Optional[np.ndarray] = None,
        views_uncertainty: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info("Running Black-Litterman optimization")
        
        n_assets = len(expected_returns)
        
        if market_caps is None:
            market_caps = np.ones(n_assets) / n_assets
        else:
            market_caps = market_caps / market_caps.sum()
        
        implied_returns = self.risk_aversion * (covariance_matrix @ market_caps)
        
        if views_matrix is not None and views_returns is not None:
            if views_uncertainty is None:
                views_uncertainty = np.diag(np.diag(views_matrix @ (self.tau * covariance_matrix) @ views_matrix.T))
            
            tau_sigma = self.tau * covariance_matrix
            
            M1 = np.linalg.inv(tau_sigma)
            M2 = views_matrix.T @ np.linalg.inv(views_uncertainty) @ views_matrix
            M3 = np.linalg.inv(tau_sigma) @ implied_returns
            M4 = views_matrix.T @ np.linalg.inv(views_uncertainty) @ views_returns
            
            bl_returns = np.linalg.inv(M1 + M2) @ (M3 + M4)
            bl_covariance = np.linalg.inv(M1 + M2)
        else:
            bl_returns = implied_returns
            bl_covariance = self.tau * covariance_matrix
        
        mv_optimizer = MeanVarianceOptimizer(
            risk_aversion=self.risk_aversion,
            transaction_costs=self.transaction_costs,
            max_weight=self.max_weight,
            min_weight=self.min_weight
        )
        
        return mv_optimizer.optimize(bl_returns, covariance_matrix + bl_covariance, **kwargs)


class RiskParityOptimizer(StochasticOptimizer):
    def __init__(
        self,
        target_risk_contributions: Optional[np.ndarray] = None,
        max_weight: float = 0.5,
        min_weight: float = 0.01
    ):
        self.target_risk_contributions = target_risk_contributions
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def optimize(
        self, 
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info("Running Risk Parity optimization")
        
        n_assets = len(expected_returns)
        
        if self.target_risk_contributions is None:
            target_rc = np.ones(n_assets) / n_assets
        else:
            target_rc = self.target_risk_contributions
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            marginal_contrib = covariance_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            
            return np.sum((contrib - target_rc * contrib.sum()) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_ret = float(expected_returns.T @ optimal_weights)
            portfolio_vol = float(np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights))
            
            return {
                "weights": optimal_weights,
                "expected_return": portfolio_ret,
                "expected_volatility": portfolio_vol,
                "sharpe_ratio": portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0,
                "status": "optimal",
                "objective_value": result.fun
            }
        else:
            logger.error(f"Risk parity optimization failed: {result.message}")
            return {
                "weights": np.ones(n_assets) / n_assets,
                "expected_return": 0,
                "expected_volatility": 0,
                "sharpe_ratio": 0,
                "status": "failed",
                "objective_value": None
            }


class MonteCarloOptimizer(StochasticOptimizer):
    def __init__(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.05,
        base_optimizer: Optional[StochasticOptimizer] = None
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.base_optimizer = base_optimizer or MeanVarianceOptimizer()
    
    def optimize(
        self, 
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info(f"Running Monte Carlo optimization with {self.n_simulations} simulations")
        
        n_assets = len(expected_returns)
        
        simulated_returns = np.random.multivariate_normal(
            expected_returns, 
            covariance_matrix, 
            self.n_simulations
        )
        
        all_weights = []
        all_metrics = []
        
        for i in range(min(100, self.n_simulations // 100)):
            sample_returns = simulated_returns[i * (self.n_simulations // 100):(i + 1) * (self.n_simulations // 100)]
            mean_returns = sample_returns.mean(axis=0)
            
            result = self.base_optimizer.optimize(mean_returns, covariance_matrix, **kwargs)
            
            if result["status"] in ["optimal", "optimal_inaccurate"]:
                all_weights.append(result["weights"])
                all_metrics.append({
                    "return": result["expected_return"],
                    "volatility": result["expected_volatility"],
                    "sharpe": result["sharpe_ratio"]
                })
        
        if all_weights:
            weights_array = np.array(all_weights)
            mean_weights = weights_array.mean(axis=0)
            
            portfolio_returns = simulated_returns @ mean_weights
            var_estimate = np.percentile(portfolio_returns, self.confidence_level * 100)
            cvar_estimate = portfolio_returns[portfolio_returns <= var_estimate].mean()
            
            portfolio_ret = float(expected_returns.T @ mean_weights)
            portfolio_vol = float(np.sqrt(mean_weights.T @ covariance_matrix @ mean_weights))
            
            return {
                "weights": mean_weights,
                "expected_return": portfolio_ret,
                "expected_volatility": portfolio_vol,
                "sharpe_ratio": portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0,
                "var": var_estimate,
                "cvar": cvar_estimate,
                "weight_std": weights_array.std(axis=0),
                "status": "optimal",
                "n_successful_optimizations": len(all_weights)
            }
        else:
            return {
                "weights": np.ones(n_assets) / n_assets,
                "expected_return": 0,
                "expected_volatility": 0,
                "sharpe_ratio": 0,
                "var": 0,
                "cvar": 0,
                "weight_std": np.zeros(n_assets),
                "status": "failed",
                "n_successful_optimizations": 0
            }


class RegimeAwareOptimizer:
    def __init__(
        self,
        regime_optimizers: Dict[int, StochasticOptimizer],
        regime_transition_matrix: Optional[np.ndarray] = None,
        lookback_window: int = 20
    ):
        self.regime_optimizers = regime_optimizers
        self.regime_transition_matrix = regime_transition_matrix
        self.lookback_window = lookback_window
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrices: Dict[int, np.ndarray],
        regime_probabilities: np.ndarray,
        current_regime: int,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info("Running regime-aware optimization")
        
        regime_weights = {}
        regime_metrics = {}
        
        for regime, optimizer in self.regime_optimizers.items():
            if regime in covariance_matrices:
                result = optimizer.optimize(
                    expected_returns, 
                    covariance_matrices[regime], 
                    **kwargs
                )
                regime_weights[regime] = result["weights"]
                regime_metrics[regime] = result
        
        if self.regime_transition_matrix is not None:
            future_regime_probs = self._predict_future_regimes(
                current_regime, 
                regime_probabilities
            )
        else:
            future_regime_probs = regime_probabilities
        
        final_weights = np.zeros_like(expected_returns)
        for regime, weights in regime_weights.items():
            final_weights += future_regime_probs[regime] * weights
        
        portfolio_ret = float(expected_returns.T @ final_weights)
        
        weighted_covariance = np.zeros_like(covariance_matrices[current_regime])
        for regime, prob in enumerate(future_regime_probs):
            if regime in covariance_matrices and prob > 0:
                weighted_covariance += prob * covariance_matrices[regime]
        
        portfolio_vol = float(np.sqrt(final_weights.T @ weighted_covariance @ final_weights))
        
        return {
            "weights": final_weights,
            "expected_return": portfolio_ret,
            "expected_volatility": portfolio_vol,
            "sharpe_ratio": portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0,
            "regime_weights": regime_weights,
            "regime_probabilities": future_regime_probs,
            "regime_metrics": regime_metrics,
            "status": "optimal"
        }
    
    def _predict_future_regimes(
        self, 
        current_regime: int, 
        current_probabilities: np.ndarray
    ) -> np.ndarray:
        future_probs = current_probabilities.copy()
        
        for _ in range(self.lookback_window):
            future_probs = future_probs @ self.regime_transition_matrix
        
        return future_probs


class PortfolioOptimizationEngine:
    def __init__(self):
        self.optimizers = {
            "mean_variance": MeanVarianceOptimizer(),
            "black_litterman": BlackLittermanOptimizer(),
            "risk_parity": RiskParityOptimizer(),
            "monte_carlo": MonteCarloOptimizer()
        }
        
    def optimize_portfolio(
        self,
        method: str,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimizer = self.optimizers[method]
        return optimizer.optimize(expected_returns, covariance_matrix, **kwargs)
    
    def add_optimizer(self, name: str, optimizer: StochasticOptimizer):
        self.optimizers[name] = optimizer
    
    def compare_optimizers(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        if methods is None:
            methods = list(self.optimizers.keys())
        
        results = []
        
        for method in methods:
            try:
                result = self.optimize_portfolio(
                    method, expected_returns, covariance_matrix, **kwargs
                )
                
                results.append({
                    "method": method,
                    "expected_return": result["expected_return"],
                    "expected_volatility": result["expected_volatility"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "status": result["status"],
                    "max_weight": result["weights"].max(),
                    "min_weight": result["weights"].min(),
                    "concentration": np.sum(result["weights"] ** 2)
                })
                
            except Exception as e:
                logger.error(f"Error with {method} optimizer: {e}")
                results.append({
                    "method": method,
                    "expected_return": np.nan,
                    "expected_volatility": np.nan,
                    "sharpe_ratio": np.nan,
                    "status": "error",
                    "max_weight": np.nan,
                    "min_weight": np.nan,
                    "concentration": np.nan
                })
        
        return pd.DataFrame(results)