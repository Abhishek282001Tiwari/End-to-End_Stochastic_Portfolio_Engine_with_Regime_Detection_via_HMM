import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from src.optimization.objectives.risk_measures import RiskMeasures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class StochasticProcess(Enum):
    """Available stochastic processes"""
    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    JUMP_DIFFUSION = "jump_diffusion"
    MEAN_REVERSION = "mean_reversion"
    HESTON_STOCHASTIC_VOLATILITY = "heston"
    REGIME_SWITCHING = "regime_switching"
    LEVY_PROCESS = "levy"
    FRACTIONAL_BROWNIAN_MOTION = "fbm"
    CIR_PROCESS = "cir"


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation"""
    n_simulations: int = 10000
    time_horizon: int = 252  # Trading days
    dt: float = 1/252  # Daily time step
    random_seed: Optional[int] = None
    parallel_processing: bool = True
    n_workers: Optional[int] = None
    confidence_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    antithetic_variates: bool = True
    control_variates: bool = False


@dataclass
class ProcessParameters:
    """Parameters for stochastic processes"""
    mu: Union[float, np.ndarray] = 0.08  # Drift/mean
    sigma: Union[float, np.ndarray] = 0.20  # Volatility
    kappa: Optional[float] = None  # Mean reversion speed
    theta: Optional[float] = None  # Long-term mean
    jump_intensity: Optional[float] = None  # Jump frequency
    jump_mean: Optional[float] = None  # Jump size mean
    jump_std: Optional[float] = None  # Jump size volatility
    hurst: Optional[float] = None  # Hurst parameter for fractional BM
    regime_transition_matrix: Optional[np.ndarray] = None
    regime_parameters: Optional[List[Dict]] = None


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation"""
    paths: np.ndarray
    final_values: np.ndarray
    statistics: Dict[str, float]
    risk_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    process_type: StochasticProcess
    parameters: ProcessParameters
    config: SimulationConfig
    execution_time: float


class BaseStochasticProcess(ABC):
    """Abstract base class for stochastic processes"""
    
    def __init__(self, parameters: ProcessParameters):
        self.parameters = parameters
    
    @abstractmethod
    def simulate_path(
        self, 
        n_steps: int, 
        dt: float, 
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """Simulate a single path of the stochastic process"""
        pass
    
    @abstractmethod
    def simulate_multiple_paths(
        self, 
        n_paths: int, 
        n_steps: int, 
        dt: float,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """Simulate multiple paths efficiently"""
        pass


class GeometricBrownianMotion(BaseStochasticProcess):
    """Geometric Brownian Motion process: dS = μS dt + σS dW"""
    
    def simulate_path(
        self, 
        n_steps: int, 
        dt: float, 
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        mu = self.parameters.mu
        sigma = self.parameters.sigma
        
        # Generate random increments
        dW = random_state.normal(0, np.sqrt(dt), n_steps)
        
        # Calculate log returns
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        
        # Convert to price path (assuming S0 = 1)
        path = np.exp(np.cumsum(log_returns))
        return np.concatenate([[1.0], path])
    
    def simulate_multiple_paths(
        self, 
        n_paths: int, 
        n_steps: int, 
        dt: float,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        mu = self.parameters.mu
        sigma = self.parameters.sigma
        
        # Generate all random increments at once
        dW = random_state.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Calculate log returns
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        
        # Convert to price paths
        paths = np.exp(np.cumsum(log_returns, axis=1))
        
        # Add initial value
        initial_values = np.ones((n_paths, 1))
        return np.hstack([initial_values, paths])


class JumpDiffusionProcess(BaseStochasticProcess):
    """Merton Jump Diffusion: dS = μS dt + σS dW + S dJ"""
    
    def simulate_path(
        self, 
        n_steps: int, 
        dt: float, 
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        mu = self.parameters.mu
        sigma = self.parameters.sigma
        jump_intensity = self.parameters.jump_intensity or 0.1
        jump_mean = self.parameters.jump_mean or -0.1
        jump_std = self.parameters.jump_std or 0.2
        
        # Diffusion component
        dW = random_state.normal(0, np.sqrt(dt), n_steps)
        
        # Jump component
        jump_times = random_state.poisson(jump_intensity * dt, n_steps)
        jump_sizes = random_state.normal(jump_mean, jump_std, n_steps)
        jumps = jump_times * jump_sizes
        
        # Combined process
        log_returns = (mu - 0.5 * sigma**2 - jump_intensity * jump_mean) * dt + sigma * dW + jumps
        
        path = np.exp(np.cumsum(log_returns))
        return np.concatenate([[1.0], path])
    
    def simulate_multiple_paths(
        self, 
        n_paths: int, 
        n_steps: int, 
        dt: float,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        mu = self.parameters.mu
        sigma = self.parameters.sigma
        jump_intensity = self.parameters.jump_intensity or 0.1
        jump_mean = self.parameters.jump_mean or -0.1
        jump_std = self.parameters.jump_std or 0.2
        
        # Diffusion components
        dW = random_state.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Jump components
        jump_times = random_state.poisson(jump_intensity * dt, (n_paths, n_steps))
        jump_sizes = random_state.normal(jump_mean, jump_std, (n_paths, n_steps))
        jumps = jump_times * jump_sizes
        
        # Combined process
        log_returns = ((mu - 0.5 * sigma**2 - jump_intensity * jump_mean) * dt + 
                      sigma * dW + jumps)
        
        paths = np.exp(np.cumsum(log_returns, axis=1))
        initial_values = np.ones((n_paths, 1))
        return np.hstack([initial_values, paths])


class MeanReversionProcess(BaseStochasticProcess):
    """Ornstein-Uhlenbeck process: dX = κ(θ - X) dt + σ dW"""
    
    def simulate_path(
        self, 
        n_steps: int, 
        dt: float, 
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        kappa = self.parameters.kappa or 2.0
        theta = self.parameters.theta or 0.0
        sigma = self.parameters.sigma
        
        path = np.zeros(n_steps + 1)
        path[0] = theta  # Start at long-term mean
        
        for i in range(n_steps):
            dW = random_state.normal(0, np.sqrt(dt))
            path[i + 1] = (path[i] + kappa * (theta - path[i]) * dt + 
                          sigma * dW)
        
        return path
    
    def simulate_multiple_paths(
        self, 
        n_paths: int, 
        n_steps: int, 
        dt: float,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        kappa = self.parameters.kappa or 2.0
        theta = self.parameters.theta or 0.0
        sigma = self.parameters.sigma
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = theta
        
        for i in range(n_steps):
            dW = random_state.normal(0, np.sqrt(dt), n_paths)
            paths[:, i + 1] = (paths[:, i] + 
                              kappa * (theta - paths[:, i]) * dt + 
                              sigma * dW)
        
        return paths


class HestonStochasticVolatility(BaseStochasticProcess):
    """Heston model with stochastic volatility"""
    
    def simulate_path(
        self, 
        n_steps: int, 
        dt: float, 
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Heston parameters
        mu = self.parameters.mu
        kappa = self.parameters.kappa or 2.0  # Vol mean reversion speed
        theta = self.parameters.theta or 0.04  # Long-term vol
        sigma_v = self.parameters.sigma or 0.3  # Vol of vol
        rho = -0.7  # Correlation between price and vol
        
        # Initialize
        S = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)
        S[0] = 1.0
        V[0] = theta
        
        for i in range(n_steps):
            # Correlated random numbers
            Z1 = random_state.normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * random_state.normal()
            
            # Update variance (with reflection at zero)
            dV = kappa * (theta - V[i]) * dt + sigma_v * np.sqrt(max(V[i], 0)) * np.sqrt(dt) * Z2
            V[i + 1] = max(V[i] + dV, 0)
            
            # Update price
            dS = mu * S[i] * dt + np.sqrt(max(V[i], 0)) * S[i] * np.sqrt(dt) * Z1
            S[i + 1] = S[i] + dS
        
        return S
    
    def simulate_multiple_paths(
        self, 
        n_paths: int, 
        n_steps: int, 
        dt: float,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        # For simplicity, use sequential simulation
        # In practice, this could be vectorized
        paths = np.zeros((n_paths, n_steps + 1))
        
        for path_idx in range(n_paths):
            path = self.simulate_path(n_steps, dt, random_state)
            paths[path_idx] = path
        
        return paths


class RegimeSwitchingProcess(BaseStochasticProcess):
    """Regime-switching model with different parameters per regime"""
    
    def simulate_path(
        self, 
        n_steps: int, 
        dt: float, 
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        transition_matrix = self.parameters.regime_transition_matrix
        regime_params = self.parameters.regime_parameters
        
        if transition_matrix is None or regime_params is None:
            raise ValueError("Regime switching requires transition matrix and parameters")
        
        n_regimes = len(regime_params)
        path = np.zeros(n_steps + 1)
        path[0] = 1.0
        
        # Initialize regime
        current_regime = random_state.choice(n_regimes)
        
        for i in range(n_steps):
            # Get current regime parameters
            mu = regime_params[current_regime]['mu']
            sigma = regime_params[current_regime]['sigma']
            
            # Generate return
            dW = random_state.normal(0, np.sqrt(dt))
            log_return = (mu - 0.5 * sigma**2) * dt + sigma * dW
            path[i + 1] = path[i] * np.exp(log_return)
            
            # Switch regime
            transition_probs = transition_matrix[current_regime]
            current_regime = random_state.choice(n_regimes, p=transition_probs)
        
        return path
    
    def simulate_multiple_paths(
        self, 
        n_paths: int, 
        n_steps: int, 
        dt: float,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        
        paths = np.zeros((n_paths, n_steps + 1))
        
        for path_idx in range(n_paths):
            path = self.simulate_path(n_steps, dt, random_state)
            paths[path_idx] = path
        
        return paths


class MonteCarloEngine:
    """Comprehensive Monte Carlo simulation engine"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.risk_measures = RiskMeasures()
        
        # Set up process mappings
        self.process_classes = {
            StochasticProcess.GEOMETRIC_BROWNIAN_MOTION: GeometricBrownianMotion,
            StochasticProcess.JUMP_DIFFUSION: JumpDiffusionProcess,
            StochasticProcess.MEAN_REVERSION: MeanReversionProcess,
            StochasticProcess.HESTON_STOCHASTIC_VOLATILITY: HestonStochasticVolatility,
            StochasticProcess.REGIME_SWITCHING: RegimeSwitchingProcess
        }
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        if config.n_workers is None:
            self.config.n_workers = min(mp.cpu_count(), 8)
    
    def simulate_portfolio_paths(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        process_type: StochasticProcess = StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
        process_parameters: Optional[ProcessParameters] = None
    ) -> SimulationResult:
        """
        Simulate portfolio paths using specified stochastic process
        """
        logger.info(f"Running Monte Carlo simulation with {self.config.n_simulations} paths")
        start_time = datetime.now()
        
        # Set default parameters if not provided
        if process_parameters is None:
            process_parameters = ProcessParameters(
                mu=expected_returns,
                sigma=np.sqrt(np.diag(covariance_matrix))
            )
        
        # Create process instance
        if process_type not in self.process_classes:
            raise ValueError(f"Unsupported process type: {process_type}")
        
        process_class = self.process_classes[process_type]
        
        # Run simulation
        if self.config.parallel_processing and self.config.n_simulations > 1000:
            paths = self._simulate_parallel(
                process_class, process_parameters, weights, covariance_matrix
            )
        else:
            paths = self._simulate_sequential(
                process_class, process_parameters, weights, covariance_matrix
            )
        
        # Calculate portfolio paths
        portfolio_paths = self._calculate_portfolio_paths(paths, weights)
        
        # Calculate results
        final_values = portfolio_paths[:, -1]
        statistics = self._calculate_statistics(portfolio_paths, final_values)
        risk_metrics = self._calculate_risk_metrics(portfolio_paths, final_values)
        confidence_intervals = self._calculate_confidence_intervals(final_values)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return SimulationResult(
            paths=portfolio_paths,
            final_values=final_values,
            statistics=statistics,
            risk_metrics=risk_metrics,
            confidence_intervals=confidence_intervals,
            process_type=process_type,
            parameters=process_parameters,
            config=self.config,
            execution_time=execution_time
        )
    
    def _simulate_sequential(
        self,
        process_class,
        parameters: ProcessParameters,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Sequential simulation"""
        
        n_assets = len(weights)
        n_steps = self.config.time_horizon
        dt = self.config.dt
        
        # Simulate each asset separately for multivariate case
        if isinstance(parameters.mu, np.ndarray) and len(parameters.mu) > 1:
            all_paths = np.zeros((self.config.n_simulations, n_assets, n_steps + 1))
            
            for asset_idx in range(n_assets):
                asset_params = ProcessParameters(
                    mu=parameters.mu[asset_idx] if isinstance(parameters.mu, np.ndarray) else parameters.mu,
                    sigma=parameters.sigma[asset_idx] if isinstance(parameters.sigma, np.ndarray) else parameters.sigma,
                    kappa=parameters.kappa,
                    theta=parameters.theta,
                    jump_intensity=parameters.jump_intensity,
                    jump_mean=parameters.jump_mean,
                    jump_std=parameters.jump_std
                )
                
                process = process_class(asset_params)
                asset_paths = process.simulate_multiple_paths(
                    self.config.n_simulations, n_steps, dt
                )
                all_paths[:, asset_idx, :] = asset_paths
            
            return all_paths
        else:
            # Single asset or identical parameters
            process = process_class(parameters)
            paths = process.simulate_multiple_paths(
                self.config.n_simulations, n_steps, dt
            )
            return paths.reshape(self.config.n_simulations, 1, -1)
    
    def _simulate_parallel(
        self,
        process_class,
        parameters: ProcessParameters,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Parallel simulation using multiprocessing"""
        
        n_workers = self.config.n_workers
        sims_per_worker = self.config.n_simulations // n_workers
        remaining_sims = self.config.n_simulations % n_workers
        
        # Create simulation tasks
        tasks = []
        for i in range(n_workers):
            n_sims = sims_per_worker + (1 if i < remaining_sims else 0)
            seed = self.config.random_seed + i if self.config.random_seed else None
            
            tasks.append((
                process_class, parameters, weights, covariance_matrix,
                n_sims, self.config.time_horizon, self.config.dt, seed
            ))
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self._simulate_worker, task) for task in tasks]
            results = [future.result() for future in as_completed(futures)]
        
        # Combine results
        if len(results[0].shape) == 3:  # Multi-asset
            return np.vstack(results)
        else:  # Single asset
            combined = np.vstack(results)
            return combined.reshape(self.config.n_simulations, 1, -1)
    
    @staticmethod
    def _simulate_worker(task):
        """Worker function for parallel simulation"""
        (process_class, parameters, weights, covariance_matrix,
         n_sims, n_steps, dt, seed) = task
        
        random_state = np.random.RandomState(seed) if seed else None
        n_assets = len(weights)
        
        if isinstance(parameters.mu, np.ndarray) and len(parameters.mu) > 1:
            all_paths = np.zeros((n_sims, n_assets, n_steps + 1))
            
            for asset_idx in range(n_assets):
                asset_params = ProcessParameters(
                    mu=parameters.mu[asset_idx] if isinstance(parameters.mu, np.ndarray) else parameters.mu,
                    sigma=parameters.sigma[asset_idx] if isinstance(parameters.sigma, np.ndarray) else parameters.sigma,
                    kappa=parameters.kappa,
                    theta=parameters.theta,
                    jump_intensity=parameters.jump_intensity,
                    jump_mean=parameters.jump_mean,
                    jump_std=parameters.jump_std
                )
                
                process = process_class(asset_params)
                asset_paths = process.simulate_multiple_paths(n_sims, n_steps, dt, random_state)
                all_paths[:, asset_idx, :] = asset_paths
            
            return all_paths
        else:
            process = process_class(parameters)
            return process.simulate_multiple_paths(n_sims, n_steps, dt, random_state)
    
    def _calculate_portfolio_paths(
        self, 
        asset_paths: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """Calculate portfolio paths from individual asset paths"""
        
        if asset_paths.ndim == 3:  # Multi-asset case
            # asset_paths shape: (n_simulations, n_assets, n_steps + 1)
            portfolio_paths = np.zeros((asset_paths.shape[0], asset_paths.shape[2]))
            
            for sim_idx in range(asset_paths.shape[0]):
                # Calculate portfolio value at each time step
                for time_idx in range(asset_paths.shape[2]):
                    portfolio_paths[sim_idx, time_idx] = np.sum(
                        weights * asset_paths[sim_idx, :, time_idx]
                    )
        else:
            # Single asset case
            portfolio_paths = asset_paths
        
        return portfolio_paths
    
    def _calculate_statistics(
        self, 
        portfolio_paths: np.ndarray, 
        final_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive portfolio statistics"""
        
        # Returns calculation
        returns = np.diff(portfolio_paths, axis=1) / portfolio_paths[:, :-1]
        portfolio_returns = returns.mean(axis=1)
        
        statistics = {
            # Final value statistics
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'median_final_value': np.median(final_values),
            
            # Return statistics
            'mean_return': np.mean(portfolio_returns),
            'std_return': np.std(portfolio_returns),
            'annualized_return': np.mean(portfolio_returns) * 252,
            'annualized_volatility': np.std(portfolio_returns) * np.sqrt(252),
            
            # Distribution statistics
            'skewness': pd.Series(portfolio_returns).skew(),
            'kurtosis': pd.Series(portfolio_returns).kurtosis(),
            
            # Probability metrics
            'probability_of_loss': np.mean(final_values < 1.0),
            'probability_of_profit': np.mean(final_values > 1.0),
            'expected_profit': np.mean(np.maximum(final_values - 1.0, 0)),
            'expected_loss': np.mean(np.maximum(1.0 - final_values, 0))
        }
        
        return statistics
    
    def _calculate_risk_metrics(
        self, 
        portfolio_paths: np.ndarray, 
        final_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        returns = np.diff(portfolio_paths, axis=1) / portfolio_paths[:, :-1]
        portfolio_returns = returns.mean(axis=1)
        
        risk_metrics = {}
        
        # VaR and CVaR at different confidence levels
        for confidence_level in self.config.confidence_levels:
            var_value = self.risk_measures.value_at_risk(portfolio_returns, confidence_level)
            cvar_value = self.risk_measures.conditional_value_at_risk(portfolio_returns, confidence_level)
            
            level_pct = int(confidence_level * 100)
            risk_metrics[f'var_{level_pct}%'] = var_value
            risk_metrics[f'cvar_{level_pct}%'] = cvar_value
        
        # Maximum drawdown
        max_dd, _, _ = self.risk_measures.maximum_drawdown(portfolio_returns)
        risk_metrics['maximum_drawdown'] = max_dd
        
        # Other risk measures
        risk_metrics['semi_deviation'] = self.risk_measures.semi_deviation(portfolio_returns)
        risk_metrics['ulcer_index'] = self.risk_measures.ulcer_index(portfolio_returns)
        risk_metrics['pain_index'] = self.risk_measures.pain_index(portfolio_returns)
        
        # Tail measures
        risk_metrics['tail_ratio'] = self.risk_measures.tail_ratio(portfolio_returns)
        
        return risk_metrics
    
    def _calculate_confidence_intervals(
        self, 
        final_values: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        
        confidence_intervals = {}
        
        for confidence_level in self.config.confidence_levels:
            alpha = confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(final_values, lower_percentile)
            upper_bound = np.percentile(final_values, upper_percentile)
            
            level_pct = int((1 - confidence_level) * 100)
            confidence_intervals[f'{level_pct}%_ci'] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def scenario_analysis(
        self,
        weights: np.ndarray,
        base_returns: np.ndarray,
        base_covariance: np.ndarray,
        scenarios: Dict[str, Dict[str, Any]]
    ) -> Dict[str, SimulationResult]:
        """
        Run scenario analysis with different parameter sets
        
        scenarios: Dictionary with scenario name as key and parameter modifications as values
        """
        logger.info(f"Running scenario analysis with {len(scenarios)} scenarios")
        
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Running scenario: {scenario_name}")
            
            # Modify parameters for scenario
            modified_returns = base_returns * scenario_params.get('return_multiplier', 1.0)
            modified_covariance = base_covariance * scenario_params.get('volatility_multiplier', 1.0)**2
            
            process_type = scenario_params.get('process_type', StochasticProcess.GEOMETRIC_BROWNIAN_MOTION)
            
            # Create process parameters
            process_params = ProcessParameters(
                mu=modified_returns,
                sigma=np.sqrt(np.diag(modified_covariance)),
                jump_intensity=scenario_params.get('jump_intensity'),
                jump_mean=scenario_params.get('jump_mean'),
                jump_std=scenario_params.get('jump_std'),
                kappa=scenario_params.get('kappa'),
                theta=scenario_params.get('theta')
            )
            
            # Run simulation
            result = self.simulate_portfolio_paths(
                weights, modified_returns, modified_covariance, 
                process_type, process_params
            )
            
            results[scenario_name] = result
        
        return results
    
    def generate_simulation_report(
        self, 
        result: SimulationResult,
        scenario_results: Optional[Dict[str, SimulationResult]] = None
    ) -> str:
        """Generate comprehensive simulation report"""
        
        report = "MONTE CARLO SIMULATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Simulation configuration
        report += "SIMULATION CONFIGURATION\n"
        report += "-" * 25 + "\n"
        report += f"Process Type: {result.process_type.value}\n"
        report += f"Number of Simulations: {result.config.n_simulations:,}\n"
        report += f"Time Horizon: {result.config.time_horizon} days\n"
        report += f"Execution Time: {result.execution_time:.2f} seconds\n\n"
        
        # Statistical results
        report += "PORTFOLIO STATISTICS\n"
        report += "-" * 22 + "\n"
        stats = result.statistics
        report += f"Mean Final Value: {stats['mean_final_value']:.4f}\n"
        report += f"Std Final Value: {stats['std_final_value']:.4f}\n"
        report += f"Annualized Return: {stats['annualized_return']:.2%}\n"
        report += f"Annualized Volatility: {stats['annualized_volatility']:.2%}\n"
        report += f"Skewness: {stats['skewness']:.2f}\n"
        report += f"Kurtosis: {stats['kurtosis']:.2f}\n"
        report += f"Probability of Loss: {stats['probability_of_loss']:.2%}\n"
        report += f"Expected Profit: {stats['expected_profit']:.4f}\n"
        report += f"Expected Loss: {stats['expected_loss']:.4f}\n\n"
        
        # Risk metrics
        report += "RISK METRICS\n"
        report += "-" * 15 + "\n"
        risk = result.risk_metrics
        for metric, value in risk.items():
            if 'var' in metric or 'cvar' in metric:
                report += f"{metric.upper()}: {abs(value):.2%}\n"
            else:
                report += f"{metric.replace('_', ' ').title()}: {abs(value):.4f}\n"
        report += "\n"
        
        # Confidence intervals
        report += "CONFIDENCE INTERVALS (Final Values)\n"
        report += "-" * 35 + "\n"
        for level, (lower, upper) in result.confidence_intervals.items():
            report += f"{level}: [{lower:.4f}, {upper:.4f}]\n"
        report += "\n"
        
        # Scenario comparison
        if scenario_results:
            report += "SCENARIO ANALYSIS\n"
            report += "-" * 20 + "\n"
            
            for scenario_name, scenario_result in scenario_results.items():
                scenario_stats = scenario_result.statistics
                report += f"{scenario_name}:\n"
                report += f"  Mean Final Value: {scenario_stats['mean_final_value']:.4f}\n"
                report += f"  Annualized Return: {scenario_stats['annualized_return']:.2%}\n"
                report += f"  Probability of Loss: {scenario_stats['probability_of_loss']:.2%}\n"
                report += f"  95% VaR: {abs(scenario_result.risk_metrics.get('var_5%', 0)):.2%}\n\n"
        
        report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report


def create_default_scenarios() -> Dict[str, Dict[str, Any]]:
    """Create default scenario analysis configurations"""
    
    scenarios = {
        'Base Case': {
            'return_multiplier': 1.0,
            'volatility_multiplier': 1.0,
            'process_type': StochasticProcess.GEOMETRIC_BROWNIAN_MOTION
        },
        'Market Crash': {
            'return_multiplier': 0.5,
            'volatility_multiplier': 2.0,
            'process_type': StochasticProcess.JUMP_DIFFUSION,
            'jump_intensity': 0.2,
            'jump_mean': -0.15,
            'jump_std': 0.1
        },
        'High Volatility': {
            'return_multiplier': 1.0,
            'volatility_multiplier': 1.5,
            'process_type': StochasticProcess.GEOMETRIC_BROWNIAN_MOTION
        },
        'Low Volatility': {
            'return_multiplier': 1.0,
            'volatility_multiplier': 0.7,
            'process_type': StochasticProcess.GEOMETRIC_BROWNIAN_MOTION
        },
        'Mean Reversion': {
            'return_multiplier': 1.0,
            'volatility_multiplier': 1.0,
            'process_type': StochasticProcess.MEAN_REVERSION,
            'kappa': 2.0,
            'theta': 0.08
        }
    }
    
    return scenarios