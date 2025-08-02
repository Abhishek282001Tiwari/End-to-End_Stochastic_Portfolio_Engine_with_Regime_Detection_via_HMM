import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Any, Union
from scipy.special import logsumexp, digamma
from scipy.stats import multivariate_normal, norm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn import hmm
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class HMMParameters:
    """Container for HMM parameters"""
    start_prob: np.ndarray
    transition_matrix: np.ndarray
    emission_means: np.ndarray
    emission_covariances: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool


class BaseHMM(ABC):
    """Abstract base class for HMM implementations"""
    
    def __init__(self, n_components: int, random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseHMM':
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass
        
    @abstractmethod
    def score(self, X: np.ndarray) -> float:
        pass


class AdvancedBaumWelchHMM(BaseHMM):
    """Advanced HMM implementation with proper Baum-Welch algorithm and numerical stability"""
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-6,
        min_covar: float = 1e-3,
        random_state: Optional[int] = None
    ):
        super().__init__(n_components, random_state)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        
        # HMM parameters
        self.start_prob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None
        
        # Training history
        self.log_likelihood_history_ = []
        self.converged_ = False
        
    def _initialize_parameters(self, X: np.ndarray) -> None:
        """Initialize HMM parameters using K-means clustering"""
        n_samples, n_features = X.shape
        
        # Initialize using K-means for better starting points
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Initialize start probabilities
        self.start_prob_ = np.ones(self.n_components) / self.n_components
        
        # Initialize transition matrix with persistence bias
        self.transmat_ = np.full((self.n_components, self.n_components), 0.1 / (self.n_components - 1))
        np.fill_diagonal(self.transmat_, 0.9)
        
        # Initialize emission parameters from K-means results
        self.means_ = kmeans.cluster_centers_
        
        # Initialize covariances
        if self.covariance_type == "full":
            self.covars_ = np.zeros((self.n_components, n_features, n_features))
            for i in range(self.n_components):
                mask = labels == i
                if np.sum(mask) > 1:
                    self.covars_[i] = np.cov(X[mask].T) + np.eye(n_features) * self.min_covar
                else:
                    self.covars_[i] = np.eye(n_features) * self.min_covar
        elif self.covariance_type == "diag":
            self.covars_ = np.zeros((self.n_components, n_features))
            for i in range(self.n_components):
                mask = labels == i
                if np.sum(mask) > 1:
                    self.covars_[i] = np.var(X[mask], axis=0) + self.min_covar
                else:
                    self.covars_[i] = np.ones(n_features) * self.min_covar
        elif self.covariance_type == "spherical":
            self.covars_ = np.zeros(self.n_components)
            for i in range(self.n_components):
                mask = labels == i
                if np.sum(mask) > 1:
                    self.covars_[i] = np.var(X[mask]) + self.min_covar
                else:
                    self.covars_[i] = self.min_covar
                    
    def _compute_log_emission_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities with numerical stability"""
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            if self.covariance_type == "full":
                log_prob[:, i] = multivariate_normal.logpdf(X, self.means_[i], self.covars_[i])
            elif self.covariance_type == "diag":
                log_prob[:, i] = np.sum(
                    norm.logpdf(X, self.means_[i], np.sqrt(self.covars_[i])), axis=1
                )
            elif self.covariance_type == "spherical":
                diff = X - self.means_[i]
                log_prob[:, i] = (
                    -0.5 * n_features * np.log(2 * np.pi * self.covars_[i])
                    - 0.5 * np.sum(diff**2, axis=1) / self.covars_[i]
                )
                
        # Handle numerical issues
        log_prob = np.clip(log_prob, -700, 700)
        return log_prob
        
    def _forward_algorithm(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm with numerical stability"""
        n_samples = X.shape[0]
        log_emission_prob = self._compute_log_emission_probabilities(X)
        
        # Initialize forward variables
        log_alpha = np.zeros((n_samples, self.n_components))
        
        # Forward pass
        log_alpha[0] = np.log(self.start_prob_) + log_emission_prob[0]
        
        for t in range(1, n_samples):
            for j in range(self.n_components):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t-1] + np.log(self.transmat_[:, j]))
                    + log_emission_prob[t, j]
                )
        
        # Compute log likelihood
        log_likelihood = logsumexp(log_alpha[-1])
        
        return log_alpha, log_likelihood
        
    def _backward_algorithm(self, X: np.ndarray) -> np.ndarray:
        """Backward algorithm with numerical stability"""
        n_samples = X.shape[0]
        log_emission_prob = self._compute_log_emission_probabilities(X)
        
        # Initialize backward variables
        log_beta = np.zeros((n_samples, self.n_components))
        log_beta[-1] = 0  # log(1) = 0
        
        # Backward pass
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_components):
                log_beta[t, i] = logsumexp(
                    np.log(self.transmat_[i, :]) + log_emission_prob[t+1] + log_beta[t+1]
                )
        
        return log_beta
        
    def _compute_posteriors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute posterior probabilities (E-step)"""
        n_samples = X.shape[0]
        
        # Forward-backward algorithm
        log_alpha, log_likelihood = self._forward_algorithm(X)
        log_beta = self._backward_algorithm(X)
        
        # Compute gamma (state posteriors)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        
        # Compute xi (transition posteriors)
        log_emission_prob = self._compute_log_emission_probabilities(X)
        xi = np.zeros((n_samples - 1, self.n_components, self.n_components))
        
        for t in range(n_samples - 1):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    xi[t, i, j] = (
                        log_alpha[t, i] + np.log(self.transmat_[i, j]) 
                        + log_emission_prob[t+1, j] + log_beta[t+1, j]
                    )
            xi[t] = np.exp(xi[t] - logsumexp(xi[t]))
        
        return gamma, xi, log_likelihood
        
    def _update_parameters(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """Update HMM parameters (M-step)"""
        n_samples, n_features = X.shape
        
        # Update start probabilities
        self.start_prob_ = gamma[0] + 1e-8
        self.start_prob_ /= self.start_prob_.sum()
        
        # Update transition matrix
        self.transmat_ = xi.sum(axis=0) + 1e-8
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
        
        # Update emission parameters
        gamma_sum = gamma.sum(axis=0) + 1e-8
        
        # Update means
        for i in range(self.n_components):
            self.means_[i] = np.average(X, axis=0, weights=gamma[:, i])
        
        # Update covariances
        if self.covariance_type == "full":
            for i in range(self.n_components):
                diff = X - self.means_[i]
                self.covars_[i] = np.average(
                    np.array([np.outer(d, d) for d in diff]),
                    axis=0, weights=gamma[:, i]
                ) + np.eye(n_features) * self.min_covar
        elif self.covariance_type == "diag":
            for i in range(self.n_components):
                diff = X - self.means_[i]
                self.covars_[i] = np.average(diff**2, axis=0, weights=gamma[:, i]) + self.min_covar
        elif self.covariance_type == "spherical":
            for i in range(self.n_components):
                diff = X - self.means_[i]
                self.covars_[i] = np.average(np.sum(diff**2, axis=1), weights=gamma[:, i]) / n_features + self.min_covar
    
    def fit(self, X: np.ndarray) -> 'AdvancedBaumWelchHMM':
        """Fit HMM using Baum-Welch algorithm"""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X = X.astype(np.float64)
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        self.log_likelihood_history_ = []
        
        for iteration in range(self.n_iter):
            # E-step: compute posteriors
            gamma, xi, log_likelihood = self._compute_posteriors(X)
            
            # M-step: update parameters
            self._update_parameters(X, gamma, xi)
            
            self.log_likelihood_history_.append(log_likelihood)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                logger.info(f"Baum-Welch converged after {iteration + 1} iterations")
                break
                
            prev_log_likelihood = log_likelihood
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: log_likelihood = {log_likelihood:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence using Viterbi algorithm"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self._viterbi_algorithm(X)[1]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict state probabilities using forward-backward algorithm"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        gamma, _, _ = self._compute_posteriors(X)
        return gamma
    
    def score(self, X: np.ndarray) -> float:
        """Compute log likelihood of data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        _, log_likelihood = self._forward_algorithm(X)
        return log_likelihood
    
    def _viterbi_algorithm(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """Viterbi algorithm for finding most likely state sequence"""
        n_samples = X.shape[0]
        log_emission_prob = self._compute_log_emission_probabilities(X)
        
        # Initialize Viterbi variables
        log_delta = np.zeros((n_samples, self.n_components))
        psi = np.zeros((n_samples, self.n_components), dtype=int)
        
        # Initialization
        log_delta[0] = np.log(self.start_prob_) + log_emission_prob[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_components):
                transition_scores = log_delta[t-1] + np.log(self.transmat_[:, j])
                psi[t, j] = np.argmax(transition_scores)
                log_delta[t, j] = np.max(transition_scores) + log_emission_prob[t, j]
        
        # Backward pass (traceback)
        path = np.zeros(n_samples, dtype=int)
        path[-1] = np.argmax(log_delta[-1])
        
        for t in range(n_samples - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        
        log_prob = np.max(log_delta[-1])
        return log_prob, path
        
    def get_parameters(self) -> HMMParameters:
        """Get HMM parameters"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        return HMMParameters(
            start_prob=self.start_prob_.copy(),
            transition_matrix=self.transmat_.copy(),
            emission_means=self.means_.copy(),
            emission_covariances=self.covars_.copy(),
            log_likelihood=self.log_likelihood_history_[-1] if self.log_likelihood_history_ else 0.0,
            n_iter=len(self.log_likelihood_history_),
            converged=self.converged_
        )


class RegimeDetectionHMM(AdvancedBaumWelchHMM):
    """Enhanced regime detection HMM with market-specific features"""
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        regime_names: Optional[Dict[int, str]] = None
    ):
        super().__init__(n_components, covariance_type, n_iter, tol, random_state=random_state)
        
        self.feature_names = None
        self.regime_names = regime_names or {
            0: "Bull Market",
            1: "Bear Market", 
            2: "Sideways Market"
        }
        
        # Regime characteristics
        self.regime_characteristics_ = None
        self.regime_durations_ = None
        self.confidence_intervals_ = None
    
    def fit(self, X: pd.DataFrame) -> 'RegimeDetectionHMM':
        """Fit the regime detection HMM"""
        logger.info(f"Training Regime Detection HMM with {self.n_components} components")
        
        self.feature_names = list(X.columns)
        X_array = X.values.astype(np.float64)
        
        # Use parent class fit method
        super().fit(X_array)
        
        # Interpret regimes based on market characteristics
        self._interpret_regimes(X_array)
        self._analyze_regime_characteristics(X_array)
        
        logger.info("Regime Detection HMM training completed successfully")
        return self
    
    def _interpret_regimes(self, X: np.ndarray):
        """Interpret regimes based on market return characteristics"""
        if self.feature_names and "market_return" in self.feature_names:
            return_idx = self.feature_names.index("market_return")
            return_means = self.means_[:, return_idx]
            
            # Sort regimes by return characteristics
            sorted_indices = np.argsort(return_means)
            
            # Assign regime names based on returns
            if self.n_components == 3:
                self.regime_names = {
                    sorted_indices[0]: "Bear Market",
                    sorted_indices[1]: "Sideways Market", 
                    sorted_indices[2]: "Bull Market"
                }
            elif self.n_components == 4:
                self.regime_names = {
                    sorted_indices[0]: "Bear Market",
                    sorted_indices[1]: "Low Volatility",
                    sorted_indices[2]: "High Volatility", 
                    sorted_indices[3]: "Bull Market"
                }
        
        logger.info(f"Regime interpretation: {self.regime_names}")
    
    def _analyze_regime_characteristics(self, X: np.ndarray):
        """Analyze characteristics of detected regimes"""
        self.regime_characteristics_ = {}
        
        for regime in range(self.n_components):
            characteristics = {
                'mean_features': self.means_[regime],
                'covariance': self.covars_[regime] if self.covariance_type == "full" else None,
                'persistence_probability': self.transmat_[regime, regime],
                'transition_probabilities': self.transmat_[regime, :],
                'typical_duration': 1 / (1 - self.transmat_[regime, regime])
            }
            
            # Add feature-specific characteristics
            if self.feature_names:
                feature_dict = {}
                for i, feature in enumerate(self.feature_names):
                    feature_dict[feature] = {
                        'mean': self.means_[regime, i],
                        'std': np.sqrt(self.covars_[regime, i, i]) if self.covariance_type == "full" else np.sqrt(self.covars_[regime, i] if self.covariance_type == "diag" else self.covars_[regime])
                    }
                characteristics['features'] = feature_dict
            
            self.regime_characteristics_[regime] = characteristics
    
    def predict_regimes(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime sequence"""
        return self.predict(X)
    
    def predict_regime_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities"""
        return self.predict_proba(X)
    
    def decode_most_likely_sequence(self, X: pd.DataFrame) -> Tuple[float, np.ndarray]:
        """Decode most likely regime sequence using Viterbi"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
        
        X_array = X.values.astype(np.float64)
        return self._viterbi_algorithm(X_array)
    
    def get_regime_statistics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive regime statistics"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting statistics")
        
        states = self.predict_regimes(X)
        regime_probs = self.predict_regime_probabilities(X)
        
        statistics = {
            "regime_distribution": np.bincount(states, minlength=self.n_components) / len(states),
            "average_regime_duration": self._calculate_average_duration(states),
            "transition_probabilities": self.transmat_,
            "regime_means": self.means_,
            "regime_covariances": self.covars_,
            "regime_names": self.regime_names,
            "regime_characteristics": self.regime_characteristics_,
            "confidence_intervals": self._calculate_confidence_intervals(regime_probs)
        }
        
        return statistics
    
    def _calculate_average_duration(self, states: np.ndarray) -> Dict[int, float]:
        """Calculate average duration of each regime"""
        durations = {}
        
        for regime in range(self.n_components):
            regime_positions = np.where(states == regime)[0]
            
            if len(regime_positions) == 0:
                durations[regime] = 0
                continue
            
            # Find consecutive runs
            regime_durations = []
            current_duration = 1
            
            for i in range(1, len(regime_positions)):
                if regime_positions[i] == regime_positions[i-1] + 1:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_duration = 1
            
            regime_durations.append(current_duration)
            durations[regime] = np.mean(regime_durations) if regime_durations else 0
        
        return durations
    
    def _calculate_confidence_intervals(self, regime_probs: np.ndarray, confidence: float = 0.95) -> Dict[int, Dict[str, float]]:
        """Calculate confidence intervals for regime probabilities"""
        confidence_intervals = {}
        alpha = 1 - confidence
        
        for regime in range(self.n_components):
            probs = regime_probs[:, regime]
            lower = np.percentile(probs, 100 * alpha / 2)
            upper = np.percentile(probs, 100 * (1 - alpha / 2))
            mean_prob = np.mean(probs)
            
            confidence_intervals[regime] = {
                'mean': mean_prob,
                'lower': lower,
                'upper': upper,
                'confidence_level': confidence
            }
        
        return confidence_intervals
    
    def get_regime_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get detailed regime summary"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        states = self.predict_regimes(X)
        probabilities = self.predict_regime_probabilities(X)
        
        summary_data = []
        
        for i, date in enumerate(X.index):
            regime = states[i]
            regime_name = self.regime_names[regime]
            confidence = probabilities[i, regime]
            
            # Add regime probabilities for all states
            regime_probs = {f"prob_{self.regime_names[j]}": probabilities[i, j] 
                           for j in range(self.n_components)}
            
            summary_data.append({
                "date": date,
                "regime": regime,
                "regime_name": regime_name,
                "confidence": confidence,
                **regime_probs
            })
        
        return pd.DataFrame(summary_data).set_index("date")


class MixtureHMM(BaseHMM):
    """HMM with Gaussian Mixture emission distributions"""
    
    def __init__(
        self,
        n_components: int = 3,
        n_mix: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-6,
        random_state: Optional[int] = None
    ):
        super().__init__(n_components, random_state)
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        
        # Use hmmlearn's GMMHMM
        from hmmlearn.hmm import GMMHMM
        self.model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray) -> 'MixtureHMM':
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray) -> float:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.score(X)


class RegimeSwitchingModel(BaseHMM):
    """Regime-switching model with economic interpretation"""
    
    def __init__(
        self,
        n_regimes: int = 3,
        switching_variance: bool = True,
        switching_mean: bool = True,
        random_state: Optional[int] = None
    ):
        super().__init__(n_regimes, random_state)
        self.switching_variance = switching_variance
        self.switching_mean = switching_mean
        
        # Parameters for regime switching
        self.regime_means_ = None
        self.regime_vars_ = None
        self.transition_probs_ = None
        self.regime_probs_ = None
        
    def fit(self, returns: np.ndarray) -> 'RegimeSwitchingModel':
        """Fit regime switching model to return series"""
        if isinstance(returns, pd.DataFrame):
            returns = returns.values.flatten()
        
        # Initialize parameters
        self._initialize_regime_parameters(returns)
        
        # EM algorithm for regime switching
        log_likelihood = self._em_algorithm(returns)
        
        self.is_fitted = True
        logger.info(f"Regime switching model converged with log-likelihood: {log_likelihood:.4f}")
        return self
    
    def _initialize_regime_parameters(self, returns: np.ndarray):
        """Initialize regime parameters using K-means"""
        from sklearn.cluster import KMeans
        
        # Reshape for K-means
        X = returns.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        
        self.regime_means_ = np.zeros(self.n_components)
        self.regime_vars_ = np.zeros(self.n_components)
        
        for i in range(self.n_components):
            mask = labels == i
            self.regime_means_[i] = np.mean(returns[mask]) if np.any(mask) else 0
            self.regime_vars_[i] = np.var(returns[mask]) if np.any(mask) else 1
        
        # Initialize transition probabilities
        self.transition_probs_ = np.full((self.n_components, self.n_components), 0.1)
        np.fill_diagonal(self.transition_probs_, 0.8)
        self.transition_probs_ = self.transition_probs_ / self.transition_probs_.sum(axis=1, keepdims=True)
    
    def _em_algorithm(self, returns: np.ndarray, max_iter: int = 100) -> float:
        """EM algorithm for regime switching model"""
        n_obs = len(returns)
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            # E-step: compute regime probabilities
            regime_probs = self._compute_regime_probabilities(returns)
            
            # M-step: update parameters
            self._update_parameters(returns, regime_probs)
            
            # Compute log-likelihood
            ll = self._compute_log_likelihood(returns)
            
            if abs(ll - prev_ll) < 1e-6:
                break
            prev_ll = ll
        
        self.regime_probs_ = regime_probs
        return ll
    
    def _compute_regime_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """Compute filtered regime probabilities"""
        n_obs = len(returns)
        regime_probs = np.zeros((n_obs, self.n_components))
        
        # Initialize
        for i in range(self.n_components):
            regime_probs[0, i] = self._gaussian_density(returns[0], self.regime_means_[i], self.regime_vars_[i])
        regime_probs[0] /= regime_probs[0].sum()
        
        # Forward filtering
        for t in range(1, n_obs):
            for j in range(self.n_components):
                # Prediction step
                pred_prob = np.sum(regime_probs[t-1] * self.transition_probs_[:, j])
                
                # Update step
                likelihood = self._gaussian_density(returns[t], self.regime_means_[j], self.regime_vars_[j])
                regime_probs[t, j] = pred_prob * likelihood
            
            # Normalize
            regime_probs[t] /= regime_probs[t].sum()
        
        return regime_probs
    
    def _gaussian_density(self, x: float, mean: float, var: float) -> float:
        """Gaussian probability density function"""
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mean)**2 / var)
    
    def _update_parameters(self, returns: np.ndarray, regime_probs: np.ndarray):
        """Update model parameters"""
        n_obs = len(returns)
        
        # Update regime means and variances
        for i in range(self.n_components):
            weights = regime_probs[:, i]
            weight_sum = weights.sum()
            
            if weight_sum > 1e-8:
                if self.switching_mean:
                    self.regime_means_[i] = np.sum(weights * returns) / weight_sum
                
                if self.switching_variance:
                    residuals = returns - self.regime_means_[i]
                    self.regime_vars_[i] = np.sum(weights * residuals**2) / weight_sum
        
        # Update transition probabilities
        for i in range(self.n_components):
            for j in range(self.n_components):
                numerator = np.sum(regime_probs[:-1, i] * regime_probs[1:, j])
                denominator = np.sum(regime_probs[:-1, i])
                self.transition_probs_[i, j] = numerator / denominator if denominator > 1e-8 else 0
        
        # Normalize transition probabilities
        self.transition_probs_ = self.transition_probs_ / self.transition_probs_.sum(axis=1, keepdims=True)
    
    def _compute_log_likelihood(self, returns: np.ndarray) -> float:
        """Compute log-likelihood of the model"""
        log_likelihood = 0
        regime_probs = self._compute_regime_probabilities(returns)
        
        for t in range(len(returns)):
            likelihood = 0
            for i in range(self.n_components):
                likelihood += regime_probs[t, i] * self._gaussian_density(
                    returns[t], self.regime_means_[i], self.regime_vars_[i]
                )
            log_likelihood += np.log(likelihood + 1e-8)
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely regime sequence"""
        if isinstance(X, pd.DataFrame):
            X = X.values.flatten()
        
        regime_probs = self._compute_regime_probabilities(X)
        return np.argmax(regime_probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities"""
        if isinstance(X, pd.DataFrame):
            X = X.values.flatten()
        
        return self._compute_regime_probabilities(X)
    
    def score(self, X: np.ndarray) -> float:
        """Compute log-likelihood"""
        if isinstance(X, pd.DataFrame):
            X = X.values.flatten()
        
        return self._compute_log_likelihood(X)


class RollingWindowHMM:
    """HMM with rolling window estimation for adaptive regime detection"""
    
    def __init__(
        self,
        base_hmm: BaseHMM,
        window_size: int = 252,
        step_size: int = 21,
        min_window_size: int = 126
    ):
        self.base_hmm = base_hmm
        self.window_size = window_size
        self.step_size = step_size
        self.min_window_size = min_window_size
        
        self.models_history_ = []
        self.predictions_history_ = []
        self.probabilities_history_ = []
        
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit HMM on rolling windows and predict"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            dates = X.index
        else:
            X_array = X
            dates = pd.date_range('2000-01-01', periods=len(X), freq='D')
        
        n_samples = len(X_array)
        predictions = np.full(n_samples, -1)
        probabilities = np.zeros((n_samples, self.base_hmm.n_components))
        
        for start_idx in range(0, n_samples - self.min_window_size + 1, self.step_size):
            end_idx = min(start_idx + self.window_size, n_samples)
            
            if end_idx - start_idx < self.min_window_size:
                break
            
            # Extract window
            window_data = X_array[start_idx:end_idx]
            
            # Create and fit model for this window
            try:
                window_model = AdvancedBaumWelchHMM(
                    n_components=self.base_hmm.n_components,
                    random_state=self.base_hmm.random_state
                )
                window_model.fit(window_data)
                
                # Predict on the window
                window_predictions = window_model.predict(window_data)
                window_probabilities = window_model.predict_proba(window_data)
                
                # Store results
                pred_start = max(start_idx, end_idx - self.step_size)
                pred_end = end_idx
                
                predictions[pred_start:pred_end] = window_predictions[pred_start-start_idx:pred_end-start_idx]
                probabilities[pred_start:pred_end] = window_probabilities[pred_start-start_idx:pred_end-start_idx]
                
                # Store model
                self.models_history_.append({
                    'model': window_model,
                    'start_date': dates[start_idx],
                    'end_date': dates[end_idx-1],
                    'window_size': end_idx - start_idx
                })
                
            except Exception as e:
                logger.warning(f"Failed to fit model for window {start_idx}:{end_idx}: {e}")
                continue
        
        return predictions, probabilities
    
    def get_model_evolution(self) -> pd.DataFrame:
        """Get evolution of model parameters over time"""
        evolution_data = []
        
        for i, model_info in enumerate(self.models_history_):
            model = model_info['model']
            
            if hasattr(model, 'means_'):
                for regime in range(model.n_components):
                    evolution_data.append({
                        'window_idx': i,
                        'start_date': model_info['start_date'],
                        'end_date': model_info['end_date'],
                        'regime': regime,
                        'mean_return': model.means_[regime, 0] if model.means_.ndim > 1 else model.means_[regime],
                        'persistence_prob': model.transmat_[regime, regime] if hasattr(model, 'transmat_') else None
                    })
        
        return pd.DataFrame(evolution_data)


class OnlineHMMUpdater:
    def __init__(self, base_model: RegimeDetectionHMM, update_window: int = 50):
        self.base_model = base_model
        self.update_window = update_window
        self.data_buffer = []
        
    def update_with_new_data(self, new_data: pd.DataFrame) -> bool:
        logger.info("Updating HMM with new data")
        
        self.data_buffer.append(new_data)
        
        if len(self.data_buffer) >= self.update_window:
            combined_data = pd.concat(self.data_buffer[-self.update_window:])
            
            try:
                self.base_model.fit(combined_data)
                logger.info("HMM model updated successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error updating HMM model: {e}")
                return False
        
        return False


class EnsembleRegimeDetector:
    def __init__(self, models: List[RegimeDetectionHMM]):
        self.models = models
        self.model_weights = np.ones(len(models)) / len(models)
        
    def fit(self, X: pd.DataFrame):
        logger.info(f"Training ensemble of {len(self.models)} HMM models")
        
        for i, model in enumerate(self.models):
            try:
                model.fit(X)
                logger.info(f"Model {i+1}/{len(self.models)} trained successfully")
            except Exception as e:
                logger.error(f"Error training model {i+1}: {e}")
                self.model_weights[i] = 0
        
        self.model_weights = self.model_weights / self.model_weights.sum()
        
    def predict_regime_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        all_probabilities = []
        
        for model, weight in zip(self.models, self.model_weights):
            if weight > 0 and model.is_fitted:
                probabilities = model.predict_regime_probabilities(X)
                all_probabilities.append(probabilities * weight)
        
        if not all_probabilities:
            raise ValueError("No trained models available for prediction")
        
        ensemble_probabilities = np.sum(all_probabilities, axis=0)
        return ensemble_probabilities
    
    def predict_regimes(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_regime_probabilities(X)
        return np.argmax(probabilities, axis=1)