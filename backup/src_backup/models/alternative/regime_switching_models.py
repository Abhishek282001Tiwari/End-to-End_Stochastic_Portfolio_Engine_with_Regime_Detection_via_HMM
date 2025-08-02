import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
import warnings
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import numpy.random as npr

from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class RegimeSwitchingGARCH:
    """Regime-switching GARCH model for volatility forecasting"""
    
    def __init__(self, n_regimes: int = 2, garch_order: Tuple[int, int] = (1, 1)):
        self.n_regimes = n_regimes
        self.garch_order = garch_order
        self.models = {}
        self.transition_matrix = None
        self.regime_probabilities = None
        self.fitted_params = {}
        self.is_fitted = False
        
    def fit(self, returns: pd.Series, max_iter: int = 100) -> 'RegimeSwitchingGARCH':
        """Fit regime-switching GARCH model using EM algorithm"""
        logger.info(f"Fitting regime-switching GARCH with {self.n_regimes} regimes")
        
        returns_array = returns.dropna().values
        T = len(returns_array)
        
        # Initialize parameters
        self._initialize_parameters(returns_array)
        
        log_likelihood_old = -np.inf
        
        for iteration in range(max_iter):
            # E-step: Calculate regime probabilities
            self._e_step(returns_array)
            
            # M-step: Update parameters
            self._m_step(returns_array)
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(returns_array)
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < 1e-6:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            log_likelihood_old = log_likelihood
        
        self.is_fitted = True
        return self
    
    def _initialize_parameters(self, returns: np.ndarray):
        """Initialize model parameters"""
        T = len(returns)
        
        # Initialize transition matrix
        self.transition_matrix = np.full((self.n_regimes, self.n_regimes), 
                                       1.0 / self.n_regimes)
        np.fill_diagonal(self.transition_matrix, 0.9)
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize regime probabilities
        self.regime_probabilities = np.full((T, self.n_regimes), 1.0 / self.n_regimes)
        
        # Initialize GARCH parameters for each regime
        overall_var = np.var(returns)
        
        for regime in range(self.n_regimes):
            # Different volatility levels for different regimes
            if regime == 0:  # Low volatility regime
                base_var = overall_var * 0.5
            elif regime == 1:  # High volatility regime
                base_var = overall_var * 2.0
            else:
                base_var = overall_var
            
            self.fitted_params[regime] = {
                'omega': base_var * 0.1,
                'alpha': 0.1,
                'beta': 0.8,
                'mu': np.mean(returns)
            }
    
    def _e_step(self, returns: np.ndarray):
        """E-step: Calculate regime probabilities using forward-backward algorithm"""
        T = len(returns)
        
        # Forward probabilities
        alpha = np.zeros((T, self.n_regimes))
        
        # Initialize
        for regime in range(self.n_regimes):
            alpha[0, regime] = (1.0 / self.n_regimes) * self._regime_likelihood(returns[0], regime, 0)
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_regimes):
                alpha[t, j] = np.sum([alpha[t-1, i] * self.transition_matrix[i, j] 
                                    for i in range(self.n_regimes)]) * \
                             self._regime_likelihood(returns[t], j, t)
        
        # Backward probabilities
        beta = np.zeros((T, self.n_regimes))
        beta[T-1, :] = 1.0
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(self.n_regimes):
                beta[t, i] = np.sum([self.transition_matrix[i, j] * 
                                   self._regime_likelihood(returns[t+1], j, t+1) * 
                                   beta[t+1, j] for j in range(self.n_regimes)])
        
        # Calculate regime probabilities
        for t in range(T):
            normalizer = np.sum(alpha[t, :] * beta[t, :])
            if normalizer > 0:
                self.regime_probabilities[t, :] = (alpha[t, :] * beta[t, :]) / normalizer
            else:
                self.regime_probabilities[t, :] = 1.0 / self.n_regimes
    
    def _regime_likelihood(self, return_t: float, regime: int, t: int) -> float:
        """Calculate likelihood of return given regime"""
        params = self.fitted_params[regime]
        
        # Calculate conditional variance using GARCH
        if t == 0:
            h_t = params['omega'] / (1 - params['alpha'] - params['beta'])
        else:
            # This is simplified - in practice, you'd track variance over time
            h_t = params['omega'] + params['alpha'] * return_t**2 + params['beta'] * params.get('h_prev', params['omega'])
        
        # Store for next iteration
        params['h_prev'] = h_t
        
        # Normal likelihood
        return norm.pdf(return_t, loc=params['mu'], scale=np.sqrt(h_t))
    
    def _m_step(self, returns: np.ndarray):
        """M-step: Update parameters"""
        T = len(returns)
        
        # Update transition matrix
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = np.sum([self.regime_probabilities[t, i] * 
                                  self.regime_probabilities[t+1, j] 
                                  for t in range(T-1)])
                denominator = np.sum(self.regime_probabilities[:-1, i])
                
                if denominator > 0:
                    self.transition_matrix[i, j] = numerator / denominator
                else:
                    self.transition_matrix[i, j] = 1.0 / self.n_regimes
        
        # Normalize transition matrix
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Update GARCH parameters for each regime
        for regime in range(self.n_regimes):
            weights = self.regime_probabilities[:, regime]
            
            # Weighted mean
            weighted_mean = np.sum(weights * returns) / np.sum(weights)
            self.fitted_params[regime]['mu'] = weighted_mean
            
            # Update GARCH parameters using weighted maximum likelihood
            self._update_garch_params(returns, weights, regime)
    
    def _update_garch_params(self, returns: np.ndarray, weights: np.ndarray, regime: int):
        """Update GARCH parameters for a specific regime"""
        
        def garch_likelihood(params):
            omega, alpha, beta = params
            
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return np.inf
            
            T = len(returns)
            h = np.zeros(T)
            h[0] = omega / (1 - alpha - beta)
            
            log_likelihood = 0
            for t in range(T):
                if t > 0:
                    h[t] = omega + alpha * (returns[t-1] - self.fitted_params[regime]['mu'])**2 + beta * h[t-1]
                
                if h[t] <= 0:
                    return np.inf
                
                log_likelihood += weights[t] * (-0.5 * np.log(2 * np.pi * h[t]) - 
                                              0.5 * (returns[t] - self.fitted_params[regime]['mu'])**2 / h[t])
            
            return -log_likelihood
        
        # Optimize GARCH parameters
        initial_params = [self.fitted_params[regime]['omega'], 
                         self.fitted_params[regime]['alpha'], 
                         self.fitted_params[regime]['beta']]
        
        bounds = [(1e-6, 1), (0, 0.99), (0, 0.99)]
        
        result = minimize(garch_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.fitted_params[regime]['omega'] = result.x[0]
            self.fitted_params[regime]['alpha'] = result.x[1]
            self.fitted_params[regime]['beta'] = result.x[2]
    
    def _calculate_log_likelihood(self, returns: np.ndarray) -> float:
        """Calculate total log-likelihood"""
        T = len(returns)
        log_likelihood = 0
        
        for t in range(T):
            likelihood_t = 0
            for regime in range(self.n_regimes):
                likelihood_t += self.regime_probabilities[t, regime] * \
                               self._regime_likelihood(returns[t], regime, t)
            
            if likelihood_t > 0:
                log_likelihood += np.log(likelihood_t)
        
        return log_likelihood
    
    def predict_volatility(self, horizon: int = 1) -> Dict[str, np.ndarray]:
        """Predict volatility for each regime"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        volatility_forecasts = {}
        
        for regime in range(self.n_regimes):
            params = self.fitted_params[regime]
            
            # Long-run variance
            long_run_var = params['omega'] / (1 - params['alpha'] - params['beta'])
            
            # For simplicity, assume constant volatility forecast
            volatility_forecasts[f'regime_{regime}'] = np.full(horizon, np.sqrt(long_run_var))
        
        return volatility_forecasts
    
    def get_regime_classification(self, returns: pd.Series) -> pd.DataFrame:
        """Get regime classification for the returns"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before classification")
        
        # Get most likely regime for each time point
        most_likely_regimes = np.argmax(self.regime_probabilities, axis=1)
        
        results = pd.DataFrame({
            'regime': most_likely_regimes,
            **{f'prob_regime_{i}': self.regime_probabilities[:, i] 
               for i in range(self.n_regimes)}
        }, index=returns.index[:len(most_likely_regimes)])
        
        return results


class JumpDiffusionModel:
    """Jump-diffusion model for extreme market movements"""
    
    def __init__(self):
        self.params = {}
        self.is_fitted = False
        
    def fit(self, returns: pd.Series) -> 'JumpDiffusionModel':
        """Fit jump-diffusion model using maximum likelihood"""
        logger.info("Fitting jump-diffusion model")
        
        returns_array = returns.dropna().values
        
        def log_likelihood(params):
            mu, sigma, lambda_jump, mu_jump, sigma_jump = params
            
            if sigma <= 0 or sigma_jump <= 0 or lambda_jump < 0:
                return -np.inf
            
            dt = 1/252  # Daily data
            
            total_ll = 0
            for ret in returns_array:
                # Probability of no jump
                prob_no_jump = np.exp(-lambda_jump * dt) * \
                              norm.pdf(ret, loc=mu * dt, scale=sigma * np.sqrt(dt))
                
                # Probability of jump (simplified to single jump)
                prob_jump = lambda_jump * dt * np.exp(-lambda_jump * dt) * \
                           norm.pdf(ret, loc=mu * dt + mu_jump, 
                                  scale=np.sqrt(sigma**2 * dt + sigma_jump**2))
                
                total_prob = prob_no_jump + prob_jump
                
                if total_prob > 0:
                    total_ll += np.log(total_prob)
                else:
                    return -np.inf
            
            return total_ll
        
        # Initial parameters
        initial_params = [
            np.mean(returns_array),  # mu
            np.std(returns_array),   # sigma
            0.1,                     # lambda_jump
            -0.02,                   # mu_jump (negative for market crashes)
            0.05                     # sigma_jump
        ]
        
        # Optimize
        result = minimize(
            lambda p: -log_likelihood(p), 
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.params = {
                'mu': result.x[0],
                'sigma': result.x[1],
                'lambda_jump': result.x[2],
                'mu_jump': result.x[3],
                'sigma_jump': result.x[4]
            }
            self.is_fitted = True
            
            logger.info(f"Jump-diffusion model fitted successfully")
            logger.info(f"Drift: {self.params['mu']:.4f}")
            logger.info(f"Volatility: {self.params['sigma']:.4f}")
            logger.info(f"Jump intensity: {self.params['lambda_jump']:.4f}")
            logger.info(f"Jump mean: {self.params['mu_jump']:.4f}")
            logger.info(f"Jump volatility: {self.params['sigma_jump']:.4f}")
        else:
            logger.error("Jump-diffusion model fitting failed")
            raise ValueError("Optimization failed")
        
        return self
    
    def simulate_paths(
        self, 
        S0: float, 
        T: float, 
        n_steps: int, 
        n_paths: int
    ) -> np.ndarray:
        """Simulate price paths using jump-diffusion model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
        
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in range(n_paths):
            for t in range(n_steps):
                # Normal diffusion component
                dW = npr.normal(0, np.sqrt(dt))
                
                # Jump component
                jump_occurs = npr.poisson(self.params['lambda_jump'] * dt)
                jump_size = 0
                
                if jump_occurs > 0:
                    jump_size = np.sum(npr.normal(
                        self.params['mu_jump'], 
                        self.params['sigma_jump'], 
                        jump_occurs
                    ))
                
                # Update price
                paths[i, t+1] = paths[i, t] * np.exp(
                    (self.params['mu'] - 0.5 * self.params['sigma']**2) * dt + 
                    self.params['sigma'] * dW + 
                    jump_size
                )
        
        return paths
    
    def calculate_risk_metrics(self, S0: float, T: float, confidence_levels: List[float] = [0.01, 0.05]) -> Dict[str, float]:
        """Calculate risk metrics using the jump-diffusion model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before risk calculation")
        
        # Simulate many paths for risk calculation
        n_simulations = 10000
        n_steps = int(T * 252)  # Daily steps
        
        paths = self.simulate_paths(S0, T, n_steps, n_simulations)
        
        # Calculate returns
        final_prices = paths[:, -1]
        returns = (final_prices / S0) - 1
        
        risk_metrics = {}
        
        for conf_level in confidence_levels:
            var = np.percentile(returns, conf_level * 100)
            cvar = np.mean(returns[returns <= var])
            
            risk_metrics[f'var_{int(conf_level*100)}'] = var
            risk_metrics[f'cvar_{int(conf_level*100)}'] = cvar
        
        risk_metrics['expected_return'] = np.mean(returns)
        risk_metrics['volatility'] = np.std(returns)
        risk_metrics['skewness'] = pd.Series(returns).skew()
        risk_metrics['kurtosis'] = pd.Series(returns).kurtosis()
        
        return risk_metrics


class FractionalBrownianMotion:
    """Fractional Brownian Motion model for long-memory processes"""
    
    def __init__(self, hurst_exponent: float = 0.5):
        self.hurst_exponent = hurst_exponent
        self.is_fitted = False
        
    def estimate_hurst_exponent(self, returns: pd.Series) -> float:
        """Estimate Hurst exponent using R/S analysis"""
        logger.info("Estimating Hurst exponent using R/S analysis")
        
        returns_array = returns.dropna().values
        n = len(returns_array)
        
        # Different time scales
        scales = np.logspace(1, np.log10(n//4), 20).astype(int)
        rs_values = []
        
        for scale in scales:
            # Divide series into non-overlapping windows of size 'scale'
            n_windows = n // scale
            rs_window = []
            
            for i in range(n_windows):
                window = returns_array[i*scale:(i+1)*scale]
                
                # Calculate mean
                mean_window = np.mean(window)
                
                # Calculate cumulative departures
                departures = np.cumsum(window - mean_window)
                
                # Calculate range
                R = np.max(departures) - np.min(departures)
                
                # Calculate standard deviation
                S = np.std(window)
                
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        # Fit log(R/S) = H * log(n) + constant
        log_scales = np.log(scales[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Linear regression
        coeffs = np.polyfit(log_scales, log_rs, 1)
        hurst_estimate = coeffs[0]
        
        self.hurst_exponent = np.clip(hurst_estimate, 0.1, 0.9)
        self.is_fitted = True
        
        logger.info(f"Estimated Hurst exponent: {self.hurst_exponent:.3f}")
        
        return self.hurst_exponent
    
    def generate_fbm(self, n_steps: int, dt: float = 1.0) -> np.ndarray:
        """Generate fractional Brownian motion using Hosking method"""
        if not self.is_fitted:
            self.hurst_exponent = 0.5  # Standard Brownian motion
        
        H = self.hurst_exponent
        
        # Generate correlated Gaussian sequence
        gamma = np.zeros(n_steps)
        gamma[0] = 1
        
        for k in range(1, n_steps):
            gamma[k] = 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + (k-1)**(2*H))
        
        # Cholesky decomposition for correlated sequence
        from scipy.linalg import cholesky, solve_triangular
        
        # Build covariance matrix
        cov_matrix = np.zeros((n_steps, n_steps))
        for i in range(n_steps):
            for j in range(n_steps):
                cov_matrix[i, j] = 0.5 * (abs(i)**(2*H) + abs(j)**(2*H) - abs(i-j)**(2*H))
        
        # Regularize for numerical stability
        cov_matrix += 1e-10 * np.eye(n_steps)
        
        try:
            L = cholesky(cov_matrix, lower=True)
            
            # Generate independent normal random variables
            Z = npr.normal(0, 1, n_steps)
            
            # Generate correlated sequence
            fbm = solve_triangular(L, Z, lower=True) * np.sqrt(dt)
            
            return np.cumsum(fbm)
            
        except np.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed, using approximate method")
            # Fallback to approximate method
            return self._approximate_fbm(n_steps, dt)
    
    def _approximate_fbm(self, n_steps: int, dt: float) -> np.ndarray:
        """Approximate fBM generation for numerical stability"""
        H = self.hurst_exponent
        
        # Generate regular Brownian motion
        dW = npr.normal(0, np.sqrt(dt), n_steps)
        
        # Apply simple fractional integration (approximate)
        if H != 0.5:
            # Use moving average with weights
            weights = np.array([(i+1)**(-H) for i in range(min(20, n_steps))])
            weights = weights / np.sum(weights)
            
            fbm_increments = np.convolve(dW, weights, mode='same')
        else:
            fbm_increments = dW
        
        return np.cumsum(fbm_increments)
    
    def simulate_price_paths(
        self, 
        S0: float, 
        mu: float, 
        sigma: float, 
        T: float, 
        n_steps: int, 
        n_paths: int
    ) -> np.ndarray:
        """Simulate price paths using fractional Brownian motion"""
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in range(n_paths):
            # Generate fBM
            fbm = self.generate_fbm(n_steps, dt)
            
            # Convert to price path
            for t in range(n_steps):
                paths[i, t+1] = paths[i, t] * np.exp(
                    mu * dt + sigma * (fbm[t+1] - fbm[t] if t < len(fbm)-1 else 0)
                )
        
        return paths


class BehavioralFinanceModel:
    """Model incorporating behavioral finance effects"""
    
    def __init__(self):
        self.sentiment_params = {}
        self.herding_params = {}
        self.is_fitted = False
    
    def fit(
        self, 
        returns: pd.Series, 
        sentiment_data: Optional[pd.Series] = None,
        volume_data: Optional[pd.Series] = None
    ) -> 'BehavioralFinanceModel':
        """Fit behavioral finance model"""
        logger.info("Fitting behavioral finance model")
        
        # Fit sentiment effect
        if sentiment_data is not None:
            self._fit_sentiment_model(returns, sentiment_data)
        
        # Fit herding effect using volume
        if volume_data is not None:
            self._fit_herding_model(returns, volume_data)
        
        self.is_fitted = True
        return self
    
    def _fit_sentiment_model(self, returns: pd.Series, sentiment: pd.Series):
        """Fit sentiment-driven return model"""
        aligned_data = pd.DataFrame({
            'returns': returns,
            'sentiment': sentiment
        }).dropna()
        
        if len(aligned_data) > 30:
            # Simple linear regression: return = alpha + beta * sentiment + error
            X = aligned_data['sentiment'].values.reshape(-1, 1)
            y = aligned_data['returns'].values
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            self.sentiment_params = {
                'alpha': model.intercept_,
                'beta': model.coef_[0],
                'r_squared': model.score(X, y)
            }
            
            logger.info(f"Sentiment model: R² = {self.sentiment_params['r_squared']:.3f}")
    
    def _fit_herding_model(self, returns: pd.Series, volume: pd.Series):
        """Fit herding model using volume as proxy"""
        aligned_data = pd.DataFrame({
            'returns': returns,
            'volume': volume,
            'abs_returns': returns.abs()
        }).dropna()
        
        if len(aligned_data) > 30:
            # Model: |return| = alpha + beta * log(volume) + error
            # High volume often indicates herding behavior
            
            X = np.log(aligned_data['volume'].values + 1).reshape(-1, 1)
            y = aligned_data['abs_returns'].values
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            self.herding_params = {
                'alpha': model.intercept_,
                'beta': model.coef_[0],
                'r_squared': model.score(X, y)
            }
            
            logger.info(f"Herding model: R² = {self.herding_params['r_squared']:.3f}")
    
    def predict_behavioral_impact(
        self, 
        sentiment: Optional[float] = None,
        volume: Optional[float] = None
    ) -> Dict[str, float]:
        """Predict behavioral impact on returns"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        impact = {'total_impact': 0.0}
        
        # Sentiment impact
        if sentiment is not None and self.sentiment_params:
            sentiment_impact = (self.sentiment_params['alpha'] + 
                              self.sentiment_params['beta'] * sentiment)
            impact['sentiment_impact'] = sentiment_impact
            impact['total_impact'] += sentiment_impact
        
        # Herding impact (affects volatility)
        if volume is not None and self.herding_params:
            herding_impact = (self.herding_params['alpha'] + 
                            self.herding_params['beta'] * np.log(volume + 1))
            impact['herding_impact'] = herding_impact
            # Herding affects expected absolute return (volatility proxy)
        
        return impact
    
    def simulate_behavioral_returns(
        self,
        n_periods: int,
        base_mu: float = 0.0,
        base_sigma: float = 0.02,
        sentiment_series: Optional[np.ndarray] = None,
        volume_series: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate returns incorporating behavioral effects"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
        
        returns = np.zeros(n_periods)
        
        for t in range(n_periods):
            # Base return (market efficiency)
            base_return = npr.normal(base_mu, base_sigma)
            
            # Behavioral adjustments
            behavioral_adjustment = 0.0
            
            # Sentiment effect
            if sentiment_series is not None and t < len(sentiment_series) and self.sentiment_params:
                sentiment_effect = self.sentiment_params['beta'] * sentiment_series[t]
                behavioral_adjustment += sentiment_effect
            
            # Herding effect (affects volatility)
            volatility_multiplier = 1.0
            if volume_series is not None and t < len(volume_series) and self.herding_params:
                herding_effect = self.herding_params['beta'] * np.log(volume_series[t] + 1)
                volatility_multiplier = 1.0 + herding_effect * 0.1  # Scale factor
            
            # Final return
            returns[t] = (base_return * volatility_multiplier + behavioral_adjustment)
        
        return returns


def create_regime_switching_garch_model(returns: pd.Series, n_regimes: int = 2) -> RegimeSwitchingGARCH:
    """Factory function to create and fit regime-switching GARCH model"""
    model = RegimeSwitchingGARCH(n_regimes=n_regimes)
    model.fit(returns)
    return model


def create_jump_diffusion_model(returns: pd.Series) -> JumpDiffusionModel:
    """Factory function to create and fit jump-diffusion model"""
    model = JumpDiffusionModel()
    model.fit(returns)
    return model


def create_fractional_brownian_model(returns: pd.Series) -> FractionalBrownianMotion:
    """Factory function to create fractional Brownian motion model"""
    model = FractionalBrownianMotion()
    model.estimate_hurst_exponent(returns)
    return model


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate regime-switching returns
    returns_data = []
    current_regime = 0
    
    for i in range(len(dates)):
        if np.random.random() < 0.02:  # 2% chance of regime switch
            current_regime = 1 - current_regime
        
        if current_regime == 0:  # Low volatility regime
            ret = np.random.normal(0.001, 0.01)
        else:  # High volatility regime
            ret = np.random.normal(-0.001, 0.03)
        
        # Add occasional jumps
        if np.random.random() < 0.01:  # 1% chance of jump
            ret += np.random.normal(-0.05, 0.02)  # Negative jump
        
        returns_data.append(ret)
    
    returns = pd.Series(returns_data, index=dates)
    
    # Test regime-switching GARCH
    print("Testing Regime-Switching GARCH...")
    rs_garch = create_regime_switching_garch_model(returns, n_regimes=2)
    regime_classification = rs_garch.get_regime_classification(returns)
    print(f"Regime distribution: {regime_classification['regime'].value_counts()}")
    
    # Test jump-diffusion model
    print("\nTesting Jump-Diffusion Model...")
    jd_model = create_jump_diffusion_model(returns)
    risk_metrics = jd_model.calculate_risk_metrics(100, 1.0)
    print(f"Jump-diffusion risk metrics: {risk_metrics}")
    
    # Test fractional Brownian motion
    print("\nTesting Fractional Brownian Motion...")
    fbm_model = create_fractional_brownian_model(returns)
    print(f"Estimated Hurst exponent: {fbm_model.hurst_exponent:.3f}")
    
    print("\nAll alternative models tested successfully!")