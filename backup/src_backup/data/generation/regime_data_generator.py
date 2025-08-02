#!/usr/bin/env python3
"""
Regime Detection Training Data Generator

Creates labeled regime datasets for training and validating HMM models:
- Generates clear regime periods with distinct characteristics
- Creates regime transition data and mixed states
- Provides ground truth labels for model validation
- Generates regime-specific features and indicators
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class RegimeType(Enum):
    """Market regime types"""
    BULL_MARKET = "Bull Market"
    BEAR_MARKET = "Bear Market"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"
    SIDEWAYS_MARKET = "Sideways Market"
    CRISIS = "Crisis"
    RECOVERY = "Recovery"

@dataclass
class RegimeCharacteristics:
    """Characteristics of each market regime"""
    regime_type: RegimeType
    mean_return: float  # Daily mean return
    volatility: float   # Daily volatility
    skewness: float    # Return distribution skewness
    kurtosis: float    # Return distribution kurtosis
    autocorrelation: float  # First-order autocorrelation
    mean_duration: int  # Average duration in days
    transition_probs: Dict[RegimeType, float]  # Transition probabilities

class RegimeDataGenerator:
    """Generate labeled regime data for HMM training and validation"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize regime data generator
        
        Args:
            random_seed: Random seed for reproducible generation
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Define regime characteristics
        self.regime_definitions = self._create_regime_definitions()
        
        # Regime transition matrix
        self.transition_matrix = self._create_transition_matrix()
        
    def _create_regime_definitions(self) -> Dict[RegimeType, RegimeCharacteristics]:
        """Define characteristics for each regime type"""
        
        return {
            RegimeType.BULL_MARKET: RegimeCharacteristics(
                regime_type=RegimeType.BULL_MARKET,
                mean_return=0.0008,  # ~20% annual
                volatility=0.012,    # ~19% annual
                skewness=0.3,        # Slightly positive skew
                kurtosis=3.5,        # Mild fat tails
                autocorrelation=0.05,
                mean_duration=120,   # ~6 months
                transition_probs={
                    RegimeType.BULL_MARKET: 0.95,
                    RegimeType.SIDEWAYS_MARKET: 0.03,
                    RegimeType.HIGH_VOLATILITY: 0.015,
                    RegimeType.BEAR_MARKET: 0.005
                }
            ),
            
            RegimeType.BEAR_MARKET: RegimeCharacteristics(
                regime_type=RegimeType.BEAR_MARKET,
                mean_return=-0.0012,  # ~-30% annual 
                volatility=0.025,     # ~40% annual
                skewness=-0.8,        # Negative skew (crash risk)
                kurtosis=6.0,         # Fat tails
                autocorrelation=-0.1, # Negative momentum
                mean_duration=80,     # ~4 months
                transition_probs={
                    RegimeType.BEAR_MARKET: 0.92,
                    RegimeType.CRISIS: 0.03,
                    RegimeType.RECOVERY: 0.03,
                    RegimeType.HIGH_VOLATILITY: 0.02
                }
            ),
            
            RegimeType.HIGH_VOLATILITY: RegimeCharacteristics(
                regime_type=RegimeType.HIGH_VOLATILITY,
                mean_return=-0.0002,  # Slightly negative
                volatility=0.035,     # ~55% annual
                skewness=-0.5,        # Negative skew
                kurtosis=8.0,         # Very fat tails
                autocorrelation=0.0,  # No momentum
                mean_duration=40,     # ~2 months
                transition_probs={
                    RegimeType.HIGH_VOLATILITY: 0.85,
                    RegimeType.BEAR_MARKET: 0.08,
                    RegimeType.SIDEWAYS_MARKET: 0.05,
                    RegimeType.CRISIS: 0.02
                }
            ),
            
            RegimeType.LOW_VOLATILITY: RegimeCharacteristics(
                regime_type=RegimeType.LOW_VOLATILITY,
                mean_return=0.0003,   # Modest positive
                volatility=0.006,     # ~9% annual
                skewness=0.1,         # Slight positive skew
                kurtosis=2.8,         # Less than normal
                autocorrelation=0.15, # Strong momentum
                mean_duration=60,     # ~3 months
                transition_probs={
                    RegimeType.LOW_VOLATILITY: 0.92,
                    RegimeType.BULL_MARKET: 0.05,
                    RegimeType.SIDEWAYS_MARKET: 0.025,
                    RegimeType.HIGH_VOLATILITY: 0.005
                }
            ),
            
            RegimeType.SIDEWAYS_MARKET: RegimeCharacteristics(
                regime_type=RegimeType.SIDEWAYS_MARKET,
                mean_return=0.0001,   # Nearly flat
                volatility=0.015,     # ~24% annual
                skewness=0.0,         # Symmetric
                kurtosis=3.2,         # Close to normal
                autocorrelation=0.02, # Weak momentum
                mean_duration=100,    # ~5 months
                transition_probs={
                    RegimeType.SIDEWAYS_MARKET: 0.90,
                    RegimeType.BULL_MARKET: 0.04,
                    RegimeType.BEAR_MARKET: 0.03,
                    RegimeType.HIGH_VOLATILITY: 0.02,
                    RegimeType.LOW_VOLATILITY: 0.01
                }
            ),
            
            RegimeType.CRISIS: RegimeCharacteristics(
                regime_type=RegimeType.CRISIS,
                mean_return=-0.002,   # ~-50% annual
                volatility=0.045,     # ~70% annual
                skewness=-1.2,        # Extreme negative skew
                kurtosis=12.0,        # Extreme fat tails
                autocorrelation=-0.2, # Strong negative momentum
                mean_duration=30,     # ~1.5 months
                transition_probs={
                    RegimeType.CRISIS: 0.80,
                    RegimeType.BEAR_MARKET: 0.15,
                    RegimeType.RECOVERY: 0.05
                }
            ),
            
            RegimeType.RECOVERY: RegimeCharacteristics(
                regime_type=RegimeType.RECOVERY,
                mean_return=0.0015,   # ~40% annual
                volatility=0.030,     # ~48% annual
                skewness=0.8,         # Positive skew
                kurtosis=5.0,         # Fat tails
                autocorrelation=0.25, # Strong positive momentum
                mean_duration=50,     # ~2.5 months
                transition_probs={
                    RegimeType.RECOVERY: 0.85,
                    RegimeType.BULL_MARKET: 0.10,
                    RegimeType.SIDEWAYS_MARKET: 0.03,
                    RegimeType.HIGH_VOLATILITY: 0.02
                }
            )
        }
    
    def _create_transition_matrix(self) -> np.ndarray:
        """Create regime transition probability matrix"""
        
        regime_list = list(RegimeType)
        n_regimes = len(regime_list)
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i, from_regime in enumerate(regime_list):
            regime_char = self.regime_definitions[from_regime]
            
            for j, to_regime in enumerate(regime_list):
                if to_regime in regime_char.transition_probs:
                    transition_matrix[i, j] = regime_char.transition_probs[to_regime]
                else:
                    # Small probability for unspecified transitions
                    transition_matrix[i, j] = 0.001
            
            # Normalize rows to sum to 1
            row_sum = transition_matrix[i, :].sum()
            transition_matrix[i, :] /= row_sum
        
        return transition_matrix
    
    def generate_regime_sequence(self, 
                               n_periods: int,
                               start_regime: Optional[RegimeType] = None) -> Tuple[List[RegimeType], np.ndarray]:
        """
        Generate sequence of regime states over time
        
        Args:
            n_periods: Number of time periods
            start_regime: Starting regime (random if None)
            
        Returns:
            Tuple of (regime_sequence, transition_points)
        """
        
        regime_list = list(RegimeType)
        n_regimes = len(regime_list)
        
        # Initialize
        if start_regime is None:
            current_regime_idx = np.random.choice(n_regimes)
        else:
            current_regime_idx = regime_list.index(start_regime)
        
        regime_sequence = []
        transition_points = []
        periods_in_regime = 0
        
        for t in range(n_periods):
            current_regime = regime_list[current_regime_idx]
            regime_sequence.append(current_regime)
            
            periods_in_regime += 1
            
            # Check for regime transition
            regime_char = self.regime_definitions[current_regime]
            
            # Probability of transition increases with time in regime
            base_transition_prob = 1.0 / regime_char.mean_duration
            duration_factor = min(2.0, periods_in_regime / regime_char.mean_duration)
            transition_prob = base_transition_prob * duration_factor
            
            if np.random.random() < transition_prob:
                # Transition to new regime
                transition_probs = self.transition_matrix[current_regime_idx, :]
                # Exclude staying in same regime for forced transition
                transition_probs[current_regime_idx] = 0
                transition_probs /= transition_probs.sum()
                
                new_regime_idx = np.random.choice(n_regimes, p=transition_probs)
                
                if new_regime_idx != current_regime_idx:
                    current_regime_idx = new_regime_idx
                    transition_points.append(t)
                    periods_in_regime = 0
        
        return regime_sequence, np.array(transition_points)
    
    def generate_returns_from_regimes(self, 
                                    regime_sequence: List[RegimeType],
                                    add_noise: bool = True) -> pd.Series:
        """
        Generate return series based on regime sequence
        
        Args:
            regime_sequence: Sequence of regimes
            add_noise: Whether to add idiosyncratic noise
            
        Returns:
            Series of returns
        """
        
        n_periods = len(regime_sequence)
        returns = np.zeros(n_periods)
        
        for t in range(n_periods):
            regime = regime_sequence[t]
            regime_char = self.regime_definitions[regime]
            
            # Base return from regime
            base_return = regime_char.mean_return
            
            # Add volatility with proper distribution
            if regime_char.skewness != 0:
                # Use skewed normal distribution
                volatility_component = stats.skewnorm.rvs(
                    a=regime_char.skewness,
                    scale=regime_char.volatility
                )
            else:
                volatility_component = np.random.normal(0, regime_char.volatility)
            
            # Add fat tails if specified
            if regime_char.kurtosis > 3.5:
                # Mix in some t-distribution for fat tails
                df = max(3, 15 - regime_char.kurtosis)  # Lower df = fatter tails
                fat_tail_component = stats.t.rvs(df) * regime_char.volatility * 0.3
                volatility_component = 0.7 * volatility_component + 0.3 * fat_tail_component
            
            # Add autocorrelation
            if t > 0 and regime_char.autocorrelation != 0:
                autocorr_component = regime_char.autocorrelation * returns[t-1]
                base_return += autocorr_component
            
            # Combine components
            total_return = base_return + volatility_component
            
            # Add idiosyncratic noise if requested
            if add_noise:
                noise = np.random.normal(0, regime_char.volatility * 0.1)
                total_return += noise
            
            returns[t] = total_return
        
        return pd.Series(returns)
    
    def generate_regime_features(self, 
                               returns: pd.Series,
                               window_sizes: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        Generate features that help identify regimes
        
        Args:
            returns: Return series
            window_sizes: Rolling window sizes for feature calculation
            
        Returns:
            DataFrame with regime detection features
        """
        
        features = pd.DataFrame(index=returns.index)
        
        # Basic return features
        features['returns'] = returns
        features['abs_returns'] = np.abs(returns)
        features['squared_returns'] = returns ** 2
        
        # Rolling statistics for different windows
        for window in window_sizes:
            prefix = f'rolling_{window}d'
            
            # Volatility measures
            features[f'{prefix}_volatility'] = returns.rolling(window).std()
            features[f'{prefix}_realized_vol'] = np.sqrt(returns.rolling(window).apply(lambda x: (x**2).sum()))
            
            # Return measures
            features[f'{prefix}_mean_return'] = returns.rolling(window).mean()
            features[f'{prefix}_cumulative_return'] = (1 + returns).rolling(window).apply(lambda x: x.prod()) - 1
            
            # Higher moments
            features[f'{prefix}_skewness'] = returns.rolling(window).skew()
            features[f'{prefix}_kurtosis'] = returns.rolling(window).kurt()
            
            # Momentum and mean reversion
            features[f'{prefix}_momentum'] = returns.rolling(window).sum()
            features[f'{prefix}_mean_reversion'] = (returns - returns.rolling(window).mean()).rolling(window).mean()
            
            # Volatility clustering
            features[f'{prefix}_vol_ratio'] = (features[f'{prefix}_volatility'] / 
                                             features[f'{prefix}_volatility'].rolling(window).mean())
            
            # Extreme movements
            features[f'{prefix}_max_return'] = returns.rolling(window).max()
            features[f'{prefix}_min_return'] = returns.rolling(window).min()
            features[f'{prefix}_range'] = features[f'{prefix}_max_return'] - features[f'{prefix}_min_return']
            
            # Autocorrelation
            features[f'{prefix}_autocorr'] = returns.rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
        
        # Cross-window features
        features['vol_short_long_ratio'] = features['rolling_5d_volatility'] / features['rolling_60d_volatility']
        features['return_short_long_ratio'] = features['rolling_5d_mean_return'] / features['rolling_60d_mean_return']
        
        # VIX-like fear index (based on short-term volatility spikes)
        features['fear_index'] = (features['rolling_5d_volatility'] / 
                                features['rolling_60d_volatility'].rolling(252).mean()) * 100
        
        # Regime change indicators
        features['volatility_spike'] = (features['rolling_5d_volatility'] > 
                                      features['rolling_60d_volatility'] * 2).astype(int)
        
        features['extreme_return'] = (np.abs(returns) > features['rolling_20d_volatility'] * 2).astype(int)
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def create_labeled_dataset(self, 
                             start_date: str = "2010-01-01",
                             end_date: str = "2024-01-01",
                             frequency: str = "D") -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive labeled dataset for regime detection
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            frequency: Data frequency
            
        Returns:
            Dictionary containing labeled datasets
        """
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_periods = len(dates)
        
        # Generate regime sequence
        regime_sequence, transition_points = self.generate_regime_sequence(n_periods)
        
        # Generate returns from regimes
        returns = self.generate_returns_from_regimes(regime_sequence)
        returns.index = dates
        
        # Generate features
        features = self.generate_regime_features(returns)
        
        # Create labels
        regime_labels = pd.Series([regime.value for regime in regime_sequence], 
                                index=dates, name='regime')
        regime_codes = pd.Series([list(RegimeType).index(regime) for regime in regime_sequence],
                               index=dates, name='regime_code')
        
        # Create regime probabilities (ground truth with some uncertainty)
        regime_probs = self._create_regime_probabilities(regime_sequence, transition_points, dates)
        
        # Create confidence scores
        confidence_scores = self._create_confidence_scores(regime_sequence, transition_points, dates)
        
        # Combine into datasets
        datasets = {
            'returns': pd.DataFrame({
                'returns': returns,
                'regime': regime_labels,
                'regime_code': regime_codes,
                'confidence': confidence_scores
            }),
            
            'features': features.join(regime_labels).join(regime_codes),
            
            'regime_probabilities': regime_probs,
            
            'transitions': pd.DataFrame({
                'transition_date': dates[transition_points] if len(transition_points) > 0 else [],
                'from_regime': [regime_sequence[t-1].value for t in transition_points] if len(transition_points) > 0 else [],
                'to_regime': [regime_sequence[t].value for t in transition_points] if len(transition_points) > 0 else []
            }),
            
            'regime_statistics': self._calculate_regime_statistics(regime_sequence, returns, dates)
        }
        
        return datasets
    
    def _create_regime_probabilities(self, 
                                   regime_sequence: List[RegimeType], 
                                   transition_points: np.ndarray,
                                   dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create regime probability matrix (ground truth with uncertainty)"""
        
        regime_list = list(RegimeType)
        n_regimes = len(regime_list)
        n_periods = len(dates)
        
        # Initialize probability matrix
        prob_matrix = np.zeros((n_periods, n_regimes))
        
        for t in range(n_periods):
            current_regime = regime_sequence[t]
            current_idx = regime_list.index(current_regime)
            
            # Base probability for true regime
            base_prob = 0.9
            
            # Reduce certainty near transitions
            distance_to_transition = float('inf')
            if len(transition_points) > 0:
                distances = np.abs(transition_points - t)
                distance_to_transition = distances.min()
            
            # Reduce confidence near transitions
            if distance_to_transition <= 5:  # Within 5 days of transition
                uncertainty_factor = 0.5 + 0.1 * distance_to_transition
                base_prob *= uncertainty_factor
            
            # Assign probabilities
            prob_matrix[t, current_idx] = base_prob
            
            # Distribute remaining probability among other regimes
            remaining_prob = 1 - base_prob
            other_indices = [i for i in range(n_regimes) if i != current_idx]
            
            if other_indices:
                # Weight by transition probabilities
                transition_probs = self.transition_matrix[current_idx, other_indices]
                transition_probs /= transition_probs.sum()
                
                for i, other_idx in enumerate(other_indices):
                    prob_matrix[t, other_idx] = remaining_prob * transition_probs[i]
        
        # Create DataFrame
        columns = [regime.value for regime in regime_list]
        regime_probs = pd.DataFrame(prob_matrix, index=dates, columns=columns)
        
        return regime_probs
    
    def _create_confidence_scores(self,
                                regime_sequence: List[RegimeType],
                                transition_points: np.ndarray, 
                                dates: pd.DatetimeIndex) -> pd.Series:
        """Create confidence scores for regime identification"""
        
        n_periods = len(dates)
        confidence = np.ones(n_periods)
        
        # Reduce confidence near transitions
        for t in range(n_periods):
            distance_to_transition = float('inf')
            
            if len(transition_points) > 0:
                distances = np.abs(transition_points - t)
                distance_to_transition = distances.min()
            
            if distance_to_transition <= 10:  # Within 10 days of transition
                confidence[t] = 0.3 + 0.07 * distance_to_transition
            elif distance_to_transition <= 20:  # Within 20 days
                confidence[t] = 0.8 + 0.01 * (distance_to_transition - 10)
        
        # Add some random noise
        confidence += np.random.normal(0, 0.05, n_periods)
        confidence = np.clip(confidence, 0.1, 1.0)
        
        return pd.Series(confidence, index=dates, name='confidence')
    
    def _calculate_regime_statistics(self,
                                   regime_sequence: List[RegimeType],
                                   returns: pd.Series,
                                   dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Calculate statistics for each regime period"""
        
        # Group consecutive periods by regime
        regime_periods = []
        current_regime = regime_sequence[0]
        start_idx = 0
        
        for i in range(1, len(regime_sequence)):
            if regime_sequence[i] != current_regime:
                # End of current regime period
                regime_periods.append({
                    'regime': current_regime.value,
                    'start_date': dates[start_idx],
                    'end_date': dates[i-1],
                    'duration': i - start_idx,
                    'mean_return': returns.iloc[start_idx:i].mean(),
                    'volatility': returns.iloc[start_idx:i].std(),
                    'total_return': (1 + returns.iloc[start_idx:i]).prod() - 1,
                    'max_drawdown': (returns.iloc[start_idx:i].cumsum() - 
                                   returns.iloc[start_idx:i].cumsum().expanding().max()).min(),
                    'sharpe_ratio': returns.iloc[start_idx:i].mean() / returns.iloc[start_idx:i].std() 
                                  if returns.iloc[start_idx:i].std() > 0 else 0
                })
                
                current_regime = regime_sequence[i]
                start_idx = i
        
        # Add final period
        regime_periods.append({
            'regime': current_regime.value,
            'start_date': dates[start_idx],
            'end_date': dates[-1],
            'duration': len(regime_sequence) - start_idx,
            'mean_return': returns.iloc[start_idx:].mean(),
            'volatility': returns.iloc[start_idx:].std(),
            'total_return': (1 + returns.iloc[start_idx:]).prod() - 1,
            'max_drawdown': (returns.iloc[start_idx:].cumsum() - 
                           returns.iloc[start_idx:].cumsum().expanding().max()).min(),
            'sharpe_ratio': returns.iloc[start_idx:].mean() / returns.iloc[start_idx:].std()
                          if returns.iloc[start_idx:].std() > 0 else 0
        })
        
        return pd.DataFrame(regime_periods)
    
    def create_validation_datasets(self,
                                 n_datasets: int = 10,
                                 periods_per_dataset: int = 1000) -> List[Dict[str, pd.DataFrame]]:
        """Create multiple validation datasets with different characteristics"""
        
        validation_sets = []
        
        for i in range(n_datasets):
            # Use different random seed for each dataset
            np.random.seed(self.random_seed + i * 100)
            
            # Create synthetic dates
            start_date = datetime(2015, 1, 1) + timedelta(days=i * 30)
            dates = pd.date_range(start=start_date, periods=periods_per_dataset, freq='D')
            
            # Generate regime sequence with varying characteristics
            regime_sequence, transition_points = self.generate_regime_sequence(
                periods_per_dataset,
                start_regime=list(RegimeType)[i % len(RegimeType)]
            )
            
            # Generate returns
            returns = self.generate_returns_from_regimes(regime_sequence, add_noise=True)
            returns.index = dates
            
            # Generate features
            features = self.generate_regime_features(returns)
            
            # Create labels
            regime_labels = pd.Series([regime.value for regime in regime_sequence], 
                                    index=dates, name='regime')
            
            validation_sets.append({
                'dataset_id': i,
                'returns': returns,
                'features': features,
                'labels': regime_labels,
                'transitions': transition_points,
                'regime_sequence': regime_sequence
            })
        
        return validation_sets
    
    def save_regime_data(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "data/generated/regimes"):
        """Save regime datasets to files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main datasets
        for name, df in datasets.items():
            df.to_csv(os.path.join(output_dir, f"{name}.csv"))
        
        # Save regime definitions
        regime_info = []
        for regime_type, characteristics in self.regime_definitions.items():
            regime_info.append({
                'regime': regime_type.value,
                'mean_return': characteristics.mean_return,
                'volatility': characteristics.volatility,
                'skewness': characteristics.skewness,
                'kurtosis': characteristics.kurtosis,
                'autocorrelation': characteristics.autocorrelation,
                'mean_duration': characteristics.mean_duration
            })
        
        regime_df = pd.DataFrame(regime_info)
        regime_df.to_csv(os.path.join(output_dir, "regime_definitions.csv"), index=False)
        
        # Save transition matrix
        regime_list = [regime.value for regime in RegimeType]
        transition_df = pd.DataFrame(self.transition_matrix, 
                                   index=regime_list, 
                                   columns=regime_list)
        transition_df.to_csv(os.path.join(output_dir, "transition_matrix.csv"))
        
        print(f"Regime data saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Create generator
    generator = RegimeDataGenerator(random_seed=42)
    
    # Generate labeled dataset
    regime_data = generator.create_labeled_dataset(
        start_date="2015-01-01",
        end_date="2024-01-01"
    )
    
    # Save data
    generator.save_regime_data(regime_data)
    
    # Generate validation datasets
    validation_sets = generator.create_validation_datasets(n_datasets=5)
    
    # Print summary
    print("Generated regime data summary:")
    for name, df in regime_data.items():
        print(f"{name}: {len(df)} records")
    
    print(f"Generated {len(validation_sets)} validation datasets")