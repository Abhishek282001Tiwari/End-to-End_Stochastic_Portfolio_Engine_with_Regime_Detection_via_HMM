import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import mode
import warnings

from src.models.hmm.hmm_engine import RegimeDetectionHMM, EnsembleRegimeDetector
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class AdaptiveFeatureSelector:
    """Adaptive feature selection for regime detection"""
    
    def __init__(self, method: str = "recursive", n_features: int = 10):
        self.method = method
        self.n_features = n_features
        self.selected_features = None
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaptiveFeatureSelector':
        logger.info(f"Performing adaptive feature selection using {self.method}")
        
        if self.method == "recursive":
            # Use Random Forest for recursive feature elimination
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.selector = RFE(estimator, n_features_to_select=self.n_features)
            
        elif self.method == "univariate":
            # Use statistical tests
            self.selector = SelectKBest(score_func=f_classif, k=self.n_features)
            
        elif self.method == "importance":
            # Use feature importance from tree-based models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            self.selected_features = importances.nlargest(self.n_features).index.tolist()
            
            return self
        
        # Fit selector
        self.selector.fit(X, y)
        
        if hasattr(self.selector, 'get_support'):
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            raise ValueError("Feature selector must be fitted first")
        
        return X[self.selected_features]
    
    def get_feature_importance_ranking(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get ranking of all features by importance"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        return importances.sort_values(ascending=False)


class MLRegimeDetector:
    """Machine learning-based regime detector"""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_selector = None
        
    def _create_model(self):
        """Create the underlying ML model"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 10),
                random_state=42
            )
            
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 6),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=42
            )
            
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 6),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=42
            )
            
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 6),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=42,
                verbose=-1
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, use_feature_selection: bool = True) -> 'MLRegimeDetector':
        logger.info(f"Training ML regime detector using {self.model_type}")
        
        self._create_model()
        
        # Feature selection
        if use_feature_selection:
            self.feature_selector = AdaptiveFeatureSelector(n_features=min(15, len(X.columns)))
            self.feature_selector.fit(X, y)
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Log feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_scaled.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 most important features:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Apply feature selection if used
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Apply feature selection if used
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return pd.Series()
        
        if self.feature_selector:
            feature_names = self.feature_selector.selected_features
        else:
            feature_names = self.scaler.feature_names_in_
        
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)


class HybridRegimeDetector:
    """Hybrid detector combining HMM and ML approaches"""
    
    def __init__(
        self,
        hmm_weight: float = 0.6,
        ml_weight: float = 0.4,
        n_components: int = 3
    ):
        self.hmm_weight = hmm_weight
        self.ml_weight = ml_weight
        self.n_components = n_components
        
        # Initialize models
        self.hmm_model = RegimeDetectionHMM(n_components=n_components)
        self.ml_model = MLRegimeDetector(model_type="random_forest")
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, known_regimes: Optional[pd.Series] = None) -> 'HybridRegimeDetector':
        logger.info("Training hybrid regime detector")
        
        # Fit HMM model
        self.hmm_model.fit(X)
        
        # If we have known regimes, use them for ML training
        if known_regimes is not None:
            self.ml_model.fit(X, known_regimes)
        else:
            # Use HMM predictions as pseudo-labels for ML training
            hmm_predictions = self.hmm_model.predict_regimes(X)
            self.ml_model.fit(X, pd.Series(hmm_predictions, index=X.index))
        
        self.is_fitted = True
        
        return self
    
    def predict_regime_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from both models
        hmm_probs = self.hmm_model.predict_regime_probabilities(X)
        ml_probs = self.ml_model.predict_proba(X)
        
        # Combine probabilities using weighted average
        combined_probs = (self.hmm_weight * hmm_probs + self.ml_weight * ml_probs)
        
        # Normalize to ensure probabilities sum to 1
        combined_probs = combined_probs / combined_probs.sum(axis=1, keepdims=True)
        
        return combined_probs
    
    def predict_regimes(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_regime_probabilities(X)
        return np.argmax(probs, axis=1)


class EnsembleRegimeDetector:
    """Advanced ensemble regime detector with multiple models"""
    
    def __init__(self, models: Optional[List[Any]] = None):
        if models is None:
            # Default ensemble
            self.models = [
                RegimeDetectionHMM(n_components=3, covariance_type="full"),
                RegimeDetectionHMM(n_components=3, covariance_type="diag"),
                MLRegimeDetector(model_type="random_forest"),
                MLRegimeDetector(model_type="gradient_boosting"),
                MLRegimeDetector(model_type="xgboost")
            ]
        else:
            self.models = models
        
        self.model_weights = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, validation_split: float = 0.2) -> 'EnsembleRegimeDetector':
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        # Split data for validation-based weighting
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        
        # Train all models
        trained_models = []
        model_scores = []
        
        for i, model in enumerate(self.models):
            try:
                logger.info(f"Training model {i+1}/{len(self.models)}: {type(model).__name__}")
                
                if isinstance(model, RegimeDetectionHMM):
                    model.fit(X_train)
                    
                    # Evaluate on validation set
                    val_predictions = model.predict_regimes(X_val)
                    
                    # Use log-likelihood as score for HMM
                    score = model.score(X_val)
                    
                elif isinstance(model, MLRegimeDetector):
                    # Use HMM to generate pseudo-labels for training
                    hmm_temp = RegimeDetectionHMM(n_components=3)
                    hmm_temp.fit(X_train)
                    train_labels = hmm_temp.predict_regimes(X_train)
                    
                    model.fit(X_train, pd.Series(train_labels, index=X_train.index))
                    
                    # Evaluate on validation set
                    val_predictions = model.predict(X_val)
                    val_labels = hmm_temp.predict_regimes(X_val)
                    
                    score = accuracy_score(val_labels, val_predictions)
                
                trained_models.append(model)
                model_scores.append(score)
                
                logger.info(f"Model {i+1} score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training model {i+1}: {e}")
                continue
        
        self.models = trained_models
        
        # Calculate model weights based on performance
        if model_scores:
            scores_array = np.array(model_scores)
            # Use softmax to convert scores to weights
            exp_scores = np.exp(scores_array - np.max(scores_array))
            self.model_weights = exp_scores / np.sum(exp_scores)
            
            logger.info("Model weights:")
            for i, (model, weight) in enumerate(zip(self.models, self.model_weights)):
                logger.info(f"  {type(model).__name__}: {weight:.4f}")
        else:
            # Equal weights if no scores available
            self.model_weights = np.ones(len(self.models)) / len(self.models)
        
        self.is_fitted = True
        
        return self
    
    def predict_regime_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        all_probabilities = []
        
        for model, weight in zip(self.models, self.model_weights):
            try:
                if isinstance(model, RegimeDetectionHMM):
                    probs = model.predict_regime_probabilities(X)
                elif isinstance(model, MLRegimeDetector) and hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                else:
                    # Convert hard predictions to probabilities
                    predictions = model.predict(X) if hasattr(model, 'predict') else model.predict_regimes(X)
                    n_classes = 3  # Assuming 3 regimes
                    probs = np.zeros((len(predictions), n_classes))
                    for i, pred in enumerate(predictions):
                        probs[i, pred] = 1.0
                
                # Weight the probabilities
                weighted_probs = probs * weight
                all_probabilities.append(weighted_probs)
                
            except Exception as e:
                logger.error(f"Error getting predictions from {type(model).__name__}: {e}")
                continue
        
        if not all_probabilities:
            raise ValueError("No valid predictions from ensemble models")
        
        # Combine weighted probabilities
        ensemble_probs = np.sum(all_probabilities, axis=0)
        
        # Normalize to ensure probabilities sum to 1
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        
        return ensemble_probs
    
    def predict_regimes(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_regime_probabilities(X)
        return np.argmax(probs, axis=1)
    
    def get_model_consensus(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get consensus information from all models"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting consensus")
        
        all_predictions = []
        model_names = []
        
        for model in self.models:
            try:
                if hasattr(model, 'predict_regimes'):
                    predictions = model.predict_regimes(X)
                else:
                    predictions = model.predict(X)
                
                all_predictions.append(predictions)
                model_names.append(type(model).__name__)
                
            except Exception as e:
                logger.error(f"Error getting predictions from {type(model).__name__}: {e}")
        
        if not all_predictions:
            return {}
        
        # Calculate consensus metrics
        predictions_array = np.array(all_predictions).T  # Shape: (n_samples, n_models)
        
        consensus_metrics = {
            'model_names': model_names,
            'individual_predictions': all_predictions,
            'majority_vote': [],
            'consensus_strength': [],
            'disagreement_rate': []
        }
        
        for i in range(len(X)):
            sample_predictions = predictions_array[i]
            
            # Majority vote
            majority_vote = mode(sample_predictions, keepdims=True)[0][0]
            consensus_metrics['majority_vote'].append(majority_vote)
            
            # Consensus strength (percentage of models agreeing with majority)
            consensus_strength = np.mean(sample_predictions == majority_vote)
            consensus_metrics['consensus_strength'].append(consensus_strength)
            
            # Disagreement rate (percentage of unique predictions)
            unique_predictions = len(np.unique(sample_predictions))
            disagreement_rate = unique_predictions / len(sample_predictions)
            consensus_metrics['disagreement_rate'].append(disagreement_rate)
        
        return consensus_metrics


class AdaptiveLearningSystem:
    """Adaptive learning system that adjusts to changing market conditions"""
    
    def __init__(self, base_detector: Any, adaptation_window: int = 252):
        self.base_detector = base_detector
        self.adaptation_window = adaptation_window
        self.performance_history = []
        self.adaptation_triggers = []
        
    def fit(self, X: pd.DataFrame, **kwargs) -> 'AdaptiveLearningSystem':
        """Initial training"""
        self.base_detector.fit(X, **kwargs)
        return self
    
    def adaptive_update(
        self, 
        new_data: pd.DataFrame, 
        true_regimes: Optional[pd.Series] = None
    ) -> bool:
        """Adaptively update the model based on new data"""
        logger.info("Performing adaptive model update")
        
        # Evaluate current performance if true regimes are available
        if true_regimes is not None:
            predictions = self.base_detector.predict_regimes(new_data)
            current_accuracy = accuracy_score(true_regimes, predictions)
            self.performance_history.append(current_accuracy)
            
            # Check if performance has degraded significantly
            if len(self.performance_history) > 10:
                recent_performance = np.mean(self.performance_history[-5:])
                historical_performance = np.mean(self.performance_history[-10:-5])
                
                performance_drop = historical_performance - recent_performance
                
                if performance_drop > 0.1:  # 10% drop in accuracy
                    logger.info(f"Performance drop detected: {performance_drop:.3f}")
                    self.adaptation_triggers.append({
                        'date': new_data.index[-1],
                        'reason': 'performance_drop',
                        'drop_amount': performance_drop
                    })
                    
                    # Retrain on recent data
                    recent_data = new_data.tail(self.adaptation_window)
                    
                    if isinstance(self.base_detector, RegimeDetectionHMM):
                        self.base_detector.fit(recent_data)
                    elif isinstance(self.base_detector, MLRegimeDetector):
                        # Use latest predictions as pseudo-labels
                        pseudo_labels = self.base_detector.predict(recent_data)
                        self.base_detector.fit(recent_data, pd.Series(pseudo_labels, index=recent_data.index))
                    
                    return True
        
        return False
    
    def detect_regime_change(self, new_data: pd.DataFrame, threshold: float = 0.8) -> bool:
        """Detect if a regime change has occurred"""
        if len(new_data) < 10:
            return False
        
        # Get regime probabilities for recent data
        recent_probs = self.base_detector.predict_regime_probabilities(new_data.tail(10))
        
        # Check if there's high uncertainty (no regime with high probability)
        max_probs = np.max(recent_probs, axis=1)
        uncertain_periods = np.mean(max_probs < threshold)
        
        if uncertain_periods > 0.7:  # 70% of recent periods are uncertain
            logger.info(f"High uncertainty detected: {uncertain_periods:.2%} of recent periods")
            return True
        
        return False


def create_advanced_ensemble(
    data: pd.DataFrame,
    n_components: int = 3,
    validation_split: float = 0.2
) -> EnsembleRegimeDetector:
    """Create an advanced ensemble detector with optimal configuration"""
    
    models = [
        # HMM variants
        RegimeDetectionHMM(n_components=n_components, covariance_type="full"),
        RegimeDetectionHMM(n_components=n_components, covariance_type="diag"),
        RegimeDetectionHMM(n_components=n_components, covariance_type="spherical"),
        
        # ML variants
        MLRegimeDetector(model_type="random_forest", n_estimators=200, max_depth=10),
        MLRegimeDetector(model_type="gradient_boosting", n_estimators=200, learning_rate=0.1),
        MLRegimeDetector(model_type="xgboost", n_estimators=200, learning_rate=0.1),
        
        # Hybrid model
        HybridRegimeDetector(hmm_weight=0.6, ml_weight=0.4, n_components=n_components)
    ]
    
    ensemble = EnsembleRegimeDetector(models)
    ensemble.fit(data, validation_split=validation_split)
    
    return ensemble


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Simulate regime-switching data
    n_features = 8
    regime_data = []
    current_regime = 0
    
    for i in range(len(dates)):
        # Switch regimes occasionally
        if np.random.random() < 0.01:
            current_regime = np.random.choice([0, 1, 2])
        
        # Generate features based on current regime
        if current_regime == 0:  # Bull market
            features = np.random.normal([0.002, 0.15, 0.8, 15, 0.02, 0.1, 0.05, 0.03], 
                                      [0.01, 0.05, 0.2, 3, 0.01, 0.03, 0.02, 0.01])
        elif current_regime == 1:  # Bear market
            features = np.random.normal([-0.003, 0.25, 1.2, 25, -0.01, -0.05, 0.08, 0.04], 
                                      [0.015, 0.08, 0.3, 5, 0.02, 0.04, 0.03, 0.02])
        else:  # Sideways market
            features = np.random.normal([0.0, 0.12, 0.6, 18, 0.005, 0.02, 0.06, 0.025], 
                                      [0.008, 0.03, 0.15, 2, 0.008, 0.02, 0.02, 0.01])
        
        regime_data.append(features)
    
    feature_names = ['returns', 'volatility', 'volume_ratio', 'vix', 'yield_curve', 
                    'sentiment', 'momentum', 'mean_reversion']
    
    X = pd.DataFrame(regime_data, columns=feature_names, index=dates)
    
    # Test ensemble detector
    ensemble = create_advanced_ensemble(X)
    
    # Get predictions
    regime_probs = ensemble.predict_regime_probabilities(X)
    regime_predictions = ensemble.predict_regimes(X)
    
    # Get consensus information
    consensus = ensemble.get_model_consensus(X)
    
    print(f"Ensemble trained with {len(ensemble.models)} models")
    print(f"Regime distribution: {np.bincount(regime_predictions) / len(regime_predictions)}")
    print(f"Average consensus strength: {np.mean(consensus['consensus_strength']):.3f}")
    print(f"Average disagreement rate: {np.mean(consensus['disagreement_rate']):.3f}")