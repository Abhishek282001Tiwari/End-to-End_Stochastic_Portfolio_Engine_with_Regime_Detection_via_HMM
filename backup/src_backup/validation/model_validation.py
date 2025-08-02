import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
from scipy.stats import kstest, jarque_bera, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

from src.models.hmm.hmm_engine import RegimeDetectionHMM
from src.models.ml_enhancements.ensemble_methods import MLRegimeDetector, EnsembleRegimeDetector
from src.optimization.portfolio.stochastic_optimizer import PortfolioOptimizationEngine
from src.backtesting.engine.backtesting_engine import BacktestingEngine, BacktestConfig
from src.utils.performance_analytics import PerformanceAnalytics
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelValidationFramework:
    """Comprehensive framework for validating financial models"""
    
    def __init__(self):
        self.validation_results = {}
        self.performance_analytics = PerformanceAnalytics()
        
    def validate_regime_detection_model(
        self,
        model: Any,
        data: pd.DataFrame,
        true_regimes: Optional[pd.Series] = None,
        test_size: float = 0.3,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Comprehensive validation of regime detection models"""
        logger.info("Starting regime detection model validation")
        
        validation_results = {
            'model_type': type(model).__name__,
            'validation_date': datetime.now(),
            'data_characteristics': self._analyze_data_characteristics(data),
            'statistical_tests': {},
            'cross_validation': {},
            'backtesting_performance': {},
            'regime_accuracy': {},
            'stability_tests': {}
        }
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Fit model on training data
        if hasattr(model, 'fit'):
            model.fit(train_data)
        
        # 1. Statistical Tests
        validation_results['statistical_tests'] = self._perform_statistical_tests(
            model, train_data, test_data
        )
        
        # 2. Cross-validation
        validation_results['cross_validation'] = self._perform_time_series_cross_validation(
            model, data, n_splits
        )
        
        # 3. Regime accuracy if true regimes available
        if true_regimes is not None:
            validation_results['regime_accuracy'] = self._validate_regime_accuracy(
                model, test_data, true_regimes.iloc[split_idx:]
            )
        
        # 4. Stability tests
        validation_results['stability_tests'] = self._perform_stability_tests(
            model, data
        )
        
        # 5. Out-of-sample performance
        validation_results['out_of_sample'] = self._evaluate_out_of_sample_performance(
            model, train_data, test_data
        )
        
        self.validation_results[f"{type(model).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = validation_results
        
        return validation_results
    
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of input data"""
        characteristics = {
            'n_observations': len(data),
            'n_features': len(data.columns),
            'date_range': (data.index[0], data.index[-1]),
            'missing_values': data.isnull().sum().to_dict(),
            'feature_statistics': {}
        }
        
        for col in data.columns:
            characteristics['feature_statistics'][col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis(),
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        return characteristics
    
    def _perform_statistical_tests(
        self,
        model: Any,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform various statistical tests on the model"""
        tests = {}
        
        try:
            # Get model predictions
            if hasattr(model, 'predict_regimes'):
                train_predictions = model.predict_regimes(train_data)
                test_predictions = model.predict_regimes(test_data)
            elif hasattr(model, 'predict'):
                train_predictions = model.predict(train_data)
                test_predictions = model.predict(test_data)
            else:
                logger.warning("Model doesn't have predict method")
                return tests
            
            # Test for regime persistence
            tests['regime_persistence'] = self._test_regime_persistence(train_predictions)
            
            # Test for regime switching frequency
            tests['switching_frequency'] = self._test_switching_frequency(train_predictions)
            
            # Ljung-Box test for residual autocorrelation (if applicable)
            if len(train_data.columns) == 1:  # Single feature case
                residuals = self._calculate_residuals(model, train_data)
                if residuals is not None:
                    tests['ljung_box'] = self._ljung_box_test(residuals)
            
            # Test for distributional assumptions
            tests['normality_tests'] = self._test_normality_assumptions(train_data)
            
        except Exception as e:
            logger.error(f"Error in statistical tests: {e}")
            tests['error'] = str(e)
        
        return tests
    
    def _test_regime_persistence(self, predictions: np.ndarray) -> Dict[str, float]:
        """Test average regime persistence"""
        changes = np.diff(predictions)
        n_changes = np.sum(changes != 0)
        avg_persistence = len(predictions) / (n_changes + 1) if n_changes > 0 else len(predictions)
        
        return {
            'average_persistence_days': avg_persistence,
            'regime_changes': int(n_changes),
            'change_frequency': n_changes / len(predictions)
        }
    
    def _test_switching_frequency(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Test regime switching frequency"""
        unique_regimes = np.unique(predictions)
        regime_counts = {int(regime): np.sum(predictions == regime) for regime in unique_regimes}
        
        total_switches = np.sum(np.diff(predictions) != 0)
        
        return {
            'unique_regimes': len(unique_regimes),
            'regime_distribution': regime_counts,
            'total_switches': int(total_switches),
            'switches_per_period': total_switches / len(predictions)
        }
    
    def _calculate_residuals(self, model: Any, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate model residuals if possible"""
        try:
            if hasattr(model, 'score'):
                # For HMM models, use negative log-likelihood as proxy
                return None  # Placeholder
            else:
                return None
        except:
            return None
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation"""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        try:
            result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            
            return {
                'statistic': float(result['lb_stat'].iloc[-1]),
                'p_value': float(result['lb_pvalue'].iloc[-1]),
                'significant_autocorr': float(result['lb_pvalue'].iloc[-1]) < 0.05
            }
        except Exception as e:
            logger.error(f"Ljung-Box test failed: {e}")
            return {'error': str(e)}
    
    def _test_normality_assumptions(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Test normality assumptions for each feature"""
        normality_tests = {}
        
        for col in data.columns:
            col_data = data[col].dropna()
            
            if len(col_data) > 20:
                # Jarque-Bera test
                jb_stat, jb_pvalue = jarque_bera(col_data)
                
                # Shapiro-Wilk test (for smaller samples)
                if len(col_data) <= 5000:
                    sw_stat, sw_pvalue = stats.shapiro(col_data)
                else:
                    sw_stat, sw_pvalue = np.nan, np.nan
                
                # Kolmogorov-Smirnov test against normal distribution
                ks_stat, ks_pvalue = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                
                normality_tests[col] = {
                    'jarque_bera_stat': float(jb_stat),
                    'jarque_bera_pvalue': float(jb_pvalue),
                    'shapiro_wilk_stat': float(sw_stat),
                    'shapiro_wilk_pvalue': float(sw_pvalue),
                    'ks_stat': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'is_normal_jb': float(jb_pvalue) > 0.05,
                    'is_normal_sw': float(sw_pvalue) > 0.05,
                    'is_normal_ks': float(ks_pvalue) > 0.05
                }
        
        return normality_tests
    
    def _perform_time_series_cross_validation(
        self,
        model: Any,
        data: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            try:
                # Split data
                train_fold = data.iloc[train_idx]
                test_fold = data.iloc[test_idx]
                
                # Clone and fit model
                if hasattr(model, 'fit'):
                    # Create a copy of the model
                    if hasattr(model, '__class__'):
                        fold_model = model.__class__(**model.__dict__)
                    else:
                        fold_model = model
                    
                    fold_model.fit(train_fold)
                    
                    # Evaluate
                    if hasattr(fold_model, 'score'):
                        score = fold_model.score(test_fold)
                    elif hasattr(fold_model, 'predict_regimes'):
                        # For regime detection, use consistency score
                        predictions = fold_model.predict_regimes(test_fold)
                        score = self._calculate_regime_consistency_score(predictions)
                    else:
                        score = 0.0
                    
                    cv_scores.append(score)
                    fold_results.append({
                        'fold': fold,
                        'train_size': len(train_fold),
                        'test_size': len(test_fold),
                        'score': score
                    })
                    
            except Exception as e:
                logger.error(f"Error in CV fold {fold}: {e}")
                cv_scores.append(np.nan)
                fold_results.append({
                    'fold': fold,
                    'error': str(e)
                })
        
        valid_scores = [s for s in cv_scores if not np.isnan(s)]
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(valid_scores) if valid_scores else np.nan,
            'std_cv_score': np.std(valid_scores) if valid_scores else np.nan,
            'fold_results': fold_results,
            'n_successful_folds': len(valid_scores)
        }
    
    def _calculate_regime_consistency_score(self, predictions: np.ndarray) -> float:
        """Calculate consistency score for regime predictions"""
        if len(predictions) < 2:
            return 0.0
        
        # Measure how consistent the regime assignments are
        # Higher score for more persistent regimes
        changes = np.sum(np.diff(predictions) != 0)
        consistency_score = 1.0 - (changes / (len(predictions) - 1))
        
        return consistency_score
    
    def _validate_regime_accuracy(
        self,
        model: Any,
        test_data: pd.DataFrame,
        true_regimes: pd.Series
    ) -> Dict[str, Any]:
        """Validate regime detection accuracy against true regimes"""
        if hasattr(model, 'predict_regimes'):
            predicted_regimes = model.predict_regimes(test_data)
        elif hasattr(model, 'predict'):
            predicted_regimes = model.predict(test_data)
        else:
            return {'error': 'Model has no prediction method'}
        
        # Align predictions with true regimes
        min_length = min(len(predicted_regimes), len(true_regimes))
        predicted_regimes = predicted_regimes[:min_length]
        true_regimes = true_regimes.iloc[:min_length]
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(true_regimes, predicted_regimes)
        
        # Calculate per-class metrics
        unique_regimes = np.unique(np.concatenate([true_regimes, predicted_regimes]))
        
        precision_scores = {}
        recall_scores = {}
        f1_scores = {}
        
        for regime in unique_regimes:
            try:
                precision_scores[int(regime)] = precision_score(
                    true_regimes, predicted_regimes, labels=[regime], average=None
                )[0] if regime in predicted_regimes else 0.0
                
                recall_scores[int(regime)] = recall_score(
                    true_regimes, predicted_regimes, labels=[regime], average=None
                )[0] if regime in true_regimes else 0.0
                
                f1_scores[int(regime)] = f1_score(
                    true_regimes, predicted_regimes, labels=[regime], average=None
                )[0] if regime in predicted_regimes and regime in true_regimes else 0.0
                
            except Exception as e:
                logger.error(f"Error calculating metrics for regime {regime}: {e}")
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_regimes, predicted_regimes)
        
        return {
            'overall_accuracy': float(accuracy),
            'precision_by_regime': precision_scores,
            'recall_by_regime': recall_scores,
            'f1_by_regime': f1_scores,
            'confusion_matrix': conf_matrix.tolist(),
            'n_samples': int(min_length)
        }
    
    def _perform_stability_tests(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Test model stability across different time periods"""
        stability_results = {}
        
        # Split data into multiple periods
        n_periods = 4
        period_length = len(data) // n_periods
        
        period_scores = []
        period_regimes = []
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(data)
            
            period_data = data.iloc[start_idx:end_idx]
            
            try:
                if hasattr(model, 'predict_regimes'):
                    predictions = model.predict_regimes(period_data)
                    period_regimes.append(predictions)
                    
                    # Calculate consistency score for this period
                    score = self._calculate_regime_consistency_score(predictions)
                    period_scores.append(score)
                    
            except Exception as e:
                logger.error(f"Error in stability test period {i}: {e}")
                period_scores.append(np.nan)
        
        valid_scores = [s for s in period_scores if not np.isnan(s)]
        
        stability_results['period_scores'] = period_scores
        stability_results['score_stability'] = {
            'mean': np.mean(valid_scores) if valid_scores else np.nan,
            'std': np.std(valid_scores) if valid_scores else np.nan,
            'min': np.min(valid_scores) if valid_scores else np.nan,
            'max': np.max(valid_scores) if valid_scores else np.nan
        }
        
        # Test regime distribution stability
        if period_regimes:
            regime_distributions = []
            for regimes in period_regimes:
                unique_regimes, counts = np.unique(regimes, return_counts=True)
                distribution = {int(regime): count/len(regimes) for regime, count in zip(unique_regimes, counts)}
                regime_distributions.append(distribution)
            
            stability_results['regime_distribution_stability'] = regime_distributions
        
        return stability_results
    
    def _evaluate_out_of_sample_performance(
        self,
        model: Any,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate out-of-sample performance"""
        performance = {}
        
        try:
            # Get predictions on test data
            if hasattr(model, 'predict_regimes'):
                test_predictions = model.predict_regimes(test_data)
                performance['regime_predictions'] = test_predictions.tolist()
                
                # Calculate regime statistics
                unique_regimes, counts = np.unique(test_predictions, return_counts=True)
                performance['regime_distribution'] = {
                    int(regime): int(count) for regime, count in zip(unique_regimes, counts)
                }
                
                # Calculate prediction confidence if available
                if hasattr(model, 'predict_regime_probabilities'):
                    probabilities = model.predict_regime_probabilities(test_data)
                    max_probs = np.max(probabilities, axis=1)
                    performance['prediction_confidence'] = {
                        'mean': float(np.mean(max_probs)),
                        'std': float(np.std(max_probs)),
                        'min': float(np.min(max_probs)),
                        'max': float(np.max(max_probs))
                    }
            
            # Model scoring if available
            if hasattr(model, 'score'):
                performance['model_score'] = float(model.score(test_data))
            
        except Exception as e:
            logger.error(f"Error in out-of-sample evaluation: {e}")
            performance['error'] = str(e)
        
        return performance
    
    def validate_portfolio_optimization_model(
        self,
        optimizer: PortfolioOptimizationEngine,
        returns_data: pd.DataFrame,
        benchmark_returns: pd.Series,
        test_period_days: int = 252
    ) -> Dict[str, Any]:
        """Validate portfolio optimization model performance"""
        logger.info("Validating portfolio optimization model")
        
        validation_results = {
            'model_type': 'PortfolioOptimization',
            'validation_date': datetime.now(),
            'backtest_results': {},
            'risk_metrics': {},
            'benchmark_comparison': {}
        }
        
        # Set up backtest configuration
        end_date = returns_data.index[-1]
        start_date = end_date - timedelta(days=test_period_days)
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000,
            rebalance_frequency='M',
            transaction_costs=0.001,
            benchmark_symbol='SPY',
            lookback_window=60,
            optimization_method='mean_variance'
        )
        
        # Run backtest
        try:
            backtesting_engine = BacktestingEngine(config, optimizer)
            
            test_returns = returns_data[returns_data.index >= start_date]
            test_benchmark = benchmark_returns[benchmark_returns.index >= start_date]
            
            backtest_results = backtesting_engine.run_backtest(
                test_returns, test_benchmark
            )
            
            validation_results['backtest_results'] = {
                'total_return': backtest_results.performance_metrics.get('total_return', 0),
                'sharpe_ratio': backtest_results.performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': backtest_results.performance_metrics.get('max_drawdown', 0),
                'alpha': backtest_results.performance_metrics.get('alpha', 0),
                'information_ratio': backtest_results.performance_metrics.get('information_ratio', 0)
            }
            
            # Calculate additional risk metrics
            portfolio_returns = backtest_results.portfolio_returns
            
            risk_metrics = self.performance_analytics.calculate_comprehensive_metrics(
                portfolio_returns, test_benchmark
            )
            
            validation_results['risk_metrics'] = risk_metrics
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization validation: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        report = "MODEL VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Model Type: {validation_results.get('model_type', 'Unknown')}\n"
        report += f"Validation Date: {validation_results.get('validation_date', 'Unknown')}\n\n"
        
        # Data characteristics
        if 'data_characteristics' in validation_results:
            data_char = validation_results['data_characteristics']
            report += "DATA CHARACTERISTICS:\n"
            report += f"  Observations: {data_char.get('n_observations', 'N/A')}\n"
            report += f"  Features: {data_char.get('n_features', 'N/A')}\n"
            report += f"  Date Range: {data_char.get('date_range', 'N/A')}\n\n"
        
        # Statistical tests
        if 'statistical_tests' in validation_results:
            stat_tests = validation_results['statistical_tests']
            report += "STATISTICAL TESTS:\n"
            
            if 'regime_persistence' in stat_tests:
                persistence = stat_tests['regime_persistence']
                report += f"  Average Regime Persistence: {persistence.get('average_persistence_days', 'N/A'):.1f} days\n"
                report += f"  Regime Changes: {persistence.get('regime_changes', 'N/A')}\n"
            
            if 'switching_frequency' in stat_tests:
                switching = stat_tests['switching_frequency']
                report += f"  Unique Regimes: {switching.get('unique_regimes', 'N/A')}\n"
                report += f"  Total Switches: {switching.get('total_switches', 'N/A')}\n"
            
            report += "\n"
        
        # Cross-validation results
        if 'cross_validation' in validation_results:
            cv_results = validation_results['cross_validation']
            report += "CROSS-VALIDATION RESULTS:\n"
            report += f"  Mean CV Score: {cv_results.get('mean_cv_score', 'N/A'):.4f}\n"
            report += f"  Std CV Score: {cv_results.get('std_cv_score', 'N/A'):.4f}\n"
            report += f"  Successful Folds: {cv_results.get('n_successful_folds', 'N/A')}\n\n"
        
        # Regime accuracy
        if 'regime_accuracy' in validation_results:
            accuracy = validation_results['regime_accuracy']
            report += "REGIME ACCURACY:\n"
            report += f"  Overall Accuracy: {accuracy.get('overall_accuracy', 'N/A'):.2%}\n"
            
            if 'precision_by_regime' in accuracy:
                report += "  Precision by Regime:\n"
                for regime, precision in accuracy['precision_by_regime'].items():
                    report += f"    Regime {regime}: {precision:.2%}\n"
            
            report += "\n"
        
        # Stability tests
        if 'stability_tests' in validation_results:
            stability = validation_results['stability_tests']
            if 'score_stability' in stability:
                score_stab = stability['score_stability']
                report += "STABILITY TESTS:\n"
                report += f"  Mean Period Score: {score_stab.get('mean', 'N/A'):.4f}\n"
                report += f"  Score Std Dev: {score_stab.get('std', 'N/A'):.4f}\n"
                report += f"  Score Range: {score_stab.get('min', 'N/A'):.4f} - {score_stab.get('max', 'N/A'):.4f}\n\n"
        
        # Portfolio optimization specific results
        if 'backtest_results' in validation_results:
            backtest = validation_results['backtest_results']
            report += "BACKTEST PERFORMANCE:\n"
            report += f"  Total Return: {backtest.get('total_return', 'N/A'):.2%}\n"
            report += f"  Sharpe Ratio: {backtest.get('sharpe_ratio', 'N/A'):.2f}\n"
            report += f"  Maximum Drawdown: {backtest.get('max_drawdown', 'N/A'):.2%}\n"
            report += f"  Alpha: {backtest.get('alpha', 'N/A'):.2%}\n"
            report += f"  Information Ratio: {backtest.get('information_ratio', 'N/A'):.2f}\n\n"
        
        # Recommendations
        report += "VALIDATION SUMMARY:\n"
        report += self._generate_recommendations(validation_results)
        
        return report
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> str:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check cross-validation performance
        if 'cross_validation' in validation_results:
            cv_score = validation_results['cross_validation'].get('mean_cv_score')
            if cv_score is not None:
                if cv_score > 0.8:
                    recommendations.append("✓ Model shows good cross-validation performance")
                elif cv_score > 0.6:
                    recommendations.append("⚠ Model shows moderate cross-validation performance")
                else:
                    recommendations.append("✗ Model shows poor cross-validation performance - consider model refinement")
        
        # Check regime accuracy
        if 'regime_accuracy' in validation_results:
            accuracy = validation_results['regime_accuracy'].get('overall_accuracy')
            if accuracy is not None:
                if accuracy > 0.7:
                    recommendations.append("✓ Regime detection accuracy is acceptable")
                else:
                    recommendations.append("✗ Low regime detection accuracy - consider feature engineering or model selection")
        
        # Check stability
        if 'stability_tests' in validation_results:
            stability = validation_results['stability_tests'].get('score_stability', {})
            std_score = stability.get('std')
            if std_score is not None:
                if std_score < 0.1:
                    recommendations.append("✓ Model shows good stability across time periods")
                else:
                    recommendations.append("⚠ Model shows instability across time periods - consider adaptive approaches")
        
        # Check portfolio performance
        if 'backtest_results' in validation_results:
            sharpe = validation_results['backtest_results'].get('sharpe_ratio')
            if sharpe is not None:
                if sharpe > 1.0:
                    recommendations.append("✓ Portfolio shows strong risk-adjusted returns")
                elif sharpe > 0.5:
                    recommendations.append("⚠ Portfolio shows moderate risk-adjusted returns")
                else:
                    recommendations.append("✗ Portfolio shows poor risk-adjusted returns")
        
        if not recommendations:
            recommendations.append("No specific recommendations available based on current validation results")
        
        return "\n".join(f"  {rec}" for rec in recommendations)


def create_validation_framework() -> ModelValidationFramework:
    """Factory function to create validation framework"""
    return ModelValidationFramework()


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Generate multi-feature data
    n_features = 5
    data = pd.DataFrame(
        np.random.randn(len(dates), n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=dates
    )
    
    # Generate true regimes for testing
    true_regimes = pd.Series(
        np.random.choice([0, 1, 2], size=len(dates), p=[0.4, 0.4, 0.2]),
        index=dates
    )
    
    # Create and test validation framework
    validation_framework = create_validation_framework()
    
    # Test with HMM model
    hmm_model = RegimeDetectionHMM(n_components=3)
    
    validation_results = validation_framework.validate_regime_detection_model(
        hmm_model, data, true_regimes, test_size=0.3, n_splits=3
    )
    
    # Generate report
    report = validation_framework.generate_validation_report(validation_results)
    print(report)