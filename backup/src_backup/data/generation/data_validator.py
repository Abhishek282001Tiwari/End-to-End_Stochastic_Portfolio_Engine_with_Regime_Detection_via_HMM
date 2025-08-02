#!/usr/bin/env python3
"""
Data Validation Framework

Comprehensive validation system for generated financial data:
- Statistical property validation
- Cross-asset correlation checks
- Data quality and completeness validation
- Realistic market behavior verification
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

@dataclass
class ValidationResult:
    """Result of a validation check"""
    test_name: str
    passed: bool
    score: float  # 0-1 score
    message: str
    details: Dict[str, Any] = None

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    overall_score: float
    validation_results: List[ValidationResult]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class DataValidator:
    """Validate generated financial data for realism and quality"""
    
    def __init__(self):
        """Initialize data validator"""
        self.validation_results = []
        
        # Expected ranges for financial data
        self.expected_ranges = {
            'daily_return_mean': (-0.001, 0.001),  # -25% to +25% annual
            'daily_return_std': (0.005, 0.050),   # 8% to 80% annual volatility
            'annual_sharpe': (-2.0, 3.0),         # Reasonable Sharpe ratio range
            'max_drawdown': (-0.8, 0.0),          # Max 80% drawdown
            'skewness': (-3.0, 3.0),              # Reasonable skewness range
            'kurtosis': (1.0, 20.0),              # Reasonable kurtosis range
            'autocorr_lag1': (-0.3, 0.3),         # First-order autocorrelation
            'price_min': (0.01, float('inf')),    # Prices should be positive
            'volume_min': (1, float('inf'))       # Volume should be positive
        }
        
        # Correlation thresholds
        self.correlation_thresholds = {
            'same_sector_min': 0.3,     # Same sector assets should be correlated
            'same_sector_max': 0.9,     # But not perfectly correlated
            'cross_sector_max': 0.7,    # Cross-sector correlation shouldn't be too high
            'bond_equity_max': 0.3,     # Bonds and equities should have low correlation
            'commodity_equity_max': 0.5  # Commodities and equities moderate correlation
        }
    
    def validate_single_asset(self, data: pd.DataFrame, asset_name: str) -> List[ValidationResult]:
        """Validate a single asset's data"""
        
        results = []
        
        # Basic data structure validation
        results.append(self._validate_data_structure(data, asset_name))
        
        # Price validation
        if 'Close' in data.columns:
            results.extend(self._validate_price_data(data, asset_name))
        
        # Volume validation  
        if 'Volume' in data.columns:
            results.append(self._validate_volume_data(data, asset_name))
        
        # OHLC consistency validation
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            results.append(self._validate_ohlc_consistency(data, asset_name))
        
        # Return distribution validation
        if 'Close' in data.columns:
            results.extend(self._validate_return_distribution(data, asset_name))
        
        # Time series properties validation
        results.extend(self._validate_time_series_properties(data, asset_name))
        
        return results
    
    def validate_portfolio_data(self, data_dict: Dict[str, pd.DataFrame]) -> DataQualityReport:
        """Validate entire portfolio dataset"""
        
        all_results = []
        
        # Validate individual assets
        for asset_name, asset_data in data_dict.items():
            if isinstance(asset_data, pd.DataFrame) and not asset_data.empty:
                asset_results = self.validate_single_asset(asset_data, asset_name)
                all_results.extend(asset_results)
        
        # Cross-asset validation
        if len(data_dict) > 1:
            cross_asset_results = self._validate_cross_asset_relationships(data_dict)
            all_results.extend(cross_asset_results)
        
        # Portfolio-level validation
        portfolio_results = self._validate_portfolio_properties(data_dict)
        all_results.extend(portfolio_results)
        
        # Calculate overall score
        scores = [result.score for result in all_results if result.score is not None]
        overall_score = np.mean(scores) if scores else 0.0
        
        # Generate summary statistics
        summary_stats = self._calculate_summary_statistics(data_dict)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results)
        
        return DataQualityReport(
            overall_score=overall_score,
            validation_results=all_results,
            summary_statistics=summary_stats,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _validate_data_structure(self, data: pd.DataFrame, asset_name: str) -> ValidationResult:
        """Validate basic data structure"""
        
        issues = []
        score = 1.0
        
        # Check for required columns
        required_cols = ['Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            score -= 0.5
        
        # Check for empty data
        if data.empty:
            issues.append("Dataset is empty")
            score = 0.0
        
        # Check for reasonable data length
        if len(data) < 30:
            issues.append(f"Dataset too short: {len(data)} rows")
            score -= 0.3
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.05:  # More than 5% missing
            issues.append(f"High missing value percentage: {missing_pct:.2%}")
            score -= missing_pct
        
        # Check index
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append("Index is not datetime")
            score -= 0.2
        
        message = f"Data structure validation for {asset_name}: " + (
            "PASSED" if score > 0.7 else f"ISSUES: {'; '.join(issues)}"
        )
        
        return ValidationResult(
            test_name=f"data_structure_{asset_name}",
            passed=score > 0.7,
            score=max(0, score),
            message=message,
            details={'issues': issues}
        )
    
    def _validate_price_data(self, data: pd.DataFrame, asset_name: str) -> List[ValidationResult]:
        """Validate price data properties"""
        
        results = []
        
        if 'Close' not in data.columns:
            return results
        
        prices = data['Close'].dropna()
        
        # Price positivity check
        negative_prices = (prices <= 0).sum()
        price_positive_score = 1.0 - (negative_prices / len(prices))
        
        results.append(ValidationResult(
            test_name=f"price_positivity_{asset_name}",
            passed=negative_prices == 0,
            score=price_positive_score,
            message=f"Price positivity for {asset_name}: {negative_prices} negative prices",
            details={'negative_count': negative_prices}
        ))
        
        # Price continuity check (no extreme gaps)
        returns = prices.pct_change().dropna()
        extreme_returns = (np.abs(returns) > 0.5).sum()  # > 50% daily moves
        continuity_score = 1.0 - min(0.5, extreme_returns / len(returns))
        
        results.append(ValidationResult(
            test_name=f"price_continuity_{asset_name}",
            passed=extreme_returns / len(returns) < 0.01,  # Less than 1% extreme moves
            score=continuity_score,
            message=f"Price continuity for {asset_name}: {extreme_returns} extreme moves",
            details={'extreme_moves': extreme_returns, 'total_moves': len(returns)}
        ))
        
        # Price trend realism (not monotonic)
        price_changes = np.diff(prices)
        same_direction_pct = max(
            (price_changes > 0).sum() / len(price_changes),
            (price_changes < 0).sum() / len(price_changes)
        )
        trend_score = 1.0 - max(0, same_direction_pct - 0.7)  # Penalize if > 70% same direction
        
        results.append(ValidationResult(
            test_name=f"price_trend_realism_{asset_name}",
            passed=same_direction_pct < 0.8,
            score=trend_score,
            message=f"Price trend realism for {asset_name}: {same_direction_pct:.1%} same direction",
            details={'same_direction_pct': same_direction_pct}
        ))
        
        return results
    
    def _validate_volume_data(self, data: pd.DataFrame, asset_name: str) -> ValidationResult:
        """Validate volume data"""
        
        if 'Volume' not in data.columns:
            return ValidationResult(
                test_name=f"volume_data_{asset_name}",
                passed=False,
                score=0.0,
                message=f"Volume data missing for {asset_name}"
            )
        
        volumes = data['Volume'].dropna()
        issues = []
        score = 1.0
        
        # Volume positivity
        negative_volumes = (volumes < 0).sum()
        if negative_volumes > 0:
            issues.append(f"{negative_volumes} negative volumes")
            score -= 0.3
        
        # Volume realism (not constant)
        if volumes.std() == 0:
            issues.append("Volume is constant")
            score -= 0.5
        
        # Volume-price relationship (higher volume on big moves)
        if 'Close' in data.columns:
            returns = data['Close'].pct_change().abs()
            corr = returns.corr(volumes)
            if corr < 0.1:  # Expect some positive correlation
                issues.append(f"Weak volume-volatility correlation: {corr:.3f}")
                score -= 0.2
        
        message = f"Volume validation for {asset_name}: " + (
            "PASSED" if not issues else f"ISSUES: {'; '.join(issues)}"
        )
        
        return ValidationResult(
            test_name=f"volume_data_{asset_name}",
            passed=len(issues) == 0,
            score=max(0, score),
            message=message,
            details={'issues': issues}
        )
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame, asset_name: str) -> ValidationResult:
        """Validate OHLC data consistency"""
        
        issues = []
        score = 1.0
        
        for idx, row in data.iterrows():
            open_price, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
            
            # High should be >= max(open, close)
            if high < max(open_price, close):
                issues.append(f"High < max(Open, Close) at {idx}")
                score -= 0.01
            
            # Low should be <= min(open, close)  
            if low > min(open_price, close):
                issues.append(f"Low > min(Open, Close) at {idx}")
                score -= 0.01
            
            # Reasonable spread
            spread = (high - low) / close if close > 0 else 0
            if spread > 0.5:  # More than 50% intraday range
                issues.append(f"Excessive intraday range at {idx}: {spread:.1%}")
                score -= 0.005
        
        # Limit issues reporting
        if len(issues) > 10:
            issues = issues[:10] + [f"... and {len(issues) - 10} more issues"]
        
        message = f"OHLC consistency for {asset_name}: " + (
            "PASSED" if score > 0.9 else f"{len(issues)} issues found"
        )
        
        return ValidationResult(
            test_name=f"ohlc_consistency_{asset_name}",
            passed=score > 0.9,
            score=max(0, score),
            message=message,
            details={'issues': issues[:5]}  # Limit details
        )
    
    def _validate_return_distribution(self, data: pd.DataFrame, asset_name: str) -> List[ValidationResult]:
        """Validate return distribution properties"""
        
        results = []
        
        if 'Close' not in data.columns:
            return results
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 30:
            return results
        
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Mean return validation
        annual_mean = mean_return * 252
        mean_in_range = self.expected_ranges['daily_return_mean'][0] * 252 <= annual_mean <= self.expected_ranges['daily_return_mean'][1] * 252
        
        results.append(ValidationResult(
            test_name=f"return_mean_{asset_name}",
            passed=abs(annual_mean) < 0.5,  # Less than 50% annual return
            score=1.0 - min(1.0, abs(annual_mean) / 1.0),  # Penalize extreme means
            message=f"Return mean for {asset_name}: {annual_mean:.1%} annual",
            details={'annual_mean': annual_mean}
        ))
        
        # Volatility validation
        annual_vol = std_return * np.sqrt(252)
        vol_in_range = self.expected_ranges['daily_return_std'][0] * np.sqrt(252) <= annual_vol <= self.expected_ranges['daily_return_std'][1] * np.sqrt(252)
        
        results.append(ValidationResult(
            test_name=f"return_volatility_{asset_name}",
            passed=0.05 <= annual_vol <= 1.0,  # 5% to 100% annual vol
            score=1.0 if vol_in_range else 0.5,
            message=f"Return volatility for {asset_name}: {annual_vol:.1%} annual",
            details={'annual_volatility': annual_vol}
        ))
        
        # Skewness validation
        skew_reasonable = abs(skew) < 3.0
        
        results.append(ValidationResult(
            test_name=f"return_skewness_{asset_name}",
            passed=skew_reasonable,
            score=1.0 - min(1.0, abs(skew) / 5.0),
            message=f"Return skewness for {asset_name}: {skew:.2f}",
            details={'skewness': skew}
        ))
        
        # Kurtosis validation (fat tails)
        kurt_reasonable = 1.0 <= kurt <= 20.0
        
        results.append(ValidationResult(
            test_name=f"return_kurtosis_{asset_name}",
            passed=kurt_reasonable,
            score=1.0 if kurt_reasonable else 0.5,
            message=f"Return kurtosis for {asset_name}: {kurt:.2f}",
            details={'kurtosis': kurt}
        ))
        
        # Normality test (should NOT be perfectly normal)
        _, normality_p = stats.normaltest(returns)
        normality_score = 1.0 if normality_p < 0.05 else 0.5  # Want to reject normality
        
        results.append(ValidationResult(
            test_name=f"return_non_normality_{asset_name}",
            passed=normality_p < 0.05,
            score=normality_score,
            message=f"Return non-normality for {asset_name}: p-value {normality_p:.4f}",
            details={'normality_p_value': normality_p}
        ))
        
        return results
    
    def _validate_time_series_properties(self, data: pd.DataFrame, asset_name: str) -> List[ValidationResult]:
        """Validate time series properties"""
        
        results = []
        
        if 'Close' not in data.columns:
            return results
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 50:
            return results
        
        # Autocorrelation test
        autocorr_lag1 = returns.autocorr(lag=1)
        autocorr_reasonable = abs(autocorr_lag1) < 0.3
        
        results.append(ValidationResult(
            test_name=f"autocorrelation_{asset_name}",
            passed=autocorr_reasonable,
            score=1.0 - min(1.0, abs(autocorr_lag1) / 0.5),
            message=f"Autocorrelation lag-1 for {asset_name}: {autocorr_lag1:.3f}",
            details={'autocorr_lag1': autocorr_lag1}
        ))
        
        # Volatility clustering (ARCH effects)
        squared_returns = returns ** 2
        vol_clustering = squared_returns.autocorr(lag=1)
        
        results.append(ValidationResult(
            test_name=f"volatility_clustering_{asset_name}",
            passed=vol_clustering > 0.05,  # Should have some clustering
            score=min(1.0, vol_clustering / 0.2) if vol_clustering > 0 else 0.5,
            message=f"Volatility clustering for {asset_name}: {vol_clustering:.3f}",
            details={'volatility_clustering': vol_clustering}
        ))
        
        # Stationarity (returns should be stationary) - simplified test
        try:
            # Simple variance ratio test as proxy for stationarity
            returns_clean = returns.dropna()
            if len(returns_clean) > 100:
                # Calculate variance ratio
                var_short = returns_clean.tail(50).var()
                var_long = returns_clean.var()
                variance_ratio = var_short / var_long if var_long > 0 else 1.0
                
                # Should be close to 1 for stationary series
                stationary = 0.5 <= variance_ratio <= 2.0
                
                results.append(ValidationResult(
                    test_name=f"stationarity_{asset_name}",
                    passed=stationary,
                    score=1.0 if stationary else 0.7,
                    message=f"Stationarity proxy for {asset_name}: variance ratio {variance_ratio:.3f}",
                    details={'variance_ratio': variance_ratio}
                ))
        except Exception:
            pass  # Skip if test fails
        
        return results
    
    def _validate_cross_asset_relationships(self, data_dict: Dict[str, pd.DataFrame]) -> List[ValidationResult]:
        """Validate relationships between assets"""
        
        results = []
        
        # Extract returns for correlation analysis
        returns_data = {}
        
        for asset_name, asset_data in data_dict.items():
            if isinstance(asset_data, pd.DataFrame) and 'Close' in asset_data.columns:
                returns = asset_data['Close'].pct_change().dropna()
                if len(returns) > 100:  # Sufficient data
                    returns_data[asset_name] = returns
        
        if len(returns_data) < 2:
            return results
        
        # Align returns data
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 50:
            return results
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Validate correlation ranges
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                correlations.append((asset1, asset2, correlation))
        
        # Check for perfect correlations (suspicious)
        perfect_corrs = [(a1, a2, c) for a1, a2, c in correlations if abs(c) > 0.98]
        
        results.append(ValidationResult(
            test_name="perfect_correlations",
            passed=len(perfect_corrs) == 0,
            score=1.0 - min(1.0, len(perfect_corrs) / len(correlations)),
            message=f"Perfect correlations found: {len(perfect_corrs)}",
            details={'perfect_correlations': perfect_corrs[:5]}
        ))
        
        # Check correlation distribution
        corr_values = [c for _, _, c in correlations]
        avg_correlation = np.mean(np.abs(corr_values))
        
        results.append(ValidationResult(
            test_name="correlation_distribution",
            passed=0.1 <= avg_correlation <= 0.6,
            score=1.0 if 0.1 <= avg_correlation <= 0.6 else 0.7,
            message=f"Average absolute correlation: {avg_correlation:.3f}",
            details={'avg_abs_correlation': avg_correlation}
        ))
        
        return results
    
    def _validate_portfolio_properties(self, data_dict: Dict[str, pd.DataFrame]) -> List[ValidationResult]:
        """Validate portfolio-level properties"""
        
        results = []
        
        # Asset count validation
        valid_assets = sum(1 for data in data_dict.values() 
                         if isinstance(data, pd.DataFrame) and not data.empty)
        
        results.append(ValidationResult(
            test_name="asset_count",
            passed=valid_assets >= 5,
            score=min(1.0, valid_assets / 10),
            message=f"Valid assets in portfolio: {valid_assets}",
            details={'valid_asset_count': valid_assets}
        ))
        
        # Data coverage validation (same time periods)
        date_ranges = []
        for asset_name, asset_data in data_dict.items():
            if isinstance(asset_data, pd.DataFrame) and not asset_data.empty:
                if isinstance(asset_data.index, pd.DatetimeIndex):
                    date_ranges.append((asset_data.index.min(), asset_data.index.max()))
        
        if date_ranges:
            min_start = max(start for start, _ in date_ranges)
            max_end = min(end for _, end in date_ranges)
            overlap_days = (max_end - min_start).days if max_end > min_start else 0
            
            results.append(ValidationResult(
                test_name="data_coverage",
                passed=overlap_days >= 365,  # At least 1 year overlap
                score=min(1.0, overlap_days / 1000),  # Score based on overlap
                message=f"Data overlap period: {overlap_days} days",
                details={'overlap_days': overlap_days}
            ))
        
        return results
    
    def _calculate_summary_statistics(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate summary statistics for the dataset"""
        
        stats = {
            'total_assets': len(data_dict),
            'asset_types': {},
            'date_range': {},
            'data_quality_metrics': {}
        }
        
        # Asset type breakdown
        for asset_name, asset_data in data_dict.items():
            if isinstance(asset_data, pd.DataFrame) and not asset_data.empty:
                # Classify asset type based on name
                if any(etf in asset_name for etf in ['AGG', 'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'TIP', 'VTEB']):
                    asset_type = 'Bonds'
                elif any(comm in asset_name for comm in ['GLD', 'SLV', 'USO', 'DBA', 'PDBC', 'IAU']):
                    asset_type = 'Commodities'  
                elif any(intl in asset_name for intl in ['VEA', 'VWO', 'EFA', 'EEM']):
                    asset_type = 'International'
                elif any(sect in asset_name for sect in ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP']):
                    asset_type = 'Sectors'
                else:
                    asset_type = 'Equities'
                
                stats['asset_types'][asset_type] = stats['asset_types'].get(asset_type, 0) + 1
        
        # Date range analysis
        all_dates = []
        for asset_data in data_dict.values():
            if isinstance(asset_data, pd.DataFrame) and isinstance(asset_data.index, pd.DatetimeIndex):
                all_dates.extend(asset_data.index.tolist())
        
        if all_dates:
            stats['date_range'] = {
                'start_date': min(all_dates).strftime('%Y-%m-%d'),
                'end_date': max(all_dates).strftime('%Y-%m-%d'),
                'total_days': (max(all_dates) - min(all_dates)).days
            }
        
        # Data quality metrics
        total_missing = 0
        total_records = 0
        
        for asset_data in data_dict.values():
            if isinstance(asset_data, pd.DataFrame):
                total_missing += asset_data.isnull().sum().sum()
                total_records += len(asset_data) * len(asset_data.columns)
        
        stats['data_quality_metrics'] = {
            'missing_data_pct': total_missing / total_records if total_records > 0 else 0,
            'total_records': total_records
        }
        
        return stats
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Analyze failed tests
        failed_tests = [result for result in validation_results if not result.passed]
        
        # Group by test type
        test_types = {}
        for result in failed_tests:
            test_type = result.test_name.split('_')[0]
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)
        
        # Generate specific recommendations
        if 'price' in test_types:
            recommendations.append("Review price generation algorithms for realism and consistency")
        
        if 'return' in test_types:
            recommendations.append("Adjust return distribution parameters to match realistic market behavior")
        
        if 'correlation' in test_types:
            recommendations.append("Review cross-asset correlation structure for realistic relationships")
        
        if 'ohlc' in test_types:
            recommendations.append("Fix OHLC consistency issues in data generation")
        
        if 'volume' in test_types:
            recommendations.append("Improve volume data generation and volume-price relationships")
        
        # Score-based recommendations
        low_score_tests = [result for result in validation_results if result.score < 0.7]
        
        if len(low_score_tests) > len(validation_results) * 0.3:
            recommendations.append("Overall data quality is low - consider regenerating dataset")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable - no major issues detected")
        
        return recommendations
    
    def export_report(self, report: DataQualityReport, output_file: str):
        """Export validation report to file"""
        
        with open(output_file, 'w') as f:
            f.write("# Data Quality Validation Report\n\n")
            f.write(f"Generated on: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Overall Score: {report.overall_score:.2f}/1.00\n\n")
            
            f.write("## Summary Statistics\n")
            for key, value in report.summary_statistics.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            f.write("## Validation Results\n")
            passed_tests = sum(1 for result in report.validation_results if result.passed)
            total_tests = len(report.validation_results)
            f.write(f"Passed: {passed_tests}/{total_tests} tests\n\n")
            
            for result in report.validation_results:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                f.write(f"{status} {result.test_name}: {result.message} (Score: {result.score:.2f})\n")
            
            f.write("\n## Recommendations\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"Validation report exported to {output_file}")


# Example usage
if __name__ == "__main__":
    # Example validation
    validator = DataValidator()
    
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    sample_data = {
        'AAPL': pd.DataFrame({
            'Open': 150 + np.random.normal(0, 5, len(dates)),
            'High': 155 + np.random.normal(0, 5, len(dates)), 
            'Low': 145 + np.random.normal(0, 5, len(dates)),
            'Close': 150 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))),
            'Volume': np.random.lognormal(15, 0.5, len(dates)).astype(int)
        }, index=dates),
        
        'GOOGL': pd.DataFrame({
            'Open': 2500 + np.random.normal(0, 50, len(dates)),
            'High': 2550 + np.random.normal(0, 50, len(dates)),
            'Low': 2450 + np.random.normal(0, 50, len(dates)), 
            'Close': 2500 + np.cumsum(np.random.normal(0.0008, 0.025, len(dates))),
            'Volume': np.random.lognormal(14, 0.4, len(dates)).astype(int)
        }, index=dates)
    }
    
    # Validate portfolio
    report = validator.validate_portfolio_data(sample_data)
    
    # Print summary
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Passed Tests: {sum(1 for r in report.validation_results if r.passed)}/{len(report.validation_results)}")
    
    # Export report
    validator.export_report(report, "validation_report.txt")