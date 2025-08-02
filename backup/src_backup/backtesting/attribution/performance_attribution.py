import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from src.models.factors.factor_models import FamaFrenchFactorModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BrinsonAttribution:
    def __init__(self):
        self.attribution_results = {}
        
    def calculate_brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_returns: pd.DataFrame,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        logger.info("Calculating Brinson performance attribution")
        
        common_dates = portfolio_weights.index.intersection(benchmark_weights.index)
        common_dates = common_dates.intersection(sector_returns.index)
        
        if len(common_dates) == 0:
            raise ValueError("No common dates found between portfolio, benchmark, and returns data")
        
        results = {
            'allocation_effect': [],
            'selection_effect': [],
            'interaction_effect': [],
            'total_effect': []
        }
        
        for date in common_dates:
            pw = portfolio_weights.loc[date].fillna(0)
            bw = benchmark_weights.loc[date].fillna(0)
            sr = sector_returns.loc[date].fillna(0)
            
            common_sectors = pw.index.intersection(bw.index).intersection(sr.index)
            
            if len(common_sectors) > 0:
                pw_aligned = pw[common_sectors]
                bw_aligned = bw[common_sectors]
                sr_aligned = sr[common_sectors]
                
                benchmark_return = (bw_aligned * sr_aligned).sum()
                
                allocation_effect = (pw_aligned - bw_aligned) * (sr_aligned - benchmark_return)
                selection_effect = bw_aligned * (sr_aligned - sr_aligned.mean())
                interaction_effect = (pw_aligned - bw_aligned) * (sr_aligned - sr_aligned.mean())
                
                total_effect = allocation_effect + selection_effect + interaction_effect
                
                results['allocation_effect'].append(allocation_effect.sum())
                results['selection_effect'].append(selection_effect.sum())
                results['interaction_effect'].append(interaction_effect.sum())
                results['total_effect'].append(total_effect.sum())
            else:
                for key in results.keys():
                    results[key].append(0)
        
        attribution_df = pd.DataFrame(results, index=common_dates)
        
        return {
            'daily_attribution': attribution_df,
            'cumulative_attribution': attribution_df.cumsum(),
            'total_attribution': attribution_df.sum()
        }


class FactorAttribution:
    def __init__(self, factor_model: Optional[FamaFrenchFactorModel] = None):
        self.factor_model = factor_model
        
    def calculate_factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_loadings: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        logger.info("Calculating factor-based performance attribution")
        
        if factor_loadings is None and self.factor_model is not None:
            factor_loadings = self.factor_model.factor_loadings
        
        if factor_loadings is None:
            raise ValueError("Factor loadings must be provided or factor model must be fitted")
        
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        
        if len(common_dates) == 0:
            raise ValueError("No common dates between portfolio returns and factor returns")
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        factor_contributions = pd.DataFrame(index=common_dates, columns=factors_aligned.columns)
        
        for factor in factors_aligned.columns:
            factor_contribution = 0
            
            for asset, loadings in factor_loadings.items():
                if isinstance(loadings, dict) and 'betas' in loadings:
                    if factor in loadings['betas']:
                        beta = loadings['betas'][factor]
                        factor_contribution += beta * factors_aligned[factor]
            
            factor_contributions[factor] = factor_contribution
        
        alpha_contribution = portfolio_aligned - factor_contributions.sum(axis=1)
        
        total_factor_contribution = factor_contributions.sum(axis=1)
        
        return {
            'factor_contributions': factor_contributions,
            'alpha_contribution': alpha_contribution,
            'total_factor_contribution': total_factor_contribution,
            'factor_summary': factor_contributions.sum(),
            'alpha_summary': alpha_contribution.sum(),
            'explained_variance': np.corrcoef(portfolio_aligned, total_factor_contribution)[0, 1] ** 2
        }


class RegimeAttribution:
    def __init__(self):
        self.regime_names = {0: "Bear Market", 1: "Sideways Market", 2: "Bull Market"}
        
    def calculate_regime_attribution(
        self,
        portfolio_returns: pd.Series,
        regime_history: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        logger.info("Calculating regime-based performance attribution")
        
        common_dates = portfolio_returns.index.intersection(regime_history.index)
        
        if len(common_dates) == 0:
            raise ValueError("No common dates between portfolio returns and regime history")
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        regimes_aligned = regime_history.loc[common_dates]
        
        if benchmark_returns is not None:
            benchmark_aligned = benchmark_returns.loc[common_dates]
            excess_returns = portfolio_aligned - benchmark_aligned
        else:
            excess_returns = portfolio_aligned
        
        regime_performance = {}
        regime_statistics = {}
        
        unique_regimes = regimes_aligned.unique()
        unique_regimes = unique_regimes[~pd.isna(unique_regimes)]
        
        for regime in unique_regimes:
            regime_mask = regimes_aligned == regime
            regime_returns = portfolio_aligned[regime_mask]
            regime_excess = excess_returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_performance[regime] = {
                    'total_return': (1 + regime_returns).prod() - 1,
                    'annualized_return': ((1 + regime_returns).prod() ** (252 / len(regime_returns))) - 1,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(regime_returns),
                    'frequency': len(regime_returns) / len(portfolio_aligned),
                    'avg_excess_return': regime_excess.mean(),
                    'excess_volatility': regime_excess.std() * np.sqrt(252),
                    'win_rate': (regime_returns > 0).mean()
                }
                
                regime_statistics[regime] = {
                    'periods': len(regime_returns),
                    'contribution_to_total_return': regime_returns.sum(),
                    'contribution_percentage': regime_returns.sum() / portfolio_aligned.sum() if portfolio_aligned.sum() != 0 else 0
                }
        
        regime_transitions = self._analyze_regime_transitions(regimes_aligned, portfolio_aligned)
        
        return {
            'regime_performance': regime_performance,
            'regime_statistics': regime_statistics,
            'regime_transitions': regime_transitions,
            'regime_names': self.regime_names
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1)
        return drawdown.min()
    
    def _analyze_regime_transitions(
        self,
        regime_history: pd.Series,
        portfolio_returns: pd.Series
    ) -> Dict[str, Any]:
        transitions = []
        
        for i in range(1, len(regime_history)):
            if regime_history.iloc[i] != regime_history.iloc[i-1]:
                transitions.append({
                    'date': regime_history.index[i],
                    'from_regime': regime_history.iloc[i-1],
                    'to_regime': regime_history.iloc[i],
                    'return_on_transition': portfolio_returns.iloc[i]
                })
        
        if transitions:
            transition_df = pd.DataFrame(transitions)
            
            transition_performance = {}
            for _, row in transition_df.iterrows():
                key = f"{int(row['from_regime'])}_to_{int(row['to_regime'])}"
                if key not in transition_performance:
                    transition_performance[key] = []
                transition_performance[key].append(row['return_on_transition'])
            
            transition_stats = {}
            for key, returns in transition_performance.items():
                transition_stats[key] = {
                    'count': len(returns),
                    'avg_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'success_rate': np.mean(np.array(returns) > 0)
                }
            
            return {
                'transitions': transition_df,
                'transition_stats': transition_stats,
                'total_transitions': len(transitions)
            }
        else:
            return {
                'transitions': pd.DataFrame(),
                'transition_stats': {},
                'total_transitions': 0
            }


class ComprehensiveAttribution:
    def __init__(
        self,
        factor_model: Optional[FamaFrenchFactorModel] = None
    ):
        self.brinson = BrinsonAttribution()
        self.factor = FactorAttribution(factor_model)
        self.regime = RegimeAttribution()
        
    def calculate_full_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[pd.DataFrame] = None,
        sector_returns: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        regime_history: Optional[pd.Series] = None,
        factor_loadings: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        logger.info("Calculating comprehensive performance attribution")
        
        attribution_results = {
            'summary': self._calculate_summary_stats(portfolio_returns, benchmark_returns)
        }
        
        if (portfolio_weights is not None and benchmark_weights is not None and 
            sector_returns is not None):
            try:
                attribution_results['brinson'] = self.brinson.calculate_brinson_attribution(
                    portfolio_weights, benchmark_weights, sector_returns,
                    portfolio_returns, benchmark_returns
                )
            except Exception as e:
                logger.error(f"Brinson attribution failed: {e}")
        
        if factor_returns is not None:
            try:
                attribution_results['factor'] = self.factor.calculate_factor_attribution(
                    portfolio_returns, factor_returns, factor_loadings
                )
            except Exception as e:
                logger.error(f"Factor attribution failed: {e}")
        
        if regime_history is not None:
            try:
                attribution_results['regime'] = self.regime.calculate_regime_attribution(
                    portfolio_returns, regime_history, benchmark_returns
                )
            except Exception as e:
                logger.error(f"Regime attribution failed: {e}")
        
        return attribution_results
    
    def _calculate_summary_stats(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        excess_returns = portfolio_returns - benchmark_returns
        
        return {
            'total_excess_return': excess_returns.sum(),
            'annualized_excess_return': excess_returns.mean() * 252,
            'excess_volatility': excess_returns.std() * np.sqrt(252),
            'information_ratio': (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0,
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0, 1],
            'beta': np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns),
            'up_capture': self._calculate_capture_ratio(portfolio_returns, benchmark_returns, 'up'),
            'down_capture': self._calculate_capture_ratio(portfolio_returns, benchmark_returns, 'down')
        }
    
    def _calculate_capture_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        direction: str
    ) -> float:
        if direction == 'up':
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
        
        if mask.sum() == 0:
            return 0
        
        portfolio_filtered = portfolio_returns[mask]
        benchmark_filtered = benchmark_returns[mask]
        
        portfolio_return = (1 + portfolio_filtered).prod() - 1
        benchmark_return = (1 + benchmark_filtered).prod() - 1
        
        return portfolio_return / benchmark_return if benchmark_return != 0 else 0
    
    def generate_attribution_report(
        self,
        attribution_results: Dict[str, Any]
    ) -> str:
        report = "PERFORMANCE ATTRIBUTION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        if 'summary' in attribution_results:
            summary = attribution_results['summary']
            report += "SUMMARY STATISTICS:\n"
            report += f"Total Excess Return: {summary['total_excess_return']:.2%}\n"
            report += f"Annualized Excess Return: {summary['annualized_excess_return']:.2%}\n"
            report += f"Information Ratio: {summary['information_ratio']:.2f}\n"
            report += f"Tracking Error: {summary['tracking_error']:.2%}\n"
            report += f"Beta: {summary['beta']:.2f}\n"
            report += f"Up Capture: {summary['up_capture']:.2%}\n"
            report += f"Down Capture: {summary['down_capture']:.2%}\n\n"
        
        if 'brinson' in attribution_results:
            brinson = attribution_results['brinson']['total_attribution']
            report += "BRINSON ATTRIBUTION:\n"
            report += f"Allocation Effect: {brinson['allocation_effect']:.2%}\n"
            report += f"Selection Effect: {brinson['selection_effect']:.2%}\n"
            report += f"Interaction Effect: {brinson['interaction_effect']:.2%}\n"
            report += f"Total Effect: {brinson['total_effect']:.2%}\n\n"
        
        if 'factor' in attribution_results:
            factor = attribution_results['factor']
            report += "FACTOR ATTRIBUTION:\n"
            report += f"Alpha Contribution: {factor['alpha_summary']:.2%}\n"
            report += f"Explained Variance: {factor['explained_variance']:.2%}\n"
            
            factor_summary = factor['factor_summary']
            for factor_name, contribution in factor_summary.items():
                report += f"{factor_name}: {contribution:.2%}\n"
            report += "\n"
        
        if 'regime' in attribution_results:
            regime = attribution_results['regime']
            report += "REGIME ATTRIBUTION:\n"
            
            for regime_id, performance in regime['regime_performance'].items():
                regime_name = regime['regime_names'].get(regime_id, f"Regime {regime_id}")
                report += f"{regime_name}:\n"
                report += f"  Frequency: {performance['frequency']:.1%}\n"
                report += f"  Annualized Return: {performance['annualized_return']:.2%}\n"
                report += f"  Volatility: {performance['volatility']:.2%}\n"
                report += f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
            
            transitions = regime['regime_transitions']
            report += f"Total Regime Transitions: {transitions['total_transitions']}\n\n"
        
        return report