import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from src.optimization.objectives.risk_measures import RiskMeasures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HedgingStrategy(ABC):
    @abstractmethod
    def calculate_hedge_ratio(
        self,
        portfolio_returns: pd.Series,
        hedge_instrument_returns: pd.Series,
        **kwargs
    ) -> float:
        pass
    
    @abstractmethod
    def get_hedge_position(
        self,
        portfolio_value: float,
        hedge_ratio: float,
        **kwargs
    ) -> Dict[str, Any]:
        pass


class MinimumVarianceHedge(HedgingStrategy):
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        
    def calculate_hedge_ratio(
        self,
        portfolio_returns: pd.Series,
        hedge_instrument_returns: pd.Series,
        **kwargs
    ) -> float:
        logger.info("Calculating minimum variance hedge ratio")
        
        common_dates = portfolio_returns.index.intersection(hedge_instrument_returns.index)
        
        if len(common_dates) < self.lookback_window:
            logger.warning(f"Insufficient data for hedge calculation: {len(common_dates)} < {self.lookback_window}")
            return 0.0
        
        recent_dates = common_dates[-self.lookback_window:]
        
        portfolio_recent = portfolio_returns.loc[recent_dates]
        hedge_recent = hedge_instrument_returns.loc[recent_dates]
        
        covariance = np.cov(portfolio_recent, hedge_recent)[0, 1]
        hedge_variance = np.var(hedge_recent)
        
        if hedge_variance == 0:
            return 0.0
        
        hedge_ratio = -covariance / hedge_variance
        
        return hedge_ratio
    
    def get_hedge_position(
        self,
        portfolio_value: float,
        hedge_ratio: float,
        hedge_price: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        hedge_notional = hedge_ratio * portfolio_value
        hedge_units = hedge_notional / hedge_price
        
        return {
            'hedge_ratio': hedge_ratio,
            'hedge_notional': hedge_notional,
            'hedge_units': hedge_units,
            'hedge_price': hedge_price,
            'effectiveness': self._estimate_hedge_effectiveness(hedge_ratio, **kwargs)
        }
    
    def _estimate_hedge_effectiveness(self, hedge_ratio: float, **kwargs) -> float:
        if 'correlation' in kwargs:
            correlation = kwargs['correlation']
            return correlation ** 2
        return abs(hedge_ratio) / (1 + abs(hedge_ratio))


class DeltaNeutralHedge(HedgingStrategy):
    def __init__(self):
        self.delta_sensitivity = {}
        
    def calculate_delta_exposure(
        self,
        portfolio_weights: pd.Series,
        asset_deltas: pd.Series
    ) -> float:
        common_assets = portfolio_weights.index.intersection(asset_deltas.index)
        
        if len(common_assets) == 0:
            return 0.0
        
        portfolio_delta = (portfolio_weights[common_assets] * asset_deltas[common_assets]).sum()
        
        return portfolio_delta
    
    def calculate_hedge_ratio(
        self,
        portfolio_returns: pd.Series,
        hedge_instrument_returns: pd.Series,
        portfolio_delta: Optional[float] = None,
        hedge_delta: float = 1.0,
        **kwargs
    ) -> float:
        if portfolio_delta is None:
            beta = np.cov(portfolio_returns, hedge_instrument_returns)[0, 1] / np.var(hedge_instrument_returns)
            portfolio_delta = beta
        
        hedge_ratio = -portfolio_delta / hedge_delta
        
        return hedge_ratio
    
    def get_hedge_position(
        self,
        portfolio_value: float,
        hedge_ratio: float,
        hedge_price: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        hedge_notional = hedge_ratio * portfolio_value
        hedge_units = hedge_notional / hedge_price
        
        return {
            'hedge_ratio': hedge_ratio,
            'hedge_notional': hedge_notional,
            'hedge_units': hedge_units,
            'hedge_price': hedge_price,
            'delta_neutrality': abs(hedge_ratio) < 0.1
        }


class VaRHedge(HedgingStrategy):
    def __init__(self, confidence_level: float = 0.05, target_var: float = 0.02):
        self.confidence_level = confidence_level
        self.target_var = target_var
        
    def calculate_hedge_ratio(
        self,
        portfolio_returns: pd.Series,
        hedge_instrument_returns: pd.Series,
        **kwargs
    ) -> float:
        logger.info("Calculating VaR-based hedge ratio")
        
        def portfolio_var_objective(hedge_ratio):
            hedged_returns = portfolio_returns + hedge_ratio * hedge_instrument_returns
            current_var = abs(RiskMeasures.value_at_risk(hedged_returns, self.confidence_level))
            return (current_var - self.target_var) ** 2
        
        result = minimize(
            portfolio_var_objective,
            x0=[-0.5],
            bounds=[(-2.0, 0.0)],
            method='L-BFGS-B'
        )
        
        if result.success:
            return result.x[0]
        else:
            logger.warning("VaR hedge optimization failed, using minimum variance hedge")
            return MinimumVarianceHedge().calculate_hedge_ratio(portfolio_returns, hedge_instrument_returns)
    
    def get_hedge_position(
        self,
        portfolio_value: float,
        hedge_ratio: float,
        hedge_price: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        hedge_notional = hedge_ratio * portfolio_value
        hedge_units = hedge_notional / hedge_price
        
        hedged_returns = kwargs.get('hedged_returns')
        if hedged_returns is not None:
            achieved_var = abs(RiskMeasures.value_at_risk(hedged_returns, self.confidence_level))
            var_reduction = 1 - (achieved_var / abs(RiskMeasures.value_at_risk(kwargs.get('unhedged_returns'), self.confidence_level)))
        else:
            var_reduction = None
        
        return {
            'hedge_ratio': hedge_ratio,
            'hedge_notional': hedge_notional,
            'hedge_units': hedge_units,
            'hedge_price': hedge_price,
            'target_var': self.target_var,
            'var_reduction': var_reduction
        }


class TailRiskHedge(HedgingStrategy):
    def __init__(self, tail_threshold: float = 0.05, max_hedge_cost: float = 0.02):
        self.tail_threshold = tail_threshold
        self.max_hedge_cost = max_hedge_cost
        
    def calculate_hedge_ratio(
        self,
        portfolio_returns: pd.Series,
        hedge_instrument_returns: pd.Series,
        option_premium: float = 0.01,
        **kwargs
    ) -> float:
        logger.info("Calculating tail risk hedge ratio")
        
        portfolio_tail_events = portfolio_returns <= portfolio_returns.quantile(self.tail_threshold)
        
        if portfolio_tail_events.sum() == 0:
            return 0.0
        
        tail_portfolio_returns = portfolio_returns[portfolio_tail_events]
        tail_hedge_returns = hedge_instrument_returns[portfolio_tail_events]
        
        if len(tail_hedge_returns) == 0 or tail_hedge_returns.std() == 0:
            return 0.0
        
        hedge_beta_in_tail = np.cov(tail_portfolio_returns, tail_hedge_returns)[0, 1] / np.var(tail_hedge_returns)
        
        cost_adjusted_ratio = hedge_beta_in_tail * (1 - option_premium / self.max_hedge_cost)
        
        return max(-2.0, min(0.0, -abs(cost_adjusted_ratio)))
    
    def get_hedge_position(
        self,
        portfolio_value: float,
        hedge_ratio: float,
        hedge_price: float = 1.0,
        option_premium: float = 0.01,
        **kwargs
    ) -> Dict[str, Any]:
        hedge_notional = hedge_ratio * portfolio_value
        hedge_units = hedge_notional / hedge_price
        hedge_cost = abs(hedge_notional) * option_premium
        
        return {
            'hedge_ratio': hedge_ratio,
            'hedge_notional': hedge_notional,
            'hedge_units': hedge_units,
            'hedge_price': hedge_price,
            'hedge_cost': hedge_cost,
            'cost_ratio': hedge_cost / portfolio_value,
            'tail_protection': abs(hedge_ratio) > 0.1
        }


class RegimeAwareHedge:
    def __init__(self, base_strategies: Dict[str, HedgingStrategy]):
        self.base_strategies = base_strategies
        self.regime_hedge_mappings = {
            0: 'tail_risk',      # Bear market - use tail risk hedge
            1: 'minimum_variance', # Sideways - use minimum variance hedge
            2: 'delta_neutral'    # Bull market - use delta neutral hedge
        }
        
    def calculate_regime_hedge_ratio(
        self,
        current_regime: int,
        regime_probability: float,
        portfolio_returns: pd.Series,
        hedge_instrument_returns: pd.Series,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info(f"Calculating regime-aware hedge for regime {current_regime}")
        
        strategy_name = self.regime_hedge_mappings.get(current_regime, 'minimum_variance')
        strategy = self.base_strategies.get(strategy_name)
        
        if strategy is None:
            logger.warning(f"Strategy {strategy_name} not found, using minimum variance")
            strategy = self.base_strategies.get('minimum_variance', MinimumVarianceHedge())
        
        base_hedge_ratio = strategy.calculate_hedge_ratio(
            portfolio_returns, hedge_instrument_returns, **kwargs
        )
        
        confidence_adjusted_ratio = base_hedge_ratio * regime_probability
        
        return {
            'base_hedge_ratio': base_hedge_ratio,
            'confidence_adjusted_ratio': confidence_adjusted_ratio,
            'regime': current_regime,
            'regime_probability': regime_probability,
            'strategy_used': strategy_name
        }


class DynamicHedgingEngine:
    def __init__(self):
        self.hedging_strategies = {
            'minimum_variance': MinimumVarianceHedge(),
            'delta_neutral': DeltaNeutralHedge(),
            'var_hedge': VaRHedge(),
            'tail_risk': TailRiskHedge()
        }
        
        self.regime_aware_hedge = RegimeAwareHedge(self.hedging_strategies)
        
        self.hedge_history = []
        self.current_hedges = {}
        
    def calculate_optimal_hedges(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: float,
        hedge_instruments: Dict[str, pd.Series],
        current_regime: Optional[int] = None,
        regime_probability: Optional[float] = None,
        hedging_budget: float = 0.05
    ) -> Dict[str, Any]:
        logger.info("Calculating optimal hedge positions")
        
        hedge_recommendations = {}
        total_hedge_cost = 0
        
        for instrument_name, instrument_returns in hedge_instruments.items():
            try:
                if current_regime is not None and regime_probability is not None:
                    hedge_result = self.regime_aware_hedge.calculate_regime_hedge_ratio(
                        current_regime,
                        regime_probability,
                        portfolio_returns,
                        instrument_returns
                    )
                    
                    strategy_name = hedge_result['strategy_used']
                    hedge_ratio = hedge_result['confidence_adjusted_ratio']
                    
                else:
                    strategy_name = 'minimum_variance'
                    strategy = self.hedging_strategies[strategy_name]
                    hedge_ratio = strategy.calculate_hedge_ratio(
                        portfolio_returns, instrument_returns
                    )
                
                hedge_position = self.hedging_strategies[strategy_name].get_hedge_position(
                    portfolio_value, hedge_ratio
                )
                
                hedge_cost = hedge_position.get('hedge_cost', abs(hedge_position['hedge_notional']) * 0.01)
                
                if total_hedge_cost + hedge_cost <= portfolio_value * hedging_budget:
                    hedge_recommendations[instrument_name] = {
                        'strategy': strategy_name,
                        'position': hedge_position,
                        'expected_cost': hedge_cost
                    }
                    
                    total_hedge_cost += hedge_cost
                    
                else:
                    logger.info(f"Skipping hedge {instrument_name} due to budget constraints")
                    
            except Exception as e:
                logger.error(f"Error calculating hedge for {instrument_name}: {e}")
        
        hedge_summary = {
            'recommendations': hedge_recommendations,
            'total_hedge_cost': total_hedge_cost,
            'hedge_budget_utilization': total_hedge_cost / (portfolio_value * hedging_budget),
            'portfolio_value': portfolio_value,
            'number_of_hedges': len(hedge_recommendations)
        }
        
        self.hedge_history.append({
            'timestamp': pd.Timestamp.now(),
            'hedge_summary': hedge_summary,
            'regime': current_regime
        })
        
        return hedge_summary
    
    def execute_hedge_rebalancing(
        self,
        current_hedges: Dict[str, float],
        target_hedges: Dict[str, float],
        transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        logger.info("Executing hedge rebalancing")
        
        rebalancing_trades = {}
        total_transaction_cost = 0
        
        all_instruments = set(current_hedges.keys()) | set(target_hedges.keys())
        
        for instrument in all_instruments:
            current_position = current_hedges.get(instrument, 0)
            target_position = target_hedges.get(instrument, 0)
            
            trade_size = target_position - current_position
            
            if abs(trade_size) > 0.001:
                trade_cost = abs(trade_size) * transaction_cost
                
                rebalancing_trades[instrument] = {
                    'current_position': current_position,
                    'target_position': target_position,
                    'trade_size': trade_size,
                    'trade_cost': trade_cost,
                    'trade_direction': 'buy' if trade_size > 0 else 'sell'
                }
                
                total_transaction_cost += trade_cost
        
        return {
            'trades': rebalancing_trades,
            'total_transaction_cost': total_transaction_cost,
            'number_of_trades': len(rebalancing_trades)
        }
    
    def evaluate_hedge_effectiveness(
        self,
        unhedged_returns: pd.Series,
        hedged_returns: pd.Series,
        hedge_costs: float = 0
    ) -> Dict[str, float]:
        logger.info("Evaluating hedge effectiveness")
        
        unhedged_vol = unhedged_returns.std() * np.sqrt(252)
        hedged_vol = hedged_returns.std() * np.sqrt(252)
        
        vol_reduction = (unhedged_vol - hedged_vol) / unhedged_vol if unhedged_vol > 0 else 0
        
        unhedged_var = abs(RiskMeasures.value_at_risk(unhedged_returns, 0.05))
        hedged_var = abs(RiskMeasures.value_at_risk(hedged_returns, 0.05))
        
        var_reduction = (unhedged_var - hedged_var) / unhedged_var if unhedged_var > 0 else 0
        
        unhedged_max_dd = abs(RiskMeasures.maximum_drawdown(unhedged_returns)[0])
        hedged_max_dd = abs(RiskMeasures.maximum_drawdown(hedged_returns)[0])
        
        dd_reduction = (unhedged_max_dd - hedged_max_dd) / unhedged_max_dd if unhedged_max_dd > 0 else 0
        
        return_impact = hedged_returns.mean() - unhedged_returns.mean()
        
        risk_adjusted_return_improvement = (
            (hedged_returns.mean() / hedged_vol) - (unhedged_returns.mean() / unhedged_vol)
            if hedged_vol > 0 and unhedged_vol > 0 else 0
        )
        
        return {
            'volatility_reduction': vol_reduction,
            'var_reduction': var_reduction,
            'drawdown_reduction': dd_reduction,
            'return_impact': return_impact,
            'risk_adjusted_improvement': risk_adjusted_return_improvement,
            'hedge_cost_ratio': hedge_costs / abs(unhedged_returns.sum()) if unhedged_returns.sum() != 0 else 0,
            'net_benefit': vol_reduction - (hedge_costs / abs(unhedged_returns.sum()) if unhedged_returns.sum() != 0 else 0)
        }
    
    def get_hedging_dashboard(self) -> Dict[str, Any]:
        if not self.hedge_history:
            return {'status': 'No hedging history available'}
        
        recent_hedges = self.hedge_history[-10:]
        
        avg_hedge_cost = np.mean([h['hedge_summary']['total_hedge_cost'] for h in recent_hedges])
        avg_num_hedges = np.mean([h['hedge_summary']['number_of_hedges'] for h in recent_hedges])
        
        return {
            'total_hedge_sessions': len(self.hedge_history),
            'recent_average_hedge_cost': avg_hedge_cost,
            'recent_average_num_hedges': avg_num_hedges,
            'latest_hedge_summary': recent_hedges[-1]['hedge_summary'] if recent_hedges else None,
            'hedging_frequency': len([h for h in recent_hedges if h['hedge_summary']['number_of_hedges'] > 0])
        }