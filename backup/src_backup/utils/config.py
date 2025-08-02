import yaml
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel


class DataConfig(BaseModel):
    sources: Dict[str, bool]
    refresh_frequency: str
    lookback_days: int
    database: Dict[str, Any]
    cache: Dict[str, Any]


class HMMConfig(BaseModel):
    n_components: int
    covariance_type: str
    n_iter: int
    random_state: int
    features: list[str]
    training: Dict[str, Any]


class PortfolioConfig(BaseModel):
    optimization: Dict[str, Any]
    constraints: Dict[str, float]
    risk: Dict[str, float]


class BacktestingConfig(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float
    benchmark: str
    metrics: list[str]


class Config(BaseModel):
    data: DataConfig
    hmm: HMMConfig
    portfolio: PortfolioConfig
    backtesting: BacktestingConfig
    logging: Dict[str, Any]
    monitoring: Dict[str, Any]


def load_config(config_path: str = "config/config.yaml") -> Config:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return Config(**config_data)


def get_config() -> Config:
    return load_config()