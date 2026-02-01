"""Trading strategies module."""
from .strategies import (
    BaseStrategy,
    SMAcrossoverStrategy,
    RSIStrategy,
    MomentumStrategy,
    BollingerBandsStrategy,
    MACDStrategy,
    STRATEGIES,
    get_strategy,
    get_available_strategies
)

__all__ = [
    'BaseStrategy',
    'SMAcrossoverStrategy',
    'RSIStrategy',
    'MomentumStrategy',
    'BollingerBandsStrategy',
    'MACDStrategy',
    'STRATEGIES',
    'get_strategy',
    'get_available_strategies'
]
