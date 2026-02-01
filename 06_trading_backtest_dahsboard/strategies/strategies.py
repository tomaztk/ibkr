"""
Trading strategies implementation.
Each strategy generates buy/sell signals based on technical indicators.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import ta


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize strategy with parameters.
        
        Args:
            params: Dictionary of strategy-specific parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added signal columns
        """
        pass
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the strategy."""
        return {}
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that required columns exist."""
        required = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required)


class SMAcrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Buy when fast SMA crosses above slow SMA.
    Sell when fast SMA crosses below slow SMA.
    """
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast_period': 10,
            'slow_period': 50
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate SMA crossover signals."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        params = {**self.get_default_params(), **self.params}
        fast_period = params['fast_period']
        slow_period = params['slow_period']
        
        df = df.copy()
        
        # Calculate SMAs
        df['sma_fast'] = df['close'].rolling(window=fast_period).mean()
        df['sma_slow'] = df['close'].rolling(window=slow_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1  # Long
        df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1  # Short
        
        # Position changes (actual trades)
        df['position'] = df['signal'].diff()
        
        # Mark entry/exit points
        df['buy_signal'] = df['position'] == 2  # From -1 to 1
        df['sell_signal'] = df['position'] == -2  # From 1 to -1
        
        return df


class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index Strategy.
    
    Buy when RSI crosses above oversold level.
    Sell when RSI crosses below overbought level.
    """
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 14,
            'overbought': 70,
            'oversold': 30
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI signals."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        params = {**self.get_default_params(), **self.params}
        period = params['period']
        overbought = params['overbought']
        oversold = params['oversold']
        
        df = df.copy()
        
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=period
        ).rsi()
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when RSI was below oversold and crosses above
        df['rsi_prev'] = df['rsi'].shift(1)
        buy_condition = (df['rsi'] > oversold) & (df['rsi_prev'] <= oversold)
        sell_condition = (df['rsi'] < overbought) & (df['rsi_prev'] >= overbought)
        
        # Track position
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Forward fill signals to maintain position
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        df['buy_signal'] = buy_condition
        df['sell_signal'] = sell_condition
        
        return df


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.
    
    Buy when momentum is positive and above threshold.
    Sell when momentum turns negative.
    """
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'threshold': 0.0
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        params = {**self.get_default_params(), **self.params}
        period = params['period']
        threshold = params['threshold']
        
        df = df.copy()
        
        # Calculate momentum (Rate of Change)
        df['momentum'] = df['close'].pct_change(periods=period) * 100
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['momentum'] > threshold, 'signal'] = 1  # Long
        df.loc[df['momentum'] < -threshold, 'signal'] = -1  # Short
        
        # Position changes
        df['position'] = df['signal'].diff()
        
        df['buy_signal'] = (df['signal'] == 1) & (df['signal'].shift(1) != 1)
        df['sell_signal'] = (df['signal'] == -1) & (df['signal'].shift(1) != -1)
        
        return df


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.
    
    Buy when price touches lower band (mean reversion).
    Sell when price touches upper band.
    """
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'std_dev': 2.0
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands signals."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        params = {**self.get_default_params(), **self.params}
        period = params['period']
        std_dev = params['std_dev']
        
        df = df.copy()
        
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=df['close'],
            window=period,
            window_dev=std_dev
        )
        
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Generate signals (mean reversion)
        df['signal'] = 0
        
        # Buy when price crosses below lower band
        buy_condition = (df['close'] <= df['bb_lower'])
        # Sell when price crosses above upper band
        sell_condition = (df['close'] >= df['bb_upper'])
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Forward fill to maintain position
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        df['buy_signal'] = buy_condition & (df['close'].shift(1) > df['bb_lower'].shift(1))
        df['sell_signal'] = sell_condition & (df['close'].shift(1) < df['bb_upper'].shift(1))
        
        return df


class MACDStrategy(BaseStrategy):
    """
    MACD Strategy.
    
    Buy when MACD crosses above signal line.
    Sell when MACD crosses below signal line.
    """
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD signals."""
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        
        params = {**self.get_default_params(), **self.params}
        
        df = df.copy()
        
        # Calculate MACD
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=params['slow_period'],
            window_fast=params['fast_period'],
            window_sign=params['signal_period']
        )
        
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
        df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1
        
        # Position changes
        df['position'] = df['signal'].diff()
        
        df['buy_signal'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['sell_signal'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        return df


# Strategy registry for easy access
STRATEGIES = {
    'SMA Crossover': SMAcrossoverStrategy,
    'RSI': RSIStrategy,
    'Momentum': MomentumStrategy,
    'Bollinger Bands': BollingerBandsStrategy,
    'MACD': MACDStrategy
}


def get_strategy(name: str, params: Dict[str, Any] = None) -> BaseStrategy:
    """
    Get a strategy instance by name.
    
    Args:
        name: Strategy name
        params: Strategy parameters
        
    Returns:
        Strategy instance
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    
    return STRATEGIES[name](params)


def get_available_strategies() -> list:
    """Get list of available strategy names."""
    return list(STRATEGIES.keys())
