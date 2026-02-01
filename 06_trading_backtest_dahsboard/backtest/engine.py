"""
Backtesting engine with realistic simulation including commissions and slippage.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class PositionSide(Enum):
    """Position side enumeration."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    side: PositionSide
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0


@dataclass
class BacktestResult:
    """Container for backtest results."""
    # Core metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Time metrics
    avg_trade_duration: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Time series data
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    
    # Portfolio values
    initial_capital: float = 0.0
    final_capital: float = 0.0


class BacktestEngine:
    """
    Comprehensive backtesting engine with realistic simulation.
    
    Features:
    - Realistic commission modeling
    - Slippage simulation
    - Position sizing (fixed, percent, risk-based)
    - Multiple position management
    - Detailed trade analytics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
        position_sizing: str = 'percent',  # 'fixed', 'percent', 'risk'
        position_size: float = 0.95,  # 95% of capital or fixed shares
        risk_per_trade: float = 0.02,  # 2% risk per trade for risk-based
        allow_short: bool = True,
        allow_fractional: bool = True
    ):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
            position_sizing: Method for position sizing
            position_size: Size parameter (meaning depends on method)
            risk_per_trade: Risk per trade for risk-based sizing
            allow_short: Allow short positions
            allow_fractional: Allow fractional shares
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.risk_per_trade = risk_per_trade
        self.allow_short = allow_short
        self.allow_fractional = allow_fractional
        
        # State variables
        self.reset()
    
    def reset(self):
        """Reset engine state for new backtest."""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        self.entry_time = None
        self.trades: List[Trade] = []
        self.equity_history = []
    
    def _calculate_position_size(
        self, 
        price: float, 
        signal: int,
        atr: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on sizing method.
        
        Args:
            price: Current price
            signal: Trade signal (1 for long, -1 for short)
            atr: Average True Range for risk-based sizing
            
        Returns:
            Number of shares to trade
        """
        if self.position_sizing == 'fixed':
            shares = self.position_size
        
        elif self.position_sizing == 'percent':
            # Use percentage of current capital
            trade_value = self.capital * self.position_size
            shares = trade_value / price
        
        elif self.position_sizing == 'risk':
            # Risk-based position sizing using ATR
            if atr is None or atr == 0:
                atr = price * 0.02  # Default to 2% if no ATR
            
            risk_amount = self.capital * self.risk_per_trade
            shares = risk_amount / atr
        
        else:
            raise ValueError(f"Unknown position sizing method: {self.position_sizing}")
        
        if not self.allow_fractional:
            shares = int(shares)
        
        return abs(shares)
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to price."""
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        return abs(trade_value) * self.commission
    
    def _open_position(
        self, 
        timestamp: pd.Timestamp, 
        price: float, 
        signal: int,
        atr: Optional[float] = None
    ):
        """Open a new position."""
        is_long = signal > 0
        
        if not self.allow_short and not is_long:
            return
        
        # Apply slippage
        execution_price = self._apply_slippage(price, is_long)
        
        # Calculate position size
        shares = self._calculate_position_size(execution_price, signal, atr)
        trade_value = shares * execution_price
        
        # Check if we have enough capital
        if trade_value > self.capital * 0.99:  # Leave 1% buffer
            shares = (self.capital * 0.99) / execution_price
            if not self.allow_fractional:
                shares = int(shares)
            trade_value = shares * execution_price
        
        if shares <= 0:
            return
        
        # Calculate commission
        commission = self._calculate_commission(trade_value)
        slippage_cost = abs(execution_price - price) * shares
        
        # Update state
        self.position = shares if is_long else -shares
        self.position_side = PositionSide.LONG if is_long else PositionSide.SHORT
        self.entry_price = execution_price
        self.entry_time = timestamp
        self.capital -= commission  # Deduct entry commission
        
        # Record trade entry
        self.trades.append(Trade(
            entry_time=timestamp,
            entry_price=execution_price,
            side=self.position_side,
            size=shares,
            commission=commission,
            slippage_cost=slippage_cost
        ))
    
    def _close_position(self, timestamp: pd.Timestamp, price: float):
        """Close the current position."""
        if self.position == 0:
            return
        
        is_long = self.position > 0
        
        # Apply slippage
        execution_price = self._apply_slippage(price, not is_long)
        
        # Calculate trade value and PnL
        trade_value = abs(self.position) * execution_price
        entry_value = abs(self.position) * self.entry_price
        
        if is_long:
            pnl = trade_value - entry_value
        else:
            pnl = entry_value - trade_value
        
        # Calculate commission
        commission = self._calculate_commission(trade_value)
        slippage_cost = abs(execution_price - price) * abs(self.position)
        
        # Update capital
        self.capital += pnl - commission
        
        # Update trade record
        if self.trades:
            trade = self.trades[-1]
            trade.exit_time = timestamp
            trade.exit_price = execution_price
            trade.pnl = pnl - trade.commission - commission
            trade.pnl_pct = trade.pnl / entry_value * 100
            trade.commission += commission
            trade.slippage_cost += slippage_cost
        
        # Reset position
        self.position = 0.0
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        self.entry_time = None
    
    def _calculate_equity(self, price: float) -> float:
        """Calculate current equity including unrealized PnL."""
        equity = self.capital
        
        if self.position != 0:
            position_value = abs(self.position) * price
            entry_value = abs(self.position) * self.entry_price
            
            if self.position > 0:
                unrealized_pnl = position_value - entry_value
            else:
                unrealized_pnl = entry_value - position_value
            
            equity += unrealized_pnl
        
        return equity
    
    def run(
        self, 
        df: pd.DataFrame,
        signal_column: str = 'signal'
    ) -> BacktestResult:
        """
        Run backtest on data with signals.
        
        Args:
            df: DataFrame with OHLCV data and signals
            signal_column: Name of signal column (1=long, -1=short, 0=flat)
            
        Returns:
            BacktestResult with all metrics
        """
        self.reset()
        
        if signal_column not in df.columns:
            raise ValueError(f"Signal column '{signal_column}' not found in data")
        
        # Calculate ATR for risk-based sizing
        if 'atr' not in df.columns:
            df = df.copy()
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
        
        equity_curve = []
        
        for idx, row in df.iterrows():
            current_signal = row[signal_column]
            price = row['close']
            atr = row.get('atr', None)
            
            # Check for position changes
            if self.position == 0 and current_signal != 0:
                # Open new position
                self._open_position(idx, price, current_signal, atr)
            
            elif self.position != 0:
                position_signal = 1 if self.position > 0 else -1
                
                if current_signal == 0 or current_signal != position_signal:
                    # Close position
                    self._close_position(idx, price)
                    
                    # Open opposite position if signal changed
                    if current_signal != 0:
                        self._open_position(idx, price, current_signal, atr)
            
            # Record equity
            equity = self._calculate_equity(price)
            equity_curve.append({'timestamp': idx, 'equity': equity})
        
        # Close any open position at the end
        if self.position != 0:
            self._close_position(df.index[-1], df['close'].iloc[-1])
        
        # Build results
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        return self._calculate_metrics(equity_df['equity'])
    
    def _calculate_metrics(self, equity_curve: pd.Series) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        result = BacktestResult()
        result.initial_capital = self.initial_capital
        result.final_capital = self.capital
        result.equity_curve = equity_curve
        result.trades = self.trades
        
        # Return metrics
        result.total_return = self.capital - self.initial_capital
        result.total_return_pct = (self.capital / self.initial_capital - 1) * 100
        
        # Calculate daily returns
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            
            # Annualized return (assuming minute data, ~252 trading days, ~390 mins/day)
            periods_per_year = 252 * 390  # For minute data
            if len(returns) > 0:
                total_return_decimal = self.capital / self.initial_capital
                years = len(returns) / periods_per_year
                if years > 0:
                    result.annualized_return = (total_return_decimal ** (1/years) - 1) * 100
            
            # Volatility (annualized)
            result.volatility = returns.std() * np.sqrt(periods_per_year) * 100
            
            # Sharpe Ratio (assuming 0 risk-free rate)
            if returns.std() != 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                result.sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
            
            # Drawdown
            rolling_max = equity_curve.cummax()
            drawdown = equity_curve - rolling_max
            drawdown_pct = (drawdown / rolling_max) * 100
            
            result.max_drawdown = abs(drawdown.min())
            result.max_drawdown_pct = abs(drawdown_pct.min())
            result.drawdown_curve = drawdown_pct
            
            # Calmar Ratio
            if result.max_drawdown_pct != 0:
                result.calmar_ratio = result.annualized_return / result.max_drawdown_pct
        
        # Trade statistics
        completed_trades = [t for t in self.trades if t.exit_time is not None]
        result.total_trades = len(completed_trades)
        
        if completed_trades:
            pnls = [t.pnl for t in completed_trades]
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p < 0]
            
            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            result.win_rate = (result.winning_trades / result.total_trades) * 100
            
            result.avg_trade = np.mean(pnls)
            result.avg_win = np.mean(winning) if winning else 0
            result.avg_loss = np.mean(losing) if losing else 0
            result.largest_win = max(pnls) if pnls else 0
            result.largest_loss = min(pnls) if pnls else 0
            
            # Profit factor
            gross_profit = sum(winning) if winning else 0
            gross_loss = abs(sum(losing)) if losing else 0
            result.profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Consecutive wins/losses
            result.max_consecutive_wins = self._max_consecutive(pnls, lambda x: x > 0)
            result.max_consecutive_losses = self._max_consecutive(pnls, lambda x: x < 0)
            
            # Average trade duration
            durations = []
            for t in completed_trades:
                if t.entry_time and t.exit_time:
                    duration = (t.exit_time - t.entry_time).total_seconds() / 60  # in minutes
                    durations.append(duration)
            result.avg_trade_duration = np.mean(durations) if durations else 0
            
            # Costs
            result.total_commission = sum(t.commission for t in completed_trades)
            result.total_slippage = sum(t.slippage_cost for t in completed_trades)
        
        return result
    
    def _max_consecutive(self, values: List[float], condition) -> int:
        """Calculate maximum consecutive values matching condition."""
        max_streak = 0
        current_streak = 0
        
        for v in values:
            if condition(v):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak


class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy parameters.
    """
    
    def __init__(
        self,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        engine_params: Dict[str, Any] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameter ranges to test
            engine_params: Parameters for backtest engine
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.engine_params = engine_params or {}
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        from itertools import product
        
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def optimize(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        metric: str = 'sharpe_ratio'
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Run optimization to find best parameters.
        
        Args:
            df: Full dataset
            train_ratio: Ratio of data for training
            metric: Metric to optimize
            
        Returns:
            Best parameters and their backtest result
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        best_params = None
        best_metric = float('-inf')
        best_result = None
        
        param_combinations = self._generate_param_combinations()
        
        for params in param_combinations:
            # Create strategy with these parameters
            strategy = self.strategy_class(params)
            
            # Generate signals on training data
            train_signals = strategy.generate_signals(train_df.copy())
            
            # Run backtest
            engine = BacktestEngine(**self.engine_params)
            result = engine.run(train_signals)
            
            # Check if this is the best
            metric_value = getattr(result, metric, 0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params
                best_result = result
        
        # Validate on test data
        if best_params:
            strategy = self.strategy_class(best_params)
            test_signals = strategy.generate_signals(test_df.copy())
            engine = BacktestEngine(**self.engine_params)
            validation_result = engine.run(test_signals)
            
            print(f"\nOptimization Results:")
            print(f"Best parameters: {best_params}")
            print(f"Training {metric}: {best_metric:.4f}")
            print(f"Validation {metric}: {getattr(validation_result, metric, 0):.4f}")
        
        return best_params, best_result
