# Algorithmic Trading Platform — Architecture

> **Stack:** Python · Streamlit · SQLite / PostgreSQL · Alpaca Markets API · Interactive Brokers (TWS / IB Gateway)

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL SERVICES                                                              │
│   🌐 Alpaca Markets API (REST + WebSocket)    🏦 IBKR TWS / IB Gateway          │
└───────────────┬────────────────────────────────────────────┬────────────────────┘
                │ fetch                                      │ TWS API (ib_insync)
                ▼                                            ▼
┌───────────────────────────┐                  ┌────────────────────────────────┐
│  DATA INGESTION LAYER     │                  │  BROKER INTEGRATION LAYER      │
│  AlpacaDataCollector      │                  │  IBKRConnector                 │
│  DatabaseManager          │                  │  OrderManager  FinanceManager  │
└───────────────┬───────────┘                  └──────────────┬─────────────────┘
                │ OHLCV data                                   │ trade signal
                ▼                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  CORE PROCESSING LAYER                                                           │
│   DataAnalyzer ──── SignalGenerator ──── BacktestEngine                          │
└──────────────────────────────────────────────────────────────────────────────────┘
                                  │ results / metrics
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  VISUALIZATION LAYER (Streamlit)                                                 │
│  StreamlitDashboard                                                              │
│  Charts · Signals · Backtest · Orders · Portfolio · Finances                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

```
trading_platform/
│
├── data/
│   ├── alpaca_collector.py        # AlpacaDataCollector
│   └── database_manager.py        # DatabaseManager
│
├── analysis/
│   ├── data_analyzer.py           # DataAnalyzer
│   └── signal_generator.py        # SignalGenerator
│
├── backtesting/
│   └── backtest_engine.py         # BacktestEngine
│
├── broker/
│   ├── ibkr_connector.py          # IBKRConnector
│   ├── order_manager.py           # OrderManager
│   └── finance_manager.py         # FinanceManager
│
├── visualization/
│   └── dashboard.py               # StreamlitDashboard (entry point)
│
├── models/
│   └── order.py                   # Order dataclass / model
│
├── config.py                      # API keys, DB path, constants
└── main.py                        # Orchestrator / scheduler
```

---

## 3. Component Details & Python Classes

---

### 3.1 DATA INGESTION LAYER

#### `AlpacaDataCollector`
**File:** `data/alpaca_collector.py`  
**Purpose:** Connects to Alpaca Markets REST and WebSocket endpoints, pulls historical OHLCV bars and live streaming data.

```python
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import pandas as pd
from datetime import datetime
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL


class AlpacaDataCollector:
    """Fetches historical and live market data from Alpaca Markets."""

    def __init__(self):
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            ALPACA_BASE_URL,
            api_version="v2",
        )
        self._stream: Stream | None = None

    # ── Historical data ────────────────────────────────────────────────────

    def fetch_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",          # e.g. "1Min", "1Hour", "1Day"
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for *symbol* between *start* and *end*."""
        bars = self.api.get_bars(symbol, timeframe, start=start.isoformat(), end=end.isoformat()).df
        bars.index = pd.to_datetime(bars.index)
        return bars

    def fetch_multi_symbol(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch for multiple symbols."""
        return {sym: self.fetch_historical_data(sym, start, end, timeframe) for sym in symbols}

    def get_ticker_info(self, symbol: str) -> dict:
        """Return latest quote, asset metadata for *symbol*."""
        asset = self.api.get_asset(symbol)
        latest = self.api.get_latest_quote(symbol)
        return {"asset": asset._raw, "latest_quote": latest._raw}

    # ── Live / streaming data ──────────────────────────────────────────────

    def fetch_live_bars(self, symbols: list[str], on_bar_callback) -> None:
        """Subscribe to real-time bars; calls *on_bar_callback(bar)* on each update."""
        self._stream = Stream(ALPACA_API_KEY, ALPACA_SECRET_KEY, data_feed="iex")

        async def _bar_handler(bar):
            on_bar_callback(bar)

        for sym in symbols:
            self._stream.subscribe_bars(_bar_handler, sym)
        self._stream.run()

    def stop_stream(self) -> None:
        if self._stream:
            self._stream.stop()
```

---

#### `DatabaseManager`
**File:** `data/database_manager.py`  
**Purpose:** Persists OHLCV bars, signals, and trades. Abstracts SQLite (dev) / PostgreSQL (prod).

```python
import sqlite3
import pandas as pd
from pathlib import Path
from config import DB_PATH


class DatabaseManager:
    """CRUD layer on top of SQLite/PostgreSQL for market data and trade records."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._init_schema()

    # ── Connection lifecycle ───────────────────────────────────────────────

    def connect(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def _init_schema(self) -> None:
        self.connect()
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                open      REAL,  high REAL, low REAL,
                close     REAL,  volume REAL,
                UNIQUE(symbol, timestamp)
            );
            CREATE TABLE IF NOT EXISTS signals (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT, signal_type TEXT,
                price     REAL, timestamp TEXT, strategy TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id   TEXT, symbol TEXT, side TEXT,
                qty        REAL,  filled_price REAL,
                status     TEXT,  timestamp TEXT
            );
        """)
        self.conn.commit()

    # ── OHLCV ─────────────────────────────────────────────────────────────

    def save_ohlcv(self, symbol: str, df: pd.DataFrame) -> None:
        df["symbol"] = symbol
        df.reset_index(inplace=True)
        df.rename(columns={"index": "timestamp"}, inplace=True)
        df.to_sql("ohlcv", self.conn, if_exists="append", index=False)
        self.conn.commit()

    def get_ohlcv(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        query = "SELECT * FROM ohlcv WHERE symbol=? ORDER BY timestamp DESC LIMIT ?"
        return pd.read_sql_query(query, self.conn, params=(symbol, limit))

    # ── Signals ───────────────────────────────────────────────────────────

    def save_signal(self, symbol: str, signal_type: str, price: float,
                    timestamp: str, strategy: str) -> None:
        self.conn.execute(
            "INSERT INTO signals (symbol,signal_type,price,timestamp,strategy) VALUES (?,?,?,?,?)",
            (symbol, signal_type, price, timestamp, strategy),
        )
        self.conn.commit()

    def get_signals(self, symbol: str | None = None, limit: int = 100) -> pd.DataFrame:
        if symbol:
            return pd.read_sql_query(
                "SELECT * FROM signals WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
                self.conn, params=(symbol, limit),
            )
        return pd.read_sql_query(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
            self.conn, params=(limit,),
        )

    # ── Trades ────────────────────────────────────────────────────────────

    def save_trade(self, order_id: str, symbol: str, side: str,
                   qty: float, filled_price: float, status: str, timestamp: str) -> None:
        self.conn.execute(
            "INSERT INTO trades (order_id,symbol,side,qty,filled_price,status,timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            (order_id, symbol, side, qty, filled_price, status, timestamp),
        )
        self.conn.commit()

    def get_trades(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            self.conn, params=(limit,),
        )
```

---

### 3.2 CORE PROCESSING LAYER

#### `DataAnalyzer`
**File:** `analysis/data_analyzer.py`  
**Purpose:** Computes technical indicators (SMA, RSI, MACD, Bollinger Bands) and detects chart patterns on OHLCV data.

```python
import pandas as pd
import numpy as np


class DataAnalyzer:
    """Technical analysis engine — computes indicators on OHLCV DataFrames."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ── Indicators ────────────────────────────────────────────────────────

    def calculate_sma(self, period: int = 20, column: str = "close") -> pd.Series:
        return self.df[column].rolling(window=period).mean()

    def calculate_ema(self, period: int = 20, column: str = "close") -> pd.Series:
        return self.df[column].ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (macd_line, signal_line, histogram)."""
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, macd - sig

    def calculate_bollinger(
        self, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (upper_band, middle_band, lower_band)."""
        mid = self.calculate_sma(period)
        std = self.df["close"].rolling(period).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    def detect_patterns(self) -> dict:
        """Detect simple candlestick patterns. Returns dict of pattern → bool."""
        c = self.df["close"]
        o = self.df["open"]
        patterns = {
            "doji": abs(c - o) / (self.df["high"] - self.df["low"] + 1e-9) < 0.1,
            "bullish_engulfing": (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1)),
        }
        return {k: bool(v.iloc[-1]) for k, v in patterns.items()}

    def generate_report(self) -> dict:
        """Return a summary dict of the latest indicator values."""
        rsi = self.calculate_rsi()
        macd, sig, hist = self.calculate_macd()
        upper, mid, lower = self.calculate_bollinger()
        return {
            "sma_20": round(self.calculate_sma(20).iloc[-1], 4),
            "sma_50": round(self.calculate_sma(50).iloc[-1], 4),
            "rsi_14": round(rsi.iloc[-1], 2),
            "macd": round(macd.iloc[-1], 4),
            "macd_signal": round(sig.iloc[-1], 4),
            "bb_upper": round(upper.iloc[-1], 4),
            "bb_lower": round(lower.iloc[-1], 4),
        }
```

---

#### `SignalGenerator`
**File:** `analysis/signal_generator.py`  
**Purpose:** Applies trading strategies to indicator data and emits BUY / SELL / HOLD signals.

```python
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from analysis.data_analyzer import DataAnalyzer


@dataclass
class Signal:
    symbol: str
    signal_type: str          # "BUY" | "SELL" | "HOLD"
    price: float
    strategy: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0   # 0–1


class SignalGenerator:
    """Converts indicator output into actionable trading signals."""

    def __init__(self, analyzer: DataAnalyzer, symbol: str):
        self.analyzer = analyzer
        self.symbol = symbol
        self._active_signals: list[Signal] = []

    # ── Strategies ────────────────────────────────────────────────────────

    def apply_strategy(self, strategy: str = "rsi_sma") -> Signal:
        """Dispatch to the requested strategy and return a Signal."""
        strategies = {
            "rsi_sma": self._rsi_sma_strategy,
            "macd_cross": self._macd_cross_strategy,
            "bb_reversion": self._bb_reversion_strategy,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        signal = strategies[strategy]()
        self._active_signals.append(signal)
        return signal

    def _rsi_sma_strategy(self) -> Signal:
        rsi = self.analyzer.calculate_rsi().iloc[-1]
        price = float(self.analyzer.df["close"].iloc[-1])
        sma = float(self.analyzer.calculate_sma(20).iloc[-1])
        if rsi < 30 and price > sma:
            sig_type = "BUY"
        elif rsi > 70 and price < sma:
            sig_type = "SELL"
        else:
            sig_type = "HOLD"
        return Signal(self.symbol, sig_type, price, "rsi_sma")

    def _macd_cross_strategy(self) -> Signal:
        macd, sig_line, _ = self.analyzer.calculate_macd()
        price = float(self.analyzer.df["close"].iloc[-1])
        cross_up = (macd.iloc[-1] > sig_line.iloc[-1]) and (macd.iloc[-2] < sig_line.iloc[-2])
        cross_dn = (macd.iloc[-1] < sig_line.iloc[-1]) and (macd.iloc[-2] > sig_line.iloc[-2])
        return Signal(self.symbol, "BUY" if cross_up else ("SELL" if cross_dn else "HOLD"), price, "macd_cross")

    def _bb_reversion_strategy(self) -> Signal:
        upper, _, lower = self.analyzer.calculate_bollinger()
        price = float(self.analyzer.df["close"].iloc[-1])
        if price < float(lower.iloc[-1]):
            return Signal(self.symbol, "BUY", price, "bb_reversion")
        if price > float(upper.iloc[-1]):
            return Signal(self.symbol, "SELL", price, "bb_reversion")
        return Signal(self.symbol, "HOLD", price, "bb_reversion")

    # ── Helpers ───────────────────────────────────────────────────────────

    def generate_buy_signal(self) -> Signal:
        price = float(self.analyzer.df["close"].iloc[-1])
        return Signal(self.symbol, "BUY", price, "manual")

    def generate_sell_signal(self) -> Signal:
        price = float(self.analyzer.df["close"].iloc[-1])
        return Signal(self.symbol, "SELL", price, "manual")

    def filter_signals(self, signal_type: str) -> list[Signal]:
        return [s for s in self._active_signals if s.signal_type == signal_type]

    def get_active_signals(self) -> list[Signal]:
        return self._active_signals
```

---

#### `BacktestEngine`
**File:** `backtesting/backtest_engine.py`  
**Purpose:** Simulates trading on historical data using a chosen strategy. Returns equity curve, trade log, and performance metrics (Sharpe, drawdown, win-rate).

```python
import pandas as pd
import numpy as np
from analysis.signal_generator import SignalGenerator, Signal
from analysis.data_analyzer import DataAnalyzer


class BacktestEngine:
    """Simulates strategy execution on historical OHLCV data."""

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10_000.0):
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self._equity_curve: list[float] = []
        self._trade_log: list[dict] = []
        self._metrics: dict = {}

    def set_strategy(self, strategy: str) -> None:
        self.strategy = strategy

    def run_backtest(self) -> dict:
        """Execute strategy row-by-row. Returns performance metrics dict."""
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        equity_curve = [capital]
        trades = []

        for i in range(50, len(self.df)):            # warm-up window = 50
            window = self.df.iloc[: i + 1].copy()
            analyzer = DataAnalyzer(window)
            gen = SignalGenerator(analyzer, symbol="backtest")
            signal: Signal = gen.apply_strategy(self.strategy)
            price = float(self.df["close"].iloc[i])

            if signal.signal_type == "BUY" and position == 0:
                position = capital / price
                entry_price = price
                capital = 0.0
                trades.append({"type": "BUY", "price": price, "index": i})

            elif signal.signal_type == "SELL" and position > 0:
                capital = position * price
                trades.append({
                    "type": "SELL", "price": price, "index": i,
                    "pnl": capital - position * entry_price,
                })
                position = 0.0

            equity_curve.append(capital + position * price)

        self._equity_curve = equity_curve
        self._trade_log = trades
        self._metrics = self.calculate_metrics()
        return self._metrics

    def calculate_metrics(self) -> dict:
        eq = pd.Series(self._equity_curve)
        returns = eq.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max
        wins = [t for t in self._trade_log if t.get("pnl", 0) > 0]
        return {
            "final_capital": round(eq.iloc[-1], 2),
            "total_return_pct": round((eq.iloc[-1] / self.initial_capital - 1) * 100, 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
            "total_trades": len(self._trade_log) // 2,
            "win_rate_pct": round(len(wins) / max(1, len(self._trade_log) // 2) * 100, 1),
        }

    def get_equity_curve(self) -> list[float]:
        return self._equity_curve

    def get_trade_log(self) -> list[dict]:
        return self._trade_log
```

---

### 3.3 BROKER INTEGRATION LAYER

#### `IBKRConnector`
**File:** `broker/ibkr_connector.py`  
**Purpose:** Wraps `ib_insync` to communicate with IBKR TWS / IB Gateway. Handles connection lifecycle, order placement, and market data subscription.

```python
from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID


class IBKRConnector:
    """Low-level connector to Interactive Brokers via ib_insync."""

    def __init__(self):
        self.ib = IB()

    def connect(self) -> None:
        self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)

    def disconnect(self) -> None:
        self.ib.disconnect()

    def place_order(self, symbol: str, side: str, qty: int,
                    order_type: str = "MKT", limit_price: float | None = None):
        """Place a buy or sell order. Returns the Trade object."""
        contract = Stock(symbol, "SMART", "USD")
        if order_type == "MKT":
            order = MarketOrder(side.upper(), qty)
        else:
            order = LimitOrder(side.upper(), qty, limit_price)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        return trade

    def cancel_order(self, trade) -> None:
        self.ib.cancelOrder(trade.order)

    def get_account_info(self) -> dict:
        account_values = self.ib.accountValues()
        return {v.tag: v.value for v in account_values}

    def subscribe_market_data(self, symbol: str):
        contract = Stock(symbol, "SMART", "USD")
        ticker = self.ib.reqMktData(contract)
        return ticker

    def get_open_orders(self) -> list:
        return self.ib.openOrders()

    def get_positions(self) -> list:
        return self.ib.positions()
```

---

#### `OrderManager`
**File:** `broker/order_manager.py`  
**Purpose:** High-level order management — translates Signals into IBKR orders, tracks status, exposes order history.

```python
from dataclasses import dataclass, field
from datetime import datetime
from analysis.signal_generator import Signal
from broker.ibkr_connector import IBKRConnector
from data.database_manager import DatabaseManager


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    order_type: str = "MKT"
    limit_price: float | None = None
    status: str = "PENDING"
    order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OrderManager:
    """Translates signals to IBKR orders and tracks their lifecycle."""

    def __init__(self, connector: IBKRConnector, db: DatabaseManager):
        self.connector = connector
        self.db = db
        self._order_history: list[Order] = []

    def submit_order(self, signal: Signal, qty: int = 1) -> Order | None:
        if signal.signal_type == "HOLD":
            return None
        order = Order(
            symbol=signal.symbol,
            side=signal.signal_type,
            qty=qty,
        )
        trade = self.connector.place_order(order.symbol, order.side, order.qty)
        order.order_id = str(trade.order.orderId)
        order.status = trade.orderStatus.status
        self._order_history.append(order)
        self.db.save_trade(
            order.order_id, order.symbol, order.side, order.qty,
            signal.price, order.status, order.timestamp.isoformat(),
        )
        return order

    def check_orders(self) -> list[dict]:
        open_orders = self.connector.get_open_orders()
        return [{"orderId": o.orderId, "status": o.orderState.status} for o in open_orders]

    def cancel_order(self, order: Order) -> None:
        # Re-fetch the trade by orderId and cancel
        open_trades = self.connector.ib.openTrades()
        for t in open_trades:
            if str(t.order.orderId) == order.order_id:
                self.connector.cancel_order(t)
                order.status = "CANCELLED"
                break

    def get_history(self) -> list[Order]:
        return self._order_history
```

---

#### `FinanceManager`
**File:** `broker/finance_manager.py`  
**Purpose:** Queries account financials from IBKR — balance, positions, P&L, equity.

```python
from broker.ibkr_connector import IBKRConnector


class FinanceManager:
    """Reads account financial data from IBKR."""

    def __init__(self, connector: IBKRConnector):
        self.connector = connector

    def get_balance(self) -> float:
        info = self.connector.get_account_info()
        return float(info.get("CashBalance", 0))

    def get_positions(self) -> list[dict]:
        positions = self.connector.get_positions()
        return [
            {
                "symbol": p.contract.symbol,
                "qty": p.position,
                "avg_cost": p.avgCost,
                "market_value": p.position * p.marketPrice if hasattr(p, "marketPrice") else None,
            }
            for p in positions
        ]

    def get_equity(self) -> float:
        info = self.connector.get_account_info()
        return float(info.get("NetLiquidation", 0))

    def calc_pnl(self) -> dict:
        info = self.connector.get_account_info()
        return {
            "realized_pnl": float(info.get("RealizedPnL", 0)),
            "unrealized_pnl": float(info.get("UnrealizedPnL", 0)),
        }

    def get_full_summary(self) -> dict:
        return {
            "balance": self.get_balance(),
            "equity": self.get_equity(),
            "positions": self.get_positions(),
            **self.calc_pnl(),
        }
```

---

### 3.4 VISUALIZATION LAYER

#### `StreamlitDashboard`
**File:** `visualization/dashboard.py`  
**Purpose:** Streamlit multi-page application. Reads from DatabaseManager and the live IBKR/Alpaca state to render interactive charts and tables.

```python
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from data.database_manager import DatabaseManager
from broker.finance_manager import FinanceManager
from broker.order_manager import OrderManager
from backtesting.backtest_engine import BacktestEngine


class StreamlitDashboard:
    """Streamlit-based trading platform dashboard."""

    def __init__(self, db: DatabaseManager, finance: FinanceManager, order_mgr: OrderManager):
        self.db = db
        self.finance = finance
        self.order_mgr = order_mgr

    def run(self) -> None:
        st.set_page_config(page_title="Trading Platform", layout="wide")
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", [
            "📈 Price Charts",
            "🔔 Signals",
            "🔁 Backtest",
            "📋 Orders",
            "💰 Finances",
            "📊 Portfolio",
        ])
        pages = {
            "📈 Price Charts": self.render_price_charts,
            "🔔 Signals": self.render_signals,
            "🔁 Backtest": self.render_backtest_results,
            "📋 Orders": self.render_orders,
            "💰 Finances": self.render_finances,
            "📊 Portfolio": self.render_portfolio,
        }
        pages[page]()

    def render_price_charts(self) -> None:
        st.title("📈 Price Charts")
        symbol = st.text_input("Symbol", value="AAPL")
        df = self.db.get_ohlcv(symbol)
        if df.empty:
            st.warning("No data found. Run the data collector first.")
            return
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
        )])
        fig.update_layout(title=symbol, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    def render_signals(self) -> None:
        st.title("🔔 Signals")
        df = self.db.get_signals()
        st.dataframe(df, use_container_width=True)

    def render_backtest_results(self) -> None:
        st.title("🔁 Backtesting Results")
        symbol = st.text_input("Symbol", value="AAPL", key="bt_sym")
        strategy = st.selectbox("Strategy", ["rsi_sma", "macd_cross", "bb_reversion"])
        if st.button("Run Backtest"):
            df = self.db.get_ohlcv(symbol)
            engine = BacktestEngine(df)
            engine.set_strategy(strategy)
            metrics = engine.run_backtest()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{metrics['total_return_pct']}%")
            col2.metric("Sharpe Ratio", metrics['sharpe_ratio'])
            col3.metric("Max Drawdown", f"{metrics['max_drawdown_pct']}%")
            st.line_chart(engine.get_equity_curve())
            st.dataframe(pd.DataFrame(engine.get_trade_log()))

    def render_orders(self) -> None:
        st.title("📋 Orders")
        open_orders = self.order_mgr.check_orders()
        st.subheader("Open Orders")
        st.json(open_orders)
        st.subheader("Trade History")
        st.dataframe(self.db.get_trades(), use_container_width=True)

    def render_finances(self) -> None:
        st.title("💰 Finances")
        summary = self.finance.get_full_summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cash Balance", f"${summary['balance']:,.2f}")
        col2.metric("Net Equity", f"${summary['equity']:,.2f}")
        col3.metric("Realized P&L", f"${summary['realized_pnl']:,.2f}")
        col4.metric("Unrealized P&L", f"${summary['unrealized_pnl']:,.2f}")
        st.subheader("Positions")
        st.dataframe(pd.DataFrame(summary["positions"]), use_container_width=True)

    def render_portfolio(self) -> None:
        st.title("📊 Portfolio Overview")
        positions = self.finance.get_positions()
        if positions:
            df = pd.DataFrame(positions)
            st.bar_chart(df.set_index("symbol")["market_value"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No open positions.")


# ── Entry point ────────────────────────────────────────────────────────────────
# Run with:  streamlit run visualization/dashboard.py

if __name__ == "__main__":
    db = DatabaseManager()
    # finance = FinanceManager(IBKRConnector())   # connect IBKR first
    # order_mgr = OrderManager(connector, db)
    # app = StreamlitDashboard(db, finance, order_mgr)
    # app.run()
    pass
```

---

## 4. Configuration

```python
# config.py

# Alpaca
ALPACA_API_KEY  = "YOUR_ALPACA_KEY"
ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"   # use paper for testing

# Database
DB_PATH = "trading.db"   # swap to PostgreSQL DSN for production

# IBKR
IBKR_HOST      = "127.0.0.1"
IBKR_PORT      = 7497     # 7497 = TWS paper | 4002 = IB Gateway paper
IBKR_CLIENT_ID = 1
```

---

## 5. Technology Stack

| Layer | Library |
|---|---|
| Alpaca data | `alpaca-trade-api` |
| IBKR broker | `ib_insync` |
| Database (dev) | `sqlite3` (stdlib) |
| Database (prod) | `psycopg2` + PostgreSQL |
| Technical analysis | `pandas`, `numpy` |
| Visualization | `streamlit`, `plotly` |
| Scheduling | `APScheduler` or `schedule` |
| Testing | `pytest` |

---

## 6. Data Flow Summary

```
Alpaca API
  └─▶ AlpacaDataCollector.fetch_historical_data()
        └─▶ DatabaseManager.save_ohlcv()
              ├─▶ DataAnalyzer (indicators)
              │     └─▶ SignalGenerator.apply_strategy()
              │           ├─▶ OrderManager.submit_order()
              │           │     └─▶ IBKRConnector.place_order()  ──▶  IBKR
              │           └─▶ StreamlitDashboard.render_signals()
              └─▶ BacktestEngine.run_backtest()
                    └─▶ StreamlitDashboard.render_backtest_results()

FinanceManager / OrderManager
  └─▶ IBKRConnector  ◀──▶  IBKR TWS
  └─▶ StreamlitDashboard (finances + orders pages)
```
