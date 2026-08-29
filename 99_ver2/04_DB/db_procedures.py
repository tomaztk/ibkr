"""
db_procedures.py — Trading System Database Procedures
======================================================
SQLite3 has no stored procedures. This module is the equivalent:
a collection of focused functions that encapsulate every DB operation
used by the trading pipeline.

Import pattern:
    from db_procedures import DB
    with DB("trading.db") as db:
        db.upsert_bars("AAPL", "daily", df)
        signals = db.get_open_signals()
        db.place_order(account_id=1, symbol="AAPL", side="BUY", qty=10,
                       signal_id=signals[0]["id"])
"""

import json
import sqlite3
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

# ---------------------------------------------------------------------------
# Connection wrapper
# ---------------------------------------------------------------------------

class DB:
    """Context-manager wrapper that enforces WAL + FK pragma on every connection."""

    def __init__(self, path: Union[str, Path] = "trading.db"):
        self.path = str(path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── context manager ──────────────────────────────────────────────────
    def __enter__(self) -> "DB":
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
            self._conn.close()
            self._conn = None
        return False     # re-raise exceptions

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DB not open — use 'with DB(...) as db:'")
        return self._conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── generic helpers ──────────────────────────────────────────────────
    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def execute(self, sql: str, params: tuple = ()) -> int:
        cur = self.conn.execute(sql, params)
        return cur.lastrowid or cur.rowcount

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def upsert_symbol(self, symbol: str, name: str = "", exchange: str = "",
                       asset_type: str = "STOCK") -> None:
        """Ensure a symbol exists in the master table."""
        self.conn.execute("""
            INSERT OR IGNORE INTO symbols (symbol, name, exchange, asset_type)
            VALUES (?, ?, ?, ?)
        """, (symbol, name, exchange, asset_type))

    def upsert_bars(self, symbol: str, resolution: str,
                    rows: list[dict]) -> int:
        """
        Bulk upsert OHLCV bars.

        rows: list of dicts with keys:
            timestamp, open, high, low, close, volume, [vwap], [trade_count]
        Returns number of rows inserted/replaced.
        """
        table = f"bars_{resolution}"
        sql = f"""
            INSERT OR REPLACE INTO {table}
                (symbol, timestamp, open, high, low, close, volume, vwap, trade_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (symbol, r["timestamp"],
             r["open"], r["high"], r["low"], r["close"],
             r.get("volume"), r.get("vwap"), r.get("trade_count"))
            for r in rows
        ]
        cur = self.conn.executemany(sql, data)
        return cur.rowcount

    def get_bars(self, symbol: str, resolution: str,
                 start: Optional[str] = None, end: Optional[str] = None,
                 limit: Optional[int] = None) -> list[dict]:
        """Return OHLCV bars, newest first."""
        table = f"bars_{resolution}"
        filters, params = [], []
        filters.append("symbol = ?");  params.append(symbol)
        if start: filters.append("timestamp >= ?"); params.append(start)
        if end:   filters.append("timestamp <= ?"); params.append(end)
        where = " AND ".join(filters)
        lim = f"LIMIT {limit}" if limit else ""
        return self.fetchall(
            f"SELECT * FROM {table} WHERE {where} ORDER BY timestamp DESC {lim}",
            tuple(params))

    # =========================================================================
    # TECHNICAL ANALYSIS
    # =========================================================================

    def upsert_indicators(self, symbol: str, resolution: str,
                          rows: list[dict]) -> int:
        """Bulk upsert computed indicator rows."""
        table = f"indicators_{resolution}"
        cols  = ["symbol","timestamp","sma_20","sma_50","sma_200",
                 "rsi_14","macd","macd_signal","macd_hist",
                 "bb_upper","bb_middle","bb_lower","bb_width","bb_pct"]
        ph  = ",".join(["?"]*len(cols))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({ph})"
        data = [tuple(r.get(c, (symbol if c=="symbol" else None)) for c in cols)
                for r in rows]
        cur = self.conn.executemany(sql, data)
        return cur.rowcount

    def upsert_patterns(self, symbol: str, resolution: str,
                        rows: list[dict]) -> int:
        """Bulk insert detected patterns (ignore duplicates)."""
        table = f"patterns_{resolution}"
        sql = f"""
            INSERT OR IGNORE INTO {table} (symbol, timestamp, pattern, direction, confidence)
            VALUES (?, ?, ?, ?, ?)
        """
        data = [(symbol, r["timestamp"], r["pattern"], r["direction"], r["confidence"])
                for r in rows]
        cur = self.conn.executemany(sql, data)
        return cur.rowcount

    def get_latest_indicators(self, symbol: str, resolution: str) -> Optional[dict]:
        """Return the most recent indicator row for a symbol."""
        view = f"v_latest_indicators_{resolution}"
        return self.fetchone(f"SELECT * FROM {view} WHERE symbol = ?", (symbol,))

    def get_recent_patterns(self, symbol: str, resolution: str,
                            days: int = 5) -> list[dict]:
        """Return patterns detected in the last N days."""
        table = f"patterns_{resolution}"
        return self.fetchall(
            f"SELECT * FROM {table} WHERE symbol=? "
            f"AND timestamp >= datetime('now','-{days} days') "
            f"ORDER BY timestamp DESC",
            (symbol,))

    # =========================================================================
    # STRATEGY ENGINE
    # =========================================================================

    def get_active_strategies(self, resolution: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM strategies WHERE active=1"
        params: tuple = ()
        if resolution:
            sql += " AND resolution=?"
            params = (resolution,)
        return self.fetchall(sql, params)

    def create_signal(self, symbol: str, strategy_id: int, timestamp: str,
                      resolution: str, signal_type: str, strength: float = 1.0,
                      reason: Optional[dict] = None) -> Optional[int]:
        """
        Insert a new signal. Returns the new row id, or None if duplicate.

        signal_type: 'BUY' | 'SELL' | 'HOLD'
        strength:    0.0 – 1.0
        reason:      dict  e.g. {"rsi":28.4,"patterns":["Hammer"],"macd":"bullish"}
        """
        try:
            cur = self.conn.execute("""
                INSERT INTO signals
                    (symbol, strategy_id, timestamp, resolution,
                     signal_type, strength, reason_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, strategy_id, timestamp, resolution,
                  signal_type, strength, json.dumps(reason or {})))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None   # duplicate — already recorded

    def get_open_signals(self, symbol: Optional[str] = None,
                         signal_type: Optional[str] = None) -> list[dict]:
        """Return all unacted-on signals, newest first."""
        sql    = "SELECT * FROM v_open_signals WHERE 1=1"
        params = []
        if symbol:      sql += " AND symbol=?";      params.append(symbol)
        if signal_type: sql += " AND signal_type=?"; params.append(signal_type)
        return self.fetchall(sql, tuple(params))

    def check_trigger_rules(self, strategy_id: int,
                            indicators: dict, patterns: list[str]) -> list[dict]:
        """
        Evaluate trigger_rules for a given strategy against live indicator values.
        Returns list of rules that fired.

        indicators: dict from get_latest_indicators()
        patterns:   list of pattern names detected at latest bar
        """
        rules  = self.fetchall(
            "SELECT * FROM trigger_rules WHERE strategy_id=? AND active=1",
            (strategy_id,))
        fired  = []
        for rule in rules:
            params = json.loads(rule["params_json"] or "{}")
            ct = rule["condition_type"]

            if ct == "RSI_OVERSOLD":
                thresh = params.get("threshold", 30)
                if indicators.get("rsi_14") is not None and \
                   indicators["rsi_14"] < thresh:
                    fired.append(rule)

            elif ct == "RSI_OVERBOUGHT":
                thresh = params.get("threshold", 70)
                if indicators.get("rsi_14") is not None and \
                   indicators["rsi_14"] > thresh:
                    fired.append(rule)

            elif ct == "MACD_CROSS":
                direction = params.get("direction","bullish")
                hist = indicators.get("macd_hist", 0) or 0
                if direction == "bullish" and hist > 0:
                    fired.append(rule)
                elif direction == "bearish" and hist < 0:
                    fired.append(rule)

            elif ct == "GOLDEN_CROSS":
                sma50  = indicators.get("sma_50")
                sma200 = indicators.get("sma_200")
                if sma50 and sma200 and sma50 > sma200:
                    fired.append(rule)

            elif ct == "DEATH_CROSS":
                sma50  = indicators.get("sma_50")
                sma200 = indicators.get("sma_200")
                if sma50 and sma200 and sma50 < sma200:
                    fired.append(rule)

            elif ct == "PATTERN_MATCH":
                required = params.get("patterns", [])
                if any(p in patterns for p in required):
                    fired.append(rule)

            elif ct == "BB_SQUEEZE":
                bw = indicators.get("bb_width")
                thresh = params.get("max_width", 0.05)
                if bw is not None and bw < thresh:
                    fired.append(rule)

        return fired

    # =========================================================================
    # ORDERS & EXECUTION
    # =========================================================================

    def place_order(self, account_id: int, symbol: str, side: str,
                    quantity: float, order_type: str = "MARKET",
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    signal_id: Optional[int] = None) -> int:
        """
        Create a new order row (status='pending').
        Call update_order_status() after submitting to IBKR.
        Returns the new order id.
        """
        cur = self.conn.execute("""
            INSERT INTO orders
                (account_id, symbol, signal_id, side, order_type,
                 quantity, limit_price, stop_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (account_id, symbol, signal_id, side, order_type,
              quantity, limit_price, stop_price))
        order_id = cur.lastrowid
        self.log_event("ORDER_CREATED", "orders", symbol,
                       f"Order #{order_id} created: {side} {quantity} {symbol}",
                       {"order_id": order_id, "side": side, "qty": quantity})
        return order_id

    def update_order_status(self, order_id: int, status: str,
                            ibkr_order_id: Optional[str] = None,
                            reject_reason: Optional[str] = None) -> None:
        """
        Update an order's status and optionally the IBKR order ID.
        Triggers handle event_log automatically.
        """
        ts_field = {
            "submitted": "submitted_at",
            "filled":    "filled_at",
            "cancelled": "cancelled_at",
        }.get(status)

        parts  = ["status=?"]
        params: list = [status]

        if ibkr_order_id:
            parts.append("ibkr_order_id=?"); params.append(ibkr_order_id)
        if reject_reason:
            parts.append("reject_reason=?"); params.append(reject_reason)
        if ts_field:
            parts.append(f"{ts_field}=?");   params.append(self._now())

        params.append(order_id)
        self.conn.execute(
            f"UPDATE orders SET {', '.join(parts)} WHERE id=?", tuple(params))

    def record_execution(self, order_id: int, ibkr_exec_id: str,
                         price: float, quantity: float,
                         commission: float = 0.0, exchange: str = "") -> int:
        """Record a fill/partial fill from IBKR."""
        cur = self.conn.execute("""
            INSERT OR IGNORE INTO executions
                (order_id, ibkr_exec_id, price, quantity, commission,
                 exchange, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (order_id, ibkr_exec_id, price, quantity, commission,
              exchange, self._now()))
        return cur.lastrowid

    def get_open_orders(self, account_id: Optional[int] = None,
                        symbol: Optional[str] = None) -> list[dict]:
        sql    = "SELECT * FROM v_open_orders WHERE 1=1"
        params = []
        if account_id: sql += " AND account_id=?"; params.append(account_id)
        if symbol:     sql += " AND symbol=?";     params.append(symbol)
        return self.fetchall(sql, tuple(params))

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def upsert_position(self, account_id: int, symbol: str,
                        quantity: float, avg_cost: float,
                        current_price: Optional[float] = None,
                        realized_pnl_delta: float = 0.0) -> None:
        """
        Create or update a position.  quantity=0 means flat (but keeps the row).
        """
        existing = self.fetchone(
            "SELECT * FROM positions WHERE account_id=? AND symbol=?",
            (account_id, symbol))

        if existing is None:
            self.conn.execute("""
                INSERT INTO positions
                    (account_id, symbol, quantity, avg_cost, current_price,
                     unrealized_pnl, realized_pnl, opened_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (account_id, symbol, quantity, avg_cost, current_price,
                  (current_price - avg_cost) * quantity if current_price else None,
                  0.0, self._now()))
        else:
            unreal = ((current_price or existing["current_price"] or avg_cost)
                      - avg_cost) * quantity
            self.conn.execute("""
                UPDATE positions SET
                    quantity=?, avg_cost=?, current_price=?,
                    unrealized_pnl=?,
                    realized_pnl = realized_pnl + ?
                WHERE account_id=? AND symbol=?
            """, (quantity, avg_cost, current_price, unreal,
                  realized_pnl_delta, account_id, symbol))

    def close_position(self, account_id: int, symbol: str,
                       exit_price: float,
                       entry_order_id: Optional[int] = None,
                       exit_order_id: Optional[int] = None) -> Optional[int]:
        """
        Mark a position as closed, record in trades_history.
        Returns the new trades_history row id.
        """
        pos = self.fetchone(
            "SELECT * FROM positions WHERE account_id=? AND symbol=? AND quantity!=0",
            (account_id, symbol))
        if pos is None:
            return None

        direction  = "LONG" if pos["quantity"] > 0 else "SHORT"
        gross_pnl  = (exit_price - pos["avg_cost"]) * abs(pos["quantity"])
        if direction == "SHORT":
            gross_pnl = -gross_pnl
        commission = 0.001 * exit_price * abs(pos["quantity"])  # placeholder
        net_pnl    = gross_pnl - commission
        cost_basis = pos["avg_cost"] * abs(pos["quantity"])
        pnl_pct    = net_pnl / cost_basis if cost_basis else 0.0

        cur = self.conn.execute("""
            INSERT INTO trades_history
                (account_id, symbol, entry_order_id, exit_order_id,
                 direction, entry_price, exit_price, quantity,
                 gross_pnl, commission, net_pnl, net_pnl_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (account_id, symbol, entry_order_id, exit_order_id,
              direction, pos["avg_cost"], exit_price, abs(pos["quantity"]),
              gross_pnl, commission, net_pnl, pnl_pct))

        # Flatten the position
        self.conn.execute(
            "UPDATE positions SET quantity=0, unrealized_pnl=0, "
            "realized_pnl=realized_pnl+? WHERE account_id=? AND symbol=?",
            (net_pnl, account_id, symbol))

        return cur.lastrowid

    def get_live_positions(self, account_id: Optional[int] = None) -> list[dict]:
        sql    = "SELECT * FROM v_live_positions WHERE 1=1"
        params = []
        if account_id: sql += " AND account_id=?"; params.append(account_id)
        return self.fetchall(sql, tuple(params))

    # =========================================================================
    # PORTFOLIO SNAPSHOTS
    # =========================================================================

    def take_portfolio_snapshot(self, account_id: int, total_value: float,
                                cash: float, unrealized_pnl: float = 0.0,
                                realized_pnl_day: float = 0.0) -> None:
        """Record end-of-day portfolio state."""
        today = self._now()[:10]
        self.conn.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
                (account_id, snapshot_date, total_value, cash, invested,
                 unrealized_pnl, realized_pnl_day)
            VALUES (?,?,?,?,?,?,?)
        """, (account_id, today, total_value, cash,
              total_value - cash, unrealized_pnl, realized_pnl_day))

    # =========================================================================
    # RISK CHECKS
    # =========================================================================

    def get_risk_params(self, account_id: int,
                        symbol: Optional[str] = None) -> dict:
        """
        Return the most specific risk_parameters row:
        symbol-specific > global (symbol IS NULL).
        """
        if symbol:
            row = self.fetchone(
                "SELECT * FROM risk_parameters WHERE account_id=? AND symbol=?",
                (account_id, symbol))
            if row:
                return row
        # Fall back to global
        row = self.fetchone(
            "SELECT * FROM risk_parameters WHERE account_id=? AND symbol IS NULL",
            (account_id,))
        return row or {}

    def check_risk_breaches(self) -> list[dict]:
        """Return positions violating size limits (from the v_risk_breaches view)."""
        return self.fetchall("SELECT * FROM v_risk_breaches")

    def can_open_position(self, account_id: int, symbol: str,
                          order_value: float) -> tuple[bool, str]:
        """
        Gate-check before placing a BUY order.
        Returns (allowed: bool, reason: str).
        """
        # Get latest portfolio value
        snap = self.fetchone(
            "SELECT total_value FROM portfolio_snapshots "
            "WHERE account_id=? ORDER BY snapshot_date DESC LIMIT 1",
            (account_id,))
        if snap is None:
            return False, "No portfolio snapshot found — run take_portfolio_snapshot() first"

        rp      = self.get_risk_params(account_id, symbol)
        tv      = snap["total_value"]
        max_pct = rp.get("max_position_pct", 0.05)

        if order_value / tv > max_pct:
            return False, (f"Order value ${order_value:,.0f} exceeds "
                           f"{max_pct*100:.0f}% of portfolio (${tv*max_pct:,.0f})")

        # Count open positions
        n_pos   = self.fetchone(
            "SELECT COUNT(*) AS n FROM positions WHERE account_id=? AND quantity!=0",
            (account_id,))["n"]
        max_pos = rp.get("max_open_positions", 10)
        if n_pos >= max_pos:
            return False, f"Max open positions reached ({n_pos}/{max_pos})"

        return True, "OK"

    # =========================================================================
    # BACKTESTING
    # =========================================================================

    def create_backtest(self, name: str, strategy_id: int, symbol: str,
                        resolution: str, start_date: str, end_date: str,
                        initial_capital: float = 100_000.0,
                        commission_pct: float = 0.001,
                        slippage_pct: float = 0.0005) -> int:
        cur = self.conn.execute("""
            INSERT INTO backtests
                (name, strategy_id, symbol, resolution, start_date, end_date,
                 initial_capital, commission_pct, slippage_pct, status)
            VALUES (?,?,?,?,?,?,?,?,?,'running')
        """, (name, strategy_id, symbol, resolution, start_date, end_date,
              initial_capital, commission_pct, slippage_pct))
        return cur.lastrowid

    def record_backtest_trade(self, backtest_id: int, symbol: str,
                              entry_time: str, exit_time: str,
                              direction: str, entry_price: float,
                              exit_price: float, quantity: float,
                              commission: float = 0.0,
                              exit_reason: str = "signal") -> int:
        gross  = (exit_price - entry_price) * quantity
        if direction == "SHORT": gross = -gross
        net    = gross - commission
        pct    = net / (entry_price * quantity) if entry_price * quantity else 0
        cur = self.conn.execute("""
            INSERT INTO backtest_trades
                (backtest_id, symbol, entry_time, exit_time, direction,
                 entry_price, exit_price, quantity, gross_pnl, commission,
                 net_pnl, net_pnl_pct, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (backtest_id, symbol, entry_time, exit_time, direction,
              entry_price, exit_price, quantity, gross, commission, net, pct,
              exit_reason))
        return cur.lastrowid

    def save_backtest_metrics(self, backtest_id: int, metrics: dict) -> None:
        """Upsert performance metrics for a completed backtest."""
        cols = ["backtest_id","total_return_pct","annualized_return",
                "sharpe_ratio","sortino_ratio","max_drawdown_pct","max_drawdown_dur",
                "win_rate","total_trades","winning_trades","losing_trades",
                "profit_factor","avg_trade_pnl","avg_win_pnl","avg_loss_pnl",
                "avg_holding_bars"]
        metrics["backtest_id"] = backtest_id
        vals = tuple(metrics.get(c) for c in cols)
        ph   = ",".join(["?"]*len(cols))
        self.conn.execute(
            f"INSERT OR REPLACE INTO backtest_metrics ({','.join(cols)}) VALUES ({ph})",
            vals)
        self.conn.execute(
            "UPDATE backtests SET status='done' WHERE id=?", (backtest_id,))

    def get_backtest_results(self, backtest_id: int) -> dict:
        """Return backtest metadata + metrics + trade list in one dict."""
        bt      = self.fetchone("SELECT * FROM backtests WHERE id=?", (backtest_id,))
        metrics = self.fetchone("SELECT * FROM backtest_metrics WHERE backtest_id=?",
                                (backtest_id,))
        trades  = self.fetchall("SELECT * FROM backtest_trades WHERE backtest_id=? "
                                "ORDER BY entry_time", (backtest_id,))
        return {"backtest": bt, "metrics": metrics, "trades": trades}

    def compare_backtests(self, ids: list[int]) -> list[dict]:
        """Return side-by-side metrics for multiple backtest runs."""
        ph = ",".join(["?"]*len(ids))
        return self.fetchall(
            f"SELECT b.name, b.symbol, b.start_date, b.end_date, m.* "
            f"FROM backtests b JOIN backtest_metrics m ON m.backtest_id=b.id "
            f"WHERE b.id IN ({ph}) ORDER BY m.total_return_pct DESC",
            tuple(ids))

    # =========================================================================
    # ANALYTICS VIEWS (call from code like a stored procedure)
    # =========================================================================

    def latest_indicators(self, symbol: str, resolution: str = "daily") -> Optional[dict]:
        return self.get_latest_indicators(symbol, resolution)

    def trade_performance(self, symbol: Optional[str] = None) -> list[dict]:
        sql    = "SELECT * FROM v_trade_performance"
        params: tuple = ()
        if symbol:
            sql += " WHERE symbol=?"
            params = (symbol,)
        return self.fetchall(sql, params)

    def pattern_backtest_summary(self, direction: Optional[str] = None) -> list[dict]:
        sql    = "SELECT * FROM v_pattern_backtest_summary"
        params: tuple = ()
        if direction:
            sql += " WHERE direction=?"
            params = (direction,)
        return self.fetchall(sql, params)

    def portfolio_history(self, account_id: int, days: int = 30) -> list[dict]:
        return self.fetchall(
            "SELECT * FROM portfolio_snapshots WHERE account_id=? "
            "AND snapshot_date >= date('now',?) "
            "ORDER BY snapshot_date ASC",
            (account_id, f"-{days} days"))

    # =========================================================================
    # AUDIT & LOGGING
    # =========================================================================

    def log_event(self, event_type: str, component: str, symbol: Optional[str],
                  message: str, data: Optional[dict] = None) -> None:
        self.conn.execute("""
            INSERT INTO event_log (event_type, component, symbol, message, data_json)
            VALUES (?,?,?,?,?)
        """, (event_type, component, symbol, message, json.dumps(data or {})))

    def log_error(self, component: str, error_type: str,
                  message: str, exc: Optional[Exception] = None) -> None:
        tb = traceback.format_exc() if exc else None
        self.conn.execute("""
            INSERT INTO error_log (component, error_type, message, traceback)
            VALUES (?,?,?,?)
        """, (component, error_type, message, tb))

    def get_recent_errors(self, component: Optional[str] = None,
                          limit: int = 50) -> list[dict]:
        sql    = "SELECT * FROM error_log WHERE 1=1"
        params = []
        if component: sql += " AND component=?"; params.append(component)
        sql += f" ORDER BY created_at DESC LIMIT {limit}"
        return self.fetchall(sql, tuple(params))


# =============================================================================
# Convenience: initialise schema from schema.sql
# =============================================================================

def init_db(db_path: Union[str, Path], schema_sql: Union[str, Path] = "schema.sql") -> None:
    """Create all tables / indexes / triggers from schema.sql."""
    sql_text = Path(schema_sql).read_text()
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(sql_text)
    print(f"  Database initialised: {db_path}")


# =============================================================================
# Quick sanity check (python3 db_procedures.py)
# =============================================================================
if __name__ == "__main__":
    import tempfile, os

    # Expects schema.sql in current directory
    schema = Path(__file__).parent / "schema.sql"
    if not schema.exists():
        print("schema.sql not found — run from the project root")
        raise SystemExit(1)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp = f.name

    try:
        init_db(tmp, schema)

        with DB(tmp) as db:
            # Verify seed data
            syms = db.fetchall("SELECT symbol FROM symbols")
            print(f"  Symbols: {[s['symbol'] for s in syms]}")

            strats = db.get_active_strategies()
            print(f"  Strategies: {[s['name'] for s in strats]}")

            # Create a test signal
            sid = db.create_signal(
                "AAPL", strategy_id=1,
                timestamp="2024-01-15T00:00:00Z",
                resolution="daily", signal_type="BUY",
                strength=0.85,
                reason={"rsi": 27.4, "patterns": ["Hammer"]})
            print(f"  Signal created: id={sid}")

            # Place an order linked to the signal
            oid = db.place_order(
                account_id=1, symbol="AAPL",
                side="BUY", quantity=10,
                signal_id=sid)
            print(f"  Order placed: id={oid}")

            # Simulate IBKR submission + fill
            db.update_order_status(oid, "submitted", ibkr_order_id="IBKR-001")
            db.update_order_status(oid, "filled")
            db.record_execution(oid, "EXEC-001", price=185.50, quantity=10,
                                commission=0.185)
            db.upsert_position(1, "AAPL", quantity=10, avg_cost=185.50,
                               current_price=185.50)
            db.take_portfolio_snapshot(1, total_value=100_000, cash=98_145,
                                       unrealized_pnl=0)

            # Risk check
            ok, reason = db.can_open_position(1, "MSFT", 5_000)
            print(f"  Risk check for $5k MSFT: {ok} — {reason}")

            # Event log
            events = db.fetchall("SELECT event_type, message FROM event_log "
                                 "ORDER BY id")
            print(f"  Events logged: {len(events)}")
            for e in events:
                print(f"    [{e['event_type']}] {e['message']}")

        print("\n  All checks passed.")
    finally:
        os.unlink(tmp)