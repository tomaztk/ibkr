"""
strategy_runner.py — Strategy Evaluation & Order Execution
===========================================================
Flow per run
  1. Load active strategies from DB
  2. For every (strategy × symbol) pair:
       a. Pull latest indicators + recent patterns
       b. Evaluate every trigger_rule in the strategy
       c. Score the results — if composite strength ≥ threshold → signal
       d. Risk gate  (position sizing, max positions, daily loss cap)
       e. Place BUY / SELL order via IBKR
       f. Track the fill, update positions, log everything
  3. Check existing positions for stop-loss / take-profit exits

Usage
  python3 strategy_runner.py                      # dry-run (no real orders)
  python3 strategy_runner.py --live               # real IBKR execution
  python3 strategy_runner.py --symbol AAPL --live
  python3 strategy_runner.py --strategy "RSI_Mean_Reversion" --live

Requirements
  pip install ib_insync   (needs TWS or IB Gateway running on localhost:7497)
  db_procedures.py + schema.sql must be present

  Run:
cd 04_DB
python3 strategy_runner.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from db_procedures import DB, init_db

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("runner.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("runner")

# =============================================================================
# Config
# =============================================================================

DB_PATH     = Path("99ver_db.db")
SCHEMA_PATH = Path("schema.sql")

# Symbols the runner watches (override with --symbol)
WATCHLIST   = ["AAPL", "MSFT"]

# IBKR connection
IBKR_HOST   = "127.0.0.1"
IBKR_PORT   = 7497          # 7497 = TWS paper, 7496 = TWS live, 4002 = Gateway
IBKR_CLIENT = 1

# Signal strength threshold to act (0–1). Rules are scored and averaged.
SIGNAL_THRESHOLD = 0.55

# Patterns to look back for (bars)
PATTERN_LOOKBACK_BARS = 3


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class EvalResult:
    """Result of evaluating one (strategy × symbol) pair."""
    symbol:      str
    strategy_id: int
    strategy_name: str
    action:      str          # "BUY" | "SELL" | "HOLD" | "CLOSE"
    strength:    float        # 0.0 – 1.0
    fired_rules: list[dict]   = field(default_factory=list)
    reason:      dict         = field(default_factory=dict)
    timestamp:   str          = field(default_factory=lambda:
                                    datetime.now(timezone.utc)
                                    .strftime("%Y-%m-%dT%H:%M:%SZ"))


@dataclass
class OrderResult:
    """Outcome of submitting one order to IBKR."""
    order_db_id:   int
    ibkr_order_id: Optional[str]
    status:        str     # "submitted" | "filled" | "dry_run" | "blocked" | "error"
    fill_price:    Optional[float] = None
    fill_qty:      Optional[float] = None
    message:       str = ""


# =============================================================================
# IBKR Broker
# =============================================================================

class IBKRBroker:
    """
    Thin wrapper around ib_insync.
    All order-placement logic lives here — the rest of the system
    never imports ib_insync directly.
    """

    def __init__(self, host: str, port: int, client_id: int, dry_run: bool = True):
        self.host      = host
        self.port      = port
        self.client_id = client_id
        self.dry_run   = dry_run
        self._ib       = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def connect(self) -> bool:
        if self.dry_run:
            log.info("IBKR  [DRY-RUN] — no real connection made")
            return True
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id,
                             timeout=10, readonly=False)
            log.info("IBKR  Connected  account=%s",
                     self._ib.managedAccounts())
            return True
        except Exception as e:
            log.error("IBKR  Connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            log.info("IBKR  Disconnected")

    # ── account info ─────────────────────────────────────────────────────
    def get_account_value(self) -> Optional[float]:
        """Return net liquidation value in base currency."""
        if self.dry_run:
            return 100_000.0    # paper value for dry-run
        try:
            vals = self._ib.accountValues()
            nlv  = next((v for v in vals
                         if v.tag == "NetLiquidation" and v.currency == "USD"), None)
            return float(nlv.value) if nlv else None
        except Exception as e:
            log.error("IBKR  get_account_value: %s", e)
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch last traded price for a symbol."""
        if self.dry_run:
            # Return a fake price; in real code this comes from the feed.
            return None
        try:
            from ib_insync import Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            ticker = self._ib.reqMktData(contract, snapshot=True)
            self._ib.sleep(2)
            return ticker.last or ticker.close
        except Exception as e:
            log.error("IBKR  get_current_price(%s): %s", symbol, e)
            return None

    # ── order placement ──────────────────────────────────────────────────
    def submit_market_order(self, symbol: str, side: str,
                            quantity: float) -> OrderResult:
        """
        Place a MARKET order.
        Returns immediately with ibkr_order_id; fill arrives via callback.
        """
        if self.dry_run:
            fake_id = f"DRY-{symbol}-{int(time.time())}"
            log.info("IBKR  [DRY-RUN] MARKET %s  %s  qty=%.0f", side, symbol, quantity)
            return OrderResult(order_db_id=0, ibkr_order_id=fake_id,
                               status="dry_run",
                               message=f"Dry-run: {side} {quantity} {symbol}")

        try:
            from ib_insync import Stock, MarketOrder
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order    = MarketOrder(side, quantity)
            trade    = self._ib.placeOrder(contract, order)
            self._ib.sleep(1)
            ibkr_id  = str(trade.order.orderId)
            log.info("IBKR  MARKET %s %s qty=%.0f  ibkr_id=%s",
                     side, symbol, quantity, ibkr_id)
            return OrderResult(order_db_id=0, ibkr_order_id=ibkr_id,
                               status="submitted")
        except Exception as e:
            log.error("IBKR  submit_market_order(%s %s): %s", side, symbol, e)
            return OrderResult(order_db_id=0, ibkr_order_id=None,
                               status="error", message=str(e))

    def submit_limit_order(self, symbol: str, side: str,
                           quantity: float, limit_price: float) -> OrderResult:
        if self.dry_run:
            fake_id = f"DRY-LMT-{symbol}-{int(time.time())}"
            log.info("IBKR  [DRY-RUN] LIMIT %s  %s  qty=%.0f @ %.4f",
                     side, symbol, quantity, limit_price)
            return OrderResult(order_db_id=0, ibkr_order_id=fake_id,
                               status="dry_run")
        try:
            from ib_insync import Stock, LimitOrder
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            order    = LimitOrder(side, quantity, limit_price)
            trade    = self._ib.placeOrder(contract, order)
            self._ib.sleep(1)
            ibkr_id  = str(trade.order.orderId)
            return OrderResult(order_db_id=0, ibkr_order_id=ibkr_id,
                               status="submitted")
        except Exception as e:
            log.error("IBKR  submit_limit_order: %s", e)
            return OrderResult(order_db_id=0, ibkr_order_id=None,
                               status="error", message=str(e))

    def wait_for_fill(self, ibkr_order_id: str,
                      timeout_sec: int = 30) -> tuple[Optional[float], Optional[float]]:
        """
        Poll until the order is filled or timeout.
        Returns (fill_price, fill_qty) or (None, None).
        """
        if self.dry_run or self._ib is None:
            return None, None
        try:
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                self._ib.sleep(1)
                for trade in self._ib.trades():
                    if str(trade.order.orderId) == ibkr_order_id:
                        if trade.orderStatus.status == "Filled":
                            return (trade.orderStatus.avgFillPrice,
                                    trade.orderStatus.filled)
            log.warning("IBKR  wait_for_fill timeout  ibkr_id=%s", ibkr_order_id)
            return None, None
        except Exception as e:
            log.error("IBKR  wait_for_fill: %s", e)
            return None, None


# =============================================================================
# Strategy Evaluator
# =============================================================================

class StrategyEvaluator:
    """
    Loads a strategy's trigger_rules from DB and scores them
    against the latest indicator + pattern snapshot.
    """

    def __init__(self, db: DB):
        self.db = db

    def evaluate(self, strategy: dict, symbol: str) -> EvalResult:
        """
        Return an EvalResult for one (strategy × symbol).
        """
        resolution = strategy["resolution"]

        # ── fetch latest data ──────────────────────────────────────────
        indicators = self.db.get_latest_indicators(symbol, resolution)
        if indicators is None:
            log.debug("  %s/%s  no indicators — skip", symbol, strategy["name"])
            return EvalResult(symbol, strategy["id"], strategy["name"],
                              "HOLD", 0.0)

        raw_patterns = self.db.get_recent_patterns(
            symbol, resolution, days=PATTERN_LOOKBACK_BARS)
        pattern_names = [p["pattern"] for p in raw_patterns]

        # ── check trigger rules ────────────────────────────────────────
        fired = self.db.check_trigger_rules(
            strategy["id"], indicators, pattern_names)

        if not fired:
            return EvalResult(symbol, strategy["id"], strategy["name"],
                              "HOLD", 0.0,
                              reason={"indicators": _ind_summary(indicators)})

        # ── composite scoring ──────────────────────────────────────────
        # Each rule contributes equally; confidence can weight in future.
        buy_rules  = [r for r in fired if r["action"] == "BUY"]
        sell_rules = [r for r in fired if r["action"] == "SELL"]
        close_rules= [r for r in fired if r["action"] == "CLOSE"]

        total = len(fired)
        buy_score  = len(buy_rules)  / total
        sell_score = len(sell_rules) / total

        # Also factor in pattern direction confirmation
        bull_pats = sum(1 for p in raw_patterns if p["direction"] == "bullish")
        bear_pats = sum(1 for p in raw_patterns if p["direction"] == "bearish")
        if pattern_names:
            buy_score  += 0.15 * (bull_pats / len(raw_patterns))
            sell_score += 0.15 * (bear_pats / len(raw_patterns))

        # Clamp to [0,1]
        buy_score  = min(buy_score,  1.0)
        sell_score = min(sell_score, 1.0)

        if close_rules:
            action, strength = "CLOSE", 1.0
        elif buy_score >= SIGNAL_THRESHOLD and buy_score >= sell_score:
            action, strength = "BUY",  round(buy_score,  4)
        elif sell_score >= SIGNAL_THRESHOLD and sell_score > buy_score:
            action, strength = "SELL", round(sell_score, 4)
        else:
            action, strength = "HOLD", round(max(buy_score, sell_score), 4)

        reason = {
            "fired_rules":  [r["name"] for r in fired],
            "buy_score":    round(buy_score,  4),
            "sell_score":   round(sell_score, 4),
            "patterns":     pattern_names,
            "indicators":   _ind_summary(indicators),
        }

        log.info("  %-6s  %-30s  action=%-5s  strength=%.2f  rules=%d",
                 symbol, strategy["name"], action, strength, len(fired))

        return EvalResult(
            symbol=symbol, strategy_id=strategy["id"],
            strategy_name=strategy["name"],
            action=action, strength=strength,
            fired_rules=fired, reason=reason,
        )


def _ind_summary(ind: dict) -> dict:
    """Return a compact snapshot of key indicators for logging."""
    return {k: round(ind[k], 4) if ind.get(k) is not None else None
            for k in ("rsi_14", "macd_hist", "bb_pct", "sma_50", "sma_200")}


# =============================================================================
# Order Manager
# =============================================================================

class OrderManager:
    """
    Bridges EvalResult → risk check → DB order record → IBKR submission.
    """

    def __init__(self, db: DB, broker: IBKRBroker,
                 account_id: int, dry_run: bool = True):
        self.db         = db
        self.broker     = broker
        self.account_id = account_id
        self.dry_run    = dry_run

    # ── position sizing ──────────────────────────────────────────────────
    def _calc_quantity(self, symbol: str, price: float) -> float:
        """
        Use risk_parameters.max_position_pct of latest portfolio value.
        Falls back to 1 share if no snapshot exists.
        """
        rp = self.db.get_risk_params(self.account_id, symbol)
        snap = self.db.fetchone(
            "SELECT total_value FROM portfolio_snapshots "
            "WHERE account_id=? ORDER BY snapshot_date DESC LIMIT 1",
            (self.account_id,))

        if snap is None or price <= 0:
            log.warning("  No portfolio snapshot or zero price — defaulting to 1 share")
            return 1.0

        max_pct  = rp.get("max_position_pct", 0.05)
        budget   = snap["total_value"] * max_pct
        quantity = budget / price
        return max(1.0, round(quantity, 0))

    # ── entry order ──────────────────────────────────────────────────────
    def handle_entry(self, result: EvalResult,
                     current_price: Optional[float]) -> Optional[OrderResult]:
        """
        Place a BUY (or short SELL) order when strategy fires.
        """
        symbol = result.symbol
        side   = result.action   # "BUY" or "SELL"

        if current_price is None:
            # Fall back to last close in DB
            bar = self.db.fetchone(
                "SELECT close FROM bars_daily WHERE symbol=? "
                "ORDER BY timestamp DESC LIMIT 1", (symbol,))
            current_price = bar["close"] if bar else 0.0

        if current_price <= 0:
            log.warning("  %s  cannot determine price — skip", symbol)
            return None

        quantity = self._calc_quantity(symbol, current_price)
        order_value = current_price * quantity

        # ── risk gate ────────────────────────────────────────────────
        ok, reason = self.db.can_open_position(
            self.account_id, symbol, order_value)
        if not ok:
            log.warning("  %s  BLOCKED — %s", symbol, reason)
            self.db.log_event("RISK_BREACH", "order_manager", symbol,
                              f"Order blocked: {reason}",
                              {"side": side, "qty": quantity, "price": current_price})
            return OrderResult(order_db_id=0, ibkr_order_id=None,
                               status="blocked", message=reason)

        # ── already have a position? skip double-entry ────────────────
        pos = self.db.fetchone(
            "SELECT quantity FROM positions WHERE account_id=? AND symbol=?",
            (self.account_id, symbol))
        if pos and side == "BUY"  and pos["quantity"] > 0:
            log.info("  %s  already LONG — skip BUY", symbol)
            return None
        if pos and side == "SELL" and pos["quantity"] < 0:
            log.info("  %s  already SHORT — skip SELL", symbol)
            return None

        # ── create signal record ──────────────────────────────────────
        signal_id = self.db.create_signal(
            symbol=symbol,
            strategy_id=result.strategy_id,
            timestamp=result.timestamp,
            resolution=self.db.fetchone(
                "SELECT resolution FROM strategies WHERE id=?",
                (result.strategy_id,))["resolution"],
            signal_type="BUY" if side == "BUY" else "SELL",
            strength=result.strength,
            reason=result.reason,
        )

        # ── DB order record ───────────────────────────────────────────
        order_db_id = self.db.place_order(
            account_id=self.account_id,
            symbol=symbol, side=side,
            quantity=quantity, order_type="MARKET",
            signal_id=signal_id,
        )

        # ── IBKR submission ───────────────────────────────────────────
        ibkr_result = self.broker.submit_market_order(symbol, side, quantity)
        ibkr_result.order_db_id = order_db_id

        if ibkr_result.status in ("submitted", "dry_run"):
            self.db.update_order_status(
                order_db_id, "submitted",
                ibkr_order_id=ibkr_result.ibkr_order_id)

            # Wait for fill (skipped in dry-run)
            fill_price, fill_qty = self.broker.wait_for_fill(
                ibkr_result.ibkr_order_id or "")

            if fill_price:
                self.db.update_order_status(order_db_id, "filled")
                self.db.record_execution(
                    order_db_id,
                    ibkr_exec_id=f"EXEC-{order_db_id}",
                    price=fill_price, quantity=fill_qty or quantity,
                    commission=fill_price * quantity * 0.001)
                self.db.upsert_position(
                    self.account_id, symbol,
                    quantity=(quantity if side == "BUY" else -quantity),
                    avg_cost=fill_price, current_price=fill_price)
                ibkr_result.fill_price = fill_price
                ibkr_result.fill_qty   = fill_qty
                ibkr_result.status     = "filled"
                log.info("  %s  FILLED  %.0f shares @ %.4f", symbol, quantity, fill_price)
            else:
                # Dry-run or async fill — record estimated position
                if self.dry_run:
                    self.db.upsert_position(
                        self.account_id, symbol,
                        quantity=(quantity if side == "BUY" else -quantity),
                        avg_cost=current_price, current_price=current_price)
        else:
            self.db.update_order_status(order_db_id, "rejected",
                                        reject_reason=ibkr_result.message)

        return ibkr_result

    # ── exit order (stop-loss / take-profit) ─────────────────────────────
    def handle_exit(self, symbol: str,
                    current_price: float) -> Optional[OrderResult]:
        """
        Check if an open position has hit stop-loss or take-profit.
        If so, place a closing order.
        """
        pos = self.db.fetchone(
            "SELECT * FROM positions WHERE account_id=? AND symbol=? AND quantity!=0",
            (self.account_id, symbol))
        if pos is None:
            return None

        rp          = self.db.get_risk_params(self.account_id, symbol)
        sl_pct      = rp.get("stop_loss_pct",   0.02)
        tp_pct      = rp.get("take_profit_pct", 0.06)
        avg_cost    = pos["avg_cost"]
        qty         = pos["quantity"]
        side        = "LONG" if qty > 0 else "SHORT"

        if side == "LONG":
            pnl_pct = (current_price - avg_cost) / avg_cost
        else:
            pnl_pct = (avg_cost - current_price) / avg_cost

        exit_reason = None
        if pnl_pct <= -sl_pct:
            exit_reason = f"STOP_LOSS  ({pnl_pct*100:.2f}%  ≤ -{sl_pct*100:.0f}%)"
        elif pnl_pct >= tp_pct:
            exit_reason = f"TAKE_PROFIT ({pnl_pct*100:.2f}%  ≥ +{tp_pct*100:.0f}%)"

        if exit_reason is None:
            return None

        log.info("  %s  EXIT triggered: %s", symbol, exit_reason)
        close_side = "SELL" if qty > 0 else "BUY"

        order_db_id = self.db.place_order(
            account_id=self.account_id,
            symbol=symbol, side=close_side,
            quantity=abs(qty), order_type="MARKET")

        ibkr_result = self.broker.submit_market_order(
            symbol, close_side, abs(qty))
        ibkr_result.order_db_id = order_db_id

        if ibkr_result.status in ("submitted", "dry_run"):
            self.db.update_order_status(order_db_id, "submitted",
                                        ibkr_order_id=ibkr_result.ibkr_order_id)
            fill_price, fill_qty = self.broker.wait_for_fill(
                ibkr_result.ibkr_order_id or "")
            exit_price = fill_price or current_price

            self.db.update_order_status(order_db_id,
                                        "filled" if fill_price else "submitted")
            self.db.close_position(self.account_id, symbol, exit_price,
                                   exit_order_id=order_db_id)
            self.db.log_event(
                "POSITION_CLOSED", "order_manager", symbol,
                exit_reason,
                {"exit_price": exit_price, "pnl_pct": round(pnl_pct, 4)})

        return ibkr_result


# =============================================================================
# Main Runner
# =============================================================================

class Runner:
    """Orchestrates one full evaluation cycle."""

    def __init__(self, db_path: Path, account_id: int,
                 dry_run: bool = True,
                 symbol_filter: Optional[str] = None,
                 strategy_filter: Optional[str] = None):

        self.db_path         = db_path
        self.account_id      = account_id
        self.dry_run         = dry_run
        self.symbol_filter   = symbol_filter
        self.strategy_filter = strategy_filter

        self.broker = IBKRBroker(IBKR_HOST, IBKR_PORT, IBKR_CLIENT,
                                 dry_run=dry_run)

    def run(self) -> None:
        log.info("=" * 65)
        log.info("  Strategy Runner  [%s]", "DRY-RUN" if self.dry_run else "LIVE")
        log.info("=" * 65)

        if not self.broker.connect():
            log.error("Cannot connect to IBKR — aborting")
            return

        try:
            with DB(self.db_path) as db:
                self._run_cycle(db)
        finally:
            self.broker.disconnect()

    def _run_cycle(self, db: DB) -> None:
        evaluator = StrategyEvaluator(db)
        order_mgr = OrderManager(db, self.broker, self.account_id,
                                 dry_run=self.dry_run)

        # ── 1. determine watchlist ─────────────────────────────────────
        symbols_query = db.fetchall("SELECT symbol FROM symbols WHERE active=1")
        all_symbols   = [r["symbol"] for r in symbols_query] or WATCHLIST
        symbols       = ([self.symbol_filter]
                         if self.symbol_filter else all_symbols)

        # ── 2. load active strategies ──────────────────────────────────
        strategies = db.get_active_strategies()
        if self.strategy_filter:
            strategies = [s for s in strategies
                          if s["name"] == self.strategy_filter]
        if not strategies:
            log.warning("No active strategies found")
            return

        log.info("Symbols: %s", symbols)
        log.info("Strategies: %s", [s["name"] for s in strategies])

        # ── 3. exit check for open positions ──────────────────────────
        log.info("─── Exit check ───────────────────────────────────────")
        open_positions = db.get_live_positions(self.account_id)
        for pos in open_positions:
            sym   = pos["symbol"]
            price = self.broker.get_current_price(sym) or pos.get("current_price")
            if price:
                order_mgr.handle_exit(sym, price)

        # ── 4. strategy evaluation loop ────────────────────────────────
        log.info("─── Strategy evaluation ──────────────────────────────")
        summary: list[EvalResult] = []

        for strategy in strategies:
            for symbol in symbols:
                try:
                    result = evaluator.evaluate(strategy, symbol)
                    summary.append(result)

                    if result.action in ("BUY", "SELL"):
                        price = self.broker.get_current_price(symbol)
                        order_mgr.handle_entry(result, price)

                    elif result.action == "CLOSE":
                        price = self.broker.get_current_price(symbol)
                        if price:
                            order_mgr.handle_exit(symbol, price)

                except Exception as e:
                    log.exception("  %s/%s  unhandled error: %s",
                                  symbol, strategy["name"], e)
                    db.log_error("runner", type(e).__name__,
                                 f"{symbol}/{strategy['name']}: {e}", e)

        # ── 5. print summary ───────────────────────────────────────────
        self._print_summary(summary, db)

    @staticmethod
    def _print_summary(results: list[EvalResult], db: DB) -> None:
        log.info("─── Run summary ──────────────────────────────────────")
        acted = [r for r in results if r.action in ("BUY", "SELL", "CLOSE")]
        holds = [r for r in results if r.action == "HOLD"]

        log.info("  Evaluated : %d combinations", len(results))
        log.info("  Acted on  : %d", len(acted))
        log.info("  Held      : %d", len(holds))

        if acted:
            log.info("  Actions:")
            for r in acted:
                log.info("    %-6s  %-5s  str=%.2f  strategy=%s",
                         r.symbol, r.action, r.strength, r.strategy_name)

        # Live position overview
        positions = db.fetchall(
            "SELECT symbol, quantity, avg_cost, unrealized_pnl "
            "FROM positions WHERE quantity != 0")
        if positions:
            log.info("  Open positions:")
            for p in positions:
                pnl = p["unrealized_pnl"] or 0
                log.info("    %-6s  qty=%-8.0f  avg=%-10.4f  unreal_pnl=%+.2f",
                         p["symbol"], p["quantity"], p["avg_cost"], pnl)


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy Runner")
    parser.add_argument("--live",     action="store_true",
                        help="Execute real orders via IBKR (default: dry-run)")
    parser.add_argument("--symbol",   default=None,
                        help="Evaluate only this symbol")
    parser.add_argument("--strategy", default=None,
                        help="Evaluate only this strategy (exact name)")
    parser.add_argument("--account",  type=int, default=1,
                        help="DB accounts.id to trade under (default: 1)")
    parser.add_argument("--db",       default=str(DB_PATH),
                        help="Path to trading.db")
    parser.add_argument("--init-db",  action="store_true",
                        help="Initialise/migrate the database schema first")
    args = parser.parse_args()

    db_path = Path(args.db)

    # Optionally bootstrap schema
    if args.init_db:
        if not SCHEMA_PATH.exists():
            log.error("schema.sql not found at %s", SCHEMA_PATH.resolve())
            sys.exit(1)
        init_db(db_path, SCHEMA_PATH)

    if not db_path.exists():
        log.error("Database not found: %s  (run with --init-db first)", db_path)
        sys.exit(1)

    runner = Runner(
        db_path=db_path,
        account_id=args.account,
        dry_run=not args.live,
        symbol_filter=args.symbol,
        strategy_filter=args.strategy,
    )
    runner.run()


if __name__ == "__main__":
    main()