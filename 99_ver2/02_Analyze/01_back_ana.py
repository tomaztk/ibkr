"""
Technical Analysis — Indicators & Chart Pattern Detection
Reads bars_daily and bars_hourly from alpaca_data.db and writes results
back into new tables: indicators_daily, indicators_hourly,
patterns_daily, patterns_hourly.

Requirements:
    pip install pandas numpy python-dotenv

Tables written:
    indicators_daily / indicators_hourly
        symbol, timestamp,
        sma_20, sma_50, sma_200,
        rsi_14,
        macd, macd_signal, macd_hist,
        bb_upper, bb_middle, bb_lower, bb_width, bb_pct

    patterns_daily / patterns_hourly
        symbol, timestamp, pattern, direction, confidence

Run:
cd 02_Analyze
python3 01_back_ana.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path("../01_Data/alpaca_data.db")

RESOLUTIONS = ["daily", "hourly"]
SYMBOLS     = ["AAPL", "MSFT"]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
CREATE_INDICATORS_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    sma_20      REAL,
    sma_50      REAL,
    sma_200     REAL,
    rsi_14      REAL,
    macd        REAL,
    macd_signal REAL,
    macd_hist   REAL,
    bb_upper    REAL,
    bb_middle   REAL,
    bb_lower    REAL,
    bb_width    REAL,
    bb_pct      REAL,
    UNIQUE(symbol, timestamp)
);
"""

CREATE_PATTERNS_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    pattern     TEXT NOT NULL,
    direction   TEXT NOT NULL,
    confidence  TEXT NOT NULL,
    UNIQUE(symbol, timestamp, pattern)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_{table}_symbol_ts
ON {table} (symbol, timestamp);
"""


def init_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for res in RESOLUTIONS:
        for sql in (CREATE_INDICATORS_SQL, CREATE_PATTERNS_SQL):
            table = sql.strip().split("EXISTS ")[1].split(" ")[0].replace("{table}",
                "indicators_" + res if "indicators" in sql else "patterns_" + res)
            cur.executescript(sql.format(table=table))
            cur.executescript(CREATE_INDEX_SQL.format(table=table))
    conn.commit()


def load_bars(conn: sqlite3.Connection, resolution: str, symbol: str) -> pd.DataFrame:
    table = f"bars_{resolution}"
    df = pd.read_sql(
        f"SELECT * FROM {table} WHERE symbol = ? ORDER BY timestamp ASC",
        conn, params=(symbol,)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def upsert_indicators(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> int:
    cols = ["symbol", "timestamp", "sma_20", "sma_50", "sma_200",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct"]
    cols = [c for c in cols if c in df.columns]
    rows = [tuple(row[c] for c in cols) for _, row in df.iterrows()]
    ph   = ", ".join(["?"] * len(cols))
    sql  = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({ph})"
    cur  = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


def upsert_patterns(conn: sqlite3.Connection, records: list, table: str) -> int:
    if not records:
        return 0
    sql = (f"INSERT OR IGNORE INTO {table} "
           f"(symbol, timestamp, pattern, direction, confidence) "
           f"VALUES (?, ?, ?, ?, ?)")
    cur = conn.cursor()
    cur.executemany(sql, records)
    conn.commit()
    return cur.rowcount


# ===========================================================================
# TECHNICAL INDICATORS
# ===========================================================================

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9
              ) -> pd.DataFrame:
    ema_fast   = series.ewm(span=fast,   min_periods=fast).mean()
    ema_slow   = series.ewm(span=slow,   min_periods=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line,
                         "macd_signal": signal_line,
                         "macd_hist": histogram})


def calc_bollinger_bands(series: pd.Series,
                         period: int = 20, std_dev: float = 2.0
                         ) -> pd.DataFrame:
    middle = series.rolling(window=period, min_periods=period).mean()
    std    = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    width  = (upper - lower) / middle.replace(0, np.nan)
    pct    = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_upper":  upper,
                         "bb_middle": middle,
                         "bb_lower":  lower,
                         "bb_width":  width,
                         "bb_pct":    pct})


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    df["sma_20"]  = calc_sma(close, 20)
    df["sma_50"]  = calc_sma(close, 50)
    df["sma_200"] = calc_sma(close, 200)
    df["rsi_14"]  = calc_rsi(close, 14)

    macd_df = calc_macd(close)
    df[["macd", "macd_signal", "macd_hist"]] = macd_df

    bb_df = calc_bollinger_bands(close)
    df[["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct"]] = bb_df

    # Round to 6 decimal places for clean storage
    num_cols = ["sma_20", "sma_50", "sma_200", "rsi_14",
                "macd", "macd_signal", "macd_hist",
                "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct"]
    df[num_cols] = df[num_cols].round(6)

    return df


# ===========================================================================
# CHART PATTERN DETECTION
# ===========================================================================
# Each detector receives the full DataFrame and returns a list of dicts:
#   { timestamp, pattern, direction, confidence }
# ---------------------------------------------------------------------------

def _is_local_high(s: pd.Series, i: int, n: int = 2) -> bool:
    """True if s[i] is the highest value in a window of n bars each side."""
    window = s.iloc[max(0, i - n): i + n + 1]
    return s.iloc[i] == window.max() and window.nunique() > 1


def _is_local_low(s: pd.Series, i: int, n: int = 2) -> bool:
    window = s.iloc[max(0, i - n): i + n + 1]
    return s.iloc[i] == window.min() and window.nunique() > 1


# --- Candlestick patterns ---------------------------------------------------

def detect_doji(df: pd.DataFrame) -> list:
    """Body is ≤ 10 % of the total candle range → indecision."""
    records = []
    body   = (df["close"] - df["open"]).abs()
    rng    = df["high"] - df["low"]
    ratio  = body / rng.replace(0, np.nan)
    mask   = ratio <= 0.10
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "Doji", "neutral", "medium"))
    return records


def detect_hammer(df: pd.DataFrame) -> list:
    """Small body near top, long lower wick ≥ 2× body — bullish reversal."""
    records = []
    body    = (df["close"] - df["open"]).abs()
    rng     = df["high"] - df["low"]
    lower_w = df[["open", "close"]].min(axis=1) - df["low"]
    upper_w = df["high"] - df[["open", "close"]].max(axis=1)
    mask = (lower_w >= 2 * body) & (upper_w <= 0.3 * body) & (body > 0)
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "Hammer", "bullish", "medium"))
    return records


def detect_shooting_star(df: pd.DataFrame) -> list:
    """Small body near bottom, long upper wick ≥ 2× body — bearish reversal."""
    records = []
    body    = (df["close"] - df["open"]).abs()
    lower_w = df[["open", "close"]].min(axis=1) - df["low"]
    upper_w = df["high"] - df[["open", "close"]].max(axis=1)
    mask = (upper_w >= 2 * body) & (lower_w <= 0.3 * body) & (body > 0)
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "Shooting Star", "bearish", "medium"))
    return records


def detect_engulfing(df: pd.DataFrame) -> list:
    """Current candle body fully engulfs previous candle body."""
    records = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        prev_body_lo = min(prev["open"], prev["close"])
        prev_body_hi = max(prev["open"], prev["close"])
        curr_body_lo = min(curr["open"], curr["close"])
        curr_body_hi = max(curr["open"], curr["close"])
        if (curr_body_lo < prev_body_lo and curr_body_hi > prev_body_hi):
            direction = "bullish" if curr["close"] > curr["open"] else "bearish"
            records.append((str(curr["timestamp"]), "Engulfing", direction, "high"))
    return records


def detect_morning_star(df: pd.DataFrame) -> list:
    """3-bar bullish reversal: large red, small body, large green."""
    records = []
    for i in range(2, len(df)):
        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        body1 = c1["open"] - c1["close"]           # red
        body2 = abs(c2["close"] - c2["open"])       # small
        body3 = c3["close"] - c3["open"]            # green
        rng1  = c1["high"] - c1["low"] or 1
        if (body1 > 0.6 * rng1 and
                body2 < 0.3 * rng1 and
                body3 > 0.6 * (c3["high"] - c3["low"] or 1)):
            records.append((str(c3["timestamp"]), "Morning Star", "bullish", "high"))
    return records


def detect_evening_star(df: pd.DataFrame) -> list:
    """3-bar bearish reversal: large green, small body, large red."""
    records = []
    for i in range(2, len(df)):
        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        body1 = c1["close"] - c1["open"]
        body2 = abs(c2["close"] - c2["open"])
        body3 = c3["open"] - c3["close"]
        rng1  = c1["high"] - c1["low"] or 1
        if (body1 > 0.6 * rng1 and
                body2 < 0.3 * rng1 and
                body3 > 0.6 * (c3["high"] - c3["low"] or 1)):
            records.append((str(c3["timestamp"]), "Evening Star", "bearish", "high"))
    return records


# --- Multi-bar structural patterns -----------------------------------------

def detect_double_top(df: pd.DataFrame, window: int = 20, tol: float = 0.02) -> list:
    """Two highs within tol % of each other separated by a trough."""
    records = []
    highs   = df["high"]
    for i in range(window, len(df) - 1):
        seg = highs.iloc[i - window: i + 1]
        peaks = [j for j in range(1, len(seg) - 1)
                 if _is_local_high(seg, j)]
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(seg.iloc[p1] - seg.iloc[p2]) / seg.iloc[p1] <= tol:
                ts = str(df["timestamp"].iloc[i])
                records.append((ts, "Double Top", "bearish", "high"))
    return records


def detect_double_bottom(df: pd.DataFrame, window: int = 20, tol: float = 0.02) -> list:
    """Two lows within tol % of each other separated by a peak."""
    records = []
    lows = df["low"]
    for i in range(window, len(df) - 1):
        seg = lows.iloc[i - window: i + 1]
        troughs = [j for j in range(1, len(seg) - 1)
                   if _is_local_low(seg, j)]
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if abs(seg.iloc[t1] - seg.iloc[t2]) / seg.iloc[t1] <= tol:
                ts = str(df["timestamp"].iloc[i])
                records.append((ts, "Double Bottom", "bullish", "high"))
    return records


def detect_golden_cross(df: pd.DataFrame) -> list:
    """SMA 50 crosses above SMA 200 — strong bullish signal."""
    records = []
    if "sma_50" not in df.columns or "sma_200" not in df.columns:
        return records
    prev = df.shift(1)
    mask = (prev["sma_50"] <= prev["sma_200"]) & (df["sma_50"] > df["sma_200"])
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "Golden Cross", "bullish", "high"))
    return records


def detect_death_cross(df: pd.DataFrame) -> list:
    """SMA 50 crosses below SMA 200 — strong bearish signal."""
    records = []
    if "sma_50" not in df.columns or "sma_200" not in df.columns:
        return records
    prev = df.shift(1)
    mask = (prev["sma_50"] >= prev["sma_200"]) & (df["sma_50"] < df["sma_200"])
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "Death Cross", "bearish", "high"))
    return records


def detect_rsi_oversold(df: pd.DataFrame, threshold: float = 30.0) -> list:
    """RSI crosses back above the oversold threshold — potential reversal up."""
    records = []
    if "rsi_14" not in df.columns:
        return records
    prev = df["rsi_14"].shift(1)
    mask = (prev <= threshold) & (df["rsi_14"] > threshold)
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "RSI Oversold Bounce", "bullish", "medium"))
    return records


def detect_rsi_overbought(df: pd.DataFrame, threshold: float = 70.0) -> list:
    """RSI crosses back below the overbought threshold — potential reversal down."""
    records = []
    if "rsi_14" not in df.columns:
        return records
    prev = df["rsi_14"].shift(1)
    mask = (prev >= threshold) & (df["rsi_14"] < threshold)
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "RSI Overbought Drop", "bearish", "medium"))
    return records


def detect_macd_crossover(df: pd.DataFrame) -> list:
    """MACD line crosses above signal line — bullish momentum."""
    records = []
    if "macd" not in df.columns:
        return records
    prev = df.shift(1)
    bull = (prev["macd"] <= prev["macd_signal"]) & (df["macd"] > df["macd_signal"])
    bear = (prev["macd"] >= prev["macd_signal"]) & (df["macd"] < df["macd_signal"])
    for ts in df.loc[bull, "timestamp"]:
        records.append((str(ts), "MACD Bullish Cross", "bullish", "medium"))
    for ts in df.loc[bear, "timestamp"]:
        records.append((str(ts), "MACD Bearish Cross", "bearish", "medium"))
    return records


def detect_bb_squeeze(df: pd.DataFrame, pct: float = 0.10) -> list:
    """BB width in the bottom pct % of its rolling range — low volatility / breakout pending."""
    records = []
    if "bb_width" not in df.columns:
        return records
    rolling_min = df["bb_width"].rolling(50, min_periods=20).min()
    rolling_max = df["bb_width"].rolling(50, min_periods=20).max()
    norm = (df["bb_width"] - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)
    mask = norm <= pct
    for ts in df.loc[mask, "timestamp"]:
        records.append((str(ts), "BB Squeeze", "neutral", "medium"))
    return records


# Registry of all detectors
PATTERN_DETECTORS = [
    detect_doji,
    detect_hammer,
    detect_shooting_star,
    detect_engulfing,
    detect_morning_star,
    detect_evening_star,
    detect_double_top,
    detect_double_bottom,
    detect_golden_cross,
    detect_death_cross,
    detect_rsi_oversold,
    detect_rsi_overbought,
    detect_macd_crossover,
    detect_bb_squeeze,
]


def detect_all_patterns(df: pd.DataFrame, symbol: str) -> list:
    """Run every detector and return a flat list of DB-ready tuples."""
    all_records = []
    for detector in PATTERN_DETECTORS:
        try:
            hits = detector(df)
            for ts, pattern, direction, confidence in hits:
                all_records.append((symbol, ts, pattern, direction, confidence))
        except Exception as e:
            print(f"   {detector.__name__} failed: {e}")
    return all_records


# ===========================================================================
# Main
# ===========================================================================

def print_summary(conn: sqlite3.Connection) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {'TABLE':<26} {'SYMBOL':<8} {'ROWS':>8}")
    print(f"{'─' * 65}")
    cur = conn.cursor()
    tables = [f"{kind}_{res}"
              for kind in ("indicators", "patterns")
              for res in RESOLUTIONS]
    for table in tables:
        cur.execute(f"SELECT symbol, COUNT(*) FROM {table} "
                    f"GROUP BY symbol ORDER BY symbol")
        for symbol, count in cur.fetchall():
            print(f"  {table:<26} {symbol:<8} {count:>8,}")
    print(f"{'─' * 65}")


def main() -> None:
    if not DB_PATH.exists():
        print(f"  Database not found at {DB_PATH.resolve()}")
        print("   Run fetch_alpaca_data.py first.")
        return

    print("=" * 65)
    print("  Technical Analysis — Indicators & Pattern Detection")
    print("=" * 65)

    with sqlite3.connect(DB_PATH) as conn:
        # Ensure output tables exist
        cur = conn.cursor()
        for res in RESOLUTIONS:
            cur.executescript(CREATE_INDICATORS_SQL.format(table=f"indicators_{res}"))
            cur.executescript(CREATE_INDEX_SQL.format(table=f"indicators_{res}"))
            cur.executescript(CREATE_PATTERNS_SQL.format(table=f"patterns_{res}"))
            cur.executescript(CREATE_INDEX_SQL.format(table=f"patterns_{res}"))
        conn.commit()

        for symbol in SYMBOLS:
            print(f"\n Symbol: {symbol}")
            print(f" {'─' * 50}")

            for res in RESOLUTIONS:
                print(f"\n  [{res}]")

                # --- Load ---
                df = load_bars(conn, res, symbol)
                if df.empty:
                    print(f"    No data found in bars_{res} for {symbol}")
                    continue
                print(f"    Loaded {len(df):,} bars "
                      f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")

                # --- Indicators ---
                df = compute_indicators(df)
                df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
                df["symbol"]    = symbol

                ind_table = f"indicators_{res}"
                n = upsert_indicators(conn, df, ind_table)
                print(f"    ✓ Indicators  → {ind_table}  ({n:,} rows upserted)")

                # Re-parse timestamp as string for pattern detectors
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

                # --- Patterns ---
                pat_table = f"patterns_{res}"
                records   = detect_all_patterns(df, symbol)
                n = upsert_patterns(conn, records, pat_table)
                print(f"    ✓ Patterns    → {pat_table}  ({len(records):,} detected, "
                      f"{n:,} new rows inserted)")

                # Print pattern breakdown
                if records:
                    from collections import Counter
                    counts = Counter(r[2] for r in records)
                    for pattern, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                        print(f"       {cnt:>5}×  {pattern}")

        print_summary(conn)

    print(f"\n Analysis complete  →  {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()