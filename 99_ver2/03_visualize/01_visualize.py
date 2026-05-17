"""
Technical Analysis Dashboard — Interactive Visualization
Reads from alpaca_data.db (bars, indicators, patterns tables) and
renders a self-contained, interactive HTML file using Plotly.

Usage:
    cd 03_visualize
    python3 01_visualize.py                      # all symbols, daily
    python3 01_visualize.py --symbol AAPL        # single symbol
    python3 01_visualize.py --res hourly         # hourly resolution
    python3 01_visualize.py --symbol MSFT --res hourly --days 30

Output:
    ta_dashboard_<symbol>_<res>.html   (open in any browser)

Requirements:
    pip install pandas plotly

Run:
cd 03_visualize
python3 01_visualize.py
"""

import argparse
import sqlite3
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta, timezone
 
# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH  = Path("../01_Data/alpaca_data.db")
SYMBOLS  = ["AAPL", "MSFT"]
 
# Pattern marker styles  {pattern: (symbol_shape, color)}
PATTERN_STYLES = {
    "Doji":               ("diamond",         "#94a3b8"),
    "Hammer":             ("triangle-up",     "#22d3ee"),
    "Shooting Star":      ("triangle-down",   "#f97316"),
    "Engulfing":          ("star",            "#facc15"),
    "Morning Star":       ("star-triangle-up","#4ade80"),
    "Evening Star":       ("star-triangle-down","#f87171"),
    "Double Top":         ("x",               "#ef4444"),
    "Double Bottom":      ("cross",           "#22c55e"),
    "Golden Cross":       ("star-square",     "#fbbf24"),
    "Death Cross":        ("bowtie",          "#a78bfa"),
    "RSI Oversold Bounce":("arrow-up",        "#34d399"),
    "RSI Overbought Drop":("arrow-down",      "#fb7185"),
    "MACD Bullish Cross": ("arrow-bar-up",    "#6ee7b7"),
    "MACD Bearish Cross": ("arrow-bar-down",  "#fca5a5"),
    "BB Squeeze":         ("circle",          "#cbd5e1"),
}
 
DIRECTION_COLOR = {
    "bullish": "#22c55e",
    "bearish": "#ef4444",
    "neutral": "#94a3b8",
}
 
# ── Data loading ─────────────────────────────────────────────────────────────
 
def load_data(conn: sqlite3.Connection, symbol: str, res: str,
              days: Optional[int] = None) -> tuple:
    """Return (bars_with_indicators, patterns) DataFrames."""
 
    # Bars + indicators (LEFT JOIN so we always have price data)
    bars_q = f"""
        SELECT  b.timestamp, b.open, b.high, b.low, b.close, b.volume,
                i.sma_20, i.sma_50, i.sma_200,
                i.rsi_14,
                i.macd, i.macd_signal, i.macd_hist,
                i.bb_upper, i.bb_middle, i.bb_lower, i.bb_width, i.bb_pct
        FROM    bars_{res}      AS b
        LEFT JOIN indicators_{res} AS i
               ON  b.symbol    = i.symbol
               AND b.timestamp = i.timestamp
        WHERE   b.symbol = ?
        ORDER BY b.timestamp ASC
    """
    bars = pd.read_sql(bars_q, conn, params=(symbol,))
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
 
    # Patterns
    pat_q = f"""
        SELECT timestamp, pattern, direction, confidence
        FROM   patterns_{res}
        WHERE  symbol = ?
        ORDER BY timestamp ASC
    """
    patterns = pd.read_sql(pat_q, conn, params=(symbol,))
    patterns["timestamp"] = pd.to_datetime(patterns["timestamp"], utc=True)
 
    # Optional date filter
    if days:
        cutoff = bars["timestamp"].max() - timedelta(days=days)
        bars     = bars[bars["timestamp"] >= cutoff].copy()
        patterns = patterns[patterns["timestamp"] >= cutoff].copy()
 
    return bars, patterns
 
 
# ── Chart builder ─────────────────────────────────────────────────────────────
 
def build_dashboard(bars: pd.DataFrame, patterns: pd.DataFrame,
                    symbol: str, res: str) -> go.Figure:
 
    ts = bars["timestamp"]
 
    # ── Layout: 4 rows (Price+BB, Volume, RSI, MACD) ──────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.12, 0.19, 0.19],
        subplot_titles=("", "Volume", "RSI (14)", "MACD (12/26/9)"),
    )
 
    # ── Row 1 — Candlesticks ──────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=ts, open=bars["open"], high=bars["high"],
        low=bars["low"],  close=bars["close"],
        name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#166534", decreasing_fillcolor="#7f1d1d",
        line_width=1,
    ), row=1, col=1)
 
    # SMAs
    for col, color, dash in [
        ("sma_20",  "#38bdf8", "solid"),
        ("sma_50",  "#fb923c", "dot"),
        ("sma_200", "#a78bfa", "dash"),
    ]:
        if bars[col].notna().any():
            fig.add_trace(go.Scatter(
                x=ts, y=bars[col], name=col.upper().replace("_", " "),
                line=dict(color=color, width=1.2, dash=dash),
                hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
            ), row=1, col=1)
 
    # Bollinger Bands (filled channel)
    if bars["bb_upper"].notna().any():
        fig.add_trace(go.Scatter(
            x=pd.concat([ts, ts.iloc[::-1]]),
            y=pd.concat([bars["bb_upper"], bars["bb_lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(148,163,184,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="BB Band", showlegend=True, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ts, y=bars["bb_upper"], name="BB Upper",
            line=dict(color="#94a3b8", width=0.8, dash="dot"),
            hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ts, y=bars["bb_lower"], name="BB Lower",
            line=dict(color="#94a3b8", width=0.8, dash="dot"),
            hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
        ), row=1, col=1)
 
    # ── Pattern markers on price chart ────────────────────────────────────
    _add_pattern_markers(fig, bars, patterns, row=1)
 
    # ── Row 2 — Volume ───────────────────────────────────────────────────
    colors = ["#166534" if c >= o else "#7f1d1d"
              for c, o in zip(bars["close"], bars["open"])]
    fig.add_trace(go.Bar(
        x=ts, y=bars["volume"],
        marker_color=colors, name="Volume",
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)
 
    # ── Row 3 — RSI ───────────────────────────────────────────────────────
    if bars["rsi_14"].notna().any():
        fig.add_trace(go.Scatter(
            x=ts, y=bars["rsi_14"], name="RSI 14",
            line=dict(color="#f472b6", width=1.5),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=3, col=1)
        # Reference lines
        for level, color, label in [(70, "#ef4444", "OB 70"), (30, "#22c55e", "OS 30")]:
            fig.add_hline(y=level, line_color=color, line_dash="dot",
                          line_width=0.8, row=3, col=1,
                          annotation_text=label, annotation_font_color=color,
                          annotation_font_size=9)
        fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.05,
                      row=3, col=1, line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="#22c55e", opacity=0.05,
                      row=3, col=1, line_width=0)
 
    # ── Row 4 — MACD ──────────────────────────────────────────────────────
    if bars["macd"].notna().any():
        hist_colors = ["#22c55e" if v >= 0 else "#ef4444"
                       for v in bars["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=ts, y=bars["macd_hist"], name="MACD Hist",
            marker_color=hist_colors, opacity=0.6,
            hovertemplate="Hist: %{y:.4f}<extra></extra>",
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=ts, y=bars["macd"], name="MACD",
            line=dict(color="#38bdf8", width=1.4),
            hovertemplate="MACD: %{y:.4f}<extra></extra>",
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=ts, y=bars["macd_signal"], name="Signal",
            line=dict(color="#fb923c", width=1.4, dash="dot"),
            hovertemplate="Signal: %{y:.4f}<extra></extra>",
        ), row=4, col=1)
        fig.add_hline(y=0, line_color="#475569", line_width=0.6, row=4, col=1)
 
    # ── Layout polish ─────────────────────────────────────────────────────
    n_bars   = len(bars)
    date_rng = (ts.min().date(), ts.max().date()) if n_bars else ("", "")
    n_pat    = len(patterns)
 
    fig.update_layout(
        title=dict(
            text=(f"<b>{symbol}</b> — {res.capitalize()} — "
                  f"{date_rng[0]} → {date_rng[1]}  "
                  f"<span style='font-size:13px;color:#64748b'>"
                  f"({n_bars:,} bars · {n_pat:,} patterns)</span>"),
            font=dict(size=18, color="#f1f5f9"),
            x=0.01,
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="'IBM Plex Mono', monospace", size=11, color="#94a3b8"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155",
            borderwidth=1, font=dict(size=10),
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=40, t=80, b=40),
        height=900,
    )
 
    # Axis styles
    axis_style = dict(
        gridcolor="#1e293b", zerolinecolor="#334155",
        tickfont=dict(size=10, color="#64748b"),
    )
    for i in range(1, 5):
        fig.update_xaxes(axis_style, row=i, col=1)
        fig.update_yaxes(axis_style, row=i, col=1)
 
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                     title_font=dict(size=10))
    fig.update_yaxes(title_text="Vol",  row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="RSI",  row=3, col=1, range=[0, 100],
                     title_font=dict(size=10))
    fig.update_yaxes(title_text="MACD", row=4, col=1, title_font=dict(size=10))
 
    return fig
 
 
def _add_pattern_markers(fig: go.Figure, bars: pd.DataFrame,
                          patterns: pd.DataFrame, row: int) -> None:
    """One scatter trace per pattern type, markers placed at candle high/low."""
    if patterns.empty:
        return
 
    # Merge to get price at each pattern timestamp
    merged = patterns.merge(
        bars[["timestamp", "high", "low", "close"]],
        on="timestamp", how="left"
    ).dropna(subset=["close"])
 
    for pattern, grp in merged.groupby("pattern"):
        style = PATTERN_STYLES.get(pattern, ("circle", "#94a3b8"))
        shape, _ = style
 
        # Bullish → marker below low; bearish → above high; neutral → at close
        direction = grp["direction"].iloc[0]
        if direction == "bullish":
            y_vals = grp["low"] * 0.995
            pos = "below"
        elif direction == "bearish":
            y_vals = grp["high"] * 1.005
            pos = "above"
        else:
            y_vals = grp["close"]
            pos = "middle center"
 
        color = DIRECTION_COLOR.get(direction, "#94a3b8")
 
        # Hover text
        hover = [
            f"<b>{row_['pattern']}</b><br>"
            f"Dir: {row_['direction']}<br>"
            f"Conf: {row_['confidence']}<br>"
            f"Time: {row_['timestamp'].strftime('%Y-%m-%d %H:%M')}"
            for _, row_ in grp.iterrows()
        ]
 
        fig.add_trace(go.Scatter(
            x=grp["timestamp"],
            y=y_vals,
            mode="markers+text",
            marker=dict(symbol=shape, size=10, color=color,
                        line=dict(color="#0f172a", width=0.5)),
            text=[pattern[0]] * len(grp),          # first letter as tiny label
            textposition=pos,
            textfont=dict(size=7, color=color),
            name=pattern,
            hovertext=hover,
            hoverinfo="text",
            legendgroup=f"pat_{pattern}",
            showlegend=True,
        ), row=row, col=1)
 
 
# ── Comparison overlay (two date ranges on same chart) ──────────────────────
 
def build_comparison(conn: sqlite3.Connection, symbol: str, res: str,
                     split_date: str) -> go.Figure:
    """
    Split the price series at split_date into 'backtest' and 'live' windows.
    Overlays normalised close prices so you can compare shape and drawdowns.
    """
    bars, patterns = load_data(conn, symbol, res)
    if bars.empty:
        print(f"  No data for {symbol}/{res}")
        return go.Figure()
 
    split = pd.Timestamp(split_date, tz="UTC")
    hist  = bars[bars["timestamp"] <  split].copy()
    live  = bars[bars["timestamp"] >= split].copy()
 
    if hist.empty or live.empty:
        print(f"  Split date {split_date} is outside the data range. "
              f"Range: {bars['timestamp'].min().date()} → {bars['timestamp'].max().date()}")
        return go.Figure()
 
    # Normalise to 100 at each window start
    hist["norm"] = hist["close"] / hist["close"].iloc[0] * 100
    live["norm"] = live["close"] / live["close"].iloc[0] * 100
 
    # Align x-axis by bar index rather than calendar date
    hist = hist.reset_index(drop=True)
    live = live.reset_index(drop=True)
 
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["norm"],
        name=f"Backtest  (before {split_date})",
        line=dict(color="#38bdf8", width=2),
        hovertemplate="Bar %{x}: %{y:.2f}<extra>Backtest</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=live.index, y=live["norm"],
        name=f"Live data (from  {split_date})",
        line=dict(color="#f472b6", width=2),
        hovertemplate="Bar %{x}: %{y:.2f}<extra>Live</extra>",
    ))
 
    # RSI overlay (secondary y)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["rsi_14"],
        name="RSI — Backtest", yaxis="y2",
        line=dict(color="#38bdf8", width=1, dash="dot"), opacity=0.5,
        hovertemplate="RSI: %{y:.1f}<extra>BT</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=live.index, y=live["rsi_14"],
        name="RSI — Live", yaxis="y2",
        line=dict(color="#f472b6", width=1, dash="dot"), opacity=0.5,
        hovertemplate="RSI: %{y:.1f}<extra>Live</extra>",
    ))
 
    # Pattern count comparison
    hist_pats = patterns[patterns["timestamp"] < split]
    live_pats = patterns[patterns["timestamp"] >= split]
    pat_compare = (
        pd.concat([
            hist_pats["pattern"].value_counts().rename("backtest"),
            live_pats["pattern"].value_counts().rename("live"),
        ], axis=1).fillna(0).astype(int)
    )
 
    annotation_lines = [f"<b>Pattern comparison  (backtest vs live)</b>"]
    for pat, row in pat_compare.iterrows():
        annotation_lines.append(f"  {pat:<26} {row['backtest']:>4} vs {row['live']:<4}")
    annotation_text = "<br>".join(annotation_lines)
 
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> — Backtest vs Live  ({split_date} split)",
            font=dict(size=17, color="#f1f5f9"), x=0.01,
        ),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="'IBM Plex Mono', monospace", size=11, color="#94a3b8"),
        yaxis=dict(title="Normalised Price (base=100)",
                   gridcolor="#1e293b", title_font=dict(size=10)),
        yaxis2=dict(title="RSI", overlaying="y", side="right",
                    range=[0, 100], showgrid=False,
                    tickfont=dict(size=9, color="#64748b")),
        xaxis=dict(title="Bar index", gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155",
                    borderwidth=1, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=60, r=80, t=80, b=60),
        height=600,
        annotations=[dict(
            xref="paper", yref="paper", x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            text=annotation_text,
            align="left",
            showarrow=False,
            font=dict(family="'IBM Plex Mono', monospace", size=9, color="#94a3b8"),
            bgcolor="rgba(15,23,42,0.7)", bordercolor="#334155",
            borderpad=8,
        )],
    )
    return fig
 
 
# ── Patterns heatmap ─────────────────────────────────────────────────────────
 
def build_pattern_heatmap(conn: sqlite3.Connection, symbol: str, res: str) -> go.Figure:
    """Calendar heatmap of total patterns per day."""
    pat_q = f"""
        SELECT date(timestamp) AS day, pattern, direction, COUNT(*) AS n
        FROM   patterns_{res}
        WHERE  symbol = ?
        GROUP BY day, pattern, direction
        ORDER BY day ASC
    """
    df = pd.read_sql(pat_q, conn, params=(symbol,))
    if df.empty:
        return go.Figure()
 
    pivot = df.pivot_table(index="pattern", columns="day", values="n",
                           aggfunc="sum", fill_value=0)
 
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#0f172a"],
            [0.15, "#1e3a5f"],
            [0.40, "#1d4ed8"],
            [0.70, "#06b6d4"],
            [1.0,  "#f0f9ff"],
        ],
        hovertemplate="Date: %{x}<br>Pattern: %{y}<br>Count: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title="Signals", tickfont=dict(size=9, color="#94a3b8"),
            title_font=dict(size=10, color="#94a3b8"),
        ),
    ))
 
    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> — Pattern Frequency Heatmap ({res})",
                   font=dict(size=16, color="#f1f5f9"), x=0.01),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(family="'IBM Plex Mono', monospace", size=10, color="#94a3b8"),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=160, r=40, t=70, b=80),
        height=max(350, 30 * len(pivot.index) + 100),
    )
    return fig
 
 
# ── HTML export (multi-chart page) ──────────────────────────────────────────
 
def export_html(figs: list, out_path: Path) -> None:
    """Combine multiple Plotly figures into a single dark-themed HTML page."""
    divs = []
    scripts = []
    for label, fig in figs:
        div_id = label.replace(" ", "_").lower()
        html_chunk = fig.to_html(full_html=False, include_plotlyjs=False,
                                 div_id=div_id)
        divs.append(f'<h2 class="section-title">{label}</h2>\n{html_chunk}')
 
    plotly_cdn = (
        '<script src="https://cdn.plot.ly/plotly-2.30.0.min.js">'
        '</script>'
    )
 
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TA Dashboard</title>
{plotly_cdn}
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap"
      rel="stylesheet"/>
<style>
  *       {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body    {{ background: #020617; color: #94a3b8;
             font-family: 'IBM Plex Mono', monospace;
             padding: 24px 32px; }}
  h1      {{ color: #f1f5f9; font-size: 1.4rem; margin-bottom: 4px; }}
  .meta   {{ color: #475569; font-size: 0.78rem; margin-bottom: 32px; }}
  .section-title {{
             color: #64748b; font-size: 0.72rem; text-transform: uppercase;
             letter-spacing: 0.12em; margin: 40px 0 8px;
             padding-bottom: 6px; border-bottom: 1px solid #1e293b; }}
  .chart-wrap {{ border-radius: 8px; overflow: hidden;
                 border: 1px solid #1e293b; margin-bottom: 8px; }}
  .js-plotly-plot {{ border-radius: 8px; }}
</style>
</head>
<body>
<h1>Technical Analysis Dashboard</h1>
<p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
{''.join(f'<div class="chart-wrap">{d}</div>' for d in divs)}
</body>
</html>"""
 
    out_path.write_text(page, encoding="utf-8")
    print(f"  Saved → {out_path.resolve()}")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(description="TA Visualization Dashboard")
    parser.add_argument("--symbol",  default=None,
                        help="Symbol to visualize (default: all in SYMBOLS)")
    parser.add_argument("--res",     default="daily",
                        choices=["daily", "hourly"],
                        help="Resolution (default: daily)")
    parser.add_argument("--days",    type=int, default=None,
                        help="Only show last N days of data")
    parser.add_argument("--compare", default=None, metavar="YYYY-MM-DD",
                        help="Show backtest-vs-live comparison split at this date")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for HTML files (default: .)")
    args = parser.parse_args()
 
    if not DB_PATH.exists():
        print(f"  Database not found: {DB_PATH.resolve()}")
        print("   Run fetch_alpaca_data.py then 01_back_ana.py first.")
        return
 
    symbols  = [args.symbol] if args.symbol else SYMBOLS
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    with sqlite3.connect(DB_PATH) as conn:
        for symbol in symbols:
            print(f"\n Building dashboard for {symbol} [{args.res}]...")
            figs: list = []
 
            # Main price + indicators chart
            bars, patterns = load_data(conn, symbol, args.res, args.days)
            if bars.empty:
                print(f"  No bar data found — skipping {symbol}")
                continue
 
            figs.append((
                f"{symbol} — Price, Indicators & Patterns",
                build_dashboard(bars, patterns, symbol, args.res),
            ))
 
            # Pattern frequency heatmap
            heat_fig = build_pattern_heatmap(conn, symbol, args.res)
            if heat_fig.data:
                figs.append((f"{symbol} — Pattern Heatmap", heat_fig))
 
            # Backtest comparison (if requested)
            if args.compare:
                cmp_fig = build_comparison(conn, symbol, args.res, args.compare)
                if cmp_fig.data:
                    figs.append((
                        f"{symbol} — Backtest vs Live ({args.compare})",
                        cmp_fig,
                    ))
 
            suffix = f"_last{args.days}d" if args.days else ""
            out_file = out_dir / f"ta_dashboard_{symbol}_{args.res}{suffix}.html"
            export_html(figs, out_path=out_file)
 
    print("\n Done. Open the HTML file(s) in your browser.")
 
 
if __name__ == "__main__":
    main()