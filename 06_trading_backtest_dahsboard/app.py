"""
Trading Dashboard - Main Streamlit Application
A comprehensive dashboard for historical data analysis and strategy backtesting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import SYMBOLS, SYMBOL_DESCRIPTIONS, DATA_DIR, TIMEFRAMES
from utils.data_fetcher import AlpacaDataFetcher
from strategies import get_strategy, get_available_strategies, STRATEGIES
from backtest import BacktestEngine, BacktestResult

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #ff1744;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data(symbol: str, timeframe: str = "1Min") -> pd.DataFrame:
    """Load data from parquet file."""
    filepath = DATA_DIR / f"{symbol}_{timeframe}.parquet"
    
    if filepath.exists():
        df = pd.read_parquet(filepath)
        return df
    return pd.DataFrame()


def get_available_data() -> dict:
    """Get information about available data files."""
    fetcher = AlpacaDataFetcher(data_dir=str(DATA_DIR))
    return fetcher.get_available_data()


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    height: int = 600,
    show_volume: bool = True
) -> go.Figure:
    """Create an interactive candlestick chart."""
    
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, 'Volume'),
            row_heights=[0.7, 0.3]
        )
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=(title,))
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#00c853',
            decreasing_line_color='#ff1744'
        ),
        row=1, col=1
    )
    
    # Volume bar chart
    if show_volume and 'volume' in df.columns:
        colors = ['#00c853' if close >= open else '#ff1744' 
                  for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig


def add_indicators_to_chart(
    fig: go.Figure,
    df: pd.DataFrame,
    indicators: list
) -> go.Figure:
    """Add technical indicators to the chart."""
    
    if 'sma_fast' in df.columns and 'SMA Fast' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sma_fast'],
                name='SMA Fast', line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_slow' in df.columns and 'SMA Slow' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sma_slow'],
                name='SMA Slow', line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'bb_upper' in df.columns and 'Bollinger Bands' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_upper'],
                name='BB Upper', line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_lower'],
                name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_middle'],
                name='BB Middle', line=dict(color='gray', width=1)
            ),
            row=1, col=1
        )
    
    # Add buy/sell signals
    if 'buy_signal' in df.columns:
        buy_signals = df[df['buy_signal']]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['low'] * 0.995,
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ),
            row=1, col=1
        )
    
    if 'sell_signal' in df.columns:
        sell_signals = df[df['sell_signal']]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['high'] * 1.005,
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ),
            row=1, col=1
        )
    
    return fig


def create_equity_chart(result: BacktestResult) -> go.Figure:
    """Create equity curve chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Equity Curve', 'Drawdown'),
        row_heights=[0.7, 0.3]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name='Equity',
            line=dict(color='#2196f3', width=2),
            fill='tozeroy',
            fillcolor='rgba(33,150,243,0.1)'
        ),
        row=1, col=1
    )
    
    # Initial capital line
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        row=1, col=1
    )
    
    # Drawdown
    if len(result.drawdown_curve) > 0:
        fig.add_trace(
            go.Scatter(
                x=result.drawdown_curve.index,
                y=result.drawdown_curve.values,
                name='Drawdown',
                line=dict(color='#ff1744', width=1),
                fill='tozeroy',
                fillcolor='rgba(255,23,68,0.2)'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create RSI indicator chart."""
    fig = go.Figure()
    
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['rsi'],
                name='RSI', line=dict(color='purple', width=1)
            )
        )
    
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                      annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        height=200,
        title="RSI Indicator",
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def display_metrics(result: BacktestResult):
    """Display backtest metrics in a nice format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"${result.total_return:,.2f}",
            f"{result.total_return_pct:.2f}%"
        )
        st.metric(
            "Win Rate",
            f"{result.win_rate:.1f}%",
            f"{result.winning_trades}/{result.total_trades} trades"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}"
        )
        st.metric(
            "Profit Factor",
            f"{result.profit_factor:.2f}" if result.profit_factor != float('inf') else "∞"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"-{result.max_drawdown_pct:.2f}%",
            f"-${result.max_drawdown:,.2f}"
        )
        st.metric(
            "Sortino Ratio",
            f"{result.sortino_ratio:.2f}"
        )
    
    with col4:
        st.metric(
            "Avg Trade",
            f"${result.avg_trade:.2f}"
        )
        st.metric(
            "Volatility (Ann.)",
            f"{result.volatility:.2f}%"
        )


def display_trade_table(result: BacktestResult):
    """Display trade history table."""
    if not result.trades:
        st.warning("No trades to display")
        return
    
    trades_data = []
    for t in result.trades:
        if t.exit_time:
            trades_data.append({
                'Entry Time': t.entry_time,
                'Exit Time': t.exit_time,
                'Side': 'LONG' if t.side.value == 1 else 'SHORT',
                'Size': t.size,
                'Entry Price': f"${t.entry_price:.2f}",
                'Exit Price': f"${t.exit_price:.2f}",
                'PnL': f"${t.pnl:.2f}",
                'PnL %': f"{t.pnl_pct:.2f}%",
                'Commission': f"${t.commission:.2f}"
            })
    
    if trades_data:
        df = pd.DataFrame(trades_data)
        st.dataframe(df, use_container_width=True)


def main():
    """Main dashboard application."""
    
    st.title("📈 Trading Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header(" Settings")
        
        # Check for available data
        available_data = get_available_data()
        available_symbols = list(set(info['symbol'] for info in available_data.values()))
        
        if not available_symbols:
            st.warning("No data available. Please run data download first.")
            st.code("python utils/data_fetcher.py")
            available_symbols = SYMBOLS  # Show as options anyway
        
        # Symbol selection
        selected_symbol = st.selectbox(
            "Select Symbol",
            available_symbols,
            format_func=lambda x: f"{x} - {SYMBOL_DESCRIPTIONS.get(x, x)}"
        )
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
            list(TIMEFRAMES.keys()),
            index=0
        )
        
        st.markdown("---")
        st.header("📊 Analysis Options")
        
        # Date range filter
        st.subheader("Date Range")
        
        # Load data to get date range
        df = load_data(selected_symbol, timeframe)
        
        if not df.empty:
            min_date = df.index.min().date()
            max_date = df.index.max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (df.index.date >= start_date) & (df.index.date <= end_date)
                df = df[mask]
        
        st.markdown("---")
        
        # Strategy selection
        st.header("🎯 Strategy")
        selected_strategy = st.selectbox(
            "Select Strategy",
            get_available_strategies()
        )
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        strategy_class = STRATEGIES[selected_strategy]
        strategy_instance = strategy_class()
        default_params = strategy_instance.get_default_params()
        
        strategy_params = {}
        for param, default in default_params.items():
            if isinstance(default, int):
                strategy_params[param] = st.number_input(
                    param.replace('_', ' ').title(),
                    value=default,
                    min_value=1
                )
            elif isinstance(default, float):
                strategy_params[param] = st.number_input(
                    param.replace('_', ' ').title(),
                    value=default,
                    format="%.4f"
                )
        
        st.markdown("---")
        
        # Backtest settings
        st.header(" Backtest Settings")
        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=100000,
            min_value=1000,
            step=1000
        )
        
        commission = st.slider(
            "Commission (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        ) / 100
        
        slippage = st.slider(
            "Slippage (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01
        ) / 100
        
        position_size = st.slider(
            "Position Size (%)",
            min_value=10,
            max_value=100,
            value=95
        ) / 100
        
        run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)
    
    # Main content area
    if df.empty:
        st.warning(f"No data available for {selected_symbol}. Please download data first.")
        
        st.info("""
        ### How to Download Data
        
        1. Copy the `.env.example` file to `.env`
        2. Add your Alpaca API credentials
        3. Run the data fetcher:
        ```bash
        python utils/data_fetcher.py
        ```
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "📈 Backtest Results", "📋 Trade History", "📉 Analysis"])
    
    with tab1:
        st.header(f"{selected_symbol} - {SYMBOL_DESCRIPTIONS.get(selected_symbol, '')}")
        
        # Data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", f"{len(df):,}")
        with col2:
            st.metric("Start Date", df.index.min().strftime('%Y-%m-%d'))
        with col3:
            st.metric("End Date", df.index.max().strftime('%Y-%m-%d'))
        with col4:
            latest_price = df['close'].iloc[-1]
            price_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:.2f}%")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            show_volume = st.checkbox("Show Volume", value=True)
            indicators = st.multiselect(
                "Indicators",
                ["SMA Fast", "SMA Slow", "Bollinger Bands"],
                default=[]
            )
        
        strategy = get_strategy(selected_strategy, strategy_params)
        df_with_signals = strategy.generate_signals(df.copy())
        
        
        fig = create_candlestick_chart(
            df_with_signals,
            title=f"{selected_symbol} Price",
            show_volume=show_volume
        )
        

        if indicators:
            fig = add_indicators_to_chart(fig, df_with_signals, indicators)
        
        st.plotly_chart(fig, use_container_width=True)
        

        if selected_strategy == 'RSI' and 'rsi' in df_with_signals.columns:
            st.plotly_chart(create_rsi_chart(df_with_signals), use_container_width=True)
    
    with tab2:
        if run_backtest or 'backtest_result' in st.session_state:
            if run_backtest:
                # Run backtest
                with st.spinner("Running backtest..."):
                    strategy = get_strategy(selected_strategy, strategy_params)
                    df_with_signals = strategy.generate_signals(df.copy())
                    
                    engine = BacktestEngine(
                        initial_capital=initial_capital,
                        commission=commission,
                        slippage=slippage,
                        position_size=position_size
                    )
                    
                    result = engine.run(df_with_signals)
                    st.session_state['backtest_result'] = result
                    st.session_state['backtest_params'] = {
                        'symbol': selected_symbol,
                        'strategy': selected_strategy,
                        'params': strategy_params
                    }
            
            result = st.session_state.get('backtest_result')
            
            if result:
                st.header(" Backtest Results")
                
                # Display key metrics
                display_metrics(result)
                
                st.markdown("---")
                
                # Equity curve
                st.subheader("Equity Curve")
                fig = create_equity_chart(result)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Trade Statistics")
                    stats_data = {
                        "Metric": [
                            "Total Trades", "Winning Trades", "Losing Trades",
                            "Average Win", "Average Loss", "Largest Win", "Largest Loss",
                            "Max Consecutive Wins", "Max Consecutive Losses",
                            "Avg Trade Duration (min)"
                        ],
                        "Value": [
                            result.total_trades,
                            result.winning_trades,
                            result.losing_trades,
                            f"${result.avg_win:.2f}",
                            f"${result.avg_loss:.2f}",
                            f"${result.largest_win:.2f}",
                            f"${result.largest_loss:.2f}",
                            result.max_consecutive_wins,
                            result.max_consecutive_losses,
                            f"{result.avg_trade_duration:.1f}"
                        ]
                    }
                    st.table(pd.DataFrame(stats_data))
                
                with col2:
                    st.subheader("Cost Analysis")
                    cost_data = {
                        "Cost Type": ["Total Commission", "Total Slippage", "Total Costs"],
                        "Amount": [
                            f"${result.total_commission:.2f}",
                            f"${result.total_slippage:.2f}",
                            f"${result.total_commission + result.total_slippage:.2f}"
                        ]
                    }
                    st.table(pd.DataFrame(cost_data))
                    
                    st.subheader("Risk Metrics")
                    risk_data = {
                        "Metric": ["Annualized Return", "Annualized Volatility", "Calmar Ratio"],
                        "Value": [
                            f"{result.annualized_return:.2f}%",
                            f"{result.volatility:.2f}%",
                            f"{result.calmar_ratio:.2f}"
                        ]
                    }
                    st.table(pd.DataFrame(risk_data))
        else:
            st.info("👈 Configure your strategy and click 'Run Backtest' to see results")
    
    with tab3:
        st.header("Trade History")
        
        result = st.session_state.get('backtest_result')
        if result and result.trades:
            display_trade_table(result)
        else:
            st.info("Run a backtest to see trade history")
    
    with tab4:
        st.header("📉 Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Statistics")
            stats = df['close'].describe()
            st.write(stats)
            
            # Returns distribution
            returns = df['close'].pct_change().dropna()
            
            fig = go.Figure(data=[go.Histogram(x=returns * 100, nbinsx=50)])
            fig.update_layout(
                title="Returns Distribution (%)",
                xaxis_title="Return %",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Volume Statistics")
            if 'volume' in df.columns:
                vol_stats = df['volume'].describe()
                st.write(vol_stats)
                
                # Volume over time
                daily_volume = df['volume'].resample('D').sum()
                fig = go.Figure(data=[
                    go.Bar(x=daily_volume.index, y=daily_volume.values)
                ])
                fig.update_layout(
                    title="Daily Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume"
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
