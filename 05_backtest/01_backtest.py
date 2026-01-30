import pandas as pd


def generate_signals(df):
    df = df.copy()
    df['signal'] = 0  # 1 = BUY, -1 = SELL

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        buy = prev['SMA20'] <= prev['SMA50'] and curr['SMA20'] > curr['SMA50']
        sell = prev['SMA20'] >= prev['SMA50'] and curr['SMA20'] < curr['SMA50']

        if buy:
            df.iloc[i, df.columns.get_loc('signal')] = 1
        elif sell:
            df.iloc[i, df.columns.get_loc('signal')] = -1

    return df




def run_backtest(
    df,
    initial_capital=100_000,
    position_size=1  # shares
):
    cash = initial_capital
    position = 0
    entry_price = 0

    trades = []
    equity_curve = []

    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        date = df.index[i]

        # BUY
        if signal == 1 and position == 0:
            position = position_size
            entry_price = price
            entry_date = date

        # SELL
        elif signal == -1 and position > 0:
            exit_price = price
            exit_date = date

            pnl = (exit_price - entry_price) * position

            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': position,
                'pnl': pnl
            })

            cash += pnl
            position = 0

        equity = cash + position * price
        equity_curve.append(equity)

    trades_df = pd.DataFrame(trades)
    df['equity'] = equity_curve

    return df, trades_df



def performance_metrics(df, trades_df, initial_capital):
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
    total_return = total_pnl / initial_capital * 100

    equity = df['equity']
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max

    return {
        'trades': len(trades_df),
        'total_pnl': round(total_pnl, 2),
        'return_pct': round(total_return, 2),
        'max_drawdown_pct': round(drawdown.min() * 100, 2)
    }


df = compute_indicators(df)
df = df[df['SMA50'].notna()]      # critical
df = generate_signals(df)

df, trades_df = run_backtest(
    df,
    initial_capital=100_000,
    position_size=10
)

stats = performance_metrics(df, trades_df, 100_000)

print(stats)
print(trades_df.head())
