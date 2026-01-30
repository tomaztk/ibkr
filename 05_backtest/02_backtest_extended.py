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


### Koncepti


## rule for stopčloss / take-rpofit
#mal nerodno
def apply_stops(price, entry_price, stop_pct, take_pct):
    stop_price = entry_price * (1 - stop_pct) ### Stop
    take_price = entry_price * (1 + take_pct)  ## Take

    if price <= stop_price:
        return 'STOP', stop_price
    if price >= take_price:
        return 'TAKE', take_price
    return None, None

"""exit_reason, exit_price = apply_stops(
    price=price,
    entry_price=entry_price,
    stop_pct=0.02,
    take_pct=0.04
)
"""


# risk positioning and risk sizing

def position_size(capital, risk_pct, entry_price, stop_price):
    risk_amount = capital * risk_pct
    risk_per_share = abs(entry_price - stop_price)
    return int(risk_amount / risk_per_share)

"""
qty = position_size(
    capital=equity,
    risk_pct=0.01,   # 1% risk
    entry_price=entry,
    stop_price=entry * 0.98
)
"""

test_pnl = 1

# get some stocks from parquet filetek :)
train_days = []
test_days = []

## out of sample testing / Backtesting
#sprehodim se po rolling oknu v interval, ki gan e poznam

def walk_forward(df, train_days=252, test_days=63):
    results = []

    for start in range(0, len(df) - train_days - test_days, test_days):
        train = df.iloc[start:start+train_days]
        test = df.iloc[start+train_days:start+train_days+test_days]

        # fit params on train
        # run backtest on test
        results.append(test_pnl)

    return results
