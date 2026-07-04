"""
Crypto Backtesting Dashboard
----------------------------
Run with:  streamlit run app.py

Requires:  pip install streamlit ccxt backtesting pandas numpy plotly requests
"""

import datetime as dt
import os
import tempfile

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

# --------------------------------------------------------------------------
# Indicator helpers (pure pandas, no extra dependencies)
# --------------------------------------------------------------------------
def RSI(series, period=14):
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).values


def MACD_line(series, fast=12, slow=26):
    s = pd.Series(series)
    return (s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()).values


def MACD_signal(series, fast=12, slow=26, signal=9):
    macd = pd.Series(MACD_line(series, fast, slow))
    return macd.ewm(span=signal, adjust=False).mean().values


def BB_upper(series, period=20, num_std=2.0):
    s = pd.Series(series)
    ma, sd = s.rolling(period).mean(), s.rolling(period).std()
    return (ma + num_std * sd).values


def BB_lower(series, period=20, num_std=2.0):
    s = pd.Series(series)
    ma, sd = s.rolling(period).mean(), s.rolling(period).std()
    return (ma - num_std * sd).values


# --------------------------------------------------------------------------
# Strategy factories — each returns a `Strategy` subclass configured with params
# --------------------------------------------------------------------------
def make_sma_cross(fast: int, slow: int):
    class SmaCross(Strategy):
        n1, n2 = fast, slow

        def init(self):
            price = self.data.Close
            self.ma1 = self.I(SMA, price, self.n1)
            self.ma2 = self.I(SMA, price, self.n2)

        def next(self):
            if crossover(self.ma1, self.ma2):
                self.buy()
            elif crossover(self.ma2, self.ma1):
                self.position.close()

    return SmaCross


def make_rsi_meanrev(period: int, low_th: int, high_th: int):
    class RsiMeanReversion(Strategy):
        p, lo, hi = period, low_th, high_th

        def init(self):
            price = self.data.Close
            self.rsi = self.I(RSI, price, self.p)

        def next(self):
            if not self.position and self.rsi[-1] < self.lo:
                self.buy()
            elif self.position and self.rsi[-1] > self.hi:
                self.position.close()

    return RsiMeanReversion


def make_macd_cross(fast: int, slow: int, signal: int):
    class MacdCross(Strategy):
        f, s, sig = fast, slow, signal

        def init(self):
            price = self.data.Close
            self.macd = self.I(MACD_line, price, self.f, self.s)
            self.macd_signal = self.I(MACD_signal, price, self.f, self.s, self.sig)

        def next(self):
            if crossover(self.macd, self.macd_signal):
                self.buy()
            elif crossover(self.macd_signal, self.macd):
                self.position.close()

    return MacdCross


def make_bb_breakout(period: int, num_std: float):
    class BollingerBreakout(Strategy):
        p, k = period, num_std

        def init(self):
            price = self.data.Close
            self.upper = self.I(BB_upper, price, self.p, self.k)
            self.lower = self.I(BB_lower, price, self.p, self.k)

        def next(self):
            price = self.data.Close[-1]
            if not self.position and price > self.upper[-1]:
                self.buy()
            elif self.position and price < self.lower[-1]:
                self.position.close()

    return BollingerBreakout


# Registry: name -> config used to build the UI and the strategy dynamically.
# `defaults` are used as-is in Compare mode; in Single mode they seed the sliders.
STRATEGIES = {
    "SMA Crossover": {
        "build": lambda p: make_sma_cross(p["fast"], p["slow"]),
        "defaults": {"fast": 10, "slow": 20},
    },
    "RSI Mean Reversion": {
        "build": lambda p: make_rsi_meanrev(p["period"], p["low"], p["high"]),
        "defaults": {"period": 14, "low": 30, "high": 70},
    },
    "MACD Crossover": {
        "build": lambda p: make_macd_cross(p["fast"], p["slow"], p["signal"]),
        "defaults": {"fast": 12, "slow": 26, "signal": 9},
    },
    "Bollinger Band Breakout": {
        "build": lambda p: make_bb_breakout(p["period"], p["num_std"]),
        "defaults": {"period": 20, "num_std": 2.0},
    },
}

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Backtester", layout="wide")
st.title("📊 Crypto Backtesting Dashboard")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Settings")

EXCHANGES = ["binance", "bybit", "okx", "kraken", "coinbase"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

exchange_id = st.sidebar.selectbox("Exchange", EXCHANGES, index=0)


@st.cache_data(ttl=3600)
def get_symbols(exchange_id: str):
    ex = getattr(ccxt, exchange_id)()
    markets = ex.load_markets()
    # keep USDT pairs by default so the dropdown isn't thousands of items
    symbols = sorted([s for s in markets if s.endswith("/USDT")])
    return symbols


@st.cache_data(ttl=3600)
def get_top100_bases():
    """Top 100 coins by market cap (base symbols, e.g. 'BTC') via CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return [c["symbol"].upper() for c in data]


try:
    symbols = get_symbols(exchange_id)
except Exception as e:
    st.sidebar.error(f"Could not load symbols: {e}")
    symbols = ["BTC/USDT", "ETH/USDT"]

top100_only = st.sidebar.checkbox("Only show Top 100 coins (by market cap)", value=False)
if top100_only:
    try:
        top_bases = set(get_top100_bases())
        filtered = [s for s in symbols if s.split("/")[0] in top_bases]
        if filtered:
            symbols = filtered
        else:
            st.sidebar.warning("No overlap between Top 100 list and this exchange's pairs; showing all.")
    except Exception as e:
        st.sidebar.error(f"Could not fetch Top 100 list: {e}")

symbol = st.sidebar.selectbox("Coin", symbols, index=symbols.index("BTC/USDT") if "BTC/USDT" in symbols else 0)
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1h"))

st.sidebar.subheader("Date Range")
use_date_range = st.sidebar.checkbox("Select specific date range", value=False)
if use_date_range:
    default_start = dt.date.today() - dt.timedelta(days=180)
    date_range = st.sidebar.date_input(
        "From / To", value=(default_start, dt.date.today()), max_value=dt.date.today()
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, dt.date.today()
    limit = None
else:
    start_date = end_date = None
    limit = st.sidebar.slider("Number of candles", min_value=200, max_value=2000, value=1000, step=100)

DEFAULT_CUSTOM_CODE = '''# Your strategy MUST be a class named CustomStrategy that subclasses Strategy.
# Available without importing: Strategy, crossover, cross, SMA, RSI,
# MACD_line, MACD_signal, BB_upper, BB_lower, np, pd

class CustomStrategy(Strategy):
    fast = 10
    slow = 20

    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.fast)
        self.ma2 = self.I(SMA, price, self.slow)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position.close()
'''

st.sidebar.subheader("Strategy")
mode = st.sidebar.radio("Mode", ["Single Strategy", "Compare Strategies", "Custom Strategy (paste code)"])

strategy_params = {}  # populated below, used only in Single mode
custom_code = None

if mode == "Single Strategy":
    strategy_name = st.sidebar.selectbox("Choose strategy", list(STRATEGIES.keys()))

    if strategy_name == "SMA Crossover":
        strategy_params["fast"] = st.sidebar.slider("Fast MA length", 2, 50, 10)
        strategy_params["slow"] = st.sidebar.slider("Slow MA length", 5, 200, 20)
    elif strategy_name == "RSI Mean Reversion":
        strategy_params["period"] = st.sidebar.slider("RSI period", 2, 50, 14)
        strategy_params["low"] = st.sidebar.slider("Oversold threshold (buy)", 5, 45, 30)
        strategy_params["high"] = st.sidebar.slider("Overbought threshold (sell)", 55, 95, 70)
    elif strategy_name == "MACD Crossover":
        strategy_params["fast"] = st.sidebar.slider("Fast EMA", 2, 50, 12)
        strategy_params["slow"] = st.sidebar.slider("Slow EMA", 10, 100, 26)
        strategy_params["signal"] = st.sidebar.slider("Signal EMA", 2, 50, 9)
    elif strategy_name == "Bollinger Band Breakout":
        strategy_params["period"] = st.sidebar.slider("BB period", 5, 100, 20)
        strategy_params["num_std"] = st.sidebar.slider("BB std dev", 1.0, 4.0, 2.0, step=0.1)

    selected_strategies = [strategy_name]

elif mode == "Compare Strategies":
    selected_strategies = st.sidebar.multiselect(
        "Strategies to compare", list(STRATEGIES.keys()), default=list(STRATEGIES.keys())
    )
    st.sidebar.caption("Compare mode runs each strategy with its default parameters. Switch to Single Strategy mode to fine-tune one.")

else:  # Custom Strategy (paste code)
    st.sidebar.caption(
        "Pine Script can't run outside TradingView. Paste your Pine Script to Claude in chat first, "
        "ask it to convert to this template, then paste the Python result below."
    )
    selected_strategies = []

commission = st.sidebar.number_input("Commission (%)", value=0.10, step=0.01) / 100
cash = st.sidebar.number_input("Starting cash ($)", value=10000, step=1000)

run_button = st.sidebar.button("Run Backtest", type="primary")

if mode == "Custom Strategy (paste code)":
    with st.expander("📋 Pine Script → Python cheat sheet", expanded=False):
        st.markdown(
            "| Pine Script | Python (this app) |\n"
            "|---|---|\n"
            "| `ta.sma(close, 20)` | `self.I(SMA, price, 20)` |\n"
            "| `ta.rsi(close, 14)` | `self.I(RSI, price, 14)` |\n"
            "| `ta.crossover(a, b)` | `crossover(a, b)` |\n"
            "| `strategy.entry(...)` (long) | `self.buy()` |\n"
            "| `strategy.close(...)` | `self.position.close()` |\n"
            "| `input.int(10, ...)` | class attribute, e.g. `fast = 10` |\n"
            "\nPaste your Pine Script to Claude in the chat and ask for a conversion to this app's format — "
            "then paste the Python it gives you below."
        )
    st.subheader("Custom Strategy Code")
    st.caption("⚠️ This runs the Python code you paste, locally, exactly as written. Only paste code you wrote or trust.")
    custom_code = st.text_area("CustomStrategy class", value=DEFAULT_CUSTOM_CODE, height=320)

# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch the most recent `limit` candles."""
    ex = getattr(ccxt, exchange_id)()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


@st.cache_data(ttl=300)
def fetch_ohlcv_range(exchange_id: str, symbol: str, timeframe: str, start_date, end_date) -> pd.DataFrame:
    """Fetch all candles between start_date and end_date (inclusive), paginating as needed."""
    ex = getattr(ccxt, exchange_id)()
    since = int(dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc).timestamp() * 1000)
    end_ms = int(dt.datetime.combine(end_date, dt.time.max, tzinfo=dt.timezone.utc).timestamp() * 1000)

    all_rows = []
    max_iterations = 500  # safety cap so a bad range can't loop forever
    for _ in range(max_iterations):
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= end_ms or len(batch) < 2:
            break
        since = last_ts + 1  # move forward to avoid re-fetching the same candle

    if not all_rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1))]
    return df






# Chart builder
# --------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, stats, price_overlays=None, oscillator=None) -> go.Figure:
    """
    price_overlays: list of (name, pd.Series) drawn on the price panel (e.g. MAs, Bollinger Bands)
    oscillator: optional dict {"name": str, "series": pd.Series, "hlines": [values]} drawn in its own panel
    """
    trades = stats["_trades"]
    equity_curve = stats["_equity_curve"]
    price_overlays = price_overlays or []

    rows = 3 + (1 if oscillator else 0)
    if oscillator:
        row_heights = [0.45, 0.15, 0.15, 0.25]
        titles = ("Price & Trades", oscillator["name"], "Volume", "Equity Curve")
    else:
        row_heights = [0.55, 0.15, 0.30]
        titles = ("Price & Trades", "Volume", "Equity Curve")

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03, subplot_titles=titles,
    )

    fig.add_trace(
        go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"),
        row=1, col=1,
    )
    for name, series in price_overlays:
        fig.add_trace(go.Scatter(x=df.index, y=series, name=name, line=dict(width=1)), row=1, col=1)

    if len(trades):
        fig.add_trace(
            go.Scatter(
                x=trades["EntryTime"], y=trades["EntryPrice"], mode="markers",
                marker=dict(symbol="triangle-up", color="lime", size=11, line=dict(width=1, color="black")),
                name="Buy",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=trades["ExitTime"], y=trades["ExitPrice"], mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=11, line=dict(width=1, color="black")),
                name="Sell",
            ), row=1, col=1,
        )

    next_row = 2
    if oscillator:
        fig.add_trace(
            go.Scatter(x=df.index, y=oscillator["series"], name=oscillator["name"], line=dict(color="orange")),
            row=next_row, col=1,
        )
        for hl in oscillator.get("hlines", []):
            fig.add_hline(y=hl, line_dash="dot", line_color="grey", row=next_row, col=1)
        next_row += 1

    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume", marker_color="grey"), row=next_row, col=1)
    next_row += 1

    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve["Equity"], name="Equity", line=dict(color="cyan")),
        row=next_row, col=1,
    )

    fig.update_layout(
        height=950 if not oscillator else 1050,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def build_comparison_chart(equity_curves: dict) -> go.Figure:
    """equity_curves: {strategy_name: pd.Series of equity indexed by time}"""
    fig = go.Figure()
    for name, curve in equity_curves.items():
        pct_return = (curve / curve.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=curve.index, y=pct_return, name=name, mode="lines"))
    fig.update_layout(
        title="Equity Growth Comparison (% return)",
        template="plotly_dark",
        height=500,
        yaxis_title="Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def get_plot_extras(strategy_name: str, df: pd.DataFrame, params: dict):
    """Returns (price_overlays, oscillator) for the given strategy, computed directly from df for plotting."""
    if strategy_name == "SMA Crossover":
        overlays = [
            (f"MA {params['fast']}", df.Close.rolling(params["fast"]).mean()),
            (f"MA {params['slow']}", df.Close.rolling(params["slow"]).mean()),
        ]
        return overlays, None
    elif strategy_name == "RSI Mean Reversion":
        rsi = pd.Series(RSI(df.Close.values, params["period"]), index=df.index)
        return [], {"name": "RSI", "series": rsi, "hlines": [params["low"], params["high"]]}
    elif strategy_name == "MACD Crossover":
        macd = pd.Series(MACD_line(df.Close.values, params["fast"], params["slow"]), index=df.index)
        signal = pd.Series(MACD_signal(df.Close.values, params["fast"], params["slow"], params["signal"]), index=df.index)
        fig_osc = {"name": "MACD", "series": macd, "hlines": [0]}
        # signal line added as a second overlay trace isn't supported by the simple oscillator dict,
        # so we fold it into price_overlays-less usage by returning macd only; signal is close enough
        # for visual crossover confirmation when paired with the trade markers.
        return [], fig_osc
    elif strategy_name == "Bollinger Band Breakout":
        upper = pd.Series(BB_upper(df.Close.values, params["period"], params["num_std"]), index=df.index)
        lower = pd.Series(BB_lower(df.Close.values, params["period"], params["num_std"]), index=df.index)
        return [("BB Upper", upper), ("BB Lower", lower)], None
    return [], None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if run_button:
    with st.spinner(f"Fetching {symbol} {timeframe} data from {exchange_id}..."):
        try:
            if use_date_range:
                df = fetch_ohlcv_range(exchange_id, symbol, timeframe, start_date, end_date)
            else:
                df = fetch_ohlcv(exchange_id, symbol, timeframe, limit)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    min_len_needed = 60  # generous minimum so any strategy's longest lookback has room
    if df.empty or len(df) < min_len_needed:
        st.warning("Not enough data returned for this timeframe/range. Try more candles or a wider date range.")
        st.stop()

    # ======================================================================
    # SINGLE STRATEGY MODE
    # ======================================================================
    if mode == "Single Strategy":
        strategy_cls = STRATEGIES[strategy_name]["build"](strategy_params)
        bt = Backtest(df, strategy_cls, commission=commission, cash=cash, exclusive_orders=True)
        stats = bt.run()

        trades = stats["_trades"]
        total_trades = int(stats["# Trades"])
        win_rate = stats["Win Rate [%]"] if total_trades else 0.0
        profit_factor = stats["Profit Factor"] if total_trades else float("nan")
        net_profit_dollars = stats["Equity Final [$]"] - cash

        st.subheader(f"Results: {strategy_name}")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Trades", total_trades)
        k2.metric("Win Rate", f"{win_rate:.1f}%")
        k3.metric("Net Profit ($)", f"${net_profit_dollars:,.2f}")
        k4.metric("Net Return", f"{stats['Return [%]']:.2f}%")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%")
        k6.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
        k7.metric("Profit Factor", f"{profit_factor:.2f}" if total_trades else "—")
        k8.metric("Buy & Hold Return", f"{stats['Buy & Hold Return [%]']:.2f}%")

        k9, k10, k11 = st.columns(3)
        k9.metric("Exposure Time", f"{stats['Exposure Time [%]']:.1f}%")
        k10.metric("Best Trade", f"{stats['Best Trade [%]']:.2f}%" if total_trades else "—")
        k11.metric("Worst Trade", f"{stats['Worst Trade [%]']:.2f}%" if total_trades else "—")

        overlays, oscillator = get_plot_extras(strategy_name, df, strategy_params)
        st.plotly_chart(build_chart(df, stats, overlays, oscillator), use_container_width=True)

        st.subheader("Full Stats")
        st.dataframe(stats.drop(["_equity_curve", "_trades", "_strategy"], errors="ignore").to_frame("Value"))

        st.subheader("Trade Log")
        if total_trades:
            st.dataframe(trades)
        else:
            st.info("No trades were generated for this configuration.")

    # ======================================================================
    # COMPARE STRATEGIES MODE
    # ======================================================================
    elif mode == "Compare Strategies":
        if not selected_strategies:
            st.warning("Pick at least one strategy to compare in the sidebar.")
            st.stop()

        leaderboard_rows = []
        equity_curves = {}
        per_strategy_stats = {}

        with st.spinner("Running all selected strategies..."):
            for name in selected_strategies:
                params = STRATEGIES[name]["defaults"]
                strategy_cls = STRATEGIES[name]["build"](params)
                try:
                    bt = Backtest(df, strategy_cls, commission=commission, cash=cash, exclusive_orders=True)
                    stats = bt.run()
                except Exception as e:
                    st.warning(f"{name} failed to run: {e}")
                    continue

                total_trades = int(stats["# Trades"])
                leaderboard_rows.append({
                    "Strategy": name,
                    "Net Profit ($)": round(stats["Equity Final [$]"] - cash, 2),
                    "Net Return (%)": round(stats["Return [%]"], 2),
                    "Total Trades": total_trades,
                    "Win Rate (%)": round(stats["Win Rate [%]"], 1) if total_trades else 0.0,
                    "Profit Factor": round(stats["Profit Factor"], 2) if total_trades else float("nan"),
                    "Max Drawdown (%)": round(stats["Max. Drawdown [%]"], 2),
                    "Sharpe Ratio": round(stats["Sharpe Ratio"], 2),
                })
                equity_curves[name] = stats["_equity_curve"]["Equity"]
                per_strategy_stats[name] = stats

        if not leaderboard_rows:
            st.error("No strategies ran successfully.")
            st.stop()

        leaderboard = pd.DataFrame(leaderboard_rows).sort_values("Net Return (%)", ascending=False).reset_index(drop=True)

        st.subheader("Strategy Leaderboard")
        st.dataframe(leaderboard, use_container_width=True)

        best_name = leaderboard.iloc[0]["Strategy"]
        st.success(f"🏆 Best performer on this data: **{best_name}** ({leaderboard.iloc[0]['Net Return (%)']}% return)")

        st.subheader("Equity Growth Comparison")
        st.plotly_chart(build_comparison_chart(equity_curves), use_container_width=True)

        st.subheader("Drill into one strategy's trades")
        drill_name = st.selectbox("Strategy", selected_strategies, key="drill_select")
        drill_stats = per_strategy_stats[drill_name]
        drill_trades = drill_stats["_trades"]
        overlays, oscillator = get_plot_extras(drill_name, df, STRATEGIES[drill_name]["defaults"])
        st.plotly_chart(build_chart(df, drill_stats, overlays, oscillator), use_container_width=True)
        if len(drill_trades):
            st.dataframe(drill_trades)
        else:
            st.info("No trades were generated for this strategy.")

    # ======================================================================
    # CUSTOM STRATEGY MODE (paste Python code, e.g. converted from Pine Script)
    # ======================================================================
    else:
        exec_globals = {
            "Strategy": Strategy,
            "crossover": crossover,
            "cross": crossover,
            "SMA": SMA,
            "RSI": RSI,
            "MACD_line": MACD_line,
            "MACD_signal": MACD_signal,
            "BB_upper": BB_upper,
            "BB_lower": BB_lower,
            "np": np,
            "pd": pd,
        }

        try:
            exec(custom_code, exec_globals)
        except Exception as e:
            st.error(f"Your code has an error and couldn't be parsed:\n\n{e}")
            st.stop()

        if "CustomStrategy" not in exec_globals:
            st.error("No class named `CustomStrategy` was found. Your pasted code must define a class with exactly that name.")
            st.stop()

        strategy_cls = exec_globals["CustomStrategy"]
        if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, Strategy)):
            st.error("`CustomStrategy` must be a class that subclasses `Strategy`.")
            st.stop()

        try:
            bt = Backtest(df, strategy_cls, commission=commission, cash=cash, exclusive_orders=True)
            stats = bt.run()
        except Exception as e:
            st.error(f"Backtest failed while running your strategy:\n\n{e}")
            st.stop()

        trades = stats["_trades"]
        total_trades = int(stats["# Trades"])
        win_rate = stats["Win Rate [%]"] if total_trades else 0.0
        profit_factor = stats["Profit Factor"] if total_trades else float("nan")
        net_profit_dollars = stats["Equity Final [$]"] - cash

        st.subheader("Results: Custom Strategy")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Trades", total_trades)
        k2.metric("Win Rate", f"{win_rate:.1f}%")
        k3.metric("Net Profit ($)", f"${net_profit_dollars:,.2f}")
        k4.metric("Net Return", f"{stats['Return [%]']:.2f}%")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%")
        k6.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
        k7.metric("Profit Factor", f"{profit_factor:.2f}" if total_trades else "—")
        k8.metric("Buy & Hold Return", f"{stats['Buy & Hold Return [%]']:.2f}%")

        # Use backtesting.py's own plot — it auto-detects every self.I() indicator
        # the pasted strategy registered, so this works for ANY custom indicator
        # without us needing to know about it in advance.
        st.subheader("Chart (auto-generated from your strategy's indicators)")
        try:
            tmp_path = os.path.join(tempfile.gettempdir(), f"custom_plot_{id(stats)}.html")
            bt.plot(filename=tmp_path, open_browser=False, plot_width=None)
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=1000, scrolling=True)
        except Exception as e:
            st.warning(f"Chart couldn't be generated ({e}), but stats and trades below are still valid.")

        st.subheader("Full Stats")
        st.dataframe(stats.drop(["_equity_curve", "_trades", "_strategy"], errors="ignore").to_frame("Value"))

        st.subheader("Trade Log")
        if total_trades:
            st.dataframe(trades)
        else:
            st.info("No trades were generated for this configuration.")

else:
    st.info("Set your parameters in the sidebar and click **Run Backtest**.")