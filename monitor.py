import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import glob

# Use the following code to start from console
# streamlit run monitor.py

# Configure page layout
st.set_page_config(page_title="Backtest Real-time Monitor", layout="wide")

# Define base directory for snapshots
SNAPSHOT_BASE_DIR = r'D:\BTC_ACE\snapshots'

# [SETUP] Unified chart height for visual consistency
CHART_HEIGHT = 450 

st.title("📈 QG-ACE (Quant-Graph Adaptive Cognitive Engine)")

placeholder = st.empty()

def get_latest_snapshot_path(base_dir):
    search_pattern = os.path.join(base_dir, '*', 'time_series_snapshot.csv')
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def load_data():
    file_path = get_latest_snapshot_path(SNAPSHOT_BASE_DIR)
    if file_path is None:
        return None, None
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df, file_path
    except Exception as e:
        return None, file_path

def calculate_metrics(df, initial_equity):
    """Calculate strategy performance metrics: cumulative return, Sharpe ratio, max drawdown, Alpha"""
    if df is None or df.empty or len(df) < 2:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'current_alpha': 0.0
        }

    # 1. Cumulative Return (%)
    final_equity = df['net_value'].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100

    # 2. Max Drawdown (%)
    equity_curve = df['net_value'].values
    running_max = pd.Series(equity_curve).expanding().max().values
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # 3. Sharpe Ratio (assuming daily returns, annualized 252 days)
    daily_returns = df['net_value'].pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
    else:
        sharpe_ratio = 0.0

    # 4. Win Rate (%)
    trades = df[df['decision_type'].isin(['CLOSE', 'CLOSE_TP_HIT', 'CLOSE_SL_HIT'])]
    if len(trades) > 0:
        winning_trades = len(trades[trades['alpha'] > 0])
        win_rate = winning_trades / len(trades) * 100
    else:
        win_rate = 0.0

    # 5. Current Alpha (strategy excess return %)
    initial_price = df['close'].iloc[0]
    benchmark_return = (df['close'].iloc[-1] - initial_price) / initial_price * 100
    current_alpha = total_return - benchmark_return

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'current_alpha': current_alpha
    }

while True:
    df, current_file_path = load_data()
    refresh_id = str(time.time())

    with placeholder.container():
        if df is None or df.empty:
            st.warning(f"Scanning directory: {SNAPSHOT_BASE_DIR} ...\nNo time_series_snapshot.csv file found yet")
        else:
            st.caption(f"📂 Monitoring latest backtest: `{current_file_path}`")
            st.divider()

            # --- 0. Data Preprocessing ---
            initial_equity = df['net_value'].iloc[0]
            initial_price = df['close'].iloc[0]
            df['benchmark_equity'] = (df['close'] / initial_price) * initial_equity

            # Calculate strategy performance metrics
            metrics = calculate_metrics(df, initial_equity)

            # --- 1. Top Core Metrics KPI ---
            latest = df.iloc[-1]
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                excess_return = (latest['net_value'] - latest['benchmark_equity']) / initial_equity * 100
                st.metric("Net Value", f"{latest['net_value']:,.2f}",
                          delta=f"vs Benchmark {excess_return:+.2f}%")
            with col2:
                st.metric("Price", f"{latest['price']:,.2f}")
            with col3:
                pos_pct = latest.get('quantity_pct', 0) * 100
                st.metric("Position %", f"{pos_pct:.1f}%")
            with col4:
                st.metric("Alpha (Excess Return)", f"{metrics['current_alpha']:+.2f}%")
            with col5:
                st.metric("Return", f"{metrics['total_return']:+.2f}%")
            with col6:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")

            # Second row KPIs
            col7, col8 = st.columns(2)
            with col7:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
            with col8:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

            # --- 2. Chart Area ---

            # [Chart A]: Price and Trading Signals
            # ------------------------------------------------------
            fig_trade = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_trade.add_trace(go.Scatter(x=df['timestamp'], y=df['close'],
                                             mode='lines', name='Close Price',
                                             line=dict(color='#636EFA', width=1)), secondary_y=False)

            # Filter out all non-HOLD trading points
            trades = df[df['decision_type'] != 'HOLD']

            if not trades.empty:
                # [KEY MODIFY] Added color mapping for CLOSE_TP_HIT
                # LONG: green, SHORT: red, CLOSE: orange (regular close), CLOSE_TP_HIT: purple (take profit), CLOSE_SL_HIT: gray (stop loss, reserved)
                color_map = {
                    'LONG': 'green',
                    'SHORT': 'red',
                    'CLOSE': 'orange',
                    'CLOSE_TP_HIT': 'purple',  # Added: take profit purple
                    'CLOSE_SL_HIT': 'gray'     # Reserved: stop loss gray
                }

                # Use map to assign colors, default to black for undefined types
                colors = trades['decision_type'].map(color_map).fillna('black')

                fig_trade.add_trace(go.Scatter(
                    x=trades['timestamp'],
                    y=trades['price'],
                    mode='markers',
                    name='Trade Signals',
                    marker=dict(size=9, color=colors, symbol='triangle-up'), # Slightly increased size
                    text=trades['decision_type']
                ), secondary_y=False)

            fig_trade.add_trace(go.Scatter(
                x=df['timestamp'], y=df['quantity_pct'],
                mode='lines', name='Position %',
                line=dict(width=0),
                fill='tozeroy', fillcolor='rgba(128, 128, 128, 0.2)'
            ), secondary_y=True)

            fig_trade.update_layout(title="Price Action and Trading Signals", height=CHART_HEIGHT, margin=dict(l=10, r=60, t=30, b=10))
            fig_trade.update_yaxes(title_text="Price", secondary_y=False)
            fig_trade.update_yaxes(title_text="Position", range=[0, 1.1], secondary_y=True)
            
            st.plotly_chart(fig_trade, use_container_width=True, key=f"trade_chart_{refresh_id}")


            # [Chart B]: Equity Curve Comparison
            # ------------------------------------------------------
            fig_equity = go.Figure()

            fig_equity.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['benchmark_equity'],
                mode='lines',
                name='Benchmark (Buy & Hold)',
                line=dict(color='#A9A9A9', width=1.5, dash='dash')
            ))

            fig_equity.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['net_value'],
                mode='lines',
                name='Strategy Equity',
                line=dict(color='#00CC96', width=2)
            ))

            fig_equity.update_layout(
                title="Strategy Performance vs Benchmark",
                height=CHART_HEIGHT,
                margin=dict(l=10, r=60, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_equity, use_container_width=True, key=f"equity_chart_{refresh_id}")

            # --- 3. Recent Data Display ---
            with st.expander("View Latest Detailed Data", expanded=False):
                st.dataframe(df.tail(10).sort_values('timestamp', ascending=False),
                             use_container_width=True,
                             key=f"data_table_{refresh_id}")

    time.sleep(60)