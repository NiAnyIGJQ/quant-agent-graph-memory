# -*- coding: utf-8 -*-
# main.py
import backtrader as bt
import matplotlib.pyplot as plt
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import logging

from ace_trading.logging_config import setup_logging
from ace_trading.engine import BacktestRunner
from ace_trading.agents.qwen_agent import QwenAgent
from ace_trading.config_graph_vector import (
    USE_VECTOR_DB, USE_GRAPH_DB, VECTOR_DB_MODEL_PATH,
    VECTOR_DB_BATCH_SIZE, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)

if __name__ == '__main__':
    # [FIX] Windows UTF-8 encoding support
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # [NEW] Initialize logging system
    log = setup_logging(log_dir="logs")

    # [NEW] Initialize prompt logging system
    from ace_trading.prompt_logger import init_prompt_logger
    prompt_log = init_prompt_logger(log_dir="prompt_logs")

    # [NEW] Initialize token statistics logging system (shares session_dir with prompt_log)
    from ace_trading.token_logger import init_token_logger
    token_log = init_token_logger(session_dir=prompt_log.get_session_dir())
    print(f"[TOKEN_LOG] Token statistics enabled, saving to: {token_log.session_dir}\n")

    start_time = time.time()
    timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f"[TIME] Program started at: {timestamp_str}")
    log.info(f"Program started at: {timestamp_str}")
    data_dir = "history_data"
    if not os.path.exists(data_dir):
        print("Please run the data download script first!")
        exit()

    print("[AGENT] Initializing multi-period Qwen Agent...")
    agent = QwenAgent()

    # ====================================================================
    # [NEW] Initialize Vector Database
    # ====================================================================
    vec_db = None
    if USE_VECTOR_DB:
        try:
            from ace_trading.LSTM.vector_database import VectorDatabase
            print("[VECTOR_DB] Initializing vector database...")
            vec_db = VectorDatabase(model_path=VECTOR_DB_MODEL_PATH)

            # [FIX] Vector database should start empty
            vec_db.clear_vectors()
            # During backtesting, when a trade is completed (notify_trade triggered),
            # the market snapshot from when the trade was opened is added to the vector database
            # This ensures the vector database only contains market data from completed trades
            # for training and similarity search
            print("[VECTOR_DB] Vector database initialized (empty state)")
            print("[VECTOR_DB] Completed trade data will be gradually added to the vector database during backtesting")

        except Exception as e:
            print(f"[VECTOR_DB_ERROR] Initialization failed: {e}")
            vec_db = None

    # ====================================================================
    # [NEW] Initialize Graph Database
    # ====================================================================
    graph_db = None
    if USE_GRAPH_DB:
        try:
            from ace_trading.graph.quant_graph_manager import QGAGraphManager
            print("[GRAPH_DB] Initializing graph database...")
            URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            USER = os.getenv("NEO4J_USER", "neo4j")
            PASSWORD = os.environ["NEO4J_PASSWORD"]  # Force environment variable setting

            graph_db = QGAGraphManager(URI, USER, PASSWORD)

            # Initialize schema (idempotent)
            print("[GRAPH_DB] Initializing Schema...")
            graph_db.clear_database()
            graph_db.initialize_schema()
            print("[GRAPH_DB] Graph database initialization complete")
        except Exception as e:
            print(f"[GRAPH_DB_ERROR] Initialization failed: {e}")
            graph_db = None

    # To calculate corresponding indicators, actual backtest time will be later than start.
    # Please keep actual time after 2025-03-29, otherwise perpetual contract volume data will be missing
    runner = BacktestRunner(
        agent=agent,
        data_dir=data_dir,
        start="2024-11-01",
        # start="2021-01-01",  # [MODIFY] Start from 2021 to warm up to 2025, ensuring sufficient high-cycle data
        # start="2025-04-01",
        end="2025-11-23",
        cash=100000.0,
        leverage=1, # Enable leverage
    )

    # ====================================================================
    # [NEW] Attach Vector DB and Graph DB to runner
    # ====================================================================
    if vec_db:
        runner.vec_db = vec_db
        print("[SETUP] Vector Database attached to runner")

    if graph_db:
        runner.graph_db = graph_db
        print("[SETUP] Graph Database attached to runner")
    # runner = BacktestRunner(
    #     agent=agent,
    #     data_dir=data_dir,
    #     start="2021-01-01",
    #     end="2021-06-01",
    #     cash=100000.0,
    #     leverage=1.0, # Enable leverage
    #     reflect_period=80 # Reflect every 80 periods
    # )

    # Add indicators (RSI uses standard version)
    runner.add_indicator(bt.indicators.MACDHisto, period_me1=12, period_me2=26, period_signal=9, name='macd',plot=False)
    runner.add_indicator(bt.indicators.RSI_Safe, period=6, name='rsi_6',plot=False)
    runner.add_indicator(bt.indicators.RSI_Safe, period=12, name='rsi_12',plot=False)
    runner.add_indicator(bt.indicators.RSI_Safe, period=24, name='rsi_24',plot=False)

    # Get cerebro and enable T+0
    # [MODIFY] Now supports specifying vector time granularity (vector_tf) independently from backtest time granularity (main_tf)
    # Example: Use 4H data for backtesting, but use 1H data for vector generation and similarity matching
    # cerebro = runner.run(start_backtest=False, main_tf='4h', vector_tf='1h')
    # Current config: Use 4H data for backtesting, 4H data for vector matching
    # [FIX] Changed to start_backtest=True, let run() method execute backtest internally
    try:
        result_tuple = runner.run(start_backtest=True, main_tf='1h', vector_tf='1h')
        # result_tuple = runner.run(start_backtest=True, main_tf='4h', vector_tf='4h')
        # run() now returns (results, cerebro, snapshot_dir) tuple
        if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
            results, cerebro, snapshot_dir = result_tuple
        elif isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            # Compatibility: Old version returns 2 values
            results, cerebro = result_tuple
            snapshot_dir = None
        else:
            # Compatibility: If non-tuple returned, treat as results
            results = result_tuple
            cerebro = None
            snapshot_dir = None
    except Exception as e:
        print(f"[ERROR] Backtest failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("[DONE] Backtest execution completed")

    if results and cerebro is not None:
        strat = results[0]
        end_val = cerebro.broker.getvalue()
        print(f"\n[END] Ending capital: {end_val:.2f}")
        # =========================================
        # [NOTE] Key modification: Move metrics printing before plotting
        # =========================================
        print("\n" + "="*40)
        print("[STATS] Strategy Performance Evaluation")
        print("="*40)

        # 1. Sharpe Ratio (safely retrieve, handle None for insufficient data)
        sharpe = strat.analyzers.sharpe.get_analysis()
        # If backtest time is too short or no trades, sharperatio may be None
        sr_val = sharpe.get('sharperatio')
        if sr_val is None:
            print(f"[•] Sharpe Ratio: N/A (insufficient data)")
        else:
            print(f"[•] Sharpe Ratio: {sr_val:.4f}")

        # 2. Maximum Drawdown
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get('max', {}).get('drawdown', 0.0)
        max_money = drawdown.get('max', {}).get('moneydown', 0.0)
        print(f"[•] Max Drawdown: {max_dd:.2f}%")
        print(f"   - Drawdown amount: {max_money:.2f}")

        # 3. Returns
        returns = strat.analyzers.returns.get_analysis()
        rnorm = returns.get('rnorm100', 0.0)
        print(f"[•] Annual Return: {rnorm:.2f}%")

        # 4. Trade Statistics — Use robust parsing, compatible with different analyzer return formats
        trades = strat.analyzers.trades.get_analysis()
        # Print raw analysis result for debugging (can be commented out if needed)
        # print("\n🔎 [DEBUG] TradeAnalyzer raw output:")
        # try:
        #     import json
        #     print(json.dumps(trades, default=str, ensure_ascii=False, indent=2))
        # except Exception:
        #     print(repr(trades))

        # Helper function: Find first number (int/float) from nested structure
        def _find_number(obj):
            if obj is None:
                return None
            if isinstance(obj, (int, float)):
                return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    res = _find_number(v)
                    if res is not None:
                        return res
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    res = _find_number(v)
                    if res is not None:
                        return res
            return None

        total_trades = _find_number(trades.get('total')) or 0
        won_trades = _find_number(trades.get('won')) or 0
        lost_trades = _find_number(trades.get('lost')) or 0

        # In some versions, total may be a dict with won/lost inside; if total looks like 0, try fallback with won+lost
        try:
            total_trades = int(total_trades)
            won_trades = int(won_trades)
            lost_trades = int(lost_trades)
        except Exception:
            total_trades = int(total_trades) if isinstance(total_trades, (int, float)) else 0
            won_trades = int(won_trades) if isinstance(won_trades, (int, float)) else 0
            lost_trades = int(lost_trades) if isinstance(lost_trades, (int, float)) else 0

        if total_trades == 0 and (won_trades or lost_trades):
            total_trades = won_trades + lost_trades

        win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0

        # Net PnL: TradeAnalyzer's pnl field structure varies across versions/configs, prioritize finding numbers
        pnl_val = _find_number(trades.get('pnl')) or 0.0
        try:
            pnl_net = float(pnl_val)
        except Exception:
            pnl_net = 0.0

        print(f"[•] Trade Statistics:")
        print(f"   - Total trades: {total_trades}")
        print(f"   - Win rate: {win_rate:.2%}")
        print(f"   - Winning trades: {won_trades} | Losing trades: {lost_trades}")
        print(f"   - Net PnL: {pnl_net:.2f}")
        print("="*40 + "\n")

        # [FIX] In main.py's results handling branch, strat is already the strategy instance
        # Use snapshot_dir passed from runner.run() to avoid duplicate directory creation
        # If snapshot_dir is None, save_snapshot will automatically create a new directory
        saved_snapshot_dir = runner.save_snapshot(
            cerebro, strat,
            out_dir=snapshot_dir if snapshot_dir else None,  # Use existing snapshot_dir or None
            include_pickle=False
        )
        print(f"Snapshot saved to: {saved_snapshot_dir}")

    # <--- 3. Calculate elapsed time (also recommended before plotting, otherwise includes viewing time)
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    if elapsed_seconds < 60:
        print(f"\n[TIME] Total elapsed: {elapsed_seconds:.2f} seconds")
    else:
        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        print(f"\n[TIME] Total elapsed: {minutes} min {seconds} sec")

    # [NEW] Generate token statistics report
    print("\n" + "="*80)
    print("[TOKEN_STATS] Generating token consumption statistics report...")
    print("="*80)
    try:
        token_log.print_summary()
        print(f"[TOKEN_STATS] Detailed report saved to: {token_log.session_dir}")
    except Exception as e:
        print(f"[TOKEN_STATS_ERROR] Failed to generate statistics report: {e}")


    print("[STATS] Plotting...")
    if cerebro is not None:
        plt.rcParams['figure.figsize'] = [18, 10]
        # --- [CORE MODIFY] Explicitly specify plotting range ---
        # start: Only plot after 2025-03-30
        # end: Plot until backtest ends
        cerebro.plot(
            iplot=False,
            # style='candlestick',
            subplot=False,
            volume=False,
            # start=datetime.date(2025, 11, day=12), # <--- Set X-axis start point
            start=datetime.date(2025, 3, 28), # <--- Set X-axis start point
            end=datetime.date(2025, 11, 23)   # <--- Set X-axis end point
        )
    else:
        print("[WARN] cerebro is None, skipping plot")
