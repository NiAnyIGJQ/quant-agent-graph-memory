# ace_trading/engine.py
import backtrader as bt
import pandas as pd
import os
import csv
import json
import shutil
from typing import Any, Optional
from datetime import datetime
from ace_trading.framework import MarginCommInfo


class TradeDecisionLogger:
    """
    【实时交易决策日志记录器】
    记录所有的 Agent 决策、订单执行状态、成交结果等，实时写入文件
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = None
        self.csv_writer = None
        self.decision_queue = {}  # 存储待执行的决策 {decision_id: decision_info}

        # 初始化日志文件
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            write_header = not os.path.exists(log_path)
            self.log_file = open(log_path, 'a', newline='', encoding='utf-8')

            fieldnames = [
                'timestamp',           # 决策时间
                'bar_count',          # K线编号
                'decision_type',      # LONG/SHORT/CLOSE/HOLD
                'decision_price',     # 决策时价格
                'decision_quantity',  # 决策仓位比例
                'take_profit',        # 止盈价
                'stop_loss',          # 止损价
                'decision_reason',    # 决策原因（简短）
                'order_status',       # 订单状态（PENDING/SUBMITTED/FILLED/FAILED/CANCELED）
                'order_executed_price', # 成交价
                'order_executed_size',  # 成交数量
                'position_before',    # 执行前持仓
                'position_after',     # 执行后持仓
                'realized_pnl',       # 平仓盈亏
                'error_msg',          # 错误信息（如有）
                'notes',              # 其他备注
                'similar_pattern_found' # 【新增】是否匹配到相似历史行情
            ]
            self.csv_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
            if write_header:
                self.csv_writer.writeheader()
                self.log_file.flush()
        except Exception as e:
            print(f"[LOG_ERROR] Failed to initialize trade decision logger: {e}")
            self.log_file = None
            self.csv_writer = None

    def log_decision(self, timestamp: str, bar_count: int, decision_type: str,
                     decision_price: float, decision_quantity: float,
                     take_profit: float = 0.0, stop_loss: float = 0.0,
                     reason: str = "", position_before: float = 0.0,
                     similar_pattern_found: bool = False):
        """
        记录 Agent 的交易决策
        """
        if not self.csv_writer:
            return

        try:
            row = {
                'timestamp': timestamp,
                'bar_count': bar_count,
                'decision_type': decision_type,
                'decision_price': f"{decision_price:.2f}",
                'decision_quantity': f"{decision_quantity:.2%}",
                'take_profit': f"{take_profit:.2f}" if take_profit > 0 else "N/A",
                'stop_loss': f"{stop_loss:.2f}" if stop_loss > 0 else "N/A",
                'decision_reason': reason[:100],  # 限制长度
                'order_status': 'PENDING',
                'order_executed_price': 'N/A',
                'order_executed_size': 'N/A',
                'position_before': f"{position_before:.4f}",
                'position_after': 'N/A',
                'realized_pnl': 'N/A',
                'error_msg': '',
                'notes': f'Decision made at bar {bar_count}',
                'similar_pattern_found': 'YES' if similar_pattern_found else 'NO'
            }
            self.csv_writer.writerow(row)
            self.log_file.flush()

            # 存储决策ID以便后续跟踪
            decision_id = f"{timestamp}_{bar_count}_{decision_type}"
            self.decision_queue[decision_id] = row
        except Exception as e:
            print(f"[LOG_ERROR] Failed to log decision: {e}")

    def log_order_execution(self, timestamp: str, order_type: str, is_buy: bool,
                           executed_price: float, executed_size: float,
                           position_before: float, position_after: float,
                           error_msg: str = ""):
        """
        记录订单执行结果
        """
        if not self.csv_writer:
            return

        try:
            status = 'FAILED' if error_msg else 'FILLED'
            row = {
                'timestamp': timestamp,
                'bar_count': '',
                'decision_type': 'SYSTEM_ORDER',
                'decision_price': 'N/A',
                'decision_quantity': 'N/A',
                'take_profit': 'N/A',
                'stop_loss': 'N/A',
                'decision_reason': f"{order_type}: {['Sell', 'Buy'][is_buy]}",
                'order_status': status,
                'order_executed_price': f"{executed_price:.2f}" if executed_price > 0 else 'N/A',
                'order_executed_size': f"{executed_size:.4f}",
                'position_before': f"{position_before:.4f}",
                'position_after': f"{position_after:.4f}",
                'realized_pnl': 'N/A',
                'error_msg': error_msg,
                'notes': f"{order_type} order execution",
                'similar_pattern_found': 'N/A'
            }
            self.csv_writer.writerow(row)
            self.log_file.flush()
        except Exception as e:
            print(f"[LOG_ERROR] Failed to log order execution: {e}")

    def log_trade_close(self, timestamp: str, bar_count: int, trade_direction: str,
                       realized_pnl: float, duration_bars: int, position_before: float):
        """
        记录平仓结果
        """
        if not self.csv_writer:
            return

        try:
            outcome = 'WIN' if realized_pnl > 0 else ('LOSS' if realized_pnl < 0 else 'BREAK_EVEN')
            row = {
                'timestamp': timestamp,
                'bar_count': bar_count,
                'decision_type': 'CLOSE',
                'decision_price': 'N/A',
                'decision_quantity': 'N/A',
                'take_profit': 'N/A',
                'stop_loss': 'N/A',
                'decision_reason': f"Trade closed: {outcome}",
                'order_status': 'COMPLETED',
                'order_executed_price': 'N/A',
                'order_executed_size': 'N/A',
                'position_before': f"{position_before:.4f}",
                'position_after': '0.0000',
                'realized_pnl': f"{realized_pnl:+.2f}",
                'error_msg': '',
                'notes': f"{trade_direction} trade closed, duration: {duration_bars} bars, outcome: {outcome}",
                'similar_pattern_found': 'N/A'
            }
            self.csv_writer.writerow(row)
            self.log_file.flush()
        except Exception as e:
            print(f"[LOG_ERROR] Failed to log trade close: {e}")

    def log_error(self, timestamp: str, bar_count: int, decision_type: str,
                 error_msg: str, decision_price: float = 0.0):
        """
        记录执行错误
        """
        if not self.csv_writer:
            return

        try:
            row = {
                'timestamp': timestamp,
                'bar_count': bar_count,
                'decision_type': decision_type,
                'decision_price': f"{decision_price:.2f}" if decision_price > 0 else 'N/A',
                'decision_quantity': 'N/A',
                'take_profit': 'N/A',
                'stop_loss': 'N/A',
                'decision_reason': 'Error occurred',
                'order_status': 'FAILED',
                'order_executed_price': 'N/A',
                'order_executed_size': 'N/A',
                'position_before': 'N/A',
                'position_after': 'N/A',
                'realized_pnl': 'N/A',
                'error_msg': error_msg[:200],
                'notes': f"Error during {decision_type} execution",
                'similar_pattern_found': 'N/A'
            }
            self.csv_writer.writerow(row)
            self.log_file.flush()
        except Exception as e:
            print(f"[LOG_ERROR] Failed to log error: {e}")

    def close(self):
        """关闭日志文件"""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass

class FundingRateData(bt.feeds.PandasData):
    """资金费率数据源"""
    lines = ('funding_rate',) 
    params = (
        ('funding_rate', 'funding_rate'), # 自动匹配 CSV 中的列
        ('open', None), ('high', None), ('low', None), ('close', None), ('volume', None), ('openinterest', None),
    )
    plotinfo = dict(plot=False)

class OpenInterestData(bt.feeds.PandasData):
    """持仓量数据源"""
    lines = ('oi_usd', 'oi_contracts') # 定义两条新线：美元价值 和 合约张数
    params = (
        ('oi_usd', 'OI_Usd'),          # 对应 CSV 列名 'OI_Usd'
        ('oi_contracts', 'OI_Contracts'), # 对应 CSV 列名 'OI_Contracts'
        ('open', None), ('high', None), ('low', None), ('close', None), ('volume', None), ('openinterest', None),
    )
    plotinfo = dict(plot=False)


class MarketSentimentFormatter:
    """
    【市场情绪数据格式化工具】
    统一处理永续合约资金费率和持仓量数据的展示与解释
    """
    
    @staticmethod
    def format_funding_rate(funding_rate_list: list, limit: int = 5) -> dict:
        """
        格式化资金费率数据
        
        返回:
        {
            'current': 当前费率值,
            'display': "0.05%" 或 "-0.03%" 格式,
            'trend': "↑" / "↓" / "→" (上升/下降/持平),
            'interpretation': 市场意义解读,
            'status': "高位多头拥挤" / "低位空头拥挤" / "中性"
        }
        """
        if not funding_rate_list or len(funding_rate_list) == 0:
            return {
                'current': 0,
                'display': "N/A",
                'trend': "→",
                'interpretation': "资金费率数据不足",
                'status': "N/A"
            }
        
        current_fr = funding_rate_list[-1]
        prev_fr = funding_rate_list[-2] if len(funding_rate_list) > 1 else current_fr
        
        # 计算趋势
        if current_fr > prev_fr:
            trend = "↑"
        elif current_fr < prev_fr:
            trend = "↓"
        else:
            trend = "→"
        
        # 格式化显示
        fr_display = f"{current_fr*100:+.3f}%"
        
        # 市场意义解读
        if current_fr > 0.05:
            status = "多头拥挤"
            interpretation = f"正值过高({fr_display})代表多头严重拥挤，潜在回调风险。短线做空风险低，但需警惕反弹。"
        elif current_fr > 0.02:
            status = "轻度多头拥挤"
            interpretation = f"正值适度({fr_display})表示多头有轻微拥挤，可维持多头持仓但需防守。"
        elif current_fr > 0:
            status = "中立偏多"
            interpretation = f"小幅正值({fr_display})表示多头略占优势，交易环境略利多。"
        elif current_fr > -0.02:
            status = "中性"
            interpretation = f"接近零值({fr_display})表示多空力量平衡，市场处于中立状态。"
        elif current_fr > -0.05:
            status = "轻度空头拥挤"
            interpretation = f"负值适度({fr_display})表示空头有轻微拥挤，可维持空头但需防守。"
        else:
            status = "空头拥挤"
            interpretation = f"负值过低({fr_display})代表空头严重拥挤，潜在轧空风险。短线做多风险低，但需警惕下跌。"
        
        return {
            'current': float(current_fr),
            'display': fr_display,
            'trend': trend,
            'interpretation': interpretation,
            'status': status
        }
    
    @staticmethod
    def format_open_interest(oi_usd_list: list, oi_contracts_list: list = None, limit: int = 5) -> dict:
        """
        格式化持仓量数据
        
        返回:
        {
            'current_usd': 当前USD持仓量,
            'current_contracts': 当前合约张数,
            'display_usd': "1.23B" 格式化显示,
            'display_contracts': "45.67K" 格式化显示,
            'change_pct': 变化百分比,
            'trend': "↑增加" / "↓减少" / "→持平",
            'interpretation': 市场意义解读,
            'price_oi_signal': 价格与OI关系分析
        }
        """
        result = {
            'current_usd': 0,
            'current_contracts': 0,
            'display_usd': "N/A",
            'display_contracts': "N/A",
            'change_pct': 0.0,
            'trend': "→",
            'interpretation': "持仓量数据不足",
            'price_oi_signal': "N/A"
        }
        
        if not oi_usd_list or len(oi_usd_list) == 0:
            return result
        
        current_oi_usd = oi_usd_list[-1]
        prev_oi_usd = oi_usd_list[-2] if len(oi_usd_list) > 1 else current_oi_usd
        
        # 计算变化百分比
        change_pct = ((current_oi_usd - prev_oi_usd) / prev_oi_usd * 100) if prev_oi_usd != 0 else 0
        
        # 计算趋势
        if change_pct > 0.5:
            trend = "↑增加"
        elif change_pct < -0.5:
            trend = "↓减少"
        else:
            trend = "→持平"
        
        # 格式化显示（美元）
        if abs(current_oi_usd) >= 1e9:
            display_usd = f"{current_oi_usd/1e9:.2f}B"
        elif abs(current_oi_usd) >= 1e6:
            display_usd = f"{current_oi_usd/1e6:.2f}M"
        elif abs(current_oi_usd) >= 1e3:
            display_usd = f"{current_oi_usd/1e3:.2f}K"
        else:
            display_usd = f"{current_oi_usd:.2f}"
        
        # 格式化显示（合约张数）
        display_contracts = "N/A"
        if oi_contracts_list and len(oi_contracts_list) > 0:
            current_contracts = oi_contracts_list[-1]
            if abs(current_contracts) >= 1e6:
                display_contracts = f"{current_contracts/1e6:.2f}M"
            elif abs(current_contracts) >= 1e3:
                display_contracts = f"{current_contracts/1e3:.2f}K"
            else:
                display_contracts = f"{current_contracts:.0f}"
            result['current_contracts'] = float(current_contracts)
        
        # 市场意义解读
        if change_pct > 5:
            interpretation = f"持仓量显著上升({change_pct:+.1f}%，现价{display_usd})，表示新资金大幅入场，市场热度上升，趋势方向更易确立。"
            price_oi_signal = "【价格↑+OI↑】→ 真突破信号，建议顺势加仓"
        elif change_pct > 2:
            interpretation = f"持仓量温和上升({change_pct:+.1f}%，现价{display_usd})，市场参与度增加，有利于趋势持续。"
            price_oi_signal = "【价格↑+OI↑】→ 趋势延续，可维持多头"
        elif change_pct > -2:
            interpretation = f"持仓量基本持平({change_pct:+.1f}%，现价{display_usd})，市场参与度稳定，交易动力不足。"
            price_oi_signal = "【OI持平】→ 动力不足，需等待突破"
        elif change_pct > -5:
            interpretation = f"持仓量温和下降({change_pct:+.1f}%，现价{display_usd})，多头部分获利了结，可能出现技术性回调。"
            price_oi_signal = "【价格↑+OI↓】→ 空头平仓，属于弱势上涨，注意回调"
        else:
            interpretation = f"持仓量显著下降({change_pct:+.1f}%，现价{display_usd})，大量持仓平仓，多头动能衰竭风险，需谨慎加仓。"
            price_oi_signal = "【价格↑+OI↓明显】→ 弱势信号，禁止追高"
        
        result.update({
            'current_usd': float(current_oi_usd),
            'current_contracts': float(oi_contracts_list[-1]) if oi_contracts_list and len(oi_contracts_list) > 0 else 0,
            'display_usd': display_usd,
            'display_contracts': display_contracts,
            'change_pct': float(change_pct),
            'trend': trend,
            'interpretation': interpretation,
            'price_oi_signal': price_oi_signal
        })
        
        return result
    
    @staticmethod
    def format_market_sentiment(funding_rate_list: list, oi_usd_list: list, oi_contracts_list: list = None) -> str:
        """
        生成完整的市场情绪数据格式化文本，直接用于Agent提示词
        
        返回格式如用户所需的示例：
        【市场情绪数据】
        1. 资金费率 (Funding Rate): {fr_display}
           * 正值过高代表多头拥挤，可能回调；负值过低代表空头拥挤，可能轧空。
        2. 持仓量 (Open Interest): {oi_display}
           * 价格涨+OI涨=真突破；价格涨+OI跌=空头平仓(弱势上涨)。
        """
        fr_data = MarketSentimentFormatter.format_funding_rate(funding_rate_list)
        oi_data = MarketSentimentFormatter.format_open_interest(oi_usd_list, oi_contracts_list)
        
        sentiment_text = f"""
【市场情绪数据】
1. 资金费率 (Funding Rate): {fr_data['display']} {fr_data['trend']}
   * 当前状态: {fr_data['status']}
   * 解读: {fr_data['interpretation']}

2. 持仓量 (Open Interest): {oi_data['display_usd']} (合约:{oi_data['display_contracts']}) {oi_data['trend']}
   * 变化: {oi_data['change_pct']:+.1f}%
   * 信号: {oi_data['price_oi_signal']}
   * 解读: {oi_data['interpretation']}
"""
        return sentiment_text

    @staticmethod
    def format_market_sentiment_current(funding_rate_current: float = None,
                                       funding_rate_delta: float = None,
                                       oi_usd_current: float = None,
                                       oi_usd_change_pct: float = None,
                                       oi_contracts_current: float = None) -> str:
        """
        简洁版：仅使用当期永续合约数据（当前值 + 可选变化率），用于减少提示词长度与令牌消耗。
        返回短文本，示例：
        资金费率: +0.003% (较前:+0.001%)；持仓量: 1.76B (变化:+0.0%)
        """
        fr_display = "N/A"
        oi_display = "N/A"
        fr_trend = ""
        oi_trend = ""

        try:
            if funding_rate_current is not None:
                fr_display = f"{funding_rate_current*100:+.3f}%"
                if funding_rate_delta is not None:
                    fr_trend = f" (变化:{funding_rate_delta*100:+.3f}%)"
        except Exception:
            fr_display = "N/A"

        try:
            if oi_usd_current is not None:
                v = oi_usd_current
                if abs(v) >= 1e9:
                    oi_display = f"{v/1e9:.2f}B"
                elif abs(v) >= 1e6:
                    oi_display = f"{v/1e6:.2f}M"
                elif abs(v) >= 1e3:
                    oi_display = f"{v/1e3:.2f}K"
                else:
                    oi_display = f"{v:.0f}"
                if oi_usd_change_pct is not None:
                    oi_trend = f" (变化:{oi_usd_change_pct:+.1f}%)"
        except Exception:
            oi_display = "N/A"

        parts = ["【市场情绪（当期值）】"]
        parts.append(f"资金费率: {fr_display}{fr_trend}")
        parts.append(f"持仓量(USD): {oi_display}{oi_trend}")
        if oi_contracts_current is not None:
            parts.append(f"持仓量(合约): {oi_contracts_current:.0f}")

        return "\n".join(parts)


class UniversalLLMStrategy(bt.Strategy):
    params = (
        ('agent', None),
        ('lookback', 20),
        ('indicators_config', []),
        ('snapshot_csv', None),
        ('leverage', 1.0),
        ('high_low_window', 20),
        ('vec_db', None),
        ('graph_db', None),
        ('h1_data', None),  # 【新增】1h向量数据，供next()方法使用
        ('main_tf', '1h'),  # 【新增】主要交易周期，默认为1h
    )

    # 【注解】不再设置minperiod，改为在prenext()中直接调用next()
    # 这样可以规避高周期稀疏数据导致的minperiod阻塞问题

    def __init__(self):
        # self.main_data = self.datas[0]
        self.main_data = self.datas[0]

        self.order = None # 主订单（市价开仓）
        self.stop_order = None  # 止损挂单
        self.limit_order = None # 止盈挂单

        self.start_price = None
        self.start_value = None
        self.roi_msg = "Init..."
        self.history_log = []
        self.current_tp = 0.0
        self.current_sl = 0.0
        # 【新增】记录最后触发的平仓方式（用于notify_trade中识别）
        self.last_close_reason = None
        # 【新增】标志：上一个bar是否刚完成平仓（用于跳过重复的HOLD记录）
        self.just_closed_position = False
        # 已实现盈亏与交易统计
        self.realized_pnl = 0.0
        self.closed_trades = 0
        self.win_trades = 0
        self.lose_trades = 0
        # 每个时间点的运行快照（用于导出 time-series CSV）
        # 每项为 dict: timestamp, price, net_value, alpha, decision_made(bool), decision_type, reflect_happened(bool), reflect_content
        self.tick_snapshot_rows = []
        self.weekly_indicators = {}
        # 【新增】K线计数用于调试
        self.bar_count = 0
        # 【新增】跳过计数器：在Agent决策后跳过4个next窗口以加快测试
        self.skip_next_bars = 0  # 剩余需要跳过的bars数
        # 【新增】上一期的资金费率和持仓量缓存，用于计算变化值
        self.prev_funding_rate = None
        self.prev_oi_usd = None
        # 【新增】持仓元数据：保存开仓时的市场状态快照，供平仓时写入Graph DB和向量库
        self.position_metadata = {
            'market_states': [],      # 开仓时观察到的市场状态 (RSI_Oversold, Price_NearLow 等)
            'base_states': [],        # 硬编码指标状态
            'additional_states': [],  # Agent补充的状态
            'confidence': {},         # 状态置信度
            'entry_price': 0.0,       # 开仓价格
            'entry_time': '',         # 开仓时间
            'entry_bars': 0,          # 开仓时的 bar 计数
            'entry_decision': '',     # 开仓决策 (LONG/SHORT)
            'entry_ohlcv': None,      # 【新增】开仓时的30期OHLCV数据，用于向量库
        }
        # snapshot file writer (streaming)
        self._snapshot_file = None
        self._snapshot_writer = None
        try:
            if getattr(self.params, 'snapshot_csv', None):
                csv_path = self.params.snapshot_csv
                write_header = not os.path.exists(csv_path)
                self._snapshot_file = open(csv_path, 'a', newline='', encoding='utf-8')
                fieldnames = ['timestamp', 'price', 'open', 'high', 'low', 'close', 'net_value', 'alpha', 'position_size', 'avg_price', 'decision_made', 'decision_type', 'quantity_pct', 'take_profit_price', 'stop_loss_price', 'similar_pattern_found']
                self._snapshot_writer = csv.DictWriter(self._snapshot_file, fieldnames=fieldnames)
                if write_header:
                    self._snapshot_writer.writeheader()
                    self._snapshot_file.flush()
        except Exception:
            self._snapshot_file = None
            self._snapshot_writer = None

        # 【新增】初始化实时交易决策日志记录器
        self.trade_decision_logger = None
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            log_dir = os.path.join(repo_root, 'logs', ts)
            os.makedirs(log_dir, exist_ok=True)
            trade_log_path = os.path.join(log_dir, 'trade_decisions.csv')
            self.trade_decision_logger = TradeDecisionLogger(trade_log_path)
            print(f"[LOGGER_INIT] Trade decision logger initialized: {trade_log_path}")
        except Exception as e:
            print(f"[LOGGER_ERROR] Failed to initialize trade decision logger: {e}")
            self.trade_decision_logger = None


        self.mtf_indicators = {}
        try:
            for i, d in enumerate(self.datas):
                tf_name = d._name
                if d._name in ['funding_feed', 'oi_feed']:
                    continue

                # 【关键修复】只在主周期上初始化指标！
                # 高周期数据（1W、1M）太稀疏，指标初始化会失败，且会阻塞minperiod计算
                # 只有主周期（通常为4h或1h）才需要完整的技术指标计算
                main_period = self.datas[0]._name  # 第一个非辅助数据源就是主周期
                if tf_name != main_period:
                    print(f"[INIT_IND] Skipping indicators for {tf_name} (not main period: {main_period})", flush=True)
                    self.mtf_indicators[tf_name] = {}  # 创建空字典占位
                    continue

                print(f"[INIT_IND] Initializing indicators for {tf_name}...", flush=True)
                self.mtf_indicators[tf_name] = {}
                for conf in self.params.indicators_config:
                    ind_cls = conf['cls']
                    name = conf.get('name', ind_cls.__name__.lower())
                    kwargs = conf.get('kwargs', {}).copy()
                    user_plot_setting = kwargs.get('plot')

                    # 如果用户没指定，默认只在主图(i=0)画
                    if user_plot_setting is None:
                        should_plot = (i == 0)
                    else:
                        should_plot = user_plot_setting
                    # 如果 main.py 里传了 plot=False，这里会保留 False
                    if 'plot' in kwargs:
                        del kwargs['plot']

                    try:
                        ind = ind_cls(d, **kwargs)  #所有子图都不绘制
                        ind.plotinfo.plot = should_plot
                        self.mtf_indicators[tf_name][name] = ind
                    except Exception as e:
                        # 【修复】如果指标计算失败（如数据太稀疏），跳过该指标
                        # 这通常发生在高周期数据（1W、1M）上，数据太少导致 IndexError
                        print(f"[WARN_IND] Failed to add {name} indicator for {tf_name}: {e}", flush=True)

                print(f"[INIT_IND] {tf_name}: {len(self.mtf_indicators[tf_name])} indicators added", flush=True)

            print(f"[INIT] Strategy init done: {len(self.mtf_indicators)} timeframes, {sum(len(v) for v in self.mtf_indicators.values())} indicators", flush=True)
        except Exception as e:
            print(f"[ERROR_INIT] Failed to initialize indicators: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    def _sanitize_csv_field(self, val):
        """把可能包含换行的文本字段转为单行，使用显式的 \n 表示换行，避免破坏 CSV 格式。"""
        if val is None:
            return ''
        s = str(val)
        # 统一替换各种换行符为字面 \n
        s = s.replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')
        return s

    def _extract_market_states(self, state: dict, agent=None) -> dict:
        """
        提取当前市场的状态标签（仅从Agent S动态提取）

        【改进】
        - 移除硬编码的LAYER 1（RSI、MACD、价格位置等）
        - 只保留Agent S结构化输出的状态
        - 简化结果结构：仅保留Agent S提取的状态

        返回：{
            'base_states': [],              # 【已移除】硬编码状态
            'additional_states': [...],     # Agent S 提取的状态（现在是主要来源）
            'all_states': [...],            # 合并后的所有状态
            'confidence': {...},            # 各状态的置信度
        }
        """
        result = {
            'base_states': [],              # 保持空列表以兼容现有代码
            'additional_states': [],
            'all_states': [],
            'confidence': {}
        }

        try:
            # ===== LAYER: Agent S 结构化输出状态（唯一来源）=====
            agent_s_states = []

            if agent:
                try:
                    agent_s_output = getattr(agent, '_agent_s_output_obj', None)
                    if agent_s_output:
                        # 【改进】支持新的AgentSOutput格式（Dict格式状态）
                        if hasattr(agent_s_output, 'current_states'):
                            # 新格式：current_states 是Dict，需要提取keys
                            if isinstance(agent_s_output.current_states, dict):
                                agent_s_states = list(agent_s_output.current_states.keys())
                            else:
                                # 兼容旧格式（List格式）
                                agent_s_states = agent_s_output.current_states if isinstance(agent_s_output.current_states, list) else []
                        elif hasattr(agent_s_output, 'data') and hasattr(agent_s_output.data, 'current_states'):
                            # 旧格式兼容：StateComparisonResult
                            agent_s_states = agent_s_output.data.current_states
                        else:
                            agent_s_states = []

                        if agent_s_states:
                            result['additional_states'] = agent_s_states
                            # 为 Agent S 提取的状态设置置信度
                            for state_name in agent_s_states:
                                if state_name not in result['confidence']:
                                    result['confidence'][state_name] = 0.9  # Agent S 状态为主要来源
                        else:
                            # 如果Agent S未提取到任何状态，使用默认值
                            result['additional_states'] = ['NoState']
                            result['confidence']['NoState'] = 0.5

                except Exception as e:
                    print(f"[AGENT_S_STATE_EXTRACT_ERROR] 从 Agent S 提取状态失败: {e}")
                    result['additional_states'] = ['NoState']
                    result['confidence']['NoState'] = 0.5
            else:
                # 没有Agent实例时的降级处理
                print(f"[WARN] Agent instance not provided, using NoState fallback")
                result['additional_states'] = ['NoState']
                result['confidence']['NoState'] = 0.5

            # 合并所有状态（现在仅包含Agent S的状态）
            result['all_states'] = result['additional_states']

            # 确保all_states不为空
            if not result['all_states']:
                result['all_states'] = ['NoState']
                result['confidence']['NoState'] = 0.5

        except Exception as e:
            print(f"[WARN] 市场状态提取失败: {e}")
            result['all_states'] = ['NoState']
            result['additional_states'] = ['NoState']
            result['confidence']['NoState'] = 0.5

        return result


    def log(self, txt):
        dt = self.main_data.datetime.datetime(0)
        print(f'{dt} | {txt}')

    def _print_dashboard(self, title):
        """打印账户看板"""
        val = self.broker.getvalue()
        pos = self.position.size
        cost = self.position.price
        cur_price = self.main_data.close[0]
        
        strat_pnl = val - self.start_value
        strat_roi = (strat_pnl / self.start_value) if (self.start_value and self.start_value != 0) else 0.0
        
        # 重新计算 Alpha
        bench_pnl = cur_price - self.start_price
        bench_roi = (bench_pnl / self.start_price) if (self.start_price and self.start_price != 0) else 0.0
        alpha = strat_roi - bench_roi

        # TP/SL 距离百分比
        if self.current_tp > 0:
            tp_diff = (self.current_tp - cur_price) / cur_price if cur_price != 0 else 0.0
            tp_str = f"{self.current_tp:.0f}({tp_diff:+.2%})"
        else: tp_str = "未设"

        if self.current_sl > 0:
            sl_diff = (self.current_sl - cur_price) / cur_price if cur_price != 0 else 0.0
            sl_str = f"{self.current_sl:.0f}({sl_diff:+.2%})"
        else: sl_str = "unset"
        dt_str = self.main_data.datetime.datetime(0).isoformat()
        print(f"      [Dashboard] ({title}  @ {cur_price:.0f} | {dt_str})")
        print(f"      ├─ Cash: {self.broker.getcash():.2f}")
        print(f"      ├─ Position: {pos:.4f} (cost:{cost:.2f})")
        print(f"      ├─ Plan: TP:{tp_str} | SL:{sl_str}")
        print(f"      ├─ Equity: {val:.2f}")
        print(f"      ├─ PnL: {strat_pnl:+.2f} ({strat_roi:+.2%})")
        print(f"      ├─ Realized: {self.realized_pnl:+.2f} | Trades: {self.closed_trades} (Win:{self.win_trades} Loss:{self.lose_trades})")
        print(f"      └─ Alpha: {alpha:+.2%}")

    def update_protection(self):
        """
        每次仓位变动后，强制更新止盈止损挂单。
        """
        # 1. 先撤销所有旧的保护挂单，防止重复
        try:
            if self.stop_order:
                self.cancel(self.stop_order)
                self.stop_order = None
        except Exception as e:
            print(f"[WARN] Failed to cancel stop order: {e}")
            self.stop_order = None  # 强制清理，即使cancel失败

        try:
            if self.limit_order:
                self.cancel(self.limit_order)
                self.limit_order = None
        except Exception as e:
            print(f"[WARN] Failed to cancel limit order: {e}")
            self.limit_order = None  # 强制清理，即使cancel失败

        # 2. 如果没持仓，就不需要挂单
        if self.position.size == 0:
            self.current_sl = 0.0
            self.current_tp = 0.0
            return

        # 3. 根据持仓方向挂单
        size = self.position.size
        cur_price = self.main_data.close[0]

        # 多头持仓 (size > 0)
        if size > 0:
            # 止损卖单 (Stop Sell) - 价格下跌触发
            if self.current_sl > 0 and self.current_sl < cur_price:
                try:
                    self.stop_order = self.sell(
                        data=self.main_data,
                        exectype=bt.Order.Stop,
                        price=self.current_sl,
                        size=size,
                        transmit=True # 立即发送
                    )
                    print(f"       [RISK] Long SL order @ {self.current_sl}")
                except Exception as e:
                    print(f"       [WARN] Failed to place long SL: {e}")
                    self.stop_order = None

            # 止盈卖单 (Limit Sell)
            if self.current_tp > 0 and self.current_tp > cur_price:
                try:
                    self.limit_order = self.sell(
                        data=self.main_data,
                        exectype=bt.Order.Limit,
                        price=self.current_tp,
                        size=size,
                        transmit=True
                    )
                    print(f"       [RISK] Long TP order @ {self.current_tp}")
                except Exception as e:
                    print(f"       [WARN] Failed to place long TP: {e}")
                    self.limit_order = None

        # 空头持仓 (size < 0)
        elif size < 0:
            # 止损买单 (Stop Buy) -> 价格上涨触发
            if self.current_sl > 0 and self.current_sl > cur_price:
                try:
                    self.stop_order = self.buy(
                        data=self.main_data,
                        exectype=bt.Order.Stop,
                        price=self.current_sl,
                        size=abs(size),
                        transmit=True
                    )
                    print(f"       [RISK] Short SL order @ {self.current_sl}")
                except Exception as e:
                    print(f"       [WARN] Failed to place short SL: {e}")
                    self.stop_order = None

            # 止盈买单 (Limit Buy) -> 价格下跌触发
            if self.current_tp > 0 and self.current_tp < cur_price:
                try:
                    self.limit_order = self.buy(
                        data=self.main_data,
                        exectype=bt.Order.Limit,
                        price=self.current_tp,
                        size=abs(size),
                        transmit=True
                    )
                    print(f"       [RISK] Short TP order @ {self.current_tp}")
                except Exception as e:
                    print(f"       [WARN] Failed to place short TP: {e}")
                    self.limit_order = None




    def notify_order(self, order):
        if order.status in [order.Completed]:
            # 判断是否为主订单（市价单通常是主订单，ref 对应 self.order）
            # 但更简单的方法是：任何成交都意味着仓位可能变了，所以更新风控
            if not order.parent: # 排除子订单（如果用了bracket的话，但这里我们是手动挂单）
                t = "买入" if order.isbuy() else "卖出"
                print(f"      >>> [DONE] {t}: {order.executed.size:.4f} @ {order.executed.price:.2f}")

                # --- 构建日志条目并加入 Agent 记忆 ---
                dt_str = self.main_data.datetime.datetime(0).isoformat()
                # 判断成交类型
                order_type = "MARKET"
                if order == self.stop_order: order_type = "STOP_LOSS"
                elif order == self.limit_order: order_type = "TAKE_PROFIT"

                log_entry = f"{dt_str}: [SYSTEM] {order_type} FILLED {t} {order.executed.size:.4f} @ {order.executed.price:.2f}"
                self.history_log.append(log_entry)

                # 【新增】记录到交易日志文件
                try:
                    if self.trade_decision_logger:
                        position_before = self.position.size  # 成交前的持仓
                        # 成交后的持仓需要特殊处理，因为此时 notify_order 已经更新了 position
                        position_after = self.position.size  # 成交后的持仓

                        self.trade_decision_logger.log_order_execution(
                            timestamp=dt_str,
                            order_type=order_type,
                            is_buy=order.isbuy(),
                            executed_price=order.executed.price,
                            executed_size=order.executed.size,
                            position_before=position_before,
                            position_after=position_after
                        )
                except Exception as e:
                    print(f"[LOG_ERROR] Failed to log order execution: {e}")
                # -----------------------------------------


                # 如果是止损止盈单触发了，要重置计划
                if order == self.stop_order:
                    print(f"      >>>  止损触发！")
                    # 【新增】在重置前记录平仓方式，供notify_trade使用
                    self.last_close_reason = "SL_HIT"
                    self.current_sl = 0
                    self.current_tp = 0
                elif order == self.limit_order:
                    print(f"      >>>  止盈触发！")
                    # 【新增】在重置前记录平仓方式，供notify_trade使用
                    self.last_close_reason = "TP_HIT"
                    self.current_sl = 0
                    self.current_tp = 0

                # 任何成交发生后，重新评估挂单
                # 注意：必须在 notify_order 里调用，因为此时 self.position 已经更新
                self.update_protection()

                self._print_dashboard("[TRADE_SNAPSHOT]")

            # 清理引用
            if order == self.order: self.order = None
            if order == self.stop_order: self.stop_order = None
            if order == self.limit_order: self.limit_order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 【修复】无论是什么订单出问题，都要清理所有相关引用，防止幽灵订单
            status_name = order.getstatusname()
            dt_str = self.main_data.datetime.datetime(0).isoformat()

            if order == self.order:
                print(f"      >>> [ERROR] 主订单异常: {status_name}")
                self.order = None

                # 【新增】记录订单执行错误
                try:
                    if self.trade_decision_logger:
                        self.trade_decision_logger.log_error(
                            timestamp=dt_str,
                            bar_count=self.bar_count,
                            decision_type=order.getordername(),
                            error_msg=f"Order {status_name}: {order.getstatusname()}",
                            decision_price=self.main_data.close[0]
                        )
                except Exception as e:
                    print(f"[LOG_ERROR] Failed to log order error: {e}")

            # 【新增】也要清理止损和止盈订单，即使它们不是立即发生的异常
            # 这样可以防止在后续 update_protection() 中出现引用混乱
            if order == self.stop_order:
                print(f"      >>> [WARN] 止损单异常: {status_name}")
                self.stop_order = None
                self.current_sl = 0  # 也重置止损价

                # 【新增】记录止损单错误
                try:
                    if self.trade_decision_logger:
                        self.trade_decision_logger.log_error(
                            timestamp=dt_str,
                            bar_count=self.bar_count,
                            decision_type='STOP_LOSS',
                            error_msg=f"Stop order {status_name}",
                            decision_price=self.main_data.close[0]
                        )
                except Exception as e:
                    print(f"[LOG_ERROR] Failed to log stop order error: {e}")

            if order == self.limit_order:
                print(f"      >>> [WARN] 止盈单异常: {status_name}")
                self.limit_order = None
                self.current_tp = 0  # 也重置止盈价

                # 【新增】记录止盈单错误
                try:
                    if self.trade_decision_logger:
                        self.trade_decision_logger.log_error(
                            timestamp=dt_str,
                            bar_count=self.bar_count,
                            decision_type='TAKE_PROFIT',
                            error_msg=f"Limit order {status_name}",
                            decision_price=self.main_data.close[0]
                        )
                except Exception as e:
                    print(f"[LOG_ERROR] Failed to log limit order error: {e}")

# --- notify_trade 用于计算盈亏 ---
    def notify_trade(self, trade):
        """记录已实现盈亏并写入图数据库。"""
        if not trade.isclosed:
            return

        # 【新增】标记已平仓，供 next() 使用以跳过重复的快照记录
        self.just_closed_position = True

        # 计算盈亏
        pnl = trade.pnlcomm if trade.pnlcomm else trade.pnl
        self.realized_pnl += pnl
        self.closed_trades += 1
        if pnl > 0: self.win_trades += 1
        else: self.lose_trades += 1

        # 记录盈亏到 Agent 记忆
        dt_str = self.main_data.datetime.datetime(0).isoformat()
        log_entry = f"{dt_str}: [PnL] Trade Closed. Net: {pnl:+.2f} (Acc: {self.realized_pnl:+.2f})"
        self.history_log.append(log_entry)

        print(f"      >>>  Trade Settled: PnL {pnl:+.2f} | Total: {self.realized_pnl:+.2f}")

        # 【新增】记录平仓快照到snapshot（记录TP/SL触发的信息）
        try:
            cur_price = self.main_data.close[0]
            cur_val = self.broker.getvalue()

            # 判断平仓原因：TP/SL/MANUAL
            # 【修复】使用 last_close_reason（在notify_order中设置），而不是检查current_tp/current_sl
            # 因为current_tp和current_sl在notify_order中已经被重置为0了
            close_reason = self.last_close_reason if self.last_close_reason else "MANUAL"
            # 平仓后清除标记，为下一个平仓做准备
            self.last_close_reason = None
            # 【修复】使用开仓时保存的 entry_decision，而不是平仓时的 trade.size
            trade_direction = self.position_metadata.get('entry_decision', 'LONG')
            if trade_direction not in ['LONG', 'SHORT']:
                trade_direction = 'LONG' if trade.size > 0 else 'SHORT'  # 回退方案

            # 额外的验证（备用方案）：如果没有last_close_reason，尝试通过订单类型判断
            if close_reason == "MANUAL" and (self.current_tp > 0 or self.current_sl > 0):
                if trade_direction == 'LONG':
                    # 多头：价格上涨到TP或下跌到SL
                    if self.current_tp > 0 and cur_price >= self.current_tp * 0.995:  # 允许5个基点的偏差
                        close_reason = "TP_HIT"
                    elif self.current_sl > 0 and cur_price <= self.current_sl * 1.005:
                        close_reason = "SL_HIT"
                else:  # SHORT
                    # 空头：价格下跌到TP或上涨到SL
                    if self.current_tp > 0 and cur_price <= self.current_tp * 1.005:
                        close_reason = "TP_HIT"
                    elif self.current_sl > 0 and cur_price >= self.current_sl * 0.995:
                        close_reason = "SL_HIT"

            # 【新增】记录平仓原因，用于验证向量库和图数据库更新
            print(f"      [NOTIFY_TRADE] Close Reason: {close_reason} | Trade PnL: {pnl:+.2f}")

            # 创建平仓快照，遵循现有格式
            close_snap_row = {
                'timestamp': dt_str,
                'price': float(cur_price),
                'open': float(self.main_data.open[0]) if hasattr(self.main_data, 'open') else '',
                'high': float(self.main_data.high[0]) if hasattr(self.main_data, 'high') else '',
                'low': float(self.main_data.low[0]) if hasattr(self.main_data, 'low') else '',
                'close': float(self.main_data.close[0]) if hasattr(self.main_data, 'close') else '',
                'net_value': float(cur_val),
                'alpha': float(pnl),  # 平仓的实现盈亏
                'position_size': 0.0,  # 平仓后持仓为0
                'avg_price': '',  # 已平仓，无平均价格
                'decision_made': True,
                'decision_type': f'CLOSE_{close_reason}',  # 例如：CLOSE_TP_HIT
                'quantity_pct': 0.0,
                'take_profit_price': float(self.current_tp) if self.current_tp > 0 else 0.0,
                'stop_loss_price': float(self.current_sl) if self.current_sl > 0 else 0.0,
                'similar_pattern_found': 'N/A'  # 【新增】平仓时不适用相似行情匹配
            }

            # 写入内存列表
            try:
                self.tick_snapshot_rows.append(close_snap_row)
            except Exception:
                pass

            # 写入CSV文件
            if self._snapshot_writer and self._snapshot_file:
                try:
                    self._snapshot_writer.writerow(close_snap_row)
                    self._snapshot_file.flush()
                    print(f"      [SNAPSHOT] Close recorded: {close_reason} @ {cur_price:.1f}, PnL: {pnl:+.2f}")
                except Exception as e:
                    print(f"      [SNAPSHOT_WARN] Failed to write close snapshot: {e}")
        except Exception as e:
            print(f"      [SNAPSHOT_ERROR] Error recording close snapshot: {e}")

        try:
            if self.trade_decision_logger:
                # 【修复】使用开仓时保存的 entry_decision，而不是平仓时的 trade.size
                # 原因：平仓时 trade.size = 0，会导致判断错误
                trade_direction = self.position_metadata.get('entry_decision', 'LONG')
                if trade_direction not in ['LONG', 'SHORT']:
                    trade_direction = 'LONG' if trade.size > 0 else 'SHORT'  # 回退方案
                duration_bars = int(trade.barlen) if hasattr(trade, 'barlen') else 1

                self.trade_decision_logger.log_trade_close(
                    timestamp=dt_str,
                    bar_count=self.bar_count,
                    trade_direction=trade_direction,
                    realized_pnl=pnl,
                    duration_bars=duration_bars,
                    position_before=trade.size  # 平仓前的持仓
                )
        except Exception as e:
            print(f"[LOG_ERROR] Failed to log trade close: {e}")

        # ============================================
        # 【新增】PHASE 3: 创建 TradeContext 黑匣子
        # ============================================
        from ace_trading.framework import TradeContext
        import uuid
        import time

        # 生成唯一的事件ID
        event_id = f"trade_{str(uuid.uuid4())[:8]}_{int(time.time() * 1000) % 10000}"

        # 获取开仓时的Agent输出（来自position_metadata）
        s_output = self.position_metadata.get('s_output')  # AgentSOutput
        a_output = self.position_metadata.get('a_output')  # AgentAOutput
        b_output = self.position_metadata.get('b_output')  # AgentBOutput

        # 获取图数据库搜索摘要
        graph_search_summary = self.position_metadata.get('graph_search_summary', '')

        # 创建 TradeContext 对象
        trade_context = TradeContext(
            event_id=event_id,
            entry_timestamp=dt_str,
            s_output=s_output,
            a_output=a_output,
            b_output=b_output,
            graph_search_summary=graph_search_summary
        )

        # 将 TradeContext 挂载到订单对象上（便于后续引用）
        trade.trade_context = trade_context

        print(f"[TRADE_CONTEXT] Created for trade {event_id}: s_output={s_output is not None}, a_output={a_output is not None}, b_output={b_output is not None}")

        # ============================================
        # Agent C-1: 写入向量数据库 (Vector DB Writer)
        # ============================================
        if self.params.vec_db is not None:
            try:
                entry_ohlcv = self.position_metadata.get('entry_ohlcv')
                entry_time = self.position_metadata.get('entry_time', '')

                # 只有当开仓时成功保存了OHLCV数据时，才添加到向量库
                if entry_ohlcv is not None and len(entry_ohlcv) == 30:
                    # 将OHLCV数据添加到向量库
                    # 使用开仓时间作为时间戳
                    vec_timestamp = pd.Timestamp(entry_time)
                    self.params.vec_db.add_vector(entry_ohlcv, vec_timestamp)

                    # 确定交易结果
                    outcome = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAK_EVEN')
                    trade_direction = 'LONG' if self.position_metadata.get('entry_decision') == 'LONG' else 'SHORT'

                    print(f"[VECTOR_DB] Trade ({trade_direction} {outcome}) entry pattern added at {entry_time}")
                    
                    # 【新增】记录 Agent C 的向量库操作
                    try:
                        from ace_trading.prompt_logger import get_prompt_logger
                        prompt_logger = get_prompt_logger()
                        if prompt_logger:
                            from datetime import datetime
                            timestamp = datetime.now().isoformat()
                            
                            # 构建 Agent C 的输入提示词（向量库操作的说明）
                            agent_c_prompt = f"""Agent C - Trade Pattern Writer

Your role: Write  trade patterns to the vector database for future pattern matching.

Trade Information:
- Entry Time: {entry_time}
- Trade Direction: {trade_direction}
- Entry Decision: {self.position_metadata.get('entry_decision', 'Unknown')}
- OHLCV Data Points: {len(entry_ohlcv)} (30 hours of 1H K-line data)
- Pattern Type: {trade_direction} Entry Pattern

Market Context at Entry:
- Entry Price: {self.position_metadata.get('entry_price', 'N/A')}
- Market States: {self.position_metadata.get('market_states', [])}
- Base States: {self.position_metadata.get('base_states', [])}
- Additional States: {self.position_metadata.get('additional_states', [])}

Action: Store this pattern in vector DB for similarity matching
Status: Completed
Vector ID: {entry_time}
"""
                            
                            # 构建输出（向量库操作结果）
                            agent_c_output = f"""Trade Pattern Successfully Written to Vector Database

Vector Database Entry:
- Timestamp: {entry_time}
- Pattern Type: {trade_direction} Entry Pattern
- Outcome: {outcome}
- Market States Used: {self.position_metadata.get('market_states', [])}
- OHLCV Window: 30 hours of 1H data (1632 hourly bars)
- Embedding Dimension: 128
- Status: Available for future similarity search

Pattern will be used to match similar market conditions in future trading.
"""
                            
                            prompt_logger.log_agent_prompt(
                                agent_name="C",
                                bar_number=self.bar_count,
                                timestamp=timestamp,
                                prompt_text=agent_c_prompt,
                                additional_info={
                                    "trade_direction": trade_direction,
                                    "outcome": outcome,
                                    "entry_time": entry_time,
                                    "market_states": str(self.position_metadata.get('market_states', [])),
                                    "pnl": f"{pnl:+.2f}"
                                },
                                output_text=agent_c_output,
                                output_info={
                                    "operation": "Vector DB Write",
                                    "status": "Success",
                                    "output_length": len(agent_c_output)
                                }
                            )
                    except Exception as log_err:
                        print(f"[PROMPT_LOG_ERROR] Failed to log Agent C operation: {log_err}")
                else:
                    print(f"[VECTOR_DB_WARN] Failed to add entry pattern: OHLCV data not available")

            except Exception as e:
                print(f"[VECTOR_DB_ERROR] Failed to add trade to vector DB: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        # ============================================
        # Agent C-2: 写入图数据库 (Reflection Writer)
        # ============================================
        if self.params.graph_db is not None:
            try:
                # 生成Event ID (格式: YYYYMMDD_HHMMSS_P)
                # 【修复】统一使用开仓时间，而非平仓时间，以便与向量库查询匹配
                # 图库记录的是"开仓时的市场状态"，用于给后续相似的开仓时机提供经验
                entry_time = self.position_metadata.get('entry_time', '')
                if entry_time:
                    # 从ISO格式的entry_time提取秒级时间戳
                    entry_dt = pd.Timestamp(entry_time)
                    event_id = entry_dt.strftime('%Y%m%d_%H%M%S') + '_P'
                else:
                    # 回退方案：如果没有entry_time则使用当前时间
                    trade_dt = self.main_data.datetime.datetime(0)
                    event_id = trade_dt.strftime('%Y%m%d_%H%M%S') + '_P'

                # 确定交易结果
                outcome = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAK_EVEN')

                # 获取交易持续时间（单位：分钟，假设1分钟K线）
                duration_bars = int(trade.barlen) if hasattr(trade, 'barlen') else 1
                duration_minutes = duration_bars * 1  # 1分钟K线

                # 构建事件数据
                # 【修复】timestamp 应该记录开仓时间（市场状态时刻），而非平仓时间
                event_data = {
                    'event_id': event_id,
                    'timestamp': entry_time,  # 使用开仓时间，对应市场状态被观察的时刻
                    'duration': duration_minutes,
                    'realized_pnl': float(pnl)
                }

                # 【修复】使用开仓时保存的 entry_decision，而不是平仓时的 trade.size
                # 原因：平仓时 trade.size = 0，会导致判断错误（所有交易都变成 SHORT）
                trade_direction = self.position_metadata.get('entry_decision', 'LONG')
                if trade_direction not in ['LONG', 'SHORT']:
                    trade_direction = 'LONG' if trade.size > 0 else 'SHORT'  # 回退方案

                # 【修复】使用开仓时保存的市场状态，而不是无用的 'TradeCompleted' 标签
                # 从 position_metadata 中获取开仓时的市场状态（现在包括硬编码和Agent补充的）
                all_market_states = self.position_metadata.get('market_states', [])
                base_states = self.position_metadata.get('base_states', [])
                additional_states = self.position_metadata.get('additional_states', [])
                confidence = self.position_metadata.get('confidence', {})

                # ===============================================
                # 【新增】提取 Agent S 输出数据
                # ===============================================
                agent_s_output = self.position_metadata.get('s_output')
                agent_s_states = {
                    'current_states': {},
                    'matched_states': {},
                    'missing_states': {},
                    'novel_states': {}
                }
                try:
                    if agent_s_output:
                        if hasattr(agent_s_output, 'current_states'):
                            agent_s_states['current_states'] = dict(agent_s_output.current_states) if isinstance(agent_s_output.current_states, dict) else {}
                        if hasattr(agent_s_output, 'matched_states'):
                            agent_s_states['matched_states'] = dict(agent_s_output.matched_states) if isinstance(agent_s_output.matched_states, dict) else {}
                        if hasattr(agent_s_output, 'missing_states'):
                            agent_s_states['missing_states'] = dict(agent_s_output.missing_states) if isinstance(agent_s_output.missing_states, dict) else {}
                        if hasattr(agent_s_output, 'novel_states'):
                            agent_s_states['novel_states'] = dict(agent_s_output.novel_states) if isinstance(agent_s_output.novel_states, dict) else {}
                        print(f"[GRAPH_DB] Found Agent S states: {len(agent_s_states['current_states'])} current, {len(agent_s_states['matched_states'])} matched")
                except Exception as e:
                    print(f"[GRAPH_DB_WARN] Failed to extract Agent S states: {e}")

                # ===============================================
                # 【新增】提取 Agent A 输出数据
                # ===============================================
                agent_a_pattern_name = None
                agent_a_confidence = None
                agent_a_core_dna_states = []  # 【新增】核心DNA状态
                agent_a_context_states = []   # 【新增】环境Context状态
                agent_a_core_logic_states = [] # 【兼容性】向后兼容，等同于core_dna_states
                agent_a_pattern_description = None
                try:
                    a_output = self.position_metadata.get('a_output')
                    if a_output:
                        if hasattr(a_output, 'pattern_name'):
                            agent_a_pattern_name = a_output.pattern_name
                        if hasattr(a_output, 'confidence'):
                            agent_a_confidence = a_output.confidence

                        # 【新增】优先提取core_dna_states（核心DNA）
                        if hasattr(a_output, 'core_dna_states'):
                            agent_a_core_dna_states = list(a_output.core_dna_states) if isinstance(a_output.core_dna_states, (list, tuple)) else []
                            agent_a_core_logic_states = agent_a_core_dna_states  # 【兼容性】复制给core_logic_states
                        # 【后备】如果没有core_dna_states，检查core_logic_states（旧代码兼容）
                        elif hasattr(a_output, 'core_logic_states'):
                            agent_a_core_logic_states = list(a_output.core_logic_states) if isinstance(a_output.core_logic_states, (list, tuple)) else []
                            agent_a_core_dna_states = agent_a_core_logic_states  # 将旧字段映射到新字段

                        # 【新增】提取context_states（环境Context）
                        if hasattr(a_output, 'context_states'):
                            agent_a_context_states = list(a_output.context_states) if isinstance(a_output.context_states, (list, tuple)) else []

                        if hasattr(a_output, 'pattern_description'):
                            agent_a_pattern_description = a_output.pattern_description

                        # 【新增】验证 pattern_name 不包含禁止的结果标签
                        if agent_a_pattern_name:
                            forbidden_keywords = ['_WIN', '_LOSS', '_BREAK_EVEN', '_DRAW', '_BREAKEVEN']
                            for keyword in forbidden_keywords:
                                if keyword.lower() in agent_a_pattern_name.lower():
                                    print(f"[PATTERN_NAME_WARNING] Agent A pattern contains forbidden outcome keyword '{keyword}': {agent_a_pattern_name}")
                                    # 尝试清除 outcome 标签
                                    cleaned_name = agent_a_pattern_name
                                    for kw in forbidden_keywords:
                                        cleaned_name = cleaned_name.replace(kw, '').replace(kw.lower(), '')
                                    cleaned_name = cleaned_name.rstrip('_')  # 移除末尾的下划线
                                    if cleaned_name and cleaned_name != agent_a_pattern_name:
                                        print(f"[PATTERN_NAME_CLEANED] Cleaned pattern from '{agent_a_pattern_name}' to '{cleaned_name}'")
                                        agent_a_pattern_name = cleaned_name
                                    break

                        # 【新增】详细日志
                        print(f"[GRAPH_DB] Agent A output: pattern={agent_a_pattern_name}, core_dna={agent_a_core_dna_states}, context={len(agent_a_context_states)} items")
                except Exception as e:
                    print(f"[GRAPH_DB_WARN] Failed to extract Agent A pattern: {e}")

                # ===============================================
                # 【新增】提取 Agent B 输出数据
                # ===============================================
                agent_b_action = None
                agent_b_quantity = None
                agent_b_confidence = None
                agent_b_reasoning = None
                agent_b_risk_reward = None
                try:
                    b_output = self.position_metadata.get('b_output')
                    if b_output:
                        if hasattr(b_output, 'action'):
                            agent_b_action = b_output.action
                        if hasattr(b_output, 'quantity_pct'):
                            agent_b_quantity = b_output.quantity_pct
                        if hasattr(b_output, 'confidence'):
                            agent_b_confidence = b_output.confidence
                        if hasattr(b_output, 'reasoning'):
                            agent_b_reasoning = b_output.reasoning
                        if hasattr(b_output, 'risk_reward_ratio'):
                            agent_b_risk_reward = b_output.risk_reward_ratio
                        print(f"[GRAPH_DB] Found Agent B decision: {agent_b_action} with quantity {agent_b_quantity:.1%}")
                except Exception as e:
                    print(f"[GRAPH_DB_WARN] Failed to extract Agent B decision: {e}")

                # 使用所有状态作为核心逻辑状态
                core_logic_states = all_market_states if all_market_states else ['Unknown_Entry_State']

                # 如果没有保存的状态，则使用默认值
                if not core_logic_states or core_logic_states == ['NoState']:
                    core_logic_states = ['Unknown_Entry_State']

                # 【关键改进】强制对core_logic_states进行排序
                # 确保相同状态但不同顺序的Pattern识别为同一个Pattern
                # 例如：['State_B', 'State_A'] 和 ['State_A', 'State_B'] 都会被排序为 ['State_A', 'State_B']
                # 这样在生成Hash时能确保得到相同的Hash值
                core_logic_states = sorted(core_logic_states)
                print(f"[HASH_STABILITY] Sorted core_logic_states for Hash stability: {core_logic_states}", flush=True)

                # all_context_states 包含核心逻辑状态
                all_context_states = core_logic_states

                # 输出日志
                print(f"[STATE_CONTEXT] 平仓时使用开仓状态: {', '.join(core_logic_states)} -> {outcome}", flush=True)

                # 调用图数据库写入
                # 【改进】优先使用 Agent A 的 Pattern 名称
                # 【修复】备用格式只包含交易方向，不包含 outcome
                # outcome 应通过 Event -[:RESULTED_IN]-> Outcome 关系记录
                pattern_name = agent_a_pattern_name if agent_a_pattern_name else f'Trade_{trade_direction}'
                pattern_confidence = agent_a_confidence if agent_a_confidence is not None else 0.5

                # 转换 match_confidence 为整数百分比（0-100）
                match_confidence_int = int(pattern_confidence * 100) if isinstance(pattern_confidence, float) else pattern_confidence

                # 【优化】构建完整的 agent_insight 字典，包含 Agent S/A/B 的完整数据
                agent_insight = {
                    'pattern_name': pattern_name,
                    'description': f'{trade_direction} trade with states: {", ".join(core_logic_states)}',
                    'base_states': base_states,
                    'additional_states': additional_states,
                    'confidence': confidence,

                    # Agent S 的状态检测信息
                    'agent_s': {
                        'current_states': agent_s_states.get('current_states', {}),
                        'matched_states': agent_s_states.get('matched_states', {}),
                        'missing_states': agent_s_states.get('missing_states', {}),
                        'novel_states': agent_s_states.get('novel_states', {})
                    } if agent_s_output else None,

                    # Agent A 的模式分析信息
                    'agent_a': {
                        'pattern_name': agent_a_pattern_name,
                        'pattern_description': agent_a_pattern_description,
                        'core_dna_states': agent_a_core_dna_states,  # 【新增】核心DNA状态
                        'context_states': agent_a_context_states,    # 【新增】环境Context状态
                        'core_logic_states': agent_a_core_logic_states,  # 【兼容性】向后兼容
                        'confidence': agent_a_confidence
                    } if agent_a_pattern_name else None,

                    # Agent B 的决策信息
                    'agent_b': {
                        'action': agent_b_action,
                        'quantity_pct': agent_b_quantity,
                        'confidence': agent_b_confidence,
                        'reasoning': agent_b_reasoning,
                        'risk_reward_ratio': agent_b_risk_reward
                    } if agent_b_action else None
                }

                self.params.graph_db.insert_trade_reflection(
                    event_data=event_data,
                    core_logic_states=core_logic_states,
                    all_context_states=all_context_states,
                    agent_insight=agent_insight,
                    decision=trade_direction,
                    outcome=outcome,
                    match_confidence=match_confidence_int
                )
                print(f"[GRAPH_DB] Trade {event_id} logged successfully with outcome: {outcome}")

                # 【新增】记录 Agent C-2 的图数据库操作
                try:
                    from ace_trading.prompt_logger import get_prompt_logger
                    prompt_logger = get_prompt_logger()
                    if prompt_logger:
                        from datetime import datetime
                        timestamp = datetime.now().isoformat()

                        # 构建 Agent S 信息摘要
                        agent_s_summary = ""
                        if agent_s_output:
                            current_count = len(agent_s_states.get('current_states', {}))
                            matched_count = len(agent_s_states.get('matched_states', {}))
                            missing_count = len(agent_s_states.get('missing_states', {}))
                            novel_count = len(agent_s_states.get('novel_states', {}))
                            agent_s_summary = f"""
Agent S (State Sensor):
- Current States: {current_count} detected
- Matched States: {matched_count} matched to history
- Missing States: {missing_count} (expected but absent)
- Novel States: {novel_count} (new discoveries)"""

                        # 构建 Agent A 信息摘要
                        agent_a_summary = ""
                        if agent_a_pattern_name:
                            agent_a_summary = f"""
Agent A (Pattern Analyzer):
- Pattern: {agent_a_pattern_name}
- Description: {agent_a_pattern_description or 'N/A'}
- Core DNA States: {', '.join(agent_a_core_dna_states) if agent_a_core_dna_states else 'N/A'}
- Context States: {', '.join(agent_a_context_states) if agent_a_context_states else 'N/A'}
- Confidence: {agent_a_confidence:.1%}"""

                        # 构建 Agent B 信息摘要
                        agent_b_summary = ""
                        if agent_b_action:
                            agent_b_summary = f"""
Agent B (Decision Maker):
- Action: {agent_b_action}
- Position Size: {agent_b_quantity:.1%}
- Confidence: {agent_b_confidence:.1%}
- Risk/Reward Ratio: {agent_b_risk_reward:.2f}"""

                        # 构建 Agent C-2 的输入提示词（图数据库操作的说明）
                        agent_c2_prompt = f"""Agent C-2 - Trade Reflection Writer

Your role: Write trade reflections and patterns to the graph database for strategic learning.

Trade Information:
- Event ID: {event_id}
- Trade Timestamp: {dt_str}
- Trade Duration: {duration_minutes} minutes
- Trade Direction: {trade_direction}
- Outcome: {outcome}
- Realized PnL: {pnl:+.2f}

Market States Used:
- Core States: {', '.join(core_logic_states)}
- Base States: {', '.join(base_states)}
- Additional States: {', '.join(additional_states)}{agent_s_summary}{agent_a_summary}{agent_b_summary}

Action: Record this trade reflection and update graph database
Status: Completed
"""

                        # 构建输出（图数据库操作结果）
                        agent_c2_output = f"""Trade Reflection Successfully Written to Graph Database

Graph Database Entry:
- Event ID: {event_id}
- Pattern Name: {pattern_name}
- Core Logic States: {', '.join(core_logic_states)}
- All Context States: {', '.join(all_context_states)}
- Trade Duration: {duration_minutes} minutes
- Realized PnL: {pnl:+.2f}
- Confidence: {pattern_confidence:.1%}
- Status: Recorded and indexed for pattern learning

Agent Integration Summary:{agent_s_summary}{agent_a_summary}{agent_b_summary}

This trade reflection will be used to:
1. Refine market state classifications (Agent S)
2. Build historical pattern knowledge (Agent A)
3. Improve risk/reward decision making (Agent B)
4. Enable continuous strategy learning and optimization
"""

                        prompt_logger.log_agent_prompt(
                            agent_name="C",
                            bar_number=self.bar_count,
                            timestamp=timestamp,
                            prompt_text=agent_c2_prompt,
                            additional_info={
                                "event_id": event_id,
                                "trade_direction": trade_direction,
                                "outcome": outcome,
                                "duration_minutes": duration_minutes,
                                "pnl": f"{pnl:+.2f}",
                                "core_states": str(core_logic_states),
                                "agent_s_states": str(len(agent_s_states.get('current_states', {}))),
                                "agent_a_pattern": agent_a_pattern_name or "None",
                                "agent_b_action": agent_b_action or "None"
                            },
                            output_text=agent_c2_output,
                            output_info={
                                "operation": "Graph DB Write",
                                "status": "Success",
                                "output_length": len(agent_c2_output),
                                "states_count": len(core_logic_states),
                                "agents_integrated": sum([1 for x in [agent_s_output, agent_a_pattern_name, agent_b_action] if x])
                            }
                        )
                except Exception as log_err:
                    print(f"[PROMPT_LOG_ERROR] Failed to log Agent C-2 operation: {log_err}")
                
            except Exception as e:
                print(f"[GRAPH_DB_ERROR] Failed to log trade to graph DB: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    # ---------------------------------------




    def prenext(self):
        """在 next() 被调用前的预热阶段调用，通常用于积累数据以计算指标"""
        self.bar_count += 1
        if self.bar_count % 100 == 0:
            print(f"[PRENEXT] Bar {self.bar_count}: Still in warmup phase", flush=True)

    def next(self):
        self.bar_count += 1
        # 【调试】前1条K线详细输出
        if self.bar_count == 1:
            print(f"[NEXT-FIRST] Bar 1: len(self.datas)={len(self.datas)}", flush=True)
            for i, d in enumerate(self.datas):
                print(f"  [{i}] {d._name}: len={len(d)}", flush=True)
        # 【调试】前几条K线详细输出
        if self.bar_count <= 5:
            print(f"[NEXT] Bar {self.bar_count}: Entering next()", flush=True)
        # 【调试】每10000条K线输出一次进度
        elif self.bar_count % 10000 == 0:
            dt = self.main_data.datetime.datetime(0)
            print(f"[BAR] {self.bar_count}: {dt}", flush=True)

        # 【修复】如果刚完成平仓，则跳过本 bar 的快照记录和决策，以避免记录多余的 HOLD
        # 平仓信息已在 notify_trade() 中完整记录
        # 同时清除 last_close_reason，防止后续误用
        if self.just_closed_position:
            self.just_closed_position = False
            self.last_close_reason = None  # 【新增】清除平仓原因标记，防止后续误判
            print(f"[NEXT_SKIP] Skipping bar {self.bar_count} snapshot recording (just closed position)", flush=True)
            return

        # 【新增】跳过逻辑：Agent决策后跳过4个bars以加快测试
        if self.skip_next_bars > 0:
            self.skip_next_bars -= 1
            print(f"[SKIP] Skipping bar {self.bar_count} (Agent cooldown: {self.skip_next_bars} bars remaining)", flush=True)
            return

        if self.order:
            return

        # 【修复】只检查主周期是否有足够lookback数据，不检查高周期数据
        # 这样即使1w/1d等高周期数据不足，主周期(1m)也能正常执行
        if len(self.main_data) < self.params.lookback:
            return

        if self.start_price is None:
            self.start_price = self.main_data.close[0]
            self.start_value = self.broker.getvalue()
            print(f"[START] Bar {self.bar_count}: Started at {self.main_data.datetime.datetime(0)}", flush=True)
            return

        # 计算 Alpha (用于 Agent 状态)
        cur_val = self.broker.getvalue()
        cur_price = self.main_data.close[0]
        strat_roi = ((cur_val - self.start_value) / self.start_value) if (self.start_value and self.start_value != 0) else 0.0
        bench_roi = ((cur_price - self.start_price) / self.start_price) if (self.start_price and self.start_price != 0) else 0.0
        alpha = strat_roi - bench_roi
        self.roi_msg = f"Alpha: {alpha:+.2%} (Strat {strat_roi:+.2%} vs Bench {bench_roi:+.2%})"

        window_size = self.params.lookback
        ohlcv_window = 30  # 固定30根K线用于LSTM预处理

        funding_rate_window = []
        oi_usd_window = []
        oi_contracts_window = []  # 【新增】合约张数
        ohlcv_df = None  # 存储30期OHLCV数据

        # 提取数据
        weekly_state = {}
        mtf_state = {}
        hl_window = self.params.high_low_window

        for d in self.datas:
            tf_name = d._name

            # --- 处理资金费率 ---
            if tf_name == 'funding_feed':
                # 获取过去 window_size 的数据
                # get(ago=0, size=N) 返回的是 tuple 或 array
                raw_data = d.funding_rate.get(ago=0, size=window_size)
                funding_rate_window = list(raw_data)

                # 回退逻辑：如果提取为空或均为 None，则尝试获取当前值作为最小回退
                try:
                    if not funding_rate_window or all(x is None for x in funding_rate_window):
                        # 尝试读取当前刻度的单条值
                        try:
                            last_val = float(d.funding_rate[0])
                            funding_rate_window = [last_val]
                        except Exception:
                            funding_rate_window = []
                except Exception:
                    # 在一些非常规数据源上 all() 可能失败，保守处理
                    pass

                # 前向填充：如果仅有单值，复制以便后续计算能得到前后对比
                if funding_rate_window and len(funding_rate_window) == 1:
                    funding_rate_window = [funding_rate_window[0], funding_rate_window[0]]

                print(f"[FUNDING_RATE] Extracted {len(funding_rate_window)} values: {[f'{x:.5f}' if (x is not None) else 'None' for x in funding_rate_window[-5:]]}")
                continue # 跳过常规 K 线处理
            # --- 处理持仓量 ---
            if tf_name == 'oi_feed':
                raw_oi = d.oi_usd.get(ago=0, size=window_size)
                oi_usd_window = list(raw_oi)
                # 【新增】同时提取合约张数
                raw_contracts = d.oi_contracts.get(ago=0, size=window_size)
                oi_contracts_window = list(raw_contracts)

                # 回退逻辑：如果提取为空或均为 None，则尝试读取当前值
                try:
                    if not oi_usd_window or all(x is None for x in oi_usd_window):
                        try:
                            last_oi = float(d.oi_usd[0])
                            oi_usd_window = [last_oi]
                        except Exception:
                            oi_usd_window = []

                    if (not oi_contracts_window or all(x is None for x in oi_contracts_window)):
                        try:
                            last_contracts = float(d.oi_contracts[0])
                            oi_contracts_window = [last_contracts]
                        except Exception:
                            oi_contracts_window = []
                except Exception:
                    pass

                # 前向填充：如果仅有单值，复制以便后续计算能得到前后对比
                if oi_usd_window and len(oi_usd_window) == 1:
                    oi_usd_window = [oi_usd_window[0], oi_usd_window[0]]
                if oi_contracts_window and len(oi_contracts_window) == 1:
                    oi_contracts_window = [oi_contracts_window[0], oi_contracts_window[0]]

                print(f"[OPEN_INTEREST] Extracted {len(oi_usd_window)} OI values (USD): {[f'{x:.0f}' if (x is not None) else 'None' for x in oi_usd_window[-5:]]}")
                continue # 跳过常规 K 线处理

            # 【改进】对于1W和1M等高周期数据，总是尝试获取所有可用的历史数据（不限制lookback）
            # 这样可以确保从回测开始就有足够的20期历史用于市场分析
            is_sparse_tf = tf_name.lower() in ['1w', '1m']  # 识别稀疏周期
            # 【关键】对于高周期数据，用一个较大的固定值（如500）来获取足够的历史
            # 因为len(d)在回测初期可能不够，但由于fromdate=data_load_start，数据是充足的
            lookback_size = 500 if is_sparse_tf else window_size  # 高周期用500（足够获取20期+），常规周期用window_size

            # 提取价格数据
            try:
                prices = list(d.close.get(ago=0, size=lookback_size))
            except Exception:
                prices = []

            # 回退逻辑：如果仍然失败，手动逐个访问
            if not prices or len(prices) < 2:
                try:
                    max_bars = len(d)
                    if max_bars > 0:
                        prices = [float(d.close[-i]) for i in range(min(max_bars, 100), 0, -1)]
                        prices.reverse()
                    else:
                        prices = []
                except Exception as e:
                    print(f"[WARN] Failed to extract prices for {tf_name}: {e}")
                    prices = [float(d.close[0])] if len(d) > 0 else []

            # 提取成交量数据
            try:
                volumes = list(d.volume.get(ago=0, size=lookback_size))
            except Exception:
                volumes = []

            # 回退逻辑：如果仍然失败，手动逐个访问
            if not volumes or len(volumes) < 2:
                try:
                    max_bars = len(d)
                    if max_bars > 0:
                        volumes = [float(d.volume[-i]) for i in range(min(max_bars, 100), 0, -1)]
                        volumes.reverse()
                    else:
                        volumes = []
                except Exception as e:
                    print(f"[WARN] Failed to extract volumes for {tf_name}: {e}")
                    volumes = [float(d.volume[0])] if len(d) > 0 else []

            highs = list(d.high.get(ago=0, size=hl_window))
            lows = list(d.low.get(ago=0, size=hl_window))

            # 【修复】处理空列表的情况（在早期bars或稀疏高周期数据可能发生）
            highest_high = max(highs) if highs else 0.0
            lowest_low = min(lows) if lows else 0.0

            # 【新增】调试日志：输出高周期数据的提取情况
            if is_sparse_tf:
                print(f"[HIGH_TF_DEBUG] {tf_name}: extracted {len(prices)} prices, {len(volumes)} volumes (lookback_size={lookback_size}, len(d)={len(d)})", flush=True)

            # --- 【新增】提取30期OHLCV数据用于LSTM预处理（使用动态向量时间粒度数据）---
            if ohlcv_df is None and tf_name == self.main_data._name:
                try:
                    # 获取当前时间
                    cur_time = self.main_data.datetime.datetime(0)

                    # 【修改】使用动态向量时间粒度数据（由run()中的vector_tf参数决定）
                    # 这允许用户灵活指定向量匹配的时间粒度，可以获得不同粒度的市场模式识别
                    if self.params.h1_data is not None and len(self.params.h1_data) > 0:
                        # 从向量时间粒度数据中提取最近的30期OHLCV
                        # h1_data 现在指向 vector_tf_data（由run()初始化）

                        # 找到小于等于当前时间的所有向量数据
                        available_vector = self.params.h1_data[self.params.h1_data.index <= cur_time]

                        if len(available_vector) >= ohlcv_window:
                            # 取最后30期向量数据
                            vector_window = available_vector.tail(ohlcv_window)

                            opens = vector_window['Open'].values
                            close_list = vector_window['Close'].values
                            highs_ohlcv = vector_window['High'].values
                            lows_ohlcv = vector_window['Low'].values
                            volumes = vector_window['Volume'].values

                            # 构建DataFrame: Open,High,Low,Close,Volume (符合preprocess函数要求)
                            ohlcv_df = pd.DataFrame({
                                'Open': opens,
                                'High': highs_ohlcv,
                                'Low': lows_ohlcv,
                                'Close': close_list,
                                'Volume': volumes
                            })

                            print(f"[VECTOR_DATA] Using vector timeframe data (window: {vector_window.index[0].strftime('%Y-%m-%d %H:%M:%S')} to {vector_window.index[-1].strftime('%Y-%m-%d %H:%M:%S')})")
                        else:
                            # 数据不足30期，无法生成向量
                            print(f"[VECTOR_DATA_WARN] Insufficient vector data: {len(available_vector)} < {ohlcv_window} bars")
                            ohlcv_df = None
                    else:
                        # 没有加载向量数据，退用主周期数据（兼容模式）
                        print("[VECTOR_DATA_WARN] Vector timeframe data not available, using main timeframe data as fallback")

                        opens = list(d.open.get(ago=0, size=ohlcv_window))
                        close_list = list(d.close.get(ago=0, size=ohlcv_window))
                        highs_ohlcv = list(d.high.get(ago=0, size=ohlcv_window))
                        lows_ohlcv = list(d.low.get(ago=0, size=ohlcv_window))
                        volumes = list(d.volume.get(ago=0, size=ohlcv_window))

                        # 构建DataFrame: Open,High,Low,Close,Volume (符合preprocess函数要求)
                        ohlcv_df = pd.DataFrame({
                            'Open': opens,
                            'High': highs_ohlcv,
                            'Low': lows_ohlcv,
                            'Close': close_list,
                            'Volume': volumes
                        })
                except Exception as e:
                    print(f"[WARN] Failed to extract OHLCV data: {e}")
                    ohlcv_df = None
            
            ind_values = {}
            for name, ind in self.mtf_indicators[tf_name].items():
                aliases = ind.lines.getlinealiases()
                if len(aliases) > 1:
                    ind_data = {}
                    for alias in aliases:
                        line = getattr(ind.lines, alias)
                        ind_data[alias] = list(line.get(ago=0, size=self.params.lookback))
                    ind_values[name] = ind_data
                else:
                    line = getattr(ind.lines, aliases[0])
                    ind_values[name] = list(line.get(ago=0, size=self.params.lookback))
            
            mtf_state[tf_name] = {
                'prices': prices,
                'volumes': volumes,
                'indicators': ind_values, 
                'cur_price': d.close[0],
                'cur_volume': d.volume[0],
                'highest_high': highest_high,
                'lowest_low': lowest_low,
                'window_len': hl_window
            }

        agent_state = {
            'dt_str':self.main_data.datetime.datetime(0).isoformat(),
            'mtf_data': mtf_state,
            'weekly_data': weekly_state,
            'cash': self.broker.getcash(),
            'position': self.position.size,
            'avg_price': self.position.price if self.position.size != 0 else 0.0,
            'roi_status': self.roi_msg,
            'history': self.history_log[-100:],
            'cur_price': cur_price, # 全局最新价
            'current_tp': self.current_tp,
            'current_sl': self.current_sl,
            'funding_rate_list': funding_rate_window,
            'oi_usd_list': oi_usd_window,
            'oi_contracts_list': oi_contracts_window,  # 【新增】合约张数列表
            # 当期简洁值，供 Agent 使用更小的提示词：
            'funding_rate_current': (funding_rate_window[-1] if funding_rate_window else None),
            'funding_rate_delta': (
                (funding_rate_window[-1] - funding_rate_window[-2]) if len(funding_rate_window) > 1 and funding_rate_window[-1] != funding_rate_window[-2]
                else (funding_rate_window[-1] - self.prev_funding_rate if self.prev_funding_rate is not None and funding_rate_window else None)
            ),
            'oi_usd_current': (oi_usd_window[-1] if oi_usd_window else None),
            'oi_usd_change_pct': (
                ( (oi_usd_window[-1]-oi_usd_window[-2]) / oi_usd_window[-2]*100 ) if len(oi_usd_window) > 1 and oi_usd_window[-2] not in (0,None) else
                ( (oi_usd_window[-1]-self.prev_oi_usd) / self.prev_oi_usd*100 if self.prev_oi_usd is not None and self.prev_oi_usd != 0 and oi_usd_window else None)
            ),
            'oi_contracts_current': (oi_contracts_window[-1] if oi_contracts_window else None),
            'ohlcv_data': ohlcv_df,  # 【新增】30期OHLCV数据, 可直接用于preprocess()
            # 【新增】成交量信息
            'volume_current': (self.main_data.volume[0] if hasattr(self.main_data, 'volume') else None),
            'volume_list': (list(self.main_data.volume.get(ago=0, size=window_size)) if hasattr(self.main_data, 'volume') else []),
        }

        # [VECTOR DB INTEGRATION] Query similar historical patterns
        similar_vectors = None
        similar_event_ids = []
        has_similar_pattern = False  # 【新增】标记是否找到相似行情
        if self.params.vec_db is not None and ohlcv_df is not None:
            try:
                ohlcv_array = ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
                similar_vectors = self.params.vec_db.query_similar(
                    ohlcv_array,
                    top_k=5,
                    threshold=0.9 # 相似度阈值超过1则禁用
                )
                # Extract event IDs from timestamps
                # 【说明】向量库返回的是开仓时间戳，提取秒级格式以匹配图库的event_id
                similar_event_ids = [r['timestamp'].strftime('%Y%m%d_%H%M%S') for r in similar_vectors]
                has_similar_pattern = len(similar_event_ids) > 0  # 【新增】设置标记
                print(f"[VECTOR_DB] Found {len(similar_event_ids)} similar patterns: {similar_event_ids[:5]}")
            except Exception as e:
                print(f"[VECTOR_DB_ERROR] Query failed: {e}")
                similar_vectors = None
                similar_event_ids = []
                has_similar_pattern = False  # 【新增】设置标记

        # [GRAPH DB INTEGRATION] Query historical pattern analysis
        graph_insight = None
        if self.params.graph_db is not None and similar_event_ids:
            try:
                # Add _P suffix to match Graph DB event_id format
                # 【说明】similar_event_ids 是从向量库提取的开仓时间戳（秒级）
                # 图库的event_id也是基于开仓时间，所以现在可以完美匹配
                similar_event_ids_with_suffix = [f"{eid}_P" for eid in similar_event_ids]
                graph_insight = self.params.graph_db.query_similar_events_insight(similar_event_ids_with_suffix)
                if graph_insight and 'dominant_pattern' in graph_insight:
                    dominant = graph_insight['dominant_pattern']
                    if dominant is not None:
                        print(f"[GRAPH_DB] Dominant pattern: {dominant.get('name', 'Unknown')} (win_rate: {dominant.get('baseline_win_rate', 0):.1%})")
            except Exception as e:
                print(f"[GRAPH_DB_ERROR] Query failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                graph_insight = None
        else:
            if self.params.graph_db is None:
                print(f"[GRAPH_DB_DEBUG] graph_db is None - not initialized")
            elif not similar_event_ids:
                print(f"[GRAPH_DB_DEBUG] similar_event_ids is empty - no matching patterns found in vector DB")

        # Add vector and graph DB results to agent state (optional keys)
        agent_state['similar_vectors'] = similar_vectors if similar_vectors else []
        agent_state['graph_insight'] = graph_insight if graph_insight else None

        # 触发决策
        self.log(f"P:{cur_price:.0f} | Calling agent...")
        print(f"[AGENT_CALL] Bar {self.bar_count}: Calling agent.decide() at {self.main_data.datetime.datetime(0)}")
        try:
            # 【新增】Pass leverage info to agent for risk management education
            self.params.agent._bar_count = self.bar_count
            self.params.agent._current_leverage = self.params.leverage  # 传递当前系统杠杆倍数
            self.params.agent._main_tf = self.params.main_tf  # 【新增】传递主要交易周期
            print(f"[LEVERAGE_INFO] Current system leverage: {self.params.leverage:.1f}x | Agent will use this for risk guidance")
            print(f"[TRADING_CONFIG] Main trading timeframe: {self.params.main_tf} | Agent will use this for decision context")
            decision = self.params.agent.decide(agent_state)
            print(f"[AGENT_RESP] Got decision: {decision.action}")
        except Exception as e:
            print(f"[AGENT_ERROR] Error calling agent: {e}")
            raise

        # ============================================
        # 【新增】PHASE 3: 从 Agent 中提取结构化输出
        # ============================================
        # 从 Agent 实例中获取刚生成的 Agent S/A 输出对象
        agent_s_output_obj = getattr(self.params.agent, '_agent_s_output_obj', None)
        agent_a_output_obj = getattr(self.params.agent, '_agent_a_output_obj', None)
        agent_b_output_obj = getattr(self.params.agent, '_agent_b_output_obj', None)

        if agent_s_output_obj:
            # 【改进】支持新的AgentSOutput格式（Dict格式状态）
            try:
                if hasattr(agent_s_output_obj, 'current_states'):
                    num_states = len(agent_s_output_obj.current_states) if isinstance(agent_s_output_obj.current_states, dict) else len(agent_s_output_obj.current_states) if isinstance(agent_s_output_obj.current_states, list) else 0
                elif hasattr(agent_s_output_obj, 'data'):
                    num_states = len(agent_s_output_obj.data.current_states)
                else:
                    num_states = 0
                print(f"[AGENT_S_EXTRACT] Extracted AgentSOutput: states={num_states}")
            except Exception as e:
                print(f"[AGENT_S_EXTRACT_ERROR] {e}")
        if agent_a_output_obj:
            # 【改进】支持新的AgentAOutput格式（结构化的pattern和逻辑状态）
            try:
                if hasattr(agent_a_output_obj, 'pattern_name'):
                    pattern_name = agent_a_output_obj.pattern_name
                    confidence = agent_a_output_obj.confidence
                    print(f"[AGENT_A_EXTRACT] Extracted AgentAOutput: pattern={pattern_name}, confidence={confidence:.1%}")
                else:
                    print(f"[AGENT_A_EXTRACT_ERROR] AgentAOutput missing pattern_name attribute")
            except Exception as e:
                print(f"[AGENT_A_EXTRACT_ERROR] {e}")

        dt_str = self.main_data.datetime.datetime(0).isoformat()

        # 【新增】如果有交易动作，保存 Agent 输出到 position_metadata
        if decision.action in ['LONG', 'SHORT']:
            self.position_metadata['s_output'] = agent_s_output_obj
            self.position_metadata['a_output'] = agent_a_output_obj
            self.position_metadata['b_output'] = agent_b_output_obj
            print(f"[POSITION_METADATA] Saved Agent outputs (S/A/B) for TradeContext")


        # 记录时间序列快照（在做出决策后记录）
        # 【修复】如果刚刚平仓完成（position.size == 0 且持有挂单），则不记录快照
        # 因为平仓信息已经在 notify_trade() 中完整记录过了
        snap_row = None
        try:
            cur_val = self.broker.getvalue()
            cur_price = self.main_data.close[0]
            # 已计算 alpha 变量在上方
            snap_row = {
                'timestamp': dt_str,
                'price': float(cur_price),
                'net_value': float(cur_val),
                'alpha': float(alpha),
                'decision_made': False,
                'decision_type': 'HOLD',
                'similar_pattern_found': has_similar_pattern  # 【新增】记录是否匹配到相似行情
            }
        except Exception:
            snap_row = None

        if decision.action != 'HOLD':
            # 计算 TP/SL 百分比用于日志
            tp_str = ""
            if decision.take_profit_price > 0:
                tp_pct = (decision.take_profit_price - cur_price) / cur_price if cur_price != 0 else 0.0
                tp_str = f"TP:{decision.take_profit_price:.0f}({tp_pct:+.2%})"

            sl_str = ""
            if decision.stop_loss_price > 0:
                sl_pct = (decision.stop_loss_price - cur_price) / cur_price if cur_price != 0 else 0.0
                sl_str = f"SL:{decision.stop_loss_price:.0f}({sl_pct:+.2%})"

            print(f"      >>> [AGENT] @ {cur_price:.0f} [{decision.action}] Pos:{decision.quantity_pct:.0%} {tp_str} {sl_str}")
            print(f"          Reason: {decision.reasoning[:30]}...")

            entry = f"{dt_str}: {decision.action} @ {cur_price:.0f} {tp_str} {sl_str}"
            self.history_log.append(entry)

            # 【修复】CLOSE 决策不在这里记录，而是在 notify_trade() 中记录，以避免重复
            # 只有 LONG/SHORT 决策才在这里记录
            if decision.action != 'CLOSE':
                # 【新增】记录 LONG/SHORT 决策到交易日志
                try:
                    if self.trade_decision_logger:
                        self.trade_decision_logger.log_decision(
                            timestamp=dt_str,
                            bar_count=self.bar_count,
                            decision_type=decision.action,
                            decision_price=cur_price,
                            decision_quantity=decision.quantity_pct,
                            take_profit=decision.take_profit_price,
                            stop_loss=decision.stop_loss_price,
                            reason=decision.reasoning[:100],
                            position_before=self.position.size,
                            similar_pattern_found=has_similar_pattern  # 【新增】传递相似行情标记
                        )
                except Exception as e:
                    print(f"[LOG_ERROR] Failed to log decision: {e}")

                # 标记快照中的决策信息（仅用于 snapshot CSV，不关联 trade_decision_logger）
                # 【修复】只有非CLOSE决策才记录到快照，CLOSE决策由notify_trade()负责记录
                try:
                    if snap_row is not None:
                        snap_row['decision_made'] = True
                        snap_row['decision_type'] = decision.action
                except Exception:
                    pass
        else:
            # HOLD 时打印看板
            self._print_dashboard("[HOLD_WATCH]")
            print(f"      >>> [AGENT] [HOLD] Reason: {decision.reasoning[:50]}...")

        # 将快照以流式方式写入 CSV（若已配置），并同时保留于内存列表
        try:
            if snap_row is not None:
                # 追加到内存（保留历史小样本）
                try:
                    self.tick_snapshot_rows.append(snap_row)
                except Exception:
                    pass

                if self._snapshot_writer:
                    try:
                        row_out = {
                            'timestamp': snap_row.get('timestamp',''),
                            'price': snap_row.get('price',''),
                            'open': float(self.main_data.open[0]) if hasattr(self.main_data, 'open') else '',
                            'high': float(self.main_data.high[0]) if hasattr(self.main_data, 'high') else '',
                            'low': float(self.main_data.low[0]) if hasattr(self.main_data, 'low') else '',
                            'close': float(self.main_data.close[0]) if hasattr(self.main_data, 'close') else snap_row.get('price',''),
                            'net_value': snap_row.get('net_value',''),
                            'alpha': snap_row.get('alpha',''),
                            'position_size': float(self.position.size) if hasattr(self, 'position') else '',
                            'avg_price': float(self.position.price) if hasattr(self.position, 'price') and self.position.size != 0 else snap_row.get('avg_price',''),
                            'decision_made': bool(snap_row.get('decision_made', False)),
                            'decision_type': snap_row.get('decision_type',''),
                            'quantity_pct': getattr(decision, 'quantity_pct', '') if decision is not None else '',
                            'take_profit_price': getattr(decision, 'take_profit_price', '') if decision is not None else '',
                            'stop_loss_price': getattr(decision, 'stop_loss_price', '') if decision is not None else '',
                            'similar_pattern_found': 'YES' if snap_row.get('similar_pattern_found', False) else 'NO'  # 【新增】格式化输出
                        }
                        self._snapshot_writer.writerow(row_out)
                        try:
                            self._snapshot_file.flush()
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

        # 执行交易
        if decision.action != 'HOLD':
            # 【新增】开仓时保存市场状态快照（混合策略：硬编码 + Agent 补充）
            if decision.action in ['LONG', 'SHORT']:
                # 调用 _extract_market_states，同时传入 agent 让其补充状态
                agent_instance = getattr(self.params, 'agent', None)
                state_result = self._extract_market_states(agent_state, agent=agent_instance)

                # 取出所有合并的状态
                all_market_states = state_result.get('all_states', [])
                base_states = state_result.get('base_states', [])
                additional_states = state_result.get('additional_states', [])
                confidence = state_result.get('confidence', {})

                # 【关键改进】对所有状态列表进行排序，确保Hash稳定性
                # 这样无论Agent S以什么顺序返回状态，都能被正确地映射到同一Pattern
                all_market_states_sorted = sorted(all_market_states) if all_market_states else []
                base_states_sorted = sorted(base_states) if base_states else []
                additional_states_sorted = sorted(additional_states) if additional_states else []

                print(f"[ENTRY_STATE_SORTED] Original: {all_market_states} → Sorted: {all_market_states_sorted}", flush=True)

                dt_str = self.main_data.datetime.datetime(0).isoformat()
                # 【修复】保存开仓时的OHLCV数据，供平仓时添加到向量库
                entry_ohlcv = None
                if ohlcv_df is not None:
                    entry_ohlcv = ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume']].values

                # 【关键修复】保留之前保存的 Agent 输出（s_output, a_output, b_output）
                # 这些输出是在第 1892-1894 行保存的，不应该被清除
                preserved_agent_s_output = self.position_metadata.get('s_output')
                preserved_agent_a_output = self.position_metadata.get('a_output')
                preserved_agent_b_output = self.position_metadata.get('b_output')

                self.position_metadata = {
                    'market_states': all_market_states_sorted,  # 【改进】使用已排序的状态
                    'base_states': base_states_sorted,          # 【改进】使用已排序的状态
                    'additional_states': additional_states_sorted,  # 【改进】使用已排序的状态
                    'confidence': confidence,
                    'entry_price': cur_price,
                    'entry_time': dt_str,
                    'entry_bars': self.bar_count,
                    'entry_decision': decision.action,
                    'entry_ohlcv': entry_ohlcv,  # 【新增】开仓时的OHLCV数据
                    # 【修复】恢复之前保存的 Agent 输出
                    's_output': preserved_agent_s_output,
                    'a_output': preserved_agent_a_output,
                    'b_output': preserved_agent_b_output,
                }

                # 日志输出
                print(f"[STATE_SNAPSHOT] 开仓时市场状态（Agent S）: {', '.join(all_market_states_sorted)}", flush=True)

            # 更新计划
            if decision.action in ['LONG', 'SHORT']:
                if decision.take_profit_price > 0: self.current_tp = decision.take_profit_price
                if decision.stop_loss_price > 0: self.current_sl = decision.stop_loss_price
            elif decision.action == 'CLOSE':
                # 【新增】标记为主动平仓（Agent B决策），而不是自动TP/SL平仓
                self.last_close_reason = "MANUAL"
                self.current_tp = 0.0
                self.current_sl = 0.0

            total_val = self.broker.getvalue()
            target_val = total_val * decision.quantity_pct * self.params.leverage
            max_allowed = total_val * self.params.leverage * 0.98
            target_val = min(target_val, max_allowed)

            try:
                if decision.action == 'LONG':
                    self.order = self.order_target_value(data=self.main_data, target=target_val)
                elif decision.action == 'SHORT':
                    self.order = self.order_target_value(data=self.main_data, target=-target_val)
                elif decision.action == 'CLOSE':
                    if self.position.size != 0:
                        # 【改进】平仓时：
                        # 1. 首先尝试 order_target_value(0) - 更可靠
                        # 2. 如果失败，回退到 self.close()
                        try:
                            self.order = self.order_target_value(data=self.main_data, target=0)
                            print(f"[CLOSE] Using order_target_value(0) to close position")
                        except Exception as e:
                            print(f"[CLOSE_WARN] order_target_value failed: {e}, fallback to close()")
                            self.order = self.close(data=self.main_data)
            except Exception as e:
                print(f"[ORDER_ERROR] Failed to place order: {e}")
                self.order = None
                # 【新增】强制清理保护挂单，以便后续重试
                try:
                    self.update_protection()
                except Exception:
                    pass

            # 【新增】Agent决策后跳过4个bars以加快测试速度
            # self.skip_next_bars = 4
            self.skip_next_bars = 0
            print(f"[AGENT_DECISION] Action: {decision.action} | Skipping next 4 bars for faster testing", flush=True)

        # 【新增】更新上一期的资金费率和持仓量缓存，供下一个next周期使用
        if funding_rate_window:
            self.prev_funding_rate = funding_rate_window[-1]
        if oi_usd_window:
            self.prev_oi_usd = oi_usd_window[-1]


class BacktestRunner:
    def __init__(self, agent, data_dir, start, end, cash=100000.0, commission=0.001, leverage=1.0, high_low_window=20):
        self.agent = agent
        self.data_dir = data_dir
        self.start = start
        self.end = end
        self.cash = cash
        self.commission = commission
        self.leverage = leverage
        self.high_low_window = high_low_window
        self.indicators_config = []

        # Vector DB and Graph DB optional integrations
        self.vec_db: Optional[Any] = None  # Will be initialized by caller if vector DB is enabled
        self.graph_db: Optional[Any] = None  # Will be initialized by caller if graph DB is enabled
        self.h1_data: Optional[Any] = None  # 【已弃用】保留兼容性，由 vector_tf_data 替代
        self.vector_tf_data: Optional[Any] = None  # 【新增】动态向量数据，由run()方法根据vector_tf参数初始化
        self.vector_tf: str = '1h'  # 【新增】向量时间粒度，默认为1h

    def add_indicator(self, cls, name=None, **kwargs):
        if name is None: name = cls.__name__.lower()
        self.indicators_config.append({'cls': cls, 'name': name, 'kwargs': kwargs})

    def run(self, start_backtest=True, main_tf='1h', vector_tf='1h'):
        # 【新增】保存向量时间粒度配置和主要交易周期
        self.vector_tf = vector_tf
        self.main_tf = main_tf  # 【新增】保存主要交易周期，供Agent使用

        cerebro = bt.Cerebro()

        # --- 添加性能分析 ---
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, timeframe=bt.TimeFrame.Days, compression=1)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        # -----------------------------------------
        tf_weights = {
            '1m': 1,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080,
            '1M': 43200  # 【新增】月度（约30天的分钟数）
        }
        all_tfs = {
                    '1m':  'BTC-USDT_1m_2024_2025.csv',
                    '15m': 'BTC-USDT_15m_2021_2025.csv',
                    '1h':  'BTC-USDT_1H_2021_2025.csv',
                    '4h':  'BTC-USDT_4H_2021_2025.csv',
                    '1d':  'BTC-USDT_1D_2021_2025.csv',
                    '1w':  'BTC-USDT_1W_2021_2025.csv',
                    '1M':  'BTC-USDT_1mon_2021_2025.csv'  # 【新增】月度数据
                }
        if main_tf not in all_tfs:
            print(f"[ERROR] 错误: 不支持的主周期 '{main_tf}'")
            return None
        
        main_weight = tf_weights[main_tf]
        
        # 1. 确定加载顺序：主周期必须第一个加载
        # 2. 过滤逻辑：只加载 >= 主周期的数据
        load_order = [main_tf] # 先放主周期

        # 按照权重顺序添加（确保一致性）
        for tf in ['15m', '1h', '4h', '1d', '1w', '1M']:  # 【修改】加入 '1M'
            if tf == main_tf:
                continue # 跳过主周期(已添加)
            if tf_weights[tf] >= main_weight:  # 只添加比主周期大(或相等)的数据
                load_order.append(tf)
                print(f"[DATA_LOAD] Added {tf} (weight={tf_weights[tf]}) to load_order")

        print(f"[DATA LOAD] Dynamic loading: {load_order} (filtered < {main_tf})")

        print(f"[DATA] Adding {len(load_order)} feeds to cerebro...")

        # 【关键】第一遍：只加载主周期数据，用来建立对齐的时间索引
        start_date = pd.to_datetime(self.start)
        end_date = pd.to_datetime(self.end)

        # 【新增】为了高周期数据（尤其是1M月度）有足够的历史，
        # 数据加载范围应该比交易范围更早。从start之前12个月开始加载，
        # 这样在回测真正开始时，高周期数据就有了完整的历史
        data_load_start = start_date - pd.DateOffset(months=24)  # 提前13个月加载数据

        for tf_name in load_order:
                    if tf_name != main_tf:
                        continue  # 先跳过高周期

                    file_name = all_tfs[tf_name]
                    path = os.path.join(self.data_dir, file_name)
                    if not os.path.exists(path): print(f"[ERROR] Missing file: {path}"); return None

                    try:
                        # 【优化】对于1m数据，先在pandas中过滤再加载到backtrader，避免加载无用数据
                        print(f"  [LOAD] Loading {tf_name} data (PRIMARY FEED)...")
                        df = pd.read_csv(path)
                        df.rename(columns={'ts':'datetime', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'vol':'Volume'}, inplace=True)
                        df['datetime'] = pd.to_datetime(df['datetime'])

                        # 【关键】数据加载范围：data_load_start → end_date（包含预热期）
                        # 策略执行范围：start_date → end_date（用户指定的交易范围）
                        # DataFrame包含完整历史，但Backtrader只从start_date开始执行next()
                        df = df[(df['datetime'] >= data_load_start) & (df['datetime'] <= end_date)]
                        print(f"  [OK] {tf_name}: {len(df):,} rows (data from {data_load_start.date()}, execution from {start_date.date()})")

                        df.set_index('datetime', inplace=True)
                        df.sort_index(ascending=True, inplace=True)

                        data = bt.feeds.PandasData(
                            dataname=df,  # type: ignore
                            name=tf_name, # 这里的 name 对应策略里 d._name # type: ignore
                            fromdate=start_date,  # type: ignore - 策略从用户指定的start_date开始执行
                            todate=end_date # type: ignore
                        )
                        cerebro.adddata(data)
                        print(f"  [ADD] {tf_name} added to cerebro (PRIMARY, with warmup from {data_load_start.date()})", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Failed to load {tf_name}: {e}")
                        return None

        # 【修复】删除不必要的时间索引生成
        # Backtrader 本身支持不同频率的多周期数据，无需手动对齐
        # 这里不再生成 full_time_index 和进行 reindex + ffill

        # 【修复】第二遍：加载高周期数据 - 保持原始稀疏格式
        # Backtrader 会自动处理多周期同步
        for tf_name in load_order:
                    if tf_name == main_tf:
                        continue  # 主周期已加载

                    file_name = all_tfs[tf_name]
                    path = os.path.join(self.data_dir, file_name)
                    if not os.path.exists(path): print(f"[ERROR] Missing file: {path}"); return None

                    try:
                        print(f"  [LOAD] Loading {tf_name} data (sparse native format, no reindex)...")
                        df = pd.read_csv(path)
                        df.rename(columns={'ts':'datetime', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'vol':'Volume'}, inplace=True)
                        df['datetime'] = pd.to_datetime(df['datetime'])

                        # 【关键】数据加载范围：data_load_start → end_date（包含预热期）
                        # 这确保高周期数据（1W、1M）有足够的历史bars供策略预热使用
                        # 策略执行范围：start_date → end_date（由fromdate参数控制）
                        df = df[(df['datetime'] >= data_load_start) & (df['datetime'] <= end_date)]
                        print(f"  [OK] {tf_name}: {len(df):,} rows (data from {data_load_start.date()}, execution from {start_date.date()})")

                        df.set_index('datetime', inplace=True)
                        df.sort_index(ascending=True, inplace=True)

                        # 【修复】不做 reindex 和 ffill
                        # 让 Backtrader 的原生多周期机制自动处理
                        # Backtrader 会在主周期的每个 next() 调用时同步所有 feeds
                        # 如果高周期在该时刻没有新 K线，会自动使用前一个 K线

                        # 不做任何处理，保持原始数据格式

                        data = bt.feeds.PandasData(
                            dataname=df,  # type: ignore
                            name=tf_name, # 这里的 name 对应策略里 d._name # type: ignore
                            fromdate=data_load_start,  # type: ignore - 【修复】使用data_load_start以包含预热期的历史数据，这样len(d)才能获取到足够的20期+
                            todate=end_date # type: ignore
                        )
                        # --- 【修改】只让主周期绘图，其他周期隐藏 ---
                        data.plotinfo.plot = False
                        # ------------------------------------------
                        cerebro.adddata(data)
                        print(f"  [ADD] {tf_name} added to cerebro (native sparse, backtrader will auto-sync, with warmup from {data_load_start.date()})", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Failed to load {tf_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        return None

        # ==========================================
        # 新增：加载辅助数据 (资金费率 & 持仓量)
        # ==========================================
        # 1. 加载资金费率 (Funding Rate)
        fr_path = os.path.join(self.data_dir, 'BTC-USDT-SWAP_FundingRate.csv')
        if os.path.exists(fr_path):
            try:
                df_fr = pd.read_csv(fr_path)
                df_fr['datetime'] = pd.to_datetime(df_fr['datetime'])
                # 去重：防止同一时间有多条数据
                df_fr.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
                df_fr.set_index('datetime', inplace=True)
                df_fr.sort_index(inplace=True)

                # 重采样并填充：对齐到主周期 (如 1H)，ffill 填补空缺
                # 【修改】对于1m周期，资金费率以小时粒度为准，无需过度细化
                resample_freq = main_tf if main_tf not in ['1m', '15m'] else '1H'
                df_fr = df_fr.resample(resample_freq).ffill()

                # 【新增】为缺失的早期日期补充数据，确保从data_load_start日期开始有数据
                # 如果数据的最早日期晚于data_load_start，用0填充早期数据（诚实地表示"无数据"）
                # 【修复】使用 data_load_start 而非 start_date，确保与主数据加载范围一致

                if df_fr.index[0] > data_load_start:
                    # 创建完整的日期范围
                    full_date_range = pd.date_range(start=data_load_start, end=df_fr.index[-1], freq=resample_freq)
                    df_fr = df_fr.reindex(full_date_range)
                    # 用0填充缺失的早期数据（表示无法获取，防止传递错误信息给agent）
                    df_fr = df_fr.fillna(0.0)
                    first_valid_idx = df_fr[df_fr != 0].index[0] if (df_fr != 0).any().any() else df_fr.index[0]
                    print(f"  [ADD_FR] Filled funding_rate with 0 from {data_load_start} to {first_valid_idx} (data not available)", flush=True)

                # 截取回测时间段（使用 data_load_start 以包含预热期）
                df_fr = df_fr[(df_fr.index >= data_load_start) &
                              (df_fr.index <= end_date)]

                fr_data = FundingRateData(
                    dataname=df_fr,
                    name='funding_feed'  # 关键标识
                )
                cerebro.adddata(fr_data)
                print(f"  [ADD] funding_feed added: {len(df_fr)} rows (from {df_fr.index[0]} to {df_fr.index[-1]})", flush=True)
            except Exception as e:
                print(f"[WARN] Funding rate load failed: {e}")

        # 2. 加载持仓量 (Open Interest)
        oi_path = os.path.join(self.data_dir, 'BTC-USDT-SWAP_OpenInterest_4H.csv')
        if os.path.exists(oi_path):
            try:
                df_oi = pd.read_csv(oi_path)
                df_oi['datetime'] = pd.to_datetime(df_oi['datetime'])
                # 去重 (你提供的样本有重复行，必须去重)
                df_oi.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
                df_oi.set_index('datetime', inplace=True)
                df_oi.sort_index(inplace=True)

                # 重采样：虽然源数据可能是 1H，但为了保险起见，强制对齐主周期
                # 持仓量用 ffill 比较合理（如果某小时缺失，认为持仓量未变）
                # 【修改】对于1m周期，持仓量以4H粒度为准，无需过度细化
                resample_freq = main_tf if main_tf not in ['1m', '15m', '1h'] else '4H'
                df_oi = df_oi.resample(resample_freq).ffill()

                # 【新增】为缺失的早期日期补充数据，确保从data_load_start日期开始有数据
                # 如果数据的最早日期晚于data_load_start，用0填充早期数据（诚实地表示"无数据"）
                # 【修复】使用 data_load_start 而非 start_date，确保与主数据加载范围一致

                if df_oi.index[0] > data_load_start:
                    # 创建完整的日期范围
                    full_date_range = pd.date_range(start=data_load_start, end=df_oi.index[-1], freq=resample_freq)
                    df_oi = df_oi.reindex(full_date_range)
                    # 用0填充缺失的早期数据（表示无数据，防止传递错误信息给agent）
                    df_oi = df_oi.fillna(0.0)
                    first_valid_idx = df_oi[(df_oi != 0).any(axis=1)].index[0] if (df_oi != 0).any().any() else df_oi.index[0]
                    print(f"  [ADD_OI] Filled OI data with 0 from {data_load_start} to {first_valid_idx} (data not available)", flush=True)

                # 截取回测时间段（使用 data_load_start 以包含预热期）
                df_oi = df_oi[(df_oi.index >= data_load_start) &
                              (df_oi.index <= end_date)]

                oi_data = OpenInterestData(
                    dataname=df_oi,
                    name='oi_feed'  # 关键标识
                )
                cerebro.adddata(oi_data)
                print(f"  [ADD] oi_feed added: {len(df_oi)} rows (from {df_oi.index[0]} to {df_oi.index[-1]})", flush=True)
            except Exception as e:
                print(f"[WARN] Open Interest load failed: {e}")

        # ==========================================
        # 【新增】为向量库加载动态时间粒度数据（独立于主回测周期）
        # ==========================================
        # 说明：根据 vector_tf 参数动态加载向量数据，允许用户指定向量匹配的时间粒度
        # 例如：可以在 4H 回测中使用 1H 数据生成向量，获得更细粒度的模式识别
        try:
            print(f"  [LOAD] Loading {vector_tf.upper()} data for vector database...")

            # 确定向量数据文件名（使用 all_tfs 字典中的映射）
            vector_file = all_tfs.get(vector_tf, f'BTC-USDT_{vector_tf.upper()}_2021_2025.csv')
            vector_path = os.path.join(self.data_dir, vector_file)

            if os.path.exists(vector_path):
                df_vector = pd.read_csv(vector_path)
                df_vector.rename(columns={'ts':'datetime', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'vol':'Volume'}, inplace=True)
                df_vector['datetime'] = pd.to_datetime(df_vector['datetime'])

                # 过滤时间范围（使用 data_load_start 以包含预热期）
                df_vector = df_vector[(df_vector['datetime'] >= data_load_start) & (df_vector['datetime'] <= end_date)]
                print(f"  [OK] {vector_tf.upper()} data for vector: {len(df_vector):,} rows (with warmup from {data_load_start.date()})")

                df_vector.set_index('datetime', inplace=True)
                df_vector.sort_index(ascending=True, inplace=True)

                # 【修改】存储到 vector_tf_data 而非 h1_data
                self.vector_tf_data = df_vector

                print(f"  [OK] {vector_tf.upper()} data cached for vector database operations (vector_tf='{vector_tf}')")
            else:
                print(f"[WARN] Vector data file not found: {vector_path}")
                self.vector_tf_data = None

        except Exception as e:
            print(f"[WARN] Failed to load {vector_tf.upper()} data for vector DB: {e}")
            self.vector_tf_data = None

        # 预先创建 snapshot 目录并把 CSV 路径传入策略以便流式写入
        try:
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            snapshot_dir = os.path.join(repo_root, 'snapshots', ts)
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_csv = os.path.join(snapshot_dir, 'time_series_snapshot.csv')
        except Exception:
            snapshot_dir = None
            snapshot_csv = None

        cerebro.addstrategy(UniversalLLMStrategy,
                            agent=self.agent,
                            indicators_config=self.indicators_config,
                            leverage=self.leverage,
                            high_low_window=20,
                            snapshot_csv=snapshot_csv,
                            vec_db=self.vec_db,
                            graph_db=self.graph_db,
                            h1_data=self.vector_tf_data,  # 【修改】传递动态向量数据（vector_tf_data）
                            main_tf=self.main_tf)  # 【新增】传递主要交易周期给Strategy
        cerebro.broker.setcash(self.cash)
        
        try:
            comm_info = MarginCommInfo(commission=self.commission, interest=0.01, leverage=self.leverage)
            cerebro.broker.addcommissioninfo(comm_info)
        except NameError:
            cerebro.broker.setcommission(commission=self.commission)

        print(f"[INIT] Engine initialized | Cash: {self.cash}")

        if not start_backtest:
            return cerebro

        # 【调试】在回测前检查 feeds 的数据
        print(f"[DEBUG] Total feeds: {len(cerebro.datas)}")
        for i, d in enumerate(cerebro.datas):
            print(f"[DEBUG] Feed {i} ({d._name}): buffer_len={len(d)}")

        print("[DEBUG] Starting cerebro.run()...")
        results = cerebro.run(runonce=False)
        print(f"[DEBUG] cerebro.run() completed with {len(results) if results else 0} results")

        # 【修复】移除自动保存快照的逻辑，统一在main.py中保存
        # 这样避免重复创建snapshot目录
        # 快照将在main.py的最后统一保存，包含所有分析结果

        end_val = cerebro.broker.getvalue()
        perf_pct = ((end_val - self.cash) / self.cash) if (self.cash and self.cash != 0) else 0.0
        print(f"\n[FINISH] 结束资金: {end_val:.2f} (收益: {perf_pct:.2%})")

        # 【修复】返回 (results, cerebro, snapshot_dir) 以供 main.py 后续使用
        # snapshot_dir 是在run()开始时创建的，用于流式写入CSV
        # 传递给main.py后，save_snapshot会使用这个已存在的目录，避免重复创建
        return results, cerebro, snapshot_dir if 'snapshot_dir' in locals() else None

    def save_snapshot(self, cerebro, strat, out_dir: Optional[str] = None, include_pickle: bool = False,
                     save_vector_db: bool = True):
        """保存回测快照：analyzers、broker 状态、strategy 日志与 trade_history.csv。

        - out_dir: 指定输出目录；默认为项目根下的 `snapshots/<timestamp>`。
        - include_pickle: 若为 True，会尝试 pickle 整个 `cerebro` 与 `strat`（可能失败，依赖对象可序列化性）。
        - save_vector_db: 是否保存向量数据库到磁盘（默认True）。
        返回写入的目录路径或 None（失败）。
        """
        try:
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if out_dir is None:
                out_dir = os.path.join(repo_root, 'snapshots', ts)
            os.makedirs(out_dir, exist_ok=True)

            # 【新增】保存向量数据库到磁盘
            if save_vector_db and self.vec_db is not None:
                try:
                    from ace_trading.LSTM.vector_db_manager import VectorDatabaseManager
                    manager = VectorDatabaseManager(out_dir)
                    manager.save_from_vec_db(self.vec_db)
                    manager.create_summary_report(os.path.join(out_dir, 'vector_db_summary.txt'))
                    print(f"[VECTOR_DB_SAVE] 向量数据库已保存到: {out_dir}")
                except Exception as e:
                    print(f"[VECTOR_DB_SAVE_ERROR] 保存向量数据库失败: {e}")

            analyzer_names = ['sharpe', 'drawdown', 'returns', 'trades']
            analyzers_data = {}
            for name in analyzer_names:
                try:
                    ana = getattr(strat.analyzers, name)
                    analyzers_data[name] = ana.get_analysis()
                except Exception as e:
                    analyzers_data[name] = {'error': str(e)}

            with open(os.path.join(out_dir, 'analyzers.json'), 'w', encoding='utf-8') as f:
                json.dump(analyzers_data, f, default=str, ensure_ascii=False, indent=2)

            meta = {
                'broker_value': cerebro.broker.getvalue(),
                'broker_cash': cerebro.broker.getcash(),
                'position_size': getattr(strat.position, 'size', None),
                'position_price': getattr(strat.position, 'price', None),
                'start_cash': self.cash,
                'params': {
                    'leverage': self.leverage,
                    'commission': self.commission,
                    'start': self.start,
                    'end': self.end
                }
            }
            with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, default=str, ensure_ascii=False, indent=2)

            try:
                hist = getattr(strat, 'history_log', [])
                with open(os.path.join(out_dir, 'history_log.txt'), 'w', encoding='utf-8') as f:
                    for line in hist:
                        f.write(line + '\n')
            except Exception:
                pass

            # 如果策略使用了流式 CSV 写入（snapshot_csv），优先关闭并复制该文件，避免覆盖或丢失列
            try:
                copied = False
                # strat 可能是策略实例
                snap_path = None
                if hasattr(strat, 'params') and getattr(getattr(strat, 'params'), 'snapshot_csv', None):
                    snap_path = strat.params.snapshot_csv

                # 如果策略仍持有打开的文件句柄，先 flush/close
                try:
                    if hasattr(strat, '_snapshot_file') and getattr(strat, '_snapshot_file') is not None:
                        try:
                            strat._snapshot_file.flush()
                        except Exception:
                            pass
                        try:
                            strat._snapshot_file.close()
                        except Exception:
                            pass
                except Exception:
                    pass

                # 若存在流式写入文件，复制到 out_dir（原子操作）
                if snap_path and os.path.exists(snap_path):
                    try:
                        shutil.copy(snap_path, os.path.join(out_dir, 'time_series_snapshot.csv'))
                        copied = True
                    except Exception:
                        copied = False

                # 如果没有可复制的流式文件，则回退到使用内存 rows 写入（字段包含更多列以兼容流式文件）
                if not copied:
                    rows = getattr(strat, 'tick_snapshot_rows', None)
                    if rows:
                        csv_path = os.path.join(out_dir, 'time_series_snapshot.csv')
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            fieldnames = ['timestamp', 'price', 'open', 'high', 'low', 'close', 'net_value', 'alpha', 'position_size', 'avg_price', 'decision_made', 'decision_type', 'quantity_pct', 'take_profit_price', 'stop_loss_price', 'similar_pattern_found']
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            for r in rows:
                                writer.writerow({
                                    'timestamp': r.get('timestamp',''),
                                    'price': r.get('price',''),
                                    'open': r.get('open',''),
                                    'high': r.get('high',''),
                                    'low': r.get('low',''),
                                    'close': r.get('close',''),
                                    'net_value': r.get('net_value',''),
                                    'alpha': r.get('alpha',''),
                                    'position_size': r.get('position_size',''),
                                    'avg_price': r.get('avg_price',''),
                                    'decision_made': bool(r.get('decision_made', False)),
                                    'decision_type': r.get('decision_type',''),
                                    'quantity_pct': r.get('quantity_pct',''),
                                    'take_profit_price': r.get('take_profit_price',''),
                                    'stop_loss_price': r.get('stop_loss_price',''),
                                    'similar_pattern_found': 'YES' if r.get('similar_pattern_found', False) else ('N/A' if r.get('similar_pattern_found') == 'N/A' else 'NO')  # 【新增】支持布尔值和字符串格式
                                })
            except Exception:
                pass

            th_path = os.path.join(repo_root, 'trade_history.csv')
            if os.path.exists(th_path):
                try:
                    shutil.copy(th_path, os.path.join(out_dir, 'trade_history.csv'))
                except Exception:
                    pass

            # 【新增】生成周期净值汇总报告（日、周、月）
            try:
                snap_path = None
                if hasattr(strat, 'params') and getattr(getattr(strat, 'params'), 'snapshot_csv', None):
                    snap_path = strat.params.snapshot_csv

                # 如果流式快照存在，从最新复制的文件读取
                if not snap_path:
                    snap_path = os.path.join(out_dir, 'time_series_snapshot.csv')

                if snap_path and os.path.exists(snap_path):
                    df = pd.read_csv(snap_path)

                    # 转换时间戳为 datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                        df = df.sort_index()

                        # 【日净值汇总】
                        daily_nav = df['net_value'].resample('D').agg(['first', 'last', 'min', 'max'])
                        daily_nav.columns = ['open', 'close', 'low', 'high']
                        daily_nav['daily_return_pct'] = ((daily_nav['close'] / daily_nav['open']) - 1) * 100
                        daily_nav.to_csv(os.path.join(out_dir, 'daily_nav.csv'))
                        print(f"[NAV_REPORT] 日净值汇总: {len(daily_nav)} 个交易日")

                        # 【周净值汇总】
                        weekly_nav = df['net_value'].resample('W').agg(['first', 'last', 'min', 'max'])
                        weekly_nav.columns = ['open', 'close', 'low', 'high']
                        weekly_nav['weekly_return_pct'] = ((weekly_nav['close'] / weekly_nav['open']) - 1) * 100
                        weekly_nav.to_csv(os.path.join(out_dir, 'weekly_nav.csv'))
                        print(f"[NAV_REPORT] 周净值汇总: {len(weekly_nav)} 周")

                        # 【月净值汇总】
                        monthly_nav = df['net_value'].resample('M').agg(['first', 'last', 'min', 'max'])
                        monthly_nav.columns = ['open', 'close', 'low', 'high']
                        monthly_nav['monthly_return_pct'] = ((monthly_nav['close'] / monthly_nav['open']) - 1) * 100
                        monthly_nav.to_csv(os.path.join(out_dir, 'monthly_nav.csv'))
                        print(f"[NAV_REPORT] 月净值汇总: {len(monthly_nav)} 月")

                        # 【汇总统计】生成一个总体统计报告
                        summary = {
                            'total_days': len(daily_nav),
                            'total_weeks': len(weekly_nav),
                            'total_months': len(monthly_nav),
                            'daily_avg_return': daily_nav['daily_return_pct'].mean(),
                            'daily_std_return': daily_nav['daily_return_pct'].std(),
                            'best_day': daily_nav['daily_return_pct'].max(),
                            'worst_day': daily_nav['daily_return_pct'].min(),
                            'weekly_avg_return': weekly_nav['weekly_return_pct'].mean(),
                            'best_week': weekly_nav['weekly_return_pct'].max(),
                            'worst_week': weekly_nav['weekly_return_pct'].min(),
                            'monthly_avg_return': monthly_nav['monthly_return_pct'].mean(),
                            'best_month': monthly_nav['monthly_return_pct'].max(),
                            'worst_month': monthly_nav['monthly_return_pct'].min(),
                        }

                        with open(os.path.join(out_dir, 'nav_summary.json'), 'w', encoding='utf-8') as f:
                            json.dump(summary, f, default=str, ensure_ascii=False, indent=2)

                        print(f"[NAV_REPORT] 统计完成:")
                        print(f"  - 日平均收益: {summary['daily_avg_return']:.4f}%")
                        print(f"  - 周平均收益: {summary['weekly_avg_return']:.4f}%")
                        print(f"  - 月平均收益: {summary['monthly_avg_return']:.4f}%")
            except Exception as e:
                print(f"[NAV_REPORT_ERROR] 净值汇总生成失败: {e}")

            if include_pickle:
                try:
                    import pickle
                    with open(os.path.join(out_dir, 'cerebro.pkl'), 'wb') as f:
                        pickle.dump(cerebro, f)
                    with open(os.path.join(out_dir, 'strat.pkl'), 'wb') as f:
                        pickle.dump(strat, f)
                except Exception as e:
                    with open(os.path.join(out_dir, 'pickle_error.txt'), 'w', encoding='utf-8') as f:
                        f.write(str(e))

            return out_dir
        except Exception as e:
            try:
                err_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'snapshots')
                os.makedirs(err_dir, exist_ok=True)
                with open(os.path.join(err_dir, 'last_error.txt'), 'w', encoding='utf-8') as f:
                    f.write(str(e))
            except Exception:
                pass
            return None

        return cerebro