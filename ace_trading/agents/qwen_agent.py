# ace_trading/agents/qwen_agent.py
import math
import time
import threading
import re
import sys
import json

from typing import Dict, Any, List, Optional
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from ace_trading.framework import BaseAgent, TradeDecision, StateComparisonResult,  LogicDefinition, AgentBOutput, ExecutionPlan
from ace_trading.config import OLLAMA_BASE_URL, MODEL_NAME,ZHIPU_API_KEY ,ZHIPU_URL,ZHIPU_MODEL, ENABLE_THINKING,QWEN_MODEL,QWEN_API,QWEN_URL
from ace_trading.agents.tools import calculate_support_resistance
from ace_trading.agents.output_schemas import AgentSOutput, MarketStateClassification, DeepMarketAnalysis, AgentAOutput as SchemaAgentAOutput, AgentBOutput as SchemaAgentBOutput  # 【新增】导入AgentBOutput结构化模型
from ace_trading.prompt_logger import get_prompt_logger  # 【新增】导入提示词日志
from ace_trading.token_logger import get_token_logger  # 【新增】导入Token日志
from ace_trading.engine import MarketSentimentFormatter  # 【新增】导入市场情绪格式化工具
import uuid  # 【新增】用于生成事件ID

class LoadingSpinner:
    """在控制台显示旋转动画，提示用户程序正在运行"""
    def __init__(self, message="Thinking"):
        self.message = message
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True # 设置为守护线程，防止主程序挂掉时它还在转

    def start(self):
        self.stop_event.clear()
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self._spin)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        # 清除行
        sys.stdout.write('\r' + ' ' * (len(self.message) + 20) + '\r')
        sys.stdout.flush()

    def _spin(self):
        chars = ['-', '\\', '|', '/']  # ASCII spinner instead of braille
        i = 0
        while not self.stop_event.is_set():
            # 打印 \r 使光标回到行首，实现覆盖效果
            try:
                sys.stdout.write(f'\r[*] {self.message} {chars[i % len(chars)]} ')
                sys.stdout.flush()
            except UnicodeEncodeError:
                # Skip printing on systems with encoding issues
                pass
            time.sleep(0.1)
            i += 1


class QwenAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=QWEN_MODEL,
            openai_api_key=QWEN_API,
            base_url=QWEN_URL,
            temperature=0.1,
            timeout=900,  # 【修复】从600秒改为900秒（15分钟），适配thinking模型的思考时间
            max_retries=2
        )
        
        # 【新增】针对Agent S的LLM实例，支持结构化输出
        self.llm_agent_s = ChatOpenAI(
            model=QWEN_MODEL,
            openai_api_key=QWEN_API,
            base_url=QWEN_URL,
            temperature=0.1,
            timeout=900,  # 【修复】从600秒改为900秒（15分钟），适配thinking模型的思考时间
            max_retries=2
        )
        
        # 【新增】针对Agent A的LLM实例，支持结构化输出
        self.llm_agent_a = ChatOpenAI(
            model=QWEN_MODEL,
            openai_api_key=QWEN_API,
            base_url=QWEN_URL,
            temperature=0.1,
            timeout=900,  # 【修复】从600秒改为900秒（15分钟），适配thinking模型的思考时间
            max_retries=2
        )
        
        # 【新增】针对Agent B的LLM实例，支持结构化输出
        self.llm_agent_b = ChatOpenAI(
            model=QWEN_MODEL,
            openai_api_key=QWEN_API,
            base_url=QWEN_URL,
            temperature=0.1,
            timeout=900,  # 【修复】从600秒改为900秒（15分钟），适配thinking模型的思考时间
            max_retries=2
        )

        # 【新增】历史缓冲区：保存Agent S、A、B的最近N期输出
        self.history_s = []      # 保存 AgentSOutput 对象列表
        self.history_a = []      # 保存 SchemaAgentAOutput 对象列表
        self.history_b = []      # 保存 SchemaAgentBOutput 对象列表
        self.history_m = []      # 【新增】保存最近36期月度数据的格式化字符串
        self.max_history = 5     # 保留最近5期历史
        self.max_monthly_history = 36  # 【新增】保留最近36期月度数据

        self.system_prompt = "你是一位专业的BTC比特币摇摆交易员，专门从事4小时周期的中期趋势交易。"

        # 针对4小时交易周期，重点关注持续的趋势和结构性突破
        self.tactical_instruction = """
【BTC 4小时周期交易原则】
1. **摇摆设置识别**：在几天内形成的多根K线确认的趋势方向
2. **支撑阻力结构**：使用日线和周线级别的摇摆高点/低点作为关键结构位
3. **趋势确认**：通过动量指标确认持续的超买或超卖状态
4. **成交量验证**：突破关键位置时需要有成交量的配合确认机构参与
5. **持仓规模**：维持适当规模的头寸，通常持有几小时到一两天的周期
6. **出场纪律**：在合理的止损位出场，目标收益相对于风险保持较高水平
7. **趋势对齐**：参考日线和周线的宏观趋势方向，与更大周期的趋势保持一致
8. **持仓原则**：除非出现明确的结构性突破或反转信号，否则默认持仓
9. **高置信度交易**：宁可错过机会也要追求高质量的交易信号，拒绝低置信度的机会
10. **风险管理**：根据BTC的日级别波动幅度来决定头寸规模，不被短期噪声所迷惑
        """

        # 【新增】Agent S 专用实例，支持结构化输出（Dict状态）
        self.agent_s = create_agent(
            model=self.llm,
            response_format=AgentSOutput
        )

        # 【新增】Agent A 专用实例，支持结构化输出（Pattern逻辑）
        self.agent_a = create_agent(
            model=self.llm,
            response_format=SchemaAgentAOutput
        )

        # 【改进】Agent B 专用实例，采用新的标准化 AgentBOutput 结构
        self.agent_b = create_agent(
            model=self.llm,
            response_format=SchemaAgentBOutput
        )

    def update_prompt(self, new_tactic: str):
        self.tactical_instruction = new_tactic

    # 【新增】=== 历史格式化函数 ===
    def _format_raw_history_s(self) -> str:
        """直接将历史Agent S输出转成字符串"""
        if not self.history_s:
            return ""

        text = "【Agent S 历史输出 - 最近N期参考】\n"
        for i, h in enumerate(self.history_s, 1):
            text += f"\n--- 第{i}期 ---\n"
            text += f"reasoning_trace: {h.reasoning_trace}\n"
            text += f"current_states: {h.current_states}\n"
            text += f"matched_states: {h.matched_states}\n"
            text += f"missing_states: {h.missing_states}\n"
            text += f"novel_states: {h.novel_states}\n"
        return text

    def _format_raw_history_a(self) -> str:
        """直接将历史Agent A输出转成字符串"""
        if not self.history_a:
            return ""

        text = "【Agent A 历史输出 - 最近N期参考】\n"
        for i, h in enumerate(self.history_a, 1):
            text += f"\n--- 第{i}期 ---\n"
            text += f"reasoning_trace: {h.reasoning_trace}\n"
            text += f"pattern_name: {h.pattern_name}\n"
            text += f"pattern_description: {h.pattern_description}\n"
            text += f"confidence: {h.confidence:.1%}\n"
            text += f"core_logic_states: {h.core_logic_states}\n"
        return text

    def _format_raw_history_b(self) -> str:
        """直接将历史Agent B输出转成字符串，包含所有字段"""
        if not self.history_b:
            return ""

        text = "【Agent B 历史决策 - 最近N期参考】\n"
        for i, h in enumerate(self.history_b, 1):
            text += f"\n--- 第{i}期 ---\n"
            text += f"action: {h.action}\n"
            text += f"quantity_pct: {h.quantity_pct:.1%}\n"
            text += f"take_profit_price: {h.take_profit_price:.2f}\n"
            text += f"stop_loss_price: {h.stop_loss_price:.2f}\n"
            text += f"limit_price: {h.limit_price:.2f}\n"
            text += f"confidence: {h.confidence:.1%}\n"
            text += f"risk_reward_ratio: {h.risk_reward_ratio:.2f}\n"
            if hasattr(h, 'reasoning') and h.reasoning:
                # 截取前200字符的reasoning，避免过长
                reasoning_preview = h.reasoning[:200] + "..." if len(h.reasoning) > 200 else h.reasoning
                text += f"reasoning: {reasoning_preview}\n"
            if hasattr(h, 'position_adjustment_reason') and h.position_adjustment_reason:
                text += f"position_adjustment_reason: {h.position_adjustment_reason}\n"
        return text

    def _format_monthly_history(self, mtf_data: dict) -> str:
        """【新增】使用现有的 _format_tf_data() 方法处理月度数据"""
        if '1M' not in mtf_data:
            return ""

        try:
            # 使用现有方法，自动计算RSI、MACD等指标
            monthly_formatted = self._format_tf_data('1M', mtf_data['1M'], real_price=mtf_data['1M'].get('cur_price', 0), limit=36)

            # 存储到历史缓冲区
            self.history_m.append(monthly_formatted)
            if len(self.history_m) > self.max_monthly_history:
                self.history_m.pop(0)

            return f"【月度数据 - 用于长期趋势识别】\n{monthly_formatted}"
        except Exception as e:
            print(f"[MONTHLY_FORMAT_ERROR] {e}")
            return ""

    # VECTOR DB & GRAPH DB INTEGRATION METHODS
    # ============================================================================

    def _format_graph_insight(self, insight: Dict[str, Any]) -> str:
        """Format graph DB output for agent consumption"""
        if not insight or 'dominant_pattern' not in insight or insight['dominant_pattern'] is None:
            return "No historical pattern data available"

        dominant = insight.get('dominant_pattern', {})
        attribution = insight.get('attribution_analysis', {})

        # Format dominant pattern
        pattern_str = f"""[HISTORICAL PATTERN MATCH]
Pattern Name: {dominant.get('name', 'Unknown')}
Pattern Definition: {', '.join(dominant.get('definition', []))}
Historical Win Rate: {dominant.get('baseline_win_rate', 0):.1%}
Similar Trade Count: {dominant.get('match_count', 0)}"""

        # Format success drivers
        success_drivers = attribution.get('success_drivers', [])
        if success_drivers:
            success_str = "\n[SUCCESS DRIVERS] (Present in these conditions, win rate increases):\n"
            for driver in success_drivers:
                success_str += f"  - {driver['state']}: {driver['win_rate']:.1%} win rate (appears {driver['freq']} times) | {driver['insight']}\n"
        else:
            success_str = "\n[SUCCESS DRIVERS] None identified"

        # Format failure drivers
        failure_drivers = attribution.get('failure_drivers', [])
        if failure_drivers:
            failure_str = "\n[FAILURE DRIVERS] (Avoid these conditions, high lose rate):\n"
            for driver in failure_drivers:
                failure_str += f"  - {driver['state']}: {driver['win_rate']:.1%} win rate (appears {driver['freq']} times) | {driver['insight']}\n"
        else:
            failure_str = "\n[FAILURE DRIVERS] None identified"

        # Format common context
        common_context = attribution.get('common_context', [])
        if common_context:
            context_str = "\n[COMMON CONTEXT] (Present in most similar trades):\n"
            for ctx in common_context:
                context_str += f"  - {ctx['state']}: {ctx['prevalence']:.1%} prevalence | {ctx['insight']}\n"
        else:
            context_str = "\n[COMMON CONTEXT] None identified"

        return pattern_str + success_str + failure_str + context_str

    def _agent_s_analyze_with_graph(self, market_context: str, graph_insight: Dict[str, Any], state: Dict[str, Any]) -> AgentSOutput:
        """
        【新增】Agent S：融合市场数据与图数据库信息的完整分析
        
        输入：
        - market_context: 当前市场数据的格式化字符串
        - graph_insight: 来自quant_graph_manager的相似事件分析结果（含dominant_pattern & attribution_analysis）
        - state: 完整的agent_state
        
        输出：
        AgentSOutput - 包含完整思维链和Dict格式的四类状态（current/matched/missing/novel）
        """
        try:
            # 提取账户信息作为激励机制
            account_info = state.get('account_info', {})
            account_pnl = account_info.get('unrealized_pnl', 0)
            account_pnl_pct = account_info.get('unrealized_pnl_pct', 0)
            position_info = account_info.get('position_info', {})
            
            account_context = f"""
"""

            # 提取市场情绪信息
            cur_price = state.get('cur_price', 0)
            market_emotion_score = state.get('market_emotion_score', 0)
            market_emotion_label = state.get('market_emotion_label', 'Neutral')
            
            dt_str = state.get('dt_str', 0)
            sentiment_prob_dict = state.get('sentiment_prob_dict', {})
            market_emotion_str = f"{market_emotion_label} (Score: {market_emotion_score:.2f})"
            sentiment_prob_str = ", ".join([f"{k}: {v:.1%}" for k, v in sentiment_prob_dict.items()])

            current_market_summary = f"""【当前市场状态 - 结构化摘要】
Time:{dt_str}
Price: {cur_price:.2f}
Market Emotion: {market_emotion_str}
Sentiment Probability: {sentiment_prob_str}
"""

            # 构建图数据库信息的完整提示词部分
            graph_prompt_part = ""
            if graph_insight and 'dominant_pattern' in graph_insight and graph_insight['dominant_pattern'] is not None:
                dominant = graph_insight['dominant_pattern']
                pattern_definition = dominant.get('definition', [])
                
                graph_prompt_part = f"""【历史Pattern信息 - 来自图数据库】
Pattern名称: {dominant.get('name', 'Unknown')}
Pattern定义 (核心状态): {', '.join(pattern_definition)}
历史胜率: {dominant.get('baseline_win_rate', 0):.1%}
相似交易数: {dominant.get('match_count', 0)}

"""
                
                # 添加归因分析信息
                attribution = graph_insight.get('attribution_analysis', {})
                
                common_context = attribution.get('common_context', [])
                if common_context:
                    graph_prompt_part += "【背景共识 - 相似交易普遍存在的状态】\n"
                    for ctx in common_context:
                        graph_prompt_part += f"  - {ctx['state']}: {ctx['prevalence']:.1%}普及度 | {ctx['insight']}\n"
                    graph_prompt_part += "\n"
                
                success_drivers = attribution.get('success_drivers', [])
                if success_drivers:
                    graph_prompt_part += "【成功驱动因子 - Alpha】\n"
                    for driver in success_drivers:
                        graph_prompt_part += f"  - {driver['state']}: {driver['win_rate']:.1%}胜率 | 出现{driver['freq']}次 | {driver['insight']}\n"
                    graph_prompt_part += "\n"
                
                failure_drivers = attribution.get('failure_drivers', [])
                if failure_drivers:
                    graph_prompt_part += "【致死因子 - Risk】\n"
                    for driver in failure_drivers:
                        graph_prompt_part += f"  - {driver['state']}: {driver['win_rate']:.1%}胜率 | {driver['insight']}\n"
                    graph_prompt_part += "\n"
            else:
                graph_prompt_part = """【注意】当前没有历史Pattern数据可参考。
请只分析当前市场状态，将所有识别的状态都放入 current_states，同时将其完整复制到 novel_states。
"""

            # 构建Agent S的完整提示词
            history_s_str = "无" #停用短期记忆memory
            # history_s_str = self._format_raw_history_s()
            agent_s_llm_prompt = f"""{account_context}

{current_market_summary}

【市场数据详情 -】
{market_context}

{graph_prompt_part}

{history_s_str}

【历史参考提示】
请参考上述历史输出信息，确保你的分析具有连续性和一致性。避免与之前的分析发生不必要的大幅反转。
如果发现与历史显著不同的信号，请在推理中详细解释原因。

【Agent S 任务】
你是BTC多周期市场状态分析专家 Agent S。请基于上述所有信息，进行多时间周期的市场状态分析：

1. 识别不同时间周期（15分钟、1小时、4小时、日线、周线）的所有关键状态（趋势结构、支撑阻力、超买超卖、成交量确认等）
   - 分别分析各个周期的独立状态
   - 特别注意不同周期之间的共鸣和冲突信号


2. 如果有历史Pattern数据：
   - 判断当前哪些状态与历史Pattern定义一致（matched_states）
   - 判断历史Pattern需要但当前缺失的状态（missing_states）
   - 判断当前独有的新状态，这些可能是新机会或风险（novel_states）
3. 如果没有历史Pattern数据：
   - 将识别的所有当前状态放入 current_states 和 novel_states

4. 当前的情况与历史虽然state一样，但是是否存在新的机会，例如新高，新低，同时不要因为历史胜率100%或0%就盲目死守历史准则，应该做到辩证判别
【多周期状态分析参考】
周期特性指导：
- **15分钟周期**：短期噪声过滤、精确入场点位确认（MA5/MA20快线交叉、RSI微观极值等）
- **1小时周期**：短期趋势确认、早期信号识别（突破日内支撑/阻力、小级别结构）
- **4小时周期**（核心）：主要操作周期的趋势、支撑/阻力、形态完成、成交量确认
- **日线周期**：中期趋势方向、关键结构位（突破日线支撑/阻力是重大信号）、宏观情绪
- **周线周期**：长期趋势确认、重大阻力位（周线突破往往启动中期行情）、市场结构、当周级支撑被突破意味着趋势反转

【关键状态参考示例】
可以包含但不限于以下状态（按周期+类型组织）：
- **趋势类**：Trend_Uptrend_4H, Trend_Downtrend_1H, Trend_Consolidation_Daily, etc.
- **结构类**：Structure_DailySupport, Structure_WeeklyResistance, Structure_Breakout_4H, etc.
- **动量类**：RSI_Overbought_1H, RSI_Oversold_4H, MACD_Bullish_Daily, Momentum_Extreme_Hourly, etc.
- **成交量**：Volume_Spike_4H, Volume_Confirmation_Daily, Volume_Declining_1H, etc.
- **价格位置**：Price_NearSupport_4H, Price_AtResistance_Daily, Price_Extreme_Weekly, etc.
- **周期共鸣**：MTF_Alignment_Bullish (多周期同向), MTF_Conflict_Bearish (周期背离), etc.

【牛市/熊市/猴市状态范例 - 按周期识别市场行情】
以下状态代表在某个具体周期上呈现的行情特征：
- **牛市状态**（连续新高或下一个高点反弹后新低不跌破低于前一个高点）：1m_Bull, 15m_Bull, 1h_Bull, 4h_Bull, 1d_Bull, 1w_Bull, 1M_Bull
- **熊市状态**（连续新低或下一个低点反弹后不创新高超过前一个低点）：1m_Bear, 15m_Bear, 1h_Bear, 4h_Bear, 1d_Bear, 1w_Bear, 1M_Bear
- **猴市状态**（区间震荡，高低点无新突破）：1m_Monkey, 15m_Monkey, 1h_Monkey, 4h_Monkey, 1d_Monkey, 1w_Monkey, 1M_Monkey
- **无行情** （高低点无突破，无大波动）： 1d_Peace, 4h_Peace

【状态名称格式规范】
状态名称必须遵守以下格式（严格约束）：
- 【格式】：大写字母开头，后续跟字母/数字/下划线，总长度≥2
- 【示例】：RSI_Overbought_4H, Volume_Spike_Daily, Trend_Uptrend_1H, MTF_Alignment_Bullish, Price_NearSupport_4H
- 【反例】：rsi overbought（小写）, RSI OVERBOUGHT（空格）, R（太短）
- 【建议】：使用 类别_描述_周期 的模式，如 RSI_*_4H, Volume_*_Daily, Price_*_1H, Trend_*_Weekly 等

返回以下JSON格式（必须有效的Python可解析JSON）：
{{
    "reasoning_trace": "【思维链】详细解释你的分析过程。为什么识别某个状态，与历史如何对比，新状态的风险或机会",
    "current_states": {{"状态名": "判定理由和数据证据", ...}},
    "matched_states": {{"状态名": "与历史Pattern的匹配程度说明", ...}},
    "missing_states": {{"状态名": "缺失的具体证据", ...}},
    "novel_states": {{"状态名": "这是新风险还是机会，为什么值得注意", ...}}
}}

【重要提示】
- 状态名必须严格遵守格式规范，否则会被拒绝！
- 每个状态的值（理由）应该具体、有数据支撑，不要模糊
- 思维链应该是完整的推理过程，包括如何比对历史Pattern
- 返回的JSON必须有效，所有字符串中的引号要正确转义 """

            # 调用LLM进行结构化分析
            spinner = LoadingSpinner(message="Agent S analyzing with graph insights...")
            spinner.start()

            # 【修复】直接调用 LLM，不经过 agent 框架，避免无限循环
            # 使用 with_structured_output 来强制返回 AgentSOutput 格式
            result = self.llm_agent_s.with_structured_output(AgentSOutput).invoke(agent_s_llm_prompt)

            spinner.stop()

            # 【新增】记录token使用（从文本估算）
            token_logger = get_token_logger()
            if token_logger:
                try:
                    # 估算输出token（从结构化结果）
                    if isinstance(result, dict) and 'structured_response' in result:
                        output_obj = result['structured_response']
                    else:
                        output_obj = result

                    output_text = str(output_obj) if output_obj else ""

                    token_logger.log_from_prompt_and_output(
                        agent_name="S",
                        prompt_text=agent_s_llm_prompt,
                        output_text=output_text,
                        bar_number=getattr(self, '_bar_count', None),
                        model=QWEN_MODEL
                    )
                except Exception as e:
                    print(f"[TOKEN_LOG_ERROR] Failed to log Agent S tokens: {e}")

            # 【新增】立即记录Agent S的真实输入提示词和完整输出对象（在方法内部）
            try:
                from ace_trading.prompt_logger import get_prompt_logger
                from datetime import datetime
                prompt_logger = get_prompt_logger()
                if prompt_logger:
                    timestamp = datetime.now().isoformat()

                    # 解析Agent S的结构化输出，以便提前获取信息用于日志
                    if isinstance(result, dict) and 'structured_response' in result:
                        temp_output = result['structured_response']
                    else:
                        temp_output = result

                    # 构建输出摘要（用于summary日志）并计数
                    output_summary = ""
                    current_count = 0
                    matched_count = 0
                    missing_count = 0
                    novel_count = 0

                    if isinstance(temp_output, dict):
                        current_count = len(temp_output.get('current_states', {}))
                        matched_count = len(temp_output.get('matched_states', {}))
                        missing_count = len(temp_output.get('missing_states', {}))
                        novel_count = len(temp_output.get('novel_states', {}))
                        output_summary = f"States found: current={current_count}, matched={matched_count}, missing={missing_count}, novel={novel_count}"
                    elif hasattr(temp_output, 'current_states'):
                        current_count = len(temp_output.current_states) if isinstance(temp_output.current_states, dict) else 0
                        matched_count = len(temp_output.matched_states) if isinstance(temp_output.matched_states, dict) else 0
                        missing_count = len(temp_output.missing_states) if isinstance(temp_output.missing_states, dict) else 0
                        novel_count = len(temp_output.novel_states) if isinstance(temp_output.novel_states, dict) else 0
                        output_summary = f"States found: current={current_count}, matched={matched_count}, missing={missing_count}, novel={novel_count}"

                    prompt_logger.log_agent_prompt(
                        agent_name="S",
                        bar_number=getattr(self, '_bar_count', 0),
                        timestamp=timestamp,
                        prompt_text=agent_s_llm_prompt,  # 【修复】记录真实的LLM输入提示词
                        additional_info={
                            "cur_price": f"{cur_price:.2f}",
                            "position": f"{state.get('position', 0):.4f}",
                            "cash": f"{state.get('cash', 0):.2f}",
                            "analysis_method": "unified_graph_integration",
                            "graph_available": graph_insight is not None,
                            "prompt_length": len(agent_s_llm_prompt)
                        },
                        output_text=output_summary,  # 保留摘要用于快速查看
                        output_info={
                            "output_type": "AgentSOutput with Dict states",
                            "states_summary": output_summary,
                            "current_states_count": current_count,
                            "matched_states_count": matched_count,
                            "missing_states_count": missing_count,
                            "novel_states_count": novel_count
                        },
                        output_object=temp_output  # 【关键改进】直接传入完整的AgentSOutput对象，将被序列化为JSON
                    )
            except Exception as log_err:
                print(f"[PROMPT_LOGGER_ERROR] Failed to log Agent S prompt: {log_err}")
            if isinstance(result, dict) and 'structured_response' in result:
                agent_s_output = result['structured_response']
            else:
                agent_s_output = result

            # 确保输出是AgentSOutput对象
            if isinstance(agent_s_output, AgentSOutput):
                # 【新增】记录到历史缓冲区
                self.history_s.append(agent_s_output)
                if len(self.history_s) > self.max_history:
                    self.history_s.pop(0)
                return agent_s_output
            else:
                # 降级处理：如果返回的不是AgentSOutput，则手动构造
                print(f"[AGENT_S_WARNING] 返回的不是AgentSOutput对象，类型: {type(agent_s_output)}")
                return AgentSOutput(
                    reasoning_trace="Agent S分析完成（格式转换）",
                    current_states={},
                    matched_states={},
                    missing_states={},
                    novel_states={}
                )

        except Exception as e:
            print(f"[AGENT_S_ANALYZE_ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return AgentSOutput(
                reasoning_trace=f"Agent S分析失败: {str(e)}",
                current_states={},
                matched_states={},
                missing_states={},
                novel_states={}
            )


    def _agent_a_analyze_with_structure(self, agent_a_prompt: str) -> SchemaAgentAOutput:
        """
        【新增】Agent A：深度分析与结构化输出

        输入：agent_a_prompt - 完整的Agent A提示词（包含市场数据、Agent S输出、历史模式等）
        输出：SchemaAgentAOutput - 结构化的深度分析结果（思维链 + 核心逻辑状态 + Pattern名称 + 置信度）

        关键特性：
        1. 保持原有的提示词结构完全不变
        2. 只改变输出的结构化方式（从纯文本 → 结构化对象）
        3. 使用 LangChain 的 response_format 约束（类似Agent S）
        """
        try:
            # 调用LLM进行结构化分析
            spinner = LoadingSpinner(message="Agent A analyzing with structure...")
            spinner.start()

            # 【修复】直接调用 LLM，不经过 agent 框架
            res, err = self._invoke_with_timeout(
                self.llm_agent_a.with_structured_output(SchemaAgentAOutput).invoke,
                (agent_a_prompt,),
                300
            )

            spinner.stop()

            # 【新增】记录token使用
            token_logger = get_token_logger()
            if token_logger and res:
                try:
                    output_text = str(res) if res else ""
                    token_logger.log_from_prompt_and_output(
                        agent_name="A",
                        prompt_text=agent_a_prompt,
                        output_text=output_text,
                        bar_number=getattr(self, '_bar_count', None),
                        model=QWEN_MODEL
                    )
                except Exception as e:
                    print(f"[TOKEN_LOG_ERROR] Failed to log Agent A tokens: {e}")

            if err:
                raise err

            # 解析Agent A的结构化输出
            if isinstance(res, dict) and 'structured_response' in res:
                agent_a_output = res['structured_response']
            else:
                agent_a_output = res

            # 确保输出是SchemaAgentAOutput对象
            if isinstance(agent_a_output, SchemaAgentAOutput):
                # 【新增】记录到历史缓冲区
                self.history_a.append(agent_a_output)
                if len(self.history_a) > self.max_history:
                    self.history_a.pop(0)
                return agent_a_output
            else:
                # 降级处理：如果返回的不是SchemaAgentAOutput，则手动构造
                print(f"[AGENT_A_WARNING] 返回的不是SchemaAgentAOutput对象，类型: {type(agent_a_output)}")
                return SchemaAgentAOutput(
                    reasoning_trace="Agent A分析完成（格式转换）",
                    core_dna_states=["Generic_State"],  # 至少需要1个核心DNA
                    context_states=[],
                    pattern_name="Pattern_Generic_Analysis",
                    pattern_description="通用深度分析",
                    confidence=0.5
                )

        except Exception as e:
            print(f"[AGENT_A_ANALYZE_ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return SchemaAgentAOutput(
                reasoning_trace=f"Agent A分析失败: {str(e)}",
                core_dna_states=["Error_State"],  # 至少需要1个核心DNA
                context_states=[],
                pattern_name="Pattern_Analysis_Failed",
                pattern_description="分析失败",
                confidence=0.0
            )


    # ============================================================================
    # 【新增】Agent I/O Protocol Methods - 生成完整的Agent输出包（包含思维链）
    # ============================================================================


    # EXISTING HELPER METHODS (unchanged)
    # ============================================================================

    def _extract_states_from_text(self, text: str) -> List[str]:
        """
        从 LLM 的自由文本分析中提取市场状态标签
        用于当 LLM 未按 JSON 格式返回时的 fallback 处理
        """
        import re
        states = []

        # 关键词匹配规则 - 根据 LLM 常见的表述提取状态
        keyword_patterns = {
            '多头': ['多头', '上涨', '看涨', '强势上行', '破位向上', '持续上升'],
            '空头': ['空头', '下跌', '看跌', '弱势下行', '回调', '持续下降'],
            'RSI_Overbought': ['RSI.*超买', 'RSI.*>80', 'RSI.*>70'],
            'RSI_Oversold': ['RSI.*超卖', 'RSI.*<20', 'RSI.*<30'],
            'MACD_Bullish': ['MACD.*多头', 'MACD.*柱状.*放大', 'DIF.*DEA'],
            'MACD_Bearish': ['MACD.*空头', 'MACD.*柱状.*缩小', 'MACD.*负'],
            'Price_High': ['高位', '近高点', '阻力', '突破阻力'],
            'Price_Low': ['低位', '支撑', '底部', '近低点'],
            'Consolidation': ['震荡', '盘整', '横盘', '整理'],
            'Divergence': ['背离', '乖离'],
            'Momentum_Strong': ['动能.*强', '强劲', '充足'],
            'Momentum_Weak': ['动能.*弱', '乏力', '枯竭'],
        }

        for state_label, patterns in keyword_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if state_label not in states:
                        states.append(state_label)
                    break

        # 如果没有提取到任何状态，至少返回基础状态
        if not states:
            if '多头' in text or '上涨' in text or '看涨' in text:
                states.append('多头')
            elif '空头' in text or '下跌' in text or '看跌' in text:
                states.append('空头')
            else:
                states.append('Neutral')

        return states

    def _extract_text_response(self, response: Any) -> str:
        """
        【关键方法】从LangChain复杂响应对象中提取纯文本内容
        避免将整个对话链、元数据等污染信息传入下一个Agent
        """
        # 情况1：字典响应
        if isinstance(response, dict):
            # 优先级顺序：structured_response → content → text → message
            if 'structured_response' in response:
                obj = response['structured_response']
                if hasattr(obj, 'reasoning'):
                    return str(obj.reasoning)
                else:
                    return str(obj)
            
            # 尝试从content/text/message字段提取
            for key in ['content', 'text', 'message']:
                if key in response and response[key]:
                    return str(response[key]).strip()
        
        # 情况2：对象有messages属性（LangChain AIMessage等）
        if hasattr(response, 'messages'):
            # 提取最后一条AIMessage的content
            for msg in reversed(response.messages):
                if hasattr(msg, 'content') and msg.content:
                    return str(msg.content).strip()
        
        # 情况3：对象有content属性（AIMessage）
        if hasattr(response, 'content'):
            return str(response.content).strip()
        
        # 情况4：对象有tool_calls属性（LangChain structured response）
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                if hasattr(tool_call, 'args') and isinstance(tool_call.args, dict):
                    # 对于TradeDecision等，提取reasoning字段
                    if 'reasoning' in tool_call.args:
                        return str(tool_call.args['reasoning']).strip()
        
        # 情况5：直接的字符串
        if isinstance(response, str):
            return response.strip()
        
        # 默认：仅返回简短的类型说明，不序列化整个对象
        return f"[Unable to extract text from response type: {type(response).__name__}]"

    def _invoke_with_timeout(self, fn, args=(), timeout=60):
        """类方法版本的 invoke watchdog，供 decide/reflect 复用。"""
        container = {}

        def _target():
            try:
                container['res'] = fn(*args)
            except Exception as e:
                container['exc'] = e

        th = threading.Thread(target=_target)
        th.daemon = True
        th.start()
        th.join(timeout)
        if th.is_alive():
            return None, TimeoutError(f"invoke timeout after {timeout}s")
        if 'exc' in container:
            return None, container['exc']
        return container.get('res'), None

    def _safe_format(self, val):
        """安全格式化，处理 NaN"""
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return "N/A"
        return f"{val:.2f}"

    def _format_seq(self, data_list: list, limit=5) -> str:
        """辅助函数：将列表转为箭头序列 50->52->55"""
        # 取最近 limit 个数据
        seq = data_list[-limit:]
        return "->".join([self._safe_format(x) for x in seq])

    def _format_tf_data(self, tf_name, data,real_price, limit=5,):
        """
        格式化单个周期的数据，支持多指标序列
        """
        # 1. 价格序列
        prices = data['prices'][-limit:]
        price_str = "->".join([f"{p:.0f}" for p in prices])
        cur_price = real_price

        # --- 【新增】提取 High/Low 数据 ---
        hh = data.get('highest_high', cur_price)
        ll = data.get('lowest_low', cur_price)
        win_len = data.get('window_len', 20)
        
        # 计算当前价格在区间内的相对位置 (0% = 最低点, 100% = 最高点)
        if hh != ll:
            pos_in_range = (cur_price - ll) / (hh - ll)
        else:
            pos_in_range = 0.5 # 极值相等说明无波动
            
        # 格式化区间字符串，例如: "[36000 ~ 38000] (位置: 80%)"
        range_str = f"近{win_len}K线范围:[{ll:.0f}~{hh:.0f}] (位置:{pos_in_range:.0%})"

        # --- 【新增】提取成交量数据 ---
        volumes = data.get('volumes', [])[-limit:]
        volume_str = "->".join([f"{v:.0f}" for v in volumes]) if len(volumes) > 0 else "N/A"
        cur_volume = data.get('cur_volume', 0)

        lines = []
        lines.append(f"== [{tf_name.upper()}] 现价:{cur_price:.0f} | {range_str} | 成交量:{cur_volume:.0f} ==")
        lines.append(f"  • 价格走势: {price_str}")
        lines.append(f"  • 成交量走势: {volume_str}")
        
        # 2. 指标序列处理
        # 这里的 data['indicators'] 是一个字典: {'macd': {'histo': [...], ...}, 'rsi_6': [...]}
        indicators = data['indicators']
        
        # 分组处理 RSI
        rsi_group = []
        for k in ['rsi_6', 'rsi_12', 'rsi_24']:
            if k in indicators:
                # indicators[k] 是一个 list
                seq_str = self._format_seq(indicators[k], limit)
                rsi_group.append(f"{k.upper()}:[{seq_str}]")
        if rsi_group:
            lines.append(f"  • RSI组: " + " | ".join(rsi_group))

        # 处理 MACD (包含 DIF, DEA, Histo)
        if 'macd' in indicators:
            macd_data = indicators['macd']
            # macd_data 是 dict: {'macd':[], 'signal':[], 'histo':[]}
            dif_seq = self._format_seq(macd_data.get('macd', []), limit)
            dea_seq = self._format_seq(macd_data.get('signal', []), limit)
            hist_seq = self._format_seq(macd_data.get('histo', []), limit)
            
            lines.append(f"  • MACD(12,26,9):")
            lines.append(f"    - DIF(快): [{dif_seq}]")
            lines.append(f"    - DEA(慢): [{dea_seq}]")
            lines.append(f"    - 柱状(动能): [{hist_seq}]")

        return "\n".join(lines)

    def decide(self, state: Dict[str, Any]) -> SchemaAgentBOutput:
        """
        Four-Stage Agent Cascade for 1-Minute Trading:
        1. Agent S - State Classification (match current market against known patterns)
        2. Agent A - Deep Analysis (unstructured thinking on probabilities)
        3. Agent B - Risk Assessment & Structured Decision (AgentBOutput output)
        4. Agent C - Reflection (if Agent B rejects or trade completes)
        """
        import json

        mtf_data = state.get('mtf_data', {})
        graph_insight = state.get('graph_insight')
        cur_price = state.get('price') or state.get('cur_price', 0)
        dt_str=state.get("dt_str","next_time")
        # ====================================================================
        # FORMAT MARKET DATA FOR 1-MINUTE TRADING
        # ====================================================================


        # Build 1-minute specific market context (focus on recent 5 bars)
        context_blocks = []
        for tf in ['1m']:  # Primaary focus on 1m
            if tf in mtf_data:
                context_blocks.append(self._format_tf_data(
                    tf, mtf_data[tf], real_price=cur_price, limit=15
                ))

        # Add macro confirmation (15m/1h for trend direction)
        for tf in ['15m', '1h', '4h', '1d',"1w"]:
            if tf in mtf_data:
                context_blocks.append(self._format_tf_data(
                    tf, mtf_data[tf], real_price=cur_price, limit=30
                ))

        # 【修改】Create full_market_context_b WITHOUT 1M data (for Agent B)
        full_market_context_b = "\n\n".join(context_blocks)

        # 【新增】Add monthly data for long-term analysis (36 periods) - ONLY for Agent A
        if '1M' in mtf_data:
            context_blocks.append(self._format_tf_data(
                '1M', mtf_data['1M'], real_price=cur_price, limit=36
            ))

        # 【修改】Create full_market_context WITH 1M data (for Agent A)
        full_market_context = "\n\n".join(context_blocks)

        # ====================================================================
        # 优先传递永续合约当期值（减少令牌与空列表问题），若不存在再回退到历史序列
        # ====================================================================
        market_sentiment_str = ""
        try:
            fr_cur = state.get('funding_rate_current', None)
            fr_delta = state.get('funding_rate_delta', None)
            oi_cur = state.get('oi_usd_current', None)
            oi_change = state.get('oi_usd_change_pct', None)
            oi_contracts_cur = state.get('oi_contracts_current', None)

            if fr_cur is not None or oi_cur is not None:
                # 使用简洁版格式，仅传当期值与变化率
                market_sentiment_str = MarketSentimentFormatter.format_market_sentiment_current(
                    funding_rate_current=fr_cur,
                    funding_rate_delta=fr_delta,
                    oi_usd_current=oi_cur,
                    oi_usd_change_pct=oi_change,
                    oi_contracts_current=oi_contracts_cur
                )
            else:
                # 回退：使用历史序列展示（旧逻辑）
                funding_rate_list = state.get('funding_rate_list', [])
                oi_usd_list = state.get('oi_usd_list', [])
                oi_contracts_list = state.get('oi_contracts_list', [])
                if funding_rate_list or oi_usd_list:
                    market_sentiment_str = MarketSentimentFormatter.format_market_sentiment(
                        funding_rate_list,
                        oi_usd_list,
                        oi_contracts_list
                    )
        except Exception as e:
            print(f"[WARN] Failed to format market sentiment: {e}")
            market_sentiment_str = ""

        # 将市场情绪数据添加到完整上下文
        if market_sentiment_str:
            full_market_context = full_market_context + "\n\n" + market_sentiment_str

        # Extract current position info
        pos_str = f"{state['position']:.4f}"
        cash_str = f"{state['cash']:.0f}"
        current_pos_pct = 0.0
        current_value = state['position'] * cur_price
        total_equity = state['cash'] + current_value
        if total_equity > 0:
            current_pos_pct = current_value / total_equity

        # ====================================================================
        # 【新增】格式化账户状态和操作历史
        # ====================================================================
        
        # 账户状态块
        plan_str = f"TP:{state['current_tp']:.0f} SL:{state['current_sl']:.0f}" if state['current_tp'] > 0 else "未设置"
        account_status_str = f"""【账户状态】真实账户状态
        目前的时间是:{dt_str}
        持仓: {state['position']:.4f} | 现金: {state['cash']:.0f}
        当前持仓占比: {current_pos_pct:.1%}  <--- (注意：这是你目前的仓位水平)
        持仓计划: {plan_str} | 现价: {cur_price:.2f}
        盈亏: {state['roi_status']} | 成本: {state['avg_price']:.0f}"""
        
        # 操作历史块
        history_lines = state.get('history', [])
        if history_lines:
            history_str = "\n        ".join(history_lines[-50:])  # 取最近5条
            operation_history_str = f"""【最近操作历史】
        {history_str}"""
        else:
            operation_history_str = """【最近操作历史】
        暂无交易记录"""
        
        # 分析要求说明
        analysis_note_str = """【分析要求】 注意，分析要求中的任何有关账户信息的数据仅作参考，实际数据以【账户状态】模块为准"""

        # ====================================================================
        # 【改进】STAGE 0-PROTOCOL: Agent S + 图数据库融合分析
        # ====================================================================
        # 【新增】使用新的 _agent_s_analyze_with_graph 方法，融合图数据库和市场数据
        print(f"[AGENT_S_WITH_GRAPH] Starting Agent S analysis with graph insights...")
        agent_s_output_obj: Optional[AgentSOutput] = None
        try:
            agent_s_output_obj = self._agent_s_analyze_with_graph(
                market_context=full_market_context,
                graph_insight=graph_insight if isinstance(graph_insight, dict) else {},
                state=state
            )
            if agent_s_output_obj:
                num_current = len(agent_s_output_obj.current_states) if isinstance(agent_s_output_obj.current_states, dict) else 0
                num_matched = len(agent_s_output_obj.matched_states) if isinstance(agent_s_output_obj.matched_states, dict) else 0
                print(f"[AGENT_S_WITH_GRAPH] Complete: current={num_current}, matched={num_matched}")
        except Exception as e:
            print(f"[AGENT_S_WITH_GRAPH_ERROR] {e}")
            import traceback
            traceback.print_exc()
            agent_s_output_obj = None

        # 【新增】保存 Agent S 输出到实例变量，供 engine 提取
        self._agent_s_output_obj = agent_s_output_obj
        print(f"[AGENT_OUTPUT_SAVED] Saved Agent S output for TradeContext: {agent_s_output_obj is not None}")

        # 为TradeContext生成唯一事件ID
        event_id = f"trade_{str(uuid.uuid4())[:8]}_{int(time.time() * 1000) % 10000}"

        # 【新增】记录Agent S的提示词和输出用于调试
        prompt_logger = get_prompt_logger()
        if prompt_logger and agent_s_output_obj:
            from datetime import datetime
            timestamp = datetime.now().isoformat()

            # 【改进】包含完整的 Graph DB 查询信息
            graph_insight_summary = "No graph data"
            if graph_insight and graph_insight.get('dominant_pattern'):
                dominant = graph_insight.get('dominant_pattern', {})
                attribution = graph_insight.get('attribution_analysis', {})

                graph_insight_summary = f"""[Dominant Pattern] {dominant.get('name', 'Unknown')} (Win Rate: {dominant.get('baseline_win_rate', 0):.1%})
[Success Drivers] {len(attribution.get('success_drivers', []))} identified
[Failure Drivers] {len(attribution.get('failure_drivers', []))} identified
[Common Context] {len(attribution.get('common_context', []))} identified"""

            # 记录新的Agent S提示词（从_agent_s_analyze_with_graph方法中提取关键信息）
            agent_s_prompt_summary = f"""
Agent S Unified Analysis with Graph Insights
Graph Insight Available: {graph_insight is not None}
{graph_insight_summary}
Market Context Length: {len(full_market_context)}
Current States in Result: {len(agent_s_output_obj.current_states) if isinstance(agent_s_output_obj.current_states, dict) else 0}
"""

            # 【改进】构建完整的结构化输出文本，包含所有状态信息
            current_states_str = "\n".join([f"  {k}: {v}" for k, v in agent_s_output_obj.current_states.items()]) if isinstance(agent_s_output_obj.current_states, dict) else str(agent_s_output_obj.current_states)
            matched_states_str = "\n".join([f"  {k}: {v}" for k, v in agent_s_output_obj.matched_states.items()]) if isinstance(agent_s_output_obj.matched_states, dict) else str(agent_s_output_obj.matched_states)
            missing_states_str = "\n".join([f"  {k}: {v}" for k, v in agent_s_output_obj.missing_states.items()]) if isinstance(agent_s_output_obj.missing_states, dict) else str(agent_s_output_obj.missing_states)
            novel_states_str = "\n".join([f"  {k}: {v}" for k, v in agent_s_output_obj.novel_states.items()]) if isinstance(agent_s_output_obj.novel_states, dict) else str(agent_s_output_obj.novel_states)

        # ====================================================================
        # FORMAT GRAPH DB INSIGHTS FOR AGENTS A & B
        # ====================================================================

        graph_context_str = ""
        graph_insight_status = "No graph_insight available"

        # 【调试】检查 graph_insight 的具体内容
        if graph_insight is None:
            graph_insight_status = "graph_insight is None"
            print(f"[GRAPH_DEBUG] {graph_insight_status}")
        elif not isinstance(graph_insight, dict):
            graph_insight_status = f"graph_insight is not dict: {type(graph_insight)}"
            print(f"[GRAPH_DEBUG] {graph_insight_status}")
        elif 'dominant_pattern' not in graph_insight:
            graph_insight_status = "dominant_pattern not in graph_insight"
            print(f"[GRAPH_DEBUG] {graph_insight_status}")
        elif graph_insight['dominant_pattern'] is None:
            graph_insight_status = "dominant_pattern is None"
            print(f"[GRAPH_DEBUG] {graph_insight_status}")
        else:
            # 检查 attribution_analysis
            attr = graph_insight.get('attribution_analysis', {})
            success_count = len(attr.get('success_drivers', []))
            failure_count = len(attr.get('failure_drivers', []))
            context_count = len(attr.get('common_context', []))
            pattern_name = graph_insight['dominant_pattern'].get('name', 'Unknown')
            graph_insight_status = f"✓ Pattern: {pattern_name}, Success: {success_count}, Failure: {failure_count}, Context: {context_count}"
            print(f"[GRAPH_DEBUG] {graph_insight_status}")

        if graph_insight:
            graph_context_str = self._format_graph_insight(graph_insight)

        # ====================================================================
        # STAGE 2: AGENT A - DEEP ANALYSIS (Unstructured thinking)
        # ====================================================================
        
        # 【改进】格式化Agent S的输出供Agent A使用（支持新的Dict格式）
        agent_s_output_str = ""
        if agent_s_output_obj:
            # 新格式：Dict状态
            current_states_str = "\n".join([f"  - {k}: {v}" for k, v in agent_s_output_obj.current_states.items()]) if isinstance(agent_s_output_obj.current_states, dict) else str(agent_s_output_obj.current_states)
            matched_states_str = "\n".join([f"  - {k}: {v}" for k, v in agent_s_output_obj.matched_states.items()]) if isinstance(agent_s_output_obj.matched_states, dict) else str(agent_s_output_obj.matched_states)
            missing_states_str = "\n".join([f"  - {k}: {v}" for k, v in agent_s_output_obj.missing_states.items()]) if isinstance(agent_s_output_obj.missing_states, dict) else str(agent_s_output_obj.missing_states)
            novel_states_str = "\n".join([f"  - {k}: {v}" for k, v in agent_s_output_obj.novel_states.items()]) if isinstance(agent_s_output_obj.novel_states, dict) else str(agent_s_output_obj.novel_states)
            
            agent_s_output_str = f"""【Agent S - 市场状态分析结果】

思维链：
{agent_s_output_obj.reasoning_trace}

当前识别的状态：
{current_states_str if current_states_str else '无'}

与历史Pattern匹配的状态：
{matched_states_str if matched_states_str else '无'}

缺失的历史状态：
{missing_states_str if missing_states_str else '无'}

新增的独有状态（新风险或机会）：
{novel_states_str if novel_states_str else '无'}
"""
        # 【关键】不再使用agent_a_structured，直接使用Agent S的输出供Agent A分析
        structured_analysis_context = ""

        # Pre-compute values from market data
        market_emotion_str = state.get('market_emotion_label', 'Computing...')
        sentiment_prob_str = f"{state.get('market_emotion_score', 0):.0%}"
# 停用短期记忆memory
# {self._format_raw_history_a()}

# {self._format_raw_history_b()}

# 【历史参考提示】
# 请参考你的历史分析输出和历史交易决策（来自Agent B）
# 同时注意历史决策的趋向和情况，以理解当前的交易方向和账户状态演变。
# 如果改变之前的判断，请详细解释原因


        agent_a_prompt = f"""
你是小明，一个利用BTC历史模式和当前市场数据的行情研究员，对市场进行牛熊市趋势、反转和技术分析并给出初步的交易意见。你的分析应该以历史模式匹配作为主要参考框架。你追求高夏普，低风险暴露，尽量提高交易次数，提高胜率，及时止盈见好就收，如果赚钱了你获得50%分红，亏钱了自负50%亏损，

【基础交易框架】
在参考周期内，基于支撑/阻力位、趋势方向和价格形态制定交易计划。你的止盈、止损和仓位大小应与选定的参考周期保持一致。
积极判断超长期（月度以上）的行情反转，但是要通过支撑去检验是否真真发生了反转，例如冲高回撤但没跌破上一个高点，认为仍然为牛市，冲低回调但没涨过上一个低点认为仍然是熊市

【关键执行规则】
- 止盈止损必须设置，不能为0
- 已设置的止损代表可接受的最大损失，不应主动平仓
- 不要频繁改变交易方向，避免大幅反转
- 无持仓时，根据清晰信号决定是否进场
- 有持仓时，让止盈止损自动执行，除非出现重大风险


【核心分析框架 - 历史与当前的平衡】
下面的图数据库信息包含了历史相似情况的胜败统计，这是参考但不是绝对。同时你有当前的市场数据。
**重要**：不应该完全信任任何单一信息源：
- 历史数据可能过时或存在质量问题
- 当前市场信号也不一定完全可靠（短期噪音）
- 需要两者结合判断，当两者产生矛盾时，需要谨慎权衡而不是盲目倾向某一方

{graph_context_str}

[CURRENT MARKET STATE - STRUCTURED SUMMARY]
Price: {cur_price:.2f}
Market Emotion: {market_emotion_str}
Sentiment Probability: {sentiment_prob_str}

[MARKET DATA CONTEXT]
{full_market_context}

[AGENT S - TECHNICAL STATE CLASSIFICATION]
{agent_s_output_str}
{structured_analysis_context}



[AGENT A - DEEP ANALYSIS]

【基于当前市场数据的分析】
1. **当前市场的直观信号**：


2. **当前状态与历史模式的匹配度**：


【基于历史模式的分析】
1. **历史模式评估**：上述HISTORICAL PATTERN MATCH显示了什么样的胜率和条件？
   - 这个模式在历史中出现过多少次？
   - SUCCESS DRIVERS中有多少个已经在当前市场出现？
   - FAILURE DRIVERS中有任何已经出现的吗？

2. **当前市场与历史模式的匹配度**：
   - 当前的市场状态与历史高胜率情况有多接近？
   - 哪些COMMON CONTEXT在当前市场中存在？
   - 缺少哪些成功条件？

3. **趋势结构确认**：
   - 当前BTC的不同周期趋势是否与历史成功模式的趋势方向一致？
   - 当前的支撑/阻力位在哪里？

【账户状态分析】

{account_status_str}

{operation_history_str}

4. 当前持仓占比为 {current_pos_pct:.1%}，与当前的模式匹配度是否一致？

【历史交易决策分析】
7. 参考历史交易决策（来自Agent B），当前的交易趋向是什么？
8. 历史决策的信心度如何？当前市场是否支持继续维持这个方向？
9. 如果当前与历史模式不同，原因是什么？是市场条件改变了，还是之前的判断有误？

【重要提示 - 止损线的含义】
已设置的止损线(SL)代表当前能承受的最大损失，这个损失是可以接受的。不应该因为轻微的负面波动就主动平仓。
止损和止盈机制会自动执行，你的任务是基于当前信号决定是否继续持仓(HOLD)或者是否有强烈信号要求操作。

【重要提示 - 信息矛盾时的处理】
当历史数据和当前市场信号产生矛盾时，请你仔细思考

{analysis_note_str}

【关键约束 - Pattern 命名规范】
【重要】Pattern 名称代表"市场状态组合和交易逻辑"，**绝不应该包含交易结果（WIN/LOSS/BREAK_EVEN）**
Pattern 名称必须遵守以下格式规则，否则系统将拒绝使用：

【错误示例】❌ 不要这样做：
- Trade_SHORT_WIN（包含了交易结果 WIN）
- Trade_LONG_LOSS（包含了交易结果 LOSS）
- Pattern_RSI_Oversold_WIN（包含了交易结果 WIN）
- Pattern_Breakout_Volume_BREAK_EVEN（包含了交易结果 BREAK_EVEN）

【正确示例】✅ 应该这样做：
- Pattern_RSI_Oversold_Support（超卖反弹）
- Pattern_Breakout_VolumeSpike（突破验证）
- Pattern_Reversal_Divergence_RSI（背离反转）
- Pattern_Trend_Confirmation_MA（趋势确认）
- Pattern_MomentumShift_MACD（动量转换）
- Pattern_HigherHigh_HigherLow（更高的高点和更高的低点）

【命名规则汇总】
1. 所有 Pattern 名称必须以 "Pattern_" 开头
2. 类型名（TypeName）：描述模式的核心特征（Oversold, Breakout, Reversal, Consolidation, Trend, Support, Resistance 等）
3. 触发信号（Trigger）：触发这个模式的具体信号（VolumeSpike, RSI, MA, Divergence, Momentum 等）
4. **绝不包含 WIN/LOSS/BREAK_EVEN/DRAW 等交易结果标签**
5. 使用下划线连接单词，每个单词首字母大写

【为什么有这个约束】
- Pattern 代表市场条件和逻辑信号，与交易结果无关
- 同一个 Pattern（例如超卖反弹）既可能导致 WIN，也可能导致 LOSS
- 交易结果应该通过图数据库的 Event -[:RESULTED_IN]-> Outcome 关系记录
- 将 Outcome 和 Pattern 混合会破坏模式学习和归因分析

【多周期Pattern命名参考范例】
牛市Pattern示例: Pattern_Bull_1M, Pattern_Bull_1W, Pattern_Bull_4H, Pattern_Bull_1H, Pattern_Bull_1D
熊市Pattern示例: Pattern_Bear_1M, Pattern_Bear_1W, Pattern_Bear_4H, Pattern_Bear_1H, Pattern_Bear_1D
猴市Pattern示例: Pattern_Monkey_1M, Pattern_Monkey_1W, Pattern_Monkey_4H, Pattern_Monkey_1H, Pattern_Monkey_1D
无行情Pattern示例： Pattern_1d_Peace, Pattern_4h_Peace
【通用牛熊猴市判定规则】（适用所有周期）
以下状态代表在某个具体周期上呈现的行情特征：
**牛市判定:**
- 高点：连续创新高，或至少没有跌破上一个高点
- 低点：连续抬高，每次回调不破前期低点
- 确认：即使有回调，也要验证未跌破关键支撑（上升趋势的前期低点）

**熊市判定:**
- 高点：连续创新低，或至少没有涨过上一个高点
- 低点：连续下移，每次反弹不创新高
- 确认：即使有反弹，也要验证未涨过关键阻力（下降趋势的前期高点）

**猴市(震荡市)判定:**
- 高低点：一定周期内（如5-10根K线）既无新高也无新低
- 振幅：高波动，反复在支撑阻力间冲击但无有效突破
- 低波动：持续在区间平盘整理，成交量枯竭，没有交易机会

- **无行情** （高低点无突破，无大波动）：

**关键反转识别:**
- 从牛市到熊市：冲高但未涨过上期高点 → 警惕，需要确认跌破上升支撑线
- 从熊市到牛市：冲低但未跌过上期低点 → 警惕，需要确认突破下降阻力线
- 完全反转信号：跌破/突破关键支撑/阻力位 + 成交量配合 → 趋势反转确认


【核心DNA识别要求 - 强制分离核心与环境】
你必须严格区分"核心"与"环境"状态：

【核心DNA（1-3个状态）】
这些是定义该Pattern本质的特征。去掉任何一个，模式就不成立。
示例：
- 如果Pattern是"超卖反弹"，核心DNA可能是: ['RSI_Oversold', 'Support_Hit']
- 如果Pattern是"突破确认"，核心DNA可能是: ['Breakout_Confirmed']
- 如果Pattern是"趋势背离"，核心DNA可能是: ['Price_Higher', 'MACD_Lower']
- 如果Pattern是"趋势背离"，核心DNA可能是: ['Bull_1w', 'MACD_Lower']

【环境Context（0个或多个状态）】
这些是同时存在但不必要的背景信息。用于后续的微观归因分析，帮助理解为什么同一Pattern有时赚有时赔。
示例：
- 为"超卖反弹"提供Context: ['Volume_Spike', 'Bull_Sentiment', 'High_Liquidity']
- 为"突破确认"提供Context: ['Strong_Trend', 'Low_VIX']

【输出要求】
请提供详细的思考过程，包括：
- 对历史模式匹配度的评估
- SUCCESS/FAILURE DRIVERS的分析
- 当前市场与历史高概率情况的对比
- Pattern匹配的信号强度评估（强/中/弱）
- 根据目前的分析给出一个具体的Pattern和理由



【Pattern 输出格式要求】
Pattern 名称必须严格遵守 "Pattern_TypeName_Trigger" 或 "Pattern_ComplexName" 格式。
系统会自动验证名称不包含任何禁止的结果标签（WIN/LOSS 等），违反规则的输出将被过滤或使用备用值。


"""

        # 【新增】STAGE 2-REVISED: AGENT A - DEEP ANALYSIS WITH STRUCTURED OUTPUT
        # ====================================================================

        # Call Agent A for deep thinking (now with structured output)
        agent_a_output_obj: Optional[SchemaAgentAOutput] = None
        agent_a_output_str = ""
        try:
            agent_a_output_obj = self._agent_a_analyze_with_structure(agent_a_prompt)

            if agent_a_output_obj:
                # 格式化Agent A的结构化输出供Agent B使用
                core_dna_str = ", ".join(agent_a_output_obj.core_dna_states) if agent_a_output_obj.core_dna_states else "无"
                context_str = ", ".join(agent_a_output_obj.context_states) if agent_a_output_obj.context_states else "无"

                agent_a_output_str = f"""【Agent A - 深度分析结果（结构化）】

【思维链】
{agent_a_output_obj.reasoning_trace}

【Pattern识别】
Pattern名称: {agent_a_output_obj.pattern_name}
Pattern描述: {agent_a_output_obj.pattern_description}

【核心DNA（1-3个）】
{core_dna_str}

【环境Context】
{context_str}

【置信度评分】
{agent_a_output_obj.confidence:.1%}
"""
                # print(f"[AGENT_A] Analysis:\n{agent_a_output_str}\n")
            else:
                agent_a_output_str = "Agent A analysis failed"
                print(f"[AGENT_A_ERROR] Agent A returned None")

        except Exception as e:
            print(f"[AGENT_A_ERROR] {e}")
            agent_a_output_str = "Agent A analysis failed"

        # 【新增】保存 Agent A 输出到实例变量，供 engine 提取
        self._agent_a_output_obj = agent_a_output_obj
        print(f"[AGENT_OUTPUT_SAVED] Saved Agent A output for TradeContext: {agent_a_output_obj is not None}")

        # 【新增】记录 Agent A 的提示词和输出
        if prompt_logger:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            try:
                prompt_logger.log_agent_prompt(
                    agent_name="A",
                    bar_number=getattr(self, '_bar_count', 0),
                    timestamp=timestamp,
                    prompt_text=agent_a_prompt,
                    additional_info={
                        "cur_price": f"{cur_price:.2f}",
                        "position": f"{state.get('position', 0):.4f}",
                        "cash": f"{state.get('cash', 0):.2f}",
                        "context_length": len(full_market_context),
                        "graph_insight_status": graph_insight_status,  # 【新增】图数据库状态
                        "pattern_identified": agent_a_output_obj.pattern_name if agent_a_output_obj else "None",
                        "confidence": f"{agent_a_output_obj.confidence:.1%}" if agent_a_output_obj else "N/A"
                    },
                    output_text=agent_a_output_str,
                    output_info={
                        "output_type": "StructuredAgentAOutput",
                        "output_length": len(agent_a_output_str),
                        "analysis_type": "Deep Market Analysis with Pattern Recognition",
                        "pattern_name": agent_a_output_obj.pattern_name if agent_a_output_obj else "None",
                        "core_dna_states_count": len(agent_a_output_obj.core_dna_states) if agent_a_output_obj else 0,
                        "context_states_count": len(agent_a_output_obj.context_states) if agent_a_output_obj else 0,
                        "graph_insight_debug": graph_insight_status  # 【新增】调试信息
                    }
                )
            except Exception as log_err:
                print(f"[PROMPT_LOGGER_ERROR] Failed to log Agent A output: {log_err}")


        # ====================================================================
        # STAGE 3: AGENT B - RISK ASSESSMENT & STRUCTURED DECISION
        # ====================================================================

        # 【新增】从 state 或其他方式获取当前杠杆倍率和交易周期
        current_leverage = getattr(self, '_current_leverage', 1.0)
        main_tf = getattr(self, '_main_tf', '1h')  # 【新增】获取主要交易周期
        # 最大杠杆倍数：通常是当前杠杆的5-10倍，但这里我们只展示当前配置允许的上限
        # 如果后续想要支持动态杠杆，可以将这个值更新
        max_leverage = current_leverage * 5.0  # 如果要支持杠杆升级，可以最多升到5倍

        # 【新增】生成详细的杠杆管理教育文案
        leverage_education_str = f"""
【⚠️ 杠杆机制详解 - 必须理解】

【当前系统配置】
- 当前杠杆倍率: {current_leverage:.1f}x
- 主要交易周期: {main_tf.upper()}
- 账户本金: {cash_str} USDT
- 当前持仓: {pos_str} BTC


【杠杆的核心含义】
杠杆倍率 {current_leverage:.1f}x 意味着：
- 你可以用 {cash_str} 的账户，控制最多 {float(cash_str) * current_leverage:,.0f} USDT 的头寸
- 例如：决定 quantity_pct=50% 时，实际控制的头寸 = {cash_str} × 50% × {current_leverage:.1f}x = {float(cash_str) * 0.5 * current_leverage:,.0f} USDT

【杠杆如何影响风险和收益】
1. **收益放大**：价格上涨 1%，你的账户收益率 = 1% × {current_leverage:.1f}x = {1.0 * current_leverage:.1f}%
2. **损失放大**：价格下跌 1%，你的账户亏损率 = 1% × {current_leverage:.1f}x = -{1.0 * current_leverage:.1f}%
3. **爆仓风险**：如果账户亏损达到本金，就会被强制平仓（虽然系统会在 98% 时就停止加仓）

【仓位大小与风险的关系】
- quantity_pct × leverage = 实际杠杆倍数
- 例如，quantity_pct=50%, leverage={current_leverage:.1f}x → 实际控制 {0.5 * current_leverage:.1f}x 的头寸



"""
# 停用短期记忆memory        
# {self._format_raw_history_b()}

# 【前几期思考情况】
# 请辩证看待你前几期的决策记录和原因，不要过度依赖你自己前几期的思考
# 不要因为历史都在HOLD错过明显的交易机会



#减少你的通勤时间，因为你晕车晕地铁，只能在公司旁边买房，另外，你重要的亲戚生病了，每周需要一笔不小的医疗费，因此请你争取每一次机会进行交易。
        agent_b_prompt = f"""
你是李华，一个风险控制和交易决策者。你应该充分利用图数据库返回的历史模式匹配数据作为主要决策依据。基于Agent A的深度分析进行最终决策。
你追求高收益和风险平衡，如果赚钱了你获得50分红，亏钱了自负50%亏损，请努力分析工作.下面是你的同事给你提供的具体情况和交易准则

{account_status_str}

{leverage_education_str}

【核心决策原则】
请你积极参考你的同事小明发送的分析内容，并结合HISTORICAL PATTERNS ，利用行情对二者进行进行核实和辩证看待后做出交易决策

【基础交易框架】
在选定的参考周期内，基于支撑/阻力位、趋势方向和价格形态制定交易计划。你的止盈、止损和仓位大小应与选定的参考周期保持一致。
你是典型的风险厌恶型，每次交易应该只进行符合目前平均波动率的的止盈止损进行风险暴露，争取更高的夏普比例，积少成多得到收益，特殊情况也可满仓
在判断行情将在短期开始“猴市”行情波动时，不要追涨杀跌，可以围绕均线涨后做空，跌后做多，但一定要在目前是大猴市的前提下
在判断行情长期处于牛市时，不要进行大幅的做空，可以抓住大涨后的小空的机会，并在大跌后做多
在判断行情长期处于熊市时，不要进行大幅的做多，可以抓住大跌后的小多机会，并在大涨后做空
积极判断超长期（月度以上）的行情反转，但是要通过支撑去检验是否真真发生了反转，例如冲高回撤但没跌破上一个高点，认为仍然为牛市，冲低回调但没涨过上一个低点认为仍然是熊市

【交易周期的含义】
你正在 {main_tf.upper()} 周期进行交易，这意味着：
- 所有的支撑/阻力位判断应该基于 {main_tf.upper()} 周期，并一定留有余量防止被波动冲破止盈止损线
- 止盈止损的距离应该与 {main_tf.upper()} 周期的平均波动幅度相符
- 趋势判断应该以比 {main_tf.upper()} 更高的周期为主
- 你应该发挥 {main_tf.upper()} 周期的优势尽量减小风险敞口，同时获得收益

【关键执行规则】
- 止盈止损必须设置，不能为0
- 最小化风险暴露，止盈止损最好控制在3%以内，同时见好就收，存在高风险就撤离
- 已设置的止损代表可接受的最大损失，
- 有持仓时，如果不进行调仓，则使用HOLD决策，如果进行调仓再使用LONG或者SHORT以及需要变更的仓位水平

{analysis_note_str}

[HISTORICAL PATTERNS ]
{graph_context_str}

[MARKET DATA ]
Current Price: {cur_price:.2f}
Position: {pos_str} | Cash: {cash_str}
Position Percentage: {current_pos_pct:.1%}
TP: {state['current_tp']:.2f} | SL: {state['current_sl']:.2f}

{full_market_context_b}



{operation_history_str}

【操作历史提示】
这是你之前的对仓位的操作情况






[同事小明 DEEP ANALYSIS]代表当下行情与历史行情的差异
{agent_a_output_str}
请你批判辩证看待同事小明的分析，重点参考他对行情的判断，不要过度依赖你自己前几期的思考


【核心分析框架 - 历史与当前的平衡】
下面的图数据库信息包含了历史相似情况的胜败统计，这是参考但不是绝对。同时你有当前的市场数据。
**重要**：不应该完全信任任何单一信息源：
- 历史数据可能过时或存在质量问题
- 当前市场信号也不一定完全可靠（短期噪音）
- 需要两者结合判断，当两者产生矛盾时，需要谨慎权衡而不是盲目倾向某一方


【决策输出】
你可以做多或者做空，你的目标仓位会乘以杠杆别率，你可以通过对二者的简单计算得到你最终仓位，并进行风险控制
在调整仓位时使用LONG\SHORT与新的仓位水平、在不调整仓位时使用HOLD、在关闭交易或者平仓时使用CLOSE



"""

        # 【新增】记录 Agent B 的提示词（先不记录输出，等执行后再添加）
        prompt_logger = get_prompt_logger()

        # Call Agent B with retries
        max_retries = 3
        spinner = None
        agent_b_output = ""  # 用于记录输出的变量

        for attempt in range(max_retries):
            try:
                spinner = LoadingSpinner(message="Agent B deciding...")
                spinner.start()
                # 【修复】直接调用 LLM，不经过 agent 框架
                res, err = self._invoke_with_timeout(
                    self.llm_agent_b.with_structured_output(SchemaAgentBOutput).invoke,
                    (agent_b_prompt,),
                    300
                )
                if spinner:
                    spinner.stop()

                if err:
                    raise err

                # 【修复】保留原本逻辑，同时适配直接调用返回的对象
                if isinstance(res, dict):
                    decision = res.get("structured_response")
                    agent_b_output = str(decision) if decision else ""
                elif isinstance(res, SchemaAgentBOutput):
                    # 【新增】适配直接调用返回的结构化对象
                    decision = res
                    agent_b_output = str(decision) if decision else ""
                else:
                    decision = None
                    agent_b_output = ""

                if decision:
                    print(f"[AGENT_B] Decision: {decision}")

                    # 【新增】记录到历史缓冲区
                    self.history_b.append(decision)
                    if len(self.history_b) > self.max_history:
                        self.history_b.pop(0)

                    # 【新增】记录token使用
                    token_logger = get_token_logger()
                    if token_logger:
                        try:
                            token_logger.log_from_prompt_and_output(
                                agent_name="B",
                                prompt_text=agent_b_prompt,
                                output_text=agent_b_output,
                                bar_number=getattr(self, '_bar_count', None),
                                model=QWEN_MODEL
                            )
                        except Exception as e:
                            print(f"[TOKEN_LOG_ERROR] Failed to log Agent B tokens: {e}")

                    # 【新增】记录 Agent B 的提示词和输出
                    if prompt_logger:
                        from datetime import datetime
                        timestamp = datetime.now().isoformat()
                        prompt_logger.log_agent_prompt(
                            agent_name="B",
                            bar_number=getattr(self, '_bar_count', 0),
                            timestamp=timestamp,
                            prompt_text=agent_b_prompt,
                            additional_info={
                                "cur_price": f"{cur_price:.2f}",
                                "position": pos_str,
                                "cash": cash_str,
                                "position_pct": f"{current_pos_pct:.1%}",
                                "context_length": len(full_market_context),
                                "agent_a_output_length": len(agent_a_output_str),
                                "graph_insight_status": graph_insight_status,  # 【新增】图数据库状态
                                "attempt": attempt + 1
                            },
                            output_text=agent_b_output,
                            output_info={
                                "output_length": len(agent_b_output),
                                "decision_type": str(decision.action) if hasattr(decision, 'action') else "Unknown",
                                "graph_insight_debug": graph_insight_status  # 【新增】调试信息
                            }
                        )
                    self._agent_b_output_obj = decision
                    return decision
                else:
                    print(f"[WARNING] Attempt {attempt + 1}: Parse failed, retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(10)

            except Exception as e:
                if spinner:
                    spinner.stop()
                print(f"[AGENT_B_ERROR] Attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10 + attempt * 5)

        # Fallback to HOLD if all retries fail
        print("[FALLBACK] Max retries exhausted, forcing HOLD")

        # 【新增】记录失败时的 Agent B 日志
        if prompt_logger:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            prompt_logger.log_agent_prompt(
                agent_name="B",
                bar_number=getattr(self, '_bar_count', 0),
                timestamp=timestamp,
                prompt_text=agent_b_prompt,
                additional_info={
                    "cur_price": f"{cur_price:.2f}",
                    "position": pos_str,
                    "cash": cash_str,
                    "position_pct": f"{current_pos_pct:.1%}",
                    "context_length": len(full_market_context),
                    "agent_a_output_length": len(agent_a_output_str),
                    "graph_insight_status": graph_insight_status,  # 【新增】图数据库状态
                    "max_retries": max_retries,
                    "status": "FAILED"
                },
                output_text="[FALLBACK] Decision failed after max retries, forcing HOLD",
                output_info={
                    "output_length": 0,
                    "decision_type": "HOLD (Fallback)",
                    "status": "Failed",
                    "graph_insight_debug": graph_insight_status  # 【新增】调试信息
                }
            )

        return SchemaAgentBOutput(
            action="HOLD",
            quantity_pct=0.0,
            limit_price=0,
            take_profit_price=0,
            stop_loss_price=0,
            reasoning="Decision failed after retries",
            confidence=0.0,
            risk_reward_ratio=1.0
        )
