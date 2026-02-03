# ace_trading/agents/output_schemas.py
"""
Agent 输出模型定义（Pydantic v2）
支持 LangChain 结构化输出（response_format）与图数据库集成
"""
from typing import Dict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field, StringConstraints

# ====================================================================
# 【新增】Agent S 状态名称的标准格式约束
# ====================================================================
# 定义标准状态名称格式：
# 允许：RSI_Overbought, Vol_Spike, MA20_Support, Price_AtHigh
# 正则含义：大写字母开头，后面跟字母数字下划线，长度至少2
# 示例合法格式：
#   - RSI_High / RSI_Overbought
#   - Volume_Spike / Vol_Spike
#   - Price_AtHigh / Price_AtLow
#   - Trend_Uptrend / Trend_Strong
#   - MA20_Support / EMA50_Resistance
StandardStateName = Annotated[str, StringConstraints(pattern=r"^[A-Z][a-zA-Z0-9_]*$", min_length=2)]

# ====================================================================
# 【新增】Agent A Pattern 名称的标准格式约束
# ====================================================================
# 定义标准 Pattern 命名格式：
# 允许：Pattern_Breakout_VolumeSpike, Pattern_Reversal_RSI, Pattern_Consolidation_Break
# 正则含义：必须以 Pattern_ 开头，后跟大写字母/数字/下划线，长度至少3
# 示例合法格式：
#   - Pattern_Breakout_VolumeSpike
#   - Pattern_Reversal_RSI
#   - Pattern_Consolidation_Break
#   - Pattern_MomentumShift
#   - Pattern_LongSetup_Strong
StandardPatternName = Annotated[str, StringConstraints(pattern=r"^Pattern_[A-Z][a-zA-Z0-9_]*$", min_length=10)]


class AgentSOutput(BaseModel):
    """
    Agent S 的融合输出结构：
    直接包含思维链，且所有状态字段均为字典格式（State -> Reason）。
    
    状态名称必须遵守StandardStateName格式：大写字母开头，长度≥2
    """
    
    # 1. 总体思考
    reasoning_trace: str = Field(
        ..., 
        description="【思维链】详细的观察与对比分析过程。解释为什么判定某些状态存在，以及新状态的影响。"
    )

    # 2. 状态字典 (Key=状态名, Value=具体的判定理由/数据证据)
    # 使用 StandardStateName 约束状态名称格式
    current_states: Dict[StandardStateName, str] = Field(
        default_factory=dict,
        description="客观检测到的显著状态。格式：{'RSI_Overbought': 'RSI数值为78，超过70阈值', 'Vol_Spike': '成交量是均值的3倍'}"
    )
    
    matched_states: Dict[StandardStateName, str] = Field(
        default_factory=dict,
        description="【符合项】与历史Pattern一致的状态。Value应说明匹配的程度。"
    )
    
    missing_states: Dict[StandardStateName, str] = Field(
        default_factory=dict,
        description="【缺失项】历史需要但当前没有的状态。Value应解释缺失的证据（例如：'要求放量，但当前量能萎缩'）。"
    )
    
    novel_states: Dict[StandardStateName, str] = Field(
        default_factory=dict,
        description="【新增项】当前独有的新状态。Value应解释为什么这是一个值得注意的新风险或机会。"
    )


class MarketStateClassification(BaseModel):
    """
    Agent S 输出：市场状态分类与技术面评估
    
    用于图数据库记录当前市场状态快照
    """
    states: List[str] = Field(
        description="识别的市场状态列表，如 ['Overbought', 'Uptrend']"
    )
    state_descriptions: Dict[str, str] = Field(
        description="各状态的详细描述，键为状态名，值为描述"
    )
    confidence_scores: Dict[str, float] = Field(
        description="各状态的置信度 (0-1)，用于加权"
    )
    trend_direction: str = Field(
        description="趋势方向: UPTREND / DOWNTREND / CONSOLIDATION"
    )
    signal_strength: str = Field(
        description="信号强度: STRONG / MEDIUM / WEAK"
    )
    key_support: float = Field(
        description="关键支撑位"
    )
    key_resistance: float = Field(
        description="关键阻力位"
    )
    overbought_oversold: Optional[str] = Field(
        default=None,
        description="超买/超卖状态: Overbought / Oversold / Neutral"
    )
    summary: str = Field(
        description="一句话市场状态总结"
    )


class DeepMarketAnalysis(BaseModel):
    """
    Agent A 输出：深度市场分析与概率评估
    
    用于风险评估、历史模式匹配、情绪指标综合判断
    """
    pattern_match: str = Field(
        description="与历史模式的匹配度与描述"
    )
    success_drivers_present: List[str] = Field(
        description="当前已出现的成功交易驱动因素"
    )
    success_drivers_missing: List[str] = Field(
        description="缺失的成功交易驱动因素"
    )
    failure_risks: List[str] = Field(
        description="当前已出现的失败风险因素"
    )
    market_emotion: str = Field(
        description="市场整体情绪: Bullish / Bearish / Neutral / Extremely_Bullish / Extremely_Bearish"
    )
    sentiment_score: float = Field(
        description="市场情绪评分 (-1.0 极度空头 到 +1.0 极度多头)",
        ge=-1.0,
        le=1.0
    )
    probability_estimate: float = Field(
        description="当前交易机会的成功概率 (0-1)",
        ge=0.0,
        le=1.0
    )
    key_risks: List[str] = Field(
        description="当前交易的关键风险列表"
    )
    technical_alignment: str = Field(
        description="市场情绪与技术指标的一致性: Perfect_Align / Good_Align / Partial_Align / Divergence"
    )
    funding_rate_interpretation: Optional[str] = Field(
        default=None,
        description="资金费率的市场含义"
    )
    open_interest_interpretation: Optional[str] = Field(
        default=None,
        description="持仓量的市场含义"
    )
    recommendation: str = Field(
        description="综合建议: Strong_LONG / LONG / Weak_LONG / HOLD / Weak_SHORT / SHORT / Strong_SHORT"
    )
    detailed_reasoning: str = Field(
        description="详细推理过程（可包含多段分析）"
    )


class TradeDecision(BaseModel):
    """
    Agent B 输出：最终交易决策

    包含仓位管理、风险控制、决策信心等信息
    """
    action: str = Field(
        description="交易行为: LONG / SHORT / HOLD / CLOSE"
    )
    quantity_pct: float = Field(
        description="目标仓位百分比 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    limit_price: float = Field(
        description="限价单价格 (0 表示市价)"
    )
    take_profit_price: float = Field(
        description="止盈价格"
    )
    stop_loss_price: float = Field(
        description="止损价格"
    )
    reasoning: str = Field(
        description="决策原因与逻辑"
    )
    confidence: float = Field(
        description="决策信心度 (0-1)",
        ge=0.0,
        le=1.0
    )
    risk_reward_ratio: float = Field(
        description="风险回报比 (TP距离 / SL距离)"
    )
    position_adjustment_reason: Optional[str] = Field(
        default=None,
        description="如果与当前仓位不同，解释调整原因"
    )


# ====================================================================
# 【新增】Agent A 结构化输出模型 - 增强版（核心DNA + 环境Context分层）
# ====================================================================
class AgentAOutput(BaseModel):
    """
    Agent A 结构化输出：定义本次交易的核心逻辑（Pattern）与决策依据

    【核心改进】显式分离"核心DNA"与"环境Context"
    - core_dna_states (1-3个): 定义该Pattern本质的最少状态。无此特征则Pattern不成立
    - context_states (任意): 同时存在但非必要的背景信息，用于微观归因分析

    用于：
    1. 供 Agent B 精确引用核心逻辑
    2. 供图数据库学习 Pattern 与收益的关联
    3. 供系统学习什么样的 Pattern 应该增加/降低信心度
    """

    # 1. 总体思考链
    reasoning_trace: str = Field(
        ...,
        description="【思维链】深度归因分析。解释当前市场条件如何匹配历史Pattern，关键驱动因素是什么，置信度评估的依据。最后输出看多或者看空或者未发现明显交易机会或者信号"
    )

    # ============================================
    # 【核心DNA】用于生成Pattern Hash
    # ============================================
    core_dna_states: List[StandardStateName] = Field(
        min_length=1,
        max_length=3,
        description="【核心DNA - 最多3个】定义该Pattern本质的1-3个关键状态。这些状态缺一不可。"
                    "【重要】强制限制为3个以内，确保Pattern稳定性和可复用性。"
                    "示例：['RSI_Oversold', 'Support_Hit'] 或 ['Breakout_Confirmed']"
    )

    # ============================================
    # 【环境Context】不参与Hash，用于微观归因分析
    # ============================================
    context_states: List[StandardStateName] = Field(
        default_factory=list,
        description="【环境Context - 任意个数】同时存在但非核心的辅助状态或环境特征。"
                    "用于微观归因：找出为什么同一Pattern有时赚有时赔。"
                    "示例：['Volume_Spike', 'Bull_Sentiment', 'High_Liquidity']"
    )

    # ============================================
    # 【保留】Pattern信息与置信度
    # ============================================
    # 3. Pattern 名称（标准化命名）
    pattern_name: StandardPatternName = Field(
        description="Pattern 的标准化命名。格式：Pattern_TypeName_Trigger。【重要】绝不包含交易结果(WIN/LOSS/BREAK_EVEN/DRAW)。示例：Pattern_Oversold_Support, Pattern_Breakout_VolumeSpike, Pattern_Reversal_Divergence"
    )

    # 4. Pattern 描述
    pattern_description: str = Field(
        description="Pattern 的简洁描述（一到两句话）。说明这个Pattern是什么、为什么在当前条件下出现。"
    )

    # 5. 置信度
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="对本次信号的置信度评分 (0.0-1.0)。综合考虑：历史胜率、当前状态的完整性、缺失的驱动因素、市场情绪一致性。"
    )

    # ============================================
    # 【兼容性】向后兼容旧代码
    # ============================================
    @property
    def core_logic_states(self) -> List[str]:
        """向后兼容：旧代码可继续访问core_logic_states（等同于core_dna_states）"""
        return self.core_dna_states


# ====================================================================
# 【新增】Agent B 结构化输出模型 - 标准化决策输出
# ====================================================================
# 定义交易动作的标准枚举
StandardTradeAction = Literal["LONG", "SHORT", "HOLD", "CLOSE"]


class AgentBOutput(BaseModel):
    """
    Agent B 结构化输出：最终风险控制与交易决策

    用于：
    1. 供 engine.py 执行交易决策
    2. 供图数据库学习决策与收益的关联
    3. 供系统优化风险控制策略

    采用与 Agent S/A 相同的 LangChain response_format 约束模式
    """

    # 1. 交易动作（严格枚举）
    action: StandardTradeAction = Field(
        description="交易行为：LONG（做多）/ SHORT（做空）/ HOLD（持观）/ CLOSE（平仓）"
    )

    # 2. 目标仓位百分比（0-1范围）
    quantity_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="目标仓位百分比。范围：0.0（空仓）- 1.0（满仓）。例如：0.5 表示投入50%资金"
    )

    # 3. 止盈价格
    take_profit_price: float = Field(
        description="止盈价格。设置为目标利润水平。若为0或负数，表示未设置止盈"
    )

    # 4. 止损价格
    stop_loss_price: float = Field(
        description="止损价格。设置为可容忍的最大亏损水平。若为0或负数，表示未设置止损"
    )

    # 5. 限价单价格（可选）
    limit_price: float = Field(
        default=0,
        description="限价单价格。0表示使用市价单。用于精确进场价格控制"
    )

    # 6. 决策原因与逻辑
    reasoning: str = Field(
        description="【决策原因】详细解释为什么做出此决策。包括：关键因素分析、风险评估、市场情绪判断、账户状态考虑等"
    )

    # 7. 决策信心度
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="对本次决策的信心度评分 (0.0-1.0)。综合考虑：历史胜率、当前状态完整性、缺失因素、情绪一致性"
    )

    # 8. 风险回报比
    risk_reward_ratio: float = Field(
        default=1.0,
        description="风险回报比 = TP距离 / SL距离。用于风险管理评估。长期来看，在确定长线趋势进行操作时，为了防止被波动平仓导致亏损，风险回报比应该小于1 ，相反，短线操作，风险回报比应该不大于1"
    )

    # 9. 仓位调整原因（可选）
    position_adjustment_reason: Optional[str] = Field(
        default=None,
        description="如果目标仓位与当前仓位不同，说明调整原因。例如：'风险上升，降低仓位至30%' 或 '机会明确，加仓至70%'"
    )

