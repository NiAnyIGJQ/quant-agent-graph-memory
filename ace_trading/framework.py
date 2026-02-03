# ace_trading/framework.py
import backtrader as bt
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List, Optional
from abc import ABC, abstractmethod

# 1. Agent S (感知与对比) - 输出结构
# 【改进】AgentSOutput 现在由 output_schemas.py 提供，这里保留兼容性定义
class StateComparisonResult(BaseModel):
    """
    【已弃用】Agent S 的旧结构化输出格式
    保留此定义以备向后兼容，新系统应使用 output_schemas.AgentSOutput
    """
    current_states: List[str] = Field(default=[], description="当前检测到的所有显著状态")
    matched_states: List[str] = Field(default=[], description="【符合项】当前状态 与 历史Pattern定义 的交集")
    missing_states: List[str] = Field(default=[], description="【缺失项】历史Pattern定义中有，但当前缺失的状态")
    novel_states: List[str] = Field(default=[], description="【新增项】当前有，但历史Pattern定义中没有的状态（潜在风险/机会）")


# 2. Agent A (逻辑归因) - 输出结构
class LogicDefinition(BaseModel):
    """
    Agent A 的结构化输出：定义本次交易的核心逻辑（Pattern）
    """
    core_logic_states: List[str] = Field(..., min_length=1, description="【核心逻辑】本次交易最关键的归因状态")
    pattern_name: str = Field(..., description="Pattern 的标准化命名")
    pattern_description: str = Field(..., description="Pattern 的一句话描述")
    confidence: float = Field(..., ge=0.0, le=1.0, description="对本次信号的置信度评分")






# 3. Agent B (执行决策) - 输出结构
class ExecutionPlan(BaseModel):
    """
    Agent B 的结构化输出：具体的交易指令
    """
    action: Literal['LONG', 'SHORT', 'NEUTRAL'] = Field(..., description="最终决策动作")
    quantity_pct: float = Field(..., ge=0.0, le=1.0, description="目标仓位比例")
    take_profit_price: float = Field(..., description="止盈价格")
    stop_loss_price: float = Field(..., description="止损价格")

class AgentBOutput(BaseModel):
    """
    Agent B 的完整输出包
    """
    reasoning_trace: str = Field(..., description="【思维链】风控审查过程")
    data: ExecutionPlan = Field(..., description="结构化数据")


# 4. 跨时空记忆传递 - TradeContext (黑匣子)
class TradeContext(BaseModel):
    """
    【跨时空记忆包】
    这个对象在开仓时创建，挂载在 Order/Position 上。
    平仓时，Agent C 读取这个对象来写入图数据库。
    """
    event_id: str = Field(..., description="事件唯一标识")
    entry_timestamp: str = Field(..., description="开仓时间戳")

    # 保存完整的 Agent 输出历史
    s_output: Optional[Any] = Field(None, description="Agent S 的输出")  # 类型将通过下方导入和 model_rebuild() 更新
    a_output: Optional[Any] = Field(None, description="Agent A 的输出")  # 类型将通过下方导入和 model_rebuild() 更新
    b_output: Optional[Any] = Field(None, description="Agent B 的输出")  # 接受任何类型的 AgentBOutput（兼容新旧格式）

    # 辅助信息
    graph_search_summary: str = Field(default="", description="开仓时参考的历史 Pattern 摘要")


# ==========================================
# 原有交易决策结构 (兼容保留)
# ==========================================

# 1. 交易决策结构 (包含止盈止损)
class TradeDecision(BaseModel):
    action: Literal['LONG', 'SHORT', 'CLOSE', 'HOLD'] = Field(
        description="交易动作: LONG(做多), SHORT(做空), CLOSE(平仓), HOLD(观望)"
    )
    quantity_pct: float = Field(
        description="目标仓位比例 (0.0 - 1.0)。配合 leverage 可实现杠杆。",
        ge=0.0, le=1.0
    )
    limit_price: float = Field(
        description="挂单价格。0 代表市价单。"
    )
    take_profit_price: float = Field(
        description="预期止盈价格。"
    )
    stop_loss_price: float = Field(
        description="预期止损价格。"
    )
    reasoning: str = Field(default="", description="决策理由")

# 3. Agent 抽象基类
class BaseAgent(ABC):
    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> Any:
        pass
        """
        【改进】现在返回 Any 类型以支持多种输出格式
        - TradeDecision (旧格式，向后兼容)
        - SchemaAgentBOutput (新标准化格式)
        """

# 4. 佣金模式 (支持小数/利息/杠杆)
class MarginCommInfo(bt.CommInfoBase):
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
        ('interest', 0.01),       # 默认 1% 年化借贷利息
        ('interest_long', False), # 多单不收利息
        ('leverage', 1.0),        # 默认杠杆倍数
        ('automargin', False),
    )

    def getsize(self, price, cash):
        """允许小数交易"""
        if self.p.commtype == self.COMM_PERC:
            # 防御性检查：price 可能为 0 或 None，避免 ZeroDivisionError
            try:
                if price is None or price == 0:
                    return 0
                return (cash * self.p.leverage) / price
            except Exception:
                return 0
        return super().getsize(price, cash)

    def _get_credit_interest(self, data, size, price, days, dt0, dt1):
        """计算做空利息"""
        if size < 0: # 做空收取利息
            return abs(size) * price * (self.p.interest / 365.0) * days
        return 0.0


# ==========================================
# 【重要】模型重建 - 导入实际类并解决前向引用
# ==========================================
# 为了避免循环导入，AgentSOutput 在文件末尾导入
# 然后调用 model_rebuild() 来解决前向引用
try:
    from ace_trading.agents.output_schemas import AgentSOutput
    # 将 TradeContext 中的前向引用更新为实际类型
    TradeContext.model_rebuild()
except ImportError:
    # 如果导入失败（模块尚未初始化），继续不中断
    pass