# ace_trading/agents/tools.py
from langchain_core.tools import tool
from typing import List

@tool
def calculate_support_resistance(prices: List[float]) -> dict:
    """
    计算最近价格序列的支撑位(最小值)和阻力位(最大值)。
    用于辅助判断价格区间。
    """
    if not prices: return {}
    return {"resistance": max(prices), "support": min(prices)}