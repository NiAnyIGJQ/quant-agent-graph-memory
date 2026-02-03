"""
QGA 图数据库查询功能测试
测试用途：验证 query_similar_events_insight 的查询和归因分析能力
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# 添加相对路径以导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ace_trading', 'graph'))

from quant_graph_manager import QGAGraphManager


def pretty_print_raw_result(result: Dict[str, Any]) -> None:
    """打印原始返回值"""
    print("\n" + "="*80)
    print("【原始返回值 (Raw Result)】")
    print("="*80)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_and_interpret_result(result: Dict[str, Any]) -> None:
    """解析和解释返回的结果"""
    print("\n" + "="*80)
    print("【解析与解释 (Parsed Interpretation)】")
    print("="*80)
    
    # 提取主导模式信息
    pattern = result.get('dominant_pattern')
    attribution = result.get('attribution_analysis', {})
    
    if pattern is None:
        print("⚠️  未找到主导模式。这可能意味着：")
        print("   1. 相似事件中没有明显的模式集中度")
        print("   2. 数据库中没有这些 Event ID 的记录")
        print("   3. Event 节点未与 Pattern 节点关联")
        return
    
    print("\n✅ 主导模式信息:")
    print("-" * 80)
    print(f"  模式名称: {pattern['name']}")
    print(f"  模式描述: {pattern['description']}")
    print(f"  建议动作: {pattern['action']}")
    print(f"  基准胜率: {pattern['baseline_win_rate']:.1%}")
    print(f"  匹配事件数: {pattern['match_count']}")
    print(f"  模式定义 (核心 State): {pattern['definition']}")
    
    # 解析成功驱动因子
    success_drivers = attribution.get('success_drivers', [])
    if success_drivers:
        print("\n✅ 成功驱动因子 (Alpha 特征 - 增强胜率):")
        print("-" * 80)
        for idx, driver in enumerate(success_drivers, 1):
            print(f"  {idx}. {driver['state']}")
            print(f"     - 条件胜率: {driver['win_rate']:.1%}")
            print(f"     - 出现频次: {driver['freq']} 次")
            print(f"     - 业务解读: {driver.get('insight', 'N/A')}")
    else:
        print("\n⚠️  没有发现成功驱动因子。")
    
    # 解析失败驱动因子
    failure_drivers = attribution.get('failure_drivers', [])
    if failure_drivers:
        print("\n❌ 失败驱动因子 (Risk 特征 - 降低胜率):")
        print("-" * 80)
        for idx, driver in enumerate(failure_drivers, 1):
            print(f"  {idx}. {driver['state']}")
            print(f"     - 条件胜率: {driver['win_rate']:.1%}")
            print(f"     - 出现频次: {driver['freq']} 次")
            print(f"     - 业务解读: {driver.get('insight', 'N/A')}")
    else:
        print("\n⚠️  没有发现失败驱动因子。")
    
    # 解析背景共识
    common_context = attribution.get('common_context', [])
    if common_context:
        print("\n🌍 背景共识 (Common Context - 普遍特征):")
        print("-" * 80)
        for idx, ctx in enumerate(common_context, 1):
            print(f"  {idx}. {ctx['state']}")
            print(f"     - 普及度: {ctx['prevalence']:.1%}")
            print(f"     - 业务解读: {ctx.get('insight', 'N/A')}")
    else:
        print("\n⚠️  没有发现背景共识特征。")
    
    # 打印分析摘要
    print("\n" + "="*80)
    print("【分析摘要 (Summary)】")
    print("="*80)
    
    print(f"✓ 已分析 {len(success_drivers)} 个成功因子")
    print(f"✓ 已识别 {len(failure_drivers)} 个失败因子")
    print(f"✓ 已提取 {len(common_context)} 个背景共识特征")
    
    if success_drivers or failure_drivers:
        print("\n📊 交易策略建议:")
        if success_drivers:
            success_states = [d['state'] for d in success_drivers]
            print(f"   ✓ 在这些条件出现时进行交易: {', '.join(success_states)}")
        if failure_drivers:
            failure_states = [d['state'] for d in failure_drivers]
            print(f"   ✗ 避免在这些条件下交易: {', '.join(failure_states)}")


def test_query_similar_events(manager: QGAGraphManager, event_ids: List[str]) -> None:
    """执行查询测试"""
    print("\n" + "="*80)
    print("【查询测试开始】")
    print("="*80)
    print(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"查询的 Event IDs: {event_ids}")
    
    try:
        # 执行查询
        result = manager.query_similar_events_insight(event_ids)
        
        # 输出原始结果
        pretty_print_raw_result(result)
        
        # 解析和解释结果
        parse_and_interpret_result(result)
        
        print("\n✅ 查询测试完成！")
        
    except Exception as e:
        print(f"\n❌ 查询测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("QGA 图数据库查询功能测试套件")
    print("="*80)
    print(f"测试启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置数据库连接
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = os.getenv("NEO4J_PASSWORD", "1018418086")
    
    print(f"\n🔗 正在连接到数据库...")
    print(f"   URI: {URI}")
    print(f"   USER: {USER}")
    
    manager = QGAGraphManager(URI, USER, PASSWORD)
    
    try:
        # =========================================================
        # 测试用例 1: 查询特定的相似 Event IDs
        # =========================================================
        test_event_ids = ["20250714_000000_P", "20250725_000000_P","20250709_160000_P"]
        
        print("\n" + "-"*80)
        print("【测试用例 1】查询两个相似事件的图谱分析")
        print("-"*80)
        test_query_similar_events(manager, test_event_ids)
        
        # =========================================================
        # 测试用例 2 (可选): 如果需要测试更多 Event IDs
        # =========================================================
        # 取消下面的注释以测试更多数据
        # extended_ids = test_event_ids + ["20250715_000000_P", "20250720_000000_P"]
        # print("\n" + "-"*80)
        # print("【测试用例 2】查询四个相似事件的图谱分析")
        # print("-"*80)
        # test_query_similar_events(manager, extended_ids)
        
        print("\n" + "="*80)
        print("✅ 所有测试完成")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 主测试流程失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.close()
        print("\n🔌 数据库连接已关闭")


if __name__ == "__main__":
    main()
