import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase, basic_auth

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QGAGraphManager:
    """
    QGA-Pattern 3.0 图数据库管理器
    核心职责：
    1. 维护三角闭环拓扑结构 (Triangle Topology)。
    2. 实现 Pattern 的自动哈希与归纳。
    3. 提供量化归因与回测查询接口。
    """

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connected to Neo4j Database.")

    def close(self):
        self.driver.close()
        logger.info("Connection closed.")

    def _generate_pattern_hash(self, state_list: List[str]) -> str:
        """
        生成 Pattern 的唯一哈希值。

        规则：
        1. 对 State 名称列表进行排序（确保无序的状态列表也能产生同样的Hash）
        2. 拼接后取 MD5
        3. 确保 [S1, S2] 和 [S2, S1] 指向同一个 Pattern

        示例：
        - input: ['RSI_Oversold', 'Support_Hit']
        - sorted: ['RSI_Oversold', 'Support_Hit']  (已按字典序排序)
        - combined: 'RSI_Oversold_Support_Hit'
        - hash: md5(combined)

        - input: ['Support_Hit', 'RSI_Oversold']  (顺序不同)
        - sorted: ['RSI_Oversold', 'Support_Hit']  (排序结果相同!)
        - combined: 'RSI_Oversold_Support_Hit'
        - hash: md5(combined)  (得到相同的Hash!)
        """
        if not state_list:
            raise ValueError("Pattern definition cannot be empty.")

        # 【关键】排序确保顺序一致性
        sorted_states = sorted(state_list)
        combined_str = "_".join(sorted_states)
        hash_value = hashlib.md5(combined_str.encode('utf-8')).hexdigest()

        # 【调试日志】记录Hash生成过程
        logger.info(f"[HASH_GENERATION] Input states: {state_list}")
        logger.info(f"[HASH_GENERATION] Sorted states: {sorted_states}")
        logger.info(f"[HASH_GENERATION] Combined string: {combined_str}")
        logger.info(f"[HASH_GENERATION] Generated Hash: {hash_value}")

        return hash_value

    # ==========================================
    # 1. 初始化部分 (Schema & Constraints)
    # ==========================================
    
    def initialize_schema(self):
        """
        初始化数据库 Schema，建立强约束和索引。
        对应文档 4. 约束与索引 部分。
        """
        constraints = [
            # 唯一性约束
            "CREATE CONSTRAINT constraint_state_unique IF NOT EXISTS FOR (s:State) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT constraint_pattern_unique IF NOT EXISTS FOR (p:Pattern) REQUIRE p.pattern_hash IS UNIQUE",
            "CREATE CONSTRAINT constraint_decision_unique IF NOT EXISTS FOR (d:Decision) REQUIRE d.action IS UNIQUE",
            "CREATE CONSTRAINT constraint_outcome_unique IF NOT EXISTS FOR (o:Outcome) REQUIRE o.class IS UNIQUE",
            "CREATE CONSTRAINT constraint_event_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
        ]

        indexes = [
            # 性能索引
            "CREATE INDEX index_event_timestamp IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
            "CREATE INDEX index_pattern_weight IF NOT EXISTS FOR (p:Pattern) ON (p.weight)",
        ]

        with self.driver.session() as session:
            # 1. 执行约束
            for query in constraints:
                try:
                    session.run(query)
                    logger.info(f"Constraint executed: {query}")
                except Exception as e:
                    logger.warning(f"Constraint execution failed (might exist): {e}")

            # 2. 执行索引
            for query in indexes:
                try:
                    session.run(query)
                    logger.info(f"Index executed: {query}")
                except Exception as e:
                    logger.warning(f"Index execution failed: {e}")
            
            # 3. 预置基础节点 (可选)
            # 预先创建 Decision 和 Outcome 枚举，确保图谱基础完整
            self._seed_enums(session)

    def _seed_enums(self, session):
        """预置 Decision 和 Outcome 枚举节点"""
        session.run("MERGE (:Decision {action: 'LONG',name : 'LONG'})")
        session.run("MERGE (:Decision {action: 'SHORT',name : 'SHORT'})")
        session.run("MERGE (:Decision {action: 'NEUTRAL',name : 'NEUTRAL'})")

        session.run("MERGE (:Outcome {class: 'WIN',name : 'WIN'})")
        session.run("MERGE (:Outcome {class: 'LOSS',name : 'LOSS'})")

        # 可选：增加一个 BREAK_EVEN (保本) 节点，用于过滤噪音
        session.run("MERGE (:Outcome {class: 'BREAK_EVEN', name: 'BREAK_EVEN'})")

        logger.info("Seeded Decision and Outcome enums.")

    # ==========================================
    # 2. 写入部分 (Slow Stream Write)
    # ==========================================

    def insert_trade_reflection(self,
                                event_data: Dict[str, Any],
                                core_logic_states: List[str],
                                all_context_states: List[str],
                                agent_insight: Dict[str, str],
                                decision: str,
                                outcome: str,
                                match_confidence:int=1
                                ):
        """
        【改进】写入交易反思

        关键改动：
        - core_logic_states 应该已在调用端排序，但这里再做一次确保（幂等操作）
        - core_logic_states 现在接受任意长度（由Agent S动态确定）
        - all_context_states 包含所有环境信息（仍可较长）
        """

        # 【防守】再次确保core_logic_states已排序
        # 因为调用端已经排序，这里相当于幂等操作，确保Hash稳定
        core_logic_states = sorted(core_logic_states)
        logger.info(f"[INSERT_ENSURE_SORTED] Final core_logic_states (sorted): {core_logic_states}")

        # 1. 验证核心状态不为空
        if len(core_logic_states) == 0:
            logger.warning(f"[INSERT_WARN] core_logic_states is empty, skipping insertion")
            raise ValueError(f"core_logic_states cannot be empty")

        # 2. 计算 Pattern 哈希（【基于排序后的core_logic_states】）
        pattern_hash = self._generate_pattern_hash(core_logic_states)
        
        cypher_query = """
        // --- 1. 确保枚举节点存在 ---
        MERGE (d:Decision {action: $decision})
        MERGE (o:Outcome {class: $outcome})

        // --- 2. 处理 Pattern (半公有逻辑枢纽) ---
        MERGE (p:Pattern {pattern_hash: $pattern_hash})
        ON CREATE SET 
            p.name = $pattern_name,
            p.description = $pattern_desc,
            p.weight = 1.0,
            p.created_at = datetime()
        ON MATCH SET
            // 【修复】根据 confidence 加权更新，而不是固定的 0.1
            p.weight = p.weight + $conf

        // --- 3. 链接 Pattern 定义 (COMPOSED_OF) ---
        // [关键修正点开始]
        WITH p, d, o
        UNWIND $core_states AS state_name
        MERGE (s:State {name: state_name})
        MERGE (p)-[:COMPOSED_OF]->(s)
        
        // [关键修正操作]: UNWIND 后数据变成了多行，必须在这里聚合回一行！
        // 我们用 collect(s) 把刚才拆散的状态再收集起来，虽然不需要用它，但能把行数重置为 1
        WITH p, d, o, collect(s) as _ignored_states

        // --- 4. 链接 Pattern 建议 (SUGGESTS) ---
        MERGE (p)-[:SUGGESTS]->(d)

        // --- 5. 创建 Event 节点 (核心锚点) ---
        // 此时执行流只有 1 行，所以 Event 只会被创建一次
        WITH p, d, o
        CREATE (e:Event {
            event_id: $event_id,
            timestamp: datetime($timestamp),
            duration: $duration,
            realized_pnl: $pnl
        })

        // --- 6. 建立三角闭环逻辑 ---
        
        // 路径 A: 逻辑归因
        MERGE (e)-[:MATCHES {confidence: $conf}]->(p)
        
        // 路径 B: 结果验证
        MERGE (e)-[:RESULTED_IN]->(o)
        
        // 路径 C: 全量观察 (HAS_CONTEXT)
        // 这里再次 UNWIND 是安全的，因为 e 已经创建好了，我们只是在 e 上加边
        WITH e
        UNWIND $all_context AS ctx_state_name
        MERGE (cs:State {name: ctx_state_name})
        MERGE (e)-[:HAS_CONTEXT]->(cs)


        // ==========================================
        // 5. [新增] 建立稀疏时间链条 (Time Chaining)
        // ==========================================
        // 重新聚合 e，确保上下文处理完毕后只有一行
        WITH DISTINCT e
        
        // 查找最近的一个【过去】Event
        MATCH (last_e:Event)
        WHERE last_e.timestamp < e.timestamp
        WITH e, last_e
        ORDER BY last_e.timestamp DESC
        LIMIT 1
        
        // 建立链条并计算间隔
        MERGE (last_e)-[r:NEXT_REFLECTION]->(e)
        SET r.interval_minutes = duration.inSeconds(last_e.timestamp, e.timestamp).minutes
        """

        with self.driver.session() as session:
            session.run(cypher_query, 
                        decision=decision,
                        outcome=outcome,
                        pattern_hash=pattern_hash,
                        pattern_name=agent_insight.get("pattern_name", "Unknown_Pattern"),
                        pattern_desc=agent_insight.get("description", ""),
                        core_states=core_logic_states,
                        event_id=event_data["event_id"],
                        timestamp=event_data["timestamp"],
                        duration=event_data["duration"],
                        pnl=event_data["realized_pnl"],
                        all_context=all_context_states,
                        conf=match_confidence
            )
            logger.info(f"Trade Event {event_data['event_id']} inserted successfully with Pattern {pattern_hash}.")


    # ==========================================
    # 3. 读取部分 (Fast Stream Query - Advanced Attribution)
    # ==========================================

    def query_similar_events_insight(self, similar_event_ids: List[str]) -> Dict[str, Any]:
        """
        图谱归因推理接口 (Deep Attribution & Context Analysis)。

        根据向量库返回的相似历史 Event ID，进行三维度的图谱分析：
        1. 【宏观模式】(Macro): 识别主导 Pattern 及其核心定义（由哪些 State 组成）。
        2. 【背景共识】(Context): 识别这些相似行情共同所处的“大环境”特征（土壤）。
        3. 【微观归因】(Micro): 分析导致盈亏差异的“胜负手”特征（Alpha/Risk）。

        Args:
            similar_event_ids (List[str]): 
                由向量数据库 (LSTM/Transformer) 检索出的 Top-K 相似 Event ID 列表。
                例如: ["20231201_BTC", "20231105_ETH", ...]

        Returns:
            Dict[str, Any]: 结构化的分析报告，包含以下字段：
            
            - dominant_pattern (Dict | None): 主导模式信息
                - name (str): 模式名称 (e.g., "RSI_Div_Vol_Low")
                - description (str): 模式描述
                - definition (List[str]): [新增] 该模式由哪些核心 State 组成。
                                          (e.g., ["RSI_High", "Vol_Low"])
                                          这代表了“为什么”被归类为这个模式。
                - action (str): 建议动作 (LONG/SHORT/NEUTRAL)
                - baseline_win_rate (float): 该模式在相似组中的基础胜率。
                - match_count (int): 相似组中有多少个 Event 属于该模式。
            
            - attribution_analysis (Dict): 差异化归因分析
                - common_context (List[Dict]): [新增] 背景共识 (土壤)。
                    - state (str): 状态名称
                    - prevalence (float): 普及度，即在相似组中出现的频率 (>60%)。
                    - insight (str): 业务解读
                - success_drivers (List[Dict]): 成功因子 (Alpha)。
                    - state (str): 状态名称
                    - win_rate (float): 该特征出现时的条件胜率 (显著高于基准)。
                    - freq (int): 出现频次
                - failure_drivers (List[Dict]): 致死因子 (Risk)。
                    - state (str): 状态名称
                    - win_rate (float): 该特征出现时的条件胜率 (显著低于基准)。
        """
        
        # =========================================================
        # 查询 1: 宏观 Pattern 分析 + 模式定义提取
        # =========================================================
        # [逻辑增强]: 增加了 MATCH (p)-[:COMPOSED_OF]->(def_s) 
        # 目的：不仅要名字，还要知道这个 Pattern 的“骨架”是什么，防止 Agent 幻觉。
        query_pattern = """
        MATCH (e:Event) WHERE e.event_id IN $ids
        MATCH (e)-[:MATCHES]->(p:Pattern)
        MATCH (e)-[:RESULTED_IN]->(o:Outcome)
        MATCH (p)-[:SUGGESTS]->(d:Decision)
        
        // 1. 聚合统计模式表现，找出主导模式
        WITH p, d, count(e) AS occurrence, 
             sum(CASE WHEN o.class = 'WIN' THEN 1 ELSE 0 END) * 1.0 / count(e) AS win_rate
        ORDER BY occurrence DESC LIMIT 1
        
        // 2. [关键]: 获取该主导 Pattern 的核心定义 States
        MATCH (p)-[:COMPOSED_OF]->(def_s:State)
        
        RETURN 
            p.name AS pattern_name,
            p.description AS description,
            d.action AS suggested_action,
            occurrence,
            win_rate,
            collect(def_s.name) AS pattern_definition // 返回定义列表
        """

        # =========================================================
        # 查询 2: 微观归因与背景分析 (Micro Attribution & Context)
        # =========================================================
        # [逻辑增强]: 计算 prevalence (普及度)，用于识别 Common Context
        query_attribution = """
        MATCH (e:Event) WHERE e.event_id IN $ids
        MATCH (e)-[:RESULTED_IN]->(o:Outcome)
        
        // 获取全量观察路径 (关键：利用三角结构的底边，查找所有相关的环境特征)
        MATCH (e)-[:HAS_CONTEXT]->(s:State)
        
        // 聚合统计每个 State 的表现
        WITH s.name AS state_name,
             count(e) AS total_freq,
             sum(CASE WHEN o.class = 'WIN' THEN 1 ELSE 0 END) AS win_count,
             sum(CASE WHEN o.class = 'LOSS' THEN 1 ELSE 0 END) AS loss_count,
             count(e) * 1.0 / size($ids) AS prevalence // [新增]: 计算该特征在当前群体中的普及度
             
        // 计算该特征下的特定胜率 (Conditional Win Rate)
        WITH state_name, total_freq, win_count, loss_count, prevalence,
             (win_count * 1.0 / total_freq) AS conditional_win_rate
        
        // 过滤：只看那些稍微有点代表性的特征 (比如至少出现过 2 次)
        WHERE total_freq >= 2
        
        RETURN 
            state_name,
            total_freq,
            win_count,
            loss_count,
            conditional_win_rate,
            prevalence
        ORDER BY conditional_win_rate DESC
        """
        result_payload = {}

        try:
            with self.driver.session() as session:
                # --- 第一步：执行 Pattern 分析 ---
                try:
                    res_p = session.run(query_pattern, ids=similar_event_ids).single()
                except Exception as e:
                    logger.error(f"Pattern query failed: {e}")
                    res_p = None

                baseline_win_rate = 0.0
                pattern_def_states = [] 

                if res_p is not None:
                    try:
                        baseline_win_rate = res_p.get('win_rate', 0.0)
                        pattern_def_states = res_p.get('pattern_definition', [])

                        result_payload['dominant_pattern'] = {
                            "name": res_p.get('pattern_name', 'Unknown'),
                            "description": res_p.get('description', ''),
                            "definition": pattern_def_states,
                            "action": res_p.get('suggested_action', 'NEUTRAL'),
                            "baseline_win_rate": round(baseline_win_rate, 2),
                            "match_count": res_p.get('occurrence', 0)
                        }
                    except Exception as e:
                        logger.error(f"Error processing pattern result: {e}")
                        result_payload['dominant_pattern'] = None
                else:
                    result_payload['dominant_pattern'] = None

                # --- 第二步：执行归因与背景分析 ---
                if result_payload['dominant_pattern']:
                    try:
                        res_attr = session.run(query_attribution, ids=similar_event_ids)

                        success_drivers = []
                        failure_drivers = []
                        common_context = []

                        # [优化1]: 动态阈值与频次
                        num_similar_events = len(similar_event_ids)
                        # 如果样本很少，只要降低 10% 就很严重；样本多则要求 15%
                        threshold = 0.10 if num_similar_events < 5 else 0.15
                        # 如果样本很少，哪怕只出现 1 次导致大亏，也要记录
                        min_freq_req = 1 if num_similar_events < 5 else 2

                        for record in res_attr:
                            s_name = record['state_name']
                            win_rate = record['conditional_win_rate']
                            freq = record['total_freq']
                            prevalence = record['prevalence']

                            # 过滤频次过低的噪音
                            if freq < min_freq_req:
                                continue

                            is_definition = s_name in pattern_def_states

                            # A. 识别背景共识 (Common Context)
                            if prevalence >= 0.6 and not is_definition:
                                common_context.append({
                                    "state": s_name,
                                    "prevalence": round(prevalence, 2),
                                    "insight": f"High probability context ({prevalence:.0%})"
                                })

                            # B. 识别差异驱动因子 (Drivers)
                            if not is_definition:
                                # Success factor: 相对基准显著提升
                                if win_rate > (baseline_win_rate + threshold):
                                    success_drivers.append({
                                        "state": s_name,
                                        "win_rate": round(win_rate, 2),
                                        "freq": freq,
                                        "insight": f"Alpha Driver: Boosts win rate to {win_rate:.0%}"
                                    })

                                # Failure factor: [核心优化逻辑]
                                # 满足以下任一条件即视为致死因子：
                                # 1. 相对逻辑：比基准胜率显著低 (原逻辑)
                                # 2. 绝对逻辑：只要出现，胜率就低于 33% (绝对烂，不管基准是多少)
                                is_worse_than_baseline = win_rate < (baseline_win_rate - threshold)
                                is_absolute_failure = win_rate < 0.33
                                
                                if is_worse_than_baseline or is_absolute_failure:
                                    failure_drivers.append({
                                        "state": s_name,
                                        "win_rate": round(win_rate, 2),
                                        "freq": freq,
                                        "insight": f"Risk Factor: Win rate drops to {win_rate:.0%}"
                                    })

                        result_payload['attribution_analysis'] = {
                            "success_drivers": success_drivers,
                            "failure_drivers": failure_drivers,
                            "common_context": common_context
                        }
                    except Exception as e:
                        logger.error(f"Attribution query failed: {e}")
                        result_payload['attribution_analysis'] = {}
                else:
                    result_payload['attribution_analysis'] = {}
        except Exception as e:
            logger.error(f"query_similar_events_insight failed: {e}")
            return {
                'dominant_pattern': None,
                'attribution_analysis': {}
            }

        return result_payload
        # result_payload = {}

        # try:
        #     with self.driver.session() as session:
        #         # --- 第一步：执行 Pattern 分析 ---
        #         try:
        #             res_p = session.run(query_pattern, ids=similar_event_ids).single()
        #         except Exception as e:
        #             logger.error(f"Pattern query failed: {e}")
        #             res_p = None

        #         baseline_win_rate = 0.0
        #         pattern_def_states = [] # 用于存储模式定义的 State，后续做排除用

        #         if res_p is not None:
        #             try:
        #                 baseline_win_rate = res_p.get('win_rate', 0.0)
        #                 pattern_def_states = res_p.get('pattern_definition', [])

        #                 result_payload['dominant_pattern'] = {
        #                     "name": res_p.get('pattern_name', 'Unknown'),
        #                     "description": res_p.get('description', ''),
        #                     "definition": pattern_def_states,
        #                     "action": res_p.get('suggested_action', 'NEUTRAL'),
        #                     "baseline_win_rate": round(baseline_win_rate, 2),
        #                     "match_count": res_p.get('occurrence', 0)
        #                 }
        #             except Exception as e:
        #                 logger.error(f"Error processing pattern result: {e}")
        #                 result_payload['dominant_pattern'] = None
        #         else:
        #             result_payload['dominant_pattern'] = None

        #         # --- 第二步：执行归因与背景分析 (仅当存在主导模式时) ---
        #         if result_payload['dominant_pattern']:
        #             try:
        #                 res_attr = session.run(query_attribution, ids=similar_event_ids)

        #                 success_drivers = []
        #                 failure_drivers = []
        #                 common_context = []

        #                 # Calculate adaptive threshold based on sample size
        #                 # For small sample sizes (< 5 events), use 10% threshold; for larger, use 15%
        #                 num_similar_events = len(similar_event_ids)
        #                 threshold = 0.10 if num_similar_events < 5 else 0.15

        #                 for record in res_attr:
        #                     s_name = record['state_name']
        #                     win_rate = record['conditional_win_rate']
        #                     freq = record['total_freq']
        #                     prevalence = record['prevalence']

        #                     # [辩证逻辑]:
        #                     # 如果一个 State 本身就是 Pattern 的定义部分（如 RSI_High），
        #                     # 那么它不是"导致差异"的原因，它是"基准"。我们跳过它，只寻找额外的变量。
        #                     is_definition = s_name in pattern_def_states

        #                     # A. 识别背景共识 (Common Context)
        #                     # 逻辑：无论输赢，大家都具备的特征 (>60%)，代表了行情的"土壤"。
        #                     if prevalence >= 0.6 and not is_definition:
        #                         common_context.append({
        #                             "state": s_name,
        #                             "prevalence": round(prevalence, 2),
        #                             "insight": f"Historical similar patterns: {prevalence:.0%} probability with [{s_name}]"
        #                         })

        #                     # B. 识别差异驱动因子 (Drivers)
        #                     if not is_definition:
        #                         # Success factor: win rate significantly higher than baseline
        #                         if win_rate > (baseline_win_rate + threshold):
        #                             success_drivers.append({
        #                                 "state": s_name,
        #                                 "win_rate": round(win_rate, 2),
        #                                 "freq": freq,
        #                                 "insight": f"When [{s_name}] appears, win rate increases to {win_rate:.0%}"
        #                             })

        #                         # Failure factor: win rate significantly lower than baseline
        #                         elif win_rate < (baseline_win_rate - threshold):
        #                             failure_drivers.append({
        #                                 "state": s_name,
        #                                 "win_rate": round(win_rate, 2),
        #                                 "freq": freq,
        #                                 "insight": f"When [{s_name}] appears, win rate drops to {win_rate:.0%}"
        #                             })

        #                 result_payload['attribution_analysis'] = {
        #                     "success_drivers": success_drivers, # 赢家独有
        #                     "failure_drivers": failure_drivers, # 输家独有
        #                     "common_context": common_context    # 大家共有
        #                 }
        #             except Exception as e:
        #                 logger.error(f"Attribution query failed: {e}")
        #                 result_payload['attribution_analysis'] = {}
        #         else:
        #             # 如果没有主导模式，归因分析也无从谈起
        #             result_payload['attribution_analysis'] = {}
        # except Exception as e:
        #     logger.error(f"query_similar_events_insight failed: {e}")
        #     return {
        #         'dominant_pattern': None,
        #         'attribution_analysis': {}
        #     }

        # return result_payload

    # ==========================================
    # 辅助工具: 清空数据库 (仅用于测试环境)
    # ==========================================
    def clear_database(self):
        """
        危险操作：清空当前数据库中的所有节点和关系。
        保留索引和约束。
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("⚠️ Database has been CLEARED (Nodes and Relationships removed).")

# ==========================================
# 4. 使用示例 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 配置你的数据库连接
    import os
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = os.getenv("NEO4J_PASSWORD", "1018418086")

    manager = QGAGraphManager(URI, USER, PASSWORD)

    try:
        # 0. 【新增】每次测试前清空数据，避免重复 ID 报错
        # 注意：在生产环境中绝对不要调用这个！
        print(">> Cleaning previous test data...")
        manager.clear_database()

        # 1. 初始化 Schema (如果约束已存在，它会跳过，所以这里很安全)
        print(">> Initializing Schema...")
        manager.initialize_schema()

        # ==========================================
        # 场景设定：
        # 我们交易一种 "RSI超卖反弹" (Oversold Rebound) 策略。
        # 假设：
        # - [成功案例] 通常伴随着 "Vol_Spike" (放量) 和 "Support_Level" (支撑位)。
        # - [失败案例] 通常发生在 "Friday_Afternoon" (周五下午) 或 "Low_Liquidity" (流动性差)。
        # ==========================================

        print("\n>> Simulating Batch Trade Insertions (Building History)...")
        
        # --- 案例 1: 成功的交易 (放量支撑) ---
        manager.insert_trade_reflection(
            event_data={
                "event_id": "EVT_WIN_001",
                "timestamp": datetime.now().isoformat(),
                "duration": 60, "realized_pnl": 200.0
            },
            core_logic_states=["RSI_Oversold", "Support_Level"], 
            all_context_states=["RSI_Oversold", "Support_Level", "Vol_Spike", "Bull_Sentiment"], 
            agent_insight={"pattern_name": "Std_Oversold_Rebound", "description": "标准超卖反弹"},
            decision="LONG", outcome="WIN"
        )

        # --- 案例 2: 成功的交易 (放量支撑，稍微不同的噪音) ---
        manager.insert_trade_reflection(
            event_data={
                "event_id": "EVT_WIN_002",
                "timestamp": datetime.now().isoformat(),
                "duration": 45, "realized_pnl": 180.0
            },
            core_logic_states=["RSI_Oversold", "Support_Level"], 
            all_context_states=["RSI_Oversold", "Support_Level", "Vol_Spike", "News_Neutral"], 
            agent_insight={"pattern_name": "Std_Oversold_Rebound", "description": "标准超卖反弹"},
            decision="LONG", outcome="WIN"
        )

        # --- 案例 3: 失败的交易 (周五下午) ---
        manager.insert_trade_reflection(
            event_data={
                "event_id": "EVT_LOSS_001",
                "timestamp": datetime.now().isoformat(),
                "duration": 120, "realized_pnl": -100.0
            },
            core_logic_states=["RSI_Oversold", "Support_Level"], 
            all_context_states=["RSI_Oversold", "Support_Level", "Friday_Afternoon", "Low_Liquidity"], 
            agent_insight={"pattern_name": "Std_Oversold_Rebound", "description": "标准超卖反弹"},
            decision="LONG", outcome="LOSS"
        )

        # --- 案例 4: 失败的交易 (又是周五) ---
        manager.insert_trade_reflection(
            event_data={
                "event_id": "EVT_LOSS_002",
                "timestamp": datetime.now().isoformat(),
                "duration": 90, "realized_pnl": -120.0
            },
            core_logic_states=["RSI_Oversold", "Support_Level"],
            all_context_states=["RSI_Oversold", "Support_Level", "Friday_Afternoon", "Choppy_Market"],
            agent_insight={"pattern_name": "Std_Oversold_Rebound", "description": "标准超卖反弹"},
            decision="LONG", outcome="LOSS"
        )
        
        print(">> Data Insertion Complete. Graph now holds logic about 'Vol_Spike' vs 'Friday_Afternoon'.")

        # ==========================================
        # 3. 模拟读取与归因推理
        # ==========================================
        print("\n>> Simulating Retrieval & Attribution Analysis...")
        
        # 假设向量数据库认为上述 4 个历史事件都与【当前行情】相似
        sim_ids = ["EVT_WIN_001", "EVT_WIN_002", "EVT_LOSS_001", "EVT_LOSS_002"]
        
        insight = manager.query_similar_events_insight(sim_ids)
        
        # --- 格式化输出结果 ---
        import json
        print(json.dumps(insight, indent=2, ensure_ascii=False))
        
        print("\n>> Analysis Interpretation:")
        pattern = insight.get('dominant_pattern')
        attr = insight.get('attribution_analysis', {})
        
        if pattern:
            print(f"1. 主导模式: {pattern['name']} (基准胜率: {pattern['baseline_win_rate']:.0%})")
            
            print("2. 成功关键因子 (Success Drivers):")
            for driver in attr.get('success_drivers', []):
                print(f"   - {driver['state']}: 胜率提升至 {driver['win_rate']:.0%} (出现频次: {driver['freq']})")
                
            print("3. 失败致死因子 (Failure Drivers):")
            for driver in attr.get('failure_drivers', []):
                print(f"   - {driver['state']}: 胜率降至 {driver['win_rate']:.0%} (出现频次: {driver['freq']})")

    except Exception as e:
        import traceback
        traceback.print_exc() # 打印完整错误堆栈以便调试
    finally:
        manager.close()