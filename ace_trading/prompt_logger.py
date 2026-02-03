"""
Agent 提示词日志系统
记录每一轮回测中传入给每个 Agent 的完整提示词

使用方法：
    from ace_trading.prompt_logger import PromptLogger

    prompt_log = PromptLogger(log_dir="prompt_logs")
    prompt_log.log_agent_prompt("Agent A", bar_number, timestamp, prompt_text)
"""

import os
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Any
from pydantic import BaseModel


class PromptLogger:
    """
    专门记录 Agent 提示词的日志系统

    每次回测会生成一个目录，包含：
    - prompt_summary.log: 简要摘要（timestamp, bar, agent, prompt长度）
    - agent_a_prompts.log: Agent A 的完整提示词
    - agent_b_prompts.log: Agent B 的完整提示词
    - agent_s_prompts.log: Agent S 的完整提示词
    - agent_c_reflection.log: Agent C 的反思日志
    """

    def __init__(self, log_dir: str = "prompt_logs"):
        """
        初始化提示词日志系统

        Args:
            log_dir: 日志目录（默认 "prompt_logs"）
        """
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(log_dir, f"backtest_{self.timestamp}")

        # 创建会话目录
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)

        # 初始化各Agent的日志文件
        self.summary_file = os.path.join(self.session_dir, "prompt_summary.log")
        self.agent_a_file = os.path.join(self.session_dir, "agent_a_prompts.log")
        self.agent_b_file = os.path.join(self.session_dir, "agent_b_prompts.log")
        self.agent_s_file = os.path.join(self.session_dir, "agent_s_prompts.log")
        self.agent_c_file = os.path.join(self.session_dir, "agent_c_reflections.log")

        # 创建摘要头
        self._write_summary_header()

        print(f"\n[PROMPT_LOG] 提示词日志已启用")
        print(f"[PROMPT_LOG] 保存到: {os.path.abspath(self.session_dir)}\n")

    def _write_summary_header(self):
        """写入摘要文件的表头"""
        header = f"""
{'='*100}
Agent 提示词记录摘要
回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}

格式: [Timestamp] Bar #XXXXXX | Agent=X | Length=XXXXX chars

记录的文件:
- agent_a_prompts.log: Agent A 的完整提示词
- agent_b_prompts.log: Agent B 的完整提示词
- agent_s_prompts.log: Agent S 的完整提示词
- agent_c_reflections.log: Agent C 的交易反思

{'='*100}
"""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(header)

    def log_agent_prompt(self, agent_name: str, bar_number: int,
                        timestamp: str, prompt_text: str,
                        additional_info: Optional[dict] = None,
                        output_text: Optional[str] = None,
                        output_info: Optional[dict] = None,
                        output_object: Optional[Any] = None):
        """
        记录单个 Agent 的提示词和输出

        Args:
            agent_name: Agent 名称（"A", "B", "S", "C"）
            bar_number: Bar 编号
            timestamp: 时间戳字符串
            prompt_text: 完整的提示词文本
            additional_info: 额外信息（如输入数据摘要）
            output_text: Agent 的输出文本（可选）
            output_info: 输出相关信息（可选）
            output_object: Agent 的输出对象，将被序列化为 JSON（可选，优先级高于output_text）
        """
        # 确定目标文件
        if agent_name == "A":
            target_file = self.agent_a_file
        elif agent_name == "B":
            target_file = self.agent_b_file
        elif agent_name == "S":
            target_file = self.agent_s_file
        elif agent_name == "C":
            target_file = self.agent_c_file
        else:
            print(f"[PROMPT_LOG_ERROR] 未知的 Agent: {agent_name}")
            return

        # 写入完整提示词和输出
        separator = f"\n{'='*100}\n"
        header = f"[Bar #{bar_number:06d}] {timestamp}\n"

        if additional_info:
            info_str = f"【输入数据】\n"
            for key, value in additional_info.items():
                info_str += f"  - {key}: {value}\n"
            info_str += "\n"
        else:
            info_str = ""

        # 构建提示词部分
        prompt_section = f"{separator}{header}{info_str}【输入提示词】\n{prompt_text}\n"

        # 构建输出部分
        output_section = ""
        if output_object is not None or output_text:
            if output_info:
                output_info_str = f"【输出信息】\n"
                for key, value in output_info.items():
                    output_info_str += f"  - {key}: {value}\n"
                output_info_str += "\n"
            else:
                output_info_str = ""

            # 优先使用 output_object（序列化为JSON）
            if output_object is not None:
                try:
                    # 如果是 Pydantic BaseModel，使用 model_dump_json()
                    if isinstance(output_object, BaseModel):
                        # 【修复】Pydantic v2 不支持 ensure_ascii 参数，需要先转为dict再用json.dumps
                        output_dict = output_object.model_dump()
                        output_json = json.dumps(output_dict, indent=2, ensure_ascii=False, default=str)
                    else:
                        # 否则尝试普通的JSON序列化
                        output_json = json.dumps(output_object, indent=2, ensure_ascii=False, default=str)

                    output_content = f"{output_json}\n"
                    output_len = len(output_json)
                except Exception as e:
                    # 降级处理：如果JSON序列化失败，使用字符串表示
                    output_content = f"[JSON Serialization Error: {e}]\n{str(output_object)}\n"
                    output_len = len(output_content)
            else:
                output_content = output_text if output_text else ""
                output_len = len(output_content)

            output_section = f"\n【Agent输出】\n{output_info_str}{output_content}"

        content = prompt_section + output_section

        with open(target_file, 'a', encoding='utf-8') as f:
            f.write(content)

        # 写入摘要
        prompt_len = len(prompt_text)
        if output_object is not None:
            # 获取序列化后的长度
            try:
                if isinstance(output_object, BaseModel):
                    # 【修复】Pydantic v2 不支持 ensure_ascii 参数
                    output_dict = output_object.model_dump()
                    output_json = json.dumps(output_dict, ensure_ascii=False, default=str)
                else:
                    output_json = json.dumps(output_object, ensure_ascii=False, default=str)
                output_len = len(output_json)
            except:
                output_len = len(str(output_object))
        else:
            output_len = len(output_text) if output_text else 0

        summary_line = f"[{timestamp}] Bar #{bar_number:06d} | Agent {agent_name} | Prompt={prompt_len:,} chars | Output={output_len:,} chars\n"

        with open(self.summary_file, 'a', encoding='utf-8') as f:
            f.write(summary_line)

    def log_market_state(self, bar_number: int, timestamp: str,
                        state_info: dict):
        """
        记录市场状态快照（用于理解提示词的输入）

        Args:
            bar_number: Bar 编号
            timestamp: 时间戳
            state_info: 市场状态信息字典
        """
        state_file = os.path.join(self.session_dir, "market_states.log")

        content = f"""
{'='*100}
[Bar #{bar_number:06d}] {timestamp}
{'='*100}
当前价格: {state_info.get('cur_price', 'N/A')}
持仓量: {state_info.get('position', 0.0):.4f}
可用资金: {state_info.get('cash', 0.0):.2f}
当前TP: {state_info.get('current_tp', 0.0):.2f}
当前SL: {state_info.get('current_sl', 0.0):.2f}
Alpha: {state_info.get('roi_status', 'N/A')}

关键指标:
  1m RSI: {state_info.get('rsi_1m', 'N/A')}
  15m RSI: {state_info.get('rsi_15m', 'N/A')}
  1h RSI: {state_info.get('rsi_1h', 'N/A')}

  MACD 状态: {state_info.get('macd_status', 'N/A')}

市场状态标签: {state_info.get('market_states', [])}
"""
        with open(state_file, 'a', encoding='utf-8') as f:
            f.write(content + '\n')

    def get_session_dir(self) -> str:
        """获取本次回测的会话目录"""
        return self.session_dir

    def get_summary_file(self) -> str:
        """获取摘要文件路径"""
        return self.summary_file


# 全局单例，在 main.py 中使用
_prompt_logger: Optional[PromptLogger] = None


def get_prompt_logger() -> Optional[PromptLogger]:
    """获取全局的 PromptLogger 实例"""
    return _prompt_logger


def init_prompt_logger(log_dir: str = "prompt_logs") -> PromptLogger:
    """初始化全局的 PromptLogger 实例"""
    global _prompt_logger
    _prompt_logger = PromptLogger(log_dir=log_dir)
    return _prompt_logger


if __name__ == '__main__':
    # 测试示例
    prompt_log = PromptLogger()

    # 模拟 Agent A 的提示词
    test_prompt_a = """
[MARKET DATA - 1 MINUTE LEVEL]
Current Price: 82611.50

[1M TIMEFRAME - FOCUS ON RECENT 5 BARS]
Bar 1: 82631.0 (H: 82632.0, L: 82610.0, V: 150.23)
Bar 2: 82611.0 (H: 82625.0, L: 82605.0, V: 145.67)
Bar 3: 82600.0 (H: 82620.0, L: 82598.0, V: 142.15)
Bar 4: 82615.0 (H: 82625.0, L: 82600.0, V: 155.89)
Bar 5: 82620.0 (H: 82635.0, L: 82615.0, V: 160.42)

Indicators: RSI_6=45.67 MACD=0.15 HIST=0.08

[AGENT A - DEEP ANALYSIS]
Please perform unstructured, deep thinking...
"""

    prompt_log.log_agent_prompt(
        agent_name="A",
        bar_number=2,
        timestamp="2025-04-01T00:34:00",
        prompt_text=test_prompt_a,
        additional_info={
            "cur_price": 82611.50,
            "position": 0.0,
            "cash": 100000.0
        }
    )

    print(f"✓ 测试提示词已记录到: {prompt_log.get_session_dir()}")
