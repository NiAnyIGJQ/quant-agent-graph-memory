"""
Token消耗统计日志系统
记录每次回测的token使用情况

使用方法：
    from ace_trading.token_logger import TokenLogger

    token_logger = TokenLogger(log_dir="logs")
    token_logger.log_token_usage(agent_name="A", input_tokens=1500, output_tokens=300)
    token_logger.generate_summary()
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from collections import defaultdict


class TokenLogger:
    """
    专门记录token消耗的日志系统

    每次回测生成以下文件：
    - token_usage.csv: 详细的token使用记录（每次调用）
    - token_summary.json: 汇总统计信息
    - token_report.txt: 可读的统计报告
    """

    def __init__(self, session_dir: Optional[str] = None, log_dir: str = "logs"):
        """
        初始化Token日志系统

        Args:
            session_dir: 指定会话目录（如果已有），否则自动创建
            log_dir: 日志根目录（默认 "logs"）
        """
        if session_dir and os.path.exists(session_dir):
            # 使用现有的session_dir（由prompt_logger创建）
            self.session_dir = session_dir
        else:
            # 创建新的session_dir
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_dir = os.path.join(log_dir, timestamp)
            Path(self.session_dir).mkdir(parents=True, exist_ok=True)

        # 初始化日志文件路径
        self.csv_file = os.path.join(self.session_dir, "token_usage.csv")
        self.summary_file = os.path.join(self.session_dir, "token_summary.json")
        self.report_file = os.path.join(self.session_dir, "token_report.txt")

        # 统计数据（内存中累计）
        self.stats = defaultdict(lambda: {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        })

        # 全局统计
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

        # 创建CSV文件并写入表头
        self._init_csv()

        print(f"[TOKEN_LOG] Token统计已启用")
        print(f"[TOKEN_LOG] 保存到: {os.path.abspath(self.session_dir)}\n")

    def _init_csv(self):
        """初始化CSV文件"""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'bar_number',
                'agent_name',
                'input_tokens',
                'output_tokens',
                'total_tokens',
                'model',
                'notes'
            ])

    def log_token_usage(self,
                       agent_name: str,
                       input_tokens: int,
                       output_tokens: int,
                       bar_number: Optional[int] = None,
                       model: str = "qwen",
                       notes: str = ""):
        """
        记录单次token使用

        Args:
            agent_name: Agent名称（"A", "B", "S", "C"）
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            bar_number: K线编号（可选）
            model: 模型名称（默认"qwen"）
            notes: 备注信息（可选）
        """
        timestamp = datetime.now().isoformat()
        total_tokens = input_tokens + output_tokens

        # 写入CSV
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                bar_number if bar_number is not None else '',
                agent_name,
                input_tokens,
                output_tokens,
                total_tokens,
                model,
                notes
            ])

        # 更新统计数据
        self.stats[agent_name]['calls'] += 1
        self.stats[agent_name]['input_tokens'] += input_tokens
        self.stats[agent_name]['output_tokens'] += output_tokens
        self.stats[agent_name]['total_tokens'] += total_tokens

        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

    def estimate_tokens_from_text(self, text: str) -> int:
        """
        估算文本的token数量（近似值）

        规则：
        - 英文：约0.75个token/词
        - 中文：约1.5个token/字符
        - 混合文本：使用简化估算

        Args:
            text: 待估算的文本

        Returns:
            估算的token数量
        """
        if not text:
            return 0

        # 统计中文字符数
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

        # 统计英文单词数（简化：按空格分割）
        english_words = len(text.split())

        # 估算token数
        # 中文：1.5 token/字符
        # 英文：0.75 token/词
        estimated_tokens = int(chinese_chars * 1.5 + english_words * 0.75)

        return estimated_tokens

    def log_from_prompt_and_output(self,
                                   agent_name: str,
                                   prompt_text: str,
                                   output_text: str,
                                   bar_number: Optional[int] = None,
                                   model: str = "qwen"):
        """
        从提示词和输出文本估算并记录token使用

        Args:
            agent_name: Agent名称
            prompt_text: 输入提示词文本
            output_text: 输出文本
            bar_number: K线编号（可选）
            model: 模型名称
        """
        input_tokens = self.estimate_tokens_from_text(prompt_text)
        output_tokens = self.estimate_tokens_from_text(output_text)

        self.log_token_usage(
            agent_name=agent_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            bar_number=bar_number,
            model=model,
            notes="Estimated from text"
        )

    def generate_summary(self) -> Dict[str, Any]:
        """
        生成统计摘要并保存

        Returns:
            统计摘要字典
        """
        summary = {
            'session_dir': self.session_dir,
            'timestamp': datetime.now().isoformat(),
            'total_stats': {
                'total_calls': self.total_calls,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'total_tokens': self.total_tokens
            },
            'agent_stats': dict(self.stats)
        }

        # 保存JSON摘要
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 生成可读报告
        self._generate_report(summary)

        return summary

    def _generate_report(self, summary: Dict[str, Any]):
        """生成可读的统计报告"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Token消耗统计报告")
        report_lines.append("="*80)
        report_lines.append(f"回测时间: {summary['timestamp']}")
        report_lines.append(f"会话目录: {summary['session_dir']}")
        report_lines.append("")

        # 总体统计
        total = summary['total_stats']
        report_lines.append("【总体统计】")
        report_lines.append(f"  总调用次数: {total['total_calls']:,}")
        report_lines.append(f"  总输入Token: {total['total_input_tokens']:,}")
        report_lines.append(f"  总输出Token: {total['total_output_tokens']:,}")
        report_lines.append(f"  总Token消耗: {total['total_tokens']:,}")
        report_lines.append("")

        # 按Agent统计
        report_lines.append("【分Agent统计】")
        for agent_name, stats in sorted(summary['agent_stats'].items()):
            report_lines.append(f"\n  Agent {agent_name}:")
            report_lines.append(f"    调用次数: {stats['calls']:,}")
            report_lines.append(f"    输入Token: {stats['input_tokens']:,}")
            report_lines.append(f"    输出Token: {stats['output_tokens']:,}")
            report_lines.append(f"    总Token: {stats['total_tokens']:,}")

            # 计算平均值
            if stats['calls'] > 0:
                avg_input = stats['input_tokens'] / stats['calls']
                avg_output = stats['output_tokens'] / stats['calls']
                avg_total = stats['total_tokens'] / stats['calls']
                report_lines.append(f"    平均每次: 输入={avg_input:.0f}, 输出={avg_output:.0f}, 总计={avg_total:.0f}")

            # 计算占比
            if total['total_tokens'] > 0:
                pct = (stats['total_tokens'] / total['total_tokens']) * 100
                report_lines.append(f"    占比: {pct:.1f}%")

        report_lines.append("")
        report_lines.append("="*80)

        # 写入报告文件
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n[TOKEN_REPORT] 统计报告已生成: {self.report_file}")

    def print_summary(self):
        """打印统计摘要到控制台"""
        summary = self.generate_summary()

        print("\n" + "="*80)
        print("Token消耗统计")
        print("="*80)

        total = summary['total_stats']
        print(f"总调用次数: {total['total_calls']:,}")
        print(f"总Token消耗: {total['total_tokens']:,}")
        print(f"  - 输入: {total['total_input_tokens']:,}")
        print(f"  - 输出: {total['total_output_tokens']:,}")
        print("")

        print("分Agent统计:")
        for agent_name, stats in sorted(summary['agent_stats'].items()):
            pct = (stats['total_tokens'] / total['total_tokens'] * 100) if total['total_tokens'] > 0 else 0
            print(f"  Agent {agent_name}: {stats['total_tokens']:,} tokens ({pct:.1f}%) - {stats['calls']} 次调用")

        print("="*80 + "\n")


# 全局单例
_token_logger: Optional[TokenLogger] = None


def get_token_logger() -> Optional[TokenLogger]:
    """获取全局的TokenLogger实例"""
    return _token_logger


def init_token_logger(session_dir: Optional[str] = None, log_dir: str = "logs") -> TokenLogger:
    """
    初始化全局的TokenLogger实例

    Args:
        session_dir: 指定会话目录（通常与prompt_logger共享）
        log_dir: 日志根目录
    """
    global _token_logger
    _token_logger = TokenLogger(session_dir=session_dir, log_dir=log_dir)
    return _token_logger


if __name__ == '__main__':
    # 测试示例
    print("测试Token Logger...")

    token_logger = TokenLogger()

    # 模拟记录一些token使用
    token_logger.log_token_usage(
        agent_name="A",
        input_tokens=1500,
        output_tokens=300,
        bar_number=1,
        model="qwen-max",
        notes="Deep analysis"
    )

    token_logger.log_token_usage(
        agent_name="B",
        input_tokens=800,
        output_tokens=150,
        bar_number=1,
        model="qwen-max",
        notes="Decision making"
    )

    token_logger.log_token_usage(
        agent_name="S",
        input_tokens=1200,
        output_tokens=200,
        bar_number=1,
        model="qwen-max",
        notes="State classification"
    )

    # 测试文本估算
    test_text = """
    这是一段测试文本，包含中文和English words。
    We want to estimate the token count for mixed language text.
    目标是提供一个合理的近似值。
    """
    estimated = token_logger.estimate_tokens_from_text(test_text)
    print(f"\n估算token数: {estimated}")

    # 生成摘要
    token_logger.print_summary()

    print(f"\n✓ 测试完成，日志保存到: {token_logger.session_dir}")
