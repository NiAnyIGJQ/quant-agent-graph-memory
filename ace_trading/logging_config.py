#!/usr/bin/env python3
"""
Logging 配置模块
将所有 print 和 logger 输出同时保存到文件和控制台

使用方法：
    from ace_trading.logging_config import setup_logging

    log = setup_logging(
        log_file="backtest_2025_12_21.log",
        log_dir="logs"
    )

    log.info("This will be saved to file and printed to console")
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import sys


class DualStreamHandler(logging.Handler):
    """
    自定义处理器：既输出到控制台，也输出到文件
    """
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def emit(self, record):
        try:
            msg = self.format(record)
            # 写入文件
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
            # 同时输出到控制台
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_file=None, log_dir="logs", level=logging.INFO):
    """
    设置 logging 配置

    Args:
        log_file: 日志文件名 (如 "backtest_2025_12_21.log")
                 如果为 None，会自动生成 "backtest_YYYYMMDD_HHMMSS.log"
        log_dir: 日志目录 (默认 "logs")
        level: 日志级别 (默认 INFO)

    Returns:
        logger 实例，可用于 logger.info(), logger.error() 等
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 生成日志文件路径
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"backtest_{timestamp}.log"

    log_path = os.path.join(log_dir, log_file)

    # 清空现有日志文件内容（如果文件已存在）
    # 注释掉这行可以追加而不是覆盖
    # with open(log_path, 'w', encoding='utf-8'):
    #     pass

    # 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有的处理器，避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 添加文件处理器（同时写入文件和控制台）
    file_handler = DualStreamHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 打印日志文件路径信息
    startup_msg = f"\n{'='*80}\n[LOGGING] 日志已启用，保存到: {os.path.abspath(log_path)}\n{'='*80}\n"
    print(startup_msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(startup_msg)

    return root_logger


def redirect_print_to_logging():
    """
    将 print() 的输出重定向到 logging
    （可选：如果想让现有的 print() 也进入日志系统）

    使用方法：
        redirect_print_to_logging()
        print("This will be logged")  # 这会进入日志
    """
    logger = logging.getLogger("print")

    class PrintToLoggerWrapper:
        def __init__(self, logger_instance, level=logging.INFO):
            self.logger = logger_instance
            self.level = level

        def write(self, message):
            if message.strip():  # 忽略空行
                self.logger.log(self.level, message.rstrip())

        def flush(self):
            pass

    sys.stdout = PrintToLoggerWrapper(logger)


if __name__ == '__main__':
    # 测试示例
    log = setup_logging()
    log.info("这是一条信息日志")
    log.warning("这是一条警告日志")
    log.error("这是一条错误日志")
    print("[TEST] 直接使用 print() 的输出")
