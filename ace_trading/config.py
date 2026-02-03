# ace_trading/config.py
import os

# Ollama Service Configuration (read from environment variable)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL_NAME = "qwen3-14b-q8-tool:latest"
ENABLE_THINKING = True

# ZhipuAI Configuration
ZHIPU_API_KEY = os.getenv("GLM_API")
ZHIPU_MODEL = "glm-4.5-flash"
ZHIPU_URL = os.getenv("ZHIPU_URL", "https://open.bigmodel.cn/api/paas/v4/")

# QWEN Model Configuration
QWEN_MODEL = "Qwen3-Next-80B-A3B-Instruct"
QWEN_API = os.getenv("QWEN_API")
QWEN_URL = os.getenv("QWEN_URL")  # instruct URL


# Trading/Backtest Default Configuration
DEFAULT_LEVERAGE = 1.0
DEFAULT_COMMISSION = 0.001
DEFAULT_CASH = 100000.0
