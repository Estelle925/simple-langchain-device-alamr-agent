import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Milvus配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = "device_alerts"

# Ollama配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "deepseek-coder"

# 向量维度
EMBEDDING_DIMENSION = 1024

# 告警阈值配置
ALERT_THRESHOLDS = {
    "battery_level": {
        "warning": 20.0,
        "critical": 10.0
    },
    "vibration_level": {
        "warning": 0.7,
        "critical": 0.9
    },
    "temperature": {
        "warning": 75.0,
        "critical": 85.0
    }
}

# 数据存储路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images") 