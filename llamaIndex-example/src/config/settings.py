"""系统配置管理"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings(BaseSettings):
    """系统配置类"""
    
    # Ollama配置
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    embedding_model: str = Field(default="bge-m3", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="deepseek-coder:r1", env="LLM_MODEL")
    
    # Milvus配置
    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_collection_name: str = Field(default="device_alerts", env="MILVUS_COLLECTION_NAME")
    
    # API配置
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    # 告警阈值配置
    alert_threshold_battery: int = Field(default=20, env="ALERT_THRESHOLD_BATTERY")
    alert_threshold_usage: int = Field(default=80, env="ALERT_THRESHOLD_USAGE")
    alert_threshold_vibration: float = Field(default=5.0, env="ALERT_THRESHOLD_VIBRATION")
    
    # 数据存储路径
    data_path: str = Field(default="data/", env="DATA_PATH")
    image_path: str = Field(default="data/images/", env="IMAGE_PATH")
    log_path: str = Field(default="logs/", env="LOG_PATH")
    
    # 向量维度配置
    embedding_dimension: int = Field(default=1024, env="EMBEDDING_DIMENSION")
    
    # 检索配置
    top_k: int = Field(default=5, env="TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.data_path,
            self.image_path,
            self.log_path,
            os.path.dirname(self.log_file)
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    @property
    def milvus_uri(self) -> str:
        """获取Milvus连接URI"""
        return f"http://{self.milvus_host}:{self.milvus_port}"
    
    @property
    def ollama_embedding_url(self) -> str:
        """获取Ollama嵌入模型URL"""
        return f"{self.ollama_base_url}/api/embeddings"
    
    @property
    def ollama_chat_url(self) -> str:
        """获取Ollama聊天模型URL"""
        return f"{self.ollama_base_url}/api/chat"


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取配置实例（单例模式）"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """重新加载配置"""
    global _settings
    _settings = None
    return get_settings()