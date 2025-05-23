"""日志记录器工具"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

from ..config import get_settings


def setup_logger(log_file: Optional[str] = None, log_level: str = "INFO") -> None:
    """设置日志记录器"""
    settings = get_settings()
    
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # 文件输出格式
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件处理器
    log_file_path = log_file or settings.log_file
    if log_file_path:
        # 确保日志目录存在
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file_path,
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
    
    # 添加错误日志文件
    error_log_path = log_file_path.replace(".log", "_error.log") if log_file_path else "logs/error.log"
    Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        error_log_path,
        format=file_format,
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8"
    )
    
    logger.info(f"日志记录器初始化完成，级别: {log_level}")
    if log_file_path:
        logger.info(f"日志文件: {log_file_path}")


def get_logger(name: Optional[str] = None):
    """获取日志记录器实例"""
    if name:
        return logger.bind(name=name)
    return logger


class LoggerMixin:
    """日志记录器混入类"""
    
    @property
    def logger(self):
        """获取类专用的日志记录器"""
        return get_logger(self.__class__.__name__)


# 默认初始化
def init_default_logger():
    """初始化默认日志记录器"""
    settings = get_settings()
    setup_logger(
        log_file=settings.log_file,
        log_level=settings.log_level
    )


# 自动初始化
init_default_logger()