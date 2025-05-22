from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class DeviceType(Enum):
    CAMERA = "camera"
    SENSOR = "sensor"
    MACHINE = "machine"
    ROBOT = "robot"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DeviceMetrics:
    battery_level: float  # 电量百分比
    usage_count: int     # 使用次数
    vibration_level: float  # 振动级别
    temperature: float   # 温度
    humidity: float      # 湿度
    timestamp: datetime

@dataclass
class DeviceAlert:
    device_id: str
    alert_type: str
    level: AlertLevel
    message: str
    metrics: DeviceMetrics
    timestamp: datetime
    image_path: Optional[str] = None

@dataclass
class Device:
    id: str
    name: str
    type: DeviceType
    location: str
    status: str
    metrics: DeviceMetrics
    alerts: List[DeviceAlert]
    metadata: Dict[str, Any] 