"""设备数据模型"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """设备类型枚举"""
    SENSOR = "sensor"  # 传感器
    CAMERA = "camera"  # 摄像头
    MOTOR = "motor"    # 电机
    PUMP = "pump"      # 泵
    VALVE = "valve"    # 阀门
    CONTROLLER = "controller"  # 控制器
    OTHER = "other"    # 其他


class DeviceStatus(str, Enum):
    """设备状态枚举"""
    ONLINE = "online"      # 在线
    OFFLINE = "offline"    # 离线
    FAULT = "fault"        # 故障
    MAINTENANCE = "maintenance"  # 维护中
    WARNING = "warning"    # 警告


class DeviceMetrics(BaseModel):
    """设备指标数据"""
    battery_level: Optional[float] = Field(None, description="电池电量百分比", ge=0, le=100)
    usage_rate: Optional[float] = Field(None, description="使用率百分比", ge=0, le=100)
    vibration_level: Optional[float] = Field(None, description="振动强度", ge=0)
    temperature: Optional[float] = Field(None, description="温度(摄氏度)")
    humidity: Optional[float] = Field(None, description="湿度百分比", ge=0, le=100)
    pressure: Optional[float] = Field(None, description="压力值")
    voltage: Optional[float] = Field(None, description="电压值")
    current: Optional[float] = Field(None, description="电流值")
    power: Optional[float] = Field(None, description="功率值")
    
    # 自定义指标
    custom_metrics: Dict[str, Any] = Field(default_factory=dict, description="自定义指标")


class DeviceLocation(BaseModel):
    """设备位置信息"""
    building: Optional[str] = Field(None, description="建筑物")
    floor: Optional[str] = Field(None, description="楼层")
    room: Optional[str] = Field(None, description="房间")
    zone: Optional[str] = Field(None, description="区域")
    coordinates: Optional[Dict[str, float]] = Field(None, description="坐标信息 {x, y, z}")
    address: Optional[str] = Field(None, description="详细地址")


class Device(BaseModel):
    """设备模型"""
    device_id: str = Field(..., description="设备唯一标识")
    name: str = Field(..., description="设备名称")
    device_type: DeviceType = Field(..., description="设备类型")
    model: Optional[str] = Field(None, description="设备型号")
    manufacturer: Optional[str] = Field(None, description="制造商")
    serial_number: Optional[str] = Field(None, description="序列号")
    
    # 状态信息
    status: DeviceStatus = Field(default=DeviceStatus.OFFLINE, description="设备状态")
    last_seen: Optional[datetime] = Field(None, description="最后在线时间")
    
    # 位置信息
    location: Optional[DeviceLocation] = Field(None, description="设备位置")
    
    # 指标数据
    metrics: DeviceMetrics = Field(default_factory=DeviceMetrics, description="设备指标")
    
    # 图片信息
    image_urls: List[str] = Field(default_factory=list, description="设备图片URL列表")
    
    # 描述信息
    description: Optional[str] = Field(None, description="设备描述")
    tags: List[str] = Field(default_factory=list, description="设备标签")
    
    # 配置信息
    config: Dict[str, Any] = Field(default_factory=dict, description="设备配置")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
    
    def update_status(self, status: DeviceStatus, metrics: Optional[DeviceMetrics] = None):
        """更新设备状态"""
        self.status = status
        self.last_seen = datetime.now()
        self.updated_at = datetime.now()
        
        if metrics:
            self.metrics = metrics
    
    def add_image(self, image_url: str):
        """添加设备图片"""
        if image_url not in self.image_urls:
            self.image_urls.append(image_url)
            self.updated_at = datetime.now()
    
    def update_metrics(self, **kwargs):
        """更新设备指标"""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
            else:
                self.metrics.custom_metrics[key] = value
        self.updated_at = datetime.now()
    
    def is_online(self) -> bool:
        """检查设备是否在线"""
        return self.status == DeviceStatus.ONLINE
    
    def is_healthy(self) -> bool:
        """检查设备是否健康"""
        return self.status in [DeviceStatus.ONLINE, DeviceStatus.WARNING]
    
    def get_alert_context(self) -> str:
        """获取用于告警分析的上下文信息"""
        context_parts = [
            f"设备ID: {self.device_id}",
            f"设备名称: {self.name}",
            f"设备类型: {self.device_type.value}",
            f"当前状态: {self.status.value}"
        ]
        
        if self.location:
            location_info = []
            if self.location.building:
                location_info.append(f"建筑: {self.location.building}")
            if self.location.floor:
                location_info.append(f"楼层: {self.location.floor}")
            if self.location.room:
                location_info.append(f"房间: {self.location.room}")
            if location_info:
                context_parts.append(f"位置: {', '.join(location_info)}")
        
        # 添加关键指标
        metrics_info = []
        if self.metrics.battery_level is not None:
            metrics_info.append(f"电池电量: {self.metrics.battery_level}%")
        if self.metrics.usage_rate is not None:
            metrics_info.append(f"使用率: {self.metrics.usage_rate}%")
        if self.metrics.vibration_level is not None:
            metrics_info.append(f"振动强度: {self.metrics.vibration_level}")
        if self.metrics.temperature is not None:
            metrics_info.append(f"温度: {self.metrics.temperature}°C")
        
        if metrics_info:
            context_parts.append(f"关键指标: {', '.join(metrics_info)}")
        
        if self.description:
            context_parts.append(f"描述: {self.description}")
        
        return "\n".join(context_parts)
    
    def to_vector_text(self) -> str:
        """转换为用于向量化的文本"""
        text_parts = [
            f"设备 {self.name} ({self.device_type.value})",
            f"状态: {self.status.value}"
        ]
        
        if self.description:
            text_parts.append(self.description)
        
        if self.tags:
            text_parts.append(f"标签: {', '.join(self.tags)}")
        
        # 添加位置信息
        if self.location:
            location_parts = []
            for field in ['building', 'floor', 'room', 'zone']:
                value = getattr(self.location, field)
                if value:
                    location_parts.append(value)
            if location_parts:
                text_parts.append(f"位置: {' '.join(location_parts)}")
        
        # 添加指标信息
        metrics_text = []
        if self.metrics.battery_level is not None:
            metrics_text.append(f"电池{self.metrics.battery_level}%")
        if self.metrics.usage_rate is not None:
            metrics_text.append(f"使用率{self.metrics.usage_rate}%")
        if self.metrics.vibration_level is not None:
            metrics_text.append(f"振动{self.metrics.vibration_level}")
        
        if metrics_text:
            text_parts.append(" ".join(metrics_text))
        
        return " ".join(text_parts)