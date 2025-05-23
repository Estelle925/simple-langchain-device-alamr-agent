"""告警数据模型"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class AlertLevel(str, Enum):
    """告警级别枚举"""
    INFO = "info"          # 信息
    WARNING = "warning"    # 警告
    ERROR = "error"        # 错误
    CRITICAL = "critical"  # 严重
    FATAL = "fatal"        # 致命


class AlertType(str, Enum):
    """告警类型枚举"""
    DEVICE_OFFLINE = "device_offline"      # 设备离线
    BATTERY_LOW = "battery_low"            # 电池电量低
    USAGE_HIGH = "usage_high"              # 使用率过高
    VIBRATION_HIGH = "vibration_high"      # 振动过高
    TEMPERATURE_HIGH = "temperature_high"  # 温度过高
    TEMPERATURE_LOW = "temperature_low"    # 温度过低
    PRESSURE_HIGH = "pressure_high"        # 压力过高
    PRESSURE_LOW = "pressure_low"          # 压力过低
    DEVICE_FAULT = "device_fault"          # 设备故障
    MAINTENANCE_DUE = "maintenance_due"    # 维护到期
    CUSTOM = "custom"                      # 自定义告警


class AlertStatus(str, Enum):
    """告警状态枚举"""
    ACTIVE = "active"        # 活跃
    ACKNOWLEDGED = "acknowledged"  # 已确认
    RESOLVED = "resolved"    # 已解决
    SUPPRESSED = "suppressed"  # 已抑制


class AlertRule(BaseModel):
    """告警规则模型"""
    rule_id: str = Field(..., description="规则唯一标识")
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    
    # 规则条件
    alert_type: AlertType = Field(..., description="告警类型")
    device_types: List[str] = Field(default_factory=list, description="适用设备类型")
    
    # 阈值条件
    conditions: Dict[str, Any] = Field(default_factory=dict, description="告警条件")
    
    # 告警级别
    alert_level: AlertLevel = Field(default=AlertLevel.WARNING, description="告警级别")
    
    # 规则状态
    enabled: bool = Field(default=True, description="是否启用")
    
    # 抑制配置
    suppression_duration: Optional[int] = Field(None, description="抑制时长(秒)")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
    
    def matches_device(self, device_type: str) -> bool:
        """检查规则是否适用于指定设备类型"""
        return not self.device_types or device_type in self.device_types
    
    def evaluate_condition(self, metric_name: str, value: Any) -> bool:
        """评估条件是否满足"""
        if metric_name not in self.conditions:
            return False
        
        condition = self.conditions[metric_name]
        
        # 支持不同类型的条件
        if isinstance(condition, dict):
            # 范围条件
            if "min" in condition and value < condition["min"]:
                return True
            if "max" in condition and value > condition["max"]:
                return True
            if "equals" in condition and value == condition["equals"]:
                return True
        else:
            # 简单阈值条件
            return value > condition
        
        return False


class Alert(BaseModel):
    """告警模型"""
    alert_id: str = Field(..., description="告警唯一标识")
    device_id: str = Field(..., description="设备ID")
    rule_id: Optional[str] = Field(None, description="触发的规则ID")
    
    # 告警信息
    alert_type: AlertType = Field(..., description="告警类型")
    alert_level: AlertLevel = Field(..., description="告警级别")
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警消息")
    
    # 告警状态
    status: AlertStatus = Field(default=AlertStatus.ACTIVE, description="告警状态")
    
    # 触发数据
    trigger_value: Optional[Any] = Field(None, description="触发值")
    threshold_value: Optional[Any] = Field(None, description="阈值")
    metric_name: Optional[str] = Field(None, description="指标名称")
    
    # 上下文信息
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    
    # 处理信息
    acknowledged_by: Optional[str] = Field(None, description="确认人")
    acknowledged_at: Optional[datetime] = Field(None, description="确认时间")
    resolved_by: Optional[str] = Field(None, description="解决人")
    resolved_at: Optional[datetime] = Field(None, description="解决时间")
    resolution_note: Optional[str] = Field(None, description="解决备注")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    # 通知信息
    notification_sent: bool = Field(default=False, description="是否已发送通知")
    notification_channels: List[str] = Field(default_factory=list, description="通知渠道")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
    
    def acknowledge(self, user: str, note: Optional[str] = None):
        """确认告警"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()
        self.updated_at = datetime.now()
        if note:
            self.context["acknowledge_note"] = note
    
    def resolve(self, user: str, note: Optional[str] = None):
        """解决告警"""
        self.status = AlertStatus.RESOLVED
        self.resolved_by = user
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
        if note:
            self.resolution_note = note
    
    def suppress(self):
        """抑制告警"""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = datetime.now()
    
    def is_active(self) -> bool:
        """检查告警是否活跃"""
        return self.status == AlertStatus.ACTIVE
    
    def is_critical(self) -> bool:
        """检查是否为严重告警"""
        return self.alert_level in [AlertLevel.CRITICAL, AlertLevel.FATAL]
    
    def get_priority_score(self) -> int:
        """获取告警优先级分数"""
        level_scores = {
            AlertLevel.INFO: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.ERROR: 3,
            AlertLevel.CRITICAL: 4,
            AlertLevel.FATAL: 5
        }
        return level_scores.get(self.alert_level, 1)
    
    def to_vector_text(self) -> str:
        """转换为用于向量化的文本"""
        text_parts = [
            f"告警: {self.title}",
            f"级别: {self.alert_level.value}",
            f"类型: {self.alert_type.value}",
            f"设备: {self.device_id}",
            self.message
        ]
        
        if self.metric_name and self.trigger_value is not None:
            text_parts.append(f"指标 {self.metric_name} 值为 {self.trigger_value}")
        
        if self.threshold_value is not None:
            text_parts.append(f"阈值 {self.threshold_value}")
        
        # 添加上下文信息
        for key, value in self.context.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)
    
    def get_summary(self) -> str:
        """获取告警摘要"""
        return f"[{self.alert_level.value.upper()}] {self.title} - 设备 {self.device_id}"


class AlertStatistics(BaseModel):
    """告警统计信息"""
    total_alerts: int = Field(default=0, description="总告警数")
    active_alerts: int = Field(default=0, description="活跃告警数")
    critical_alerts: int = Field(default=0, description="严重告警数")
    resolved_alerts: int = Field(default=0, description="已解决告警数")
    
    # 按级别统计
    alerts_by_level: Dict[str, int] = Field(default_factory=dict, description="按级别统计")
    
    # 按类型统计
    alerts_by_type: Dict[str, int] = Field(default_factory=dict, description="按类型统计")
    
    # 按设备统计
    alerts_by_device: Dict[str, int] = Field(default_factory=dict, description="按设备统计")
    
    # 时间范围
    start_time: Optional[datetime] = Field(None, description="统计开始时间")
    end_time: Optional[datetime] = Field(None, description="统计结束时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }