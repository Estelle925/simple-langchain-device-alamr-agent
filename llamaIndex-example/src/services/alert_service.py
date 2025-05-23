"""告警服务"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger

from ..config import get_settings
from ..models import Alert, AlertLevel, AlertType, AlertStatus, AlertRule, AlertStatistics, Device
from ..vector_store import MilvusClient
from ..utils import EmbeddingService, get_embedding_service
from ..agents import AlertAgent


class AlertService:
    """告警服务类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.milvus_client: Optional[MilvusClient] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.alert_agent: Optional[AlertAgent] = None
        
        # 内存中的告警缓存
        self.alert_cache: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
    
    async def initialize(self):
        """初始化服务"""
        try:
            # 初始化Milvus客户端
            self.milvus_client = MilvusClient()
            await self.milvus_client.connect()
            
            # 初始化嵌入服务
            self.embedding_service = get_embedding_service()
            
            # 初始化告警Agent
            self.alert_agent = AlertAgent()
            await self.alert_agent.initialize()
            
            logger.info("告警服务初始化完成")
            
        except Exception as e:
            logger.error(f"告警服务初始化失败: {e}")
            raise
    
    async def create_alert(
        self,
        device_id: str,
        alert_type: AlertType,
        alert_level: AlertLevel,
        title: str,
        message: str,
        **kwargs
    ) -> Alert:
        """创建新告警"""
        try:
            # 生成告警ID
            alert_id = f"alert_{uuid.uuid4().hex[:8]}"
            
            # 创建告警对象
            alert = Alert(
                alert_id=alert_id,
                device_id=device_id,
                alert_type=alert_type,
                alert_level=alert_level,
                title=title,
                message=message,
                **kwargs
            )
            
            # 生成嵌入向量
            embedding = await self.embedding_service.generate_embedding(
                alert.to_vector_text()
            )
            
            # 存储到向量数据库
            await self.milvus_client.insert_alert(alert, embedding)
            
            # 缓存告警
            self.alert_cache[alert_id] = alert
            if alert.is_active():
                self.active_alerts[alert_id] = alert
            
            logger.info(f"创建告警成功: {alert_id}")
            return alert
            
        except Exception as e:
            logger.error(f"创建告警失败: {e}")
            raise
    
    async def analyze_and_create_alert(self, device: Device) -> Optional[Alert]:
        """分析设备状态并创建告警"""
        try:
            if not self.alert_agent:
                raise RuntimeError("告警Agent未初始化")
            
            # 使用AI Agent分析设备
            analysis_result = await self.alert_agent.analyze_device(device)
            
            # 如果需要告警，创建告警
            if analysis_result.should_alert:
                alert = await self.create_alert(
                    device_id=device.device_id,
                    alert_type=analysis_result.alert_type,
                    alert_level=analysis_result.alert_level,
                    title=analysis_result.title,
                    message=analysis_result.message,
                    context={
                        "analysis_confidence": analysis_result.confidence,
                        "reasoning": analysis_result.reasoning,
                        "suggested_actions": analysis_result.suggested_actions,
                        **analysis_result.context
                    }
                )
                
                logger.info(f"AI分析创建告警: {alert.alert_id} (置信度: {analysis_result.confidence})")
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"AI分析创建告警失败: {e}")
            raise
    
    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警信息"""
        try:
            # 先从缓存获取
            if alert_id in self.alert_cache:
                return self.alert_cache[alert_id]
            
            # 从向量数据库搜索
            query_text = f"告警ID {alert_id}"
            query_embedding = await self.embedding_service.generate_embedding(query_text)
            
            alerts = await self.milvus_client.search_alerts(
                query_embedding=query_embedding,
                top_k=1,
                score_threshold=0.9
            )
            
            if alerts and alerts[0].alert_id == alert_id:
                alert = alerts[0]
                self.alert_cache[alert_id] = alert
                if alert.is_active():
                    self.active_alerts[alert_id] = alert
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"获取告警失败: {e}")
            return None
    
    async def update_alert(self, alert_id: str, **updates) -> Optional[Alert]:
        """更新告警信息"""
        try:
            alert = await self.get_alert(alert_id)
            if not alert:
                raise ValueError(f"告警 {alert_id} 不存在")
            
            # 更新告警属性
            for key, value in updates.items():
                if hasattr(alert, key):
                    setattr(alert, key, value)
            
            alert.updated_at = datetime.now()
            
            # 重新生成嵌入向量
            embedding = await self.embedding_service.generate_embedding(
                alert.to_vector_text()
            )
            
            # 删除旧数据
            await self.milvus_client.delete_by_entity_id(alert_id, "alert")
            
            # 插入新数据
            await self.milvus_client.insert_alert(alert, embedding)
            
            # 更新缓存
            self.alert_cache[alert_id] = alert
            if alert.is_active():
                self.active_alerts[alert_id] = alert
            elif alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
            
            logger.info(f"更新告警成功: {alert_id}")
            return alert
            
        except Exception as e:
            logger.error(f"更新告警失败: {e}")
            raise
    
    async def acknowledge_alert(self, alert_id: str, user: str, note: Optional[str] = None) -> bool:
        """确认告警"""
        try:
            alert = await self.get_alert(alert_id)
            if not alert:
                raise ValueError(f"告警 {alert_id} 不存在")
            
            alert.acknowledge(user, note)
            
            # 更新数据库
            await self.update_alert(alert_id, 
                status=alert.status,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
                updated_at=alert.updated_at
            )
            
            logger.info(f"确认告警成功: {alert_id} by {user}")
            return True
            
        except Exception as e:
            logger.error(f"确认告警失败: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, user: str, note: Optional[str] = None) -> bool:
        """解决告警"""
        try:
            alert = await self.get_alert(alert_id)
            if not alert:
                raise ValueError(f"告警 {alert_id} 不存在")
            
            alert.resolve(user, note)
            
            # 更新数据库
            await self.update_alert(alert_id,
                status=alert.status,
                resolved_by=alert.resolved_by,
                resolved_at=alert.resolved_at,
                resolution_note=alert.resolution_note,
                updated_at=alert.updated_at
            )
            
            logger.info(f"解决告警成功: {alert_id} by {user}")
            return True
            
        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            return False
    
    async def suppress_alert(self, alert_id: str) -> bool:
        """抑制告警"""
        try:
            alert = await self.get_alert(alert_id)
            if not alert:
                raise ValueError(f"告警 {alert_id} 不存在")
            
            alert.suppress()
            
            # 更新数据库
            await self.update_alert(alert_id,
                status=alert.status,
                updated_at=alert.updated_at
            )
            
            logger.info(f"抑制告警成功: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"抑制告警失败: {e}")
            return False
    
    async def search_alerts(
        self,
        query: str,
        alert_level: Optional[AlertLevel] = None,
        alert_type: Optional[AlertType] = None,
        status: Optional[AlertStatus] = None,
        device_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        top_k: int = 20
    ) -> List[Alert]:
        """搜索告警"""
        try:
            # 构建查询文本
            query_parts = [query]
            if alert_level:
                query_parts.append(f"级别 {alert_level.value}")
            if alert_type:
                query_parts.append(f"类型 {alert_type.value}")
            if status:
                query_parts.append(f"状态 {status.value}")
            if device_id:
                query_parts.append(f"设备 {device_id}")
            
            query_text = " ".join(query_parts)
            
            # 生成查询嵌入
            query_embedding = await self.embedding_service.generate_embedding(query_text)
            
            # 搜索告警
            alerts = await self.milvus_client.search_alerts(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=0.3
            )
            
            # 过滤结果
            filtered_alerts = []
            for alert in alerts:
                # 级别过滤
                if alert_level and alert.alert_level != alert_level:
                    continue
                # 类型过滤
                if alert_type and alert.alert_type != alert_type:
                    continue
                # 状态过滤
                if status and alert.status != status:
                    continue
                # 设备过滤
                if device_id and alert.device_id != device_id:
                    continue
                # 时间过滤
                if start_time and alert.created_at < start_time:
                    continue
                if end_time and alert.created_at > end_time:
                    continue
                
                filtered_alerts.append(alert)
            
            logger.info(f"搜索到 {len(filtered_alerts)} 个告警")
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"搜索告警失败: {e}")
            return []
    
    async def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    async def get_critical_alerts(self) -> List[Alert]:
        """获取严重告警"""
        return await self.search_alerts(
            query="严重告警",
            alert_level=AlertLevel.CRITICAL
        ) + await self.search_alerts(
            query="致命告警",
            alert_level=AlertLevel.FATAL
        )
    
    async def get_device_alerts(self, device_id: str, limit: int = 10) -> List[Alert]:
        """获取设备告警"""
        return await self.search_alerts(
            query=f"设备 {device_id} 告警",
            device_id=device_id,
            top_k=limit
        )
    
    async def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """获取最近告警"""
        start_time = datetime.now() - timedelta(hours=hours)
        return await self.search_alerts(
            query="最近告警",
            start_time=start_time
        )
    
    async def get_alert_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> AlertStatistics:
        """获取告警统计信息"""
        try:
            # 默认统计最近24小时
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            # 获取时间范围内的告警
            alerts = await self.search_alerts(
                query="统计告警",
                start_time=start_time,
                end_time=end_time,
                top_k=1000
            )
            
            # 初始化统计
            stats = AlertStatistics(
                start_time=start_time,
                end_time=end_time
            )
            
            stats.total_alerts = len(alerts)
            
            for alert in alerts:
                # 按状态统计
                if alert.is_active():
                    stats.active_alerts += 1
                if alert.status == AlertStatus.RESOLVED:
                    stats.resolved_alerts += 1
                if alert.is_critical():
                    stats.critical_alerts += 1
                
                # 按级别统计
                level = alert.alert_level.value
                stats.alerts_by_level[level] = stats.alerts_by_level.get(level, 0) + 1
                
                # 按类型统计
                alert_type = alert.alert_type.value
                stats.alerts_by_type[alert_type] = stats.alerts_by_type.get(alert_type, 0) + 1
                
                # 按设备统计
                device_id = alert.device_id
                stats.alerts_by_device[device_id] = stats.alerts_by_device.get(device_id, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"获取告警统计失败: {e}")
            return AlertStatistics()
    
    async def cleanup_old_alerts(self, days: int = 30) -> int:
        """清理旧告警"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # 搜索旧告警
            old_alerts = await self.search_alerts(
                query="旧告警清理",
                end_time=cutoff_time,
                top_k=1000
            )
            
            cleaned_count = 0
            for alert in old_alerts:
                # 只清理已解决的告警
                if alert.status == AlertStatus.RESOLVED:
                    await self.milvus_client.delete_by_entity_id(alert.alert_id, "alert")
                    
                    # 从缓存删除
                    if alert.alert_id in self.alert_cache:
                        del self.alert_cache[alert.alert_id]
                    if alert.alert_id in self.active_alerts:
                        del self.active_alerts[alert.alert_id]
                    
                    cleaned_count += 1
            
            logger.info(f"清理了 {cleaned_count} 个旧告警")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理旧告警失败: {e}")
            return 0
    
    async def add_alert_rule(self, rule: AlertRule) -> bool:
        """添加告警规则"""
        try:
            if self.alert_agent:
                await self.alert_agent.add_alert_rule(rule)
                logger.info(f"添加告警规则成功: {rule.rule_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"添加告警规则失败: {e}")
            return False
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        try:
            if self.alert_agent:
                return await self.alert_agent.remove_alert_rule(rule_id)
            return False
            
        except Exception as e:
            logger.error(f"移除告警规则失败: {e}")
            return False
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """获取告警规则"""
        try:
            if self.alert_agent:
                return await self.alert_agent.get_alert_rules()
            return []
            
        except Exception as e:
            logger.error(f"获取告警规则失败: {e}")
            return []
    
    async def batch_process_alerts(self, alert_ids: List[str], action: str, user: str) -> Dict[str, bool]:
        """批量处理告警"""
        results = {}
        
        for alert_id in alert_ids:
            try:
                if action == "acknowledge":
                    results[alert_id] = await self.acknowledge_alert(alert_id, user)
                elif action == "resolve":
                    results[alert_id] = await self.resolve_alert(alert_id, user)
                elif action == "suppress":
                    results[alert_id] = await self.suppress_alert(alert_id)
                else:
                    results[alert_id] = False
                    logger.warning(f"未知的批量操作: {action}")
            except Exception as e:
                logger.error(f"批量处理告警 {alert_id} 失败: {e}")
                results[alert_id] = False
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"批量处理完成，成功处理 {success_count}/{len(alert_ids)} 个告警")
        return results
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.milvus_client or not await self.milvus_client.health_check():
                return False
            
            if not self.embedding_service or not await self.embedding_service.health_check():
                return False
            
            if not self.alert_agent or not await self.alert_agent.health_check():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"告警服务健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭服务"""
        try:
            if self.alert_agent:
                await self.alert_agent.close()
            
            if self.milvus_client:
                await self.milvus_client.disconnect()
            
            if self.embedding_service:
                self.embedding_service.close()
            
            # 清空缓存
            self.alert_cache.clear()
            self.active_alerts.clear()
            
            logger.info("告警服务已关闭")
            
        except Exception as e:
            logger.error(f"关闭告警服务失败: {e}")


# 全局告警服务实例
_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """获取告警服务实例（单例模式）"""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service