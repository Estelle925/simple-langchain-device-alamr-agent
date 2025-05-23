"""设备服务"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from ..config import get_settings
from ..models import Device, DeviceStatus, DeviceType, DeviceMetrics, DeviceLocation
from ..vector_store import MilvusClient
from ..utils import EmbeddingService, get_embedding_service


class DeviceService:
    """设备服务类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.milvus_client: Optional[MilvusClient] = None
        self.embedding_service: Optional[EmbeddingService] = None
        
        # 内存中的设备缓存
        self.device_cache: Dict[str, Device] = {}
    
    async def initialize(self):
        """初始化服务"""
        try:
            # 初始化Milvus客户端
            self.milvus_client = MilvusClient()
            await self.milvus_client.connect()
            
            # 初始化嵌入服务
            self.embedding_service = get_embedding_service()
            
            logger.info("设备服务初始化完成")
            
        except Exception as e:
            logger.error(f"设备服务初始化失败: {e}")
            raise
    
    async def create_device(
        self,
        device_id: str,
        name: str,
        device_type: DeviceType,
        **kwargs
    ) -> Device:
        """创建新设备"""
        try:
            # 检查设备是否已存在
            if device_id in self.device_cache:
                raise ValueError(f"设备 {device_id} 已存在")
            
            # 创建设备对象
            device = Device(
                device_id=device_id,
                name=name,
                device_type=device_type,
                **kwargs
            )
            
            # 生成嵌入向量
            embedding = await self.embedding_service.generate_embedding(
                device.to_vector_text()
            )
            
            # 存储到向量数据库
            await self.milvus_client.insert_device(device, embedding)
            
            # 缓存设备
            self.device_cache[device_id] = device
            
            logger.info(f"创建设备成功: {device_id}")
            return device
            
        except Exception as e:
            logger.error(f"创建设备失败: {e}")
            raise
    
    async def get_device(self, device_id: str) -> Optional[Device]:
        """获取设备信息"""
        try:
            # 先从缓存获取
            if device_id in self.device_cache:
                return self.device_cache[device_id]
            
            # 从向量数据库搜索
            query_text = f"设备ID {device_id}"
            query_embedding = await self.embedding_service.generate_embedding(query_text)
            
            devices = await self.milvus_client.search_devices(
                query_embedding=query_embedding,
                top_k=1,
                score_threshold=0.9
            )
            
            if devices and devices[0].device_id == device_id:
                device = devices[0]
                self.device_cache[device_id] = device
                return device
            
            return None
            
        except Exception as e:
            logger.error(f"获取设备失败: {e}")
            return None
    
    async def update_device(self, device_id: str, **updates) -> Optional[Device]:
        """更新设备信息"""
        try:
            device = await self.get_device(device_id)
            if not device:
                raise ValueError(f"设备 {device_id} 不存在")
            
            # 更新设备属性
            for key, value in updates.items():
                if hasattr(device, key):
                    setattr(device, key, value)
            
            device.updated_at = datetime.now()
            
            # 重新生成嵌入向量
            embedding = await self.embedding_service.generate_embedding(
                device.to_vector_text()
            )
            
            # 删除旧数据
            await self.milvus_client.delete_by_entity_id(device_id, "device")
            
            # 插入新数据
            await self.milvus_client.insert_device(device, embedding)
            
            # 更新缓存
            self.device_cache[device_id] = device
            
            logger.info(f"更新设备成功: {device_id}")
            return device
            
        except Exception as e:
            logger.error(f"更新设备失败: {e}")
            raise
    
    async def update_device_status(
        self,
        device_id: str,
        status: DeviceStatus,
        metrics: Optional[DeviceMetrics] = None
    ) -> Optional[Device]:
        """更新设备状态"""
        try:
            device = await self.get_device(device_id)
            if not device:
                raise ValueError(f"设备 {device_id} 不存在")
            
            # 更新状态
            device.update_status(status, metrics)
            
            # 重新生成嵌入向量
            embedding = await self.embedding_service.generate_embedding(
                device.to_vector_text()
            )
            
            # 删除旧数据并插入新数据
            await self.milvus_client.delete_by_entity_id(device_id, "device")
            await self.milvus_client.insert_device(device, embedding)
            
            # 更新缓存
            self.device_cache[device_id] = device
            
            logger.info(f"更新设备状态成功: {device_id} -> {status.value}")
            return device
            
        except Exception as e:
            logger.error(f"更新设备状态失败: {e}")
            raise
    
    async def update_device_metrics(self, device_id: str, **metrics) -> Optional[Device]:
        """更新设备指标"""
        try:
            device = await self.get_device(device_id)
            if not device:
                raise ValueError(f"设备 {device_id} 不存在")
            
            # 更新指标
            device.update_metrics(**metrics)
            
            # 重新生成嵌入向量
            embedding = await self.embedding_service.generate_embedding(
                device.to_vector_text()
            )
            
            # 删除旧数据并插入新数据
            await self.milvus_client.delete_by_entity_id(device_id, "device")
            await self.milvus_client.insert_device(device, embedding)
            
            # 更新缓存
            self.device_cache[device_id] = device
            
            logger.info(f"更新设备指标成功: {device_id}")
            return device
            
        except Exception as e:
            logger.error(f"更新设备指标失败: {e}")
            raise
    
    async def delete_device(self, device_id: str) -> bool:
        """删除设备"""
        try:
            # 从向量数据库删除
            await self.milvus_client.delete_by_entity_id(device_id, "device")
            
            # 从缓存删除
            if device_id in self.device_cache:
                del self.device_cache[device_id]
            
            logger.info(f"删除设备成功: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除设备失败: {e}")
            return False
    
    async def search_devices(
        self,
        query: str,
        device_type: Optional[DeviceType] = None,
        status: Optional[DeviceStatus] = None,
        top_k: int = 10
    ) -> List[Device]:
        """搜索设备"""
        try:
            # 构建查询文本
            query_parts = [query]
            if device_type:
                query_parts.append(f"类型 {device_type.value}")
            if status:
                query_parts.append(f"状态 {status.value}")
            
            query_text = " ".join(query_parts)
            
            # 生成查询嵌入
            query_embedding = await self.embedding_service.generate_embedding(query_text)
            
            # 搜索设备
            devices = await self.milvus_client.search_devices(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=0.5
            )
            
            # 过滤结果
            filtered_devices = []
            for device in devices:
                if device_type and device.device_type != device_type:
                    continue
                if status and device.status != status:
                    continue
                filtered_devices.append(device)
            
            logger.info(f"搜索到 {len(filtered_devices)} 个设备")
            return filtered_devices
            
        except Exception as e:
            logger.error(f"搜索设备失败: {e}")
            return []
    
    async def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        """根据类型获取设备"""
        return await self.search_devices(
            query=f"{device_type.value} 设备",
            device_type=device_type
        )
    
    async def get_devices_by_status(self, status: DeviceStatus) -> List[Device]:
        """根据状态获取设备"""
        return await self.search_devices(
            query=f"{status.value} 设备",
            status=status
        )
    
    async def get_offline_devices(self) -> List[Device]:
        """获取离线设备"""
        return await self.get_devices_by_status(DeviceStatus.OFFLINE)
    
    async def get_fault_devices(self) -> List[Device]:
        """获取故障设备"""
        return await self.get_devices_by_status(DeviceStatus.FAULT)
    
    async def get_device_statistics(self) -> Dict[str, Any]:
        """获取设备统计信息"""
        try:
            # 获取所有缓存的设备
            all_devices = list(self.device_cache.values())
            
            # 如果缓存为空，尝试从数据库获取
            if not all_devices:
                all_devices = await self.search_devices("所有设备", top_k=1000)
            
            stats = {
                "total_devices": len(all_devices),
                "by_type": {},
                "by_status": {},
                "online_rate": 0,
                "health_rate": 0,
                "avg_battery_level": 0,
                "avg_usage_rate": 0
            }
            
            if all_devices:
                online_count = 0
                healthy_count = 0
                battery_levels = []
                usage_rates = []
                
                for device in all_devices:
                    # 按类型统计
                    device_type = device.device_type.value
                    stats["by_type"][device_type] = stats["by_type"].get(device_type, 0) + 1
                    
                    # 按状态统计
                    status = device.status.value
                    stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                    
                    # 在线率统计
                    if device.is_online():
                        online_count += 1
                    
                    # 健康率统计
                    if device.is_healthy():
                        healthy_count += 1
                    
                    # 指标统计
                    if device.metrics.battery_level is not None:
                        battery_levels.append(device.metrics.battery_level)
                    if device.metrics.usage_rate is not None:
                        usage_rates.append(device.metrics.usage_rate)
                
                stats["online_rate"] = online_count / len(all_devices)
                stats["health_rate"] = healthy_count / len(all_devices)
                
                if battery_levels:
                    stats["avg_battery_level"] = sum(battery_levels) / len(battery_levels)
                if usage_rates:
                    stats["avg_usage_rate"] = sum(usage_rates) / len(usage_rates)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取设备统计失败: {e}")
            return {}
    
    async def add_device_image(self, device_id: str, image_url: str) -> bool:
        """添加设备图片"""
        try:
            device = await self.get_device(device_id)
            if not device:
                raise ValueError(f"设备 {device_id} 不存在")
            
            device.add_image(image_url)
            
            # 更新向量数据库
            embedding = await self.embedding_service.generate_embedding(
                device.to_vector_text()
            )
            await self.milvus_client.delete_by_entity_id(device_id, "device")
            await self.milvus_client.insert_device(device, embedding)
            
            # 更新缓存
            self.device_cache[device_id] = device
            
            logger.info(f"添加设备图片成功: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加设备图片失败: {e}")
            return False
    
    async def batch_update_devices(self, updates: List[Dict[str, Any]]) -> List[Device]:
        """批量更新设备"""
        updated_devices = []
        
        for update_data in updates:
            device_id = update_data.pop("device_id")
            try:
                device = await self.update_device(device_id, **update_data)
                if device:
                    updated_devices.append(device)
            except Exception as e:
                logger.error(f"批量更新设备 {device_id} 失败: {e}")
        
        logger.info(f"批量更新完成，成功更新 {len(updated_devices)} 个设备")
        return updated_devices
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.milvus_client or not await self.milvus_client.health_check():
                return False
            
            if not self.embedding_service or not await self.embedding_service.health_check():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"设备服务健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭服务"""
        try:
            if self.milvus_client:
                await self.milvus_client.disconnect()
            
            if self.embedding_service:
                self.embedding_service.close()
            
            # 清空缓存
            self.device_cache.clear()
            
            logger.info("设备服务已关闭")
            
        except Exception as e:
            logger.error(f"关闭设备服务失败: {e}")


# 全局设备服务实例
_device_service: Optional[DeviceService] = None


def get_device_service() -> DeviceService:
    """获取设备服务实例（单例模式）"""
    global _device_service
    if _device_service is None:
        _device_service = DeviceService()
    return _device_service