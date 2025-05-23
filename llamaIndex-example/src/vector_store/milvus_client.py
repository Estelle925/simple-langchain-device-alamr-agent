"""Milvus向量数据库客户端"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from pydantic import BaseModel
from loguru import logger

from ..config import get_settings
from ..models import Device, Alert


class MilvusConfig(BaseModel):
    """Milvus配置"""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "device_alerts"
    dimension: int = 1024
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    nlist: int = 1024


class MilvusClient:
    """Milvus向量数据库客户端"""
    
    def __init__(self, config: Optional[MilvusConfig] = None):
        self.config = config or MilvusConfig()
        self.settings = get_settings()
        self.collection: Optional[Collection] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """连接到Milvus数据库"""
        try:
            # 连接到Milvus
            connections.connect(
                alias="default",
                host=self.settings.milvus_host,
                port=self.settings.milvus_port
            )
            
            # 创建或获取集合
            await self._ensure_collection()
            
            self._connected = True
            logger.info(f"成功连接到Milvus: {self.settings.milvus_host}:{self.settings.milvus_port}")
            return True
            
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            return False
    
    async def disconnect(self):
        """断开Milvus连接"""
        try:
            connections.disconnect("default")
            self._connected = False
            logger.info("已断开Milvus连接")
        except Exception as e:
            logger.error(f"断开Milvus连接失败: {e}")
    
    async def _ensure_collection(self):
        """确保集合存在"""
        collection_name = self.settings.milvus_collection_name
        
        # 检查集合是否存在
        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            logger.info(f"使用现有集合: {collection_name}")
        else:
            # 创建新集合
            await self._create_collection(collection_name)
            logger.info(f"创建新集合: {collection_name}")
    
    async def _create_collection(self, collection_name: str):
        """创建Milvus集合"""
        # 定义字段模式
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
                description="主键ID"
            ),
            FieldSchema(
                name="entity_type",
                dtype=DataType.VARCHAR,
                max_length=32,
                description="实体类型(device/alert)"
            ),
            FieldSchema(
                name="entity_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="实体ID"
            ),
            FieldSchema(
                name="text_content",
                dtype=DataType.VARCHAR,
                max_length=2048,
                description="文本内容"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.settings.embedding_dimension,
                description="向量嵌入"
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="元数据JSON"
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.INT64,
                description="时间戳"
            )
        ]
        
        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="设备告警向量存储集合"
        )
        
        # 创建集合
        self.collection = Collection(
            name=collection_name,
            schema=schema
        )
        
        # 创建索引
        await self._create_index()
    
    async def _create_index(self):
        """创建向量索引"""
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"nlist": self.config.nlist}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info("创建向量索引完成")
    
    async def insert_device(self, device: Device, embedding: List[float]) -> str:
        """插入设备数据"""
        if not self._connected:
            raise RuntimeError("未连接到Milvus")
        
        # 生成唯一ID
        doc_id = f"device_{device.device_id}_{uuid.uuid4().hex[:8]}"
        
        # 准备数据
        data = [
            [doc_id],  # id
            ["device"],  # entity_type
            [device.device_id],  # entity_id
            [device.to_vector_text()],  # text_content
            [embedding],  # embedding
            [json.dumps(device.dict(), ensure_ascii=False)],  # metadata
            [int(datetime.now().timestamp() * 1000)]  # timestamp
        ]
        
        # 插入数据
        self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"插入设备数据: {device.device_id}")
        return doc_id
    
    async def insert_alert(self, alert: Alert, embedding: List[float]) -> str:
        """插入告警数据"""
        if not self._connected:
            raise RuntimeError("未连接到Milvus")
        
        # 生成唯一ID
        doc_id = f"alert_{alert.alert_id}_{uuid.uuid4().hex[:8]}"
        
        # 准备数据
        data = [
            [doc_id],  # id
            ["alert"],  # entity_type
            [alert.alert_id],  # entity_id
            [alert.to_vector_text()],  # text_content
            [embedding],  # embedding
            [json.dumps(alert.dict(), ensure_ascii=False)],  # metadata
            [int(datetime.now().timestamp() * 1000)]  # timestamp
        ]
        
        # 插入数据
        self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"插入告警数据: {alert.alert_id}")
        return doc_id
    
    async def search_similar(
        self,
        query_embedding: List[float],
        entity_type: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        if not self._connected:
            raise RuntimeError("未连接到Milvus")
        
        # 加载集合到内存
        self.collection.load()
        
        # 构建搜索参数
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": 10}
        }
        
        # 构建过滤表达式
        expr = None
        if entity_type:
            expr = f'entity_type == "{entity_type}"'
        
        # 执行搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["entity_type", "entity_id", "text_content", "metadata", "timestamp"]
        )
        
        # 处理结果
        similar_items = []
        for hits in results:
            for hit in hits:
                # 计算相似度分数（距离转换为相似度）
                similarity = 1.0 / (1.0 + hit.distance)
                
                if similarity >= score_threshold:
                    item = {
                        "id": hit.id,
                        "entity_type": hit.entity.get("entity_type"),
                        "entity_id": hit.entity.get("entity_id"),
                        "text_content": hit.entity.get("text_content"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "timestamp": hit.entity.get("timestamp"),
                        "similarity": similarity,
                        "distance": hit.distance
                    }
                    similar_items.append(item)
        
        logger.info(f"搜索到 {len(similar_items)} 个相似项")
        return similar_items
    
    async def search_devices(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Device]:
        """搜索相似设备"""
        results = await self.search_similar(
            query_embedding=query_embedding,
            entity_type="device",
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        devices = []
        for result in results:
            try:
                device_data = result["metadata"]
                device = Device(**device_data)
                devices.append(device)
            except Exception as e:
                logger.error(f"解析设备数据失败: {e}")
        
        return devices
    
    async def search_alerts(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Alert]:
        """搜索相似告警"""
        results = await self.search_similar(
            query_embedding=query_embedding,
            entity_type="alert",
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        alerts = []
        for result in results:
            try:
                alert_data = result["metadata"]
                alert = Alert(**alert_data)
                alerts.append(alert)
            except Exception as e:
                logger.error(f"解析告警数据失败: {e}")
        
        return alerts
    
    async def delete_by_entity_id(self, entity_id: str, entity_type: Optional[str] = None):
        """根据实体ID删除数据"""
        if not self._connected:
            raise RuntimeError("未连接到Milvus")
        
        # 构建删除表达式
        expr = f'entity_id == "{entity_id}"'
        if entity_type:
            expr += f' and entity_type == "{entity_type}"'
        
        # 执行删除
        self.collection.delete(expr)
        self.collection.flush()
        
        logger.info(f"删除实体数据: {entity_id}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self._connected:
            raise RuntimeError("未连接到Milvus")
        
        stats = self.collection.get_stats()
        return {
            "collection_name": self.collection.name,
            "num_entities": stats["row_count"],
            "data_size": stats.get("data_size", "unknown"),
            "index_size": stats.get("index_size", "unknown")
        }
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._connected:
                return False
            
            # 检查集合状态
            stats = await self.get_collection_stats()
            return stats["num_entities"] >= 0
            
        except Exception as e:
            logger.error(f"Milvus健康检查失败: {e}")
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connected:
            # 注意：这里不能使用async方法
            try:
                connections.disconnect("default")
            except Exception as e:
                logger.error(f"关闭Milvus连接失败: {e}")