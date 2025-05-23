"""嵌入向量服务"""

import asyncio
import aiohttp
import json
from typing import List, Optional, Union
from loguru import logger

from ..config import get_settings


class EmbeddingService:
    """嵌入向量服务"""
    
    def __init__(self):
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """确保会话存在"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        await self._ensure_session()
        
        try:
            # 构建请求数据
            payload = {
                "model": self.settings.embedding_model,
                "prompt": text
            }
            
            # 发送请求到Ollama
            async with self.session.post(
                self.settings.ollama_embedding_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result.get("embedding", [])
                    
                    if not embedding:
                        raise ValueError("返回的嵌入向量为空")
                    
                    logger.debug(f"生成嵌入向量成功，维度: {len(embedding)}")
                    return embedding
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API错误 {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            logger.error("生成嵌入向量超时")
            raise Exception("嵌入向量生成超时")
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        embeddings = []
        
        # 并发生成嵌入向量
        tasks = [self.generate_embedding(text) for text in texts]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"生成第{i+1}个文本的嵌入向量失败: {result}")
                    # 使用零向量作为fallback
                    embeddings.append([0.0] * self.settings.embedding_dimension)
                else:
                    embeddings.append(result)
            
            logger.info(f"批量生成嵌入向量完成，共{len(embeddings)}个")
            return embeddings
            
        except Exception as e:
            logger.error(f"批量生成嵌入向量失败: {e}")
            raise
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 测试生成一个简单的嵌入向量
            embedding = await self.generate_embedding("测试文本")
            return len(embedding) > 0
        except Exception as e:
            logger.error(f"嵌入服务健康检查失败: {e}")
            return False
    
    def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            asyncio.create_task(self.session.close())


class CachedEmbeddingService(EmbeddingService):
    """带缓存的嵌入向量服务"""
    
    def __init__(self, cache_size: int = 1000):
        super().__init__()
        self.cache = {}
        self.cache_size = cache_size
        self.access_order = []
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return f"{self.settings.embedding_model}:{hash(text)}"
    
    def _manage_cache_size(self):
        """管理缓存大小"""
        while len(self.cache) > self.cache_size:
            # 移除最久未访问的项
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
    
    async def generate_embedding(self, text: str) -> List[float]:
        """生成嵌入向量（带缓存）"""
        cache_key = self._get_cache_key(text)
        
        # 检查缓存
        if cache_key in self.cache:
            # 更新访问顺序
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            logger.debug(f"从缓存获取嵌入向量: {cache_key}")
            return self.cache[cache_key]
        
        # 生成新的嵌入向量
        embedding = await super().generate_embedding(text)
        
        # 存入缓存
        self.cache[cache_key] = embedding
        self.access_order.append(cache_key)
        
        # 管理缓存大小
        self._manage_cache_size()
        
        logger.debug(f"嵌入向量已缓存: {cache_key}")
        return embedding
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
        logger.info("嵌入向量缓存已清空")
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "hit_rate": len(self.access_order) / max(1, len(self.cache))
        }


# 全局嵌入服务实例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(use_cache: bool = True) -> EmbeddingService:
    """获取嵌入服务实例（单例模式）"""
    global _embedding_service
    
    if _embedding_service is None:
        if use_cache:
            _embedding_service = CachedEmbeddingService()
        else:
            _embedding_service = EmbeddingService()
    
    return _embedding_service


async def generate_text_embedding(text: str) -> List[float]:
    """便捷函数：生成文本嵌入向量"""
    service = get_embedding_service()
    async with service:
        return await service.generate_embedding(text)


async def generate_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """便捷函数：批量生成文本嵌入向量"""
    service = get_embedding_service()
    async with service:
        return await service.generate_embeddings(texts)