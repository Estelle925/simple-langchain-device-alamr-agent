"""查询智能Agent"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field
from loguru import logger

from ..config import get_settings
from ..models import Device, Alert
from ..vector_store import MilvusClient
from ..utils import EmbeddingService, get_embedding_service


class QueryResult(BaseModel):
    """查询结果"""
    success: bool = Field(..., description="查询是否成功")
    message: str = Field(..., description="查询结果消息")
    data: Dict[str, Any] = Field(default_factory=dict, description="查询数据")
    suggestions: List[str] = Field(default_factory=list, description="相关建议")


class DeviceSearchTool(BaseTool):
    """设备搜索工具"""
    name = "device_search"
    description = "根据自然语言描述搜索相关设备"
    
    def __init__(self, milvus_client: MilvusClient, embedding_service: EmbeddingService):
        super().__init__()
        self.milvus_client = milvus_client
        self.embedding_service = embedding_service
    
    def _run(self, query: str) -> str:
        """同步运行"""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """异步搜索设备"""
        try:
            # 生成查询嵌入
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # 搜索相关设备
            devices = await self.milvus_client.search_devices(
                query_embedding=query_embedding,
                top_k=5,
                score_threshold=0.5
            )
            
            # 构建结果
            result = {
                "found_devices": len(devices),
                "devices": []
            }
            
            for device in devices:
                device_info = {
                    "device_id": device.device_id,
                    "name": device.name,
                    "type": device.device_type.value,
                    "status": device.status.value,
                    "location": device.location.dict() if device.location else None,
                    "key_metrics": {
                        "battery_level": device.metrics.battery_level,
                        "usage_rate": device.metrics.usage_rate,
                        "vibration_level": device.metrics.vibration_level,
                        "temperature": device.metrics.temperature
                    }
                }
                result["devices"].append(device_info)
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"设备搜索工具执行失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class AlertSearchTool(BaseTool):
    """告警搜索工具"""
    name = "alert_search"
    description = "根据自然语言描述搜索相关告警信息"
    
    def __init__(self, milvus_client: MilvusClient, embedding_service: EmbeddingService):
        super().__init__()
        self.milvus_client = milvus_client
        self.embedding_service = embedding_service
    
    def _run(self, query: str) -> str:
        """同步运行"""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """异步搜索告警"""
        try:
            # 生成查询嵌入
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # 搜索相关告警
            alerts = await self.milvus_client.search_alerts(
                query_embedding=query_embedding,
                top_k=10,
                score_threshold=0.4
            )
            
            # 构建结果
            result = {
                "found_alerts": len(alerts),
                "alerts": [],
                "statistics": {
                    "by_level": {},
                    "by_type": {},
                    "by_status": {}
                }
            }
            
            for alert in alerts:
                alert_info = {
                    "alert_id": alert.alert_id,
                    "device_id": alert.device_id,
                    "title": alert.title,
                    "level": alert.alert_level.value,
                    "type": alert.alert_type.value,
                    "status": alert.status.value,
                    "message": alert.message,
                    "created_at": alert.created_at.isoformat(),
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                result["alerts"].append(alert_info)
                
                # 统计信息
                level = alert.alert_level.value
                alert_type = alert.alert_type.value
                status = alert.status.value
                
                result["statistics"]["by_level"][level] = result["statistics"]["by_level"].get(level, 0) + 1
                result["statistics"]["by_type"][alert_type] = result["statistics"]["by_type"].get(alert_type, 0) + 1
                result["statistics"]["by_status"][status] = result["statistics"]["by_status"].get(status, 0) + 1
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"告警搜索工具执行失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class StatisticsTool(BaseTool):
    """统计分析工具"""
    name = "statistics_analysis"
    description = "提供设备和告警的统计分析信息"
    
    def __init__(self, milvus_client: MilvusClient):
        super().__init__()
        self.milvus_client = milvus_client
    
    def _run(self, analysis_type: str) -> str:
        """同步运行"""
        return asyncio.run(self._arun(analysis_type))
    
    async def _arun(self, analysis_type: str) -> str:
        """异步统计分析"""
        try:
            # 获取集合统计信息
            stats = await self.milvus_client.get_collection_stats()
            
            result = {
                "collection_stats": stats,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"统计分析工具执行失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class QueryAgent:
    """查询智能Agent"""
    
    def __init__(self):
        self.settings = get_settings()
        self.milvus_client: Optional[MilvusClient] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.llm = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    async def initialize(self):
        """初始化Agent"""
        try:
            # 初始化Milvus客户端
            self.milvus_client = MilvusClient()
            await self.milvus_client.connect()
            
            # 初始化嵌入服务
            self.embedding_service = get_embedding_service()
            
            # 初始化LLM
            self.llm = Ollama(
                model=self.settings.llm_model,
                base_url=self.settings.ollama_base_url,
                temperature=0.1
            )
            
            # 创建工具
            tools = [
                DeviceSearchTool(self.milvus_client, self.embedding_service),
                AlertSearchTool(self.milvus_client, self.embedding_service),
                StatisticsTool(self.milvus_client)
            ]
            
            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=self._get_system_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # 创建Agent
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                max_iterations=3
            )
            
            logger.info("查询Agent初始化完成")
            
        except Exception as e:
            logger.error(f"查询Agent初始化失败: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """
你是一个专业的设备监控和告警查询助手。你可以帮助用户：

1. 搜索和查询设备信息
2. 查找相关的告警记录
3. 提供统计分析信息
4. 回答关于设备状态的问题

当用户提出查询时，请：
- 理解用户的查询意图
- 使用合适的工具获取相关信息
- 以清晰、友好的方式回答用户问题
- 提供有用的建议和后续操作

请始终以中文回答用户问题，并保持专业和友好的语调。
"""
    
    async def query(self, user_input: str) -> QueryResult:
        """处理用户查询"""
        try:
            # 调用Agent处理查询
            result = await self.agent_executor.ainvoke({
                "input": user_input
            })
            
            # 获取Agent输出
            agent_output = result.get("output", "")
            
            # 构建查询结果
            return QueryResult(
                success=True,
                message=agent_output,
                data={"query": user_input, "timestamp": datetime.now().isoformat()},
                suggestions=self._generate_suggestions(user_input)
            )
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            return QueryResult(
                success=False,
                message=f"查询处理失败: {str(e)}",
                data={"error": str(e)},
                suggestions=["请检查查询语句", "尝试使用更简单的表达", "联系技术支持"]
            )
    
    def _generate_suggestions(self, query: str) -> List[str]:
        """生成相关建议"""
        suggestions = []
        
        query_lower = query.lower()
        
        if "设备" in query or "device" in query_lower:
            suggestions.extend([
                "查看设备状态详情",
                "检查设备历史告警",
                "查询相似设备"
            ])
        
        if "告警" in query or "alert" in query_lower:
            suggestions.extend([
                "查看告警详细信息",
                "分析告警趋势",
                "查找解决方案"
            ])
        
        if "统计" in query or "分析" in query:
            suggestions.extend([
                "查看详细统计报告",
                "导出数据分析",
                "设置监控面板"
            ])
        
        # 通用建议
        if not suggestions:
            suggestions = [
                "尝试更具体的查询",
                "查看系统帮助文档",
                "联系技术支持"
            ]
        
        return suggestions[:3]  # 最多返回3个建议
    
    async def get_device_summary(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """获取设备摘要信息"""
        try:
            if device_id:
                query = f"设备ID {device_id} 的详细信息"
            else:
                query = "所有设备的概览信息"
            
            result = await self.query(query)
            return {
                "success": result.success,
                "summary": result.message,
                "data": result.data
            }
            
        except Exception as e:
            logger.error(f"获取设备摘要失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_alert_summary(self, alert_level: Optional[str] = None) -> Dict[str, Any]:
        """获取告警摘要信息"""
        try:
            if alert_level:
                query = f"{alert_level}级别的告警信息"
            else:
                query = "所有告警的概览信息"
            
            result = await self.query(query)
            return {
                "success": result.success,
                "summary": result.message,
                "data": result.data
            }
            
        except Exception as e:
            logger.error(f"获取告警摘要失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_by_keywords(self, keywords: List[str], search_type: str = "both") -> Dict[str, Any]:
        """根据关键词搜索"""
        try:
            query = f"搜索包含关键词 {', '.join(keywords)} 的{search_type}信息"
            result = await self.query(query)
            
            return {
                "success": result.success,
                "results": result.message,
                "keywords": keywords,
                "search_type": search_type
            }
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_recommendations(self, context: str) -> List[str]:
        """获取智能推荐"""
        try:
            query = f"基于以下情况，请提供相关建议和推荐操作：{context}"
            result = await self.query(query)
            
            if result.success:
                return result.suggestions + ["查看详细分析报告", "设置自动化规则"]
            else:
                return ["检查系统状态", "联系技术支持"]
                
        except Exception as e:
            logger.error(f"获取推荐失败: {e}")
            return ["系统异常，请稍后重试"]
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.milvus_client or not await self.milvus_client.health_check():
                return False
            
            if not self.embedding_service or not await self.embedding_service.health_check():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"查询Agent健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭Agent"""
        try:
            if self.milvus_client:
                await self.milvus_client.disconnect()
            
            if self.embedding_service:
                self.embedding_service.close()
            
            logger.info("查询Agent已关闭")
            
        except Exception as e:
            logger.error(f"关闭查询Agent失败: {e}")


# 导入datetime
from datetime import datetime