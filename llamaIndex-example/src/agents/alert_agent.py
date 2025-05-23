"""告警智能Agent"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field
from loguru import logger

from ..config import get_settings
from ..models import Device, Alert, AlertLevel, AlertType, AlertRule, DeviceStatus
from ..vector_store import MilvusClient
from ..utils import EmbeddingService, get_embedding_service


class AlertAnalysisResult(BaseModel):
    """告警分析结果"""
    should_alert: bool = Field(..., description="是否应该触发告警")
    alert_level: AlertLevel = Field(..., description="告警级别")
    alert_type: AlertType = Field(..., description="告警类型")
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警消息")
    confidence: float = Field(..., description="置信度", ge=0, le=1)
    reasoning: str = Field(..., description="分析推理过程")
    suggested_actions: List[str] = Field(default_factory=list, description="建议操作")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")


class DeviceAnalysisTool(BaseTool):
    """设备分析工具"""
    name = "device_analysis"
    description = "分析设备状态和指标数据，判断是否需要告警"
    
    def __init__(self, milvus_client: MilvusClient, embedding_service: EmbeddingService):
        super().__init__()
        self.milvus_client = milvus_client
        self.embedding_service = embedding_service
    
    def _run(self, device_data: str) -> str:
        """同步运行（不推荐使用）"""
        return asyncio.run(self._arun(device_data))
    
    async def _arun(self, device_data: str) -> str:
        """异步分析设备数据"""
        try:
            # 解析设备数据
            device_dict = json.loads(device_data)
            device = Device(**device_dict)
            
            # 生成查询嵌入
            query_text = f"设备告警分析 {device.to_vector_text()}"
            query_embedding = await self.embedding_service.generate_embedding(query_text)
            
            # 搜索相似的历史告警
            similar_alerts = await self.milvus_client.search_alerts(
                query_embedding=query_embedding,
                top_k=3,
                score_threshold=0.6
            )
            
            # 搜索相似设备
            similar_devices = await self.milvus_client.search_devices(
                query_embedding=query_embedding,
                top_k=3,
                score_threshold=0.6
            )
            
            # 构建分析结果
            analysis = {
                "device_info": device.dict(),
                "similar_alerts_count": len(similar_alerts),
                "similar_devices_count": len(similar_devices),
                "historical_patterns": [alert.get_summary() for alert in similar_alerts[:2]]
            }
            
            return json.dumps(analysis, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"设备分析工具执行失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class AlertHistoryTool(BaseTool):
    """告警历史工具"""
    name = "alert_history"
    description = "查询设备的历史告警记录和模式"
    
    def __init__(self, milvus_client: MilvusClient, embedding_service: EmbeddingService):
        super().__init__()
        self.milvus_client = milvus_client
        self.embedding_service = embedding_service
    
    def _run(self, query: str) -> str:
        """同步运行"""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """异步查询告警历史"""
        try:
            # 生成查询嵌入
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # 搜索相关告警
            alerts = await self.milvus_client.search_alerts(
                query_embedding=query_embedding,
                top_k=5,
                score_threshold=0.5
            )
            
            # 分析告警模式
            patterns = {
                "total_alerts": len(alerts),
                "alert_types": {},
                "alert_levels": {},
                "recent_trends": []
            }
            
            for alert in alerts:
                # 统计告警类型
                alert_type = alert.alert_type.value
                patterns["alert_types"][alert_type] = patterns["alert_types"].get(alert_type, 0) + 1
                
                # 统计告警级别
                alert_level = alert.alert_level.value
                patterns["alert_levels"][alert_level] = patterns["alert_levels"].get(alert_level, 0) + 1
                
                # 记录最近趋势
                if len(patterns["recent_trends"]) < 3:
                    patterns["recent_trends"].append({
                        "time": alert.created_at.isoformat(),
                        "type": alert_type,
                        "level": alert_level,
                        "message": alert.message[:100]
                    })
            
            return json.dumps(patterns, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"告警历史工具执行失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class AlertAgent:
    """告警智能Agent"""
    
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
        
        # 告警规则缓存
        self.alert_rules: List[AlertRule] = []
        
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
                DeviceAnalysisTool(self.milvus_client, self.embedding_service),
                AlertHistoryTool(self.milvus_client, self.embedding_service)
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
            
            # 加载默认告警规则
            await self._load_default_rules()
            
            logger.info("告警Agent初始化完成")
            
        except Exception as e:
            logger.error(f"告警Agent初始化失败: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """
你是一个专业的设备状态告警分析专家。你的任务是：

1. 分析设备状态数据和指标
2. 判断是否需要触发告警
3. 确定告警级别和类型
4. 提供详细的分析推理
5. 建议相应的处理措施

分析时请考虑：
- 设备的历史告警模式
- 相似设备的表现
- 指标的异常程度
- 业务影响的严重性

请始终以JSON格式返回分析结果，包含以下字段：
- should_alert: 是否应该告警
- alert_level: 告警级别(info/warning/error/critical/fatal)
- alert_type: 告警类型
- title: 告警标题
- message: 详细消息
- confidence: 置信度(0-1)
- reasoning: 分析推理过程
- suggested_actions: 建议操作列表
- context: 相关上下文信息
"""
    
    async def _load_default_rules(self):
        """加载默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="battery_low",
                name="电池电量低告警",
                description="当设备电池电量低于阈值时触发",
                alert_type=AlertType.BATTERY_LOW,
                conditions={"battery_level": {"max": self.settings.alert_threshold_battery}},
                alert_level=AlertLevel.WARNING
            ),
            AlertRule(
                rule_id="usage_high",
                name="使用率过高告警",
                description="当设备使用率超过阈值时触发",
                alert_type=AlertType.USAGE_HIGH,
                conditions={"usage_rate": {"min": self.settings.alert_threshold_usage}},
                alert_level=AlertLevel.ERROR
            ),
            AlertRule(
                rule_id="vibration_high",
                name="振动过高告警",
                description="当设备振动强度超过阈值时触发",
                alert_type=AlertType.VIBRATION_HIGH,
                conditions={"vibration_level": {"min": self.settings.alert_threshold_vibration}},
                alert_level=AlertLevel.CRITICAL
            ),
            AlertRule(
                rule_id="device_offline",
                name="设备离线告警",
                description="当设备离线时触发",
                alert_type=AlertType.DEVICE_OFFLINE,
                conditions={"status": {"equals": "offline"}},
                alert_level=AlertLevel.ERROR
            )
        ]
        
        self.alert_rules = default_rules
        logger.info(f"加载了{len(default_rules)}个默认告警规则")
    
    async def analyze_device(self, device: Device) -> AlertAnalysisResult:
        """分析设备状态并判断是否需要告警"""
        try:
            # 首先检查基础规则
            rule_result = await self._check_basic_rules(device)
            if rule_result:
                return rule_result
            
            # 使用AI Agent进行深度分析
            analysis_input = f"""
请分析以下设备状态数据，判断是否需要触发告警：

设备信息：
{device.get_alert_context()}

请使用可用的工具分析设备历史数据和相似设备的表现，然后给出专业的告警建议。
"""
            
            # 调用Agent
            result = await self.agent_executor.ainvoke({
                "input": analysis_input
            })
            
            # 解析Agent返回的结果
            agent_output = result.get("output", "")
            
            # 尝试从输出中提取JSON
            analysis_data = self._parse_agent_output(agent_output)
            
            # 创建分析结果
            return AlertAnalysisResult(**analysis_data)
            
        except Exception as e:
            logger.error(f"设备分析失败: {e}")
            # 返回默认的错误分析结果
            return AlertAnalysisResult(
                should_alert=True,
                alert_level=AlertLevel.ERROR,
                alert_type=AlertType.DEVICE_FAULT,
                title="设备分析异常",
                message=f"设备 {device.device_id} 分析过程中出现异常: {str(e)}",
                confidence=0.5,
                reasoning="由于分析过程异常，建议人工检查设备状态",
                suggested_actions=["人工检查设备", "检查网络连接", "重启分析服务"]
            )
    
    async def _check_basic_rules(self, device: Device) -> Optional[AlertAnalysisResult]:
        """检查基础告警规则"""
        for rule in self.alert_rules:
            if not rule.enabled or not rule.matches_device(device.device_type.value):
                continue
            
            # 检查设备状态
            if "status" in rule.conditions:
                status_condition = rule.conditions["status"]
                if "equals" in status_condition:
                    if device.status.value == status_condition["equals"]:
                        return self._create_rule_alert(device, rule, "status", device.status.value)
            
            # 检查指标条件
            metrics = device.metrics
            for metric_name, condition in rule.conditions.items():
                if metric_name == "status":
                    continue
                
                metric_value = getattr(metrics, metric_name, None)
                if metric_value is not None and rule.evaluate_condition(metric_name, metric_value):
                    return self._create_rule_alert(device, rule, metric_name, metric_value)
        
        return None
    
    def _create_rule_alert(self, device: Device, rule: AlertRule, metric_name: str, value: Any) -> AlertAnalysisResult:
        """基于规则创建告警结果"""
        threshold = rule.conditions.get(metric_name, {})
        
        return AlertAnalysisResult(
            should_alert=True,
            alert_level=rule.alert_level,
            alert_type=rule.alert_type,
            title=f"{device.name} - {rule.name}",
            message=f"设备 {device.name} 的 {metric_name} 指标异常，当前值: {value}",
            confidence=0.9,
            reasoning=f"触发规则: {rule.name}，条件: {threshold}",
            suggested_actions=self._get_suggested_actions(rule.alert_type),
            context={
                "rule_id": rule.rule_id,
                "metric_name": metric_name,
                "trigger_value": value,
                "threshold": threshold
            }
        )
    
    def _get_suggested_actions(self, alert_type: AlertType) -> List[str]:
        """根据告警类型获取建议操作"""
        action_map = {
            AlertType.BATTERY_LOW: ["检查电池状态", "准备更换电池", "减少设备使用"],
            AlertType.USAGE_HIGH: ["检查设备负载", "优化使用策略", "考虑扩容"],
            AlertType.VIBRATION_HIGH: ["检查设备固定", "检查机械部件", "停机检修"],
            AlertType.DEVICE_OFFLINE: ["检查网络连接", "检查设备电源", "重启设备"],
            AlertType.DEVICE_FAULT: ["联系技术支持", "检查设备日志", "准备备用设备"]
        }
        
        return action_map.get(alert_type, ["联系技术支持", "检查设备状态"])
    
    def _parse_agent_output(self, output: str) -> Dict[str, Any]:
        """解析Agent输出"""
        try:
            # 尝试直接解析JSON
            if output.strip().startswith("{"):
                return json.loads(output)
            
            # 尝试从文本中提取JSON
            import re
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # 如果无法解析，返回默认结果
            return {
                "should_alert": False,
                "alert_level": "info",
                "alert_type": "custom",
                "title": "AI分析结果",
                "message": output,
                "confidence": 0.5,
                "reasoning": "AI Agent分析完成",
                "suggested_actions": [],
                "context": {}
            }
            
        except Exception as e:
            logger.error(f"解析Agent输出失败: {e}")
            return {
                "should_alert": False,
                "alert_level": "info",
                "alert_type": "custom",
                "title": "分析结果解析失败",
                "message": "无法解析AI分析结果",
                "confidence": 0.1,
                "reasoning": f"输出解析错误: {str(e)}",
                "suggested_actions": ["检查AI模型配置"],
                "context": {"raw_output": output}
            }
    
    async def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        for i, rule in enumerate(self.alert_rules):
            if rule.rule_id == rule_id:
                del self.alert_rules[i]
                logger.info(f"移除告警规则: {rule_id}")
                return True
        return False
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """获取所有告警规则"""
        return self.alert_rules.copy()
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.milvus_client or not await self.milvus_client.health_check():
                return False
            
            if not self.embedding_service or not await self.embedding_service.health_check():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"告警Agent健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭Agent"""
        try:
            if self.milvus_client:
                await self.milvus_client.disconnect()
            
            if self.embedding_service:
                self.embedding_service.close()
            
            logger.info("告警Agent已关闭")
            
        except Exception as e:
            logger.error(f"关闭告警Agent失败: {e}")