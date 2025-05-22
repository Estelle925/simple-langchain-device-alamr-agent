from typing import List, Dict, Any
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
from ..config.settings import OLLAMA_BASE_URL, LLM_MODEL, ALERT_THRESHOLDS
from ..retrieval.alert_retriever import AlertRetriever
from ..chains.alert_qa_chain import AlertQAChain

class AlertAgentPromptTemplate(StringPromptTemplate):
    template = """你是一个专业的设备告警分析专家。你的任务是分析设备告警信息并提供专业的建议。

当前对话历史：
{chat_history}

当前设备告警信息：
{current_alert}

相似历史告警：
{similar_alerts}

请根据以上信息，分析当前告警的严重程度，并提供处理建议。
你可以使用以下工具：

{tools}

请使用以下格式：
思考：你的思考过程
行动：要使用的工具名称
行动输入：工具的输入参数
观察：工具返回的结果
...（可以有多轮思考和行动）
最终答案：你的最终分析和建议

{agent_scratchpad}"""

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("agent_scratchpad")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\n思考：{action.log}\n"
            thoughts += f"行动：{action.tool}\n"
            thoughts += f"行动输入：{action.tool_input}\n"
            thoughts += f"观察：{observation}\n"
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)

class AlertAgent:
    def __init__(self):
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL
        )
        self.retriever = AlertRetriever()
        self.qa_chain = AlertQAChain()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # 定义工具
        self.tools = [
            Tool(
                name="search_similar_alerts",
                func=self.retriever.search_similar_alerts,
                description="搜索与当前告警相似的历史告警记录"
            ),
            Tool(
                name="check_alert_thresholds",
                func=self._check_alert_thresholds,
                description="检查告警指标是否超过阈值"
            ),
            Tool(
                name="analyze_alert_qa",
                func=self.qa_chain.analyze_alert,
                description="使用问答链分析告警信息"
            )
        ]
        
        # 初始化Agent
        self.prompt = AlertAgentPromptTemplate()
        self.agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=self.prompt),
            output_parser=self._parse_output,
            stop=["\n观察："],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        # 初始化Agent执行器
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def _check_alert_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """检查告警指标是否超过阈值"""
        results = {}
        for metric, value in metrics.items():
            if metric in ALERT_THRESHOLDS:
                thresholds = ALERT_THRESHOLDS[metric]
                if value <= thresholds["critical"]:
                    results[metric] = "critical"
                elif value <= thresholds["warning"]:
                    results[metric] = "warning"
                else:
                    results[metric] = "normal"
        return results

    def _parse_output(self, text: str) -> AgentAction | AgentFinish:
        """解析LLM输出"""
        if "最终答案：" in text:
            return AgentFinish(
                return_values={"output": text.split("最终答案：")[-1].strip()},
                log=text
            )
        
        # 解析工具调用
        action_match = text.split("行动：")[-1].split("\n")[0].strip()
        input_match = text.split("行动输入：")[-1].split("\n")[0].strip()
        
        return AgentAction(
            tool=action_match,
            tool_input=input_match,
            log=text
        )

    def analyze_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析告警信息"""
        # 搜索相似告警
        similar_alerts = self.retriever.search_similar_alerts(
            query=alert_data["message"]
        )
        
        # 准备Agent输入
        agent_input = {
            "current_alert": alert_data,
            "similar_alerts": similar_alerts
        }
        
        # 执行Agent分析
        result = self.agent_executor.run(agent_input)
        
        # 使用问答链进行补充分析
        qa_result = self.qa_chain.analyze_alert(
            alert_data,
            question="请详细分析这个告警的原因和影响，并提供具体的处理建议。"
        )
        
        return {
            "agent_analysis": result,
            "qa_analysis": qa_result["answer"],
            "similar_alerts": qa_result["source_documents"]
        } 