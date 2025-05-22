from typing import Dict, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from ..config.settings import OLLAMA_BASE_URL, LLM_MODEL
from ..retrieval.alert_retriever import AlertRetriever

class AlertQAChain:
    def __init__(self):
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL
        )
        self.retriever = AlertRetriever()
        
        # 定义提示模板
        self.prompt_template = PromptTemplate(
            template="""你是一个专业的设备告警分析专家。请根据以下信息回答问题。

历史告警信息：
{context}

当前问题：
{question}

请提供专业的分析和建议。""",
            input_variables=["context", "question"]
        )
        
        # 初始化问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def analyze_alert(self, alert_data: Dict[str, Any], question: str = None) -> Dict[str, Any]:
        """分析告警信息"""
        # 添加告警数据到检索器
        self.retriever.add_alert(alert_data)
        
        # 如果没有提供具体问题，使用默认问题
        if question is None:
            question = f"请分析这个{alert_data['alert_type']}告警的严重程度，并提供处理建议。"
        
        # 执行问答
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "device_id": doc.metadata["device_id"],
                    "alert_type": doc.metadata["alert_type"],
                    "message": doc.metadata["message"]
                }
                for doc in result["source_documents"]
            ]
        } 