from typing import List, Dict, Any
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Milvus
from ..config.settings import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME
)

class AlertRetriever:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL
        )
        
        # 初始化向量存储
        self.vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT
            }
        )
        
        # 初始化文档分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 初始化检索器
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=InMemoryStore(),
            child_splitter=self.text_splitter
        )

    def add_alert(self, alert_data: Dict[str, Any]):
        """添加告警数据到检索器"""
        # 将告警数据转换为文本
        alert_text = self._format_alert_text(alert_data)
        
        # 添加文档
        self.retriever.add_documents([{
            "page_content": alert_text,
            "metadata": alert_data
        }])

    def search_similar_alerts(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似告警"""
        docs = self.retriever.get_relevant_documents(query, k=k)
        return [doc.metadata for doc in docs]

    def _format_alert_text(self, alert_data: Dict[str, Any]) -> str:
        """将告警数据格式化为文本"""
        return f"""
设备ID: {alert_data['device_id']}
设备名称: {alert_data['device_name']}
设备类型: {alert_data['device_type']}
位置: {alert_data['location']}
告警类型: {alert_data['alert_type']}
告警等级: {alert_data['alert_level']}
告警信息: {alert_data['message']}
指标数据:
- 电量: {alert_data['metrics']['battery_level']}%
- 使用次数: {alert_data['metrics']['usage_count']}
- 振动级别: {alert_data['metrics']['vibration_level']}
- 温度: {alert_data['metrics']['temperature']}°C
- 湿度: {alert_data['metrics']['humidity']}%
时间: {alert_data['timestamp']}
""" 