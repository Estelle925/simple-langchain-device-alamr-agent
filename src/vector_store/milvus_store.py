from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain.vectorstores import Milvus
from langchain.embeddings import OllamaEmbeddings
from ..config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL
)

class MilvusStore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL
        )
        self._connect()
        self._create_collection_if_not_exists()

    def _connect(self):
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )

    def _create_collection_if_not_exists(self):
        if not utility.has_collection(COLLECTION_NAME):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="device_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="alert_type", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="message", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
            ]
            schema = CollectionSchema(fields=fields, description="Device alerts collection")
            self.collection = Collection(name=COLLECTION_NAME, schema=schema)
            self.collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            )
        else:
            self.collection = Collection(COLLECTION_NAME)

    def add_alert(self, device_id: str, alert_type: str, message: str):
        # 生成文本嵌入
        embedding = self.embeddings.embed_query(message)
        
        # 准备插入数据
        data = [
            [device_id],
            [alert_type],
            [message],
            [embedding]
        ]
        
        # 插入数据
        self.collection.insert(data)
        self.collection.flush()

    def search_similar_alerts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        # 生成查询嵌入
        query_embedding = self.embeddings.embed_query(query)
        
        # 搜索相似告警
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["device_id", "alert_type", "message"]
        )
        
        return [
            {
                "device_id": hit.entity.get("device_id"),
                "alert_type": hit.entity.get("alert_type"),
                "message": hit.entity.get("message"),
                "distance": hit.distance
            }
            for hit in results[0]
        ] 