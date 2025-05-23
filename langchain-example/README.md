# 设备状态告警智能Agent系统简单实现

基于LangChain和Milvus的设备状态告警智能Agent系统，支持多模态数据分析和告警。

## 功能特点

- 支持设备图片、信息、电量、用量、振动等多维度数据采集
- 使用Ollama本地模型进行数据处理和分析
- 基于Milvus向量数据库进行高效检索
- 智能告警决策系统

## 项目结构

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── mock_data.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── device.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── milvus_store.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── alert_agent.py
│   └── api/
│       ├── __init__.py
│       └── main.py
└── .env
```

## 安装说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
复制 `.env.example` 到 `.env` 并填写相应配置

3. 启动服务：
```bash
python src/api/main.py
```