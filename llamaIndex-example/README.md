# 设备状态告警Agent系统

基于LangChain + LlamaIndex的智能设备状态告警系统，支持多模态数据分析和智能告警决策。

## 系统架构

- **向量数据库**: Milvus
- **本地模型**: 
  - Ollama bge-m3 (文本嵌入)
  - Ollama deepseek-coder:r1 (智能问答)
- **框架**: LangChain + LlamaIndex
- **API**: FastAPI

## 功能特点

- ✅ 设备状态数据采集和存储
- ✅ 基于向量相似度的智能检索
- ✅ 多维度告警规则配置
- ✅ 实时告警决策和通知
- 🚧 设备图片分析（规划中）
- 🚧 电量监控告警（规划中）
- 🚧 用量统计分析（规划中）
- 🚧 振动数据监测（规划中）

## 项目结构

```
llamaIndex-example/
├── README.md
├── requirements.txt
├── .env
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── device.py
│   │   └── alert.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── milvus_client.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── alert_agent.py
│   │   └── query_agent.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── device_service.py
│   │   └── alert_service.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── devices.py
│   │       └── alerts.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── embeddings.py
├── data/
│   ├── devices/
│   └── images/
├── logs/
└── tests/
    ├── __init__.py
    ├── test_agents.py
    └── test_services.py
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动Ollama服务

```bash
# 启动Ollama
ollama serve

# 拉取所需模型
ollama pull bge-m3
ollama pull deepseek-coder:r1
```

### 3. 启动Milvus

```bash
# 使用Docker启动Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v $(pwd)/milvus:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 4. 运行系统

```bash
# 启动API服务
python -m src.api.main

# 或使用uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API接口

### 设备管理
- `POST /api/devices` - 添加设备
- `GET /api/devices` - 获取设备列表
- `GET /api/devices/{device_id}` - 获取设备详情
- `PUT /api/devices/{device_id}` - 更新设备状态

### 告警管理
- `GET /api/alerts` - 获取告警列表
- `POST /api/alerts/query` - 智能告警查询
- `POST /api/alerts/rules` - 配置告警规则

## 配置说明

主要配置项在`.env`文件中：

- `OLLAMA_BASE_URL`: Ollama服务地址
- `EMBEDDING_MODEL`: 嵌入模型名称
- `LLM_MODEL`: 语言模型名称
- `MILVUS_HOST`: Milvus服务地址
- `MILVUS_PORT`: Milvus服务端口
- 告警阈值配置等

## 开发计划

### Phase 1 (当前)
- [x] 基础架构搭建
- [x] 设备数据模型
- [x] Milvus向量存储
- [x] 基础告警逻辑

### Phase 2 (下一步)
- [ ] 设备图片分析
- [ ] 多模态数据融合
- [ ] 高级告警规则引擎
- [ ] 实时数据流处理

### Phase 3 (未来)
- [ ] 机器学习预测模型
- [ ] 自适应告警阈值
- [ ] 可视化监控面板
- [ ] 移动端支持

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License