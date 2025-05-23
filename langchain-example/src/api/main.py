from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uvicorn
from ..agents.alert_agent import AlertAgent
from ..data.mock_data import generate_mock_dataset
from ..serve.alert_serve import alert_serve

app = FastAPI(title="设备告警智能分析系统")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化告警Agent
alert_agent = AlertAgent()

# 注册LangServe路由
app.include_router(alert_serve.router)

@app.post("/api/alerts/analyze")
async def analyze_alert(alert_data: Dict[str, Any]):
    """分析设备告警"""
    try:
        result = alert_agent.analyze_alert(alert_data)
        return {"status": "success", "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mock-data")
async def get_mock_data(num_devices: int = 10, alerts_per_device: int = 5):
    """获取模拟数据"""
    try:
        dataset = generate_mock_dataset(num_devices, alerts_per_device)
        return {"status": "success", "data": dataset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 