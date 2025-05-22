import requests
import json
from data.mock_data import generate_mock_dataset

def test_alert_analysis():
    # 生成模拟数据
    dataset = generate_mock_dataset(num_devices=1, alerts_per_device=1)
    
    # 选择第一个告警进行测试
    test_alert = dataset[0]
    
    # 发送告警分析请求
    response = requests.post(
        "http://localhost:8000/api/alerts/analyze",
        json=test_alert
    )
    
    # 打印结果
    print("告警数据：")
    print(json.dumps(test_alert, indent=2, ensure_ascii=False))
    print("\n分析结果：")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_alert_analysis() 