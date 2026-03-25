import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL")
)

# ========== 第一步：定义工具 ==========
tools = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a given city. "
            "Use this when the user asks about weather conditions, "
            "temperature, or whether they need an umbrella. "
            "Returns temperature in Celsius, conditions, and humidity. "
            "Does NOT provide weather forecasts or historical data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Beijing', 'San Francisco'"
                }
            },
            "required": ["city"]
        }
    }
]

# ========== 第二步：模拟工具执行 ==========
def execute_tool(tool_name, tool_input):
    """模拟工具执行 这里是 mock 的数据"""
    if tool_name == "get_weather":
        # 模拟数据，不需要真实 API
        mock_data = {
            "Beijing": {"temp": 18, "conditions": "晴", "humidity": 35},
            "Shanghai": {"temp": 22, "conditions": "多云", "humidity": 65},
            "San Francisco": {"temp": 15, "conditions": "Foggy", "humidity": 78},
        }
        city = tool_input.get("city", "")
        if city in mock_data:
            return mock_data[city]
        return {"error": True, "message": f"No weather data for '{city}'", "recoverable": True, "suggestion": "Try a major city like Beijing, Shanghai, or San Francisco."}
    return {"error": True, "message": f"Unknown tool: {tool_name}"}

# ========== 第三步：发送请求 ==========
user_message = "北京今天天气怎么样？需要带伞吗？"

print(f"👤 用户: {user_message}\n")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": user_message}]
)

print(f"🔍 stop_reason: {response.stop_reason}")
print(f"📦 content blocks: {len(response.content)}\n")

# ========== 第四步：处理 tool_use 响应 ==========
if response.stop_reason == "tool_use":
    # 提取 tool_use block
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    tool_name = tool_use_block.name
    tool_input = tool_use_block.input
    tool_use_id = tool_use_block.id

    print(f"🔧 模型要调用: {tool_name}")
    print(f"📋 参数: {json.dumps(tool_input, ensure_ascii=False)}\n")

    # 执行工具
    tool_result = execute_tool(tool_name, tool_input)
    print(f"📊 工具返回: {json.dumps(tool_result, ensure_ascii=False)}\n")

    # ========== 第五步：把结果发回给模型 ==========
    final_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content},  # 模型的 tool_use 响应
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    }
                ]
            }
        ]
    )

    print(f"🤖 最终回答: {final_response.content[0].text}")

else:
    # 模型直接回答，没有调用工具
    print(f"🤖 直接回答: {response.content[0].text}")