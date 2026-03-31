"""
任务一：跑通最基础的 tool calling
==============================
目标：理解千问 tool calling 的完整请求-响应结构

核心流程：
  定义工具 → 发送请求 → 模型返回 tool_calls → 你执行工具 → 把结果发回 → 模型生成最终回答

运行: python 01_basic_tool_calling.py
"""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

# ========== 第一步：定义工具 ==========
# 注意 OpenAI 格式比 Anthropic 多一层 {"type": "function", "function": {...}} 的包装
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a given city. "
                "Use this when the user asks about weather conditions, "
                "temperature, or whether they need an umbrella. "
                "Returns temperature in Celsius, conditions, and humidity. "
                "Does NOT provide weather forecasts or historical data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name in English, e.g. 'Beijing', 'Shanghai'. Always use English names, not Chinese."
                    }
                },
                "required": ["city"]
            }
        }
    }
]


# ========== 第二步：模拟工具执行 ==========
def execute_tool(tool_name, tool_args):
    """模拟工具执行 — 实际项目中这里会调用真实 API"""
    if tool_name == "get_weather":
        mock_data = {
            "Beijing": {"temp": 18, "conditions": "晴", "humidity": 35},
            "Shanghai": {"temp": 22, "conditions": "多云", "humidity": 65},
            "San Francisco": {"temp": 15, "conditions": "Foggy", "humidity": 78},
        }
        city = tool_args.get("city", "")
        if city in mock_data:
            return json.dumps(mock_data[city], ensure_ascii=False)
        return json.dumps({
            "error": True,
            "message": f"No weather data for '{city}'",
            "recoverable": True,
            "suggestion": "Try a major city like Beijing, Shanghai, or San Francisco."
        }, ensure_ascii=False)

    return json.dumps({"error": True, "message": f"Unknown tool: {tool_name}"})


# ========== 第三步：发送请求 ==========
user_message = "北京今天天气怎么样？需要穿羽绒服吗？"
# user_message = "你好"

print(f"👤 用户: {user_message}\n")

messages = [{"role": "user", "content": user_message}]

response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
)

assistant_message = response.choices[0].message
finish_reason = response.choices[0].finish_reason

print(f"🔍 finish_reason: {finish_reason}")
print(f"📦 content: {assistant_message.content}")
print(f"📦 tool_calls: {assistant_message.tool_calls}\n")

# ========== 第四步：处理 tool_calls 响应 ==========
#
# 关键区别（对比 Anthropic）：
#   Anthropic: stop_reason == "tool_use"，工具调用在 content blocks 里
#   OpenAI/千问: finish_reason == "tool_calls"，工具调用在 message.tool_calls 里
#
if finish_reason == "tool_calls" and assistant_message.tool_calls:
    # 把模型的响应（含 tool_calls）加入消息历史
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_call_id = tool_call.id

        print(f"🔧 模型要调用: {tool_name}")
        print(f"📋 参数: {json.dumps(tool_args, ensure_ascii=False)}")
        print(f"🆔 tool_call_id: {tool_call_id}\n")

        # 执行工具
        tool_result = execute_tool(tool_name, tool_args)
        print(f"📊 工具返回: {tool_result}\n")

        # ========== 第五步：把结果发回给模型 ==========
        #
        # 关键区别（对比 Anthropic）：
        #   Anthropic: role="user" + type="tool_result" + tool_use_id
        #   OpenAI/千问: role="tool" + tool_call_id
        #
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result
        })

    # 发送最终请求，让模型基于工具结果生成回答
    final_response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
    )

    print(f"🤖 最终回答: {final_response.choices[0].message.content}")

else:
    # 模型直接回答，没有调用工具
    print(f"🤖 直接回答: {assistant_message.content}")

# ========== 完整的消息历史（供学习观察） ==========
print(f"\n{'=' * 60}")
print("📝 完整消息历史：")
for i, msg in enumerate(messages):
    if isinstance(msg, dict):
        role = msg.get("role", "?")
        content = msg.get("content", "")[:80]
        print(f"  [{i}] {role}: {content}")
    else:
        # ChatCompletionMessage 对象
        role = msg.role
        tc = msg.tool_calls
        print(f"  [{i}] {role}: content={msg.content}, tool_calls={len(tc) if tc else 0}个")
