"""
任务二：多工具 + 观察模型选择逻辑
=================================
目标：给 Agent 多个工具，观察模型如何"选择"调用哪个

重点观察：
  - 模型是否根据用户意图选对了工具？
  - 需要两个工具时，模型会不会在一次请求里都调用？
  - 不需要工具时，模型是不是直接回答？

运行: python 02_multi_tools.py
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

# ========== 定义两个工具 ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a given city. "
                "Use this when the user asks about weather, temperature, "
                "or whether they need an umbrella or jacket. "
                "Returns temperature (Celsius), conditions, and humidity. "
                "Does NOT provide forecasts or historical weather data."
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current local time for a given timezone. "
                "Use this when the user asks what time it is in a specific location. "
                "Returns the current time in HH:MM format and the timezone name. "
                "Does NOT support scheduling or alarms — only queries current time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone string, e.g. 'Asia/Shanghai', 'America/New_York'"
                    }
                },
                "required": ["timezone"]
            }
        }
    }
]


# ========== 工具执行函数 ==========
def execute_tool(tool_name, tool_args):
    if tool_name == "get_weather":
        mock_data = {
            "Beijing": {"temp": 18, "conditions": "晴", "humidity": 35},
            "Tokyo": {"temp": 20, "conditions": "Cloudy", "humidity": 55},
            "New York": {"temp": 8, "conditions": "Rainy", "humidity": 80},
        }
        city = tool_args.get("city", "")
        result = mock_data.get(city, {
            "error": True, "message": f"No data for '{city}'",
            "recoverable": True, "suggestion": "Try Beijing, Tokyo, or New York."
        })
        return json.dumps(result, ensure_ascii=False)

    elif tool_name == "get_current_time":
        mock_times = {
            "Asia/Shanghai": {"time": "15:30", "timezone": "CST (UTC+8)"},
            "Asia/Tokyo": {"time": "16:30", "timezone": "JST (UTC+9)"},
            "America/New_York": {"time": "02:30", "timezone": "EST (UTC-5)"},
        }
        tz = tool_args.get("timezone", "")
        result = mock_times.get(tz, {
            "error": True, "message": f"Unknown timezone '{tz}'",
            "recoverable": True, "suggestion": "Use IANA format like 'Asia/Shanghai'."
        })
        return json.dumps(result, ensure_ascii=False)

    return json.dumps({"error": True, "message": f"Unknown tool: {tool_name}"})


# ========== 测试不同的用户输入 ==========
test_queries = [
    "东京现在几点了？",                    # 预期：调用 get_current_time
    "北京天气怎么样？",                    # 预期：调用 get_weather
    "纽约现在几点？天气冷不冷？",          # 预期：调用两个工具（parallel tool calls）
    "帮我写一首关于春天的诗",              # 预期：不调用任何工具
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"👤 用户: {query}\n")

    messages = [{"role": "user", "content": query}]

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
    )

    assistant_message = response.choices[0].message
    finish_reason = response.choices[0].finish_reason

    print(f"   finish_reason: {finish_reason}")

    # 如果模型有文字输出（有时模型会先说一句话再调工具）
    if assistant_message.content:
        print(f"   模型先说: {assistant_message.content[:80]}")

    # 没有工具调用，直接回答
    if not assistant_message.tool_calls:
        print(f"🤖 直接回答: {assistant_message.content[:150]}")
        continue

    # 处理工具调用
    tool_calls = assistant_message.tool_calls
    print(f"   需要调用 {len(tool_calls)} 个工具:")

    # 把模型响应加入消息历史
    messages.append(assistant_message)

    # 执行每个工具并收集结果
    for tc in tool_calls:
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments)

        print(f"   🔧 {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

        result = execute_tool(tool_name, tool_args)
        print(f"   📊 → {result}")

        # 每个工具结果作为单独的 message 追加
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result
        })

    # 把所有工具结果发回，让模型生成最终回答
    final = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
    )

    print(f"\n🤖 最终回答: {final.choices[0].message.content}")