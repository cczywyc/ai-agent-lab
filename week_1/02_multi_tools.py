import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
)

tools = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a given city. "
            "Use this when the user asks about weather, temperature, "
            "or whether they need an umbrella or jacket. "
            "Returns temperature (Celsius), conditions, and humidity. "
            "Does NOT provide forecasts or historical weather data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Beijing', 'Tokyo'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_current_time",
        "description": (
            "Get the current local time for a given timezone. "
            "Use this when the user asks what time it is in a specific location. "
            "Returns the current time in HH:MM format and the timezone name. "
            "Does NOT support scheduling or alarms — only queries current time."
        ),
        "input_schema": {
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
]

def execute_tool(tool_name, tool_input):
    if tool_name == "get_weather":
        mock_data = {
            "Beijing": {"temp": 18, "conditions": "晴", "humidity": 35},
            "Tokyo": {"temp": 20, "conditions": "Cloudy", "humidity": 55},
        }
        city = tool_input.get("city", "")
        return mock_data.get(city, {
            "error": True, "message": f"No data for '{city}'",
            "recoverable": True, "suggestion": "Try Beijing or Tokyo."
        })
    elif tool_name == "get_current_time":
        mock_times = {
            "Asia/Shanghai": {"time": "15:30", "timezone": "CST (UTC+8)"},
            "Asia/Tokyo": {"time": "16:30", "timezone": "JST (UTC+9)"},
            "America/New_York": {"time": "02:30", "timezone": "EST (UTC-5)"},
        }
        tz = tool_input.get("timezone", "")
        return mock_times.get(tz, {
            "error": True, "message": f"Unknown timezone '{tz}'",
            "recoverable": True, "suggestion": "Use IANA format like 'Asia/Shanghai'."
        })
    return {"error": True, "message": f"Unknown tool: {tool_name}"}

# ========== 测试不同的用户输入 ==========
test_queries = [
    "东京现在几点了？",
    "北京天气怎么样？",
    "纽约现在几点？天气冷不冷？",  # 可能需要调用两个工具
    "帮我写一首关于春天的诗",        # 不需要工具
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"👤 用户: {query}\n")

    messages = [{"role": "user", "content": query}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    print(f"   stop_reason: {response.stop_reason}")

    # 收集所有 tool_use blocks
    tool_blocks = [b for b in response.content if b.type == "tool_use"]
    text_blocks = [b for b in response.content if b.type == "text"]

    if text_blocks:
        print(f"   模型先说: {text_blocks[0].text[:80]}...")

    if not tool_blocks:
        print(f"🤖 直接回答: {response.content[0].text[:120]}")
        continue

    # 处理每个工具调用
    print(f"   需要调用 {len(tool_blocks)} 个工具:")
    tool_results_content = []
    for tb in tool_blocks:
        print(f"   🔧 {tb.name}({json.dumps(tb.input, ensure_ascii=False)})")
        result = execute_tool(tb.name, tb.input)
        print(f"   📊 → {json.dumps(result, ensure_ascii=False)}")
        tool_results_content.append({
            "type": "tool_result",
            "tool_use_id": tb.id,
            "content": json.dumps(result, ensure_ascii=False)
        })

    # 把所有工具结果一起发回
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results_content})

    final = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    print(f"\n🤖 最终回答: {final.content[0].text}")