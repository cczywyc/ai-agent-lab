import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
)

# ========== 工具定义 ==========
tools = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a given city. "
            "Returns temperature (Celsius), conditions, and humidity. "
            "Use when the user asks about weather. "
            "Does NOT provide forecasts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Beijing'"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_current_time",
        "description": (
            "Get the current local time for a given timezone. "
            "Returns time in HH:MM and timezone name. "
            "Use when the user asks about current time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone, e.g. 'Asia/Shanghai'"}
            },
            "required": ["timezone"]
        }
    }
]

# ========== 工具执行注册表 ==========
def get_weather(city):
    data = {
        "Beijing": {"temp": 18, "conditions": "晴", "humidity": 35},
        "Tokyo": {"temp": 20, "conditions": "Cloudy", "humidity": 55},
        "New York": {"temp": 8, "conditions": "Rainy", "humidity": 80},
    }
    return data.get(city, {"error": True, "message": f"No data for '{city}'", "recoverable": True})

def get_current_time(timezone):
    times = {
        "Asia/Shanghai": {"time": "15:30", "timezone": "CST"},
        "Asia/Tokyo": {"time": "16:30", "timezone": "JST"},
        "America/New_York": {"time": "02:30", "timezone": "EST"},
    }
    return times.get(timezone, {"error": True, "message": f"Unknown timezone '{timezone}'", "recoverable": True})

TOOL_REGISTRY = {
    "get_weather": lambda params: get_weather(params["city"]),
    "get_current_time": lambda params: get_current_time(params["timezone"]),
}

# ========== 核心：Agent Loop ==========
def run_agent(user_message, max_turns=5):
    """
    Agent 的核心循环：
    1. 发送消息给模型
    2. 如果模型要调用工具 → 执行 → 把结果追加到消息历史 → 回到 1
    3. 如果模型直接回答 → 结束
    """
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_message}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        print(f"stop_reason: {response.stop_reason}")

        # 如果模型直接回答（没有工具调用），结束循环
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if b.type == "text"),
                "[无文本输出]"
            )
            print(f"\n🤖 最终回答: {final_text}")
            return final_text

        # 模型想调用工具
        if response.stop_reason == "tool_use":
            # 把模型的响应加入消息历史
            messages.append({"role": "assistant", "content": response.content})

            # 执行所有工具调用
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    print(f"🔧 调用: {tool_name}({json.dumps(tool_input, ensure_ascii=False)})")

                    # 从注册表中查找并执行工具
                    if tool_name in TOOL_REGISTRY:
                        result = TOOL_REGISTRY[tool_name](tool_input)
                    else:
                        result = {"error": True, "message": f"Tool '{tool_name}' not found"}

                    print(f"📊 结果: {json.dumps(result, ensure_ascii=False)}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

            # 把工具结果加入消息历史
            messages.append({"role": "user", "content": tool_results})

    print("\n⚠️ 达到最大轮次限制")
    return None


# ========== 测试 ==========
if __name__ == "__main__":
    run_agent("北京今天天气怎么样？")
    run_agent("东京现在几点了？气温多少度？")
    run_agent("给我讲个笑话")  # 不需要工具
    run_agent("帮我查一下纽约的天气和时间")