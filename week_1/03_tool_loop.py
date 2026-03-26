"""
任务三：实现 Tool Call Loop 自动循环
====================================
目标：把手动的"调用-返回"过程封装成自动循环

这是构建任何 Agent 的核心骨架：
  1. 发送消息给模型
  2. 如果模型要调用工具 → 执行 → 把结果追加到消息历史 → 回到 1
  3. 如果模型直接回答 → 结束

运行: python 03_tool_loop.py
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


# ========== 工具定义 ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a given city. "
                "Returns temperature (Celsius), conditions, and humidity. "
                "Use when the user asks about weather. "
                "Does NOT provide forecasts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Beijing'"
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
                "Returns time in HH:MM and timezone name. "
                "Use when the user asks about current time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, e.g. 'Asia/Shanghai'"
                    }
                },
                "required": ["timezone"]
            }
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
    return data.get(city, {
        "error": True, "message": f"No data for '{city}'", "recoverable": True
    })


def get_current_time(timezone):
    times = {
        "Asia/Shanghai": {"time": "15:30", "timezone": "CST"},
        "Asia/Tokyo": {"time": "16:30", "timezone": "JST"},
        "America/New_York": {"time": "02:30", "timezone": "EST"},
    }
    return times.get(timezone, {
        "error": True, "message": f"Unknown timezone '{timezone}'", "recoverable": True
    })


# 注册表：工具名 → 执行函数
# 这个模式很重要 — 解耦了"工具定义"和"工具实现"
TOOL_REGISTRY = {
    "get_weather": lambda args: get_weather(args["city"]),
    "get_current_time": lambda args: get_current_time(args["timezone"]),
}


# ========== 核心：Agent Loop ==========
def run_agent(user_message, max_turns=5):
    """
    Agent 的核心循环：

    while 没有结束:
        1. 发送 messages 给模型
        2. 如果 finish_reason == "tool_calls":
              执行工具 → 追加结果到 messages → 继续循环
        3. 如果 finish_reason == "stop":
              输出最终回答 → 结束

    max_turns 防止无限循环（安全机制）
    """
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_message}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=tools,
        )

        assistant_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"finish_reason: {finish_reason}")

        # ---- 出口条件：模型直接回答，没有工具调用 ----
        if finish_reason == "stop" or not assistant_message.tool_calls:
            final_text = assistant_message.content or "[无文本输出]"
            print(f"\n🤖 最终回答: {final_text}")
            return final_text

        # ---- 模型想调用工具 ----
        # 把模型的响应（含 tool_calls）加入消息历史
        messages.append(assistant_message)

        # 执行所有工具调用
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 调用: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

            # 从注册表中查找并执行工具
            if tool_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[tool_name](tool_args)
            else:
                result = {"error": True, "message": f"Tool '{tool_name}' not found"}

            result_str = json.dumps(result, ensure_ascii=False)
            print(f"📊 结果: {result_str}")

            # 把工具结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })

        # 循环继续 → 回到顶部，带着新的 messages 再次请求模型

    print("\n⚠️ 达到最大轮次限制")
    return None


# ========== 测试 ==========
if __name__ == "__main__":
    # 测试 1：单工具调用
    run_agent("北京今天天气怎么样？")

    # 测试 2：需要两个工具
    run_agent("东京现在几点了？气温多少度？")

    # 测试 3：不需要工具
    run_agent("给我讲个笑话")

    # 测试 4：多城市查询
    run_agent("帮我查一下纽约的天气和时间")

    # 测试 5：未知城市（测试错误处理）
    run_agent("伦敦天气怎么样？")