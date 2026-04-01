"""
实验三：tools + response_format 互斥性验证
============================================
同时传入 tools 和 response_format，观察千问的实际行为：
  - 是报错？
  - 还是忽略其中一个？
  - 还是两者都生效？
  - 这直接决定周二方案 C 的可行性

测试矩阵：
  A. 只传 tools（基线）
  B. 只传 response_format（基线）
  C. tools + response_format=json_object
  D. tools + response_format=json_schema
  E. 模型选择调用工具时，response_format 是否影响最终回答格式
"""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "search_agent"))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = "qwen-plus"


# --- 工具定义（简化版，只用一个） ---
SIMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# --- response_format 定义 ---
JSON_OBJECT_FORMAT = {"type": "json_object"}

JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "structured_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        },
    },
}


def run_test(label: str, user_prompt: str, tools=None, response_format=None):
    """运行单个测试，记录结果"""
    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"  tools:           {'有' if tools else '无'}")
    print(f"  response_format: {response_format['type'] if response_format else '无'}")
    print(f"  user_prompt:     {user_prompt}")
    print(f"{'='*60}")

    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。如果需要查天气，请使用工具。用中文回答。"},
            {"role": "user", "content": user_prompt},
        ],
    }
    if tools:
        kwargs["tools"] = tools
    if response_format:
        kwargs["response_format"] = response_format

    try:
        resp = client.chat.completions.create(**kwargs)

        msg = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason

        print(f"\nfinish_reason: {finish_reason}")
        print(f"有 tool_calls: {bool(msg.tool_calls)}")
        print(f"有 content:    {bool(msg.content)}")

        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  工具调用: {tc.function.name}({tc.function.arguments})")

        if msg.content:
            print(f"\n输出内容:\n{msg.content[:500]}")
            # 尝试 JSON 解析
            try:
                parsed = json.loads(msg.content)
                print(f"\n✅ 输出是合法 JSON, keys: {list(parsed.keys())}")
            except json.JSONDecodeError:
                print(f"\n⚠️ 输出不是 JSON（纯文本）")

        return {
            "success": True,
            "finish_reason": finish_reason,
            "has_tool_calls": bool(msg.tool_calls),
            "has_content": bool(msg.content),
            "content": msg.content,
            "tool_calls": [
                {"name": tc.function.name, "args": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
        }

    except Exception as e:
        print(f"\n❌ 调用失败: {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def run_full_loop_test(label: str, user_prompt: str, response_format=None):
    """
    完整循环测试：如果模型调用了工具，模拟返回工具结果，
    观察最终回答是否受 response_format 约束。
    """
    print(f"\n{'='*60}")
    print(f"完整循环测试: {label}")
    print(f"{'='*60}")

    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。如果需要查天气，请使用工具。用中文回答。"},
            {"role": "user", "content": user_prompt},
        ],
        "tools": SIMPLE_TOOLS,
    }
    if response_format:
        kwargs["response_format"] = response_format

    try:
        # 第一轮：期望模型调用工具
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason

        print(f"\n第一轮 - finish_reason: {finish_reason}")

        if not msg.tool_calls:
            print(f"模型没有调用工具，直接回答了: {msg.content[:200] if msg.content else '无'}")
            return {"success": True, "note": "模型未调用工具", "content": msg.content}

        # 模拟工具返回
        tool_call = msg.tool_calls[0]
        print(f"工具调用: {tool_call.function.name}({tool_call.function.arguments})")

        fake_result = json.dumps({"city": "北京", "temperature": "22°C", "weather": "晴"}, ensure_ascii=False)

        kwargs["messages"].append(msg)
        kwargs["messages"].append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": fake_result,
        })

        # 第二轮：模型基于工具结果回答
        resp2 = client.chat.completions.create(**kwargs)
        msg2 = resp2.choices[0].message
        finish_reason2 = resp2.choices[0].finish_reason

        print(f"\n第二轮 - finish_reason: {finish_reason2}")
        print(f"输出:\n{msg2.content[:500] if msg2.content else '无'}")

        if msg2.content:
            try:
                parsed = json.loads(msg2.content)
                print(f"\n✅ 最终回答是合法 JSON, keys: {list(parsed.keys())}")
            except json.JSONDecodeError:
                print(f"\n⚠️ 最终回答不是 JSON")

        return {
            "success": True,
            "final_content": msg2.content,
            "final_finish_reason": finish_reason2,
        }

    except Exception as e:
        print(f"\n❌ 调用失败: {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


# ===================================================================
# 测试用例
# ===================================================================

if __name__ == "__main__":
    results = {}

    # --- A. 基线：只传 tools，触发工具调用的问题 ---
    results["A_tools_only"] = run_test(
        label="A. 基线 - 只传 tools",
        user_prompt="北京今天天气怎么样？",
        tools=SIMPLE_TOOLS,
    )

    # --- B. 基线：只传 response_format ---
    results["B_format_only"] = run_test(
        label="B. 基线 - 只传 response_format (json_object)",
        user_prompt="1+1 等于几？",
        response_format=JSON_OBJECT_FORMAT,
    )

    # --- C. tools + json_object，问一个需要工具的问题 ---
    results["C_both_tool_question"] = run_test(
        label="C. tools + json_object - 需要工具的问题",
        user_prompt="北京今天天气怎么样？",
        tools=SIMPLE_TOOLS,
        response_format=JSON_OBJECT_FORMAT,
    )

    # --- D. tools + json_object，问一个不需要工具的问题 ---
    results["D_both_no_tool_question"] = run_test(
        label="D. tools + json_object - 不需要工具的问题",
        user_prompt="1+1 等于几？",
        tools=SIMPLE_TOOLS,
        response_format=JSON_OBJECT_FORMAT,
    )

    # --- E. tools + json_schema ---
    results["E_tools_json_schema"] = run_test(
        label="E. tools + json_schema - 需要工具的问题",
        user_prompt="北京今天天气怎么样？",
        tools=SIMPLE_TOOLS,
        response_format=JSON_SCHEMA_FORMAT,
    )

    # --- F. 完整循环：tools + json_schema，工具调用后最终回答的格式 ---
    results["F_full_loop_with_schema"] = run_full_loop_test(
        label="F. 完整循环 - tools + json_schema",
        user_prompt="北京今天天气怎么样？",
        response_format=JSON_SCHEMA_FORMAT,
    )

    # --- G. 完整循环基线：tools 无 response_format ---
    results["G_full_loop_no_format"] = run_full_loop_test(
        label="G. 完整循环基线 - 只有 tools",
        user_prompt="北京今天天气怎么样？",
    )

    # --- 汇总 ---
    print(f"\n\n{'='*60}")
    print("📊 汇总 - 互斥性测试结果")
    print(f"{'='*60}")
    for name, r in results.items():
        status = "✅" if r.get("success") else "❌"
        extra = ""
        if r.get("error"):
            extra = f" | 错误: {r['error'][:60]}"
        elif r.get("has_tool_calls"):
            extra = " | 调用了工具"
        elif r.get("content"):
            try:
                json.loads(r["content"])
                extra = " | 返回 JSON"
            except (json.JSONDecodeError, TypeError):
                extra = " | 返回纯文本"
        elif r.get("note"):
            extra = f" | {r['note']}"
        print(f"  {status} {name:30s}{extra}")

    print(f"\n💡 关键结论:")
    print(f"   C 测试（tools + json_object + 工具问题）=> 看模型是调用工具还是返回 JSON")
    print(f"   E 测试（tools + json_schema + 工具问题）=> 看是否报错")
    print(f"   F 测试（完整循环 + json_schema）=> 看工具调用后最终回答是否符合 schema")
