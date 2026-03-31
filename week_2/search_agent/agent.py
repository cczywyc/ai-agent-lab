"""
agent.py — Agent 核心循环与 Trace 记录
=======================================
包含两个运行模式：
  - run_agent()            — 基础版，简洁输出
  - run_agent_with_trace() — 带完整 trace 记录，用于调试和评测

Agent Loop 的核心逻辑：
  while 未结束 and 未超过最大轮次:
      1. 发送 messages 给模型
      2. 如果 finish_reason == "tool_calls" → 执行工具 → 追加结果 → 继续
      3. 如果 finish_reason == "stop" → 输出回答 → 结束
"""
import json
from datetime import datetime

from config import client, MODEL_NAME, MAX_TURNS, SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS, execute_tool


def run_agent(user_message: str, max_turns: int = MAX_TURNS) -> str | None:
    """
    基础 Agent 运行器。

    Args:
        user_message: 用户输入
        max_turns: 最大循环轮次（安全机制）

    Returns:
        模型的最终文本回答，或 None（如果超过最大轮次）
    """
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_message}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        assistant_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"finish_reason: {finish_reason}")

        # 出口：模型直接回答
        if finish_reason == "stop" or not assistant_message.tool_calls:
            final_text = assistant_message.content or "[无文本输出]"
            print(f"\n🤖 最终回答:\n{final_text}")
            return final_text

        # 模型要调用工具
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 调用: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

            result_str = execute_tool(tool_name, tool_args)
            print(f"📊 结果: {result_str[:200]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })

    print("\n⚠️ 达到最大轮次限制")
    return None


def run_agent_with_trace(user_message: str, max_turns: int = MAX_TURNS) -> dict:
    """
    带 Trace 记录的 Agent 运行器。

    除了正常运行 Agent 外，还会记录每一轮的工具调用、结果、错误等信息，
    用于调试、评测和复盘。

    Args:
        user_message: 用户输入
        max_turns: 最大循环轮次

    Returns:
        trace 字典，包含完整的执行记录
    """
    trace = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "model": MODEL_NAME,
        "turns": [],
        "final_answer": None,
        "total_tool_calls": 0,
        "errors": [],
        "finished": False
    }

    print(f"\n{'='*60}")
    print(f"👤 用户: {user_message}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for turn in range(max_turns):
        turn_record = {"turn": turn + 1, "tool_calls": []}
        print(f"\n--- Turn {turn + 1} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        assistant_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"finish_reason: {finish_reason}")
        turn_record["finish_reason"] = finish_reason

        # 记录 token 用量（如果 API 返回了的话）
        if response.usage:
            turn_record["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        # 出口：模型直接回答
        if finish_reason == "stop" or not assistant_message.tool_calls:
            final_text = assistant_message.content or "[无文本输出]"
            trace["final_answer"] = final_text
            trace["finished"] = True
            trace["turns"].append(turn_record)
            print(f"\n🤖 最终回答:\n{final_text}")
            break

        # 模型要调用工具
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 调用: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

            result_str = execute_tool(tool_name, tool_args)

            # 解析结果判断是否有错误
            try:
                result_obj = json.loads(result_str)
                has_error = isinstance(result_obj, dict) and result_obj.get("error", False)
            except json.JSONDecodeError:
                has_error = False

            # 记录到 trace
            call_record = {
                "tool": tool_name,
                "args": tool_args,
                "has_error": has_error,
                "result_preview": result_str[:200]
            }
            turn_record["tool_calls"].append(call_record)
            trace["total_tool_calls"] += 1

            if has_error:
                error_record = {
                    "turn": turn + 1,
                    "tool": tool_name,
                    "error_type": result_obj.get("error_type", "Unknown"),
                    "message": result_obj.get("message", "")[:100]
                }
                trace["errors"].append(error_record)
                print(f"⚠️ 错误: [{error_record['error_type']}] {error_record['message']}")
            else:
                print(f"📊 结果: {result_str[:200]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })

        trace["turns"].append(turn_record)

    # 如果循环耗尽仍未结束
    if not trace["finished"]:
        trace["final_answer"] = None
        print("\n⚠️ 达到最大轮次限制")

    # 打印 trace 摘要
    _print_trace_summary(trace)

    return trace


def _print_trace_summary(trace: dict):
    """打印 trace 摘要信息"""
    print(f"\n{'='*60}")
    print("📋 Trace 摘要:")
    print(f"   模型: {trace['model']}")
    print(f"   总轮次: {len(trace['turns'])}")
    print(f"   总工具调用: {trace['total_tool_calls']}")
    print(f"   错误次数: {len(trace['errors'])}")
    print(f"   是否完成: {'✅' if trace['finished'] else '❌'}")

    if trace["errors"]:
        print("   错误详情:")
        for err in trace["errors"]:
            print(f"     Turn {err['turn']} | {err['tool']} | [{err['error_type']}] {err['message']}")

    # 打印每轮 token 用量
    total_tokens = 0
    for t in trace["turns"]:
        if "usage" in t:
            tokens = t["usage"]["total_tokens"]
            total_tokens += tokens
            print(f"   Turn {t['turn']}: {tokens} tokens")
    if total_tokens > 0:
        print(f"   总 token: {total_tokens}")