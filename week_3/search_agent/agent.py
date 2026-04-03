"""
搜索 Agent — 核心循环

v2.0 升级内容：
  1. 检查机制 1：模型 finish_reason=="stop" 时，规则检查是否应该先搜索
  2. 检查机制 2：工具连续失败 N 次后，注入降级指令
  3. 检查机制 3：上下文长度监控（简化版，只警告）
  4. 结构化 Trace 记录每一步决策

关键技术决策（基于周三实验）：
  - 工具调用阶段只传 tools，不传 response_format（两者互斥）
  - 可选的格式化阶段用单独请求 + json_schema
"""

import json
import time
import logging

from config import (
    client, MODEL, SYSTEM_PROMPT,
    MAX_TURNS, MAX_CONSECUTIVE_ERRORS, MAX_CONTEXT_CHARS,
    CORRECTION_MESSAGE, FALLBACK_MESSAGE,
)
from tools import TOOL_DEFINITIONS, execute_tool
from checks import should_have_searched
from trace import AgentTrace, TurnTrace, ToolCallTrace

logger = logging.getLogger(__name__)


def _summarize_result(result: dict, max_len: int = 200) -> str:
    """将工具返回结果压缩为简短摘要，用于 trace 记录。"""
    if result.get("error"):
        return f"[ERROR] {result.get('error_type', 'Unknown')}: {result.get('message', '')}"[:max_len]

    # 搜索结果：显示结果数
    if "results" in result:
        total = result.get("total", 0)
        if total > 0:
            first_title = result["results"][0].get("title", "")[:60]
            return f"[{total} results] First: {first_title}"
        return "[0 results]"

    # 网页内容：显示标题和长度
    if "content" in result:
        title = result.get("title", "")[:60]
        chars = result.get("char_count", 0)
        truncated = " (truncated)" if result.get("truncated") else ""
        return f"[{chars} chars{truncated}] {title}"

    return str(result)[:max_len]


def _estimate_context_chars(messages: list) -> int:
    """粗略估算 messages 的总字符数。"""
    total = 0
    for msg in messages:
        # 兼容 dict 和 OpenAI SDK 返回的 Pydantic 对象
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            total += len(content)
        # tool_calls 类型的 message 也要算
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                total += len(tc.function.arguments) if tc.function.arguments else 0
    return total


def run_agent(user_message: str, max_turns: int = None) -> tuple:
    """
    运行搜索 Agent。

    Args:
        user_message: 用户输入
        max_turns: 最大循环轮次（默认使用 config 中的 MAX_TURNS）

    Returns:
        (answer: str, trace: AgentTrace)
    """
    if max_turns is None:
        max_turns = MAX_TURNS

    # ===== 初始化 =====
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    trace = AgentTrace(user_question=user_message)
    consecutive_errors = 0
    correction_injected = False
    fallback_injected = False
    start_time = time.time()

    # ===== 主循环 =====
    for turn in range(max_turns):

        # --- 检查机制 3：上下文长度监控 ---
        context_chars = _estimate_context_chars(messages)
        if context_chars > MAX_CONTEXT_CHARS:
            logger.warning(
                f"Context size ({context_chars} chars) exceeds threshold "
                f"({MAX_CONTEXT_CHARS}). Consider trimming in future version."
            )

        # --- 调用模型 ---
        # 关键：只传 tools，不传 response_format（周三实验确认两者互斥）
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            logger.error(f"LLM call failed at turn {turn + 1}: {e}")
            answer = f"[错误] 模型调用失败: {str(e)}"
            trace.finalize(answer, start_time)
            return answer, trace

        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason

        # ===== 分支 1：模型选择直接回答 =====
        if finish == "stop":
            turn_trace = TurnTrace(
                turn_number=turn + 1,
                finish_reason="stop",
            )

            # --- 检查机制 1：应该搜索但没搜索？ ---
            if not correction_injected and should_have_searched(user_message):
                # 注入纠正指令，强制模型重新决策
                logger.info(
                    f"Turn {turn + 1}: Model skipped tools, but question "
                    f"should be searched. Injecting correction."
                )
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                })
                messages.append({
                    "role": "user",
                    "content": CORRECTION_MESSAGE,
                })
                correction_injected = True
                turn_trace.correction_injected = True
                trace.correction_triggered = True
                trace.add_turn(turn_trace)
                continue  # 重试一次

            # 正常结束
            answer = msg.content or "[模型返回空回答]"
            trace.add_turn(turn_trace)
            trace.finalize(answer, start_time)
            return answer, trace

        # ===== 分支 2：模型选择调用工具 =====
        if finish == "tool_calls" and msg.tool_calls:
            # 保留模型的完整 message（含 tool_calls）到历史
            messages.append(msg)

            turn_trace = TurnTrace(
                turn_number=turn + 1,
                finish_reason="tool_calls",
            )

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                    logger.warning(
                        f"Failed to parse arguments for {tool_name}: "
                        f"{tc.function.arguments}"
                    )

                # 执行工具
                tool_start = time.time()
                result = execute_tool(tool_name, tool_args)
                tool_duration = int((time.time() - tool_start) * 1000)

                # 判断是否错误
                is_error = isinstance(result, dict) and result.get("error", False)

                # 记录 trace
                tool_trace = ToolCallTrace(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result_success=not is_error,
                    result_summary=_summarize_result(result),
                    error_type=result.get("error_type") if is_error else None,
                    duration_ms=tool_duration,
                )
                turn_trace.tool_calls.append(tool_trace)

                # 追加工具结果到 messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                # --- 检查机制 2：连续失败计数 ---
                if is_error:
                    consecutive_errors += 1
                    logger.info(
                        f"Tool '{tool_name}' failed ({result.get('error_type')}). "
                        f"Consecutive errors: {consecutive_errors}"
                    )
                else:
                    consecutive_errors = 0  # 成功一次就重置

            # 连续失败降级检查
            if (consecutive_errors >= MAX_CONSECUTIVE_ERRORS
                    and not fallback_injected):
                logger.info(
                    f"Turn {turn + 1}: {consecutive_errors} consecutive errors. "
                    f"Injecting fallback instruction."
                )
                messages.append({
                    "role": "user",
                    "content": FALLBACK_MESSAGE,
                })
                fallback_injected = True
                turn_trace.fallback_injected = True
                trace.fallback_triggered = True

            trace.add_turn(turn_trace)
            continue

        # ===== 分支 3：其他 finish_reason =====
        # 如 "length"（输出被截断）或 "content_filter"
        logger.warning(f"Unexpected finish_reason: {finish}")
        turn_trace = TurnTrace(
            turn_number=turn + 1,
            finish_reason=finish or "unknown",
        )
        trace.add_turn(turn_trace)

        # 如果有 content 就返回
        if msg.content:
            trace.finalize(msg.content, start_time)
            return msg.content, trace

        continue

    # ===== 超时退出 =====
    answer = (
        "[达到最大轮次] Agent 在 {} 轮内未能完成任务。"
        "这通常是因为工具反复失败或模型陷入循环。"
    ).format(max_turns)
    trace.finalize(answer, start_time)
    return answer, trace


def format_answer_as_json(
        answer: str,
        user_question: str,
        trace: AgentTrace,
) -> dict:
    """
    （可选）将自由文本回答格式化为结构化 JSON。

    使用单独的 API 请求 + json_schema，不影响主流程的 tool calling。
    这是周三实验确认的正确用法：tools 和 response_format 分开用。

    仅在测试模式下调用，避免交互模式增加延迟。
    """
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "search_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete answer text.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                            },
                            "required": ["title", "url"],
                            "additionalProperties": False,
                        },
                        "description": "Information sources used.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Answer confidence level.",
                    },
                    "searched": {
                        "type": "boolean",
                        "description": "Whether search tools were used.",
                    },
                },
                "required": ["answer", "sources", "confidence", "searched"],
                "additionalProperties": False,
            },
        },
    }

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "将以下搜索助手的回答整理为指定 JSON 格式。"
                        "保留原始回答内容，提取其中的来源信息。"
                        "根据信息充分度判断 confidence。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{user_question}\n\n"
                        f"助手回答：{answer}\n\n"
                        f"是否使用了搜索：{trace.searched}"
                    ),
                },
            ],
            response_format=schema,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Format answer failed: {e}")
        return {
            "answer": answer,
            "sources": [],
            "confidence": "low",
            "searched": trace.searched,
            "_format_error": str(e),
        }
