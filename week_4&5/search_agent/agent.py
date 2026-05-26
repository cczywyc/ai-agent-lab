"""
搜索 Agent — 核心循环 v3.0

v3.0 在 v2.0 基础上扩展：
  - 检查机制 1a：模型直答时，先判 should_have_retrieved（涉及本地内容应先查本地）
  - 检查机制 1b：再判 should_have_searched（联网兜底）
    两个纠正最多各注入一次，互不重叠
  - retrieve_documents 调用结果在 trace 中记录召回 chunk 简要信息
"""

import json
import time
import logging

from config import (
    client, MODEL, SYSTEM_PROMPT,
    MAX_TURNS, MAX_CONSECUTIVE_ERRORS, MAX_CONTEXT_CHARS,
    CORRECTION_MESSAGE, FALLBACK_MESSAGE, RETRIEVAL_CORRECTION_MESSAGE,
)
from tools import TOOL_DEFINITIONS, execute_tool
from checks import should_have_searched, should_have_retrieved
from trace import AgentTrace, TurnTrace, ToolCallTrace

logger = logging.getLogger(__name__)


def _summarize_result(result: dict, max_len: int = 200) -> str:
    """将工具返回结果压缩为简短摘要，用于 trace 记录。"""
    if result.get("error"):
        return f"[ERROR] {result.get('error_type', 'Unknown')}: {result.get('message', '')}"[:max_len]

    # web_search / retrieve_documents 都有 results 字段，按第一条结构区分
    if "results" in result:
        results = result["results"] or []
        if not results:
            # 兼容两种工具的空结果
            return "[0 results]"
        first = results[0]
        if isinstance(first, dict) and "doc" in first and "section" in first:
            return (
                f"[{len(results)} chunks] Top: {first.get('doc')}#{first.get('section')} "
                f"score={first.get('score')}"
            )[:max_len]
        if isinstance(first, dict) and "url" in first:
            return f"[{len(results)} results] First: {first.get('title', '')[:60]}"
        return f"[{len(results)} results]"

    # fetch_webpage
    if "content" in result:
        title = result.get("title", "")[:60]
        chars = result.get("char_count", 0)
        truncated = " (truncated)" if result.get("truncated") else ""
        return f"[{chars} chars{truncated}] {title}"

    return str(result)[:max_len]


def _extract_retrieved_chunks(result: dict) -> list[dict]:
    """从 retrieve_documents 结果里抽出 chunk 简要信息存到 trace。"""
    if not isinstance(result, dict) or result.get("error"):
        return []
    out = []
    for r in result.get("results", []) or []:
        out.append({
            "doc": r.get("doc"),
            "section": r.get("section"),
            "chunk_id": r.get("chunk_id"),
            "score": r.get("score"),
        })
    return out


def _estimate_context_chars(messages: list) -> int:
    """粗略估算 messages 的总字符数。"""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            total += len(content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                total += len(tc.function.arguments) if tc.function.arguments else 0
    return total


def run_agent(
    user_message: str,
    max_turns: int = None,
    *,
    memory=None,
) -> tuple:
    """
    运行搜索 Agent。

    Args:
        user_message: 用户输入
        max_turns:    Agent Loop 最大循环数
        memory:       可选的 MemoryManager；传入则启用六段装配+长期记忆，
                      传 None 退化为 v3.0 RAG-only 行为（单轮无记忆）

    Returns:
        (answer: str, trace: AgentTrace)
    """
    if max_turns is None:
        max_turns = MAX_TURNS

    # ===== 初始 messages =====
    # 启用记忆时走六段装配；否则用 v3.0 RAG-only 默认
    if memory is not None:
        messages, assembly_report = memory.assemble_context(
            user_message, SYSTEM_PROMPT,
        )
        logger.info(
            f"Memory assembled: segs={assembly_report.segments_present} "
            f"trimmed={assembly_report.segments_trimmed} "
            f"chars={assembly_report.total_chars} "
            f"facts_recalled={assembly_report.facts_recalled}"
        )
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        assembly_report = None

    trace = AgentTrace(user_question=user_message)
    consecutive_errors = 0
    correction_injected = False
    retrieval_correction_injected = False
    fallback_injected = False
    has_searched = False
    has_retrieved = False
    start_time = time.time()

    for turn in range(max_turns):

        # --- 上下文长度监控 ---
        context_chars = _estimate_context_chars(messages)
        if context_chars > MAX_CONTEXT_CHARS:
            logger.warning(
                f"Context size ({context_chars} chars) exceeds threshold "
                f"({MAX_CONTEXT_CHARS}). Consider trimming in v3.x."
            )

        # --- 调用模型 ---
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

        # ===== 分支 1：模型直接回答 =====
        if finish == "stop":
            turn_trace = TurnTrace(turn_number=turn + 1, finish_reason="stop")

            # === 纠正注入语义 ===
            # 只有当模型从未碰过任何工具就给出 stop 时，才注入纠正。
            # 一旦任何工具被调用过，就尊重模型 stop 的判断
            # （否则会出现：模型已查本地 → 我们又强行让它再联网 → 多余的轮次）
            any_tool_called = has_searched or has_retrieved

            # --- 检索纠正优先级高于搜索纠正 ---
            if (not any_tool_called
                    and not retrieval_correction_injected
                    and should_have_retrieved(user_message)):
                logger.info(
                    f"Turn {turn + 1}: Model skipped tools, but question "
                    f"should trigger retrieve_documents. Injecting correction."
                )
                messages.append({"role": "assistant", "content": msg.content})
                messages.append({
                    "role": "user",
                    "content": RETRIEVAL_CORRECTION_MESSAGE,
                })
                retrieval_correction_injected = True
                turn_trace.retrieval_correction_injected = True
                trace.retrieval_correction_triggered = True
                trace.add_turn(turn_trace)
                continue

            # --- v2.0 沿用：联网搜索纠正（同样只在完全没碰过工具时） ---
            if (not any_tool_called
                    and not correction_injected
                    and should_have_searched(user_message)):
                logger.info(
                    f"Turn {turn + 1}: Model skipped tools, but question "
                    f"should be searched. Injecting correction."
                )
                messages.append({"role": "assistant", "content": msg.content})
                messages.append({"role": "user", "content": CORRECTION_MESSAGE})
                correction_injected = True
                turn_trace.correction_injected = True
                trace.correction_triggered = True
                trace.add_turn(turn_trace)
                continue

            # 正常结束
            answer = msg.content or "[模型返回空回答]"
            trace.add_turn(turn_trace)
            trace.finalize(answer, start_time)
            if memory is not None:
                memory.update_from_turn(user_message, answer, trace)
            return answer, trace

        # ===== 分支 2：模型选择调用工具 =====
        if finish == "tool_calls" and msg.tool_calls:
            messages.append(msg)

            turn_trace = TurnTrace(turn_number=turn + 1, finish_reason="tool_calls")

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

                tool_start = time.time()
                result = execute_tool(tool_name, tool_args)
                tool_duration = int((time.time() - tool_start) * 1000)

                is_error = isinstance(result, dict) and result.get("error", False)

                # retrieve_documents 的特殊 trace：记录召回的 chunk
                retrieved = (
                    _extract_retrieved_chunks(result)
                    if tool_name == "retrieve_documents" and not is_error
                    else []
                )

                tool_trace = ToolCallTrace(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result_success=not is_error,
                    result_summary=_summarize_result(result),
                    error_type=result.get("error_type") if is_error else None,
                    duration_ms=tool_duration,
                    retrieved_chunks=retrieved,
                )
                turn_trace.tool_calls.append(tool_trace)

                # 记录工具实际被调用过（即使失败也算尝试过，避免反复纠正注入）
                if tool_name == "web_search":
                    has_searched = True
                elif tool_name == "retrieve_documents":
                    has_retrieved = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                # 连续失败计数
                # retrieve_documents 的 VectorStoreNotReady 是配置问题，
                # 不应该累计到 fallback 触发（fallback 针对 fetch_webpage）
                if is_error:
                    if tool_name == "fetch_webpage":
                        consecutive_errors += 1
                    logger.info(
                        f"Tool '{tool_name}' failed ({result.get('error_type')}). "
                        f"Consecutive fetch errors: {consecutive_errors}"
                    )
                else:
                    consecutive_errors = 0

            # fetch_webpage 连续失败降级
            if (consecutive_errors >= MAX_CONSECUTIVE_ERRORS
                    and not fallback_injected):
                logger.info(
                    f"Turn {turn + 1}: {consecutive_errors} consecutive fetch errors. "
                    f"Injecting fallback instruction."
                )
                messages.append({"role": "user", "content": FALLBACK_MESSAGE})
                fallback_injected = True
                turn_trace.fallback_injected = True
                trace.fallback_triggered = True

            trace.add_turn(turn_trace)
            continue

        # ===== 分支 3：其他 finish_reason =====
        logger.warning(f"Unexpected finish_reason: {finish}")
        turn_trace = TurnTrace(turn_number=turn + 1, finish_reason=finish or "unknown")
        trace.add_turn(turn_trace)
        if msg.content:
            trace.finalize(msg.content, start_time)
            if memory is not None:
                memory.update_from_turn(user_message, msg.content, trace)
            return msg.content, trace
        continue

    # ===== 超时退出 =====
    answer = (
        "[达到最大轮次] Agent 在 {} 轮内未能完成任务。"
    ).format(max_turns)
    trace.finalize(answer, start_time)
    if memory is not None:
        memory.update_from_turn(user_message, answer, trace)
    return answer, trace
