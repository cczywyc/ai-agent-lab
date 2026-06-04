"""
nodes.py — v4.0 节点（规则：节点 = 干活 / 改 state；路由决策见 edges.py）

节点清单与 v0.3 §2.2 一一对应：
  init / assemble / agent / tools /
  inject_correction_retrieval / inject_correction_search（决策 D：共用函数体）/
  inject_fallback / update_memory / human_review（决策 F）/ finalize

v3.0 → v4.0 的对应关系标注在各节点 docstring 里。
关键实现解释（草稿留给实现层的两件事）：

1. 装配窗口切片：checkpointer 把 thread 的全量 messages 存成审计日志，
   但发给模型的窗口只取"最后一条 system 消息起"的切片——复刻 v3.0
   "每个问题由六段装配重建上下文"的语义（历史对话已被装配压缩进段 3/5，
   不该把原始消息再发一遍）。见 _context_window()。

2. memory/ 零改动（决策 A/E）：update_memory 节点从本问题窗口构造一个
   duck-type shim（MemoryManager.update_from_turn 只用到 trace.turns[*]
   .tool_calls[*].tool_name/.result_success 和 trace.searched/.retrieved），
   v3.0 的 AgentTrace 不再使用——trace 语义字段已吸收进 state。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langgraph.types import interrupt

import config as settings
from config import (
    MODEL,
    SYSTEM_PROMPT,
    MAX_TURNS,
    CORRECTION_MESSAGE,
    FALLBACK_MESSAGE,
    RETRIEVAL_CORRECTION_MESSAGE,
)
from state import AgentState, PER_QUERY_DEFAULTS
from tools import TOOL_DEFINITIONS, execute_tool  # noqa: F401  (execute_tool 供测试 monkeypatch)

logger = logging.getLogger(__name__)


# ============================================================
# 模型调用（独立成函数：测试用 nodes.call_model = stub 替换）
# ============================================================

def call_model(oai_messages: list[dict]):
    return settings.client.chat.completions.create(
        model=MODEL,
        messages=oai_messages,
        tools=TOOL_DEFINITIONS,
    )


# ============================================================
# 工具函数
# ============================================================

def _window_start(messages: list) -> int:
    """本问题窗口起点 = 最后一条 system 消息的下标（每次 assemble 都以 system 开头）。"""
    idx = 0
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            idx = i
    return idx


def _window_messages(messages: list) -> list:
    """本问题窗口内的 LangChain 消息对象。"""
    return messages[_window_start(messages):]


def _context_window(messages: list) -> list[dict]:
    """发给模型的 OpenAI 格式窗口（装配切片语义，见模块 docstring 第 1 点）。"""
    return convert_to_openai_messages(_window_messages(messages))


def _last_ai_content(state: AgentState) -> str:
    """本问题窗口内最后一条带内容的 assistant 消息（E1 观察：统一从对象取 content）。"""
    for m in reversed(_window_messages(state["messages"])):
        if isinstance(m, AIMessage):
            content = (m.content or "").strip()
            if content:
                return content
            return ""  # 最后一条 AI 消息无内容（如 tool_calls）→ 视为无答案
    return ""


def _extract_retrieved_chunks(result: dict) -> list[dict]:
    """从 retrieve_documents 结果里抽出 chunk 简要信息（v3.0 agent.py 原逻辑）。"""
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


def _memory_from(runnable_config) -> object | None:
    """从 configurable 取 use_memory 开关；开了才碰 MemoryManager（测试不带记忆）。"""
    configurable = (runnable_config or {}).get("configurable") or {}
    if not configurable.get("use_memory", False):
        return None
    from memory import get_memory
    return get_memory()


# ============================================================
# 节点
# ============================================================

def init(state: AgentState):
    """
    入口重置（决策 C，v0.3 扩充版）：
      - 首轮：为所有 per-query 字段建立默认值（E2：TypedDict 无隐式默认）
      - 后续轮：把上一问题的残留打回初值
    刻意不返回 messages（保留持久化历史）和 user_message（invoke 输入带入）。
    """
    defaults = dict(PER_QUERY_DEFAULTS)
    defaults["retrieved_chunks"] = []  # 不共享模块级列表
    return defaults


def assemble(state: AgentState, config):
    """六段装配（★保留，v3.0 memory.assemble_context / assembler.py）。"""
    memory = _memory_from(config)
    if memory is not None:
        msgs, report = memory.assemble_context(state["user_message"], SYSTEM_PROMPT)
        report_dict = asdict(report) if is_dataclass(report) else dict(vars(report))
        logger.info(
            f"Memory assembled: segs={report_dict.get('segments_present')} "
            f"trimmed={report_dict.get('segments_trimmed')} "
            f"chars={report_dict.get('total_chars')} "
            f"facts_recalled={report_dict.get('facts_recalled')}"
        )
    else:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["user_message"]},
        ]
        report_dict = None
    return {"messages": msgs, "assembly_report": report_dict}


def agent(state: AgentState):
    """
    调模型（v3.0 client.chat.completions.create）。
    turn_count 在此 +1（一次模型调用 = 一轮，替换语义：读当前值返回 +1）。
    LLM 调用失败 → 写 answer 短路通道，边上直接路由 finalize（v3.0 分支保留）。
    """
    turn = state.get("turn_count", 0) + 1
    try:
        response = call_model(_context_window(state["messages"]))
    except Exception as e:  # noqa: BLE001 — 与 v3.0 行为一致：任何调用错误都降级为错误回答
        logger.error(f"LLM call failed at turn {turn}: {e}")
        return {"answer": f"[错误] 模型调用失败: {e}", "turn_count": turn}

    choice = response.choices[0]
    msg = choice.message
    finish = choice.finish_reason
    if finish not in ("stop", "tool_calls"):
        logger.warning(f"Unexpected finish_reason: {finish}")  # 罕见分支：按 stop 处理

    ai_message: dict = {"role": "assistant", "content": msg.content or ""}
    if finish == "tool_calls" and getattr(msg, "tool_calls", None):
        ai_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return {"messages": [ai_message], "turn_count": turn}


def tools(state: AgentState):
    """
    执行工具（v3.0 execute_tool / tools.py）：
      - 写工具结果消息
      - 更新 has_*（含失败="尝试过"，防反复纠正）
      - retrieved_chunks 手动"当前+新"累加（决策 B：替换语义，E3 实证）
      - consecutive_failures：仅 fetch_webpage 失败累计（v3.0 注释保留：
        VectorStoreNotReady 是配置问题，不该触发 fallback），成功清零
    """
    last = state["messages"][-1]  # 路由保证：带 tool_calls 的 AIMessage
    out_messages = []
    has_searched = state.get("has_searched", False)
    has_retrieved = state.get("has_retrieved", False)
    chunks = list(state.get("retrieved_chunks") or [])
    failures = state.get("consecutive_failures", 0)

    for tc in last.tool_calls:  # LangChain 格式: {"name", "args", "id", "type"}
        tool_name, tool_args = tc["name"], tc["args"] or {}
        result = execute_tool(tool_name, tool_args)
        is_error = isinstance(result, dict) and result.get("error", False)

        if tool_name == "web_search":
            has_searched = True
        elif tool_name == "retrieve_documents":
            has_retrieved = True
            if not is_error:
                chunks = chunks + _extract_retrieved_chunks(result)  # 手动累加

        if is_error:
            if tool_name == "fetch_webpage":
                failures += 1
            logger.info(
                f"Tool '{tool_name}' failed ({result.get('error_type')}). "
                f"Consecutive fetch errors: {failures}"
            )
        else:
            failures = 0

        out_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(result, ensure_ascii=False),
        })

    return {
        "messages": out_messages,
        "has_searched": has_searched,
        "has_retrieved": has_retrieved,
        "retrieved_chunks": chunks,
        "consecutive_failures": failures,
    }


def inject_correction(state: AgentState, *, kind: str):
    """
    纠正注入（决策 D：两个图节点共用本函数体，functools.partial 绑定 kind）。
    模型的 stop 回复已由 agent 节点写进 messages，这里只追加纠正指令 + 置标志。
    """
    if kind == "retrieval":
        logger.info("Injecting retrieval correction.")
        return {
            "messages": [{"role": "user", "content": RETRIEVAL_CORRECTION_MESSAGE}],
            "retrieval_correction_injected": True,
            "correction_triggered": True,
        }
    logger.info("Injecting search correction.")
    return {
        "messages": [{"role": "user", "content": CORRECTION_MESSAGE}],
        "search_correction_injected": True,
        "correction_triggered": True,
    }


def inject_fallback(state: AgentState):
    """降级注入（v3.0 检查机制 2）。"""
    logger.info("Injecting fallback instruction.")
    return {
        "messages": [{"role": "user", "content": FALLBACK_MESSAGE}],
        "fallback_injected": True,
        "fallback_triggered": True,
    }


def _trace_shim(state: AgentState) -> SimpleNamespace:
    """
    为 memory.update_from_turn 构造 duck-type shim（memory/ 零改动）。
    只扫本问题窗口：配对 AIMessage.tool_calls 与 ToolMessage，恢复
    tool_name / result_success。
    """
    id_to_name: dict = {}
    calls = []
    for m in _window_messages(state["messages"]):
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                id_to_name[tc["id"]] = tc["name"]
        elif isinstance(m, ToolMessage):
            try:
                ok = not (json.loads(m.content) or {}).get("error", False)
            except (json.JSONDecodeError, TypeError, AttributeError):
                ok = True
            calls.append(SimpleNamespace(
                tool_name=id_to_name.get(m.tool_call_id, "unknown"),
                result_success=ok,
            ))
    return SimpleNamespace(
        turns=[SimpleNamespace(tool_calls=calls)],
        searched=state.get("has_searched", False),
        retrieved=state.get("has_retrieved", False),
    )


def update_memory(state: AgentState, config):
    """抽取主题/偏好/事实、写长期、eviction 触发摘要（v3.0 memory.update_from_turn）。"""
    memory = _memory_from(config)
    if memory is None:
        return {}
    answer = state.get("answer") or _last_ai_content(state)
    memory.update_from_turn(state["user_message"], answer, _trace_shim(state))
    return {}


def human_review(state: AgentState):
    """
    finalize 前的输出审批（决策 F）。开关动态读 config.INTERRUPT_ENABLED：
      - 关 → 透明放行（默认；测试/批量评测可复现）
      - 开 → interrupt() 暂停；Command(resume=...) 的语义：
          "approve"/"ok"/"yes"/"y"/"通过"/空串 → 保留草稿答案
          其他非空字符串 → 改写最终答案（写 answer 短路通道，finalize 尊重它）
    """
    if not settings.INTERRUPT_ENABLED:
        return {}
    draft = state.get("answer") or _last_ai_content(state)
    decision = interrupt({"draft_answer": draft, "hint": "回复 approve 通过，或直接给出改写后的答案"})
    if isinstance(decision, str) and decision.strip() \
            and decision.strip().lower() not in {"approve", "ok", "yes", "y", "通过"}:
        return {"answer": decision.strip()}
    return {}


def finalize(state: AgentState):
    """
    写 answer 收尾（v3.0 run_agent 返回值）。优先级：
      1. answer 已被写入（LLM 错误短路 / human_review 改写）→ 尊重
      2. 本问题窗口最后一条带内容的 assistant 消息
      3. 都没有 → 达到最大轮次（闸门收口路径）或空回答
    """
    answer = state.get("answer")
    if not answer:
        answer = _last_ai_content(state)
    if not answer:
        if state.get("turn_count", 0) >= MAX_TURNS:
            answer = f"[达到最大轮次] Agent 在 {MAX_TURNS} 轮内未能完成任务。"
        else:
            answer = "[模型返回空回答]"
    return {"answer": answer}
