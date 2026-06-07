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
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

import config as settings
from config import (
    MODEL,
    SYSTEM_PROMPT,
    MAX_TURNS,
    CORRECTION_MESSAGE,
    FALLBACK_MESSAGE,
    RETRIEVAL_CORRECTION_MESSAGE,
    PLACEHOLDER_MAX_TURNS,
    PLACEHOLDER_ERROR,
    PLACEHOLDER_EMPTY,
    is_placeholder_answer,
)
from state import AgentState, PER_QUERY_DEFAULTS
from tools import (  # noqa: F401  (execute_tool 供测试 monkeypatch)
    TOOL_DEFINITIONS,
    TOOL_EFFECTS,
    ToolEffect,
    execute_tool,
)

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
    """
    本问题窗口起点 = 最后一条「内容为 SYSTEM_PROMPT」的 system 消息下标。

    每次 assemble 的段 1 都是 SYSTEM_PROMPT，所以它就是装配块的起点锚。
    不能取"最后一条 system 消息"：记忆开启时 assemble 会产出多条 system
    （段 1 prompt / 段 2 偏好 / 段 3 摘要 / 段 4 事实），取最后一条会把
    段 1-4 切出窗口，模型看不到 SYSTEM_PROMPT（v4.0 的隐藏 bug，v4.1 修复）。
    找不到锚时退回"最后一条 system 消息"的旧行为。
    """
    idx, fallback = 0, 0
    found = False
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            fallback = i
            if m.content == SYSTEM_PROMPT:
                idx = i
                found = True
    return idx if found else fallback


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


def assemble(state: AgentState, config, *, store: BaseStore):
    """
    六段装配（★保留，v3.0 memory.assemble_context / assembler.py）。
    store 由 LangGraph 注入（compile(store=...)）——段 2 偏好 / 段 4 事实的数据源。
    """
    memory = _memory_from(config)
    if memory is not None:
        msgs, report = memory.assemble_context(state["user_message"], SYSTEM_PROMPT, store)
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

    空回答重试（第七周前重构项，06-05 复跑实证）：qwen3.7-plus 上 stop+空 content
    约两成，特征是 API 侧 fast-fail（0.8-3.5s）。节点内重试一次、不画进图——
    业务决策（纠正/降级）才进拓扑，传输层抖动留在节点里（同 try/except 先例）；
    画成 agent→agent 条件边会污染拓扑、消耗 turn_count、多落 checkpoint。
    重试不消耗 turn_count（一次有效模型交互 = 一轮）；仍空则照旧走占位符兜底。
    """
    turn = state.get("turn_count", 0) + 1
    retried = 0
    while True:
        try:
            response = call_model(_context_window(state["messages"]))
        except Exception as e:  # noqa: BLE001 — 与 v3.0 行为一致：任何调用错误都降级为错误回答
            logger.error(f"LLM call failed at turn {turn}: {e}")
            return {"answer": f"{PLACEHOLDER_ERROR} 模型调用失败: {e}", "turn_count": turn}

        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason
        # 只对 "stop 且空 content" 重试；tool_calls 配空 content 是正常形态
        if (finish == "stop" and not (msg.content or "").strip() and retried < 1):
            retried += 1
            logger.warning(f"Empty content at turn {turn} (API fast-fail?), retrying once.")
            continue
        break
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
    return {
        "messages": [ai_message],
        "turn_count": turn,
        # 本问题内手动累加（替换语义字段，init 清零）——量化"重试救回了多少"
        "empty_retries": state.get("empty_retries", 0) + retried,
    }


def tools(state: AgentState):
    """
    执行工具（v3.0 execute_tool / tools.py）。
    v4.2 瘦身：每个工具的 state 副作用由 tools.TOOL_EFFECTS 声明，循环体全通用——
    第七周加工具只登记注册表，本节点零改动。保真的三条 v3.0 语义（T10 锁）：
      - sets_flag 成功失败都置（"尝试过"，防反复纠正）
      - chunk_extractor 仅成功时抽、手动"当前+新"累加（决策 B，E3 实证）
      - 失败计数不对称：仅 counts_failures 的工具失败累加，任何工具成功都清零
    """
    last = state["messages"][-1]  # 路由保证：带 tool_calls 的 AIMessage
    out_messages = []
    flag_updates: dict = {}
    chunks = list(state.get("retrieved_chunks") or [])
    failures = state.get("consecutive_failures", 0)

    for tc in last.tool_calls:  # LangChain 格式: {"name", "args", "id", "type"}
        tool_name, tool_args = tc["name"], tc["args"] or {}
        result = execute_tool(tool_name, tool_args)
        is_error = isinstance(result, dict) and result.get("error", False)
        effect = TOOL_EFFECTS.get(tool_name, ToolEffect())

        if effect.sets_flag:
            flag_updates[effect.sets_flag] = True
        if not is_error and effect.chunk_extractor:
            chunks = chunks + effect.chunk_extractor(result)  # 手动累加

        if is_error:
            if effect.counts_failures:
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
        **flag_updates,
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
        # v4.2 踩坑 #3 收紧：携带本轮真实检索来源，事实抽取据此校验引用。
        # 恒为 list（零检索 = 空白名单 = 引用全拒收），不是 None。
        retrieved_chunks=list(state.get("retrieved_chunks") or []),
    )


def update_memory(state: AgentState, config, *, store: BaseStore):
    """
    抽取主题/偏好/事实、写长期、eviction 触发摘要（v3.0 memory.update_from_turn）。
    store 由 LangGraph 注入——长期记忆三类（偏好/事实/主题）写入这里。

    v4.2 时序反转后本节点在 human_review 之后运行：记录的是用户实际看到的
    版本（含人工改写）。answer 此时必经 finalize 组装，占位符统一在此跳过——
    旧图错误/超轮路径绕过 update_memory 的行为由这条跳过等价承接（S11 扩展版）。
    """
    memory = _memory_from(config)
    if memory is None:
        return {}
    answer = state.get("answer", "")
    if is_placeholder_answer(answer):
        # 占位符轮次没有可记的内容；写进短期记忆会把一次模型抖动
        # 级联污染到后续问题的段 5（S11）。
        logger.info("Skipping memory update: placeholder answer this turn.")
        return {}
    memory.update_from_turn(state["user_message"], answer, _trace_shim(state), store)
    return {}


def human_review(state: AgentState):
    """
    输出审批（决策 F）。v4.2 起在 finalize 之后运行——审批框里永远是
    组装完成的终稿（空回答已是占位符，人工可当场改写补救，06-05 quirk 2 修复）。
    开关动态读 config.INTERRUPT_ENABLED：
      - 关 → 透明放行（默认；测试/批量评测可复现）
      - 开 → interrupt() 暂停；Command(resume=...) 的语义：
          "approve"/"ok"/"yes"/"y"/"通过"/空串 → 保留终稿
          其他非空字符串 → 改写最终答案（update_memory 记录改写后版本）
    """
    if not settings.INTERRUPT_ENABLED:
        return {}
    draft = state.get("answer", "")  # finalize 已跑过，必有值
    decision = interrupt({"draft_answer": draft, "hint": "回复 approve 通过，或直接给出改写后的答案"})
    if isinstance(decision, str) and decision.strip() \
            and decision.strip().lower() not in {"approve", "ok", "yes", "y", "通过"}:
        return {"answer": decision.strip()}
    return {}


def finalize(state: AgentState):
    """
    组装终稿 answer（v3.0 run_agent 返回值；v4.2 起不再是图终点——
    后接 human_review 审批与 update_memory）。优先级：
      1. answer 已被写入（LLM 错误短路）→ 尊重
      2. 本问题窗口最后一条带内容的 assistant 消息
      3. 都没有 → 达到最大轮次（闸门收口路径）或空回答占位符
    """
    answer = state.get("answer")
    if not answer:
        answer = _last_ai_content(state)
    if not answer:
        if state.get("turn_count", 0) >= MAX_TURNS:
            answer = f"{PLACEHOLDER_MAX_TURNS} Agent 在 {MAX_TURNS} 轮内未能完成任务。"
        else:
            answer = PLACEHOLDER_EMPTY
    return {"answer": answer}
