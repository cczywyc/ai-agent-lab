"""
nodes.py — v5.0 节点（规则：节点 = 干活 / 改 state；路由决策见 edges.py）

v5.0 第七周：在 v4.2 单步引擎外套一层 plan 循环（planner-executor-critic）。
新增节点（设计草稿 v0.3 §2.1 / 职责边界 §1）：
  planner   入口拆解问题→plan；每步后判 done / re-plan（call_planner_model 可桩）
  step_init 每个子任务转移时把 per-subtask 标志打回初值（设计 ① 两层重置）
  critic    审 executor 单步输出 → critic_verdict（accept/retry/escalate）+ 校验引用（call_critic_model 可桩）

复用节点（v4.2 原样 / 按需参数化，职责边界 §7）：
  init      重置升级：PER_TASK_DEFAULTS（init 清）+ PER_SUBTASK_DEFAULTS（init + step_init 都清）
  assemble  按 role 参数化：role=executor 每子任务重产 SYSTEM_PROMPT 锚（E7 重锚隔离）
  agent + tools + inject_*  = executor 引擎（v4.2 内层循环原样；内层出口改接 critic 见 edges.py）
  finalize  组装**结构化调研报告**（v5.0：从 step_results + plan 拼，不再是单答案）
  human_review / update_memory  收尾链尾原样（v4.2 反转时序：finalize → review → memory）

三档模型调用各自独立成函数（测试用 nodes.call_*_model = stub 替换）：
  call_model（executor）/ call_planner_model（规划）/ call_critic_model（评审）。

关键实现解释（沿 v4.2 + v5.0 新增）：
  1. 装配窗口切片 _window_start：每子任务 assemble 重产 SYSTEM_PROMPT，窗口自锚到本子任务、
     天然隔离前序子任务的 tool 历史（E7）。messages 跨子任务只增不减、init 不清。
  2. 两档计数器分开（决策 B / E5）：empty_retries（传输层，agent 节点内、不进拓扑、per-task 累计）
     与 retry_count（业务层，critic 驱动、走 retry 边回 assemble、per-subtask）。
"""

from __future__ import annotations

import json
import logging
import re
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
    MAX_STEPS,
    MAX_REPLAN,
    CITATION_MIN_GROUNDING,
    CORRECTION_MESSAGE,
    FALLBACK_MESSAGE,
    SYNTHESIS_MESSAGE,
    RETRIEVAL_CORRECTION_MESSAGE,
    PLANNER_PROMPT,
    CRITIC_PROMPT,
    RETRY_FEEDBACK_PREFIX,
    PLACEHOLDER_MAX_TURNS,
    PLACEHOLDER_ERROR,
    PLACEHOLDER_EMPTY,
    is_placeholder_answer,
)
from state import (
    AgentState,
    fresh_subtask_defaults,
    fresh_task_defaults,
    fresh_retry_reset,
)
from tools import (  # noqa: F401  (execute_tool 供测试 monkeypatch)
    TOOL_DEFINITIONS,
    TOOL_EFFECTS,
    ToolEffect,
    execute_tool,
)
from checks import should_have_retrieved, should_have_searched

logger = logging.getLogger(__name__)


# ============================================================
# 模型调用（独立成函数：测试用 nodes.call_*_model = stub 替换）
# ============================================================

def call_model(oai_messages: list[dict]):
    """executor（agent 节点）模型调用——带工具。"""
    return settings.client.chat.completions.create(
        model=MODEL,
        messages=oai_messages,
        tools=TOOL_DEFINITIONS,
    )


def call_planner_model(messages: list[dict]) -> str:
    """planner 模型调用——纯文本（不带工具，planner 不碰检索）。返回模型文本。"""
    resp = settings.client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content or ""


def call_critic_model(messages: list[dict]) -> str:
    """critic 模型调用——纯文本（裁决 + feedback）。返回模型文本。"""
    resp = settings.client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content or ""


# ============================================================
# 工具函数（窗口切片 · 解析 · 引用校验 · 装配）
# ============================================================

def _window_start(messages: list) -> int:
    """本（子任务）窗口起点 = 最后一条「内容为 SYSTEM_PROMPT」的 system 消息下标。

    每子任务 assemble(role=executor) 的段 1 都是 SYSTEM_PROMPT，所以它是本子任务
    装配块的起点锚——窗口自锚到当前子任务，天然隔离前序子任务的 tool 历史（E7）。
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
    """本子任务窗口内的 LangChain 消息对象。"""
    return messages[_window_start(messages):]


def _context_window(messages: list) -> list[dict]:
    """发给 executor 模型的 OpenAI 格式窗口（重锚切片语义，E7）。"""
    return convert_to_openai_messages(_window_messages(messages))


def _last_ai_content(state: AgentState) -> str:
    """本子任务窗口内最后一条带内容的 assistant 消息 = executor 单步产出。"""
    for m in reversed(_window_messages(state["messages"])):
        if isinstance(m, AIMessage):
            content = (m.content or "").strip()
            return content if content else ""
    return ""


def _memory_from(runnable_config) -> object | None:
    """从 configurable 取 use_memory 开关；开了才碰 MemoryManager（测试不带记忆）。"""
    configurable = (runnable_config or {}).get("configurable") or {}
    if not configurable.get("use_memory", False):
        return None
    from memory import get_memory
    return get_memory()


def _current_subtask(state: AgentState) -> dict:
    """当前子任务（step_index 指向 plan 的项）；越界时退回"整个问题为单子任务"。"""
    plan = state.get("plan", [])
    k = state.get("step_index", 0)
    if 0 <= k < len(plan):
        return plan[k]
    return {"id": k, "query": state.get("user_message", ""), "status": "pending"}


def _parse_plan(text: str) -> list[str]:
    """从 planner 模型文本解析子任务列表：优先 JSON 字符串数组，退回按行/编号切。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.I).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    lines = [re.sub(r"^[\s\-\*\d\.、)\）]+", "", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def _parse_verdict(text: str) -> str:
    """从 critic 模型文本解析裁决；默认 accept（此时已过硬闸门）。"""
    t = (text or "").lower()
    if "escalate" in t:
        return "escalate"
    if "retry" in t:
        return "retry"
    return "accept"


def _parse_feedback(text: str) -> str:
    m = re.search(r"feedback[:：]\s*(.+)", text or "", flags=re.I)
    return m.group(1).strip() if m else ""


_CITATION_RE = re.compile(r"\[([^\[\]#]+)#([^\[\]]+)\]")


def _extract_citations(text: str) -> list[str]:
    """抽出 [doc#section] / [doc#section#chunk_id] 引用，归一成 'doc#section'。"""
    out = []
    for doc, rest in _CITATION_RE.findall(text or ""):
        section = rest.split("#")[0]
        out.append(f"{doc.strip()}#{section.strip()}")
    return out


def _citation_legal_one(doc: str, section: str, chunks: list) -> bool:
    """单条引用是否落在本步真实召回里。**doc 必须命中**某个召回 chunk（防编造来源）；
    section 容忍缩写——真实库的 section 是很长的层级路径（`A > 二、… > 认知 2：…`），模型
    常只引叶子/后缀。匹配放宽到：精确 / full 以 cited 结尾（后缀）/ cited==叶子 / cited 是叶子子串。
    （真实跑暴露：旧版精确匹配把合法但缩写的引用误杀成 retry、级联拖垮整条研究。）"""
    doc = doc.strip()
    section = section.strip()
    for c in (chunks or []):
        if str(c.get("doc", "")).strip() != doc:
            continue
        full = str(c.get("section", "")).strip()
        if not section:
            return True                       # [doc] 无 section 但 doc 已召回 → 放行
        leaf = full.split(">")[-1].strip()
        if section == full or full.endswith(section) or section == leaf or section in leaf:
            return True
    return False


def _citations_legal(citations: list[str], chunks: list) -> bool:
    """每条引用都能溯源到本步召回（严口径，供单测/store 侧白名单语义对照）。"""
    return all(
        _citation_legal_one(cit.partition("#")[0], cit.partition("#")[2], chunks)
        for cit in citations
    )


def _citation_grounding(citations: list[str], chunks: list) -> float:
    """引用"接地比例" = 能溯源到本步召回的引用占比（无引用记 1.0，不在此闸门管）。
    critic 用它做软闸门（≥ CITATION_MIN_GROUNDING 放行进 LLM 裁决）——比"全部命中"的硬闸门
    宽容：模型写富报告会引很多子节/相关节，全命中是奢望，但"多数对得上"即算接地。
    零检索而满篇引用 → 0.0（必 retry）；纯编造 → 低比例 → retry。"""
    if not citations:
        return 1.0
    legal = sum(1 for cit in citations
                if _citation_legal_one(cit.partition("#")[0], cit.partition("#")[2], chunks))
    return legal / len(citations)


def _step_summaries(state: AgentState, max_chars: int = 400) -> str:
    """已 accept 各步的结论摘要（planner 看摘要、不看检索原文；executor 看前序摘要）。"""
    lines = []
    for r in sorted(state.get("step_results", []), key=lambda r: r.get("step_id", 0)):
        if r.get("status") == "accept":
            t = (r.get("text") or "").strip().replace("\n", " ")
            lines.append(f"- [{(r.get('query') or '')[:30]}] {t[:max_chars]}")
    return "\n".join(lines)


def _executor_user_content(state: AgentState) -> str:
    """role=executor 的当前问题段（职责边界 §5 局部视角）：当前子任务 query +
    前序步结论摘要（看摘要、不看其他子任务检索原文）+ （retry 时）critic 反馈。"""
    subtask = _current_subtask(state)
    k = state.get("step_index", 0)
    total = len(state.get("plan", [])) or 1
    parts = [f"# 调研子任务 {k + 1}/{total}", f"问题：{subtask.get('query', '')}"]
    summaries = _step_summaries(state)
    if summaries:
        parts.append("\n## 前序步已得结论（摘要，供参考，勿重复检索）\n" + summaries)
    feedback = state.get("critic_feedback", "")
    if feedback:
        parts.append("\n" + RETRY_FEEDBACK_PREFIX + feedback)
    return "\n".join(parts)


# ============================================================
# 节点 · 入口与重置
# ============================================================

def init(state: AgentState):
    """
    入口重置（设计 ① 两层重置）：
      - per-task（plan / step_results / replan_count / plan_version / done / empty_retries / answer）
        只在这里随新用户问题清零。
      - per-subtask 标志在这里也建一份默认（首轮），后续每子任务由 step_init 再打回。
    刻意不返回 messages（保留 thread 历史，E7）和 user_message（invoke 输入带入）。
    """
    return {**fresh_task_defaults(), **fresh_subtask_defaults()}


def step_init(state: AgentState):
    """
    子任务转移重置（设计 ① 的降一层版：per-query → per-subtask）。
    只打回 PER_SUBTASK_DEFAULTS（含 retry_count / critic_verdict / critic_feedback /
    retrieved_chunks / turn_count 等），**不碰** per-task 字段（plan / step_results /
    replan_count / plan_version）——E2/E6 坐实"每子任务清零内层、跨子任务累加外层"。
    走在 route_after_planner → step_init → assemble 这条边上（新子任务）；retry 边走
    critic → retry_reset → assemble（轻量重置，见 retry_reset）。
    """
    return fresh_subtask_defaults()


def retry_reset(state: AgentState):
    """
    业务 retry 的轻量重置（critic → retry_reset → assemble）：把内层执行状态打回初值，
    让每次 retry 都是带满额 turn 预算的"全新一次重做"（职责边界 §5"重做该步"）——修掉
    "首次跑满 turn 的子任务一旦被 retry 就因预算耗尽直接 escalate、retry 档形同虚设"。
    保留 retry_count（业务额度，critic 刚 +1）、critic_feedback（重做指导，assemble 读它）、
    retrieved_chunks（换措辞重做仍可引用首次召回）。与 step_init 对称，区别只在这三项。
    """
    return fresh_retry_reset()


# ============================================================
# 节点 · planner（拆解 / 推进 / re-plan，职责边界 §2）
# ============================================================

def _planner_messages(state: AgentState) -> list[dict]:
    """planner 拆解的输入（decompose）。v0.5 起 escalate 改 skip-and-advance、不再 LLM re-plan，
    所以 planner 只在入口调一次模型拆解。"""
    question = state.get("user_message", "")
    return [
        {"role": "system", "content": PLANNER_PROMPT.format(max_steps=MAX_STEPS)},
        {"role": "user", "content": f"调研问题：{question}\n\n请拆解成有序子任务（JSON 字符串数组）。"},
    ]


def _planner_decompose(state: AgentState):
    raw = call_planner_model(_planner_messages(state))
    queries = _parse_plan(raw) or [state.get("user_message", "")]
    queries = queries[:MAX_STEPS]
    plan = [{"id": i, "query": q, "status": "pending"} for i, q in enumerate(queries)]
    logger.info(f"Planner decomposed into {len(plan)} subtasks.")
    return {"plan": plan, "step_index": 0, "plan_version": 1,
            "done": False, "termination_reason": ""}


def _planner_advance(state: AgentState):
    """critic accept：标记当前子任务 done，推进 step_index，判 done。"""
    plan = [dict(s) for s in state.get("plan", [])]
    k = state.get("step_index", 0)
    if 0 <= k < len(plan):
        plan[k]["status"] = "done"
    next_k = k + 1
    done, reason = False, ""
    if next_k >= len(plan):
        done, reason = True, "all_steps_done"
    elif next_k >= MAX_STEPS:
        done, reason = True, "max_steps"
    return {"plan": plan, "step_index": next_k, "done": done, "termination_reason": reason}


def _planner_escalate(state: AgentState):
    """critic escalate / retry 用尽：**skip-and-advance**（v0.5）——标记当前步 skipped、推进 step_index。
    不再 re-do 同一步（旧 re-plan 保留 step_index 重做，会因同一失败原因反复触发、烧光预算让
    后续步没机会跑——真实跑实测：一步引用误杀就拖垮整条研究）。replan_count 计累计跳过/升级数；
    达 MAX_REPLAN 提前收口（太多步失败 → 放弃剩余）。job 边界 §4 的 escalate 三选项里取"跳过"。"""
    replan = state.get("replan_count", 0) + 1
    k = state.get("step_index", 0)
    plan = [dict(s) for s in state.get("plan", [])]
    if 0 <= k < len(plan):
        plan[k]["status"] = "skipped"
    next_k = k + 1
    done, reason = False, ""
    if replan >= MAX_REPLAN:
        done, reason = True, "max_replan"          # 太多步失败，放弃剩余
    elif next_k >= len(plan):
        done, reason = True, "all_steps_done"
    elif next_k >= MAX_STEPS:
        done, reason = True, "max_steps"
    logger.info(f"Planner escalate: skip step {k} → advance to {next_k} (skip count={replan}).")
    return {"plan": plan, "step_index": next_k, "replan_count": replan,
            "done": done, "termination_reason": reason}


def planner(state: AgentState):
    """
    三态（设计 §2.1）：plan 为空 → 拆解；critic accept → 推进判 done；其余（escalate /
    retry 用尽）→ skip-and-advance（v0.5）。planner 不碰检索/工具——只读问题 + plan + 各步结论摘要。
    判 done 的依据写明（v0.3 §2.1）：advance/escalate 时若所有步走完 / 撞 MAX_STEPS / 跳过数撞
    MAX_REPLAN 则置 done=True，实际终止交条件边 route_after_planner（节点干活、边做决策）。
    """
    if not state.get("plan"):
        return _planner_decompose(state)
    if state.get("critic_verdict", "") == "accept":
        return _planner_advance(state)
    return _planner_escalate(state)


# ============================================================
# 节点 · assemble（按 role 参数化）
# ============================================================

def assemble(state: AgentState, config, *, store: BaseStore, role: str = "executor"):
    """
    六段装配从时间维度扩到空间维度（按角色挑段，设计 ④）。
    role=executor：每子任务重产 SYSTEM_PROMPT 锚（段 1）→ 窗口自锚到本子任务、隔离前序
    子任务 tool 历史（E7）；段 6 = 本子任务局部上下文（query + 前序摘要 + retry 反馈）。
    记忆开启时复用 v3.0 六段装配（段 2 偏好 / 段 3 摘要 / 段 4 长期事实召回都进窗口）——
    跨调研任务的偏好与事实仍影响每个子任务的产出（read-side 记忆不因外循环而丢）。
    （planner 的"看摘要"视角由 planner 节点内部 _planner_messages 处理，不走本节点；
    role 现仅 "executor" 一个活值，保留为前向扩展钩子——decision C 的 partial 先例。）
    """
    content = _executor_user_content(state)
    memory = _memory_from(config)
    if memory is not None:
        msgs, report = memory.assemble_context(content, SYSTEM_PROMPT, store)
        report_dict = asdict(report) if is_dataclass(report) else dict(vars(report))
        logger.info(f"Memory assembled (executor step {state.get('step_index', 0)}): "
                    f"segs={report_dict.get('segments_present')} "
                    f"facts_recalled={report_dict.get('facts_recalled')}")
    else:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},   # 段 1 锚（E7 重锚）
            {"role": "user", "content": content},
        ]
        report_dict = {"role": role, "step_index": state.get("step_index", 0)}
    return {"messages": msgs, "assembly_report": report_dict}


# ============================================================
# 节点 · executor 引擎（v4.2 内层循环原样）
# ============================================================

def agent(state: AgentState):
    """
    调 executor 模型（v4.2 原样）。turn_count 在此 +1（per-subtask，step_init 清零）。
    空回答节点内重试一次（传输层，不进拓扑、不耗 turn；empty_retries per-task 累计观测）。
    LLM 调用失败 → 追加错误占位 AI 消息（不写 answer，交 critic 判，v5.0 改动）。
    """
    turn = state.get("turn_count", 0) + 1
    retried = 0
    while True:
        try:
            response = call_model(_context_window(state["messages"]))
        except Exception as e:  # noqa: BLE001
            logger.error(f"LLM call failed at turn {turn}: {e}")
            return {
                "messages": [{"role": "assistant", "content": f"{PLACEHOLDER_ERROR} 模型调用失败: {e}"}],
                "turn_count": turn,
            }
        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason
        if finish == "stop" and not (msg.content or "").strip() and retried < 1:
            retried += 1
            logger.warning(f"Empty content at turn {turn} (API fast-fail?), retrying once.")
            continue
        break
    if finish not in ("stop", "tool_calls"):
        logger.warning(f"Unexpected finish_reason: {finish}")

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
        "empty_retries": state.get("empty_retries", 0) + retried,  # per-task 累计
    }


def tools(state: AgentState):
    """执行工具（v4.2 瘦身 TOOL_EFFECTS 声明化，节点体全通用；第七周加工具只登记注册表）。"""
    last = state["messages"][-1]  # 路由保证：带 tool_calls 的 AIMessage
    out_messages = []
    flag_updates: dict = {}
    chunks = list(state.get("retrieved_chunks") or [])
    failures = state.get("consecutive_failures", 0)

    for tc in last.tool_calls:
        tool_name, tool_args = tc["name"], tc["args"] or {}
        result = execute_tool(tool_name, tool_args)
        is_error = isinstance(result, dict) and result.get("error", False)
        effect = TOOL_EFFECTS.get(tool_name, ToolEffect())

        if effect.sets_flag:
            flag_updates[effect.sets_flag] = True
        if not is_error and effect.chunk_extractor:
            chunks = chunks + effect.chunk_extractor(result)

        if is_error:
            if effect.counts_failures:
                failures += 1
            logger.info(f"Tool '{tool_name}' failed ({result.get('error_type')}). "
                        f"Consecutive fetch errors: {failures}")
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
    """纠正注入（决策 D：两个图节点共用本函数体，functools.partial 绑定 kind）。"""
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


def inject_synthesis(state: AgentState):
    """合成提示注入（v0.5 真实跑修）：临近 turn 上限仍在检索 → 逼 executor 停手、综合产出。
    每子任务最多一次（synthesis_forced 标志，step_init/retry_reset 会重置）。"""
    logger.info("Injecting synthesis instruction (approaching turn limit).")
    return {
        "messages": [{"role": "user", "content": SYNTHESIS_MESSAGE}],
        "synthesis_forced": True,
    }


# ============================================================
# 节点 · critic（审单步，职责边界 §4 / §8）
# ============================================================

def _critic_messages(state: AgentState, query: str, text: str) -> list[dict]:
    chunks = state.get("retrieved_chunks", [])
    sources = ", ".join(f"{c.get('doc')}#{c.get('section')}" for c in chunks) or "（本步未检索到来源）"
    return [
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": (
            f"子任务问题：{query}\n\n执行器产出：\n{text}\n\n"
            f"本步真实召回来源（引用白名单）：{sources}\n\n请裁决。")},
    ]


def critic(state: AgentState):
    """
    审 executor 单步输出，写 critic_verdict（accept/retry/escalate）。
    硬闸门（比执行器更严，职责边界 §8）先于 LLM 裁决跑——空/占位、引用非法直接判 retry；
    跑满 turn 仍无答案 → escalate（重做无益，交 planner）。retry 时 retry_count +1（业务层，
    与传输层 empty_retries 分开，决策 B / E5）。结果按 step_id 写进 step_results（de-dup：
    retry 重做覆盖旧条，happy path 等价 append；节点内手动累加、不上 reducer，E3）。
    """
    step_id = state.get("step_index", 0)
    subtask = _current_subtask(state)
    query = subtask.get("query", state.get("user_message", ""))
    text = _last_ai_content(state)
    chunks = state.get("retrieved_chunks", [])
    citations = _extract_citations(text)
    retry_count = state.get("retry_count", 0)

    feedback = ""
    grounding = _citation_grounding(citations, chunks)
    if is_placeholder_answer(text):
        if state.get("turn_count", 0) >= MAX_TURNS:
            verdict = "escalate"  # 跑满 turn 仍无答案：重做无益，交 planner 跳过
        else:
            verdict, feedback = "retry", "上一步产出为空/占位符，请重新检索并给出有依据的结论。"
    elif citations and grounding < CITATION_MIN_GROUNDING:
        # 软闸门：多数引用溯源不到本步召回 → 多半是凭记忆编的，重做（而非"一条不符就毙"）
        logger.info(f"Critic step {step_id}: grounding={grounding:.2f} < {CITATION_MIN_GROUNDING} "
                    f"({len(citations)} cites) → retry")
        verdict = "retry"
        feedback = "多数引用对不上本步召回来源，请基于 retrieve_documents 的返回重写、只引检索到的 [doc#section]。"
    else:
        raw = call_critic_model(_critic_messages(state, query, text))
        verdict = _parse_verdict(raw)
        feedback = _parse_feedback(raw)

    if verdict == "retry":
        retry_count += 1

    result = {
        "step_id": step_id,
        "query": query,
        "text": text,
        "citations": citations,
        "status": verdict,
        "plan_version": state.get("plan_version", 0),
    }
    new_results = [r for r in state.get("step_results", []) if r.get("step_id") != step_id] + [result]

    logger.info(f"Critic step {step_id}: verdict={verdict} retry_count={retry_count}")
    return {
        "critic_verdict": verdict,
        "critic_feedback": feedback if verdict == "retry" else "",
        "retry_count": retry_count,
        "step_results": new_results,
    }


# ============================================================
# 节点 · 收尾链尾（v4.2 反转时序：finalize → human_review → update_memory）
# ============================================================

def _build_report(state: AgentState) -> str:
    """从 plan + step_results 组装结构化调研报告（v5.0 finalize 产出）。"""
    question = state.get("user_message", "")
    plan = state.get("plan", [])
    results_by_id = {r.get("step_id"): r for r in state.get("step_results", [])}

    lines = [f"# 调研报告：{question}", "", f"## 研究计划（{len(plan)} 步）"]
    for s in plan:
        lines.append(f"{s.get('id', 0) + 1}. {s.get('query', '')} — {s.get('status', 'pending')}")
    lines += ["", "## 分步结论"]

    any_result = False
    for s in plan:
        r = results_by_id.get(s.get("id"))
        lines.append(f"### {s.get('id', 0) + 1}. {s.get('query', '')}")
        if r and not is_placeholder_answer(r.get("text", "")) and r.get("status") == "accept":
            lines.append(r.get("text", ""))
            any_result = True
        elif r and (r.get("text") or "").strip() and not is_placeholder_answer(r.get("text", "")):
            lines.append(f"_(未通过评审：{r.get('status')})_\n{r.get('text', '')}")
            any_result = True
        else:
            lines.append("_(本步未产出有效结论)_")
        lines.append("")

    cites = sorted({c for r in state.get("step_results", []) for c in (r.get("citations") or [])})
    if cites:
        lines += ["## 引用来源"] + [f"- {c}" for c in cites] + [""]

    reason = state.get("termination_reason", "") or "done"
    lines.append(f"---\n_终止原因：{reason}；re-plan {state.get('replan_count', 0)} 次；"
                 f"plan_version={state.get('plan_version', 0)}；"
                 f"空回答重试 {state.get('empty_retries', 0)} 次_")

    report = "\n".join(lines)
    if not any_result:  # 全部步都没产出有效结论 → 占位符（与判据名单一致，防假阳性）
        return f"{PLACEHOLDER_EMPTY} 调研未能产出有效结论。\n\n{report}"
    return report


def finalize(state: AgentState):
    """组装结构化调研报告（v5.0：从 step_results + plan 拼，不再是单答案）。
    answer 已被写入（极少数短路）则尊重；否则建报告。后接 human_review（可改写）。"""
    if state.get("answer"):
        return {"answer": state["answer"]}
    return {"answer": _build_report(state)}


def human_review(state: AgentState):
    """输出审批（决策 F，v4.2 反转后在 finalize 之后——审批框永远是组装好的报告）。"""
    if not settings.INTERRUPT_ENABLED:
        return {}
    draft = state.get("answer", "")
    decision = interrupt({"draft_answer": draft, "hint": "回复 approve 通过，或直接给出改写后的报告"})
    if isinstance(decision, str) and decision.strip() \
            and decision.strip().lower() not in {"approve", "ok", "yes", "y", "通过"}:
        return {"answer": decision.strip()}
    return {}


def _trace_shim(state: AgentState) -> SimpleNamespace:
    """为 memory.update_from_turn 构造 duck-type shim（memory/ 零改动）。
    扫**全部子任务窗口**（整个调研的 messages，跨子任务）配对 tool_calls 与 ToolMessage。"""
    id_to_name: dict = {}
    calls = []
    for m in state.get("messages", []):  # v5.0：记忆记整个调研任务，扫全量 messages
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
    # 整个调研召回的全部来源（各步 retrieved_chunks 已并入 step_results.citations 的白名单语义）
    all_chunks = []
    for r in state.get("step_results", []):
        for cit in r.get("citations", []) or []:
            doc, _, section = cit.partition("#")
            all_chunks.append({"doc": doc, "section": section})
    return SimpleNamespace(
        turns=[SimpleNamespace(tool_calls=calls)],
        searched=any(r.get("status") for r in state.get("step_results", [])),  # 调研必经检索/搜索
        retrieved=bool(all_chunks),
        retrieved_chunks=all_chunks,
    )


def update_memory(state: AgentState, config, *, store: BaseStore):
    """抽取主题/偏好/事实、写长期（v3.0 memory.update_from_turn）。v4.2 时序：在 human_review 之后。
    v5.0：记录的是整份调研报告（用户实际看到的版本）；占位符报告跳过（防级联污染段 5）。"""
    memory = _memory_from(config)
    if memory is None:
        return {}
    answer = state.get("answer", "")
    if is_placeholder_answer(answer):
        logger.info("Skipping memory update: placeholder report this turn.")
        return {}
    memory.update_from_turn(state["user_message"], answer, _trace_shim(state), store)
    return {}
