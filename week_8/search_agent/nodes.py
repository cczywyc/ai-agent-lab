"""
nodes.py — v6.0 节点（规则：节点 = 干活 / 改 state；路由决策见 edges.py）

v6.0 第八周：在 v5.0 外循环上把角色节点升格成 supervisor 协调的独立 workers（升格不重写）。
  supervisor  = planner 升格：拆研究子任务 / 收编每步 findings 推进 / skip-and-advance；
                并按阶段设 active_worker + task_description 四要素派活（call_supervisor_model 可桩）
  researcher  = executor 引擎升格（assemble/agent/tools/inject_*/critic 内层 v5.0 原样复用）：
                按研究子任务检索 + 压缩 findings 回传（带引用）
  writer      = **全新**角色：把 findings 组织成初稿（保留引用），不检索不自评（call_writer_model 可桩）
  reviewer    = critic 升格：审 draft 出 verdict(accept/reject)+notes+score，不改稿（call_reviewer_model 可桩）

新增节点（设计草稿 v0.2 §三/§四/§五）：supervisor（替 planner）/ writer / reviewer。
复用节点（v5.0 原样 / 按需参数化）：
  init      重置升级：PER_TASK_DEFAULTS（含 v6.0 多 Agent 字段，init 清）+ PER_SUBTASK_DEFAULTS（init+step_init 清）
  step_init / retry_reset / assemble(role=researcher) / agent / tools / inject_*  = researcher 引擎（v5.0 原样）
  critic    researcher 内层单步自检（accept/retry/escalate），出口接 supervisor（见 edges.py）
  finalize  **改**：交付被评审的 draft（best_draft / accepted），不再从 step_results 拼报告
  human_review / update_memory  收尾链尾原样

隔离（E5，本周唯一真·新机制）：writer/reviewer 喂 LLM 的是 views.py 投影（writer_view/reviewer_view），
**绝不是 state 本身**——LangGraph 不分区、越界读 reviewer 私有字段不报错只静默串台、框架无护栏。

五档模型调用各自独立成函数（测试用 nodes.call_*_model = stub 替换）：
  call_model（researcher executor）/ call_supervisor_model（拆解）/ call_critic_model（researcher 内层自检）/
  call_writer_model（写稿）/ call_reviewer_model（审稿）。

关键实现解释（沿 v5.0 + v6.0 新增）：
  1. 装配窗口切片 _window_start：每研究子任务 assemble 重产 SYSTEM_PROMPT，窗口自锚、隔离前序 tool 历史（E7）。
  2. 计数器正交（E4）：empty_retries（传输层）/ retry_count（researcher 业务层）/ replan_count（supervisor skip）/
     review_count（writer↔reviewer 打回）——四者各兜各的，每个"谁写"≡闸门"读谁"同 key。
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
    SUPERVISOR_PROMPT,
    CRITIC_PROMPT,
    WRITER_PROMPT,
    REVIEWER_PROMPT,
    REVIEW_RUBRIC,
    MAX_REVIEW,
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
from views import (
    researcher_view,
    writer_view,
    reviewer_view,
    visible_keys,
    render_writer_input,
    render_reviewer_input,
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


def call_supervisor_model(messages: list[dict]) -> str:
    """supervisor 拆解模型调用——纯文本（不带工具，supervisor 不碰检索）。返回模型文本。
    注意：supervisor **路由**是条件函数、不接 LLM（E1）；这个函数只喂"拆研究子任务"这一次调用。"""
    resp = settings.client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content or ""


def call_critic_model(messages: list[dict]) -> str:
    """researcher 内层 critic 模型调用——纯文本（裁决 + feedback）。返回模型文本。"""
    resp = settings.client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content or ""


def call_writer_model(messages: list[dict]) -> str:
    """writer 模型调用——纯文本（把 findings 写成初稿，不带工具）。返回模型文本。"""
    resp = settings.client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content or ""


def call_reviewer_model(messages: list[dict]) -> str:
    """reviewer 模型调用——纯文本（VERDICT/SCORE/NOTES 三行，不带工具）。返回模型文本。"""
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


def _parse_review_verdict(text: str) -> str:
    """从 reviewer 文本解析 accept/reject。默认 accept（含糊时不死循环；MAX_REVIEW 闸门兜底）。"""
    t = (text or "").lower()
    m = re.search(r"verdict[:：]\s*(accept|reject)", t)
    if m:
        return m.group(1)
    if "reject" in t:
        return "reject"
    return "accept"


def _parse_review_score(text: str) -> float:
    """从 reviewer 文本解析 SCORE: 0.0~1.0（best-so-far 据此取历史最好稿）。缺省 0.5。"""
    m = re.search(r"score[:：]\s*([01](?:\.\d+)?|0?\.\d+)", text or "", flags=re.I)
    if not m:
        return 0.5
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.5


def _parse_review_notes(text: str) -> str:
    """从 reviewer 文本解析 NOTES: 逐条意见（喂下一稿，Reflexion）。"""
    m = re.search(r"notes[:：]\s*(.+)", text or "", flags=re.I | re.S)
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
    """role=researcher 的当前问题段（职责边界 §5 局部视角 / §三 角色 I/O 契约）：当前研究子任务 query +
    **handoff 携带的 boundary**（task_description 第 4 要素：别查什么、归谁管，§三/E2）+
    前序步结论摘要（看摘要、不看其他子任务检索原文）+ （retry 时）critic 反馈。
    这本身就是 researcher_view 的投影实现——researcher 只读自己该读的切片，不读 draft/review_* 等私有字段（E5）。"""
    subtask = _current_subtask(state)
    k = state.get("step_index", 0)
    total = len(state.get("plan", [])) or 1
    parts = [f"# 研究子任务 {k + 1}/{total}", f"问题：{subtask.get('query', '')}"]
    boundary = (state.get("task_description", {}) or {}).get("boundary", "")
    if boundary:
        parts.append(f"\n## 研究边界（只做这条子任务，别碰以下、那归别的角色）\n{boundary}")
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
    走在 route_supervisor → step_init → assemble 这条边上（supervisor 派 researcher 新研究子任务）；retry 边走
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
# 节点 · supervisor（= planner 升格：拆研究子任务 / 收编 findings 推进 / skip / 按阶段派活）
# ============================================================
# supervisor 三件事：① 入口拆解写作主题→研究子任务；② researcher 回来后收编/skip 并推进；
# ③ 按阶段（研究→写作→评审）设 active_worker + task_description 四要素派下一个 worker（路由本身
# 是条件函数、不接 LLM——E1）。前两件复用 v5.0 planner 的 decompose/advance/escalate 几乎原样，
# 真正新增的只是"研究做完不去 finalize 而去 writer"和"派活时携带 task_description 契约"。

def _supervisor_messages(state: AgentState) -> list[dict]:
    """supervisor 拆解的输入（decompose）：把写作主题拆成有序研究子任务。只在入口调一次模型。"""
    topic = state.get("user_message", "")
    return [
        {"role": "system", "content": SUPERVISOR_PROMPT.format(max_steps=MAX_STEPS)},
        {"role": "user", "content": f"技术写作主题：{topic}\n\n请拆解成有序的研究子任务（JSON 字符串数组）。"},
    ]


def _supervisor_decompose(state: AgentState):
    raw = call_supervisor_model(_supervisor_messages(state))
    queries = _parse_plan(raw) or [state.get("user_message", "")]
    queries = queries[:MAX_STEPS]
    plan = [{"id": i, "query": q, "status": "pending"} for i, q in enumerate(queries)]
    logger.info(f"Supervisor decomposed into {len(plan)} research subtasks.")
    return {"plan": plan, "step_index": 0, "plan_version": 1,
            "done": False, "termination_reason": ""}


# ---- task_description 四要素契约（§三）：每次 handoff 携带，缺一 worker 就漂 ----
# 第 4 条 boundary 是 v5.0 唯一系统性缺的（A 拓扑共享 state 天然不撞车，B 拓扑各自独立上下文必须显式写归属）。

def _researcher_td(state: AgentState) -> dict:
    subtask = _current_subtask(state)
    return {
        "objective": subtask.get("query", state.get("user_message", "")),
        "output_format": "findings: [{point, citations}]（每条要点带 [doc#section] 引用）",
        "tools_hint": ["retrieve_documents", "web_search", "fetch_webpage"],
        "boundary": "只做这条研究子任务的检索+压缩；别写最终稿（writer 的活）、别自评（reviewer 的活）",
    }


def _writer_td(state: AgentState) -> dict:
    return {
        "objective": "把 researcher 回传的 findings 组织成结构清晰的技术初稿",
        "output_format": "结构化初稿（引言点题 / 分点展开 / 结论），每条论断保留 [doc#section] 引用",
        "tools_hint": [],
        "boundary": "不自己检索（findings 已备齐）、不自评；返修时按 review_notes 针对性改",
    }


def _reviewer_td(state: AgentState) -> dict:
    return {
        "objective": "审 writer 初稿，按二元 rubric 出 verdict(accept/reject)+notes+score",
        "output_format": "VERDICT / SCORE / NOTES 三行",
        "tools_hint": [],
        "boundary": "不改稿、不重写（改稿是 writer 的事）；只发裁决信号 + 具体修改意见",
    }


# ---- 收编每个研究子任务的结果（critic accept → 压缩成 finding 收编推进；escalate → skip-and-advance）----

def _make_finding(state: AgentState, status: str) -> dict:
    """把当前研究子任务的 step_result 压缩成一条 finding（回收时压缩、不倒全轨迹，§四）。"""
    k = state.get("step_index", 0)
    subtask = _current_subtask(state)
    sr = next((r for r in state.get("step_results", []) if r.get("step_id") == k), None)
    if status == "ok" and sr:
        point = (sr.get("text", "") or "").strip()[:600]
        cites = list(sr.get("citations", []) or [])
    else:
        point, cites = "（本研究子任务未能产出有依据的结论，已跳过）", []
    return {"subtask": subtask.get("query", ""), "point": point, "citations": cites, "status": status}


def _advance_research(state: AgentState) -> dict:
    """critic accept：标记当前研究子任务 done、推进 step_index（不设 done——研究做完去 writer，不去 finalize）。"""
    plan = [dict(s) for s in state.get("plan", [])]
    k = state.get("step_index", 0)
    if 0 <= k < len(plan):
        plan[k]["status"] = "done"
    return {"plan": plan, "step_index": k + 1}


def _skip_research(state: AgentState) -> dict:
    """critic escalate / retry 用尽：**skip-and-advance**（v5.0 原样）——标记 skipped、replan_count+1、推进。
    replan_count（supervisor 级）独立于 review_count（writer↔reviewer 级），各兜各的正交维度（E4）。"""
    plan = [dict(s) for s in state.get("plan", [])]
    k = state.get("step_index", 0)
    if 0 <= k < len(plan):
        plan[k]["status"] = "skipped"
    replan = state.get("replan_count", 0) + 1
    logger.info(f"Supervisor skip research step {k} → advance to {k + 1} (skip count={replan}).")
    return {"plan": plan, "step_index": k + 1, "replan_count": replan}


def _supervisor_collect_research(state: AgentState) -> dict:
    """researcher 子任务回来：accept → 压缩成 finding 收编 + 推进；escalate → 记一条 skipped finding + skip。
    findings **节点内手动累加**（当前 + 新），不上累加 reducer（E3：带累加 reducer 的字段 return [] 清不掉）。"""
    verdict = state.get("critic_verdict", "")
    if verdict == "accept":
        finding = _make_finding(state, "ok")
        adv = _advance_research(state)
    else:                                    # escalate / retry 用尽 → 跳过该研究子任务
        finding = _make_finding(state, "skipped")
        adv = _skip_research(state)
    adv["findings"] = list(state.get("findings", [])) + [finding]
    return adv


def _supervisor_route(state: AgentState) -> dict:
    """按阶段（研究→写作→评审）派下一个 worker：设 active_worker + task_description（route_supervisor 读 active_worker）。
    三级跌落（E1 阶段判定）：还有研究子任务→researcher / 有 findings 无 draft→writer / 有 draft 无 verdict→reviewer。"""
    plan = state.get("plan", [])
    # 研究阶段：仍有未做研究子任务，且 skip 没烧穿 replan 闸门
    if state.get("step_index", 0) < len(plan) and state.get("replan_count", 0) < MAX_REPLAN:
        return {"active_worker": "researcher", "task_description": _researcher_td(state)}
    # 研究做完（或 replan 烧穿放弃剩余）→ 写作阶段
    ok_findings = [f for f in state.get("findings", []) if f.get("status") == "ok"]
    if not ok_findings:
        return {"active_worker": "finalize", "task_description": {}}      # 无一条有效研究产出 → 不让 writer 从空气里写，收口占位
    if not state.get("draft"):
        return {"active_worker": "writer", "task_description": _writer_td(state)}
    # 有 draft 待审（writer 写完会把 review_verdict 清成 ""）→ 评审阶段
    if state.get("review_verdict", "") == "":
        return {"active_worker": "reviewer", "task_description": _reviewer_td(state)}
    return {"active_worker": "finalize", "task_description": {}}          # 兜底（reviewer 边正常已处理 accept/best-so-far）


def supervisor(state: AgentState):
    """
    三态：plan 为空 → 拆研究子任务；刚从 researcher 回来（active_worker=researcher 且 critic 给了 verdict）
    → 收编 finding / skip 并推进；其余 → 直接按阶段派下一个 worker。最后统一用 _supervisor_route 设
    active_worker + task_description（节点干活、路由决策交 route_supervisor 边读 active_worker，E1）。
    supervisor 不自己检索/写作/评审——只拆任务、派活、判 done、收口（决策 B）。
    """
    if not state.get("plan"):
        updates = _supervisor_decompose(state)
    elif state.get("active_worker") == "researcher" and state.get("critic_verdict", ""):
        updates = _supervisor_collect_research(state)
    else:
        updates = {}
    merged = {**state, **updates}
    routing = _supervisor_route(merged)
    return {**updates, **routing}


# ============================================================
# 节点 · assemble（按 role 参数化）
# ============================================================

def assemble(state: AgentState, config, *, store: BaseStore, role: str = "researcher"):
    """
    六段装配从时间维度扩到空间维度（按角色挑段，设计 ④）。
    role=researcher：每研究子任务重产 SYSTEM_PROMPT 锚（段 1）→ 窗口自锚到本子任务、隔离前序
    子任务 tool 历史（E7）；段 6 = 本子任务局部上下文（query + handoff boundary + 前序摘要 + retry 反馈）。
    _executor_user_content 本身就是 researcher_view 的投影实现——只读 researcher 该读的切片（E5）。
    记忆开启时复用 v3.0 六段装配（段 2 偏好 / 段 3 摘要 / 段 4 长期事实召回都进窗口）——
    跨任务的偏好与事实仍影响每个研究子任务的产出（read-side 记忆不因外循环而丢）。
    （supervisor 的"看摘要"视角由 supervisor 节点内部处理，不走本节点；writer/reviewer 各有自己的
    投影节点，不走 assemble；role 现仅 "researcher" 一个活值，保留为前向扩展钩子——decision C 的 partial 先例。）
    """
    content = _executor_user_content(state)
    memory = _memory_from(config)
    if memory is not None:
        msgs, report = memory.assemble_context(content, SYSTEM_PROMPT, store)
        report_dict = asdict(report) if is_dataclass(report) else dict(vars(report))
        logger.info(f"Memory assembled (researcher step {state.get('step_index', 0)}): "
                    f"segs={report_dict.get('segments_present')} "
                    f"facts_recalled={report_dict.get('facts_recalled')}")
    else:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},   # 段 1 锚（E7 重锚）
            {"role": "user", "content": content},
        ]
        report_dict = {"role": role, "step_index": state.get("step_index", 0)}
    # 记 researcher 可见集（E5 隔离断言/trace）：契约视图 = researcher_view 的键（隔离开=窄视图、关=全 state）。
    # researcher 的实际 prompt 由 _executor_user_content 建（query+boundary+前序摘要，本身就是不泄漏 draft/review_* 的投影）。
    visible = visible_keys(researcher_view(state, settings.ISOLATION_ENABLED))
    return {"messages": msgs, "assembly_report": report_dict, "_researcher_visible": visible}


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
# 节点 · critic（researcher 内层单步自检，职责边界 §4 / §8；v5.0 原样、出口接 supervisor）
# 注意：这是 researcher **内层**对单条研究子任务的自检；draft 级的质量审是外层独立的 reviewer（critic 升格）。
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
            verdict = "escalate"  # 跑满 turn 仍无答案：重做无益，交 supervisor 跳过该研究子任务
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
# 节点 · writer（全新角色：findings → 初稿；§三 I/O 契约 + §四 隔离投影）
# ============================================================

def writer(state: AgentState):
    """
    把 researcher 回传的 findings 组织成初稿（保留引用）。**不检索、不自评**（决策 B 边界）。
    隔离（E5）：喂 LLM 的是 writer_view 投影（findings + outline + 返修时 review_notes），**不是 state 本身**——
    LangGraph 不分区、越界读 reviewer 私有字段（review_verdict/best_draft）不报错只静默串台、框架无护栏。
    返修时拿"上一稿评审意见 review_notes"针对性改（Reflexion verbal reinforcement，§五）。
    写完把 review_verdict 清成 ""——supervisor 据此把"有 draft 无 verdict"路由到 reviewer（draft 待审）。
    """
    isolation = settings.ISOLATION_ENABLED
    proj = writer_view(state, isolation)
    visible = visible_keys(proj)
    msgs = [
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content": render_writer_input(proj)},
    ]
    draft = (call_writer_model(msgs) or "").strip() or f"{PLACEHOLDER_EMPTY} writer 返回空稿。"
    logger.info(f"Writer drafted ({len(draft)} chars); visible={visible}")
    return {
        "draft": draft,
        "review_verdict": "",                       # 清空 → supervisor 路由到 reviewer 待审
        "_writer_visible": visible,                 # 可观测：writer 实际可见集（隔离断言/trace，E5）
        "worker_result": {"kind": "draft", "chars": len(draft)},   # 压缩回传信封（§四）
    }


# ============================================================
# 节点 · reviewer（critic 升格：审 draft → verdict+notes+score；不改稿；§三/§五）
# ============================================================

def reviewer(state: AgentState):
    """
    审 writer 初稿，按二元 rubric 出 verdict(accept/reject) + notes + score。**不改稿**（改稿是 writer 的事）。
    隔离（E5）：喂 LLM 的是 reviewer_view 投影（draft + rubric），不读 findings（各看各的）。
    best-so-far（§五，防 behavioral collapse）：仅当本稿 score 更优才更新 best_draft——达 MAX_REVIEW 仍未 accept
    时 finalize 取它收口（后稿可能比前稿差，取最新会交付更差的）。
    **写键≡读键铁律（E4）**：reject 时 review_count 在本节点内 +1，route_after_reviewer 也读 review_count——
    同一个 key，否则写读错位会让计数器永不达阈、死锁撞 recursion。review_count 与 replan_count 正交、别混。
    """
    isolation = settings.ISOLATION_ENABLED
    proj = reviewer_view(state, isolation)
    visible = visible_keys(proj)
    msgs = [
        {"role": "system", "content": REVIEWER_PROMPT},
        {"role": "user", "content": render_reviewer_input(proj)},
    ]
    raw = call_reviewer_model(msgs)
    verdict = _parse_review_verdict(raw)
    score = _parse_review_score(raw)
    notes = _parse_review_notes(raw)
    draft = state.get("draft", "")

    # best-so-far：空 dict 或本稿更优才更新（一个 if 的低成本兜底，仅 collapse 时 load-bearing，§五）
    best = state.get("best_draft", {}) or {}
    if not best or score > best.get("score", -1.0):
        best = {"draft": draft, "score": score}

    rc = state.get("review_count", 0)
    logger.info(f"Reviewer verdict={verdict} score={score:.2f} review_count={rc} visible={visible}")
    out = {
        "review_verdict": verdict,
        "review_notes": notes,
        "best_draft": best,
        "_reviewer_visible": visible,                          # 可观测：reviewer 可见集（隔离断言/trace，E5）
        "worker_result": {"kind": "verdict", "verdict": verdict, "score": score},
    }
    if verdict == "reject":
        out["review_count"] = rc + 1                           # 写键≡读键：route_after_reviewer 读 review_count（E4）
    return out


# ============================================================
# 节点 · 收尾链尾（v4.2 反转时序：finalize → human_review → update_memory）
# ============================================================

def _build_delivery(state: AgentState, chosen_draft: str, reason: str) -> str:
    """组装最终交付（v6.0）：把被评审的 writer 初稿包成结构化交付文档——
    研究计划一览（supervisor 拆）+ writer 正文（reviewer 已审）+ findings 引用汇总 + 可观测 meta 脚注。
    与 v5.0 从 step_results 拼报告不同：v6.0 正文是 writer 写的稿、不是各步结论的机械拼接。"""
    topic = state.get("user_message", "")
    plan = state.get("plan", [])
    findings = state.get("findings", [])
    n_skip = sum(1 for s in plan if s.get("status") == "skipped")
    best = state.get("best_draft", {}) or {}

    lines = [f"# 技术写作交付：{topic}", "", f"## 研究计划（{len(plan)} 个研究子任务，supervisor 拆）"]
    for s in plan:
        lines.append(f"{s.get('id', 0) + 1}. {s.get('query', '')} — {s.get('status', 'pending')}")
    lines += ["", "## 正文（writer 初稿，经 reviewer 评审）", "", chosen_draft, ""]

    cites = sorted({c for f in findings for c in (f.get("citations") or [])})
    if cites:
        lines += ["## 引用来源（findings 汇总）"] + [f"- {c}" for c in cites] + [""]

    lines.append(
        f"---\n_交付方式：{reason}；研究子任务 {len(plan)} 个（skip {n_skip}）；"
        f"打回 review_count={state.get('review_count', 0)}/{MAX_REVIEW}；best_score={best.get('score')}；"
        f"replan {state.get('replan_count', 0)}；空回答重试 {state.get('empty_retries', 0)} 次_")
    return "\n".join(lines)


def finalize(state: AgentState):
    """交付被评审的 draft（v6.0：accept → 取当前稿；达上限未 accept → best-so-far 收口取历史最好稿）。
    不再从 step_results 拼报告（那是 writer 的活了）。answer 已被写入（人工改写短路）则尊重。后接 human_review。"""
    if state.get("answer"):
        return {"answer": state["answer"]}

    findings = state.get("findings", [])
    draft = state.get("draft", "")
    ok_findings = [f for f in findings if f.get("status") == "ok"]
    # 研究全失败（无有效 finding）/ 没写出稿 → 占位符（与 is_placeholder_answer 名单一致，防假阳性级联）
    if not ok_findings or not draft:
        return {"answer": f"{PLACEHOLDER_EMPTY} 技术写作未能产出有效初稿"
                          f"（findings={len(findings)}、有效={len(ok_findings)}）。",
                "termination_reason": "no_findings",
                "worker_result": {"kind": "final", "chosen": None}}

    verdict = state.get("review_verdict", "")
    best = state.get("best_draft", {}) or {}
    if verdict == "accept":
        chosen = {"draft": draft, "score": best.get("score")}
        reason = "review_accept（reviewer 早退）"
    else:
        # 达 MAX_REVIEW 仍未 accept → best-so-far 收口（取历史最好稿，防 behavioral collapse，§五 / E6）。
        # best-so-far 仅在"预算内真发生 collapse（后稿更差）"时 load-bearing；单调变好则 best≡latest、机制空转。
        chosen = best or {"draft": draft, "score": None}
        reason = "review_exhausted_best_so_far（达打回上限，取历史最好稿）"
    report = _build_delivery(state, chosen.get("draft") or draft, reason)
    return {"answer": report, "termination_reason": reason,
            "worker_result": {"kind": "final", "chosen": chosen}}


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
