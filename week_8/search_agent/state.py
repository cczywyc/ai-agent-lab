"""
AgentState — v6.0 状态 schema（第八周设计草稿 v0.2 §二 落地）

v6.0 第八周升级：在 v5.0 外循环上把"角色节点"升格成 supervisor 协调的独立 workers。
  - 新增 9 个多 Agent 字段（§二）：active_worker / task_description / findings / draft /
    review_verdict / review_notes / review_count / best_draft / worker_result。全 per-task
    （init 随新主题清零），沿用 v5.0 纪律：除 messages 外一律替换语义、不上累加 reducer；
    findings 需跨研究子任务累加 → **节点内手动累加**（E3：带累加 reducer 的字段 return [] 清不掉）。
  - review_count（writer↔reviewer 级打回）与 replan_count（supervisor 级 skip）**正交、别混**（E4），
    且每个计数器"谁写"必须 ≡ 闸门边"读谁"同一个 key（E4 精确化）。
  - review_verdict **必须声明进 schema**（E5：schema 外字段的更新被框架静默丢弃、下游条件边读不到）。
  - v5.0 字段（plan / step_results / 三计数器 / per-subtask 标志）原样保留——researcher 内层照旧用。

v5.0 第七周升级：在 v4.2 单步引擎外套一层 plan 循环（planner-executor-critic）。
  - 新增 8 个字段（v0.2 锁定 7 个 + v0.3 补接 plan_version）：
    plan / step_index / step_results / critic_verdict / retry_count /
    replan_count / plan_version / done · termination_reason。
  - 重置职责降一层（设计 ①）：v4.2 的 per-query 标志在 v5.0 是**内层（单子任务）**
    状态，必须每个子任务都清零——拆成 PER_SUBTASK_DEFAULTS（init + step_init 都打回）
    与 PER_TASK_DEFAULTS（只在 init 随新问题清零）。E2/E6 坐实"两层重置并存"。
  - retry_count 是 **per-subtask**（每子任务独立业务重试额度，随 step_init 清零）；
    replan_count / plan_version 是 **per-task**（计整个调研任务，不随 step 清零）——
    两类计数器重置粒度不同，别一起放进 step_init（v0.3 §一）。

v4.2 既定纪律（沿用，降一层照旧成立）：
  - 唯一的 reducer 是 messages 的 add_messages，其余字段全部替换语义。
    step_results / retrieved_chunks 要累加，刻意不上 operator.add——E3 实证带累加
    reducer 的字段 `return []` 是 no-op、清不掉；累加在节点内手动"当前 + 新"做。
  - 条件边读字段一律 state.get(k, 默认)（不依赖 init/step_init 一定先跑过）。
  - 新增字段必须全部声明进 AgentState——E5：schema 外字段的更新被框架静默丢弃，
    下游条件边读不到（critic_verdict / done 都靠这条才被边读到）。

跨子任务窗口隔离（E7）：messages 是全图唯一累加字段、init 故意不重置它（保留 thread
历史），所以跨子任务只增不减。"executor 只看当前子任务"靠 assemble(role=executor)
每子任务重产 SYSTEM_PROMPT 锚 + nodes._window_start 切片实现，不靠清空 messages。
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ===== 累加字段（全图唯一 reducer）=====
    # 工作消息流：跨子任务只增不减、init 不清（保留 thread 历史，E7）。
    # 发给模型的窗口由 agent 节点按"最后一条 SYSTEM_PROMPT system 消息"切片
    # （nodes._window_start）——每子任务 assemble 重产锚后天然隔离前序子任务的 tool 历史。
    messages: Annotated[list, add_messages]

    # ===== 替换字段 · per-task（整个调研任务，只在 init 随新问题清零）=====
    user_message: str                      # 本轮调研问题（invoke 入参带入）
    plan: list                             # 有序子任务 [{id, query, status}]；planner 写，re-plan 整表替换
    step_index: int                        # 当前执行到第几个子任务（planner 节点内推进；外循环闸门读它）
    step_results: list                     # 每步结论 [{step_id, query, text, citations, status, plan_version}]
                                           #   executor/critic 写，节点内手动累加（决策 G，E3），init 清空
    replan_count: int                      # re-plan 次数（防绕圈闸门读它）；per-task，不随 step 清零
    plan_version: int                      # plan 版本（planner re-plan 时 +1）；executor 写 store 的结论用它打标（§8 承接）
    done: bool                             # planner 判"调研够了"写它，实际终止由条件边读（节点干活、边做决策）
    termination_reason: str                # 终止原因（all_steps_done / max_steps / max_replan）
    empty_retries: int                     # 空回答节点内重试累计（per-task 观测：整个调研的重试救回量）
    answer: str                            # 最终结构化报告（finalize 写；LLM 错误/人工改写可提前写入作短路）

    # ===== v6.0 第八周 · supervisor 多 Agent 字段（per-task，init 随新主题清零，设计草稿 §二）=====
    # 纪律：除 messages 外一律替换语义、不上累加 reducer（E3：带累加 reducer 的字段 return [] 清不掉）；
    #   findings 需跨研究子任务累加，故**节点内手动**做（当前 + 新），不上 reducer。
    active_worker: str                     # 当前被委派的 worker（researcher/writer/reviewer/finalize）；supervisor 写、route_supervisor 读
    task_description: dict                 # 委派契约四要素（objective/output_format/tools_hint/boundary，§三）；supervisor 写
    findings: list                         # researcher 压缩回传的研究要点 [{subtask, point, citations, status}]，每子任务一项；**节点内手动累加**
    draft: str                             # writer 当前初稿（每次返修覆盖）；writer 写
    review_verdict: str                    # reviewer 裁决 accept/reject（**必须声明进 schema**，E5：schema 外更新被静默丢弃）；reviewer 写、reviewer 边读
    review_notes: str                      # reviewer 本轮逐条意见，喂下一稿（Reflexion verbal reinforcement，§五）；reviewer 写
    review_count: int                      # 打回计数（**独立于 replan_count**，E4 正交维度）；reviewer 节点内 +1、reviewer 边读（写键≡读键）
    best_draft: dict                       # best-so-far {draft, score}：达 MAX_REVIEW 取它收口（防 behavioral collapse，§五）；reviewer 写（仅当更优）
    worker_result: dict                    # worker 压缩回传给 supervisor 的统一信封（隔离上下文 §六：回摘要而非全轨迹）；各 worker 写

    # ===== v6.0 可观测/trace（worker 实际可见集，E5 隔离断言用；必须声明进 schema，否则被静默丢弃）=====
    _researcher_visible: list              # researcher 投影实际读到的键集（隔离开=设计视图、关=全 state）
    _writer_visible: list                  # writer 同上（关隔离→含 reviewer 私有 review_verdict/best_draft = 串台）
    _reviewer_visible: list                # reviewer 同上

    # ===== 替换字段 · per-subtask（每个子任务都清零：init + step_init）=====
    retrieved_chunks: list                 # 当前子任务召回（critic 据此校验引用合法性）；轮内累加、step_init 清空
    has_searched: bool                     # 本子任务是否调过 web_search（含失败="尝试过"，防反复纠正）
    has_retrieved: bool                    # 本子任务是否调过 retrieve_documents
    retrieval_correction_injected: bool    # 检索纠正最多一次（per-subtask）
    search_correction_injected: bool       # 联网纠正最多一次（per-subtask）
    fallback_injected: bool                # 降级最多一次（per-subtask）
    consecutive_failures: int              # fetch_webpage 连续失败计数（本子任务内），成功清零
    turn_count: int                        # 单子任务 tool-use 循环上限保护（agent 节点内 +1，内层闸门查它）
    retry_count: int                       # 业务层重试计数（critic 驱动"换措辞重做该步"）；per-subtask
    synthesis_forced: bool                 # 本子任务是否已注入"停止检索、立即综合"提示（最多一次，防过度检索撑爆 turn 预算）
    critic_verdict: str                    # 单步裁决 accept/retry/escalate（critic 写、条件边读，E5 必须声明）
    critic_feedback: str                   # critic 给 executor 的重做提示（retry 时 assemble 带进窗口）
    correction_triggered: bool             # trace 语义字段（本子任务是否触发过纠正）
    fallback_triggered: bool               # 同上（降级）
    assembly_report: Optional[dict]        # 本子任务的装配报告（可观测）


# per-subtask 默认值：init（首轮建默认 + 跨问题重置）与 step_init（每子任务转移重置）都打回这份。
# retry_count / critic_verdict / critic_feedback 在此——每子任务独立重试额度，
# 子任务 N 不继承前序用光的额度（v0.3 §一；漏放 retry_count 是 E2 原桩踩过的坑）。
PER_SUBTASK_DEFAULTS: dict = {
    "retrieved_chunks": [],
    "has_searched": False,
    "has_retrieved": False,
    "retrieval_correction_injected": False,
    "search_correction_injected": False,
    "fallback_injected": False,
    "consecutive_failures": 0,
    "turn_count": 0,
    "retry_count": 0,
    "synthesis_forced": False,
    "critic_verdict": "",
    "critic_feedback": "",
    "correction_triggered": False,
    "fallback_triggered": False,
    "assembly_report": None,
}

# per-task 默认值：只在 init 随新用户问题清零，不随 step 转移重置（plan/step_results/
# replan_count/plan_version/done 是外层状态；empty_retries 跨子任务累计观测）。
PER_TASK_DEFAULTS: dict = {
    "plan": [],
    "step_index": 0,
    "step_results": [],
    "replan_count": 0,
    "plan_version": 0,
    "done": False,
    "termination_reason": "",
    "empty_retries": 0,
    "answer": "",
    # v6.0 多 Agent 字段（随新主题清零；findings/draft/review_* 都是整个写作任务级，不随研究子任务转移重置）
    "active_worker": "",
    "task_description": {},
    "findings": [],
    "draft": "",
    "review_verdict": "",
    "review_notes": "",
    "review_count": 0,
    "best_draft": {},      # {"draft": str, "score": float}；空 dict 表示尚无最好稿
    "worker_result": {},
}


def fresh_subtask_defaults() -> dict:
    """一份新的 per-subtask 默认值（不共享模块级列表/字典）。"""
    d = dict(PER_SUBTASK_DEFAULTS)
    d["retrieved_chunks"] = []
    return d


def fresh_task_defaults() -> dict:
    """一份新的 per-task 默认值（不共享模块级列表/字典）。"""
    d = dict(PER_TASK_DEFAULTS)
    d["plan"] = []
    d["step_results"] = []
    # v6.0 多 Agent 可变字段也各给一份新的，避免跨调用共享模块级 list/dict
    d["findings"] = []
    d["task_description"] = {}
    d["best_draft"] = {}
    d["worker_result"] = {}
    return d


# 业务 retry（critic → retry_reset → assemble）的轻量重置：把内层执行状态打回初值，
# 让每次 retry 都是带满额 turn 预算的"全新一次重做"（"重做该步"语义，职责边界 §5）；
# 但保留三项：retry_count（业务重试额度，critic 刚 +1）、critic_feedback（本次重做指导，
# assemble 要读进窗口）、retrieved_chunks（让"换措辞重做"仍能引用首次已检索来源）。
# = PER_SUBTASK_DEFAULTS - {retry_count, critic_feedback, retrieved_chunks}
# 与 step_init（新子任务全量重置 PER_SUBTASK_DEFAULTS）对称，区别只在这三项的保留。
_RETRY_RESET_KEEP = {"retry_count", "critic_feedback", "retrieved_chunks"}


def fresh_retry_reset() -> dict:
    """retry 轻量重置字段（保留 retry_count / critic_feedback / retrieved_chunks）。"""
    return {k: v for k, v in PER_SUBTASK_DEFAULTS.items() if k not in _RETRY_RESET_KEEP}
