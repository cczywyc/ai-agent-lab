"""
AgentState — v4.0 状态 schema（设计草稿 v0.3 §1 原样落地）

设计要点：
  - 唯一的 reducer 是 messages 的 add_messages，其余字段全部默认替换
    （图基本是顺序的，没有并行分支同时写同一字段）。
  - retrieved_chunks 刻意不上 operator.add：E3 实证带累加 reducer 的字段
    `return []` 是"追加空列表"的 no-op，init 无法清空。累加在 tools 节点
    手动做（"当前 + 新"），init 返回 [] 清空——轮内累加 + 跨问题清空两者兼得。
  - per-query 字段的默认值收在 PER_QUERY_DEFAULTS，init 节点整体返回。
    E2 实证 TypedDict 字段无隐式默认值（首轮裸取 KeyError），所以 init
    的职责是双重的：首轮建立默认值 + 后续轮重置上一问题的残留。
  - 条件边读 per-query 标志一律 state.get(k, 默认)（v0.3 实现约束），
    不依赖"init 一定先跑过"。
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ===== 累加字段（唯一 reducer）=====
    # 工作消息流：装配出的初始 messages、模型回复、工具结果、纠正/降级消息
    # 都追加进来。配合 checkpointer 跨轮持久化（thread 级审计日志）；
    # 发给模型的窗口由 agent 节点按"最后一条 system 消息"切片（见 nodes._context_window）。
    messages: Annotated[list, add_messages]

    # ===== 替换字段 =====
    user_message: str                      # 本轮问题；checks 和 assemble 都读它（invoke 入参带入）
    retrieved_chunks: list                 # 轮内累加（tools 节点手动）、init 清空（决策 B）
    has_searched: bool                     # 是否调过 web_search（含失败="尝试过"，防反复纠正）
    has_retrieved: bool                    # 是否调过 retrieve_documents
    retrieval_correction_injected: bool    # 检索纠正最多一次
    search_correction_injected: bool       # 联网纠正最多一次
    fallback_injected: bool                # 降级最多一次
    consecutive_failures: int              # 工具连续失败计数（仅 fetch_webpage 累计），成功清零
    turn_count: int                        # 循环上限保护（agent 节点内 +1，边上闸门查它）
    empty_retries: int                     # 空回答节点内重试累计（可观测；本问题内手动累加）
    assembly_report: Optional[dict]        # 六段装配报告（可观测）
    correction_triggered: bool             # trace 语义字段（v3.0 AgentTrace 吸收进 state）
    fallback_triggered: bool               # 同上
    answer: str                            # 最终回答（finalize 写；LLM 错误/人工改写可提前写入作短路通道）


# init 节点整体返回这份默认值（决策 C，v0.3 扩充版）。
# 注意两个刻意的"不包含"：
#   - messages：不返回 = 保留（E2 的关键不对称——返回了就会动持久化历史）
#   - user_message：每次 invoke 由输入带入，无需 init 管
PER_QUERY_DEFAULTS: dict = {
    "retrieved_chunks": [],
    "has_searched": False,
    "has_retrieved": False,
    "retrieval_correction_injected": False,
    "search_correction_injected": False,
    "fallback_injected": False,
    "consecutive_failures": 0,
    "turn_count": 0,
    "empty_retries": 0,
    "assembly_report": None,
    "correction_triggered": False,
    "fallback_triggered": False,
    "answer": "",
}
