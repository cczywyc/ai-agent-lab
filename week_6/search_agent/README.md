# 搜索 Agent v4.0 — LangGraph 状态化工作流

> 按 [第六周设计草稿 v0.3](../docs/第六周设计草稿.md) 实现（周四实现日产出）。
> 基线：`search_agent v3.0`（week_4&5）。E1–E5 实验结论（13/13 判据）全部落进实现。

## v3.0 → v4.0 变了什么

| | v3.0 | v4.0 |
|---|---|---|
| 控制流 | `run_agent` 的 317 行 for 循环 + continue | `StateGraph` 节点/条件边（`graph.py`） |
| 状态 | `run_agent` 局部变量 | `AgentState` TypedDict（`state.py`，草稿 §1 的 14 字段） |
| 跨轮持久化 | 每次调用重建 | `checkpointer`（InMemorySaver）按 `thread_id` 归档 |
| trace | `AgentTrace` 三层 dataclass | 吸收进 state（`correction_triggered` 等字段） |
| HITL | 无 | `human_review` 节点 + `INTERRUPT_ENABLED` 开关（决策 F，默认关） |
| memory/ 与 rag/ | — | **零改动**（决策 A/E：Store 迁移推迟） |

## 文件结构

```
search_agent/
├── state.py     # AgentState schema + per-query 默认值（决策 B/C，E2/E3 实证落地）
├── nodes.py     # 10 个节点：init/assemble/agent/tools/inject_*×3/update_memory/human_review/finalize
├── edges.py     # 条件边：route_after_agent / need_correction / after_tools / gate_to_agent
├── graph.py     # 状态图组装 + compile(checkpointer=InMemorySaver())
├── main.py      # CLI 入口（与 v3.0 对齐，新增 --review / --state）
├── test_graph.py# 图结构测试：桩模型+桩工具，离线可复现（37 项判据）
├── config.py    # v3.0 + INTERRUPT_ENABLED / RECURSION_LIMIT
├── tools.py / checks.py / memory/ / rag/   # v3.0 原样复制（决策 A）
└── data/        # docs 向量库复制自 week_4&5（免重新 embedding）；memory json 干净起步
```

## 实现层的三个关键解释（草稿没规定、实现时定的）

1. **装配窗口切片**：checkpointer 把 thread 的全量 messages 存成审计日志，但发给模型的
   窗口只取"**最后一条 system 消息起**"的切片（`nodes._context_window`）——复刻 v3.0
   "每个问题由六段装配重建上下文"的语义。历史对话已被装配压缩进段 3/5，不重复发原始消息。
2. **memory 零改动的代价**：`update_memory` 节点从本问题窗口构造 duck-type shim
   （`_trace_shim`）喂给 `MemoryManager.update_from_turn`——它只用到
   `trace.turns[*].tool_calls[*].tool_name/.result_success` 和 `.searched/.retrieved`。
3. **answer 短路通道**：LLM 调用失败（agent 节点）和人工改写（human_review 节点）都提前写
   `answer` 字段，`route_after_agent`/`finalize` 尊重已写入的值——v3.0 的提前 return 在图上
   的等价物。

## 跑法

```bash
../../.venv/bin/python test_graph.py            # 图结构测试（离线，无 API）
../../.venv/bin/python main.py                  # 交互模式（记忆开）
../../.venv/bin/python main.py --review         # 交互 + human_review 审批
../../.venv/bin/python main.py --query "..."    # 单次查询（--state 看最终状态）
../../.venv/bin/python main.py --test           # 8 用例实测（无记忆，出 test_report.json）
../../.venv/bin/python main.py --memory-demo    # 4 轮记忆 demo（同一 thread）
../../.venv/bin/python main.py --ingest         # 重建向量库（已含 week_6 文档目录）
```

## 实测结果（2026-06-04，LangGraph 1.2.4 / qwen-plus）

- **图结构测试**：37/37 判据通过（离线桩测，覆盖直答/两类纠正/纠正一次性/降级/闸门/interrupt 三态/同 thread 多问题）
- **8 用例实测**：7/8（88%），平均 1.8 轮 / 10.6s——见 `test_report.json`
  - 唯一失败 Case 8（"MCP 协议是什么？"预期联网、实际走本地）：单独重跑即选择 `web_search(✓)`。
    属边界用例的模型随机性——MCP 在第二周笔记里被提过，"本地优先"规则下两种选择都成立；
    v3.0 报告（5/22）同例通过是同一枚硬币的另一面，非图回归。
- **记忆 demo（4 轮同 thread）**：装配/检索纠正/偏好抽取（"请记住…"命中）/事实晋升 11 条/
  eviction 摘要全链路跑通。

## 已知行为差异（刻意的，记录在案）

- **超 MAX_TURNS 收口时不再写记忆**：v3.0 超时路径会调 `update_from_turn`，v0.3 草稿规定
  闸门"超了直接 → finalize"（跳过 update_memory）。按草稿实现。
- **异常 finish_reason**（如 length 截断）按 stop 处理并告警，不再像 v3.0 直接 return——
  罕见分支，content 会正常走纠正判定/收尾。
