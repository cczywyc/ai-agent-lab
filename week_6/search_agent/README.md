# 搜索 Agent v4.1 — LangGraph 状态化工作流 + Store 长期记忆

> v4.0 按 [第六周设计草稿](../docs/第六周设计草稿.md)（v0.3）实现；
> v4.1 按 [Store集成与重构计划](../docs/Store集成与重构计划.md) 落地决策 E——
> 长期记忆迁 LangGraph SqliteStore，验证见 [Store迁移验证记录](../docs/Store迁移验证记录.md)。
> 基线：`search_agent v3.0`（week_4&5）。E1–E5 实验结论（13/13 判据）全部落进实现。

## v4.0 → v4.1 变了什么（Store 迁移）

| | v4.0 | v4.1 |
|---|---|---|
| 长期记忆存储 | 自研 `long_term.py`：3 json + 共用向量库（下标对齐） | LangGraph **SqliteStore**（`data/ltm.db`），3 namespace：`("ltm","preferences"/"facts"/"topics")` |
| 事实写入 | 手动 `embed_texts` 批量 → `add_fact(f, vector)` | `store.batch(PutOp×N)`，索引 store 内部建（1 批 = 1 次 embed） |
| 事实召回 | 手动 `embed_query` → `recall_facts(q_vec)` | `store.search(ns, query=..., limit=k)` 原生语义搜索 |
| store 来源 | — | `compile(checkpointer=..., store=...)`，节点签名 `store: BaseStore` 注入 |
| 边界 | — | checkpointer 管 thread 内短期状态；store 管跨 thread 长期记忆 |
| 不动 | 控制流图 / 短期记忆 / 摘要 / 抽取 / 装配"挑选"逻辑（段顺序/预算/裁剪）/ 整个 `rag/` | 同左 |

顺手修了两个问题（详见验证记录 §五，回归测试 S10/S11 钉住）：
记忆开启时装配窗口丢段 1-3 的 v4.0 既有 bug；空回答轮次污染短期记忆。

## v3.0 → v4.0 变了什么

| | v3.0 | v4.0 |
|---|---|---|
| 控制流 | `run_agent` 的 317 行 for 循环 + continue | `StateGraph` 节点/条件边（`graph.py`） |
| 状态 | `run_agent` 局部变量 | `AgentState` TypedDict（`state.py`，草稿 §1 的 14 字段） |
| 跨轮持久化 | 每次调用重建 | `checkpointer`（InMemorySaver）按 `thread_id` 归档 |
| trace | `AgentTrace` 三层 dataclass | 吸收进 state（`correction_triggered` 等字段） |
| HITL | 无 | `human_review` 节点 + `INTERRUPT_ENABLED` 开关（决策 F，默认关） |
| memory/ 与 rag/ | — | **零改动**（决策 A/E：Store 迁移推迟——已在 v4.1 落地） |

## 文件结构

```
search_agent/
├── state.py     # AgentState schema + per-query 默认值（决策 B/C，E2/E3 实证落地）
├── nodes.py     # 10 个节点：init/assemble/agent/tools/inject_*×3/update_memory/human_review/finalize
│                #   v4.1：assemble/update_memory 签名加 store: BaseStore（LangGraph 注入）
├── edges.py     # 条件边：route_after_agent / need_correction / after_tools / gate_to_agent
├── graph.py     # 状态图组装 + compile(checkpointer=InMemorySaver(), store=get_ltm_store())
├── main.py      # CLI 入口（与 v3.0 对齐，新增 --review / --state）
├── test_graph.py        # 图结构测试：桩模型+桩工具，离线可复现（37 项判据）
├── test_store_memory.py # v4.1 Store×记忆测试：InMemoryStore+stub embed，离线（42 项判据）
├── migrate_ltm.py           # 一次性迁移：json 三件套 → SqliteStore（幂等，已执行）
├── verify_store_migration.py# 召回基线/对照验证（baseline 已存档，--after/--compare 可重跑）
├── config.py    # v3.0 + INTERRUPT_ENABLED / RECURSION_LIMIT
├── memory/      # v4.1 重构：ltm_store.py（Store 工厂/namespace/embed 适配）+
│                #   long_term.py（瘦身为 store 读写函数）；短期/摘要/抽取/装配"挑选"不动
├── tools.py / checks.py / rag/   # v3.0 原样（rag 向量库只剩文档 chunk）
└── data/        # docs 向量库 + ltm.db（v4.1 长期记忆）+ 旧 memory json（保留对照，已不读）
```

## 实现层的三个关键解释（草稿没规定、实现时定的）

1. **装配窗口切片**：checkpointer 把 thread 的全量 messages 存成审计日志，但发给模型的
   窗口只取本问题装配块起的切片（`nodes._context_window`）——复刻 v3.0
   "每个问题由六段装配重建上下文"的语义。历史对话已被装配压缩进段 3/5，不重复发原始消息。
   v4.1 修正锚点：取"最后一条**内容等于 SYSTEM_PROMPT** 的 system 消息"（记忆开启时
   assemble 产出多条 system，按"最后一条 system"切会把段 1-3 切掉——v4.0 既有 bug）。
2. **memory 零改动的代价**：`update_memory` 节点从本问题窗口构造 duck-type shim
   （`_trace_shim`）喂给 `MemoryManager.update_from_turn`——它只用到
   `trace.turns[*].tool_calls[*].tool_name/.result_success` 和 `.searched/.retrieved`。
3. **answer 短路通道**：LLM 调用失败（agent 节点）和人工改写（human_review 节点）都提前写
   `answer` 字段，`route_after_agent`/`finalize` 尊重已写入的值——v3.0 的提前 return 在图上
   的等价物。

## 跑法

```bash
../../.venv/bin/python test_graph.py            # 图结构测试（离线，无 API）
../../.venv/bin/python test_store_memory.py     # Store×记忆测试（离线，无 API）
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
