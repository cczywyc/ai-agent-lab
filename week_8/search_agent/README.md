# 技术写作 Agent v6.0 — supervisor 多 Agent

> 按 [第八周设计草稿](../docs/第八周设计草稿.md)（v0.2）实现；设计前提见 [多 Agent 概念](../docs/多Agent概念.md)；
> 框架机制验证见 [实验结论](../experiment/实验结论.md)（E1–E7，7/7）。
> 基线：`search_agent v5.0`（week_7，planner-executor-critic 外循环）。
> **核心认识：本周是把"一张图里的角色节点"升格成 supervisor 协调的独立 workers，不是重写引擎。**

输入一个技术写作主题 → supervisor 拆研究子任务 → researcher 逐个检索+压缩 findings →
writer 把 findings 写成初稿 → reviewer 审稿出 verdict →（reject 则带 review_notes 返修，
最多 `MAX_REVIEW` 次，达上限取 best-so-far）→ finalize 交付被评审的稿 → human_review → update_memory。

> **诚实一笔（《什么时候该用多 Agent》反例）**：researcher→writer→reviewer 是一条**强顺序依赖流水线**
> （writer 要 researcher 产出、reviewer 要 writer 草稿）。按 Anthropic"子任务强依赖 → 并行退化成串行"的判据，
> 这里 supervisor 是**串接专家、不是并行 fan-out**——本周买到的是"角色专精 + 干净的 handoff 契约"，
> 不是"真并行更强"。选 supervisor（拓扑 B）是为**学机制**；纯交付报告其实拓扑 A（一张图）更省。

## v5.0 → v6.0 变了什么（升格不重写）

| | v5.0（planner-executor-critic 一张图） | v6.0（supervisor 多 Agent） |
|---|---|---|
| 拓扑 | 一张图、共享 state、角色是节点 | supervisor 协调独立 workers，工具式 handoff 携带 task_description |
| 角色升格 | planner / executor 引擎 / critic | **planner→supervisor**、**executor 引擎→researcher**、**critic→reviewer** |
| 真正新增 | — | **writer 角色**（全新）、**task_description 四要素契约**、**writer↔reviewer 打回循环** |
| State | 22 字段 | **+9 字段**：`active_worker`/`task_description`/`findings`/`draft`/`review_verdict`/`review_notes`/`review_count`/`best_draft`/`worker_result` |
| 闸门 | 内层 `turn_count` + 外层 `replan_count` | **+`review_count`**（打回上限 `MAX_REVIEW=2`，独立于 `replan_count`——E4） |
| 隔离 | 共享池里按角色挑段（§四分层装配） | **每 worker 显式投影函数**（`views.py`）：喂 LLM 的是投影、不是 `state`（E5） |
| 产出 | 从 step_results 拼报告 | writer 写的初稿，经 reviewer 评审后交付（best_draft / accepted） |

researcher 内层引擎（`assemble→agent↔tools↔inject_*→critic`）、Store / RAG / 记忆装配 / `human_review` 审批、
接地软闸门 `CITATION_MIN_GROUNDING` + store 白名单 —— **全部原样复用**。

## 图骨架（设计草稿 §七）

```
START → init → supervisor ─route_supervisor─┬─(researcher)─► step_init ─► assemble ─► agent ─┐
                  ▲   ▲                      │   (researcher 引擎 = v5.0 内层循环原样)        │
    (critic accept收编/escalate skip)        │    agent ┬ tool_calls ► tools ┬ 失败≥2►inject_fallback┘
                  │   └── critic ─route_after_critic─┬─ supervisor          └►inject_synthesis/agent/critic
                  │      (researcher 内层自检)        └─ retry_reset ► assemble（retry 重做该研究子任务）
                  ├─(writer)──► writer ─► supervisor
                  └─(reviewer)─► reviewer ─route_after_reviewer─┬─ writer (reject & review_count<MAX_REVIEW，返修)
                                                                └─ finalize (accept 早退 / 达上限 best-so-far)
   收口：finalize（交付被评审的稿）→ human_review（审批）→ update_memory → END
```

- **supervisor 路由是条件函数、不接 LLM**（E1）：按阶段三级跌落（无 findings→researcher / 有 findings 无 draft→writer / 有 draft 无 verdict→reviewer）。supervisor 只在"拆解"时调一次 LLM。
- **researcher 引擎对外循环是黑盒**：入口 `step_init→assemble(researcher)`、出口 `critic`（accept 收编成 findings、escalate 走 skip-and-advance）。
- **writer↔reviewer 打回循环**：reviewer `route_after_reviewer` 读 `review_count`（与 reviewer 节点 reject 时 `review_count+=1` **写同一个 key**——E4 写键≡读键，否则死锁撞 recursion）。

## 隔离 = 每 worker 显式投影函数（E5，本周唯一真·新机制）

LangGraph **不分区**——每个 worker 节点物理上收到的是全 `state` dict，越界读 reviewer 私有字段
（`review_verdict`/`best_draft`）**不报错、只静默串台、框架无护栏**。所以隔离不是框架给的、是**约定**：

- `views.py` 的 `researcher_view / writer_view / reviewer_view` 是三个投影函数；worker 喂 LLM 的**只能是投影结果**。
- `config.ISOLATION_ENABLED`（默认开）关掉即对照组：worker 直接拿全 `state`，复现"writer 越界读 reviewer 私有"的串台。

## 四档闸门正交（E4 / 决策 H）

| 层 | 闸门 | 兜什么 |
|---|---|---|
| researcher 内层 | `turn_count<MAX_TURNS` + `retry_count<MAX_RETRY` + synthesis-reserve | 单研究子任务的 tool-use 上限 / 业务重做 / 过度检索 |
| supervisor 级 | `replan_count<MAX_REPLAN` | 研究子任务做不下去 → skip-and-advance |
| writer↔reviewer 级 | `review_count<MAX_REVIEW` | 这稿不够好 → 返修；达上限取 best-so-far |
| 框架 | `recursion_limit` | **只兜底**（E7：本任务收口靠显式闸门，距默认 10007 约三个数量级） |

## 跑法

```bash
cd week_8/search_agent
../../.venv/bin/python main.py --ingest            # 重建本地向量库（含 week_8 docs）
../../.venv/bin/python main.py --query "写一篇多 Agent 技术综述"   # 单次技术写作 → 交付稿
../../.venv/bin/python main.py --query "..." --state               # 附带打印最终 state
../../.venv/bin/python main.py --test              # 批量技术写作测试（无记忆、interrupt 关）
../../.venv/bin/python main.py                     # 交互模式（记忆默认开）；--review 开 human_review
```

可观测：研究子任务数 / 各子任务 status / findings 数 / `review_count`（打回）/ `replan_count`（研究 skip）/
`review_verdict` / best_score / `termination_reason`。

## 测试（离线、零 API、可复现）

```bash
../../.venv/bin/python test_graph.py        # v6.0 图结构桩测 M1–M9（48 项判据，把 E1–E7 机制搬进真实图）
../../.venv/bin/python evals.py             # routing_accuracy 种子集（E1 沉淀；真实 3/3，写歪对照 1/3）
../../.venv/bin/python test_store_memory.py # 记忆 × Store 端到端（56 项，已迁移到 v6.0 多 Agent 流）
../../.venv/bin/python test_criteria.py     # 占位符 / 判据纯函数（23 项，v4.2 口径保留）
```

| 桩测 | 验什么（设计草稿已验证清单） | E 对应 |
|---|---|---|
| M1 | happy path：supervisor 拆 → researcher → writer → reviewer accept → 交付 | — |
| M2 | supervisor 三级路由 + `routing_accuracy=3/3`、写歪被抓出 | E1 |
| M3 | task_description 四要素齐全、boundary 到达 researcher | E2 |
| M4 | 恒 reject → `review_count` 闸门收口走 best-so-far、不撞 recursion | E3 |
| M5 | `review_count` 与 `replan_count` 不串扰（写键≡读键） | E4 |
| M6 | 三 worker 可见集=各自投影；关隔离 → writer 越界读 reviewer 私有 | E5 |
| M7 | best-so-far 取历史最好稿（0.8）而非最新更差稿（0.5） | E6 |
| M8 | 内外双闸门互不误伤；`turn_count` **每次进 researcher 归零**（本周接回 plan 内循环后复证） | E7 |
| M9 | researcher 引擎保真：子任务内检索纠正 + fetch 连续失败降级 | — |

> **本周边界（真实 API 跑留周末并入 v0.x）**：prebuilt `create_supervisor` 对拍、真实模型端到端、
> writer↔reviewer 在真模型上的质量收敛。桩测只锁"框架机制成不成立"，不验"策略对不对"。
