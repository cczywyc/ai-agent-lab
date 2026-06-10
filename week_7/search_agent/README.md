# 技术调研 Agent v5.0 — planner-executor-critic 外循环

> 按 [第七周设计草稿](../docs/第七周设计草稿.md)（v0.3）实现；设计前提见
> [规划器与执行器职责边界](../docs/规划器与执行器职责边界.md)；机制验证见
> [实验结论](../experiment/实验结论.md)（E1–E7，20/20 判据）。
> 基线：`search_agent v4.2`（week_6）。**核心认识：本周是给 v4.2 套一层外循环，不是重写。**

输入一个调研问题 → planner 拆研究计划 → 逐子任务 executor 检索+总结、critic 审 →
planner 推进/重试/re-plan → finalize 组装**结构化调研报告** → human_review → update_memory。

## v4.2 → v5.0 变了什么

| | v4.2（单步引擎） | v5.0（外循环） |
|---|---|---|
| 形态 | 单问题 → 单答案的内层 tool-use 循环（`turn_count` 闸门） | 外面包一层"多子任务"plan 循环（`step_index`/`replan_count` 闸门） |
| 新增节点 | — | `planner`（拆解/推进/re-plan）、`critic`（审单步）、`step_init`（per-subtask 重置） |
| 复用节点 | — | `assemble`（按 role 参数化）、`agent`+`tools`+`inject_*`（=executor 引擎）、收尾三连原样 |
| State | 14 字段 | **+8 字段**：`plan`/`step_index`/`step_results`/`critic_verdict`/`retry_count`/`replan_count`/`plan_version`/`done`·`termination_reason` |
| 重置 | per-query（init 清一次） | **两层**：per-subtask（init+step_init 都清）/ per-task（只 init 清）——E2/E6 |
| 重试 | 传输层 `empty_retries`（v4.2 已有） | **+业务层 `retry_count`**（critic 驱动）**+计划层 `replan_count`**（escalate）——三档分计数器（E5） |
| 产出 | 单答案 | 结构化调研报告（计划 + 分步结论 + 引用汇总 + 终止元信息） |

## 图骨架（设计草稿 §五）

```
START → init → planner ─┬─(done / step≥MAX_STEPS / replan≥MAX_REPLAN)─────────────► finalize
                        └─(下一子任务)─► step_init ─► assemble(executor) ─► agent ─┐
              ┌── retry ─► retry_reset ─► assemble ──────────────────────────────┘
          critic ─┬─ accept ───────────────► planner（判 done / 推进 step_index）
                  ├─ retry(<MAX_RETRY) ────► retry_reset → assemble（轻量重置后重做该步）
                  └─ escalate / retry 用尽 ► planner（skip-and-advance：标记该步 skipped、推进）
   收口：finalize（组装报告）→ human_review（审批）→ update_memory → END
```

executor 引擎（`agent ↔ tools ↔ inject_*`，内层 `turn_count` 闸门）对外循环是黑盒：
入口 `assemble(executor)`、出口 `critic`（v4.2 里到 finalize 的内层出口，v5.0 改接 critic）。

## 设计要点（实现层）

1. **跨子任务窗口隔离（E7）**：`messages` 全图唯一累加、`init` 不清（保留 thread 历史），跨子任务只增不减。
   "executor 只看当前子任务"靠 `assemble(role=executor)` **每子任务重产 `SYSTEM_PROMPT` 锚** +
   `nodes._window_start` 切片实现——窗口自锚到当前子任务、天然隔离前序子任务 tool 历史；不靠清空 messages。
2. **两层重置 + retry 轻量重置（E2/E6 + 实现期定）**：`PER_SUBTASK_DEFAULTS`（含 `retry_count`/`turn_count`/`has_*`/`retrieved_chunks`）
   由 init + `step_init`（新子任务）全量打回；`PER_TASK_DEFAULTS`（`plan`/`step_results`/`replan_count`/`plan_version`）只 init 清。
   业务 retry 走 `critic → retry_reset → assemble`：`retry_reset` 重置 `turn_count`/`has_*`/纠正注入标志，
   **保留** `retry_count`(critic 刚 +1)/`critic_feedback`(重做指导)/`retrieved_chunks`(换措辞重做仍可引用)——
   每次 retry 是带满额 turn 预算的全新尝试（"重做该步"），避免首次跑满 turn 的子任务一被 retry 就因预算耗尽直接 escalate。
3. **三档计数器分开（E5）**：`empty_retries`（传输层，agent 节点内、不进拓扑、per-task 累计）/
   `retry_count`（业务层，critic 驱动、per-subtask）/ `replan_count`（计划层，per-task）——别合成一个。
4. **双闸门非冗余（E4）**：`step_index<MAX_STEPS`（前进步数）+ `replan_count<MAX_REPLAN`（原地绕圈）；
   escalate 恒压下 `replan_count` 闸门先收口。框架 `recursion_limit` 只兜底（E6：默认 10007 远超有界任务）。
5. **critic 比 executor 更严，但引用闸门用"接地比例"软闸门（职责边界 §8 + v0.5 真实跑修正）**：硬闸门——空/占位 → retry、
   跑满 turn 仍无答案 → escalate；**引用**——`_citation_grounding`（能溯源到本步召回的引用占比）< `CITATION_MIN_GROUNDING`(0.5) 才 retry。
   不再要求"全部引用精确命中"（真实跑暴露：模型写富报告引 20–37 条含子节/相关节，全命中是奢望、过严会把合法步全误杀成 retry 级联）。
   section 匹配也容忍缩写/后缀/叶子。store 持久化侧仍由 `extract_fact_candidates(allowed_sources)` 严格白名单兜底（S13/S14）。
6. **escalate = skip-and-advance（v0.5 真实跑修正）**：critic escalate / retry 用尽 → planner 标记该步 skipped、推进 step_index；
   `replan_count` 计跳过数、达 `MAX_REPLAN` 放弃剩余。旧"re-plan 重做同一步"会因同一失败原因反复触发、烧光预算让后续步没机会跑。
7. **`step_results` 不上 reducer（E3）**：节点内手动累加（按 step_id de-dup，retry 重做覆盖旧条），init 才能用 `[]` 清空。
8. **记忆 read-side 保留**：`assemble(executor)` 记忆开启时复用 v3.0 六段装配（偏好/摘要/长期事实召回进每子任务窗口）；
   write-side `update_memory` 记录整份报告（占位符报告跳过）。子任务结论写 store 靠 `plan_version` 打标（§8 承接）。

## 文件结构

```
search_agent/
├── state.py     # AgentState（+8 字段）+ PER_SUBTASK_DEFAULTS / PER_TASK_DEFAULTS（两层重置）+ fresh_retry_reset
├── nodes.py     # init / planner / step_init / retry_reset / assemble(role=executor) / agent / tools /
│                #   inject_*×3 / critic / finalize(报告) / human_review / update_memory
│                #   三档模型调用：call_model(executor) / call_planner_model / call_critic_model（可桩）
├── edges.py     # 外循环：route_after_planner / route_after_critic（E1 两出口 {planner, retry_reset}）；
│                #   内层：route_after_agent / need_correction / after_tools / gate_to_agent（出口改接 critic）
├── graph.py     # 状态图组装（planner-executor-critic 接线；step_init/retry_reset 都喂 assemble）+ compile
├── main.py      # CLI：交互调研 / --query / --test（批量调研）/ --ingest / 记忆模式
├── config.py    # +MAX_STEPS/MAX_REPLAN/MAX_RETRY/CITATION_MIN_GROUNDING + PLANNER_PROMPT/CRITIC_PROMPT；ingest 含 week_7 docs
├── test_graph.py        # v5.0 外循环离线测试（W1–W12，60 项；E1–E7 进真实图 + W11 retry_reset + W12 引用闸门）
├── test_criteria.py     # 占位符/判据纯函数（23 项）
├── test_store_memory.py # Store×记忆 e2e（56 项；S9/S11/S12/S14 已适配 v5.0 外循环）
├── tools.py / checks.py / rag/ / memory/   # v4.2 原样复用（加调研工具只登记 TOOL_EFFECTS）
└── data/        # docs 向量库 + ltm.db（长期记忆）
```

## 跑法

```bash
../../.venv/bin/python test_graph.py            # 外循环图结构测试（离线，无 API，44 项）
../../.venv/bin/python test_criteria.py         # 判据纯函数（离线，23 项）
../../.venv/bin/python test_store_memory.py     # Store×记忆 e2e（离线，56 项）
../../.venv/bin/python main.py --ingest         # 重建向量库（已含 week_7 docs/experiment）
../../.venv/bin/python main.py --query "调研一下 X 和 Y 怎么设计的"   # 单次调研 → 结构化报告
../../.venv/bin/python main.py --query "..." --state                  # 附带打印最终 state
../../.venv/bin/python main.py --test           # 批量调研测试（无记忆，出 test_report.json）
../../.venv/bin/python main.py                  # 交互调研模式（记忆开；/plan 看计划、/state 看摘要）
../../.venv/bin/python main.py --review         # 交互 + human_review 审批
```

## 验证状态

- **离线桩测全绿**：`test_graph.py` 60/60（W1–W12：happy path / 三路 critic / 两层重置 /
  step_results 累加 / 双闸门 / 两档计数器 / 跨子任务重锚 / executor 引擎保真 / turn 闸门到 critic / 收尾链尾 /
  W11 retry 轻量重置 / **W12 引用接地软闸门**）；`test_criteria.py` 23/23；`test_store_memory.py` 56/56。
- **机制探针**：`week_7/experiment/` E1–E7，20/20（LangGraph 1.2.4）。
- **真实 API 端到端实跑**（qwen3.7-plus，5 子任务调研问题，2026-06-09）：planner 拆解合理；critic LLM 正常裁决
  （3 步 accept、引用 12–25 条经接地软闸门放行）；**3/5 步完成**（剩 2 步因 DashScope 免费额度耗尽 403 →
  skip-and-advance 优雅收口出 3 段可读报告，坐实兜底）。**§四 token 预算实测**：摘要段峰值 ~1307 字符（受
  `MAX_STEPS`×每步 400 上限约束、有界，**无需压缩**）；真正窗口压力是子任务内检索 chunk 累加（峰值 ~12.7K tokens）——
  后续可给内层累加 tool 结果设上限/降 `RETRIEVE_TOP_K`。完整额度跑通待非免费额度复跑。
