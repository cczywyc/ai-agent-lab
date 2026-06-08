# 第七周 周三验证实验 E1–E5

用最小桩节点（不调模型、不碰 RAG，只返回固定 dict）验证 [v0.1 设计草稿](../docs/第七周设计草稿.md) 里"外循环"机制假设。
本周机制大多是第六周已验证机制的**降一层复用**（per-query → per-subtask、turn_count 闸门 → step/replan 闸门、chunks 累加 → step_results 累加），所以实验目标是**坐实"降一层后仍成立"**，外加几处精确化。
所有探针已在 **LangGraph 1.2.4** 上实跑通过。

## 环境

```bash
pip install -U langgraph        # 实测 1.2.4 / langgraph-checkpoint 4.1.1 / Python 3.12.3
```

确认的当前 import 路径（与第六周一致）：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.errors import GraphRecursionError
from functools import partial
from operator import add
```

## 跑法

```bash
python e1_critic_routing.py
python e2_two_layer_reset.py
python e3_step_results_accumulate.py
python e4_planner_loop_guard.py
python e5_retry_counters_split.py
```

## 结果一览（全部 ✓，13/13 判据）

| 实验 | 验证的决策 | 结论 |
|---|---|---|
| **E1** critic 三路条件边分发 | 设计② / 决策 A、B | ✓ 单条边按 verdict 三路分发；**三路 verdict 塌缩成两条物理边**（accept 与 escalate/retry-达上限 同落 planner） |
| **E2** 两层重置并存（per-subtask × per-task） | 设计① | ✓ 内层标志每步清零、外层 step_results 跨步累加；对照组确实泄漏（**降一层**于第六周 per-query 重置） |
| **E3** step_results 手动累加 vs reducer | 决策 G | ✓ 手动累加可清空；`operator.add` 字段 `return []` 清不掉——**降一层重演**第六周 chunks 实证 |
| **E4** 外循环双闸门 | 设计② / 决策 D、E | ✓ 双闸门在 `recursion_limit` 前收口；**escalate 恒压下 replan_count 闸门先收口**，两闸门各兜一维度 |
| **E5** 两档计数器互不串扰 | 设计③ / 决策 B | ✓ `empty_retries`（传输层、不进拓扑）与 `retry_count`（业务层、走 retry 边）各记各的、互不吃额度 |

## 几处精确化（建议并入 v0.2）

1. **E1**：`route_after_critic` 逻辑上三路、物理上两条边——`add_conditional_edges` 的 mapping 只需 `{to_planner, to_executor}` 两个出口，别写成三出口（accept、escalate、retry-达上限 三种结果里后两者同回 planner）。
2. **E4**：外层双闸门**非冗余**——`step_index < MAX_STEPS` 兜"前进步数"、`replan_count < MAX_REPLAN` 兜"原地绕圈"，是两个正交维度。实测在 critic 恒 escalate 的压力下，`replan_count` 闸门先收口（停在 `step_index=2 < 4`、`replan_count=2 = MAX_REPLAN`），坐实 `MAX_REPLAN=2`（决策 D）是 re-plan 环的有效断路器。
3. **E5**：`critic_verdict` 等 critic↔route 交互字段**必须**写进 State schema，否则 critic 的更新会被框架静默丢弃、条件边读不到——延续第六周"schema 外字段被丢弃"的纪律。

> 与第六周不同：第六周有一处真正的"意外发现"（TypedDict 无隐式默认值）。本周没有推翻设计假设的意外——E2 对照组泄漏值是 `[False, True]` 而非第六周的 `[None, True]`，正因为第六周那条修复（`init` 首轮建默认值）已带到 per-subtask 层，`None` 问题不复发。本周的产出是"降一层后机制照旧成立"的实证 + 上面三处精确化。
