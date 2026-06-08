# 第七周验证实验 E1–E5（周三）+ E6–E7（补充探针）

用最小桩节点（不调模型、不碰 RAG，只返回固定 dict）验证 [v0.2 设计草稿](../docs/第七周设计草稿.md) 里"外循环"机制假设。
本周机制大多是第六周已验证机制的**降一层复用**（per-query → per-subtask、turn_count 闸门 → step/replan 闸门、chunks 累加 → step_results 累加），所以实验目标是**坐实"降一层后仍成立"**，外加几处精确化。
**E1–E5** 是单机制隔离验；**E6–E7** 是补充探针，补两件 E1–E5 照不到的事：E6 验内外**双循环嵌套组合**、E7 验**跨子任务 messages 重锚**（executor 窗口隔离）。
所有探针已在 **LangGraph 1.2.4** 上实跑通过（E1–E5 首跑 Python 3.12.3；E6–E7 与 E1–E5 复跑 Python 3.13.9，**20/20 一致通过**）。

## 环境

```bash
pip install -U langgraph        # 实测 1.2.4 / langgraph-checkpoint 4.1.1（E1–E5 首跑 Python 3.12.3，E6–E7 与复跑 Python 3.13.9）
```

确认的当前 import 路径（与第六周一致）：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.errors import GraphRecursionError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage  # E7 用真实消息对象
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
python e6_nested_loops_probe.py   # 补充探针：嵌套双循环
python e7_message_reanchor.py     # 补充探针：messages 重锚
```

## 结果一览（全部 ✓，20/20 判据）

| 实验 | 验证的决策 | 结论 |
|---|---|---|
| **E1** critic 三路条件边分发 | 设计② / 决策 A、B | ✓ 单条边按 verdict 三路分发；**三路 verdict 塌缩成两条物理边**（accept 与 escalate/retry-达上限 同落 planner） |
| **E2** 两层重置并存（per-subtask × per-task） | 设计① | ✓ 内层标志每步清零、外层 step_results 跨步累加；对照组确实泄漏（**降一层**于第六周 per-query 重置） |
| **E3** step_results 手动累加 vs reducer | 决策 G | ✓ 手动累加可清空；`operator.add` 字段 `return []` 清不掉——**降一层重演**第六周 chunks 实证 |
| **E4** 外循环双闸门 | 设计② / 决策 D、E | ✓ 双闸门在 `recursion_limit` 前收口；**escalate 恒压下 replan_count 闸门先收口**，两闸门各兜一维度 |
| **E5** 两档计数器互不串扰 | 设计③ / 决策 B | ✓ `empty_retries`（传输层、不进拓扑）与 `retry_count`（业务层、走 retry 边）各记各的、互不吃额度 |
| **E6** 嵌套双循环 + recursion_limit 预算（补充） | 设计①② / 决策 D | ✓ step 转移归零内层 `turn_count`、内外闸门嵌套互不误伤；**假设证伪**——默认 `recursion_limit=10007`（非 25、非脚本注释的 250），有界任务永不触底兜底 |
| **E7** 跨子任务 messages 重锚（补充） | 设计① / 草稿 §四 | ✓ `assemble(role=executor)` 每子任务重产 `SYSTEM_PROMPT` 锚 + `_window_start` 切片即天然隔离；对照不重锚 → 末子任务窗口串台前面 tool 历史 |

## 几处精确化（E1–E5 已并入 v0.2；E6–E7 待并入）

1. **E1**：`route_after_critic` 逻辑上三路、物理上两条边——`add_conditional_edges` 的 mapping 只需 `{to_planner, to_executor}` 两个出口，别写成三出口（accept、escalate、retry-达上限 三种结果里后两者同回 planner）。
2. **E4**：外层双闸门**非冗余**——`step_index < MAX_STEPS` 兜"前进步数"、`replan_count < MAX_REPLAN` 兜"原地绕圈"，是两个正交维度。实测在 critic 恒 escalate 的压力下，`replan_count` 闸门先收口（停在 `step_index=2 < 4`、`replan_count=2 = MAX_REPLAN`），坐实 `MAX_REPLAN=2`（决策 D）是 re-plan 环的有效断路器。
3. **E5**：`critic_verdict` 等 critic↔route 交互字段**必须**写进 State schema，否则 critic 的更新会被框架静默丢弃、条件边读不到——延续第六周"schema 外字段被丢弃"的纪律。
4. **E6（新增）**：默认 `recursion_limit` 实值是 **10007**（源码 `langgraph._internal._config.DEFAULT_RECURSION_LIMIT`，自环实测恰在此触顶），不传 / 空 config 都走它，只有显式传值才按所传值触顶。故"正常嵌套会撞默认 limit"被证伪——有界任务（几十 super-step）距 10007 差三个数量级，收口永远由显式 `MAX_STEPS`/`MAX_REPLAN`/`MAX_TURNS` 闸门完成，**v5.0 无需显式调高**。（纠正 E6 脚本注释"250 仍不触顶"的旧猜测。）
5. **E7（新增，属文档缺口）**：草稿 §四 现只讲 token 预算压缩，**漏写了**"executor 看局部"赖以成立的隔离机制——`assemble(role=executor)` 每子任务重产 `SYSTEM_PROMPT` 锚、executor 窗口靠 `_window_start` 自锚到本子任务。§四必须补写这条，否则实现让 `assemble` 只追加 Human 不重锚 → 后一子任务串台前面全部 tool 历史（对照组实测）。

> 与第六周不同：第六周有一处真正的"意外发现"（TypedDict 无隐式默认值）。本周设计假设无被推翻——E2 对照组泄漏值是 `[False, True]` 而非第六周的 `[None, True]`，正因为第六周那条修复（`init` 首轮建默认值）已带到 per-subtask 层，`None` 问题不复发。本周产出：(1) E1–E5"降一层后机制照旧成立"的实证 + 三处精确化；(2) E6 把一处**框架假设**证伪（默认 limit=10007 而非会撞限）；(3) E7 找到一处**文档缺口**（§四 漏写重锚隔离机制）。详见 [实验结论](实验结论.md)。
