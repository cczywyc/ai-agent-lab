# 第六周 周三验证实验 E1–E5

用最小桩节点（不调模型、不碰 RAG）验证 v0.2 设计草稿里的框架机制假设。
所有探针已在 **LangGraph 1.2.4** 上实跑通过。

## 环境

```bash
pip install -U langgraph        # 实测 1.2.4 / langgraph-checkpoint 4.1.1
```

确认的当前 import 路径：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphRecursionError
from operator import add
```

## 跑法

```bash
python e1_state_passing.py
python e2_init_reset.py
python e3_chunks_accumulate.py
python e4_cycle_guard.py
python e5_interrupt_toggle.py
```

## 结果一览（全部 ✓）

| 实验 | 验证的决策 | 结论 |
|---|---|---|
| **E1** 状态传递 + add_messages 累加 | 设计① | ✓ 替换字段跨节点可见；messages 累加非覆盖 |
| **E2** init 重置 × checkpointer 持久化 | 决策 C | ✓ messages 跨轮持久化、标志被 init 重置；**意外发现**见下 |
| **E3** chunks 轮内累加 + 跨问题清空 | 决策 B | ✓ 手动累加可行；operator.add 字段 `return []` 清不掉（实证草稿推理） |
| **E4** cycle × turn_count 闸门 | 设计② | ✓ 闸门在框架 `recursion_limit` 前先收口；无闸门确实抛 `GraphRecursionError` |
| **E5** interrupt 开关透明性 | 决策 F | ✓ 关→透明；开→在 human_review 暂停、`Command(resume=)` 跑完 |

## 唯一意外发现（E2，建议进 v0.3）

TypedDict 的字段**没有隐式默认值**：无 init 的图第一轮里 `has_retrieved` 是 `None`（裸取 `state["has_retrieved"]` 会 KeyError），不是 `False`。
→ 所以 `init` 节点不止"跨轮重置"，还负责"**首轮建立默认值**"——比草稿假设更必要。
建议在 v0.3 的 §1/§2 把 init 的职责补一句"为所有 per-query 字段建立首轮默认值"。
