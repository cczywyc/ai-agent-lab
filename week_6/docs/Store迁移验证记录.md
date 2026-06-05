# Store 迁移验证记录（v4.0 → v4.1：召回不退化 + 持久化生效）

> 对应《Store集成与重构计划》§四 阶段四 / §六 产出。
> 全部数字为 2026-06-05 实跑测得（LangGraph 1.2.4 + langgraph-checkpoint-sqlite 3.1.0，
> text-embedding-v3，qwen3.7-plus），可按文末"复现方法"重跑。

---

## 一、结论

| 验证项 | 结果 |
|---|---|
| 召回不退化（阶段四-14） | ✅ hit@1 **12/12 → 12/12**，hit@3 **12/12 → 12/12** |
| 重启进程数据还在（持久化） | ✅ 新进程读 `ltm.db`：1 偏好 / 11 事实 / 7 主题，与迁移前 json 逐项一致 |
| 端到端段 2 / 段 4 进装配（阶段四-15） | ✅ 实抓模型窗口：六段全在，`facts_recalled=3` |
| 离线测试（阶段四-13） | ✅ `test_store_memory.py` 42 项 + `test_graph.py` 37 项全过 |
| 数据迁移（阶段三-11） | ✅ 1 偏好 / 11 事实 / 7 主题全量进 store，条数比对通过；原 json 保留未删 |

> 计划里写的"17 事实"是笔误，实际 `memory_facts.json` 为 **11 条**（其中 #6/#7 是
> 踩坑 #3 的抽取噪音，按计划原样迁移不处理）。

## 二、召回对照（迁移前手动 embed_query → 迁移后 store 原生 search）

标注查询集：针对现有 11 条事实设计的 12 条中文查询（见 `verify_store_migration.py`
`LABELED_QUERIES`，期望命中任一标注下标即算 hit）。线上同款参数：`top_k=3`、`min_score=0.30`。

| 查询 | 迁移前 top3 | 迁移后 top3 | hit@3 |
|---|---|---|---|
| 连续失败计数器是怎么重置的？ | [3, 0, 1] | [3, 0, 7] | ✓→✓ |
| 工具连续失败之后什么时候触发降级？ | [1, 4, 0] | [4, 1, 0] | ✓→✓ |
| 工具返回的结构化错误元数据有哪些字段？ | [2, 5, 9] | [2, 5, 8] | ✓→✓ |
| recoverable 和 suggestion 字段是干什么用的？ | [5, 2, 9] | [5, 2, 10] | ✓→✓ |
| 文档 chunking 是按什么切分的？ | [8, 9, 10] | [8, 9, 10] | ✓→✓ |
| 每个 chunk 需要携带哪些元数据？ | [9, 8, 10] | [9, 10, 8] | ✓→✓ |
| 为什么没有采用纯语义分块？ | [10, 8, 2] | [10, 8, 5] | ✓→✓ |
| fetch_webpage 连续 403 失败时 Agent 怎么办？ | [1, 4, 0] | [1, 4, 0] | ✓→✓ |
| consecutive_errors 达到阈值会发生什么？ | [4, 3, 0] | [4, 0, 3] | ✓→✓ |
| Markdown 标题切分策略是怎么设计的？ | [8, 10, 1] | [8, 10, 9] | ✓→✓ |
| 成功即重置是什么策略？ | [3, 0, 1] | [3, 0, 7] | ✓→✓ |
| 降级之后 Agent 基于什么内容回答？ | [1, 5, 4] | [4, 1, 5] | ✓→✓ |

**hit@1: 12/12 → 12/12；hit@3: 12/12 → 12/12，无退化。**

top3 顺序的局部差异来自事实**重新 embed** 的数值噪声：#0/#3、#1/#4 是近重复事实
（demo 两轮抽到了同一组结论），分数本就咬得很紧（差值 < 0.02），重嵌入后排序在
期望集内互换，hit 不受影响。明细见 `verify_ltm_baseline.json` / `verify_ltm_after.json`。

## 三、持久化验证

迁移（进程 A）→ 退出 → 全新进程 B 只读：

```
重启进程后: 1 偏好 / 11 事实 / 7 主题
偏好内容: {'以后回答涉及本地文档时，先列结论再列引用': ...}
主题计数: [('agent_loop', 4), ('tool_calling', 3), ('fallback', 3), ('rag', 3),
          ('chunking', 2), ('memory', 1), ('citation', 1)]
```

与迁移前 `memory_topics.json` / `memory_preferences.json` 逐项一致；
语义检索在重开连接后直接可用（向量随 `ltm.db` 持久化，无需重建索引）。

## 四、端到端（use_memory=True，真实模型）

问题："我们之前聊过的连续失败计数器是怎么重置的？"（探针包装 `call_model` 实抓窗口）：

- **第 1 次调用窗口**＝完整六段：SYSTEM_PROMPT(1359c) + `[用户偏好]`(51c) +
  `[历史对话摘要]`(55c) + `[相关长期事实]`(516c，召回 3 条) + 最近 3 轮 + 当前问题。
  模型**凭段 4 的长期事实直接答对**（未调工具）——长期记忆在起作用。
- 纠正规则（"我们之前"→应查本地）注入检索纠正 → 第 2 次调用发起
  `retrieve_documents` → 第 3 次调用给出带 `[doc#section]` 引用的最终回答。3 轮收口。
- `assembly_report`: `segments_present=['system','preferences','summary','facts','recent','current']`，
  `facts_recalled=3`，`total_chars=3916`，无裁剪。

## 五、过程中发现并修复的两个问题（计划外，但有测试钉住）

1. **v4.0 既有 bug：装配窗口切掉段 1-3**。`nodes._window_start` 取"最后一条 system
   消息"做窗口起点，但记忆开启时 assemble 产出多条 system（段 1/2/3/4），起点落在
   段 4 上——**发给模型的窗口丢失 SYSTEM_PROMPT、偏好、摘要**。v4.0 的 T1-T8 全部
   `use_memory=False`，从未暴露。修复：锚点改为"最后一条内容等于 SYSTEM_PROMPT 的
   system 消息"。回归测试：`test_store_memory.py::S10`。
2. **空回答污染记忆**。qwen 偶发返回空 content（v4.0 `finalize` 已有
   `[模型返回空回答]` 占位符，属已知模型抖动）；但 `update_memory` 会把
   `assistant=""` 的轮次写进短期记忆，下一问的段 5 带着空轮次会诱发模型答非所问
   （实测复现一次，根因定位见调试记录）。修复：空回答跳过本轮记忆更新。
   回归测试：`test_store_memory.py::S11`。

## 六、迁移后的代码事实

- 长期记忆对 `rag.embed` 的**两个手动 call site 消失**（写侧 `embed_texts`、读侧
  `embed_query`），`grep -rn "embed_query\|embed_texts" memory/` 仅剩
  `ltm_store.embed_for_store`（index 适配器——E2c"共用 embedding 模型"保留的那一半）。
- `Fact`↔向量"下标对齐"的手工同步消失（store 内部管 key↔向量，key=sha1(原文)，迁移幂等）。
- `rag/store.py` 的 `memory_facts` namespace 退役（旧 `.npy/.json` 文件保留在原地未删，
  对照与回滚用）；rag 向量库只剩文档 chunk。
- prefs/topics 写入带 `index=False`——语义索引只给 facts，主题计数不再有任何 embed 开销
  （离线测试 S6 钉住）。
- `store.batch(PutOp×N)` 把一批事实合并为 1 次 embed 调用（与 v4.0 手动批量等价）。
- SqliteStore 连接坑：必须 `isolation_level=None`（autocommit），否则原生 sqlite3 的
  隐式事务与 store 内部 `BEGIN` 冲突（`from_conn_string` 即如此配置，但它是
  contextmanager，进程级长连接要自建）。

## 七、复现方法

```bash
cd week_6/search_agent
../../.venv/bin/python test_store_memory.py            # 离线 42 项（stub embed，零 API）
../../.venv/bin/python test_graph.py                   # 离线 37 项（图接线回归）
../../.venv/bin/python migrate_ltm.py                  # 幂等迁移 + 条数比对
../../.venv/bin/python verify_store_migration.py --after    # 迁移后召回（真实 embed）
../../.venv/bin/python verify_store_migration.py --compare  # 与基线对照
# 基线（verify_ltm_baseline.json）依赖 v4.0 旧代码路径，已于重构前采集存档，不可重跑
```
