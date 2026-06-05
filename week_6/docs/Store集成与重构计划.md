# Store 集成与重构计划（v4.0 → v4.1：长期记忆迁移到 LangGraph Store）

> 落地决策 E（之前推迟的那条）：把长期记忆从自研 `memory/long_term.py` + 共用向量库
> 迁到 LangGraph 的 `Store`。checkpointer 管 thread 内短期状态，store 管跨 thread 长期记忆——
> 这是把长期记忆**正式交给框架**的一步。
> 基线：v4.0（已读你仓库 week_6 实际代码）。本文是计划，不含完整实现。

---

## 〇、范围

**迁**：长期记忆三类——偏好 / 已确认事实 / 主题——的存储、召回、持久化。

**不动**：控制流图（10 节点 + 条件边）、checkpointer、短期记忆（`short_term.py`，仍是 thread 内 + 装配裁剪）、摘要（`summarizer.py`）、抽取（`extractor.py`）、装配的"挑选"逻辑（`assembler.py` 的段顺序/预算/裁剪）、整个 `rag/`（文档检索仍是工具）。

---

## 一、现状盘点（迁移基线，读自你的代码）

### 数据模型（`long_term.py`）
| 类型 | 结构 | 持久化 | 进装配 |
|---|---|---|---|
| 偏好 | `dict[str,str]` | `memory_preferences.json` | 段 2，全量 |
| 已确认事实 | `list[Fact{fact,source,turn,timestamp}]` + 向量库 `namespace="memory_facts"`（**下标对齐**：metadata 第 i 项 ↔ 向量第 i 行） | `memory_facts.json` + 共用向量库 `.npy` | 段 4，语义 top-k |
| 主题 | `dict[str,int]` 计数 | `memory_topics.json` | **未进装配**（"召回加权暂未实装"） |

### 读写 call sites（要改的就这几处）
- **写**（`manager.update_from_turn`）：`set_preference` / `_extract_and_store_facts`（`embed_texts` 批量嵌入候选 → `long.add_fact(f, vector)`）/ `bump_topics` → `long.save()`（3 json + 向量库）。
- **读**（`assembler.py`）：`_format_preferences`（读 `long.preferences`）/ `_recall_facts`（`embed_query(query)` → `long.recall_facts(q_vec)`）。**topics 不读**。
- **嵌入依赖**：召回侧 `embed_query`、写入侧 `embed_texts`，都来自 `rag.embed`。
- **持久化**：`MEMORY_DIR`（= 向量库同目录）下 3 个 json + 向量库；`MemoryManager` 单例 autoload。

> 关键观察：长期记忆对外其实只有"两读三写"，迁移面比想象小。难点不在改 call site，在**选哪种 Store + 数据迁移**。

---

## 二、目标架构（Store 映射）

三个 namespace（tuple）：
- `("ltm", "preferences")` — 偏好，纯 kv，`put/get/list`，无需语义索引
- `("ltm", "facts")` — 事实，**配语义索引**
- `("ltm", "topics")` — 主题，纯 kv 计数，`get`+1+`put`

语义索引只给 facts：`index={"embed": <包装的 text-embedding-v3>, "dims": 1024, "fields": ["fact"]}`。

**最大收益**：native `search(namespace, query="...", limit=k)` 内部自己 embed。
→ 当前手动的 `embed_query`（读）和 `embed_texts`（写）这两个长期记忆 call site **消失**，长期记忆不再显式调 `rag.embed`（但仍复用同一个嵌入模型——E2c 的"共用 embedding 模型"那半保留，让位的只是"共用向量库")。
→ `Fact` 与向量"下标对齐"这套易错的手工同步也消失（store 内部管 key↔向量）。

---

## 三、关键岔路：选哪种 Store（按新证据更新 v0.2 的 B/C）

v0.2 当时把 C"原生 Store"判为"要拆 E2c + 只有内存版"。新证据推翻了这个前提：**SqliteStore 提供本地文件级持久化 + 语义搜索**（配 embed 即可），不需要 Postgres。所以重判：

| 方案 | 做法 | 持久化 | E2c | 工作量 | 选 |
|---|---|---|---|---|---|
| **C-SqliteStore** | 三类进 SqliteStore，facts 配语义 index（embed=你的 v3） | ✅ 本地 `.db` | 失"共用向量库"、留"共用 embedding" | 中 | ✅ **推荐** |
| C-InMemoryStore | 同上但内存版 | ❌ 重启即失 | 同上 | 小 | 仅开发/测试 |
| B-包一层 BaseStore | 自研 numpy 库包成 `BaseStore` 子类 | ✅ 复用现有 `.npy`/json | ✅ 完整保留 E2c | 大（要实现 batch/search 等抽象方法） | 备选 |

**推荐 C-SqliteStore**，理由贴你的目标：
1. 决策 E 的初衷就是**体验真实的 LangGraph Store**，C 学到的是标准 API；B 大半精力花在实现 `BaseStore` 抽象方法（plumbing），反而绕过了 Store 本身。
2. SqliteStore 把唯一的硬伤（InMemoryStore 重启即失）解决了，本地持久化白拿。
3. facts 的向量存储从"和文档共库"挪进 store 自管——E2c 的"共用向量库"让位，但"共用 embedding 模型"保留；而且**概念上更干净**：文档是 RAG 语料（工具），事实是长期记忆（Store），本就该分家。
4. 副作用是简化：`rag/store.py` 的 `memory_facts` namespace 退役，rag 库只剩文档 chunk。

落地节奏：**开发期先用 `InMemoryStore` 把接口和节点跑通（可离线测），再换 `SqliteStore` 落持久化**。若你更看重保住 E2c 的共用向量库，退而求其次走 B。

---

## 四、工作分解（按依赖排序）

### 阶段一 — 接口跑通（InMemoryStore，可离线测）
1. **embed 适配**：把 `rag.embed` 包成 LangGraph index 要的 `(list[str]) -> list[list[float]]`（或 Embeddings 对象），`dims=1024`。你的向量已 L2 归一化，store 的余弦搜索直接可用。
2. **`graph.py`**：`build_graph` 增加 `store` 参数，`compile(checkpointer=..., store=...)`；开发期 `InMemoryStore(index=...)`。
3. **节点访问 store**：在 `assemble` / `update_memory` 节点签名加 `store: BaseStore`（LangGraph 注入），或用 `get_store()`。
4. **`update_memory` 改写**：
   - 偏好 → `store.put(("ltm","preferences"), key, {"value": v})`
   - 事实 → `store.put(("ltm","facts"), key, {"fact":..., "source":..., "turn":...})`（**不再手动 embed**）
   - 主题 → `get` 现值 +1 后 `put`（或 list 后聚合）
5. **`assemble` 改写**：
   - 段 2 偏好 → `store.search/list(("ltm","preferences"))` 全量
   - 段 4 事实 → `store.search(("ltm","facts"), query=user_message, limit=MEMORY_FACTS_TOP_K)`，**不再传 query_vector**
6. **min_score 过滤**：native search 返回 `score`，把你的 `MEMORY_FACTS_MIN_SCORE=0.30` 阈值留在节点侧过滤（与现状语义一致）。

### 阶段二 — 重构 `memory/`（剥离已迁走的职责）
7. **`long_term.py`**：facts/prefs/topics 的"存储 + 召回 + 持久化"整体由 Store 接管 → 该文件大幅瘦身或退役（`Fact` 数据类可保留为 dict schema 约定）。
8. **`manager.py`**：`assemble_context` / `update_from_turn` 的长期记忆部分改成调 store；删除长期记忆对 `embed_query`/`embed_texts` 的用法（`rag.embed` 仍被 rag 文档检索用，不删）。短期 + 摘要 + extractor **不动**。注意：`assembler` 不再依赖 `LongTermMemory` 对象，构造参数要相应调整（`embed_query_fn` 依赖移除）。
9. **`assembler.py`**：`_format_preferences` / `_recall_facts` 改成消费 store 查询结果，**段顺序 / 预算 / 裁剪这套"挑选"核心一字不动**。

### 阶段三 — 持久化 + 数据迁移
10. **换后端**：`SqliteStore.from_conn_string(str(MEMORY_DIR/"ltm.db"))`（安装并核对包名/import：`langgraph-checkpoint-sqlite`）。`compile(checkpointer=..., store=...)`。
11. **一次性迁移脚本**：把现有 `memory_preferences.json` / `memory_facts.json` / `memory_topics.json` 的数据 `put` 进 store（facts 会被 store 重新 embed 建索引）。**别丢现有数据**（你 demo 攒的 1 偏好 / 17 事实 / 7 主题）。迁移后比对条数。
12. **明确 checkpointer / store 边界**：长期走 store；短期对话历史维持现状（`memory_short_term.json` + 图 messages），本计划不动它。

### 阶段四 — 验证（延续 E 系列）
13. **离线**（InMemoryStore + stub embed）：`put`→`search` 召回顺序对、min_score 过滤对、namespace 隔离对、prefs/topics 的 get/put 对。
14. **真实**（SqliteStore + text-embedding-v3）：复用 E2c 那 15 条标注查询，对照迁移前 `recall_facts` 的召回，确认 **hit 不退化**；**重启进程后数据还在**（持久化验证）。
15. **端到端**：`use_memory=True` 跑一轮，确认段 2 / 段 4 仍正确进装配。

---

## 五、风险与边界（重要）

- **踩坑 #3 不被 Store 解决也不被恶化**：事实抽取仍依赖 `[doc#section]` 引用格式，这是 `extractor` 侧问题，Store 不碰。事实越攒越多，这个契约要在第七周前收紧（呼应 week_4&5 复盘留的待验风险）。
- **短期记忆别塞进 Store**：那是 checkpointer / thread 内的事。经典错配——偏好存进 state 换 thread 就没了，短期塞 store 跨 thread 串味。本计划严格只迁三类长期记忆。
- **"挑选"常量留在你这侧**：`MEMORY_FACTS_TOP_K`、`MIN_SCORE`、段预算等留在 assembler/节点，不交给 store——这是你一直想保留的核心。
- **已知冗余（本计划不处理，记一笔）**：短期历史目前同时存在于 `memory_short_term.json` 和图的 messages 审计日志里。把短期完全交给 checkpointer 是另一条独立的重构线，留作后续。

---

## 六、产出

- **v4.1**：长期记忆走 LangGraph Store（SqliteStore），`memory/long_term.py` 瘦身或退役，两个 embed call site 消失。
- **设计草稿 v0.4**：B/C 岔路按 SqliteStore 新证据更新为"选 C"。
- 一份"召回不退化 + 持久化生效"的验证记录。

---

*计划基于 v4.0 实际代码；建议先 InMemoryStore 跑通接口，再 SqliteStore 落地，最后迁数据并验证召回不退化。*
