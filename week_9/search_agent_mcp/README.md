# 第九周 v7.0 — researcher 工具层接 MCP（第一个 MCP 工具服务）

把 v6.0（`week_8/search_agent`）的两个工具**包一层**暴露成 MCP 工具契约，researcher 内层经
**真实 MCP client + stdio** 调用它们——不重写工具逻辑、不动 supervisor 拓扑（决策 A/G，blast
radius 最小）。这是周三《第九周实验结论》桩测的真实化：把进程内忠实桩换成真实 FastMCP server +
stdio + 客户端站。

## 文件

| 文件 | 角色 | 对应决策 |
|---|---|---|
| `mcp_server.py` | FastMCP server，暴露 `local_search`（包 `retrieve_documents`）+ `http_fetch`（包 `fetch_webpage`）；`python mcp_server.py` 起 stdio server | A 包一层 / B 花名册 / C I/O 契约 / D stdio |
| `mcp_station.py` | MCP client 站：discover（`tools/list`）+ **派发前守卫** + **client 侧 schema 校验** + **两层错误分流**（Routing） | E 守卫迁移 / F 两层分流 / G client 插点 |
| `researcher_loop.py` | researcher 内层最小循环（`agent↔tools↔inject` 闸门原样，tools 环换接 MCP client） | G 复用边界 |
| `verify_thursday.py` | 端到端验证：真实 SDK 形态观测 + 站行为 + loop 无回归（**11/11**） | 坐实草稿 v0.2 的 [周四确认] |

## 数据流

```
researcher_loop (turn_count / empty_retries / synthesis-reserve 闸门)
      │  await station.call(name, args)
      ▼
MCPToolsStation ──①派发前守卫: name ∈ tools/list?  ──否→ ESCALATE（不发 call）
      │           ──②client 侧 inputSchema 校验    ──违→ ESCALATE（不发 call）
      │  tools/call (JSON-RPC)
      ▼  ⎯⎯ 进程边界 · stdio ⎯⎯
FastMCP server ── local_search → retrieve_documents（本地 RAG，需 embedding API）
                └ http_fetch   → fetch_webpage（HTTP 抓取）
      │  CallToolResult
      ▼
③两层分流: isError=True → INNER_RETRY（业务失败）/ 空 → INNER_RETRY / 非空 → SUCCESS（inject）
```

## 周四实测发现（发现型 → 折回设计草稿 v0.3）

草稿 v0.2 §五假设"协议层 error 走**异常**、执行层 isError 走**结果字段**，client 端 `isinstance`
干净二分"。**真实 FastMCP 证伪了这条**：

- 未知工具名 / 参数违 schema / 业务失败 **全部**回成 `isError:true` 的结果字段，client
  `call_tool` **不为前两者抛 JSON-RPC 异常**——协议层与执行层在 SDK 层**塌成一层**。
- 故两层分流的"协议层/集成问题"那一支，改由 **host 在派发前自己重建**：名字守卫 + client 侧
  `inputSchema` 校验，在 `call_tool` 之前拦掉幻觉名和错参数、判 ESCALATE；剩下的 post-dispatch
  `isError` 才是真业务失败、判 INNER_RETRY。
- 结论没变、判据挪位：决策 F 的两层路由**仍成立**，但二分判据从"SDK 失败形态"挪到"host 派发前
  名字+schema 校验"。这反而让**决策 E 的 host 守卫从'可选优化'升成'分流的承重墙'**——印证本周主轴
  "框架给的失败词汇表更弱（只一层），两层控制流仍归我自己兜"。

## 跑法

```bash
cd week_9/search_agent_mcp
../../.venv/bin/python verify_thursday.py     # 端到端验证 11/11（A 形态观测 / B 站行为 / C 无回归）
../../.venv/bin/python mcp_server.py           # 单独起 stdio server（供任意 MCP host 连）
```

依赖：`mcp==1.28.0`、`jsonschema==4.26.0`（已并入根 `requirements.txt`）。`local_search` 真检索需
`.env` 里 `DASHSCOPE_API_KEY`（embedding）+ 已构建的本地向量库（`week_8/search_agent/data`）。

## scope 诚实（留周五真实跑）

- 本周裁到 **researcher 内层 + MCP server**（决策 G / 草稿 §六），驱动用**确定性脚本**（零 LLM API），
  隔离验 MCP 接线；**真实模型驱动 researcher loop** 是周五真实跑的事。
- **未做**：把 `nodes.py` 的 `tools()` 节点（同步 LangGraph 节点）改成调异步 MCP client、搬进完整
  supervisor 图端到端跑——async-in-LangGraph 是更大一刀，与周五真实跑合并做。
- escalate 后半段（协议层 → supervisor `skip-and-advance`/`replan_count+1`）本周只验到"内层判
  ESCALATE 并退出内层"；supervisor 端复证同上，留周五端到端。
