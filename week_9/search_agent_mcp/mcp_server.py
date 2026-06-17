"""
第九周 v7.0 · MCP server（决策 A 包一层 / B 工具花名册 / C I/O 契约 / D stdio）。

把 v6.0（week_8/search_agent）的两个工具**包一层**暴露成 MCP 工具契约，不重写工具逻辑：
  - local_search ← 包 retrieve_documents（本地 RAG 检索）
  - http_fetch   ← 包 fetch_webpage（HTTP 抓取）

用官方 Python SDK 的 FastMCP（决策 A′：SDK 兜协议/JSON-RPC，精力放在工具描述+schema 上）。
传输 stdio（决策 D）：`python mcp_server.py` 即起一个 stdio server。

两层错误（决策 F）在 server 侧的落点：
  - 协议层 error（未知工具名 / 参数违 schema / server 崩）——**交给 FastMCP/SDK 自动报**，
    本文件的工具体不手写这层。
  - 执行层业务失败（检索失败 / 抓取 403 超时 / 库未就绪）——工具体里 `raise`，FastMCP
    会把它装成 `isError:true` 的 CallToolResult（业务失败、模型可据此 retry）。
  - 空结果（查无命中）——**正常返回**（`isError:false`、chunks=[]），不是失败（决策 C）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 包一层：把 v6.0 的 search_agent 放上 sys.path，直接复用其工具函数（不复制、不重写）。
_V6 = Path(__file__).resolve().parents[2] / "week_8" / "search_agent"
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

from mcp.server.fastmcp import FastMCP

from tools import retrieve_documents, fetch_webpage  # v6.0 原样函数

mcp = FastMCP("search-agent-tools")


@mcp.tool()
def local_search(query: str, top_k: int = 5) -> dict:
    """查本项目/本地库内容时用（用户自己的笔记、设计文档、周复盘）；不抓外网。

    返回 {chunks: [{text, citation}]}；citation 形如 [doc#section]，可直接引用。
    查无命中是正常业务结果（chunks=[]、非错误），不要当失败上报。
    """
    result = retrieve_documents(query, top_k)
    if result.get("error"):
        # 执行层业务失败（RetrievalError/VectorStoreNotReady/embedding API 不可达）
        # → raise，FastMCP 标 isError:true，researcher 内层据此 retry（决策 F）。
        raise RuntimeError(f"{result.get('error_type')}: {result.get('message')}")
    chunks = [
        {"text": r.get("text", ""), "citation": f"[{r.get('doc')}#{r.get('section')}]"}
        for r in result.get("results", [])
    ]
    return {"chunks": chunks}   # 空命中 → chunks=[]，isError:false（决策 C）


@mcp.tool()
def http_fetch(url: str, max_chars: int = 3000) -> dict:
    """查外部公开网页时用；不查本地库。

    返回 {content, status}。抓取失败（403/超时/黑名单域）是业务失败（isError:true），
    researcher 内层据此 retry 或换源（决策 F）。
    """
    result = fetch_webpage(url)
    if result.get("error"):
        # 执行层业务失败（403/超时/EmptyContent/BlockedDomain）→ raise → isError:true。
        raise RuntimeError(f"{result.get('error_type')}: {result.get('message')}")
    return {"content": result.get("content", "")[:max_chars], "status": 200}


if __name__ == "__main__":
    mcp.run()   # 默认 stdio 传输（决策 D）
