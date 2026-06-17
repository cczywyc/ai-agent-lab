"""
第九周 v7.0 · MCP client 站（决策 E 守卫迁移 / F 两层错误分流 / G client 插点）。

这是周三 mcp_stub.py 的【真实化】：把进程内忠实桩换成"经真实 MCP SDK + stdio 连真实
server"的 client 站。researcher 内层 `tools` 这一环（v6.0 调 execute_tool）改成调本站的
`call()`——agent 循环 / inject_* / 闸门全不动（决策 G 包一层）。

本站把"我自己掌控的控制流"落到真实 SDK 上：
  - 派发前守卫（决策 E）：名字 ∈ 发现来的 tools/list 清单？不在就本地拒、不发 tools/call。
  - 两层错误分流（决策 F）：协议层 error（SDK 抛异常）→ ESCALATE；执行层 isError（结果字段）
    → INNER_RETRY；空结果 → INNER_RETRY；成功 → SUCCESS。
  ——"看到哪类信号走哪条恢复路径"是我的控制流；MCP 只负责把失败报成 typed 信号。

[周四实测·发现型 → v0.3]：真实 FastMCP **把两层错误塌成一层**——未知工具名 / 参数违 schema /
业务失败【全部】回成 `isError:true` 的结果字段，client `call_tool` 不为前两者抛 JSON-RPC 异常
（证伪了周三草稿 §五"协议 error 走异常 / isError 走字段、isinstance 干净二分"的假设）。
→ 故两层分流的"协议层/集成问题"这一支，改由 **host 在派发前自己重建**：名字守卫 + client 侧
inputSchema 校验，在 call_tool 之前拦掉幻觉名和错参数、判 ESCALATE；剩下的 post-dispatch
`isError` 才是真业务失败、判 INNER_RETRY。这正是"框架给的失败词汇表更弱（只一层），两层控制流
仍归我自己兜"——决策 E 的 host 守卫从"可选优化"升成"分流的承重墙"。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.shared.exceptions import McpError
except Exception:  # pragma: no cover - 兼容不同 SDK 版本的导出位置
    from mcp import McpError  # type: ignore

# 决策 F 的三条恢复路径（沿用周三 stub 的命名，保持跨日连续）
ESCALATE = "escalate_to_supervisor"   # 协议层 error → skip-and-advance（replan_count）
INNER_RETRY = "inner_retry"           # isError 业务失败 / 空结果 → empty_retries（内层重试）
SUCCESS = "success"                   # 非空成功 → inject_* 注回


@dataclass
class Routing:
    path: str                       # ESCALATE / INNER_RETRY / SUCCESS
    reason: str                     # protocol_error / pre_dispatch_reject / business_error / empty_result / ok
    payload: Any = None             # 成功/空时的结构化结果
    detail: str = ""                # 失败说明
    code: Any = None                # 协议层 error 的 JSON-RPC code（若 SDK 给）


def _extract_payload(result: Any) -> Any:
    """从 CallToolResult 取结构化 payload：优先 structuredContent，回退解析 text content。
    FastMCP 对 dict 返回可能包一层 {'result': {...}}，这里顺手拆掉。"""
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        if set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                return text
    return None


def is_empty(payload: Any) -> bool:
    """空结果判别（≠ 失败）：local_search→chunks 空 / http_fetch→content 空。"""
    if payload is None:
        return True
    if isinstance(payload, dict):
        if "chunks" in payload:
            return len(payload["chunks"]) == 0
        if "content" in payload:
            return len(payload["content"]) == 0
        return len(payload) == 0
    return len(payload) == 0 if isinstance(payload, (list, str)) else False


class MCPToolsStation:
    """连真实 stdio server 的 tools 站：connect → discover → (守卫) → call → 两层分流。"""

    def __init__(self, server_script: str | None = None, *, guard: bool = True):
        script = server_script or str(Path(__file__).resolve().parent / "mcp_server.py")
        self._params = StdioServerParameters(command=sys.executable, args=[script])
        self.guard = guard
        self._session: ClientSession | None = None
        self._discovered: dict[str, Any] = {}
        self._stdio_cm = None
        self._session_cm = None

    async def __aenter__(self) -> "MCPToolsStation":
        self._stdio_cm = stdio_client(self._params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(*exc)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(*exc)

    async def discover(self) -> dict[str, Any]:
        """tools/list：发现来的清单 = 有效工具集的【权威来源】（决策 B / E）。"""
        resp = await self._session.list_tools()
        self._discovered = {t.name: t for t in resp.tools}
        return self._discovered

    @property
    def discovered_names(self) -> set[str]:
        return set(self._discovered)

    def _validate_args(self, name: str, args: dict) -> str | None:
        """client 侧 inputSchema 校验（决策 E，v0.3 升成承重墙）：拿发现来的 schema 校验，
        返回错误说明或 None。真实 SDK 不为错参数抛协议异常，故 host 这步必须自己做。"""
        schema = getattr(self._discovered.get(name), "inputSchema", None)
        if not isinstance(schema, dict):
            return None
        try:
            jsonschema.validate(args, schema)
            return None
        except jsonschema.ValidationError as e:
            return e.message

    async def call(self, name: str, args: dict) -> Routing:
        """决策 E 守卫 + 决策 F 两层分流。返回 Routing（不抛异常给上层 loop）。

        [v0.3] 因真实 SDK 把"协议层/集成问题"也塌成 isError，host 在派发前自己重建这一支：
        名字守卫 + schema 校验命中 → ESCALATE（集成问题、上报 skip）；剩下的 post-dispatch
        isError 才是真业务失败 → INNER_RETRY。两层因此仍走不同路径，分流判据从 SDK 形态
        挪到 host 派发前。"""
        if self.guard:
            # ① 派发前守卫（决策 E）：幻觉名本地拒、不触达 server。
            if name not in self._discovered:
                return Routing(ESCALATE, "pre_dispatch_reject",
                               detail=f"pre-dispatch reject (not in discovered list): {name}")
            # ② client 侧 schema 校验：错参数派发前拦（真实 SDK 会把它塞进 isError，host 抢先判协议层）。
            err = self._validate_args(name, args)
            if err is not None:
                return Routing(ESCALATE, "invalid_args", detail=err)
        # ③ 调用 + 执行层分流（决策 F）。
        try:
            result = await self._session.call_tool(name, args)
        except McpError as e:                       # 真 JSON-RPC 协议错（malformed/未知 method）
            return Routing(ESCALATE, "protocol_error",
                           detail=str(e), code=getattr(getattr(e, "error", None), "code", None))
        if getattr(result, "isError", False):       # 执行层业务失败（isError 结果字段）
            return Routing(INNER_RETRY, "business_error", detail=_first_text(result))
        payload = _extract_payload(result)
        if is_empty(payload):                       # 空结果（≠ 失败）
            return Routing(INNER_RETRY, "empty_result", payload=payload)
        return Routing(SUCCESS, "ok", payload=payload)


def _first_text(result: Any) -> str:
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            return text
    return ""
