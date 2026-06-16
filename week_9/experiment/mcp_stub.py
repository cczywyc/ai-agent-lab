"""
第九周桩测仪器：MCP 接口契约的纯进程内忠实桩（零 API、确定、可复现）。

定位：周三·实验/桩测日的"零 API 那一半"。本模块不接真实 LLM、不接真实 MCP SDK、
不碰真实 RAG/HTTP——只忠实建模 MCP 2025-11-25 的接口契约（tools/list 发现、
inputSchema 校验、两层错误模型），让 E1–E6 能在进程内确定性地验"接线机制"。

它验的是【我自己掌控的接线逻辑】：派发前守卫、两层错误在 researcher loop 里的路由
（决策 F）、turn_count/empty_retries/synthesis-reserve 在换接 tools 站后的回归。

它【不】验真实 SDK 的确切形态——那是 E3/E4 的发现型，留周四换上真实 FastMCP server
时坐实（见各处 [周四确认] 标记）。承第八周纪律：离线全绿仍是必要非充分；桩测验断路器
路径（失败怎么收口），健康中段主力路径（正常怎么跑好）是周五真实跑的事。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ── 两层错误模型（MCP 2025-11-25）────────────────────────────────────────
# 协议层错误：未知工具名 / 参数不合 schema / server 崩 → JSON-RPC error 响应。
#   本桩建模成异常，调用方在派发点 try/except 接住。
# 执行层错误：业务失败（限流/抓取失败）→ 装在结果里 is_error=True（见 ToolResult）。
class MCPProtocolError(Exception):
    """协议层错误（JSON-RPC error）。code 用标准码：-32601 未知方法 / -32602 参数非法。"""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


@dataclass
class ToolResult:
    """执行层结果：业务失败用 is_error=True 装在结果里（而非协议层异常）。"""
    content: Any            # 工具返回的 payload（chunks 列表 / 文本 / dict）
    is_error: bool = False  # True = 业务失败（限流/抓取失败），模型可据此 retry


@dataclass
class ToolDef:
    name: str
    description: str                       # 写给模型看：何时调 / 不干什么
    input_schema: dict                     # JSON Schema（required / properties / type）
    body: Callable[[dict], ToolResult]     # 桩工具体：零 API，返回固定 payload


# ── 极简 JSON-Schema 校验（只覆盖 required + type，桩测够用）──────────────
_JSON_TYPES: dict[str, Any] = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "object": dict, "array": list,
}


def validate_against_schema(args: dict, schema: dict) -> str | None:
    """返回错误说明字符串，或 None（合规）。"""
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required arg: {key}"
    for key, val in args.items():
        spec = props.get(key)
        if spec and "type" in spec:
            expected = _JSON_TYPES.get(spec["type"])
            # bool 是 int 的子类，单独防一下整型字段被布尔混入
            if expected and (not isinstance(val, expected)
                             or (spec["type"] == "integer" and isinstance(val, bool))):
                return f"arg {key!r} expected {spec['type']}, got {type(val).__name__}"
    return None


# ── 桩 MCP server（零 API）────────────────────────────────────────────────
class StubMCPServer:
    """持有 ToolDef 列表，建模 list_tools / call_tool。
    server 侧做协议层校验：未知名 → METHOD_NOT_FOUND，参数违 schema → INVALID_PARAMS。"""

    def __init__(self, tools: list[ToolDef]):
        self._tools: dict[str, ToolDef] = {t.name: t for t in tools}
        self.call_count = 0              # 探针：被真正调到几次（E2 主路径断言 server 没被调）
        self.list_changed_pending = False

    def list_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def call_tool(self, name: str, args: dict) -> ToolResult:
        self.call_count += 1
        if name not in self._tools:                                  # 协议层：未知工具名
            raise MCPProtocolError(METHOD_NOT_FOUND, f"unknown tool: {name}")
        tool = self._tools[name]
        err = validate_against_schema(args, tool.input_schema)        # 协议层：参数违 schema
        if err is not None:
            raise MCPProtocolError(INVALID_PARAMS, err)
        return tool.body(args)                                        # 执行层：返回 ToolResult

    def add_tool(self, tool: ToolDef) -> None:
        """变更工具表（E1 对照：触发 listChanged）。"""
        self._tools[tool.name] = tool
        self.list_changed_pending = True


# ── 桩 MCP client（host 侧）────────────────────────────────────────────────
class MCPClient:
    """connect → discover → （派发前守卫）→ call_tool。
    - discover(): tools/list，缓存发现来的清单 = 有效工具集的【权威来源】。
    - guard: 发 call 前比对名字 ∈ 发现来的清单，不在就本地拒、不发 call（可关，供 E2 对照）。
    - client_validates: 客户端侧 schema 校验开关。
      [周四确认] 真实 SDK 究竟在 client 还是 server 侧 fire——本桩默认 server 侧（client_validates=False）。"""

    def __init__(self, server: StubMCPServer, *, guard: bool = True, client_validates: bool = False):
        self._server = server
        self.guard = guard
        self.client_validates = client_validates
        self._discovered: dict[str, ToolDef] = {}

    def discover(self) -> dict[str, ToolDef]:
        self._discovered = {t.name: t for t in self._server.list_tools()}
        self._server.list_changed_pending = False
        return self._discovered

    @property
    def discovered_names(self) -> set[str]:
        return set(self._discovered)

    def call_tool(self, name: str, args: dict) -> ToolResult:
        # 派发前守卫：拿模型选的名字比对【发现来的清单】，不在就本地拒、不触达 server
        if self.guard and name not in self._discovered:
            raise MCPProtocolError(METHOD_NOT_FOUND, f"pre-dispatch reject (not in discovered list): {name}")
        if self.client_validates and name in self._discovered:        # 可选：客户端侧校验
            err = validate_against_schema(args, self._discovered[name].input_schema)
            if err is not None:
                raise MCPProtocolError(INVALID_PARAMS, f"client-side: {err}")
        return self._server.call_tool(name, args)


# ── 决策 F：researcher loop 的两层错误路由（这是【我自己的控制流】）────────
ESCALATE = "escalate_to_supervisor"   # → skip-and-advance（replan_count）
INNER_RETRY = "inner_retry"           # → empty_retries（内层重试）
SUCCESS = "success"                   # → inject_* 注回


@dataclass
class Routing:
    path: str       # ESCALATE / INNER_RETRY / SUCCESS
    reason: str     # protocol_error / business_error / empty_result / ok


def _is_empty(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, (list, str, dict)):
        return len(content) == 0
    return False


def route_tool_outcome(outcome: MCPProtocolError | ToolResult) -> Routing:
    """决策 F 核心——两层错误分流：
    - 协议层 error（集成坏了）→ 不 retry，上报 supervisor 走 skip
    - isError 业务失败       → 内层 retry
    - 空结果（is_error=False、content 空）→ 内层 retry（≠ 失败、≠ 上报）
    - 非空成功               → inject
    [周四确认] 真实 SDK 把协议 error / isError 暴露成什么形态，决定 outcome 怎么被接住。"""
    if isinstance(outcome, MCPProtocolError):
        return Routing(ESCALATE, "protocol_error")
    if isinstance(outcome, ToolResult):
        if outcome.is_error:
            return Routing(INNER_RETRY, "business_error")
        if _is_empty(outcome.content):
            return Routing(INNER_RETRY, "empty_result")
        return Routing(SUCCESS, "ok")
    raise TypeError(f"unexpected outcome: {outcome!r}")


# ── E6 用的最小 researcher 内层（仅为隔离验回归；端到端版周四搬进真实 v6.0 图）──
MAX_TURNS = 5
MAX_EMPTY_RETRIES = 2
SYNTHESIS_RESERVE_AT = MAX_TURNS - 1   # 最后一轮预留给综合（synthesis-reserve）


@dataclass
class LoopTrace:
    turn_count: int = 0
    empty_retries: int = 0
    synthesis_forced: bool = False
    injected: list = field(default_factory=list)
    last_path: str = ""


def run_researcher_loop(scripted_calls, tools_station) -> LoopTrace:
    """最小 agent↔tools↔inject 内层，跑确定性脚本调用序列（不接 LLM）。
    tools_station: Callable[[name, args], ToolResult]（old=dict 派发 / new=MCP client 派发）。
    闸门（turn_count / empty_retries / synthesis-reserve）全在 loop 里、与 station 无关——
    这正是决策 G "包一层" 的设计主张，E6 用它验换 station 后行为不变。"""
    tr = LoopTrace()
    calls = list(scripted_calls)
    i = 0
    while tr.turn_count < MAX_TURNS:
        # synthesis-reserve：到预留轮且已有结果 → 逼综合早退
        if tr.turn_count >= SYNTHESIS_RESERVE_AT and tr.injected:
            tr.synthesis_forced = True
            break
        if i >= len(calls):
            break
        tr.turn_count += 1
        name, args = calls[i]
        try:
            outcome = tools_station(name, args)
        except MCPProtocolError as e:
            outcome = e
        routing = route_tool_outcome(outcome)
        tr.last_path = routing.path
        if routing.path == SUCCESS:
            tr.injected.append(outcome.content)
            i += 1
        elif routing.path == INNER_RETRY:
            tr.empty_retries += 1
            if tr.empty_retries > MAX_EMPTY_RETRIES:   # 重试跑满 → retry_reset 轻量重置、推进
                tr.empty_retries = 0
                i += 1
            # 否则不推进 i：重试同一调用
        elif routing.path == ESCALATE:
            break                                       # 协议 error：退出内层、上报
    return tr


# ── 两个工具的桩工具体（零 API、固定 payload；mode 旋钮注入失败模式）────────
LOCAL_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
    "required": ["query"],
}
HTTP_FETCH_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
    "required": ["url"],
}


def make_local_search(mode: str = "ok") -> Callable[[dict], ToolResult]:
    def body(args: dict) -> ToolResult:
        if mode == "empty":                                          # 空结果（≠ 失败）
            return ToolResult(content=[], is_error=False)
        if mode == "error":                                          # 业务失败（如限流）
            return ToolResult(content="rate limited", is_error=True)
        return ToolResult(
            content=[{"text": "v6.0 用 supervisor 多 Agent", "citation": "wk8#一"}],
            is_error=False,
        )
    return body


def make_http_fetch(mode: str = "ok") -> Callable[[dict], ToolResult]:
    def body(args: dict) -> ToolResult:
        if mode == "empty":
            return ToolResult(content="", is_error=False)
        if mode == "error":                                          # 抓取失败 403/超时
            return ToolResult(content="403 forbidden", is_error=True)
        return ToolResult(content={"content": "external page text", "status": 200}, is_error=False)
    return body


def build_server(local_mode: str = "ok", http_mode: str = "ok") -> StubMCPServer:
    """标准两工具桩 server（决策 B：local_search + http_fetch）。"""
    return StubMCPServer([
        ToolDef("local_search", "查本项目/本地库内容；不抓外网", LOCAL_SEARCH_SCHEMA, make_local_search(local_mode)),
        ToolDef("http_fetch", "查外部公开网页；不查本地库", HTTP_FETCH_SCHEMA, make_http_fetch(http_mode)),
    ])
