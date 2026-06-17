"""
第九周桩测总跑：E1–E6 一次跑完，每条打印 week-8 体例的 [En] …: PASS + 实测探针值，
末行 总计 N/N。零 API、确定、可复现——不依赖 pytest（与 week_8 的 plain 脚本同构）；
granular 断言谱见 test_e1..e6（pytest，16 个函数）。

跑法：../../.venv/bin/python run_all.py
"""
from __future__ import annotations

from mcp_stub import (
    MCPClient, MCPProtocolError, ToolDef, ToolResult,
    METHOD_NOT_FOUND, INVALID_PARAMS,
    build_server, make_local_search, make_http_fetch,
    route_tool_outcome, run_researcher_loop,
    ESCALATE, INNER_RETRY, SUCCESS,
)

_passed = 0
_total = 0


def check(name: str, ok: bool, detail: str) -> None:
    global _passed, _total
    _total += 1
    if ok:
        _passed += 1
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    print(f"      {detail}")


# ── E1 工具发现 ────────────────────────────────────────────────────────────
def e1():
    server = build_server()
    client = MCPClient(server)
    discovered = client.discover()
    main_ok = (client.discovered_names == {"local_search", "http_fetch"}
               and discovered["local_search"].input_schema["required"] == ["query"]
               and discovered["http_fetch"].input_schema["required"] == ["url"])

    # 对照：变更工具表 → listChanged，重发现前缓存仍旧、重发现后才反映
    server.add_tool(ToolDef("summarize", "概括 findings",
                            {"type": "object", "properties": {}, "required": []},
                            make_local_search("ok")))
    stale = "summarize" not in client.discovered_names and server.list_changed_pending is True
    client.discover()
    refreshed = "summarize" in client.discovered_names and server.list_changed_pending is False

    check("E1 工具发现", main_ok and stale and refreshed,
          f"discover={sorted(['local_search','http_fetch'])}、schema.required 各就位；"
          f"变更后未重发现仍旧={stale}；重发现后 summarize 入清单={refreshed}")


# ── E2 派发前守卫拒幻觉名 ──────────────────────────────────────────────────
def e2():
    # 主：守卫开 → 'edit' 本地拒、server 从未被触达
    s_on = build_server(); c_on = MCPClient(s_on, guard=True); c_on.discover()
    try:
        c_on.call_tool("edit", {"query": "x"}); code_on = None
    except MCPProtocolError as e:
        code_on = e.code
    main_ok = code_on == METHOD_NOT_FOUND and s_on.call_count == 0

    # 对照：守卫关 → 'edit' 触达 server、运行期协议 error（call_count==1）
    s_off = build_server(); c_off = MCPClient(s_off, guard=False); c_off.discover()
    try:
        c_off.call_tool("edit", {"query": "x"}); code_off = None
    except MCPProtocolError as e:
        code_off = e.code
    ctrl_ok = code_off == METHOD_NOT_FOUND and s_off.call_count == 1

    check("E2 派发前守卫拒幻觉名", main_ok and ctrl_ok,
          f"守卫开 'edit' 本地拒 call_count={s_on.call_count}（派发前、从未上线）；"
          f"守卫关触达 server call_count={s_off.call_count}（运行期协议 error）——两路径都在，默认前者")


# ── E3 inputSchema 拒错参数 ────────────────────────────────────────────────
def e3():
    c = MCPClient(build_server(), guard=True); c.discover()

    def code_of(args):
        try:
            c.call_tool("local_search", args); return None
        except MCPProtocolError as e:
            return e.code

    miss = code_of({"top_k": 5})                 # 缺必填 query
    wrong = code_of({"query": 123})              # 类型错
    ok_res = c.call_tool("local_search", {"query": "v6.0 怎么演进"})
    valid_ok = ok_res.is_error is False and len(ok_res.content) == 1

    # 发现型 seam：校验切 client 侧应等价拒绝（周四对真实 SDK 时设成与其一致）
    c2 = MCPClient(build_server(), guard=True, client_validates=True); c2.discover()
    try:
        c2.call_tool("local_search", {"top_k": 5}); client_side = None
    except MCPProtocolError as e:
        client_side = e.code

    check("E3 inputSchema 拒错参数",
          miss == INVALID_PARAMS and wrong == INVALID_PARAMS and valid_ok
          and client_side == INVALID_PARAMS,
          f"缺 query→INVALID_PARAMS({miss})；类型错→INVALID_PARAMS({wrong})；"
          f"合规通过 chunks={len(ok_res.content)}；校验切 client 侧等价拒={client_side == INVALID_PARAMS}"
          f"（桩默认 server 侧校验，client_validates=False）")


# ── E4 两层错误路由分叉 ────────────────────────────────────────────────────
def e4():
    # 协议层 error（关守卫让其从 server 冒出）→ ESCALATE
    s = build_server(); c = MCPClient(s, guard=False); c.discover()
    try:
        o_proto = c.call_tool("nonexistent", {"query": "x"})
    except MCPProtocolError as e:
        o_proto = e
    r_proto = route_tool_outcome(o_proto)

    # isError 业务失败 → INNER_RETRY
    c_err = MCPClient(build_server(local_mode="error"), guard=True); c_err.discover()
    r_biz = route_tool_outcome(c_err.call_tool("local_search", {"query": "x"}))

    # 对照：空结果 → INNER_RETRY 而非 ESCALATE
    c_emp = MCPClient(build_server(local_mode="empty"), guard=True); c_emp.discover()
    r_emp = route_tool_outcome(c_emp.call_tool("local_search", {"query": "查无此项"}))

    ok = (r_proto.path == ESCALATE and r_proto.reason == "protocol_error"
          and r_biz.path == INNER_RETRY and r_biz.reason == "business_error"
          and r_emp.path == INNER_RETRY and r_emp.reason == "empty_result"
          and r_proto.path != r_biz.path)
    check("E4 两层错误路由分叉", ok,
          f"协议error→{r_proto.path}；isError→{r_biz.path}；空结果→{r_emp.path}≠{ESCALATE}"
          f"（协议层上报 skip / 执行层进内层 retry，两层真分得开）")


# ── E5 空结果≠失败 ────────────────────────────────────────────────────────
def e5():
    c = MCPClient(build_server(local_mode="empty"), guard=True); c.discover()
    res = c.call_tool("local_search", {"query": "查无此项"})
    main_ok = res.is_error is False and res.content == [] \
        and route_tool_outcome(res).reason == "empty_result"

    reasons = set()
    for mode in ("ok", "error", "empty"):
        cc = MCPClient(build_server(local_mode=mode), guard=True); cc.discover()
        reasons.add(route_tool_outcome(cc.call_tool("local_search", {"query": "x"})).reason)
    ctrl_ok = reasons == {"ok", "business_error", "empty_result"}

    check("E5 空结果≠失败", main_ok and ctrl_ok,
          f"空 is_error={res.is_error}/content={res.content}→empty_result/inner_retry；"
          f"三态 reason={sorted(reasons)} 两两可分（loop 才能各走各的恢复路径）")


# ── E6 包一层无回归 ────────────────────────────────────────────────────────
def _old_station(name, args):
    registry = {"local_search": make_local_search("ok"), "http_fetch": make_http_fetch("ok")}
    if name not in registry:
        raise MCPProtocolError(METHOD_NOT_FOUND, f"UnknownTool: {name}")
    return registry[name](args)


def e6():
    script = [("local_search", {"query": "q"}), ("http_fetch", {"url": "u"})] * 3
    old = run_researcher_loop(script, _old_station)

    client = MCPClient(build_server(), guard=True); client.discover()
    new = run_researcher_loop(script, lambda n, a: client.call_tool(n, a))

    same = ((old.turn_count, old.empty_retries, old.synthesis_forced)
            == (new.turn_count, new.empty_retries, new.synthesis_forced)
            and old.injected == new.injected and old.synthesis_forced is True)

    # 对照：扰动 station（吞内容）→ trace 不一致、被检出
    script2 = [("local_search", {"query": "q"})] * 4
    old2 = run_researcher_loop(script2, _old_station)

    def buggy(name, args):
        _old_station(name, args)
        return ToolResult(content=[], is_error=False)
    bug = run_researcher_loop(script2, buggy)
    detected = (old2.turn_count, old2.empty_retries) != (bug.turn_count, bug.empty_retries)

    check("E6 包一层无回归", same and detected,
          f"old/new station trace 全等 turn_count={new.turn_count}/empty_retries={new.empty_retries}/"
          f"synthesis_forced={new.synthesis_forced}、injected 全等；扰动 station 被检出={detected}")


if __name__ == "__main__":
    for fn in (e1, e2, e3, e4, e5, e6):
        fn()
    print(f"\n总计 {_passed}/{_total}")
    raise SystemExit(0 if _passed == _total else 1)
