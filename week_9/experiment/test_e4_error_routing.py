"""
E4 — 两层错误路由分叉。  [发现型候选·核心；跨两周悬念结案]
预判分类：发现型候选·核心——协议 error vs isError 是否真分得开（第三周 Loop Q3、
第八周 Q3 那个'纠正/降级在哪层生效'，在 typed 失败信道下结案）。
[周四确认] 真实 SDK 把协议 error / isError 暴露成什么形态，决定 outcome 怎么被接住。
跑法：pytest -v experiment/test_e4_error_routing.py
"""
import pytest

from mcp_stub import (
    MCPClient, MCPProtocolError, build_server,
    route_tool_outcome, ESCALATE, INNER_RETRY,
)


def test_e4_protocol_error_routes_to_escalate():
    """主A：协议层 error → ESCALATE（上报 supervisor 走 skip），不 retry。"""
    server = build_server()
    client = MCPClient(server, guard=False)              # 关守卫，让协议 error 从 server 冒出
    client.discover()
    try:
        outcome = client.call_tool("nonexistent", {"query": "x"})
    except MCPProtocolError as e:
        outcome = e
    r = route_tool_outcome(outcome)
    assert r.path == ESCALATE and r.reason == "protocol_error"


def test_e4_iserror_routes_to_inner_retry():
    """主B：isError 业务失败 → INNER_RETRY（内层 empty_retries），不上报。"""
    client = MCPClient(build_server(local_mode="error"), guard=True)
    client.discover()
    outcome = client.call_tool("local_search", {"query": "x"})
    r = route_tool_outcome(outcome)
    assert r.path == INNER_RETRY and r.reason == "business_error"


def test_e4_two_layers_route_differently():
    """核心断言：协议 error 与 isError 必须走【不同】路径。"""
    srv = build_server()
    c1 = MCPClient(srv, guard=False); c1.discover()
    try:
        o1 = c1.call_tool("nope", {"query": "x"})
    except MCPProtocolError as e:
        o1 = e
    c2 = MCPClient(build_server(local_mode="error"), guard=True); c2.discover()
    o2 = c2.call_tool("local_search", {"query": "x"})

    assert route_tool_outcome(o1).path == ESCALATE
    assert route_tool_outcome(o2).path == INNER_RETRY
    assert route_tool_outcome(o1).path != route_tool_outcome(o2).path


def test_e4_control_empty_routes_inner_not_escalate():
    """对照：空结果（is_error=False、空 chunks）走 INNER_RETRY 而非 ESCALATE——
    空结果绝不能被误当协议 error 上报。"""
    client = MCPClient(build_server(local_mode="empty"), guard=True)
    client.discover()
    outcome = client.call_tool("local_search", {"query": "查无此项"})
    r = route_tool_outcome(outcome)
    assert r.path == INNER_RETRY and r.reason == "empty_result"
    assert r.path != ESCALATE


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
