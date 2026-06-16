"""
E5 — 空结果 ≠ 失败：is_error=False + 空 chunks → 内层 retry，不判 error、不上报。
预判分类：巩固型。每条配对照组。
跑法：pytest -v experiment/test_e5_empty_not_failure.py
"""
from mcp_stub import MCPClient, build_server, route_tool_outcome, INNER_RETRY


def test_e5_empty_is_not_error_main():
    """主：空结果 is_error 仍为 False、被路由成 empty_result/INNER_RETRY。"""
    client = MCPClient(build_server(local_mode="empty"), guard=True)
    client.discover()
    res = client.call_tool("local_search", {"query": "查无此项"})
    assert res.is_error is False and res.content == []
    r = route_tool_outcome(res)
    assert r.path == INNER_RETRY and r.reason == "empty_result"


def test_e5_control_three_outcomes_distinguishable():
    """对照：空 / 业务失败 / 成功 三态的 reason 两两可分——loop 才能各走各的恢复路径。"""
    reasons = set()
    for mode in ("ok", "error", "empty"):
        c = MCPClient(build_server(local_mode=mode), guard=True)
        c.discover()
        reasons.add(route_tool_outcome(c.call_tool("local_search", {"query": "x"})).reason)
    assert reasons == {"ok", "business_error", "empty_result"}


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
