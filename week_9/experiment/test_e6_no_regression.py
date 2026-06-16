"""
E6 — 包一层无回归：换接 MCP client 后 turn_count/empty_retries/synthesis-reserve 不变。
预判分类：巩固型（blast radius 应够小）。每条配对照组。
注：此为隔离的最小内层模型；端到端版周四搬进真实 v6.0 图（test_graph.py）复跑——
    与第八周"E1–E7 桩测 → 搬进真实图复跑"的两层结构一致。
跑法：pytest -v experiment/test_e6_no_regression.py
"""
from mcp_stub import (
    MCPClient, MCPProtocolError, ToolResult, METHOD_NOT_FOUND,
    run_researcher_loop, build_server, make_local_search, make_http_fetch,
)


def _old_station(name, args):
    """v6.0 硬编码 tools 站：dict 派发 + UnknownTool 兜底。"""
    registry = {"local_search": make_local_search("ok"), "http_fetch": make_http_fetch("ok")}
    if name not in registry:
        raise MCPProtocolError(METHOD_NOT_FOUND, f"UnknownTool: {name}")
    return registry[name](args)


def _new_station(client):
    """v7.0 tools 站：经 MCP client 派发。"""
    def station(name, args):
        return client.call_tool(name, args)
    return station


def test_e6_no_regression_main():
    """主：同一脚本调用序列，old/new station 产出的 trace（闸门计数 + 注入）完全一致。"""
    script = [("local_search", {"query": "q"}), ("http_fetch", {"url": "u"})] * 3

    old_trace = run_researcher_loop(script, _old_station)

    client = MCPClient(build_server(), guard=True)
    client.discover()
    new_trace = run_researcher_loop(script, _new_station(client))

    assert (old_trace.turn_count, old_trace.empty_retries, old_trace.synthesis_forced) == \
           (new_trace.turn_count, new_trace.empty_retries, new_trace.synthesis_forced)
    assert old_trace.injected == new_trace.injected
    assert old_trace.synthesis_forced is True            # 这条脚本应触发 synthesis-reserve


def test_e6_control_perturbation_is_detected():
    """对照：故意扰动的 station（吞掉内容、多走 INNER_RETRY）→ trace 与 old 不一致、被检出。
    证明 E6 的绿不是空转——回归断言确有分辨力。"""
    script = [("local_search", {"query": "q"})] * 4
    old_trace = run_researcher_loop(script, _old_station)

    def buggy_station(name, args):
        _old_station(name, args)                          # 照常派发
        return ToolResult(content=[], is_error=False)     # 扰动：吞掉内容

    bug_trace = run_researcher_loop(script, buggy_station)
    assert (old_trace.turn_count, old_trace.empty_retries) != \
           (bug_trace.turn_count, bug_trace.empty_retries)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
