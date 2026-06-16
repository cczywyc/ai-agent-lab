"""
E2 — 派发前守卫拒幻觉名：吐清单外的 'edit' → 守卫本地拒、不发 call。
预判分类：巩固型。每条配对照组。
零 API 纪律：幻觉名是确定性地 call_tool('edit', ...) 喂的，无真实 LLM 选工具。
跑法：pytest -v experiment/test_e2_dispatch_guard.py
"""
import pytest

from mcp_stub import MCPClient, MCPProtocolError, METHOD_NOT_FOUND, build_server


def test_e2_guard_rejects_hallucinated_name_main():
    """主：守卫开。'edit' 不在发现清单 → 本地拒，且 server 从未被触达（call_count==0）。"""
    server = build_server()
    client = MCPClient(server, guard=True)
    client.discover()
    with pytest.raises(MCPProtocolError) as ei:
        client.call_tool("edit", {"query": "x"})
    assert ei.value.code == METHOD_NOT_FOUND
    assert server.call_count == 0                        # 关键：派发前拒，从未上线


def test_e2_control_guard_off_hits_server_protocol_error():
    """对照：守卫关。'edit' 触达 server → 协议层 error（运行期撞错，非 isError）。
    主/对照都拒，但主在【派发前】（call_count==0）、对照在【运行期 server】（call_count==1）。"""
    server = build_server()
    client = MCPClient(server, guard=False)
    client.discover()
    with pytest.raises(MCPProtocolError) as ei:
        client.call_tool("edit", {"query": "x"})
    assert ei.value.code == METHOD_NOT_FOUND
    assert server.call_count == 1                        # 触达了 server


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
