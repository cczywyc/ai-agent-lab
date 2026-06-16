"""
E1 — 工具发现：client 经 tools/list 拿到两工具的 name + schema。
预判分类：巩固型。每条配对照组。
跑法：pytest -v experiment/test_e1_discovery.py
"""
from mcp_stub import MCPClient, ToolDef, build_server, make_local_search


def test_e1_discovery_main():
    """主：discover() 拿到 local_search/http_fetch 及其 inputSchema。"""
    client = MCPClient(build_server())
    discovered = client.discover()
    assert client.discovered_names == {"local_search", "http_fetch"}
    assert discovered["local_search"].input_schema["required"] == ["query"]
    assert discovered["http_fetch"].input_schema["required"] == ["url"]


def test_e1_control_listchanged_reflects_change():
    """对照：工具表变更后 listChanged → 重新 discover 才反映变化。
    证明【发现期】是有效工具集的权威来源，重发现前 client 缓存是旧的。"""
    server = build_server()
    client = MCPClient(server)
    client.discover()
    assert "summarize" not in client.discovered_names

    server.add_tool(ToolDef("summarize", "概括 findings",
                            {"type": "object", "properties": {}, "required": []},
                            make_local_search("ok")))
    assert server.list_changed_pending is True          # server 侧已标脏
    assert "summarize" not in client.discovered_names   # 重发现前缓存仍旧

    client.discover()                                   # listChanged → 重新拉取
    assert "summarize" in client.discovered_names
    assert server.list_changed_pending is False


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
