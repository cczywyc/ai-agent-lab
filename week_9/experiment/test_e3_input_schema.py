"""
E3 — inputSchema 拒错参数。  [发现型候选]
预判分类：发现型候选——schema 校验在 client 还是 server 侧 fire，今天拿桩问出来。
[周四确认] 真实 SDK 校验在哪侧 fire——本桩默认 server 侧（client_validates=False）。
跑法：pytest -v experiment/test_e3_input_schema.py
"""
import pytest

from mcp_stub import MCPClient, MCPProtocolError, ToolResult, INVALID_PARAMS, build_server


def test_e3_schema_rejects_missing_required_main():
    """主①：缺必填 query → INVALID_PARAMS。"""
    client = MCPClient(build_server(), guard=True)
    client.discover()
    with pytest.raises(MCPProtocolError) as ei:
        client.call_tool("local_search", {"top_k": 5})
    assert ei.value.code == INVALID_PARAMS


def test_e3_schema_rejects_wrong_type_main():
    """主②：query 类型错（int 而非 string）→ INVALID_PARAMS。"""
    client = MCPClient(build_server(), guard=True)
    client.discover()
    with pytest.raises(MCPProtocolError) as ei:
        client.call_tool("local_search", {"query": 123})
    assert ei.value.code == INVALID_PARAMS


def test_e3_control_valid_args_pass():
    """对照：合规参数正常通过、拿到非空结果。"""
    client = MCPClient(build_server(), guard=True)
    client.discover()
    res = client.call_tool("local_search", {"query": "v6.0 怎么演进"})
    assert isinstance(res, ToolResult) and res.is_error is False
    assert len(res.content) == 1


def test_e3_seam_client_side_validation_equivalent():
    """发现型 seam：把校验切到 client 侧（client_validates=True）应等价拒绝。
    今天两侧都拒得住；周四对真实 SDK 时，把 client_validates 设成与其一致即可。"""
    client = MCPClient(build_server(), guard=True, client_validates=True)
    client.discover()
    with pytest.raises(MCPProtocolError) as ei:
        client.call_tool("local_search", {"top_k": 5})
    assert ei.value.code == INVALID_PARAMS


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
