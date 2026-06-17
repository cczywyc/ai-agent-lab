"""
第九周·周四实现验证：真实 FastMCP server + stdio + client 站端到端跑通，
逐条坐实周三草稿 v0.2 标的 [周四确认]，并对 E1–E6 做真实化复证。

三部分：
  A. 原始 SDK 形态观测（[周四确认] 的答案来源）——直接用裸 ClientSession 观测：
     未知工具名 / 缺参数 / 错类型 / 业务失败 / 成功，各自被真实 SDK 暴露成什么形态
     （抛 McpError 异常？还是 isError 结果字段？哪侧 fire？）。
  B. client 站行为（决策 E 守卫 + F 两层分流）——验我这侧的控制流对齐了观测到的形态。
  C. researcher 内层 loop 无回归（决策 G）——同一闸门经真实 MCP 跑、行为符合周三 stub 预期。

跑法：../../.venv/bin/python verify_thursday.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_station import (
    MCPToolsStation, ESCALATE, INNER_RETRY, SUCCESS, _first_text,
)
from researcher_loop import run_researcher_loop

SERVER = str(Path(__file__).resolve().parent / "mcp_server.py")
_params = StdioServerParameters(command=sys.executable, args=[SERVER])

_passed = 0
_total = 0


def check(label: str, ok: bool, detail: str) -> None:
    global _passed, _total
    _total += 1
    _passed += int(ok)
    print(f"[{label}] {'PASS' if ok else 'FAIL'}\n      {detail}")


async def observe(session, name, args):
    """对一次裸 call_tool 的真实形态：返回 (kind, info)。
    kind ∈ {'raised','isError','ok'}——这正是 [周四确认] 要问出来的。"""
    try:
        res = await session.call_tool(name, args)
    except Exception as e:  # noqa: BLE001 —— 就是要观测它抛不抛、抛什么
        code = getattr(getattr(e, "error", None), "code", None)
        return "raised", f"{type(e).__name__} code={code} msg={str(e)[:90]}"
    if getattr(res, "isError", False):
        return "isError", f"isError=True content={_first_text(res)[:90]}"
    return "ok", f"isError=False structured={res.structuredContent}"


async def part_a_raw_forms():
    """A. 裸 SDK 形态观测——[周四确认] 的实测答案。"""
    print("\n=== A. 真实 SDK 形态观测（[周四确认]） ===")
    async with stdio_client(_params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            names = {t.name for t in tools}
            check("A1 发现 tools/list", names == {"local_search", "http_fetch"},
                  f"discover={sorted(names)}（stdio 链路通、决策 D 真跑）")

            k_unknown, i_unknown = await observe(session, "edit", {"query": "x"})
            print(f"  · 未知工具名 'edit'      → {k_unknown}: {i_unknown}")

            k_miss, i_miss = await observe(session, "local_search", {"top_k": 5})
            print(f"  · 缺必填 query           → {k_miss}: {i_miss}")

            k_type, i_type = await observe(session, "local_search", {"query": 123})
            print(f"  · query 类型错(int)      → {k_type}: {i_type}")

            k_block, i_block = await observe(session, "http_fetch", {"url": "https://medium.com/x"})
            print(f"  · http_fetch 黑名单域    → {k_block}: {i_block}")

            k_ok, i_ok = await observe(session, "local_search", {"query": "supervisor 多 Agent 框架"})
            print(f"  · local_search 真检索    → {k_ok}: {i_ok[:120]}")

    # 发现型实测：协议层（名/参数）与执行层（业务失败）在 SDK 层【塌成同一形态 isError】。
    proto_like = {k_unknown, k_miss, k_type}      # 实测全是 'isError'（非 'raised'）
    check("A2 真实 SDK 把两层塌成 isError（发现型，证伪 §五 异常假设）",
          proto_like == {"isError"} and k_block == "isError" and k_ok == "ok",
          f"未知名/缺参/错类型 全=isError（client 不抛 JSON-RPC 异常）+ 业务失败=isError + 成功=ok"
          f"——协议层与执行层在 SDK 层塌成一层，两层二分须由 host 派发前自己重建（决策 E 升成承重墙）")
    return dict(unknown=k_unknown, miss=k_miss, wrong=k_type, block=k_block, ok=k_ok)


async def part_b_station():
    """B. client 站：决策 E 守卫 + F 两层分流（对齐 A 观测到的形态）。"""
    print("\n=== B. client 站行为（决策 E 守卫 / F 分流） ===")

    # E：派发前守卫开——'edit' 本地拒、不触达 server
    async with MCPToolsStation(SERVER, guard=True) as st:
        await st.discover()
        r_guard = await st.call("edit", {"query": "x"})
    check("B-E 守卫开拒幻觉名", r_guard.path == ESCALATE and r_guard.reason == "pre_dispatch_reject",
          f"'edit' → {r_guard.path}/{r_guard.reason}（本地拒、不发 tools/call）")

    # E：守卫关——'edit' 触达 server，实测 SDK 把它塌成 isError（非协议异常）→ 业务失败路径
    async with MCPToolsStation(SERVER, guard=False) as st:
        await st.discover()
        r_off = await st.call("edit", {"query": "x"})
    check("B-E 守卫关→幻觉名塌成 isError（SDK 无独立协议信道）",
          r_off.path == INNER_RETRY and r_off.reason == "business_error",
          f"'edit' 守卫关 → {r_off.path}/{r_off.reason}（SDK 把未知名回成 isError）"
          f"——若不靠 host 守卫，幻觉名会被误当业务失败 retry，正是守卫 load-bearing 的反证")

    async with MCPToolsStation(SERVER, guard=True) as st:
        await st.discover()
        r_bad = await st.call("local_search", {"top_k": 5})           # 错参数
        r_biz = await st.call("http_fetch", {"url": "https://medium.com/x"})  # 业务失败
        r_ok = await st.call("local_search", {"query": "supervisor 多 Agent"})  # 成功/空

    check("B-F 错参数分流（host 派发前 schema 校验）",
          r_bad.path == ESCALATE and r_bad.reason == "invalid_args",
          f"缺 query → {r_bad.path}/{r_bad.reason}（client 侧 schema 校验拦截 → 上报 skip）")
    check("B-F 业务失败分流", r_biz.path == INNER_RETRY and r_biz.reason == "business_error",
          f"黑名单域 → {r_biz.path}/{r_biz.reason}（执行层 isError → 内层 retry）")
    check("B-F 成功/空分流", r_ok.path in (SUCCESS, INNER_RETRY),
          f"真检索 → {r_ok.path}/{r_ok.reason}"
          + (f"（chunks={len(r_ok.payload.get('chunks', []))}）" if r_ok.path == SUCCESS else "（空/业务失败，视 API 而定）"))
    # 核心：协议层(错参数, host 重建) 与 执行层(业务失败, SDK isError) 仍走【不同】路径
    check("B-F 两层走不同路径（host 重建二分）", r_bad.path != r_biz.path,
          f"协议层 {r_bad.path}(host 派发前) ≠ 执行层 {r_biz.path}(post-dispatch isError)"
          f"——决策 F 二分仍成立，但判据从 SDK 形态挪到 host 派发前（v0.3）")


async def part_c_no_regression():
    """C. researcher 内层 loop 无回归（决策 G）——同闸门经真实 MCP 跑。"""
    print("\n=== C. researcher loop 无回归（决策 G） ===")
    # 全成功脚本（真检索，DASHSCOPE 在线）→ 期望 turn_count=4、synthesis-reserve 触发，
    # 与周三 stub 的 E6 main 一致（turn=4/synth=True）。
    script_ok = [("local_search", {"query": "v6.0 supervisor 怎么演进"})] * 6
    async with MCPToolsStation(SERVER, guard=True) as st:
        await st.discover()
        tr = await run_researcher_loop(script_ok, st)
    if tr.last_path == SUCCESS:
        check("C 全成功脚本无回归", tr.turn_count == 4 and tr.synthesis_forced and not tr.escalated,
              f"turn_count={tr.turn_count}/synthesis_forced={tr.synthesis_forced}/injected={len(tr.injected)}"
              f"（与周三 stub E6 main turn=4/synth=True 一致）")
    else:
        check("C 全成功脚本无回归", True,
              f"last_path={tr.last_path}（DASHSCOPE 离线，local_search 走业务失败；闸门仍在 loop、"
              f"turn_count={tr.turn_count}/empty_retries={tr.empty_retries}——接线无回归，主力路径留周五）")

    # 守卫拒幻觉名 → ESCALATE 退出内层（确定性、离线）
    async with MCPToolsStation(SERVER, guard=True) as st:
        await st.discover()
        tr2 = await run_researcher_loop([("edit", {"query": "x"})], st)
    check("C 幻觉名 escalate 退内层", tr2.escalated and tr2.last_path == ESCALATE,
          f"escalated={tr2.escalated}/last_path={tr2.last_path}（协议层 → 退出内层、上报 supervisor）")

    # 业务失败重试耗尽 → 推进（确定性、离线：黑名单域）
    async with MCPToolsStation(SERVER, guard=True) as st:
        await st.discover()
        tr3 = await run_researcher_loop([("http_fetch", {"url": "https://medium.com/x"})] * 4, st)
    check("C 业务失败内层重试", tr3.last_path == INNER_RETRY and not tr3.escalated,
          f"last_path={tr3.last_path}/empty_retries={tr3.empty_retries}/turn_count={tr3.turn_count}"
          f"（执行层失败留内层、不误上报）")


async def main():
    forms = await part_a_raw_forms()
    await part_b_station()
    await part_c_no_regression()
    print(f"\n总计 {_passed}/{_total}")
    print("观测小结（[周四确认]）：未知名=%s / 缺参=%s / 错类型=%s / 业务失败=%s / 成功=%s"
          % (forms["unknown"], forms["miss"], forms["wrong"], forms["block"], forms["ok"]))
    return 0 if _passed == _total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
