"""
第九周·周五真实跑：真实 qwen 模型 + 真实 RAG/HTTP + 真实 stdio MCP server 端到端。

周三桩测 + 周四 verify 验的全是断路器路径（确定性脚本、零 LLM）；今天第一次让**真实 LLM**
驱动一个 researcher，**经真实 MCP client + stdio** 调真实工具，验：
  T2 健康中段主力路径——模型选对 local_search → 真检索回真 chunks → 综合出带 [doc#section] 引用的真答案；
  T3 缺陷 #5 漂移在活模型下——generic 措辞的本项目题，工具 description 软锚削不削漂移（local vs http）；
  T3′ 守卫承重墙活体检验——活模型管线里塞一个清单外的名字，周四"守卫升成承重墙"兜不兜得住；
  T4-inner escalate 内层半段——业务失败/守卫拒在真实 loop 里产出 ESCALATE（喂完整图 supervisor skip，见结论 scope）。

provenance caveat（承第八周）：`qwen3.7-max` 免费额度已耗尽（403 AllocationQuota.FreeTierOnly），
沿用第八周 topic D 的 fallback → `qwen3.7-plus-2026-05-26`。

scope：本周 demo 裁到 **researcher 内层 + MCP server**（设计草稿 §六 / 决策 G）。完整 supervisor 图
async-in-LangGraph 整合（Task 1 / Task 4 supervisor 端 skip）见结论 scope 诚实——本文件验到内层
ESCALATE，supervisor 端 skip-and-advance 机器是第八周 topic D 已 real 坐实的件。

跑法：../../.venv/bin/python real_run_friday.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

_V6 = Path(__file__).resolve().parents[2] / "week_8" / "search_agent"
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

from config import client  # v6.0 的 OpenAI(DashScope) 客户端，原样复用

from mcp_station import MCPToolsStation, ESCALATE, INNER_RETRY, SUCCESS

FRIDAY_MODEL = "qwen3.7-plus-2026-05-26"   # qwen3.7-max 403 耗尽，fallback（同第八周 D）
MAX_TURNS = 5
SYNTH_RESERVE_AT = MAX_TURNS - 1
OUT = Path(__file__).resolve().parent / "friday_run.json"

RESEARCHER_SYSTEM = (
    "你是研究子 Agent，用提供的工具检索证据回答问题。\n"
    "- 关于'本项目/本地库/这个 Agent 项目/它的版本演进'的问题，优先用 local_search（查本地笔记/设计文档/周复盘）。\n"
    "- 只有问题确实需要外部公开网页内容时才用 http_fetch。\n"
    "- 最终答案必须基于检索到的 chunk，引用格式 [doc#section]，只引检索真实返回的来源，别编造引用。\n"
    "- 检索到足够信息后就停手综合作答，别反复检索。"
)

_CITE = re.compile(r"\[([^\[\]]+?#[^\[\]]+?)\]")


def discovered_to_openai_tools(discovered: dict) -> list[dict]:
    """把 MCP 发现来的工具清单转成 OpenAI function-calling 定义——模型只看见这两个、别的都没。"""
    return [
        {"type": "function", "function": {
            "name": name, "description": t.description or "", "parameters": t.inputSchema}}
        for name, t in discovered.items()
    ]


async def run_real_researcher(question: str, station: MCPToolsStation, *, label: str) -> dict:
    """真实 LLM 驱动的 researcher 内层：模型选工具 → 经 MCP 真调 → inject → 综合。"""
    tools = discovered_to_openai_tools(station._discovered)
    messages = [{"role": "system", "content": RESEARCHER_SYSTEM},
                {"role": "user", "content": question}]
    tool_log: list[dict] = []          # 每次工具调用：name/args/path/reason
    retrieved_cites: set[str] = set()  # 真检索回来的 [doc#section]（引用白名单）
    tokens = 0
    turn = 0
    final = ""
    escalated = False
    while turn < MAX_TURNS:
        force_synth = turn >= SYNTH_RESERVE_AT and retrieved_cites
        if force_synth:
            messages.append({"role": "user",
                             "content": "已检索足够，请停止检索、立即基于已有 chunk 综合作答，引用 [doc#section]。"})
        turn += 1
        kwargs = dict(model=FRIDAY_MODEL, messages=messages)
        if not force_synth:
            kwargs["tools"] = tools
        resp = client.chat.completions.create(**kwargs)
        tokens += resp.usage.total_tokens
        msg = resp.choices[0].message
        if msg.tool_calls and not force_synth:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                routing = await station.call(name, args)
                tool_log.append({"turn": turn, "name": name, "args": args,
                                 "path": routing.path, "reason": routing.reason})
                if routing.path == SUCCESS:
                    for ch in (routing.payload or {}).get("chunks", []):
                        c = ch.get("citation", "")
                        if c:
                            retrieved_cites.add(c.strip("[]"))
                    content = json.dumps(routing.payload, ensure_ascii=False)
                else:
                    if routing.path == ESCALATE:
                        escalated = True
                    content = json.dumps({"error": True, "path": routing.path,
                                          "reason": routing.reason, "detail": routing.detail[:200]},
                                         ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        else:
            final = msg.content or ""
            break

    answer_cites = set(_CITE.findall(final))
    grounded = sum(1 for c in answer_cites if c in retrieved_cites)
    grounding = round(grounded / len(answer_cites), 3) if answer_cites else None
    return {
        "label": label, "question": question, "turns": turn, "tokens": tokens,
        "tool_calls": tool_log,
        "tools_used": sorted({t["name"] for t in tool_log}),
        "retrieved_citations": sorted(retrieved_cites),
        "answer_citations": sorted(answer_cites),
        "grounding": grounding, "escalated": escalated,
        "final_answer": final,
    }


async def guard_probe(station: MCPToolsStation) -> dict:
    """T3′ 守卫承重墙活体检验：活管线里塞清单外的名字 + 错参数，看 host 派发前守卫兜不兜。"""
    r_name = await station.call("edit", {"query": "x"})                 # 幻觉名
    r_args = await station.call("local_search", {"top_k": 5})           # 缺必填 query
    r_biz = await station.call("http_fetch", {"url": "https://medium.com/x"})  # 业务失败（对照）
    return {
        "hallucinated_name": {"path": r_name.path, "reason": r_name.reason},
        "bad_args": {"path": r_args.path, "reason": r_args.reason},
        "business_fail": {"path": r_biz.path, "reason": r_biz.reason},
    }


async def main():
    results: dict = {"model": FRIDAY_MODEL}
    async with MCPToolsStation(guard=True) as station:
        await station.discover()
        results["discovered"] = sorted(station.discovered_names)

        # T2 健康中段：本地库答得出的题
        results["T2_local"] = await run_real_researcher(
            "这个 Agent 项目里，v6.0 的 supervisor 多 Agent 相比 v5.0 单 agent 改了什么？",
            station, label="T2_local")

        # T3 漂移：generic 措辞、可本地可外网的题，看软锚削不削漂移
        results["T3_drift"] = await run_real_researcher(
            "supervisor 多 Agent 模式相比单 agent 在控制流上有什么改进？",
            station, label="T3_drift")

        # T3′ 守卫承重墙活体检验（确定性，零 LLM）
        results["T3_guard"] = await guard_probe(station)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 终端摘要 ──
    print(f"\n========= 周五真实跑摘要（model={FRIDAY_MODEL}）=========")
    for key in ("T2_local", "T3_drift"):
        r = results[key]
        print(f"\n[{key}] {r['question']}")
        print(f"  turns={r['turns']} tokens={r['tokens']} tools_used={r['tools_used']} escalated={r['escalated']}")
        print(f"  tool_calls={[(t['name'], t['path']) for t in r['tool_calls']]}")
        print(f"  retrieved={len(r['retrieved_citations'])} cites；answer_cites={len(r['answer_citations'])}；grounding={r['grounding']}")
        print(f"  答案前 160 字：{(r['final_answer'] or '')[:160].replace(chr(10),' ')}")
    g = results["T3_guard"]
    print(f"\n[T3_guard 承重墙活检] 幻觉名→{g['hallucinated_name']['path']}/{g['hallucinated_name']['reason']}；"
          f"错参数→{g['bad_args']['path']}/{g['bad_args']['reason']}；业务失败→{g['business_fail']['path']}/{g['business_fail']['reason']}")
    print(f"\n详细结果已存 {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
