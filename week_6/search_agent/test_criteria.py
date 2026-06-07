"""
判据测试 — 占位符识别 + --test 用例判定（第七周前重构项：判据重审）

  C1 is_placeholder_answer：占位符名单完整
     （含 [模型返回空回答]——06-05 复跑 Case 5 假阳性的根因）
  C2 judge_case：正常用例严判口径不变
  C3 judge_case：占位符回答 → has_answer=False（Case 5 假阳性回归），
     同时 passed_legacy 复现旧口径（报告可比性）
  C4 judge_case：expect_retrieve=None → 不判该维度（Case 8 边界编码），
     legacy_expect_retrieve 提供旧口径对照

跑法：../../.venv/bin/python test_criteria.py
"""

import sys

from config import PLACEHOLDER_PREFIXES, is_placeholder_answer
from main import judge_case

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"  {'✓ PASS' if cond else '✗ FAIL'}  {name}")
    return bool(cond)


# ============================================================
# C1 is_placeholder_answer
# ============================================================

def c1():
    print("\n[C1] is_placeholder_answer：占位符名单完整")
    check("空串是占位符", is_placeholder_answer(""))
    check("[模型返回空回答] 是占位符（Case 5 假阳性根因）",
          is_placeholder_answer("[模型返回空回答]"))
    check("[达到最大轮次] 是占位符",
          is_placeholder_answer("[达到最大轮次] Agent 在 6 轮内未能完成任务。"))
    check("[错误] 是占位符",
          is_placeholder_answer("[错误] 模型调用失败: timeout"))
    check("正常回答不是占位符",
          not is_placeholder_answer("MCP 是 Anthropic 提出的协议 [doc#section]"))
    check("以占位符文字开头但非前缀位置不误判",
          not is_placeholder_answer("关于 [错误] 这个词的解释如下"))
    check("名单恰好覆盖三个占位符前缀", len(PLACEHOLDER_PREFIXES) == 3)


# ============================================================
# C2–C4 judge_case
# ============================================================

def make_state(search=False, retrieve=False, answer="正常回答 [doc#sec]"):
    return {"has_searched": search, "has_retrieved": retrieve, "answer": answer}


def c2():
    print("\n[C2] judge_case：正常用例严判口径不变")
    case = {"expect_search": True, "expect_retrieve": False}

    r = judge_case(case, make_state(search=True))
    check("全符合 → passed", r["passed"])
    check("严判口径下新旧一致", r["passed"] == r["passed_legacy"])

    r = judge_case(case, make_state(search=False))
    check("search 不符 → FAIL", not r["passed"] and not r["search_correct"])

    r = judge_case(case, make_state(search=True, retrieve=True))
    check("retrieve 多调 → FAIL（严判仍严）", not r["passed"] and not r["retrieve_correct"])


def c3():
    print("\n[C3] judge_case：占位符回答不再是有效回答（Case 5 假阳性回归）")
    case = {"expect_search": False, "expect_retrieve": True}
    r = judge_case(case, make_state(retrieve=True, answer="[模型返回空回答]"))

    check("has_answer = False", not r["has_answer"])
    check("工具路径全对也判 FAIL", not r["passed"])
    check("旧口径假阳性被记录在 passed_legacy（=True）", r["passed_legacy"])


def c4():
    print("\n[C4] judge_case：expect_retrieve=None 不判该维度（Case 8 边界编码）")
    case = {"expect_search": True, "expect_retrieve": None,
            "legacy_expect_retrieve": False}

    r = judge_case(case, make_state(search=True, retrieve=True))
    check("先本地再联网 → PASS", r["passed"] and r["retrieve_correct"])
    check("该走法在旧口径下 FAIL（记录差异）", not r["passed_legacy"])

    r = judge_case(case, make_state(search=True, retrieve=False))
    check("直接联网 → PASS", r["passed"])
    check("该走法在旧口径下也 PASS", r["passed_legacy"])

    r = judge_case(case, make_state(search=False, retrieve=True))
    check("联网兜底没发生 → FAIL（必判维度仍严）", not r["passed"])


# ============================================================
# 入口
# ============================================================

def main():
    print("=== 判据测试（离线） ===")
    for t in (c1, c2, c3, c4):
        t()

    passed = sum(1 for _, ok in CHECKS if ok)
    total = len(CHECKS)
    print(f"\n{'=' * 40}\n结果: {passed}/{total} 项判据通过", end="")
    if passed == total:
        print(" —— 全部通过 ✓")
        sys.exit(0)
    print("\n失败项:")
    for name, ok in CHECKS:
        if not ok:
            print(f"  ✗ {name}")
    sys.exit(1)


if __name__ == "__main__":
    main()
