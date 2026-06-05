"""
migrate_ltm.py — 一次性迁移：json 三件套 → LangGraph SqliteStore（计划 §四 阶段三-11）

把 v4.0 攒下的长期记忆数据迁进 store：
  memory_preferences.json → ("ltm", "preferences")   index=False
  memory_facts.json       → ("ltm", "facts")          store 重新 embed 建索引（真实 API）
  memory_topics.json      → ("ltm", "topics")         index=False

幂等：facts 的 key = sha1(原文)，重跑不产生重复；prefs/topics 同 key 覆盖。
原 json / 旧向量库文件一律保留不删（对照与回滚用）。

跑法（cwd = week_6/search_agent，需要 DASHSCOPE_API_KEY）：
  ../../.venv/bin/python migrate_ltm.py
"""

import json
from pathlib import Path

from config import (
    MEMORY_DIR, MEMORY_PREFS_FILE, MEMORY_FACTS_FILE, MEMORY_TOPICS_FILE,
)
from memory import long_term
from memory.long_term import Fact
from memory.ltm_store import get_ltm_store, LTM_DB_PATH


def _load_json(name: str, default):
    p = Path(MEMORY_DIR) / name
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    prefs: dict = _load_json(MEMORY_PREFS_FILE, {})
    facts: list = _load_json(MEMORY_FACTS_FILE, [])
    topics: dict = _load_json(MEMORY_TOPICS_FILE, {})

    print("=" * 60)
    print("  v4.0 json → v4.1 SqliteStore 长期记忆迁移")
    print("=" * 60)
    print(f"  源数据: {len(prefs)} 偏好 / {len(facts)} 事实 / {len(topics)} 主题")
    print(f"  目标库: {LTM_DB_PATH}")

    store = get_ltm_store()

    # ---- 偏好（index=False，同 key 覆盖）----
    for k, v in prefs.items():
        long_term.set_preference(store, k, v)

    # ---- 事实（store.batch 批量 put，内部重新 embed 建语义索引）----
    fact_objs = [Fact(**d) for d in facts]
    unique_keys = {long_term.fact_key(f.fact) for f in fact_objs}
    if len(unique_keys) < len(fact_objs):
        print(f"  [!] 注意: {len(fact_objs)} 条事实中有重复原文，"
              f"按 key 去重后为 {len(unique_keys)} 条")
    long_term.add_facts(store, fact_objs)

    # ---- 主题（直接 put 计数终值，不走 bump 的 +1 语义）----
    for t, n in topics.items():
        store.put(long_term.NS_TOPICS, t, {"count": int(n)}, index=False)

    # ---- 比对条数（计划：迁移后比对，别丢数据）----
    c = long_term.counts(store)
    print(f"\n  迁移后 store 计数: {c['preferences']} 偏好 / "
          f"{c['facts']} 事实 / {c['topics']} 主题")

    ok = (
        c["preferences"] == len(prefs)
        and c["facts"] == len(unique_keys)
        and c["topics"] == len(topics)
    )
    if ok:
        print("  ✅ 条数比对通过（json 三件套保留在原地，未删除）")
    else:
        print(f"  ❌ 条数不符！源: {len(prefs)}/{len(unique_keys)}/{len(topics)}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
