"""
摘要模块 — 分层压缩被裁的旧轮次

对应设计文档"设计三"：
  - 触发：token 主闸门 + 每 N 轮兜底 + 显式信号
  - 分层压缩：
    1. 被裁旧轮次先抽出「已确认事实」晋升到长期记忆（高保真，带 chunk 引用）
    2. 剩下过程性对话压成一句话级低保真摘要

实现：
  - 事实抽取走规则（extractor.extract_fact_candidates）——零成本
  - 一句话摘要走模型——但只对 evict 的旧轮调用一次，不每轮调
  - 摘要内容存到 ShortTermMemory.summary_buffer（其实持有方是 MemoryManager）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from config import client, MODEL, MEMORY_DIR, MEMORY_SUMMARY_FILE
from .extractor import extract_fact_candidates
from .short_term import ConversationTurn

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = """
你是一个对话摘要工具。把给定的若干轮对话压成「不超过 80 字」的中文摘要。
要求：
- 只记录最关键的话题流转和未解决的待办，不要复述完整问答
- 不要带任何引用、链接或工具名
- 用陈述句、不用 markdown 格式
""".strip()


class MemorySummarizer:
    """对 evict 的旧轮做事实晋升 + 一句话摘要。"""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir or MEMORY_DIR)
        self.summary_text: str = ""  # 累积摘要，每次新摘要追加在前

    # ============================================================
    # 高保真：从 evict 的回答里抽事实
    # ============================================================
    def extract_promoted_facts(
        self, turns: list[ConversationTurn]
    ) -> list[tuple[str, str, int]]:
        """
        从被裁的旧轮中，把含引用的事实抽出来。

        Returns:
            [(fact_text, source, turn_number), ...]
        """
        out: list[tuple[str, str, int]] = []
        for t in turns:
            for fact, source in extract_fact_candidates(t.assistant_message):
                out.append((fact, source, t.turn_number))
        return out

    # ============================================================
    # 低保真：一句话摘要
    # ============================================================
    def summarize_low_fidelity(
        self, turns: list[ConversationTurn]
    ) -> str:
        """对被裁旧轮调一次模型，产出一句话摘要。"""
        if not turns:
            return ""

        # 拼接对话简化文本
        lines = []
        for t in turns:
            lines.append(f"[第{t.turn_number}轮]")
            lines.append(f"用户: {t.user_message[:200]}")
            if t.tool_summaries:
                lines.append(f"工具: {'; '.join(t.tool_summaries)[:200]}")
            # 助手回答取前 300 字（事实已被前一步抽走了）
            lines.append(f"助手: {t.assistant_message[:300]}")
        convo = "\n".join(lines)

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": convo},
                ],
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Summary LLM call failed: {e}")
            # 兜底：粗略拼用户问题
            qs = "; ".join(t.user_message[:30] for t in turns)
            return f"[摘要降级] 用户问过：{qs}"

    # ============================================================
    # 组合摘要
    # ============================================================
    def append_summary(self, new_summary: str) -> None:
        """把新摘要追加到累积摘要里（保留有限长度）。"""
        if not new_summary:
            return
        if self.summary_text:
            self.summary_text = f"{self.summary_text}\n\n{new_summary}"
        else:
            self.summary_text = new_summary
        # 上限：保留最后 ~1200 字符（防摘要本身膨胀）
        if len(self.summary_text) > 1200:
            # 保留尾部，丢弃头部
            self.summary_text = "...\n" + self.summary_text[-1200:]

    # ============================================================
    # 持久化
    # ============================================================
    @property
    def _path(self) -> Path:
        return self.persist_dir / MEMORY_SUMMARY_FILE

    def save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary_text}, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.summary_text = data.get("summary", "")

    def clear(self) -> None:
        self.summary_text = ""
        if self._path.exists():
            self._path.unlink()
