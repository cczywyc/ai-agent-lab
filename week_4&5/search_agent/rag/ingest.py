"""
文档摄入与 Chunking — 按 Markdown 标题层级切，超长再按段落滑窗

设计要点（对应设计文档 RAG-1 失败模式预警①）：
  - 优先按结构切：解析 markdown 标题，section 内部聚合
  - 超过 CHUNK_MAX_CHARS 才回退到段落级滑窗（带重叠）
  - 太短的 section 与相邻同层合并，避免一堆孤立标题成单独 chunk
  - 每个 chunk 带 {doc, section, chunk_id, text}，section 是 H1>H2>H3 路径
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from config import (
    CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS,
    DEFAULT_INGEST_DIRS, PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


# ============================================================
# Markdown 解析：切成 (heading_path, body_text) 段
# ============================================================

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _parse_markdown_sections(text: str) -> list[tuple[str, str]]:
    """
    把 markdown 切成 (heading_path, body) 段。

    heading_path 形如 "RAG-1 文档摄入与 Chunking > Chunking 策略"，
    用 " > " 拼接当前位置的所有上级标题。

    没有标题的开头段统一归到 "前言"。
    """
    lines = text.split("\n")
    heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]
    sections: list[tuple[str, list[str]]] = []
    current_body: list[str] = []
    current_path = "前言"

    def _path_from_stack() -> str:
        if not heading_stack:
            return "前言"
        return " > ".join(title for _, title in heading_stack)

    in_code_block = False

    for line in lines:
        # 跳过代码块内部的 #（防止把代码注释误判为标题）
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            current_body.append(line)
            continue
        if in_code_block:
            current_body.append(line)
            continue

        m = HEADING_RE.match(line)
        if m:
            # 提交当前段
            if current_body:
                sections.append((current_path, current_body))
                current_body = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # 弹出 ≥ 当前 level 的上级（保证栈是严格递增的）
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = _path_from_stack()
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_path, current_body))

    # 把 body list 合成字符串、去掉空白段
    out: list[tuple[str, str]] = []
    for path, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            out.append((path, body))
    return out


# ============================================================
# 把 (path, body) 段切成最终 chunks
# ============================================================

def _split_long_text(text: str, target: int, max_: int) -> list[str]:
    """
    超长段落滑窗切：优先按 \n\n 分段累加，超 max_ 再硬切。
    输出每块不超过 max_ 字符，相邻块带 ~target/4 字符重叠。
    """
    if len(text) <= max_:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        if len(buf) + len(p) + 2 <= target:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            # 留尾部最后 ~target/4 字符做重叠
            overlap_chars = max(0, target // 4)
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = (tail + "\n\n" + p) if tail else p
    if buf:
        chunks.append(buf)

    # 还有超长的单段（一整段 > max_），硬切
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_:
            final.append(c)
        else:
            for i in range(0, len(c), target):
                final.append(c[i:i + max_])
    return final


def chunk_markdown_file(path: Path, doc_name: str | None = None) -> list[dict]:
    """
    把一个 markdown 文件切成 chunks。

    Returns:
        [{doc, section, chunk_id, text, path}, ...]
        doc       = 文件名（不含扩展名，去掉路径）
        section   = 标题路径
        chunk_id  = 该文件内 0-based 序号
        text      = chunk 正文（带 "## 标题路径\\n\\n正文" 前缀，提升上下文）
        path      = 原文件相对项目根的路径（trace 用）
    """
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if not raw.strip():
        return []

    doc = doc_name or path.stem

    sections = _parse_markdown_sections(raw)

    # 先把太短的 section 合并到下一个
    merged: list[tuple[str, str]] = []
    pending: tuple[str, str] | None = None
    for path_, body in sections:
        if pending is not None:
            body = pending[1] + "\n\n" + body
            # 用更深的标题作为 section 路径（通常更精准）
            new_path = path_ if len(path_) >= len(pending[0]) else pending[0]
            pending = None
        else:
            new_path = path_

        if len(body) < CHUNK_MIN_CHARS:
            pending = (new_path, body)
        else:
            merged.append((new_path, body))

    if pending is not None:
        if merged:
            # 合并到上一个
            last_path, last_body = merged[-1]
            merged[-1] = (last_path, last_body + "\n\n" + pending[1])
        else:
            merged.append(pending)

    # 切长段
    chunks: list[dict] = []
    chunk_id = 0
    try:
        rel_path = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_path = str(path)

    for sec_path, body in merged:
        sub_texts = _split_long_text(body, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS)
        for sub in sub_texts:
            text = f"# {sec_path}\n\n{sub.strip()}"
            chunks.append({
                "doc": doc,
                "section": sec_path,
                "chunk_id": chunk_id,
                "text": text,
                "path": rel_path,
            })
            chunk_id += 1

    return chunks


# ============================================================
# 目录扫描
# ============================================================

def scan_markdown_files(
    dirs: Iterable[Path] | None = None,
    *,
    exclude_dirs: Iterable[str] = (".venv", "node_modules", "__pycache__", ".git"),
) -> list[Path]:
    """
    收集要摄入的 markdown 文件。

    默认扫描 config.DEFAULT_INGEST_DIRS，可传入自定义目录列表。
    """
    target_dirs = list(dirs) if dirs else DEFAULT_INGEST_DIRS
    exclude_set = set(exclude_dirs)

    files: list[Path] = []
    for d in target_dirs:
        d = Path(d)
        if not d.exists():
            logger.warning(f"Ingest dir not found, skip: {d}")
            continue
        for p in d.rglob("*.md"):
            if any(part in exclude_set for part in p.parts):
                continue
            files.append(p)

    # 去重排序
    files = sorted(set(files))
    return files


def ingest_all(dirs: Iterable[Path] | None = None) -> list[dict]:
    """
    扫描 + chunking，返回所有 chunk（未做 embedding，留给上层）。
    """
    files = scan_markdown_files(dirs)
    logger.info(f"Ingest: found {len(files)} markdown files")

    all_chunks: list[dict] = []
    for f in files:
        chunks = chunk_markdown_file(f)
        if chunks:
            logger.info(f"  {f.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        else:
            logger.warning(f"  {f.name}: 0 chunks (empty or unparseable)")

    return all_chunks
