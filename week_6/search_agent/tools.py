"""
搜索 Agent 工具集 — v3.0

工具清单：
  1. web_search        — 搜索引擎查询（Data Tool，只读）
  2. fetch_webpage     — 网页正文提取（Data Tool，只读）
  3. retrieve_documents — 本地向量库语义检索【v3.0 新增】

v3.0 变更：
  - 新增 retrieve_documents 工具及其定义
  - 工具描述里突出"本地优先"
"""

import json
from urllib.parse import urlparse

from config import BLOCKED_DOMAINS, RETRIEVE_TOP_K, RETRIEVE_MIN_SCORE


# ============================================================
# 工具 1：web_search
# ============================================================

def web_search(query: str, max_results: int = 5) -> dict:
    """使用 DuckDuckGo 搜索引擎查询，返回精简的 title/url/snippet 列表。"""
    try:
        from ddgs import DDGS

        with DDGS(proxy=None) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        if not raw_results:
            return {
                "results": [],
                "total": 0,
                "message": f"No results found for '{query}'.",
                "suggestion": "Try different keywords or a broader search term.",
            }

        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": (r.get("body", r.get("snippet", "")))[:200],
            })

        return {"results": results, "total": len(results), "query": query}

    except Exception as e:
        return {
            "error": True,
            "error_type": "SearchError",
            "message": f"Search failed: {str(e)}",
            "recoverable": True,
            "suggestion": "Try a simpler or different query.",
        }


# ============================================================
# 工具 2：fetch_webpage
# ============================================================

def fetch_webpage(url: str) -> dict:
    """获取网页正文：URL 黑名单预过滤 + readability 提取 + 截断 3000 字符。"""
    try:
        domain = urlparse(url).netloc.lower()
        if any(blocked in domain for blocked in BLOCKED_DOMAINS):
            return {
                "error": True,
                "error_type": "BlockedDomain",
                "message": f"'{domain}' is known to block automated access.",
                "recoverable": True,
                "suggestion": "Try a different URL from the search results, or answer based on available snippets.",
            }
    except Exception:
        pass

    try:
        import httpx

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        with httpx.Client(timeout=10, follow_redirects=True) as http_client:
            resp = http_client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        try:
            from readability import Document
            import re
            doc = Document(html)
            title = doc.title()
            content = re.sub(r"<[^>]+>", "", doc.summary())
            content = re.sub(r"\s+", " ", content).strip()
        except ImportError:
            import re
            content = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.S)
            content = re.sub(r"<[^>]+>", "", content)
            content = re.sub(r"\s+", " ", content).strip()
            title = ""

        max_len = 3000
        truncated = len(content) > max_len
        content = content[:max_len]

        if not content or len(content) < 50:
            return {
                "error": True,
                "error_type": "EmptyContent",
                "message": f"Page at '{url}' returned no meaningful content.",
                "recoverable": True,
                "suggestion": "Try a different URL from the search results.",
            }

        return {
            "url": url,
            "title": title,
            "content": content,
            "truncated": truncated,
            "char_count": len(content),
        }

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        error_type = {
            403: "Forbidden",
            404: "NotFound",
            429: "RateLimited",
        }.get(status, "HTTPError")

        return {
            "error": True,
            "error_type": error_type,
            "message": f"HTTP {status} when fetching '{url}'.",
            "recoverable": status != 404,
            "suggestion": "Try a different URL from the search results.",
        }

    except httpx.TimeoutException:
        return {
            "error": True,
            "error_type": "Timeout",
            "message": f"Request timed out for '{url}'.",
            "recoverable": True,
            "suggestion": "Try a different URL, or answer based on available snippets.",
        }

    except Exception as e:
        return {
            "error": True,
            "error_type": "FetchError",
            "message": f"Failed to fetch '{url}': {str(e)}",
            "recoverable": True,
            "suggestion": "Try a different URL from the search results.",
        }


# ============================================================
# 工具 3：retrieve_documents — v3.0 新增
# ============================================================

def retrieve_documents(query: str, top_k: int = RETRIEVE_TOP_K) -> dict:
    """
    在本地向量库做语义检索，返回 chunks（含来源元数据）。

    返回结构：
        {
          "results": [
            {"doc": str, "section": str, "chunk_id": int,
             "text": str, "path": str, "score": float},
            ...
          ],
          "total": int,
          "query": str,
        }

    失败时返回统一的 error 字典。
    """
    # 延迟 import，避免 numpy 未装时 main.py 起不来
    try:
        from rag.retriever import get_retriever
    except Exception as e:
        return {
            "error": True,
            "error_type": "RAGImportError",
            "message": f"Failed to import RAG module: {e}",
            "recoverable": False,
            "suggestion": "Check that numpy is installed in the venv.",
        }

    try:
        retriever = get_retriever(namespace="docs")
    except Exception as e:
        return {
            "error": True,
            "error_type": "RetrieverInitError",
            "message": f"Retriever init failed: {e}",
            "recoverable": False,
        }

    if not retriever.is_ready():
        return {
            "error": True,
            "error_type": "VectorStoreNotReady",
            "message": "Local vector store is empty or not built.",
            "recoverable": False,
            "suggestion": "Build the index first: `python main.py --ingest`.",
        }

    try:
        hits = retriever.retrieve(query, top_k=top_k)
    except Exception as e:
        return {
            "error": True,
            "error_type": "RetrievalError",
            "message": f"Retrieval failed: {e}",
            "recoverable": True,
            "suggestion": "Try a different query or check embedding API connectivity.",
        }

    if not hits:
        return {
            "results": [],
            "total": 0,
            "query": query,
            "message": "No relevant documents found in local knowledge base.",
            "suggestion": "The topic may not be covered locally; consider web_search.",
        }

    return {
        "results": hits,
        "total": len(hits),
        "query": query,
    }


# ============================================================
# 工具定义
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Semantic search over the LOCAL knowledge base "
                "(user's own notes, design docs, and project retrospectives). "
                "**Use this FIRST** whenever the question is about this project's "
                "internals: Agent Loop design, search_agent versions, RAG plans, "
                "memory system, Tool design, weekly reviews, or anything the user "
                "wrote themselves. Returns up to top_k chunks, each with "
                "`doc`, `section`, `chunk_id`, `text`, and a relevance `score`. "
                "**Cite returned chunks as `[doc#section]` in your answer.**"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The question or key sentence to retrieve against. "
                            "Both Chinese and English are fine; no keyword "
                            "extraction needed."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "How many chunks to return. Default 5; use 3 for "
                            "specific facts, 8 for broader context."
                        ),
                        "minimum": 1,
                        "maximum": 15,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using a search engine. Returns up to 5 results, "
                "each with title, URL, and a brief snippet (max 200 chars). "
                "Use this when the question is NOT about the user's own local notes "
                "(otherwise prefer retrieve_documents). "
                "Use concise English keywords for best results (2-5 words). "
                "Does NOT fetch full page content — use fetch_webpage for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query in concise English keywords. "
                            "Example: 'MCP Model Context Protocol'. 2-5 words works best."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Fetch and extract the main text content from a webpage URL. "
                "Use this AFTER web_search when you need more detailed content "
                "than the search snippets provide. "
                "Returns the article body text, truncated to 3000 characters. "
                "Some websites may block access (403 error) — if this happens, "
                "try a different URL from the search results. "
                "Do NOT guess URLs; only use URLs returned by web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The full URL to fetch. Must be a URL from "
                            "web_search results."
                        ),
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]

# ============================================================
# 工具注册表
# ============================================================

TOOL_REGISTRY = {
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "retrieve_documents": retrieve_documents,
}


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """统一工具执行入口。"""
    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        return {
            "error": True,
            "error_type": "UnknownTool",
            "message": f"Tool '{tool_name}' is not registered.",
            "recoverable": False,
        }

    try:
        return func(**tool_args)
    except TypeError as e:
        return {
            "error": True,
            "error_type": "InvalidArguments",
            "message": f"Invalid arguments for '{tool_name}': {str(e)}",
            "recoverable": True,
            "suggestion": "Check the parameter names and types.",
        }
    except Exception as e:
        return {
            "error": True,
            "error_type": "ToolExecutionError",
            "message": f"Tool '{tool_name}' failed: {str(e)}",
            "recoverable": False,
        }
