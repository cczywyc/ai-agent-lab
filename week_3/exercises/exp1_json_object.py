"""
实验一：千问 json_object 模式验证
=================================
验证 response_format={"type": "json_object"} 的行为：
  - 返回是否是合法 JSON
  - 有 system prompt 描述 JSON 结构 vs 无描述，输出稳定性差异
"""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "search_agent"))

from dotenv import load_dotenv
from openai import OpenAI

# 加载项目根目录的 .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = "qwen-plus"


def test_json_object(label: str, system_prompt: str, user_prompt: str):
    """发送一次请求并检查返回是否为合法 JSON"""
    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"System: {system_prompt[:80]}...")
    print(f"User:   {user_prompt}")
    print(f"{'='*60}")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        print(f"\n原始输出:\n{raw}")

        # 尝试解析 JSON
        parsed = json.loads(raw)
        print(f"\n✅ JSON 解析成功")
        print(f"解析结果: {json.dumps(parsed, ensure_ascii=False, indent=2)}")
        print(f"顶层 keys: {list(parsed.keys())}")
        return {"success": True, "parsed": parsed, "raw": raw}

    except json.JSONDecodeError as e:
        print(f"\n❌ JSON 解析失败: {e}")
        return {"success": False, "error": str(e), "raw": raw}

    except Exception as e:
        print(f"\n❌ API 调用失败: {e}")
        return {"success": False, "error": str(e), "raw": None}


# ===================================================================
# 测试用例
# ===================================================================

if __name__ == "__main__":
    results = {}

    # --- 测试 1: 有详细 JSON 结构描述的 system prompt ---
    results["with_schema_desc"] = test_json_object(
        label="有结构描述的 system prompt",
        system_prompt=(
            "你是一个信息提取助手。请以 JSON 格式返回结果。"
            "JSON 结构如下：\n"
            '{"answer": "回答内容", "confidence": "high/medium/low", "sources": ["来源1", "来源2"]}'
        ),
        user_prompt="Python 3.12 有哪些主要新特性？",
    )

    # --- 测试 2: 只说"返回 JSON"，不描述具体结构 ---
    results["minimal_desc"] = test_json_object(
        label="最简描述 - 只说返回 JSON",
        system_prompt="你是一个信息助手。请以 JSON 格式返回你的回答。",
        user_prompt="Python 3.12 有哪些主要新特性？",
    )

    # --- 测试 3: system prompt 完全不提 JSON ---
    results["no_json_mention"] = test_json_object(
        label="system prompt 完全不提 JSON",
        system_prompt="你是一个有用的助手。",
        user_prompt="Python 3.12 有哪些主要新特性？",
    )

    # --- 测试 4: 中文内容 + 嵌套结构 ---
    results["chinese_nested"] = test_json_object(
        label="中文内容 + 嵌套结构",
        system_prompt=(
            "你是一个技术分析助手。请以 JSON 格式返回，结构如下：\n"
            '{"topic": "主题", "summary": "摘要", "details": [{"point": "要点", "explanation": "说明"}]}'
        ),
        user_prompt="简要分析 RAG 技术的优缺点",
    )

    # --- 汇总 ---
    print(f"\n\n{'='*60}")
    print("📊 汇总")
    print(f"{'='*60}")
    for name, r in results.items():
        status = "✅" if r["success"] else "❌"
        keys = list(r["parsed"].keys()) if r["success"] else "N/A"
        print(f"  {status} {name:20s} | 顶层 keys: {keys}")
