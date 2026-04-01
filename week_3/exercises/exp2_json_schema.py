"""
实验二：千问 json_schema 模式验证
=================================
验证 response_format={"type": "json_schema", ...} + strict: True 的行为：
  - enum 值是否被严格约束
  - required 字段是否都有
  - 中文内容在 JSON 中的编码是否正常
  - 嵌套对象和数组是否正确生成
"""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "search_agent"))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = "qwen-plus"


# ===================================================================
# 周二设计的 answer/sources/confidence 结构
# ===================================================================

AGENT_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "对用户问题的完整回答",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "信息来源列表（URL 或文档名称）",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "回答的置信度",
                },
            },
            "required": ["answer", "sources", "confidence"],
            "additionalProperties": False,
        },
    },
}


# ===================================================================
# 更复杂的 schema：嵌套对象 + 数组
# ===================================================================

DETAILED_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "detailed_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "分析主题",
                },
                "summary": {
                    "type": "string",
                    "description": "一句话总结",
                },
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "importance": {
                                "type": "string",
                                "enum": ["critical", "important", "nice_to_have"],
                            },
                        },
                        "required": ["title", "description", "importance"],
                        "additionalProperties": False,
                    },
                    "description": "分析要点列表",
                },
                "conclusion": {
                    "type": "string",
                    "description": "结论",
                },
            },
            "required": ["topic", "summary", "points", "conclusion"],
            "additionalProperties": False,
        },
    },
}


def test_json_schema(label: str, system_prompt: str, user_prompt: str, response_format: dict):
    """发送请求并验证输出是否严格符合 schema"""
    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"Schema: {response_format['json_schema']['name']}")
    print(f"{'='*60}")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )

        raw = resp.choices[0].message.content
        print(f"\n原始输出:\n{raw}")

        parsed = json.loads(raw)
        print(f"\n✅ JSON 解析成功")

        # 逐项验证
        schema_props = response_format["json_schema"]["schema"]["properties"]
        required_fields = response_format["json_schema"]["schema"]["required"]

        print(f"\n--- 字段检查 ---")

        # 检查 required 字段
        for field in required_fields:
            present = field in parsed
            print(f"  required '{field}': {'✅ 存在' if present else '❌ 缺失'}")

        # 检查 enum 约束
        for field, prop in schema_props.items():
            if "enum" in prop and field in parsed:
                valid = parsed[field] in prop["enum"]
                print(f"  enum '{field}': 值='{parsed[field]}' {'✅ 合法' if valid else '❌ 不在 enum 中'} (允许值: {prop['enum']})")

        # 检查嵌套数组中的 enum
        if "points" in parsed and isinstance(parsed["points"], list):
            print(f"  数组 'points': 共 {len(parsed['points'])} 项")
            items_schema = schema_props.get("points", {}).get("items", {})
            items_props = items_schema.get("properties", {})
            for i, item in enumerate(parsed["points"]):
                for f, p in items_props.items():
                    if "enum" in p and f in item:
                        valid = item[f] in p["enum"]
                        print(f"    points[{i}].{f}: '{item[f]}' {'✅' if valid else '❌'}")

        # 检查是否有多余字段 (additionalProperties: false)
        extra_keys = set(parsed.keys()) - set(schema_props.keys())
        if extra_keys:
            print(f"  ❌ 多余字段: {extra_keys}")
        else:
            print(f"  ✅ 无多余字段")

        # 检查中文编码
        answer_field = parsed.get("answer") or parsed.get("summary") or ""
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in answer_field)
        print(f"  中文内容: {'✅ 正常' if has_chinese else '⚠️ 无中文（可能是英文回答）'}")

        print(f"\n完整解析结果:\n{json.dumps(parsed, ensure_ascii=False, indent=2)}")
        return {"success": True, "parsed": parsed}

    except json.JSONDecodeError as e:
        print(f"\n❌ JSON 解析失败: {e}")
        print(f"原始输出: {raw}")
        return {"success": False, "error": str(e)}

    except Exception as e:
        print(f"\n❌ API 调用失败: {type(e).__name__}: {e}")
        return {"success": False, "error": str(e)}


# ===================================================================
# 测试用例
# ===================================================================

if __name__ == "__main__":
    results = {}

    # --- 测试 1: 基础 schema (answer/sources/confidence) ---
    results["basic_schema"] = test_json_schema(
        label="基础 schema - answer/sources/confidence",
        system_prompt="你是一个搜索助手。根据你的知识回答问题，并评估置信度。",
        user_prompt="什么是 prompt engineering？",
        response_format=AGENT_RESPONSE_SCHEMA,
    )

    # --- 测试 2: 基础 schema + 中文回答 ---
    results["basic_chinese"] = test_json_schema(
        label="基础 schema - 测试中文内容",
        system_prompt="你是一个中文技术助手。用中文回答问题。",
        user_prompt="解释一下什么是 Agent Loop（智能体循环）",
        response_format=AGENT_RESPONSE_SCHEMA,
    )

    # --- 测试 3: 复杂嵌套 schema ---
    results["nested_schema"] = test_json_schema(
        label="嵌套 schema - detailed_analysis",
        system_prompt="你是一个技术分析师。用中文进行详细分析。",
        user_prompt="分析 structured output 在 AI Agent 中的应用价值",
        response_format=DETAILED_ANALYSIS_SCHEMA,
    )

    # --- 测试 4: 故意诱导违反 enum ---
    results["enum_stress"] = test_json_schema(
        label="enum 压力测试 - 诱导非法值",
        system_prompt=(
            "你是一个助手。注意：confidence 必须是 high/medium/low 之一，"
            "不要用其他值如 'very high' 或 '非常高'。"
        ),
        user_prompt="1+1 等于几？这个你应该非常非常确定。",
        response_format=AGENT_RESPONSE_SCHEMA,
    )

    # --- 汇总 ---
    print(f"\n\n{'='*60}")
    print("📊 汇总")
    print(f"{'='*60}")
    for name, r in results.items():
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {name}")
