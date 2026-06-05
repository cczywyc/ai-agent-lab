# AI-Agent Learning Roadmap

大纲：我正在全面系统地学习如何构建 AI Agent 系统

目标：成为资深 AI Agent 领域应用开发工程师、架构设计者、工程落地者

以下是我计划的为期 12 周的学习计划的 roadmap



## 第 1 周：建立 Agent 基本认知

这周只做一件事：把“AI Agent 到底是什么”搞清楚。重点理解 Agent 的组成：模型、指令、工具、记忆、环境、控制循环。Microsoft 的 `AI Agents for Beginners` 很适合当第一门课，它把基础概念拆成结构化 lessons；OpenAI 的 learning track 则更偏产品与工程视角。

这周你要完成的内容：

- 通读 Microsoft 课程中的 introduction / fundamentals 相关内容
- 看一遍 OpenAI 的 building agents 概览
- 写一篇自己的总结：`Agent / Chatbot / Workflow / RAG 的区别`

这周的结果不是写代码，而是你能回答这些问题：

- 什么情况下该用 Agent，而不是固定工作流
- Agent 的“行动能力”来自哪里
- 为什么光有 LLM 不等于有 Agent

------

## 第 2 周：工具调用与单 Agent 雏形

从这周开始上手代码。核心是学会 **tool calling**，因为没有工具，Agent 只是会聊天；有了工具，它才会“做事”。OpenAI 的 Agents 文档和 Anthropic 的工程建议都把工具设计放在很核心的位置。

这周学习内容：

- function/tool schema 如何定义
- 模型如何选择工具
- 工具返回值怎么设计，才能让模型更稳定地用
- 常见失败：参数不全、结果为空、格式错乱

实战任务：

- 写一个“网页搜索 + 总结”的最小 Agent
- 至少做 2 个工具：搜索工具、网页内容提取工具
- 加上失败兜底逻辑：搜索不到怎么办、网页打不开怎么办

本周产出：

- 一个最小可运行的单 Agent demo
- 一份你自己的《Tool 设计规范》

------

## 第 3 周：提示词、结构化输出、控制循环

这周你会发现，很多 Agent 不稳定，并不是框架问题，而是 prompt、输出约束和执行循环设计得不够好。OpenAI 的实战指南提到，复杂度上升时，prompt template 和 typed output 往往比“不断加新 prompt”更可维护。

这周学习内容：

- system prompt 怎么约束角色
- structured output / JSON schema
- 基础 loop：观察 → 判断 → 调工具 → 继续/结束
- 重试和最大迭代次数控制

实战任务：

- 升级上周的 Agent
- 让它输出固定 JSON
- 支持 3 轮以内自动推理和工具调用
- 打印每轮 trace

本周产出：

- 一个“受控单 Agent”
- 一份《Agent Loop 设计笔记》

------

## 第 4 周：RAG 基础

从这周开始，你要把 Agent 和“外部知识”接起来。绝大多数有业务价值的 Agent，最终都会落到 RAG 或者知识检索上。你给的资源仓库里也把 RAG 相关框架和教程放在很重要的位置。

这周学习内容：

- chunking、embedding、top-k 检索
- 为什么只检索不够，还需要 rerank 或引用约束
- RAG 和“把整个文档塞进上下文”的区别

实战任务：

- 做一个“本地文档问答 Agent”
- 文档可以用你熟悉的技术笔记、项目设计文档
- 回答里强制带引用位置
- 用户追问时支持再次检索

本周产出：

- 一个最小 RAG Agent
- 一份《RAG 失败模式记录》，例如召回不准、引用错位、过度总结

------

## 第 5 周：记忆系统与上下文工程

这周重点是理解：**RAG 不是记忆，聊天历史也不是长期记忆。** Microsoft 课程和 Anthropic 关于 context engineering 的文章都在强调，真正的 Agent 能力，往往取决于你如何管理上下文，而不是模型自己“记住了什么”。

这周学习内容：

- 短期记忆：对话历史、工作区状态
- 长期记忆：用户偏好、任务事实、压缩摘要
- context engineering：什么该进 prompt，什么该进 memory store，什么该丢掉

实战任务：

- 给第 4 周的 RAG Agent 增加 memory
- 让它记住：
  - 用户关注的主题
  - 上一轮已确认的事实
  - 常用回答风格或格式偏好
- 加一个“记忆摘要”模块，避免上下文无限膨胀

本周产出：

- 一个带短期/长期记忆雏形的 Agent
- 一份《上下文装配策略图》

------

## 第 6 周：正式接 LangGraph，学状态化工作流

到了这周，你应该从“循环式单 Agent”升级到“可视化的状态流思维”。LangGraph 的价值不只是会画图，而是让你理解 Agent 本质上是一个 **有状态、可恢复、可插入人工节点的工作流系统**。这也是当前工程上非常重要的一条线。

这周学习内容：

- state、node、edge 的基本概念
- router / planner / executor / reviewer 的拆分方式
- checkpoint 与人工介入点

实战任务：

- 用 LangGraph 或等价状态图方式重写前面的 Agent
- 把流程拆成：
  - 任务理解
  - 检索/工具调用
  - 结果整理
  - 最终输出
- 保留中间状态和错误日志

本周产出：

- 第一个“状态化 Agent 工作流”
- 一张你自己画的 Agent 状态图

------

## 第 7 周：多步骤任务规划

这一周开始学 planning，但注意不是做“超级自由”的 Agent，而是学会让 Agent 先拆任务、再执行。Anthropic 的经验里反复强调，很多场景下，清晰的 planner-executor 模式比自由对话式多 Agent 更稳。

这周学习内容：

- planner-executor 模式
- decomposition：任务拆解
- reflection / critic：让 Agent 自检
- 终止条件和失败回滚

实战任务：

- 做一个“技术调研 Agent”
- 输入一个问题，先输出研究计划
- 再按步骤检索和总结
- 最后生成结构化报告

本周产出：

- 一个 research-style workflow agent
- 一份《规划器 vs 执行器职责边界》

------

## 第 8 周：多 Agent 入门

这周再进入多 Agent，就不会飘。OpenAI Agents SDK 支持 handoff，AutoGen 和 CrewAI 都提供了多 Agent 协作模式，但官方和工程实践都提醒你：不是越多 agent 越好。

这周学习内容：

- handoff 是什么
- centralized orchestration 和 agent-to-agent conversation 的区别
- 角色拆分原则：planner / researcher / writer / reviewer

实战任务：

- 做一个 3 角色系统：
  - researcher：查资料
  - writer：写初稿
  - reviewer：审阅和打回
- 明确每个角色的输入输出格式

本周产出：

- 第一个多 Agent demo
- 一份《什么时候该用多 Agent，什么时候不要用》

------

## 第 9 周：MCP 与外部系统接入

这周很关键，因为你会开始接真实世界。MCP 是现在很重要的开放协议方向，目的是把工具、资源和提示以统一方式暴露给 AI 应用。它不是某个框架的附属品，而是正在成为 Agent 生态的通用接口层。

这周学习内容：

- MCP 的基本概念：host、client、server
- tools / resources 的暴露方式
- 为什么 MCP 对企业集成有意义

实战任务：

- 写一个简单 MCP server
- 暴露两个工具：
  - 本地文件搜索
  - HTTP API 调用
- 再让你的 Agent 通过 MCP 调用它们

本周产出：

- 第一个 MCP 工具服务
- 一份《MCP 适合接哪些企业系统》的总结

------

## 第 10 周：评测与可观测性

从这周开始，重点不再是“能跑通”，而是“怎么知道它做得好不好”。OpenAI Agents 文档里把 eval、datasets、trace grading 放得很核心，因为不做评测的 Agent，很难真正迭代。

这周学习内容：

- success rate、tool accuracy、citation accuracy、latency
- 样例集怎么设计
- trace 怎么看
- 如何定位失败来自 prompt、工具还是检索

实战任务：

- 给你前面的一个 Agent 补 20 条测试样例
- 记录：
  - 成功/失败
  - 哪一步失败
  - 工具调用是否正确
  - 是否出现幻觉或错误引用

本周产出：

- 第一个 eval dataset
- 一份《Agent 失败归因表》

------

## 第 11 周：安全与治理

这周要补的是很多人会忽略但生产环境里必须做的部分：prompt injection、越权工具调用、恶意文档污染、敏感信息泄露。OpenAI 官方已经专门给出 Agent Builder Safety 和 prompt injection 相关指导。

这周学习内容：

- prompt injection 基本攻击方式
- 工具白名单 / 参数校验 / 权限隔离
- human approval gate
- 高风险操作的二次确认

实战任务：

- 给你的 Agent 加安全层：
  - 工具调用白名单
  - 高危动作确认
  - 文档来源可信度标签
- 手工构造 10 条攻击样例测试

本周产出：

- 一份《Agent 安全清单》
- 一个带 guardrails 的 Agent

------

## 第 12 周：整合成作品集项目

最后一周，把前 11 周的内容整合成一个真正能拿出去展示的项目。OpenAI 的实战指南明确建议，生产级 Agent 项目要能说明 use case、工具设计、guardrails、复杂度控制和评测结果，而不只是放个 demo 视频。

建议你做这两个方向之一：

### 方向 A：技术知识与诊断 Agent

很适合你当前的工程背景，比如：

- 运维日志分析 Agent
- 网络配置解释与诊断 Agent
- SQL 风险分析 Agent

### 方向 B：研发效率 Agent

比如：

- 需求分析 → 方案输出 → 代码脚手架生成
- 面试辅导 Agent
- 文档分析与知识库问答 Agent

最终项目必须包含：

- 工具调用
- 检索
- 记忆
- 多步骤 workflow
- 基础 eval
- 安全约束
- 架构图和 README



## 每周学习节奏

* 周一到周二：看概念和官方文档。
*  周三：跑官方示例。
*  周四到周五：改成自己的 demo。
*  周六：写总结和复盘。
*  周日：补评测和整理代码。

## 重点打磨的技能

1. 能设计工具，而不是只会调用工具。
2. 能控制上下文，而不是只拼接 prompt。
3. 能把任务做成状态化 workflow，而不是只写 while loop。
4. 能做基础评测和 trace，而不是靠感觉说“效果还行”。
5. 能讲清楚安全边界和复杂度取舍，而不是一上来就堆多 Agent
