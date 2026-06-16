# MCP 概念

> 第九周·周一·概念精读日。今天读 MCP 官方 spec（2025-11-25 稳定版）的 architecture / server-tools / transports / 错误处理几章。目标不是把名词背下来，而是把每个概念对上第八周已经搭好的那套东西看——它到底改了我哪一块，又没改哪一块。
>
> 读之前心里揣着两个怀疑，是第八周《多Agent概念》§七结尾留下的。一个：MCP 标准化的是接口层（工具怎么暴露、发现、调用），那调用漂移、幻觉工具名、结果不接地要不要 retry、语境锚怎么不丢这些控制流，它到底管不管？另一个：typed 工具契约能不能把缺陷 #5 那个"幻觉 `edit` 工具名、靠 `UnknownTool` 运行时兜住"提前到发现期就拒掉——在 schema 层拒，而不是等到执行层报错？带着这两个怀疑读，别一上来就默认"接了 MCP 控制流就干净了"。
>
> 基线是第八周的 v6.0，一套 supervisor 多 Agent：supervisor 拆研究子任务，researcher 逐个检索、压缩 findings，writer 组织初稿，reviewer 用二元 rubric 审稿打回（`MAX_REVIEW=2`）。researcher 内层就是 v5.0 那套 `agent↔tools↔inject_*` 引擎原样复用，工具是手写注册表（`retrieve_documents` / `web_search` / `fetch_webpage` 硬编码），靠 `UnknownTool` 兜幻觉工具名；每个 worker 给一个显式隔离投影；Store/RAG 原样复用；闸门那套是 `replan_count` / `retry_count` / synthesis-reserve / 引用接地软闸门 / escalate / skip-and-advance。
>
> 还有个版本坐标得先记一下：今天以 2025-11-25 稳定版为准。2026-07-28 那个 RC（stateless core / MCP Apps / Tasks 扩展 / 对齐 OAuth-OIDC）是 launch 以来最大的一次改版，但还是 RC，本周不碰。传输只看 stdio 和 Streamable HTTP，老的 HTTP+SSE 已经被取代，不用读。

---

## 0. 读这一遍，我想弄清的就一件事

MCP 标准化的全是"接口层"：工具怎么描述、怎么发现、怎么调用，以及失败怎么报成 typed 信号。它不碰控制流。落到我手上的具体变化是，v6.0 researcher 那套手写工具注册表加 `UnknownTool` 运行时兜底，会被换成"发现来的 typed 契约 + 两层错误信道"；但调用失败怎么降级、引用不接地要不要 retry、子任务怎么 skip-and-advance、语境锚怎么不丢，这些还全在我自己的 loop 里。

这其实是第六周（LangGraph 给图原语）、第七八周（闸门和收口全得自己搭）那条线往下走的一步，没什么本质转折。只是这回多了个更准的注脚：MCP 连"失败长什么样"都帮我标了类型（协议错误 vs 业务错误），但"看到这个失败该做什么"始终是我的活。后面那几节——架构三件套、三原语、传输、工具生命周期、企业价值——都是为了把这个判断坐实才铺的。

---

## 一、先把地图画出来：host / client / server

后面所有概念都落在这三个角色上，先把坐标系建起来。

| 角色 | 是什么 | 干什么 | 对应 v6.0 的什么 |
|---|---|---|---|
| host | 装着 LLM 的那个应用（IDE、聊天端、我的 Agent 本体） | 持有上下文窗口、决定何时调工具、把用户消息路由给模型、把工具结果喂回对话，是对话控制器 | 我的整个 v6.0 进程（supervisor 那套图） |
| client | host 内部的连接器，跟一个 server 一对一 | 管一条到某 server 的连接：发 `tools/list`、`tools/call`，收通知 | 现在没有——v6.0 直接在进程内调函数，没有"连接"这一层 |
| server | 对外暴露 tools / resources / prompts 的独立进程 | 声明自己有哪些原语、执行被调用的工具、回传结果 | 现在也没有——我的工具是进程内硬编码，没"暴露"给谁 |

读到这儿有个地方容易和第八周搞混，得专门记一下：v6.0 里 supervisor 和 researcher 是"一个进程里的多个角色"，它们之间的边界只是逻辑角色；而 MCP 的 client 和 server 是"跨进程、跨网络的接口契约"，边界是实打实的进程或网络。所以"接 MCP"不是把 supervisor-researcher 改个名字，而是在 researcher 调工具的那一刀上插进一道进程边界——工具从"本进程里的一个函数"，变成"另一个进程暴露、由 client 发现并调用的契约"。

握手阶段有一步 capability negotiation：`initialize` 的时候 client 和 server 互相声明各自支持哪些原语、支不支持动态变更（`listChanged`）。这是后面"发现期"的起点。

---

## 二、三原语：tools / resources / prompts（外加 sampling / elicitation）

MCP 把"能喂给模型的东西"分成三类一等公民，每类都有标准的 list 和 get/call 方法。三者的区别不在功能，在"谁来控制"。

| 原语 | 是什么 | 谁控制 | 对应 v6.0 的什么 |
|---|---|---|---|
| tools | 可执行动作（查库、调 API、做计算） | 模型控制（model-controlled，模型自己判断要不要调） | `retrieve_documents` / `web_search` / `fetch_webpage` |
| resources | 只读数据（文件、文档、记录） | 应用/host 控制（host 决定把哪些塞进上下文） | RAG 文档库、Store 里的长期事实——现在是 researcher 主动检索，没单独区分"只读数据"这一类 |
| prompts | 可复用模板（slash 命令式） | 用户控制（用户显式挑选触发） | 我的 SYSTEM_PROMPT / 各角色提示——现在写死在代码里，没暴露成可挑选的模板 |

外加两个把 human/host 拉进环里的原语：sampling（server 反过来请 host 帮它跑一次 LLM 推理）和 elicitation（server 中途向用户要一个输入）。后者正好对上我第六周那个 `human_review` 节点，只是 MCP 把它标准化成了"server 可以主动发起的一个请求"。

v6.0 是把这三类东西全揉在一个进程里手写的：工具是函数、RAG 是检索调用、prompt 是字符串常量。MCP 的价值就是把它们拆成三类各有标准 list/call 协议的原语。这里有个选择我现在还没定（留到 §九 Q4）：我的 RAG 检索到底该做成 tool（researcher 主动去调）还是 resource（host 被动读进上下文）？这俩"谁控制"不一样，直接决定 researcher 怎么用它。

---

## 三、传输层：stdio vs Streamable HTTP

底层消息格式统一是 JSON-RPC 2.0，请求 / 响应 / 通知三种，所以同一套工具定义在本地和远程之间是可移植的。传输只有两个要学。

| 传输 | 场景 | 本周怎么用 |
|---|---|---|
| stdio | 本机、同机进程，走标准输入输出管道 | 周四实战先用它：在本机把 server 跑通、Agent 经 client 调它，调试最省事 |
| Streamable HTTP | 远程、多客户端、HTTPS（含 SSE 流式） | 企业部署才上：多并发、可横向扩、能接 OAuth 鉴权 |

有个坑提前记下：老的 HTTP+SSE 独立传输已经被 Streamable HTTP 取代了，别去读它——MCP 这种按日期打版的协议，最容易在这种地方白花半天。本周 demo 用 stdio 就够，传输本来也不是这周的重点，接口契约才是。

---

## 四、工具的生命周期：发现 → 调用 → 失败上报

两个怀疑的答案基本都从这一节长出来。一个工具从"存在"到"被调用、失败被处理"，在 MCP 里走三步。

**发现（`tools/list`）**：调用之前，client 先发 `tools/list`，server 回一份工具清单，每个工具带 name + description + `inputSchema`（JSON Schema），新版还能带 `outputSchema`。从这一刻起，有效工具集的权威来源就住在发现期了。如果 server 声明了 `listChanged` capability，工具表一变就发通知、client 重新拉，也就是工具表能动态同步，不像我现在硬编码死。

**调用（`tools/call`）**：tools 是 model-controlled，模型根据上下文自己决定调哪个、传什么参数。这里有一句要划重点：模型住在 host 这一侧、在 MCP 之上，它生成工具名那一刻，MCP 根本不在中间。这句是后面 Q2 的全部要害。

**失败上报**：分两层，泾渭分明。

| 层 | 触发 | 怎么报 | 语义 |
|---|---|---|---|
| 协议层错误 | 未知工具名、参数非法、server 崩溃 | 标准 JSON-RPC error 响应 | "集成本身有问题"，通常别盲目 retry |
| 执行层错误 | API 限流、输入数据无效、库连不上 | 装在 tool result 里，`isError: true` | "业务级失败"，模型可以据此推理、尝试 retry 或绕过 |

对回我已有的部件上：researcher 的手写注册表对应 `tools/list`；`UnknownTool` 兜底对应两层里的协议层 error；"引用不接地软闸门触发 retry"对应 `isError` 回来之后、我自己判要不要再来一次。MCP 把"失败长什么样"分了类、报成 typed 信号，但"分类之后做什么"一点没动。

---

## 五、MCP 管什么 / 不管什么

这张表是今天最想留下来的东西，周三周四施工的时候大概会反复翻回来看。

| | MCP 管（接口层） | MCP 不管（控制流，留在我 loop） |
|---|---|---|
| 工具 | 怎么描述（name/desc/inputSchema/outputSchema）、怎么发现（`tools/list` + `listChanged`）、怎么调用（`tools/call`） | 何时该调、要不要调、调哪个（model-controlled，决策在模型/host） |
| 失败 | 报成 typed 信号（协议 error / `isError` 两层） | 失败了怎么办：retry 几次？降级？skip-and-advance？replan？ |
| 数据 | resources 的标准 list/read、prompts 的标准 get | 结果够不够好（引用接地判据）、语境锚怎么不丢 |
| 连接 | capability negotiation、传输（stdio/HTTP）、JSON-RPC | 子任务怎么拆、派几个、收口策略（supervisor 那套全是自己的） |

记一句话：MCP 给的是 typed 工具契约 + 发现/调用协议 + typed 失败信道；闸门和恢复决策一概不给。

---

## 六、回头答那两个带进来的问题

**Q1：MCP 管接口层，那控制流呢？**

不管。而且读下来发现，spec 是有意把线划在"把失败报成 typed 信号"为止的。§四§五的证据已经齐了：MCP 标准化工具的描述、发现、调用，再把失败分成协议层和执行层两类报清楚。但缺陷 #5 那一串——调用漂移、结果不接地要不要 retry、子任务要不要 skip、语境锚怎么不丢——MCP 一个都不接。`isError` 回来之后是 retry 还是降级还是早退、retry 几次、什么叫"够好了"，仍然是我 loop 里那几个闸门的事（`replan_count` / `retry_count` / 引用接地软闸门 / synthesis-reserve）。

还有更扎心的一处。缺陷 #5 里那个"supervisor 把'本项目'的题拆成通用子任务，researcher 就漂到 wikipedia 去查通用概念"——那是规划/提示层的漂移，发生在"决定调哪个工具、带什么语境"这一步，整个在 MCP 的上游。哪怕把工具全搬到 MCP 后面，这个漂移会一模一样地复现，因为 MCP 只管"我已经决定要调的那个工具，描述得好不好、调得对不对"。所以我读之前那句"大概率不管"，读完得收紧成：MCP 不是漏掉了控制流，是刻意只负责把失败报成 typed 信号，好让我的控制流拿到更干净的输入。

**Q2：typed 契约能把"幻觉工具名被运行时兜住"提前到发现期拒掉吗？**

部分能，但能提前的那块不是我最想要的，真正兜住幻觉的那块仍然是 host 侧的控制流。拆成两半看。

先看有效工具集变成权威可发现了吗？是的。`tools/list` 给 host 一份带 name + JSON Schema 的权威清单，我的手写注册表换成"发现来的"，权威来源从代码挪到发现期，还能靠 `listChanged` 动态同步。这是实打实的升级。

再看 MCP 能不能阻止模型吐出清单里没有的名字？不能。工具是 model-controlled，模型在 MCP 上游，它吐 `edit` 这个 token 的时候 MCP 不在中间。于是幻觉名最后还是在两个地方之一被抓住：要么 host 在发 `tools/call` 之前，拿模型选的名字跟发现来的清单比一下、不在就本地拒掉——这确实算"派发前拒绝"，但这一步是我自己写的控制流，跟现在的 `UnknownTool` 是同一个守卫，只是比对对象从硬编码表换成了发现来的清单；要么干脆不预检直接转发，server 回一个协议层 error——还是运行期，只是从进程内挪到了 RPC 边界。

说准一点：MCP 把"这个工具名有效吗"从一个"关于我手维护的注册表"的问题，变成一个"关于发现来的 typed 契约"的问题，但它没让这个问题消失，也没替我回答。真正能落到 schema 层、提前拒掉的是另一件事——给一个真实存在的工具传了错参数，这个能拿 `inputSchema` 在执行前校验掉；而"幻觉一个根本不存在的名字"不行。

这么一看，两个问题其实塌成了一个答案：Q2 里"幻觉名守卫仍归我"，正好是 Q1"控制流归我"的一个具体例子。MCP 升级的是接口本身和它的失败词汇表；所有守卫和恢复决策——派发前拒绝、retry 判据、skip-and-advance、语境锚保持——一律还在我 loop 里。这就把要带进第九周验证的那句"框架给原语、控制流自己兜"坐实了，而且能加一个注脚：框架这回连"失败长什么样"都帮我标了类型。

---

## 七、MCP 对企业集成为什么有意义（给周六《MCP 适合接哪些企业系统》埋个头）

价值一句话就能说清：把"每个模型 × 每个 API 都得写一遍适配器"（M×N）换成"一个工具暴露一次、所有 host 都能发现调用"（M+N）。工具定义在本地和远程之间可移植（都是同一套 JSON-RPC），换底层模型也不用重写适配器。

哪些企业系统适合接，先记个初步判据，下周展开：有稳定接口、读写语义清楚的系统——内部数据库、文件系统、企业内部 API、各种 SaaS（工单 / CRM / 文档库）。判据初稿是：接口越稳定、越能被描述成 typed 工具契约的系统，越适合 MCP；反过来，强依赖隐式上下文、语义模糊的系统，光接 MCP 解决不了问题，因为问题出在控制流、不在接口——这条正好和 §六 Q1 同根。

---

## 八、新概念 → v6.0 映射表

把今天的新概念逐个对回 v6.0 已有的东西，当认知锚点。

| MCP 概念 | v6.0 已有部件 | 跨过去多了什么 / 变了什么 |
|---|---|---|
| host / client / server | 整个 v6.0 进程（没有 client/server 之分） | 在 researcher 调工具那一刀插进进程/网络边界：工具从本进程函数变成另一进程暴露的契约 |
| tool 原语 | `retrieve_documents` / `web_search` / `fetch_webpage` 手写 | 从硬编码注册表变成发现来的 typed 契约（带 inputSchema/outputSchema） |
| `tools/list` 发现 + `listChanged` | researcher 手写工具表 | 权威来源从代码挪到发现期，而且能动态同步 |
| `inputSchema` 校验 | v6.0 没有显式参数校验 | "给真工具传错参数"能在 schema 层拒掉 |
| 两层错误（协议 error / `isError`） | `UnknownTool` 兜底 + 空回答重试 | 失败信号类型化（集成问题 vs 业务失败），但"做什么"仍归我 |
| resources 原语 | RAG 文档库 / Store | 只读数据有了标准 list/read 方法（现在是 researcher 主动检索） |
| prompts 原语 | SYSTEM_PROMPT / 各角色提示常量 | 模板可暴露、可被用户挑选触发 |
| elicitation | `human_review` 节点 | 标准化成"server 可发起的、向用户要输入"的请求 |
| 控制流（retry/skip/replan/语境锚） | `replan_count` / `retry_count` / synthesis-reserve / skip-and-advance / escalate | MCP 不接，全留在 loop（§五/§六的结论） |

---

## 九、还没解决的问题（留给周二设计日和后面几周）

**Q1：周四的 MCP server 暴露哪两个工具，怎么从 v6.0 搬过去？**（周二必须定）
roadmap 要的是"本地文件搜索 + HTTP API 调用"，正好对上我的 `retrieve_documents`（本地）和 `fetch_webpage` / `web_search`（HTTP）。问题是：在现有函数外包一层 MCP server（最省事、复用引擎），还是借机重写？我倾向包一层——这又是第六七八周"升格不重写、blast radius 最小"那条原则的一次套用，让本周的"新"只落在"插一道进程边界 + 接 client"上。

**Q2：v6.0 的 `UnknownTool` 兜底 + 空回答重试，接了 MCP 之后哪些能挪到 schema 层、哪些必须留在 loop？**（接第十一周工具白名单）
按 §六 Q2 的结论：错参数校验能挪到 `inputSchema`；幻觉名守卫挪不动，只能换成"比对发现来的清单"的派发前守卫。周二要把"哪些守卫迁移、哪些保留"列清楚，这张表直接就是第十一周工具白名单的草稿。

**Q3：researcher 内层的 tool-use loop 接了 MCP client 之后，"该搜没搜 / 连续失败降级"在哪一层兜？**（承第三周《Agent Loop 设计笔记》Q3 + 第八周 Q3）
MCP 的两层错误信号（协议 error / `isError`）怎么喂回 critic/reviewer 的 accept/retry/escalate 判据？协议层 error（集成坏了）和执行层 `isError`（业务失败）按理该触发不同的恢复路径——这正是第三周留下、单 agent 答不了的那个问题，接了 MCP 的 typed 失败信道之后，这周第一次有了能把两者区分开的素材。

**Q4：RAG 检索做成 tool 还是 resource？**
tool（researcher 主动调，model-controlled）vs resource（host 被动读进上下文，app-controlled）——这个选择改变 researcher 用它的方式，也改变"谁来决定检索什么"。周二设计日定下来，顺便想清它跟 §四"谁控制"那一列的关系。

---

*完成于 2026-06-16（第九周·周一·概念精读日）。读的材料：MCP 官方 spec 2025-11-25（architecture / server-tools / transports / 错误处理几章），辅以当前生态现状（AAIF 治理、Python SDK、2026-07-28 RC 的方向）。所有结论都接回第八周 v6.0（supervisor / researcher / writer / reviewer、手写工具注册表、`UnknownTool` 兜底、Store/RAG、§闸门）、第八周《多Agent概念》§七那两个问题、第三周《Agent Loop 设计笔记》Q3。下一站：周二设计日定 Q1–Q4。*
