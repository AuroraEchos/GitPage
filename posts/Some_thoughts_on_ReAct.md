# 关于 ReAct 的一些思考

在谈论 ReAct 之前，我们需要先对 Agent 这一概念进行一个叙述。毫无疑问，Agent 是大模型落地的主线之一，可以按照下面四个阶段来理解。

1. Chatbot 阶段

   早期的 LLM 落地主要是：用户回答问题，然后模型回答。它的价值在知识问答、写作、总结、翻译，但它的边界很明显：不能真正的操作业务系统。

2. Copilot 阶段

   然后进入 Copilot，这个阶段的任务是：人负责目标和判断，模型负责生成、补全和建议。比如一些代码助手、文档助手、客服助手等，这个阶段已经很实用，但本质还是“辅助生成”。

3. Agent 阶段

   Agent 的核心变化是：LLM + 工具调用 + 状态管理 + 多步执行。例如，用户输入帮我分析数据库里上个月销售异常，那么 Agent 会理解问题、检索 schema 、生成 SQL、执行 SQL、发现错误后修正、分析结果、生成报告。OpenAI 的 Responses API / Agents SDK 已经把 web search、file search、computer use 等工具整合进 Agent 构建链路；Anthropic 的 MCP 则试图把 Agent 和外部工具、数据源连接标准化。

4. Workflow Agent 阶段

   真正有商业价值的 Agent，不是什么都能做的通用 Agent，而是：

   > 垂直场景 + 受控工具 + 明确权限 + 可观测执行

   企业现在的问题不是“有没有 Agent 概念”，而是从 pilot 走向 scale 很难。agentic AI 正在扩散，但多数组织仍卡在规模化落地阶段。

我认为未来的形态是这样的：
```
大模型本身
↓
工具调用能力
↓
Agent Runtime
↓
MCP / API / 数据库 / 浏览器 / 文件系统
↓
业务工作流
↓
权限、审计、评估、监控
```

也就是说，未来的竞争点不是模型能力，而是模型能不能安全、稳定、可控地进入真实的业务系统。

以上是我对 Agent 的一些思考，我认为理解了上述的一些内容能够有助于理解下面我们要讨论的 ReAct。

在 Agent 领域里，ReAct 是一个非常核心，也非常现代化的思想。该思想最早来自于 ICLR 2023 的一篇论文，全名为 “ReAct: Synergizing Reasoning and Acting in Language Models”，由  Shunyu Yao  等人提出。它展示了一种将推理（Reasoning）与行动（Acting）交织在一起的通用提示范式，被广泛视为后续各类 LLM  智能体（agents）方法的重要基础之一。

ReAct 的名字正是来自于：Reasoning 与 Acting。也就是：

- 模型不仅能够思考
- 还能够行动
- 并且：
  - 根据行动结果继续思考
  - 再继续行动
  - 最终完成任务

这是一种：“LLM + Tool + Iterative Reasoning” 的范式。

这个思想直接影响了后来的：

- LangChain Agent
- OpenAI function calling agent
- AutoGPT
- BabyAGI
- Devin 类系统
- Cursor / Windsurf 类 coding agent
- 大部分 MCP Agent
- 多数现代 Tool Agent

几乎所有现代 Agent 都有 ReAct 的影子。

## 一、为什么会出现 ReAct？

先理解普通 LLM 的问题，传统的 ChatGPT 式模型都是用户输入问题，然后模型一次性生成答案，接着结束。

这在简单的问题上可以正常工作，但是普通的 LLM 没有一个长期的状态、不会调用工具、不会验证结果、不会中间修正、不会规划。于是，推理和执行是断开的，这就是 ReAct 要解决的问题。

## 二、经典的 ReAct 流程

ReAct 认为：

> 模型应该像人一样：
>
> 一边思考，
>  一边行动，
>  再根据环境反馈继续思考。

于是一个标准的 ReAct 格式如下：

```
Question:
用户问题

Thought:
我应该先做什么？

Action:
调用哪个工具？

Observation:
工具返回什么？

Thought:
下一步怎么办？

Action:
继续调用工具

Observation:
新的结果

Thought:
我已经得到答案

Final Answer:
最终输出
```

我们可以举出一个简单却真实的例子，比如：用户询问今天上海的天气怎么样？适合跑步吗？

1. 第一步是 Reason 过程，模型会思考：我需要先查询北京的天气。

2. 第二步是 Act ，模型发现我可以使用一些定义好的工具函数：weather_api("shanghai")。

3. 第三步是 Observation，模型这时候会得到 API 的返回结果，例如：

   ```
   {
     "temp": 18,
     "humidity": 35,
     "air_quality": "good"
   }
   ```

4. 第四步仍然是 Reason 过程，模型此时会根据 API 的返回结果思考：天气凉爽，空气质量不错，适合跑步。

5. 第五步是 Final Answer，模型认为可以了就会返回结果：北京今天 18℃，空气质量良好，适合户外跑步。

## 三、ReAct 和 Chain-of-Thought 的区别

我们介绍一下 CoT 思维链，CoT 只有思考，没有行动。ReAct 是推理 + 外部环境交互，这是他们两个的本质区别。

## 四、ReAct 的关键难点

1. Tool Calling

   如何去定义工具、参数结构化、执行、返回。

2. Context 管理

   Agent 最大的问题是上下文爆炸。因为 Thought、Action、Observation 这个过程会越来越长。现代系统的：memory、summarization、retrieval memory、scratchpad、state compression 都是为了解决这个问题。

3. 错误修复

   比如：API失败、tool timeout、hallucination，这些都是需要解决的。

## 五、现代 Agent 基本结构

现在主流 Agent 大致都是：

```
User Input
    ↓
Planner / Reasoner
    ↓
Tool Selection
    ↓
Tool Execution
    ↓
Observation
    ↓
Memory Update
    ↓
Next Step Reasoning
```

ReAct 是它们的思想起点。

## 六、总结

大语言模型（LLMs）是一项伟大且强大的新技术。当 LLM 与外部数据源结合时，会变得更强大。LLM 将重塑未来应用程序的形态。具体而言，未来的应用将越来越呈现 Agent 化。我们仍处于这一变革的早期阶段。虽然构建这类 Agent 化应用的原型很容易，但要构建**足够可靠、可用于生产环境的 Agent**仍然非常困难。

于是 LangChain 出现了。LangChain 的名称来自 “Language”（语言模型）和 “Chains”。

> LangChain 的存在，是为了成为构建 LLM 应用最容易的地方，同时具备灵活性并可投入生产环境。

在 ChatGPT 发布前一个月，LangChain 作为 Python 包发布。

ReAct 论文在 2022 年出现在 arXiv 后，第一批通用 Agent 被添加到 LangChain 中。

在 LangChain 的官网中明确指出：这些通用 Agent 基于 [ReAct 论文](https://arxiv.org/abs/2210.03629)。

这一切都有迹可循。

我们将在接下来的笔记中详细介绍 LangChain。