### OpenAI Function Calling Protocol

下面我们介绍一下大模型应用侧的一个核心概念：Tool Calling。这里的 Tool Calling，本质不是“模型真的执行函数”，而是：模型输出一段结构化 JSON，告诉你的程序：我想要调用哪个工具、参数是什么；真正执行函数的是你的应用代码；执行结果再作为下一轮输入交还给模型。OpenAI 官方也把 function calling 称为 tool calling，用于让模型连接外部系统和外部数据；function tool 是一种由 JSON Schema 定义的工具。

------

#### 两套常见官方协议：Chat Completions 与 Responses API

现在 OpenAI 工具调用主要有两种协议形态：

第一种是 Chat Completions API，也就是在 LangGraph / LangChain 里经常看到的`messages + tools + tool_calls` 模式。它的特点是：

```json
{
  "model": "gpt-4.1",
  "messages": [
    {
      "role": "user",
      "content": "Add 3 and 4."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add",
        "description": "Add two numbers.",
        "parameters": {
          "type": "object",
          "properties": {
            "a": { "type": "number" },
            "b": { "type": "number" }
          },
          "required": ["a", "b"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ]
}
```

第二种是 Responses API，也就是更新的统一接口风格，结构更偏向 input + output items。Responses API 的工具定义通常是更扁平的形式，例如：

```json
{
  "model": "gpt-5.5",
  "input": [
    {
      "role": "user",
      "content": "Add 3 and 4."
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "add",
      "description": "Add two numbers.",
      "parameters": {
        "type": "object",
        "properties": {
          "a": { "type": "number" },
          "b": { "type": "number" }
        },
        "required": ["a", "b"],
        "additionalProperties": false
      },
      "strict": true
    }
  ]
}
```

两者核心思想一致，但 JSON 形态不同。Chat Completions 中 function 定义嵌在 "function" 字段里；Responses API 中 function tool 通常直接写成 { "type": "function", "name": ..., "parameters": ... }。OpenAI API reference 也明确区分了 Chat Completions 与 Responses API 的工具定义形态。我们下面重点介绍一下 Chat Completions 。

------

#### Tool Calling 的完整五步流程

一般而言，工具调用的流程可以概括为五步：给模型发送可用工具列表；模型返回 tool call；应用侧执行函数；把工具执行结果发回模型；模型生成最终回答，或者继续请求更多工具调用。

```
User
  ↓
你的程序：messages + tools
  ↓
OpenAI 模型
  ↓
assistant message: tool_calls
  ↓
你的程序解析 tool_calls，执行本地函数
  ↓
你的程序追加 tool result
  ↓
再次请求模型
  ↓
assistant final answer
```

模型只负责“提出调用请求”，不负责真正执行工具。

------

#### Chat Completions API 的官方 JSON 协议

1. **请求：声明工具**

   Chat Completions 的工具列表大概长这样：

   ```json
   {
     "tools": [
       {
         "type": "function",
         "function": {
           "name": "get_weather",
           "description": "Get current weather for a given location.",
           "parameters": {
             "type": "object",
             "properties": {
               "location": {
                 "type": "string",
                 "description": "City and country, e.g. Paris, France"
               },
               "unit": {
                 "type": "string",
                 "enum": ["celsius", "fahrenheit"]
               }
             },
             "required": ["location", "unit"],
             "additionalProperties": false
           },
           "strict": true
         }
       }
     ]
   }
   ```

   这里有几个字段非常重要：

   - `type: "function"` 表示这是一个函数工具。Chat Completions 当前的 tool call 类型就是 function。
   - `function.name` 是工具名。模型后面会通过这个名字告诉你要调用哪个函数。
   - `function.description` 是工具说明。这个字段非常重要，因为模型会根据 description 判断什么时候调用这个工具。
   - `function.parameters` 是 JSON Schema。它约束模型生成哪些参数、参数类型是什么、哪些字段必填。
   - `function.strict` 开启严格模式。OpenAI 官方建议启用 strict mode，因为它能让 function call 更可靠地遵循 schema；strict mode 要求对象 schema 设置 `additionalProperties: false`，并且 properties 里的字段都放入 `required`。

2. **模型响应：assistant 返回 tool_calls**

   当模型决定需要调用工具时，Chat Completions 返回的 assistant message 中会带 `tool_calls`：

   ```json
   {
     "role": "assistant",
     "content": null,
     "tool_calls": [
       {
         "id": "call_abc123",
         "type": "function",
         "function": {
           "name": "get_weather",
           "arguments": "{\"location\":\"Paris, France\",\"unit\":\"celsius\"}"
         }
       }
     ]
   }
   ```

   注意这里最容易踩坑的地方：`function.arguments` **是一个 JSON 字符串，不是已经解析好的 JSON 对象**。也就是说你拿到的是：

   ```json
   "{\"location\":\"Paris, France\",\"unit\":\"celsius\"}"
   ```

   你的程序需要自己做：`args = json.loads(tool_call.function.arguments)`。

   模型不一定总是生成合法 JSON，也可能生成 schema 外的参数，因此应用代码应该在真正调用函数前验证参数。

3. **finish_reason: tool_calls**

   如果模型调用工具，Chat Completions 的 `finish_reason` 通常是：

   ```json
   "finish_reason": "tool_calls"
   ```

   这表示模型这一轮不是给最终自然语言答案，而是停在了“请求调用工具”的阶段。

4. **应用侧执行工具**

   假设你本地有函数：

   ```python
   def get_weather(location: str, unit: str) -> dict:
       return {
           "location": location,
           "temperature": 25,
           "unit": unit
       }
   ```

   你需要根据模型返回的 `function.name` 路由到本地函数：

   ```python
   tool_call = response.choices[0].message.tool_calls[0]
   
   name = tool_call.function.name
   args = json.loads(tool_call.function.arguments)
   
   if name == "get_weather":
       result = get_weather(**args)
   else:
       raise ValueError(f"Unknown tool: {name}")
   ```

   这里的执行权完全在你的应用侧。模型不会访问你的数据库、不会调用你的 Python 函数、不会自动执行 shell 命令。它只是输出协议化 JSON。

5. **把工具结果追加回 messages**

   执行完后，需要把工具结果作为一条 `role: "tool"` 消息追加回上下文：

   ```json
   {
     "role": "tool",
     "tool_call_id": "call_abc123",
     "content": "{\"location\":\"Paris, France\",\"temperature\":25,\"unit\":\"celsius\"}"
   }
   ```

   完整的下一轮 messages 是：

   ```json
   [
     {
       "role": "user",
       "content": "What's the weather in Paris?"
     },
     {
       "role": "assistant",
       "content": null,
       "tool_calls": [
         {
           "id": "call_abc123",
           "type": "function",
           "function": {
             "name": "get_weather",
             "arguments": "{\"location\":\"Paris, France\",\"unit\":\"celsius\"}"
           }
         }
       ]
     },
     {
       "role": "tool",
       "tool_call_id": "call_abc123",
       "content": "{\"location\":\"Paris, France\",\"temperature\":25,\"unit\":\"celsius\"}"
     }
   ]
   ```

   其中 `tool_call_id` 必须和 assistant message 里的 tool call `id` 对上。先把模型的 assistant tool call message 加入 `messages`，然后追加 `role: "tool"`、`tool_call_id`、`content`，再请求模型生成最终回答。

6. **再次请求模型，得到最终回答**

   第二次调用：

   ```json
   {
     "model": "gpt-4.1",
     "messages": [
       "... 上面的完整 messages ..."
     ],
     "tools": [
       "... 同样的 tools ..."
     ]
   }
   ```

   模型此时看到：

   1. 用户问天气；
   2. 自己刚才请求调用 `get_weather`；
   3. 工具已经返回结果；

   于是它可以生成自然语言答案：

   ```json
   {
     "role": "assistant",
     "content": "The weather in Paris is 25°C."
   }
   ```

------

以上就是一个经典的过程，这里有几个问题需要阐述一下：

1. **`content: null` 的原因？**

   用一句简要的话概括就是这一轮 assistant message 不是给用户看的最终回答，而是给程序看的工具调用指令；真正的自然语言回答要等工具执行结果返回后，模型下一轮再生成。

2. **模型知道接下来需要调用工具了，那么模型这个时候的原始输出是什么？**

   这个问题的答案是模型知道要调用工具时，它的“原始输出”不是一句普通文本，而是一个结构化的 tool call 对象，在 API 请求里通过 `tools` 参数正式声明了一组工具。模型看到这些工具定义后，如果判断需要使用工具，就会通过 API 协议返回 `tool_calls` 结构。。更准确地说，要区分两层：

   ```
   1. 模型内部到底逐 token 生成了什么
      → 这个 OpenAI 不暴露，外部看不到。
   
   2. API 返回给你的原始响应 JSON 是什么
      → 你能看到的是 assistant message 里的 tool_calls 字段。
   ```

   所以从开发者角度，模型此时的原始可见输出就是 tool_calls，不是 content。当模型决定调用工具时，它对外暴露的原始输出就是一个结构化的 tool_calls 对象；content 为 null，因为这一轮模型生成的是“调用动作”，不是“自然语言答案”。模型内部到底是如何生成这个结构的，OpenAI API 不暴露；你作为开发者只需要遵循 API 返回的 JSON 协议处理即可。

3. **现在将问题上升一个层次：模型到底吃进去的是什么？模型到底吐出来的是什么？**

   要分三层看：

   ```
   第 1 层：你代码里发送给 OpenAI API 的 HTTP JSON
   第 2 层：OpenAI 服务端真正喂给模型的内部序列化输入
   第 3 层：OpenAI API 返回给你代码的 JSON 响应
   ```

   其中，第 1 层和第 3 层你能看到；第 2 层你看不到，只能根据官方说明理解其大致机制。

   以 Chat Completions API 为例，你代码里真正发送的是类似这样的 JSON：

   ```json
   {
     "model": "gpt-4.1",
     "messages": [
       {
         "role": "developer",
         "content": "You are a helpful assistant. Use tools when needed."
       },
       {
         "role": "user",
         "content": "What's the weather in Paris?"
       }
     ],
     "tools": [
       {
         "type": "function",
         "function": {
           "name": "get_weather",
           "description": "Get current weather for a given location.",
           "parameters": {
             "type": "object",
             "properties": {
               "location": {
                 "type": "string"
               }
             },
             "required": ["location"],
             "additionalProperties": false
           },
           "strict": true
         }
       }
     ],
     "tool_choice": "auto"
   }
   ```

   这就是**应用侧可见的原始输入**。

   它里面有三类信息：

   ```
   messages:
       对话上下文，包括 developer / system / user / assistant / tool 消息。
   
   tools:
       工具定义，包括工具名、工具描述、参数 JSON Schema。
   
   tool_choice:
       工具选择策略，比如 auto、required、none、强制某个工具。
   ```

   OpenAI 官方文档说明，请求中可以包含模型可考虑使用的工具列表；如果模型判断需要工具，就可能返回一个 tool call。函数工具由 JSON Schema 定义，用来让模型把数据传给你的应用代码。

   但模型内部真正看到的输入不是你看到的 JSON 原样，这一点很关键。

   你发送的是：

   ```json
   {
     "messages": [...],
     "tools": [...]
   }
   ```

   但模型内部不一定是直接看见这份 JSON 原文。OpenAI 官方文档明确说，**在底层，函数会以模型训练过的某种语法注入到 system message 中，因此函数定义会占用上下文窗口并按输入 token 计费**。

   也就是说，服务端大致会把你的输入转换成某种内部提示序列，近似可以理解成：

   ```
   [developer]
   You are a helpful assistant. Use tools when needed.
   
   [available_tools]
   function get_weather(location: string)
   description: Get current weather for a given location.
   parameters schema:
   {
     "type": "object",
     "properties": {
       "location": {"type": "string"}
     },
     "required": ["location"],
     "additionalProperties": false
   }
   
   [user]
   What's the weather in Paris?
   ```

   但注意：**这只是便于理解的近似表示，不是 OpenAI 暴露的真实内部格式。**

   你作为开发者能确定的是：

   ```
   tools 会进入模型上下文；
   工具定义会影响模型判断；
   工具 schema 会影响模型生成参数；
   具体内部序列化格式不对外暴露。
   ```

   那么模型决定“不调用工具”时的原始输出是什么？如果模型认为自己可以直接回答，它返回的 assistant message 类似：

   ```json
   {
     "choices": [
       {
         "message": {
           "role": "assistant",
           "content": "Paris is the capital of France."
         },
         "finish_reason": "stop"
       }
     ]
   }
   ```

   这时输出主体是：`"content": "Paris is the capital of France."`

   也就是普通自然语言答案。

   那么模型决定“调用工具”时的原始输出又是什么？如果模型认为需要工具，它返回的不是普通文本，而是：

   ```json
   {
     "choices": [
       {
         "message": {
           "role": "assistant",
           "content": null,
           "tool_calls": [
             {
               "id": "call_abc123",
               "type": "function",
               "function": {
                 "name": "get_weather",
                 "arguments": "{\"location\":\"Paris\"}"
               }
             }
           ]
         },
         "finish_reason": "tool_calls"
       }
     ]
   }
   ```

   这就是你能看到的**模型原始输出 JSON**。这里的核心字段是：

   ```
   role: assistant
       说明这是模型生成的 assistant 消息。
   
   content: null
       说明这一轮不是自然语言回答。
   
   tool_calls:
       说明模型请求调用工具。
   
   tool_calls[0].id:
       这次工具调用的唯一 ID，后面 tool result 要用它对应回来。
   
   tool_calls[0].function.name:
       模型想调用的函数名。
   
   tool_calls[0].function.arguments:
       模型生成的函数参数，是 JSON 字符串。
   
   finish_reason: tool_calls
       说明模型停止生成的原因是调用了工具。
   ```

   一句话总结：

   > 模型的可见原始输入是 API JSON 里的 messages 和 tools；模型的可见原始输出不是普通文本，而可能是 assistant.content，也可能是 assistant.tool_calls。至于 OpenAI 服务端如何把这些 JSON 精确序列化成模型内部 token 序列，这个不对外暴露，只能知道工具定义会被注入上下文并参与模型生成。

4. **最后一个问题：什么时候此次对话结束？**

   循环结束有两个层面的判断：

   ```
   协议层结束：
       模型这一轮没有返回 tool_calls
   
   工程层结束：
       你的 Agent 判断任务已经完成，或者达到最大循环次数 / 出错 / 被安全策略拦截
   ```

   最核心的一句话是：只要模型返回 `tool_calls`，循环就继续；如果模型没有返回 `tool_calls`，而是返回普通 `content`，循环就结束。

   OpenAI 官方工具调用流程也是这个结构：模型可能返回 function call，你的应用执行函数并把结果返回给模型；如果模型继续返回 function call，就继续执行；如果模型返回最终响应，就结束。

   我们可以给出一个最标准的结束条件伪代码：

   ```python
   while True:
       response = client.chat.completions.create(
           model=model,
           messages=messages,
           tools=tools,
       )
   
       assistant_message = response.choices[0].message
       messages.append(assistant_message)
   
       # 结束条件：没有工具调用
       if not assistant_message.tool_calls:
           return assistant_message.content
   
       # 继续条件：有工具调用
       for tool_call in assistant_message.tool_calls:
           result = execute_tool(tool_call)
   
           messages.append({
               "role": "tool",
               "tool_call_id": tool_call.id,
               "content": json.dumps(result, ensure_ascii=False)
           })
   ```

   从 finish_reason 看循环状态，如果模型决定调用工具，通常会看到：

   ```json
   {
     "finish_reason": "tool_calls"
   }
   ```

   这表示这一轮模型停止的原因是它请求了工具调用。如果模型已经完成最终回答，通常是：

   ```json
   {
     "finish_reason": "stop"
   }
   ```

   这时一般没有 `tool_calls`，而是有普通文本：

   ```json
   {
     "role": "assistant",
     "content": "Paris 当前天气是 25°C。"
   }
   ```

   但工程上更稳妥的判断是：**优先看 `assistant_message.tool_calls` 是否为空**，不要只依赖 `finish_reason`。

   为什么不能让模型自己输出 `finish` 来结束？

   如果你是手写 ReAct Agent，可能会设计这种格式：

   ```
   Thought: ...
   Action: search
   Action Input: ...
   
   或者：
   
   Final Answer: ...
   ```

   这时结束条件通常是：

   ```
   模型输出 Final Answer
   → 结束
   ```

   但 OpenAI 官方 Tool Calling 不需要你自己设计 `finish` 标记。它已经把状态分成了两类：

   ```
   Action 阶段：
       assistant.tool_calls 不为空
   
   Answer 阶段：
       assistant.tool_calls 为空，assistant.content 有最终答案
   ```

   所以官方 Tool Calling 的结束条件更干净：

   ```
   if not assistant_message.tool_calls:
       break
   ```

   虽然理论上模型最后会停止调用工具并返回答案，但真实系统里必须加保护：最大循环次数。

   ```python
   MAX_TOOL_ROUNDS = 5
   
   for step in range(MAX_TOOL_ROUNDS):
       response = call_model(messages, tools)
       assistant_message = response.choices[0].message
       messages.append(assistant_message)
   
       if not assistant_message.tool_calls:
           return assistant_message.content
   
       for tool_call in assistant_message.tool_calls:
           result = execute_tool(tool_call)
           messages.append(make_tool_message(tool_call.id, result))
   
   raise RuntimeError("Tool calling exceeded max rounds")
   ```

   模型可能陷入这种循环：

   ```
   调用 search
   → 工具返回信息不足
   → 再调用 search
   → 工具返回信息不足
   → 再调用 search
   → ...
   ```



上述就是这次分享的内容，可以对 Tool Calling 有一个更加清晰的认知。