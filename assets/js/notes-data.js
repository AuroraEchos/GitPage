// 此文件由 tools/build_notes.py 自动生成，请勿手动编辑。
// 修改 posts/ 下的 front-matter 后运行：python3 tools/build_notes.py
window.noteCatalog = {
  "categoryLabels": {
    "llm": "LLM",
    "agent": "Agent",
    "paper": "Paper Reading",
    "other": "Other"
  },
  "notes": [
    {
      "date": "2026.08.28",
      "category": "agent",
      "title": "内部状态驱动控制流，事件流驱动观察与持久化",
      "description": "以 PhoneAgent 的 `state.py` 与 `events.py` 为例，理解“内部状态驱动控制流，事件流驱动观察与持久化”。",
      "path": "posts/Internal_State_driven_Control_Flow_And_Envent_Driven_Observation_Persistence.md"
    },
    {
      "date": "2026.08.25",
      "category": "agent",
      "title": "从“点错位置”到坐标协议层：视觉智能体跨模型坐标系统的可能性解决方案",
      "description": "一次真实 Android 真机故障如何揭示视觉模型坐标协议的不确定性，以及如何用显式契约、零触摸协商、canonical 坐标、失败关闭和完整审计构建跨模型的可靠执行系统。",
      "path": "posts/Possible Solutions for Visual Agents Across Coordinate Systems.md"
    },
    {
      "date": "2026.08.20",
      "category": "other",
      "title": "Ubuntu 26.04 LTS ChatGPT 桌面端代理生效但消息发送失败",
      "description": "Ubuntu 新版 ChatGPT 桌面端代理失效的核心不是网络问题，而是 GUI 启动环境隔离。最优解为修改本地 desktop 文件注入代理环境变量，一次配置永久生效，无需终端、无需重复操作，完美解决消息转圈、断流、发送失败问题。",
      "path": "posts/Ubuntu_26_04_LTS_ChatGPT.md"
    },
    {
      "date": "2026.08.18",
      "category": "other",
      "title": "《A Programming Paradigm for Spatiotemporal Composability》阅读分析报告",
      "description": "一个正在运行的软件系统，怎样才能像搭积木一样安全地添加、删除和替换组件，而不必重启整个系统？",
      "path": "posts/paper_reading_analysis_zh.md"
    },
    {
      "date": "2026.08.12",
      "category": "agent",
      "title": "PhoneAgent v0.2.0 Runtime Architecture",
      "description": "PhoneAgent 核心运行时架构与执行流程的设计说明与代码导读。",
      "path": "posts/Runtime Architecture and Execution Flow Technical Document.md"
    },
    {
      "date": "2026.08.01",
      "category": "llm",
      "title": "Qwen3-VL",
      "description": "梳理 Qwen3-VL 的视觉编码、DeepStack、多模态位置编码与主干数据流。",
      "path": "posts/Qwen3-VL.md"
    },
    {
      "date": "2026.07.10",
      "category": "other",
      "title": "LLM Basic Knowledge",
      "description": "从自回归解码、注意力机制、位置编码到前馈网络，梳理 LLM 的基础知识。",
      "path": "posts/LLM_Basic_Knowledge.md"
    },
    {
      "date": "2026.06.10",
      "category": "other",
      "title": "What is NAT traversal?",
      "description": "理解内网穿透中的地址转换、端口映射与通信路径。",
      "path": "posts/Intranet_penetration.md"
    },
    {
      "date": "2026.06.01",
      "category": "agent",
      "title": "OpenAI Function Calling Protocol",
      "description": "从消息协议与执行循环理解 Function Calling。",
      "path": "posts/OpenAI_Function_Calling.md"
    },
    {
      "date": "2026.05.21",
      "category": "agent",
      "title": "Some Thoughts on ReAct",
      "description": "关于 reasoning-action 循环、工具调用与错误恢复的思考。",
      "path": "posts/Some_thoughts_on_ReAct.md"
    },
    {
      "date": "2026.05.20",
      "category": "llm",
      "title": "What is KV Cache",
      "description": "为什么缓存 Key / Value 能加速自回归解码。",
      "path": "posts/KV_Cache.md"
    },
    {
      "date": "2026.05.09",
      "category": "other",
      "title": "Traffic Path of Proxy Networking",
      "description": "梳理客户端、DNS、隧道、端口与远程服务器之间的流量路径。",
      "path": "posts/Traffic_Path_of_Proxy_Networking.md"
    },
    {
      "date": "2026.04.30",
      "category": "other",
      "title": "PyTorch High-Frequency Core Operators",
      "description": "张量形状、线性代数、掩码与 Softmax 的高频操作地图。",
      "path": "posts/PyTorch_high-frequency_core_operators.md"
    },
    {
      "date": "2026.04.16",
      "category": "other",
      "title": "Linux Basic Usage",
      "description": "面向日常研发环境的 Linux 命令行基础。",
      "path": "posts/Linux_Basic_Usage.md"
    },
    {
      "date": "2026.04.06",
      "category": "agent",
      "title": "LLM Memory Mechanisms",
      "description": "短期上下文、外部存储、检索与长期记忆的工程模式。",
      "path": "posts/LLM_Memory_Mechanisms.md"
    },
    {
      "date": "2025.09.20",
      "category": "other",
      "title": "Async/Await in Python",
      "description": "用清晰的心智模型理解协程与异步程序。",
      "path": "posts/async & await in python.md"
    },
    {
      "date": "2025.09.18",
      "category": "llm",
      "title": "LoRA & QLoRA",
      "description": "大语言模型参数高效微调方法的原理与实践。",
      "path": "posts/LoRA_and_QLoRA.md"
    },
    {
      "date": "2025.09.15",
      "category": "llm",
      "title": "RMSNorm",
      "description": "RMSNorm 的计算方式，以及它为何常见于现代 LLM。",
      "path": "posts/RMSNorm.md"
    },
    {
      "date": "2025.09.10",
      "category": "llm",
      "title": "RoPE",
      "description": "旋转位置编码的直觉、公式与实现要点。",
      "path": "posts/RoPE.md"
    },
    {
      "date": "2025.09.01",
      "category": "llm",
      "title": "SDPA → MHA → GQA",
      "description": "注意力变体在表达能力、显存和推理效率之间的权衡。",
      "path": "posts/SDPA_MHA_GQA.md"
    },
    {
      "date": "2025.08.23",
      "category": "other",
      "title": "海康威视 MF5681（800 万 4K USB 摄像头）Ubuntu 使用记录",
      "description": "",
      "path": "posts/MF5681.md"
    },
    {
      "date": "2025.08.05",
      "category": "llm",
      "title": "Scaled Dot-Product Attention",
      "description": "从矩阵计算出发拆解注意力机制的核心操作。",
      "path": "posts/Scaled_DotProduct_Attention.md"
    },
    {
      "date": "2025.07.10",
      "category": "llm",
      "title": "Swish Gated Linear Unit (SwiGLU)",
      "description": "现代 Transformer 前馈网络中的门控激活函数。",
      "path": "posts/Swish_Gated_Linear_Unit.md"
    },
    {
      "date": "2025.06.15",
      "category": "other",
      "title": "Random Seed 42",
      "description": "可复现实验为什么仍然需要完整的上下文。",
      "path": "posts/Random_seed_42.md"
    }
  ]
};
