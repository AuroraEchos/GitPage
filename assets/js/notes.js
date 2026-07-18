const categoryLabels = {
  llm: "LLM 架构",
  agent: "Agent 与应用",
  engineering: "工程札记",
};

const notes = [
  { date: "2026.06.10", category: "engineering", title: "What is NAT traversal?", desc: "理解内网穿透中的地址转换、端口映射与通信路径。", path: "posts/Intranet_penetration.md" },
  { date: "2026.06.01", category: "agent", title: "OpenAI Function Calling Protocol", desc: "从消息协议与执行循环理解 Function Calling。", path: "posts/OpenAI_Function_Calling.md" },
  { date: "2026.05.21", category: "agent", title: "Some Thoughts on ReAct", desc: "关于 reasoning-action 循环、工具调用与错误恢复的思考。", path: "posts/Some_thoughts_on_ReAct.md" },
  { date: "2026.05.20", category: "llm", title: "What is KV Cache", desc: "为什么缓存 Key / Value 能加速自回归解码。", path: "posts/KV_Cache.md" },
  { date: "2026.05.09", category: "engineering", title: "Traffic Path of Proxy Networking", desc: "梳理客户端、DNS、隧道、端口与远程服务器之间的流量路径。", path: "posts/Traffic_Path_of_Proxy_Networking.md" },
  { date: "2026.04.30", category: "engineering", title: "PyTorch High-Frequency Core Operators", desc: "张量形状、线性代数、掩码与 Softmax 的高频操作地图。", path: "posts/PyTorch_high-frequency_core_operators.md" },
  { date: "2026.04.16", category: "engineering", title: "Linux Basic Usage", desc: "面向日常研发环境的 Linux 命令行基础。", path: "posts/Linux_Basic_Usage.md" },
  { date: "2026.04.06", category: "agent", title: "LLM Memory Mechanisms", desc: "短期上下文、外部存储、检索与长期记忆的工程模式。", path: "posts/LLM_Memory_Mechanisms.md" },
  { date: "2025.09.20", category: "engineering", title: "Async/Await in Python", desc: "用清晰的心智模型理解协程与异步程序。", path: "posts/async & await in python.md" },
  { date: "2025.09.18", category: "llm", title: "LoRA & QLoRA", desc: "大语言模型参数高效微调方法的原理与实践。", path: "posts/LoRA_and_QLoRA.md" },
  { date: "2025.09.15", category: "llm", title: "RMSNorm", desc: "RMSNorm 的计算方式，以及它为何常见于现代 LLM。", path: "posts/RMSNorm.md" },
  { date: "2025.09.10", category: "llm", title: "RoPE", desc: "旋转位置编码的直觉、公式与实现要点。", path: "posts/RoPE.md" },
  { date: "2025.09.01", category: "llm", title: "SDPA → MHA → GQA", desc: "注意力变体在表达能力、显存和推理效率之间的权衡。", path: "posts/SDPA_MHA_GQA.md" },
  { date: "2025.08.05", category: "llm", title: "Scaled Dot-Product Attention", desc: "从矩阵计算出发拆解注意力机制的核心操作。", path: "posts/Scaled_DotProduct_Attention.md" },
  { date: "2025.07.10", category: "llm", title: "Swish Gated Linear Unit (SwiGLU)", desc: "现代 Transformer 前馈网络中的门控激活函数。", path: "posts/Swish_Gated_Linear_Unit.md" },
  { date: "2025.06.15", category: "engineering", title: "Random Seed 42", desc: "可复现实验为什么仍然需要完整的上下文。", path: "posts/Random_seed_42.md" },
];

const list = document.querySelector("#notes-list");
const search = document.querySelector("#note-search");
const filters = [...document.querySelectorAll("[data-category]")];
let activeCategory = "all";

function renderNotes() {
  const keyword = search.value.trim().toLowerCase();
  const filtered = notes.filter((note) => {
    const categoryMatch = activeCategory === "all" || note.category === activeCategory;
    const queryMatch = !keyword || `${note.title} ${note.desc}`.toLowerCase().includes(keyword);
    return categoryMatch && queryMatch;
  });

  list.replaceChildren();
  if (!filtered.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "没有匹配的笔记。";
    list.append(empty);
    return;
  }

  filtered.forEach((note) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "note-item";
    button.innerHTML = `
      <time>${note.date}</time>
      <span class="note-title"><small>${categoryLabels[note.category]}</small><strong>${note.title}</strong></span>
      <span class="note-description">${note.desc}</span>
      <span aria-hidden="true">→</span>`;
    button.addEventListener("click", () => {
      window.location.href = `reader.html?src=${encodeURIComponent(note.path)}`;
    });
    list.append(button);
  });
}

filters.forEach((button) => {
  button.addEventListener("click", () => {
    activeCategory = button.dataset.category;
    filters.forEach((item) => item.classList.toggle("active", item === button));
    renderNotes();
  });
});

search.addEventListener("input", renderNotes);
renderNotes();
