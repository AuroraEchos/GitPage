(() => {
  const article = document.querySelector("#article");
  const titleNode = document.querySelector("#reader-title");
  const sourceNode = document.querySelector("#reader-source");
  const toc = document.querySelector("#toc-list");
  const progress = document.querySelector("#reading-progress-bar");
  const src = new URLSearchParams(location.search).get("src") || "";

  const validSource = /^posts\/[^/\\]+\.md$/i.test(src) && !src.includes("..");
  const escapeHtml = (value) => value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  const slugify = (text, index) => `${text.toLowerCase().trim().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "") || "section"}-${index + 1}`;

  function preserveDisplayMath(markdown) {
    const blocks = [];
    const content = markdown.replace(/^[ \t]*\$\$[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*\$\$[ \t]*$/gm, (_, formula) => {
      const index = blocks.push(formula) - 1;
      return `<div data-math-block="${index}"></div>`;
    });
    return { content, blocks };
  }

  function restoreDisplayMath(blocks) {
    article.querySelectorAll("[data-math-block]").forEach((node) => {
      const index = Number(node.dataset.mathBlock);
      if (!Number.isInteger(index) || index < 0 || index >= blocks.length) return;
      node.removeAttribute("data-math-block");
      node.textContent = `$$\n${blocks[index]}\n$$`;
    });
  }

  function fail(message) {
    titleNode.textContent = "无法打开这篇笔记";
    sourceNode.textContent = "请返回笔记列表重新选择";
    article.innerHTML = `<div class="reader-error"><p>${escapeHtml(message)}</p><a href="./">返回笔记</a></div>`;
    toc.closest(".toc").hidden = true;
  }

  function resolveRelativeUrls(baseUrl) {
    article.querySelectorAll("img[src]").forEach((image) => {
      const value = image.getAttribute("src");
      if (!/^(?:[a-z]+:|\/|#)/i.test(value)) image.src = new URL(value, baseUrl).href;
      image.loading = "lazy";
    });
    article.querySelectorAll("a[href]").forEach((link) => {
      const value = link.getAttribute("href");
      if (!/^(?:[a-z]+:|\/|#)/i.test(value)) link.href = new URL(value, baseUrl).href;
      if (/^https?:/i.test(link.href)) { link.target = "_blank"; link.rel = "noopener noreferrer"; }
    });
  }

  function buildToc() {
    const headings = [...article.querySelectorAll("h1, h2, h3, h4")];
    const baseLevel = headings.length
      ? Math.min(...headings.map((heading) => Number(heading.tagName.slice(1))))
      : 1;
    toc.replaceChildren();
    headings.forEach((heading, index) => {
      heading.id = slugify(heading.textContent, index);
      const level = Number(heading.tagName.slice(1)) - baseLevel + 1;
      const link = document.createElement("a");
      link.href = `#${heading.id}`;
      link.className = `toc-level-${level}`;
      link.textContent = heading.textContent;
      toc.append(link);
    });
    toc.closest(".toc").hidden = headings.length < 2;
  }

  function enhanceCode() {
    article.querySelectorAll("pre code").forEach((code) => {
      if (code.classList.contains("language-mermaid")) {
        const diagram = document.createElement("div");
        diagram.className = "mermaid";
        diagram.textContent = code.textContent;
        code.parentElement.replaceWith(diagram);
        return;
      }
      window.hljs?.highlightElement(code);
      const pre = code.parentElement;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "copy-code";
      button.textContent = "复制";
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(code.textContent);
          button.textContent = "已复制";
        } catch {
          button.textContent = "复制失败";
        }
        setTimeout(() => { button.textContent = "复制"; }, 1400);
      });
      pre.append(button);
    });
  }

  function loadMermaid() {
    return new Promise((resolve, reject) => {
      if (window.mermaid) return resolve();
      const script = document.createElement("script");
      script.src = "../assets/vendor/mermaid.min.js";
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("图表渲染库加载失败"));
      document.head.append(script);
    });
  }

  async function renderDiagrams() {
    const nodes = [...article.querySelectorAll(".mermaid")];
    if (!nodes.length) return;
    try {
      await loadMermaid();
      window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "neutral" });
      await window.mermaid.run({ nodes });
    } catch (error) { console.warn("Mermaid rendering failed", error); }
  }

  function updateProgress() {
    const start = article.offsetTop;
    const distance = Math.max(article.offsetHeight - innerHeight, 1);
    const value = Math.min(1, Math.max(0, (scrollY - start + 120) / distance));
    progress.style.transform = `scaleX(${value})`;
  }

  function updateToc() {
    const headings = [...article.querySelectorAll("h1, h2, h3, h4")];
    const links = [...toc.querySelectorAll("a")];
    if (!headings.length || !links.length) return;

    let current = headings[0];
    headings.forEach((heading) => {
      if (heading.getBoundingClientRect().top <= 150) current = heading;
    });
    if (scrollY + innerHeight >= document.documentElement.scrollHeight - 2) current = headings.at(-1);

    links.forEach((link) => {
      const isActive = link.getAttribute("href") === `#${current.id}`;
      link.classList.toggle("active", isActive);
      if (isActive) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }

  let readingStateFrame = 0;
  function scheduleReadingStateUpdate() {
    if (readingStateFrame) return;
    readingStateFrame = requestAnimationFrame(() => {
      updateProgress();
      updateToc();
      readingStateFrame = 0;
    });
  }

  async function load() {
    if (!validSource) return fail("无效的 Markdown 文件地址。");
    if (location.protocol === "file:") {
      return fail("浏览器不允许网页直接读取本地 Markdown。请通过本地 HTTP 服务或 GitHub Pages 打开网站。");
    }
    try {
      const requestUrl = new URL(`../${src}`, location.href);
      const response = await fetch(requestUrl);
      if (!response.ok) throw new Error(`文件读取失败（${response.status}）`);
      const raw = await response.text();
      // 剥掉 front-matter（tools/build_notes.py 使用的元数据块），避免被渲染成正文
      const markdown = raw.replace(/^(?:\uFEFF)?---[ \t]*\r?\n[\s\S]*?\r?\n---[ \t]*(?:\r?\n|$)/, "");
      const math = preserveDisplayMath(markdown);
      const html = window.marked.parse(math.content, { gfm: true, breaks: false });
      article.innerHTML = window.DOMPurify.sanitize(html, { ADD_ATTR: ["target", "rel", "data-math-block"] });
      restoreDisplayMath(math.blocks);
      const firstHeading = article.querySelector("h1, h2");
      const fallback = decodeURIComponent(src.split("/").pop()).replace(/\.md$/i, "").replace(/_/g, " ");
      const title = firstHeading?.textContent.trim() || fallback;
      titleNode.textContent = title;
      sourceNode.textContent = fallback;
      document.title = `${title} · Wenhao Liu`;
      if (firstHeading?.tagName === "H1") firstHeading.remove();
      resolveRelativeUrls(requestUrl);
      buildToc();
      enhanceCode();
      window.renderMathInElement?.(article, { throwOnError: false, delimiters: [
        { left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }
      ] });
      await renderDiagrams();
      scheduleReadingStateUpdate();
    } catch (error) { fail(error.message || "Markdown 文件读取失败。"); }
  }

  addEventListener("scroll", scheduleReadingStateUpdate, { passive: true });
  addEventListener("resize", scheduleReadingStateUpdate);
  load();
})();
