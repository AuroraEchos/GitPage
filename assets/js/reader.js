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
    const headings = [...article.querySelectorAll("h1, h2, h3")];
    toc.replaceChildren();
    headings.forEach((heading, index) => {
      heading.id = slugify(heading.textContent, index);
      const link = document.createElement("a");
      link.href = `#${heading.id}`;
      link.className = `toc-level-${heading.tagName.slice(1)}`;
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
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "已复制";
        setTimeout(() => { button.textContent = "复制"; }, 1400);
      });
      pre.append(button);
    });
  }

  async function renderDiagrams() {
    const nodes = article.querySelectorAll(".mermaid");
    if (!nodes.length || !window.mermaid) return;
    window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: document.documentElement.classList.contains("dark") ? "dark" : "neutral" });
    try { await window.mermaid.run({ nodes }); } catch (error) { console.warn("Mermaid rendering failed", error); }
  }

  function updateProgress() {
    const start = article.offsetTop;
    const distance = Math.max(article.offsetHeight - innerHeight, 1);
    const value = Math.min(1, Math.max(0, (scrollY - start + 120) / distance));
    progress.style.transform = `scaleX(${value})`;
  }

  async function load() {
    if (!validSource) return fail("无效的 Markdown 文件地址。");
    try {
      const requestUrl = new URL(`../${src}`, location.href);
      const response = await fetch(requestUrl);
      if (!response.ok) throw new Error(`文件读取失败（${response.status}）`);
      const markdown = await response.text();
      const html = window.marked.parse(markdown, { gfm: true, breaks: false });
      article.innerHTML = window.DOMPurify.sanitize(html, { ADD_ATTR: ["target", "rel"] });
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
      updateProgress();
    } catch (error) { fail(error.message || "Markdown 文件读取失败。"); }
  }

  addEventListener("scroll", updateProgress, { passive: true });
  addEventListener("resize", updateProgress);
  load();
})();
