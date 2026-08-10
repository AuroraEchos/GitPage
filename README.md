# Wenhao Liu — Static Personal Website

纯静态个人技术网站，不需要 Node.js、数据库或构建步骤。整体视觉采用克制、紧凑的研究者主页风格，并内置 Markdown 阅读器。

## 页面

- `about/`：关于，也是网站默认入口
- `projects/`：公开项目
- `notes/`：技术笔记列表与阅读页
- `posts/`：Markdown 原文与笔记图片；新增正文继续放在这里
- `assets/vendor/`：随站点部署的 Markdown、公式、图表和代码高亮依赖

根目录的 `index.html` 会自动跳转到 `about/`。

## 本地预览

Markdown 阅读器需要通过 HTTP 读取文章，不能直接用 `file://` 打开。可在仓库根目录运行：

```bash
python3 -m http.server 8000
```

然后访问 `http://localhost:8000/`。

## 部署到 GitHub Pages

1. 将项目内容放到 GitHub 仓库根目录并推送。
2. 打开仓库的 `Settings → Pages`。
3. 在 `Build and deployment` 中选择 `Deploy from a branch`。
4. 选择 `main` 分支和 `/ (root)` 目录。
5. 保存并等待部署完成。

所有站内资源均使用相对路径。阅读器支持文章目录、阅读进度、代码高亮与复制、KaTeX 公式、Mermaid 图表、表格及 Markdown 相对路径图片，运行时不依赖 CDN。
