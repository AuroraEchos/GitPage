# Personal Site

This site is designed to stay fully static on GitHub Pages.

## Writing Notes

1. Write Markdown in `posts/`.
2. Put images next to the Markdown file when possible.
3. Add the note to `data/posts.js` and point `path` to:

```js
reader.html?src=posts/your-note.md
```

Markdown supports plain text, tables, code blocks, images, and MathJax formulas.

Inline formula:

```md
$E = mc^2$
```

Display formula:

```md
$$
QK^\top / \sqrt{d_k}
$$
```

For image-heavy notes, this structure keeps paths simple:

```text
posts/
  20250501/
    content.md
    ViT.png
```

Then reference the image in Markdown:

```md
![ViT](ViT.png)
```
