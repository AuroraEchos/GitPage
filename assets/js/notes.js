// 笔记列表渲染逻辑。数据来自 notes-data.js（由 tools/build_notes.py 生成）。

const list = document.querySelector("#notes-list");
const search = document.querySelector("#note-search");
const filterBar = document.querySelector("#note-filters");
const catalog = window.noteCatalog;
let activeCategory = "all";

function renderLoadError() {
  const error = document.createElement("p");
  error.className = "empty";
  error.textContent = "笔记数据加载失败，请稍后重试。";
  list.replaceChildren(error);
  search.disabled = true;
}

if (!catalog || !Array.isArray(catalog.notes) || !catalog.categoryLabels) {
  renderLoadError();
} else {
  initializeNotes();
}

function initializeNotes() {
  const { notes, categoryLabels } = catalog;

  function renderFilters() {
    const categories = ["all", ...Object.keys(categoryLabels)];
    categories.forEach((category) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.category = category;
      button.textContent = category === "all" ? "全部" : categoryLabels[category];
      button.classList.toggle("active", category === activeCategory);
      button.setAttribute("aria-pressed", String(category === activeCategory));
      button.setAttribute("aria-controls", "notes-list");
      button.addEventListener("click", () => {
        activeCategory = category;
        filterBar.querySelectorAll("button").forEach((item) => {
          const isActive = item === button;
          item.classList.toggle("active", isActive);
          item.setAttribute("aria-pressed", String(isActive));
        });
        renderNotes();
      });
      filterBar.append(button);
    });
  }

  function renderNotes() {
    const keyword = search.value.trim().toLowerCase();
    const filtered = notes.filter((note) => {
      const categoryMatch = activeCategory === "all" || note.category === activeCategory;
      const queryMatch = !keyword || note.title.toLowerCase().includes(keyword);
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
      const link = document.createElement("a");
      const time = document.createElement("time");
      const title = document.createElement("span");
      const titleText = document.createElement("strong");
      const category = document.createElement("small");

      link.className = "note-item";
      link.href = `reader.html?src=${encodeURIComponent(note.path)}`;
      time.dateTime = note.date.replaceAll(".", "-");
      time.textContent = note.date;
      title.className = "note-title";
      titleText.textContent = note.title;
      category.textContent = categoryLabels[note.category];
      title.append(titleText, category);
      link.append(time, title);

      list.append(link);
    });
  }

  search.addEventListener("input", renderNotes);
  renderFilters();
  renderNotes();
}
