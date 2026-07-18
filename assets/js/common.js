(() => {
  const root = document.documentElement;
  const button = document.querySelector(".theme-button");
  const savedTheme = localStorage.getItem("wenhao-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  function setTheme(dark) {
    root.classList.toggle("dark", dark);
    if (button) {
      button.textContent = dark ? "浅色" : "深色";
      button.setAttribute("aria-label", dark ? "切换到浅色模式" : "切换到深色模式");
    }
    window.dispatchEvent(new CustomEvent("themechange", { detail: { dark } }));
  }

  setTheme(savedTheme ? savedTheme === "dark" : prefersDark);

  button?.addEventListener("click", () => {
    const dark = !root.classList.contains("dark");
    localStorage.setItem("wenhao-theme", dark ? "dark" : "light");
    setTheme(dark);
  });
})();
