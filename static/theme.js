// 主题切换：默认亮色，选择存 localStorage；页面里凡是 [data-theme-toggle] 的元素都能点
// 用法：在 <head> 里同步引入（尽早执行，避免闪白/闪黑），并在页面里放一个
//       <button data-theme-toggle></button>
(function () {
  const KEY = "doubao-asr-theme";

  function current() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function sync() {
    const dark = current() === "dark";
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.textContent = dark ? "☀️ 亮色" : "🌙 暗色";
      btn.title = dark ? "切换到亮色主题" : "切换到暗色主题";
    });
  }

  // 尽早确定主题（默认 light，只有明确选了 dark 才是暗色）
  document.documentElement.dataset.theme = localStorage.getItem(KEY) === "dark" ? "dark" : "light";

  window.toggleTheme = function () {
    document.documentElement.dataset.theme = current() === "dark" ? "light" : "dark";
    try { localStorage.setItem(KEY, document.documentElement.dataset.theme); } catch (_) {}
    sync();
  };

  function bind() {
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      if (btn.dataset.bound) return;
      btn.dataset.bound = "1";
      btn.addEventListener("click", window.toggleTheme);
    });
    sync();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
  sync();
})();
