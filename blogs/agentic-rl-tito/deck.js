(() => {
  const exportMode = new URLSearchParams(window.location.search).has("pdf");
  if (exportMode) {
    document.documentElement.classList.add("pdf-export");
    document.querySelectorAll("a[href]").forEach((link) => link.removeAttribute("href"));
  }

  const slides = Array.from(document.querySelectorAll(".slide"));
  const current = document.querySelector("[data-counter-current]");
  const total = document.querySelector("[data-counter-total]");
  const progress = document.querySelector("[data-progress]");
  const themeButton = document.querySelector(".theme-toggle");
  let index = 0;

  const clamp = (value) => Math.max(0, Math.min(slides.length - 1, value));
  const pad = (value) => String(value).padStart(2, "0");

  function indexFromHash() {
    const match = window.location.hash.match(/slide=(\d+)/);
    return match ? clamp(Number(match[1])) : 0;
  }

  function render(nextIndex, updateHash = true) {
    index = clamp(nextIndex);
    slides.forEach((slide, slideIndex) => {
      const active = slideIndex === index;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", String(!active));
    });
    current.textContent = pad(index + 1);
    total.textContent = pad(slides.length);
    progress.style.width = `${((index + 1) / slides.length) * 100}%`;
    document.title = `${slides[index].dataset.title} — TITO Gateway`;
    if (updateHash) history.replaceState(null, "", `#slide=${index}`);
  }

  function fit() {
    const scale = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
    document.documentElement.style.setProperty("--scale", String(scale));
  }

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  }

  function renderThemeButton() {
    if (themeButton) themeButton.textContent = currentTheme() === "dark" ? "Light" : "Dark";
  }

  if (themeButton) {
    themeButton.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (_) {}
      renderThemeButton();
    });
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "prev") render(index - 1);
    if (action === "next") render(index + 1);
  });

  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      render(index + 1);
    } else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) {
      event.preventDefault();
      render(index - 1);
    } else if (event.key === "Home") {
      render(0);
    } else if (event.key === "End") {
      render(slides.length - 1);
    }
  });

  window.addEventListener("resize", fit);
  window.addEventListener("hashchange", () => render(indexFromHash(), false));
  renderThemeButton();
  fit();
  render(indexFromHash(), false);
})();
