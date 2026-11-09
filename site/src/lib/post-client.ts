// Interactions on article pages: reading progress, TOC scroll-spy, copy-code,
// share, back-to-top, lazy Mermaid and lazy Giscus comments.

interface PostConfig {
  giscus: { repo: string; repoId: string; category: string; categoryId: string };
  mermaid: boolean;
  title: string;
  url: string;
}

function toast(msg: string) {
  const el = document.querySelector<HTMLElement>(".toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1800);
}

function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function progressAndTop() {
  const bar = document.querySelector<HTMLElement>(".progress");
  const article = document.querySelector<HTMLElement>(".prose");
  const top = document.querySelector<HTMLElement>("[data-to-top]");
  if (!article) return;
  let ticking = false;
  const update = () => {
    ticking = false;
    const r = article.getBoundingClientRect();
    const total = r.height - window.innerHeight * 0.6;
    const p = Math.min(1, Math.max(0, -r.top / Math.max(total, 1)));
    if (bar) bar.style.transform = `scaleX(${p})`;
    top?.classList.toggle("show", window.scrollY > 900);
  };
  addEventListener("scroll", () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(update);
    }
  }, { passive: true });
  update();
  top?.addEventListener("click", () => scrollTo({ top: 0 }));
}

function scrollSpy() {
  const links = [...document.querySelectorAll<HTMLAnchorElement>(".toc a")];
  if (!links.length) return;
  const map = new Map(links.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
  const heads = [...map.keys()].map((id) => document.getElementById(id)).filter(Boolean) as HTMLElement[];
  const visible = new Set<string>();
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id);
      let active = heads.find((h) => visible.has(h.id));
      if (!active) active = [...heads].reverse().find((h) => h.getBoundingClientRect().top < 120);
      links.forEach((a) => a.classList.remove("active"));
      if (active) map.get(active.id)?.classList.add("active");
    },
    { rootMargin: "-80px 0px -60% 0px" },
  );
  heads.forEach((h) => io.observe(h));
}

function copyButtons() {
  document.querySelectorAll<HTMLPreElement>(".prose pre").forEach((pre) => {
    if (pre.closest(".code-wrap") || pre.querySelector("code.language-mermaid")) return;
    const wrap = document.createElement("div");
    wrap.className = "code-wrap";
    pre.replaceWith(wrap);
    wrap.append(pre);
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        btn.textContent = "Copied ✓";
      } catch {
        btn.textContent = "Press ⌘C";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
    wrap.append(btn);
  });
}

function share(cfg: PostConfig) {
  document.querySelector("[data-share]")?.addEventListener("click", async () => {
    if (navigator.share) {
      try {
        await navigator.share({ title: cfg.title, url: cfg.url });
        return;
      } catch {
        /* cancelled — fall through to copy */
      }
    }
    try {
      await navigator.clipboard.writeText(cfg.url);
      toast("Link copied");
    } catch {
      toast(cfg.url);
    }
  });
}

async function mermaid() {
  const blocks = [...document.querySelectorAll<HTMLElement>("pre > code.language-mermaid")];
  if (!blocks.length) return;
  const targets = blocks.map((code) => {
    const div = document.createElement("div");
    div.className = "mermaid";
    div.dataset.src = code.textContent ?? "";
    div.textContent = div.dataset.src;
    code.parentElement!.replaceWith(div);
    return div;
  });
  // @ts-expect-error remote ESM module
  const { default: m } = await import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs");
  const run = async () => {
    targets.forEach((t) => {
      t.removeAttribute("data-processed");
      t.textContent = t.dataset.src ?? "";
    });
    m.initialize({ startOnLoad: false, theme: currentTheme() === "dark" ? "dark" : "neutral", securityLevel: "strict" });
    await m.run({ nodes: targets }).catch(() => {});
  };
  await run();
  document.addEventListener("themechange", run);
}

function comments(cfg: PostConfig) {
  const host = document.querySelector<HTMLElement>(".giscus-host");
  const btn = document.querySelector<HTMLButtonElement>("[data-load-comments]");
  if (!host || !btn) return;
  let loaded = false;
  const load = () => {
    if (loaded) return;
    loaded = true;
    btn.remove();
    const s = document.createElement("script");
    s.src = "https://giscus.app/client.js";
    s.async = true;
    s.crossOrigin = "anonymous";
    const attrs: Record<string, string> = {
      "data-repo": cfg.giscus.repo,
      "data-repo-id": cfg.giscus.repoId,
      "data-category": cfg.giscus.category,
      "data-category-id": cfg.giscus.categoryId,
      "data-mapping": "pathname",
      "data-strict": "1",
      "data-reactions-enabled": "1",
      "data-emit-metadata": "0",
      "data-input-position": "top",
      "data-theme": currentTheme(),
      "data-lang": "en",
      "data-loading": "lazy",
    };
    for (const [k, v] of Object.entries(attrs)) s.setAttribute(k, v);
    host.append(s);
  };
  btn.addEventListener("click", load);
  // Auto-load once the reader actually reaches the end of the article.
  const io = new IntersectionObserver((e) => {
    if (e.some((x) => x.isIntersecting)) {
      load();
      io.disconnect();
    }
  }, { rootMargin: "400px" });
  io.observe(btn);
  document.addEventListener("themechange", () => {
    const frame = document.querySelector<HTMLIFrameElement>("iframe.giscus-frame");
    frame?.contentWindow?.postMessage({ giscus: { setConfig: { theme: currentTheme() } } }, "https://giscus.app");
  });
}

/** Place in-article blocks (ad, recommendation) after a paragraph at the given fraction of the post. */
function placeInline() {
  const prose = document.querySelector<HTMLElement>(".prose");
  if (!prose) return;
  const paras = [...prose.children].filter((el) => el.tagName === "P" && (el.textContent?.length ?? 0) > 80);
  const used = new Set<Element>();
  document.querySelectorAll<HTMLElement>("[data-insert-at]").forEach((block) => {
    if (!block.firstElementChild || paras.length < 6) return block.remove();
    let i = Math.min(paras.length - 2, Math.max(2, Math.round(paras.length * Number(block.dataset.insertAt))));
    while (used.has(paras[i]) && i < paras.length - 1) i++;
    used.add(paras[i]);
    paras[i].after(block);
    block.hidden = false;
  });
}

export function initPost() {
  placeInline();
  const raw = document.getElementById("post-config")?.textContent;
  if (!raw) return;
  const cfg = JSON.parse(raw) as PostConfig;
  progressAndTop();
  scrollSpy();
  copyButtons();
  share(cfg);
  comments(cfg);
  if (cfg.mermaid) mermaid();
}
