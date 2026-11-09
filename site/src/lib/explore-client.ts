interface Item {
  t: string;
  u: string;
  d: string;
  e: string;
  g: [string, string][];
  m: number;
}

const PAGE = 30;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const fmt = (d: string) =>
  new Date(`${d}T00:00:00Z`).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });

export async function initExplore() {
  const root = document.querySelector<HTMLElement>("[data-explore]");
  if (!root) return;
  const q = root.querySelector<HTMLInputElement>("#q")!;
  const sort = root.querySelector<HTMLSelectElement>("#sort")!;
  const chips = [...root.querySelectorAll<HTMLButtonElement>("[data-tag]")];
  const timeline = root.querySelector<HTMLElement>(".timeline")!;
  const count = root.querySelector<HTMLElement>("[data-count]")!;
  const reset = root.querySelector<HTMLButtonElement>("[data-reset]")!;
  const list = document.querySelector<HTMLElement>("[data-results]")!;
  const more = document.querySelector<HTMLButtonElement>("[data-more]")!;

  const items: Item[] = await fetch(root.dataset.src!).then((r) => r.json());
  const params = new URLSearchParams(location.search);
  const state = {
    q: params.get("q") ?? "",
    tags: new Set((params.get("tag") ?? "").split(",").filter(Boolean)),
    month: params.get("month") ?? "",
    sort: params.get("sort") ?? "new",
    shown: PAGE,
  };
  q.value = state.q;
  sort.value = state.sort;

  // Monthly histogram
  const months = new Map<string, number>();
  for (const it of items) months.set(it.d.slice(0, 7), (months.get(it.d.slice(0, 7)) ?? 0) + 1);
  const keys = [...months.keys()].sort();
  if (keys.length) {
    const [y0, m0] = keys[0].split("-").map(Number);
    const [y1, m1] = keys[keys.length - 1].split("-").map(Number);
    const all: string[] = [];
    for (let y = y0, m = m0; y < y1 || (y === y1 && m <= m1); m === 12 ? (y++, (m = 1)) : m++)
      all.push(`${y}-${String(m).padStart(2, "0")}`);
    const max = Math.max(...months.values());
    timeline.innerHTML = all
      .map((k) => {
        const n = months.get(k) ?? 0;
        const label = new Date(`${k}-01T00:00:00Z`).toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
        return `<button type="button" data-month="${k}" style="height:${Math.max(4, (n / max) * 100)}%" title="${label}: ${n} article${n === 1 ? "" : "s"}" aria-label="${label}, ${n} articles" aria-pressed="false"></button>`;
      })
      .join("");
  }
  const bars = [...timeline.querySelectorAll<HTMLButtonElement>("[data-month]")];

  // Unknown tag from URL (e.g. coming from a small tag page) → add a chip for it.
  for (const slug of state.tags) {
    if (!chips.some((c) => c.dataset.tag === slug)) {
      const name = items.flatMap((i) => i.g).find(([s]) => s === slug)?.[1] ?? slug;
      const b = document.createElement("button");
      b.className = "chip";
      b.type = "button";
      b.dataset.tag = slug;
      b.textContent = name;
      chips[0]?.parentElement?.prepend(b);
      chips.unshift(b);
    }
  }

  const sync = () => {
    const p = new URLSearchParams();
    if (state.q) p.set("q", state.q);
    if (state.tags.size) p.set("tag", [...state.tags].join(","));
    if (state.month) p.set("month", state.month);
    if (state.sort !== "new") p.set("sort", state.sort);
    const qs = p.toString();
    history.replaceState(null, "", qs ? `?${qs}` : location.pathname);
  };

  const render = () => {
    const terms = state.q.toLowerCase().split(/\s+/).filter(Boolean);
    let res = items.filter((it) => {
      if (state.month && !it.d.startsWith(state.month)) return false;
      if (state.tags.size && ![...state.tags].every((t) => it.g.some(([s]) => s === t))) return false;
      if (terms.length) {
        const hay = `${it.t} ${it.e} ${it.g.map((g) => g[1]).join(" ")}`.toLowerCase();
        if (!terms.every((w) => hay.includes(w))) return false;
      }
      return true;
    });
    const cmp: Record<string, (a: Item, b: Item) => number> = {
      new: (a, b) => b.d.localeCompare(a.d),
      old: (a, b) => a.d.localeCompare(b.d),
      long: (a, b) => b.m - a.m,
      short: (a, b) => a.m - b.m,
    };
    res = res.sort(cmp[state.sort] ?? cmp.new);

    chips.forEach((c) => c.setAttribute("aria-pressed", String(state.tags.has(c.dataset.tag!))));
    bars.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.month === state.month)));
    const filtered = !!(state.q || state.tags.size || state.month);
    reset.hidden = !filtered;
    count.textContent = `${res.length} of ${items.length} articles${state.month ? ` · ${state.month}` : ""}`;

    list.innerHTML = res.length
      ? res
          .slice(0, state.shown)
          .map(
            (it) => `<li><a href="${it.u}">${esc(it.t)}</a><p>${esc(it.e)}</p><div class="meta"><time datetime="${it.d}">${fmt(it.d)}</time><span>${it.m} min read</span>${it.g
              .slice(0, 3)
              .map(([s, n]) => `<button type="button" class="tag" data-add-tag="${s}">${esc(n)}</button>`)
              .join("")}</div></li>`,
          )
          .join("")
      : `<li class="muted">No articles match. Try removing a filter.</li>`;
    more.hidden = res.length <= state.shown;
    more.textContent = `Show more (${res.length - state.shown} left)`;
    sync();
  };

  let t: number | undefined;
  q.addEventListener("input", () => {
    clearTimeout(t);
    t = window.setTimeout(() => {
      state.q = q.value.trim();
      state.shown = PAGE;
      render();
    }, 120);
  });
  sort.addEventListener("change", () => {
    state.sort = sort.value;
    render();
  });
  const toggleTag = (slug: string) => {
    state.tags.has(slug) ? state.tags.delete(slug) : state.tags.add(slug);
    state.shown = PAGE;
    render();
  };
  chips.forEach((c) => c.addEventListener("click", () => toggleTag(c.dataset.tag!)));
  list.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>("[data-add-tag]");
    if (b) {
      if (!chips.some((c) => c.dataset.tag === b.dataset.addTag)) {
        state.tags.add(b.dataset.addTag!);
        location.search = new URLSearchParams({ ...Object.fromEntries(new URLSearchParams(location.search)), tag: [...state.tags].join(",") }).toString();
        return;
      }
      toggleTag(b.dataset.addTag!);
      root.scrollIntoView({ block: "start" });
    }
  });
  timeline.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>("[data-month]");
    if (!b) return;
    state.month = state.month === b.dataset.month ? "" : b.dataset.month!;
    state.shown = PAGE;
    render();
  });
  reset.addEventListener("click", () => {
    state.q = q.value = "";
    state.tags.clear();
    state.month = "";
    state.shown = PAGE;
    render();
  });
  more.addEventListener("click", () => {
    state.shown += PAGE;
    render();
  });

  render();
}
