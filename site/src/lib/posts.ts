import { getCollection, type CollectionEntry } from "astro:content";

export type PostEntry = CollectionEntry<"posts">;

export interface Post {
  entry: PostEntry;
  id: string;
  slug: string;
  year: string;
  path: string; // path relative to base, e.g. /blog/2025/my-post/
  title: string;
  date: Date;
  description: string;
  tags: string[];
  categories: string[];
  readingMinutes: number;
  wordCount: number;
  hasMath: boolean;
  hasMermaid: boolean;
}

/** Jekyll-compatible slugify (default mode) so tag/category URLs stay the same. */
export function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function toList(v: string[] | string | undefined): string[] {
  if (!v) return [];
  const arr = Array.isArray(v) ? v : v.split(/[,\s]+/);
  return [...new Set(arr.map((t) => String(t).trim()).filter(Boolean))];
}

function plainText(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/[#>*_`~|-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function firstParagraph(md: string): string {
  const para = md
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .find((p) => p && !p.startsWith("#") && !p.startsWith("```") && !p.startsWith("$$"));
  return plainText(para ?? "");
}

export function clamp(text: string, max = 160): string {
  if (text.length <= max) return text;
  const cut = text.slice(0, max - 1);
  return `${cut.slice(0, cut.lastIndexOf(" ")).replace(/[,.;:]$/, "")}…`;
}

let cache: Post[] | undefined;

/** All published posts, newest first, with stable Jekyll-compatible URLs. */
export async function getPosts(): Promise<Post[]> {
  if (cache) return cache;
  const entries = await getCollection("posts", ({ data }) => !data.draft);

  const posts: Post[] = entries.map((entry) => {
    const body = entry.body ?? "";
    const words = plainText(body).split(" ").filter(Boolean).length;
    // Jekyll runs the filename through slugify for :title (collapses "--", trims "-").
    const slug = slugify(entry.id.replace(/^\d{4}-\d{2}-\d{2}-/, ""));
    const year = String(entry.data.date.getUTCFullYear());
    return {
      entry,
      id: entry.id,
      slug,
      year,
      path: `/blog/${year}/${slug}/`,
      title: entry.data.title,
      date: entry.data.date,
      description: entry.data.description ?? entry.data.excerpt ?? clamp(firstParagraph(body)),
      tags: toList(entry.data.tags),
      categories: toList(entry.data.categories),
      readingMinutes: Math.max(1, Math.round(words / 230)),
      wordCount: words,
      hasMath: /\$[^$\n]+\$|\\\(|\\\[/.test(body),
      hasMermaid: /```mermaid/.test(body),
    };
  });

  posts.sort((a, b) => b.date.valueOf() - a.date.valueOf() || b.id.localeCompare(a.id));

  // Two files can map to the same /year/slug/ URL. The newest keeps the clean URL
  // (Jekyll's behaviour), older ones get a numeric suffix so nothing is lost.
  const seen = new Map<string, number>();
  for (const p of posts) {
    const n = seen.get(p.path) ?? 0;
    seen.set(p.path, n + 1);
    if (n > 0) {
      p.slug = `${p.slug}-${n + 1}`;
      p.path = `/blog/${p.year}/${p.slug}/`;
    }
  }

  cache = posts;
  return posts;
}

export interface Term {
  name: string;
  slug: string;
  count: number;
}

export function collectTerms(posts: Post[], key: "tags" | "categories"): Term[] {
  const map = new Map<string, Term>();
  for (const p of posts) {
    for (const name of p[key]) {
      const slug = slugify(name);
      if (!slug) continue;
      const t = map.get(slug) ?? { name, slug, count: 0 };
      t.count++;
      map.set(slug, t);
    }
  }
  return [...map.values()].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

export function relatedPosts(post: Post, all: Post[], limit = 4): Post[] {
  const mine = new Set(post.tags.map(slugify));
  return all
    .filter((p) => p.id !== post.id)
    .map((p) => ({ p, score: p.tags.reduce((s, t) => s + (mine.has(slugify(t)) ? 1 : 0), 0) }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || b.p.date.valueOf() - a.p.date.valueOf())
    .slice(0, limit)
    .map((x) => x.p);
}

export function formatDate(d: Date): string {
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}
