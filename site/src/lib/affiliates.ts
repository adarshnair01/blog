import fs from "node:fs";
import path from "node:path";
import { load as yamlLoad, JSON_SCHEMA } from "js-yaml";
import { SITE } from "../site";

export interface Affiliate {
  id: string;
  name: string;
  category: string;
  blurb: string;
  why?: string;
  href: string;
  host: string;
  cta: string;
  badge?: string;
  featured: boolean;
  /** false = plain recommendation (no commission): not marked sponsored. */
  paid: boolean;
  topics: string[];
}

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

let cache: Affiliate[] | undefined;

/** Reads ../_data/affiliates.yml and builds final (tagged) links. */
export function getAffiliates(): Affiliate[] {
  if (cache) return cache;
  const file = path.resolve(process.cwd(), "../_data/affiliates.yml");
  if (!fs.existsSync(file)) return (cache = []);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw = (yamlLoad(fs.readFileSync(file, "utf8"), { schema: JSON_SCHEMA }) as any[]) ?? [];
  const { amazonTag, amazonDomain } = SITE.affiliates;

  cache = raw
    .filter((r) => r && r.id && r.name && (r.url || r.amazon))
    .map((r) => {
      let href: string;
      if (r.amazon) {
        const a = String(r.amazon).trim();
        const u = /^[A-Z0-9]{10}$/.test(a)
          ? new URL(`https://${amazonDomain}/dp/${a}`)
          : new URL(`https://${amazonDomain}/s?k=${encodeURIComponent(a)}`);
        if (amazonTag) u.searchParams.set("tag", amazonTag);
        href = u.href;
      } else {
        href = String(r.url);
      }
      return {
        id: String(r.id),
        name: String(r.name),
        category: String(r.category ?? "tools").toLowerCase(),
        blurb: String(r.blurb ?? ""),
        why: r.why ? String(r.why) : undefined,
        href,
        host: new URL(href).hostname.replace(/^www\./, ""),
        cta: String(r.cta ?? "Check it out"),
        badge: r.badge ? String(r.badge) : undefined,
        featured: Boolean(r.featured),
        paid: r.affiliate !== false,
        topics: (r.topics ?? []).map((t: unknown) => norm(String(t))),
      } satisfies Affiliate;
    });
  return cache;
}

/** Best-matching recommendations for a set of post tags; falls back to featured picks. */
export function affiliatesFor(tags: string[], limit = 3): Affiliate[] {
  const all = getAffiliates();
  const mine = new Set(tags.map(norm));
  const scored = all
    .map((a, i) => ({ a, i, score: a.topics.reduce((s, t) => s + (mine.has(t) ? 1 : 0), 0) }))
    .filter((x) => x.score > 0)
    .sort(
      (x, y) =>
        y.score - x.score || Number(y.a.paid) - Number(x.a.paid) || Number(y.a.featured) - Number(x.a.featured) || x.i - y.i,
    )
    .map((x) => x.a);
  const picks = [...scored];
  for (const a of all.filter((a) => a.featured)) if (picks.length < limit && !picks.includes(a)) picks.push(a);
  // Mix categories so a post shows e.g. a book + a course + a tool rather than three books.
  const out: Affiliate[] = [];
  const seenCat = new Set<string>();
  for (const a of picks) if (out.length < limit && !seenCat.has(a.category)) (out.push(a), seenCat.add(a.category));
  for (const a of picks) if (out.length < limit && !out.includes(a)) out.push(a);
  // Put an earning recommendation first: it gets the in-article slot.
  return out.sort((a, b) => Number(b.paid) - Number(a.paid));
}

export const categoryLabel: Record<string, string> = {
  books: "Books",
  courses: "Courses",
  tools: "Tools",
  ai: "AI assistants",
};
export const categoryIcon: Record<string, string> = {
  books: "📘",
  courses: "🎓",
  tools: "🛠️",
  ai: "🤖",
};
