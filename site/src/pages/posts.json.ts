import type { APIRoute } from "astro";
import { getPosts, slugify } from "../lib/posts";
import { url } from "../site";

// Compact index used by the interactive Explore page.
export const GET: APIRoute = async () => {
  const posts = await getPosts();
  const data = posts.map((p) => ({
    t: p.title,
    u: url(p.path),
    d: p.date.toISOString().slice(0, 10),
    e: p.description,
    g: p.tags.map((name) => [slugify(name), name]),
    m: p.readingMinutes,
  }));
  return new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json" } });
};
