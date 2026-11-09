import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { getPosts, collectTerms } from "../lib/posts";
import { SITE, absUrl } from "../site";

export const GET: APIRoute = async () => {
  const posts = await getPosts();
  const projects = await getCollection("projects");
  const latest = posts[0]?.date.toISOString().slice(0, 10);
  const pages = Math.ceil(posts.length / SITE.postsPerPage);

  const entries: { loc: string; lastmod?: string }[] = [
    { loc: absUrl("/"), lastmod: latest },
    { loc: absUrl("/explore/"), lastmod: latest },
    { loc: absUrl("/about/") },
    { loc: absUrl("/projects/") },
    { loc: absUrl("/publications/") },
    { loc: absUrl("/cv/") },
    { loc: absUrl("/repositories/") },
    { loc: absUrl("/toolkit/") },
    { loc: absUrl("/blog/tag/"), lastmod: latest },
    ...Array.from({ length: pages - 1 }, (_, i) => ({ loc: absUrl(`/page/${i + 2}/`) })),
    ...[...new Set(posts.map((p) => p.year))].map((y) => ({ loc: absUrl(`/blog/${y}/`) })),
    ...posts.map((p) => ({ loc: absUrl(p.path), lastmod: p.date.toISOString().slice(0, 10) })),
    ...collectTerms(posts, "tags")
      .filter((t) => t.count >= 2)
      .map((t) => ({ loc: absUrl(`/blog/tag/${t.slug}/`) })),
    ...collectTerms(posts, "categories").map((t) => ({ loc: absUrl(`/blog/category/${t.slug}/`) })),
    ...projects.map((p) => ({ loc: absUrl(`/projects/${p.id}/`) })),
  ];

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${entries.map((e) => `  <url><loc>${e.loc}</loc>${e.lastmod ? `<lastmod>${e.lastmod}</lastmod>` : ""}</url>`).join("\n")}
</urlset>
`;
  return new Response(xml, { headers: { "Content-Type": "application/xml" } });
};
