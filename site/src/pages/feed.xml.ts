import rss from "@astrojs/rss";
import type { APIContext } from "astro";
import { getPosts } from "../lib/posts";
import { SITE, absUrl } from "../site";

export async function GET(context: APIContext) {
  const posts = (await getPosts()).slice(0, 50);
  return rss({
    title: SITE.blogName,
    description: SITE.description,
    site: absUrl("/"),
    items: posts.map((p) => ({
      title: p.title,
      link: absUrl(p.path),
      pubDate: p.date,
      description: p.description,
      categories: p.tags,
      author: `${SITE.socials.email} (${SITE.author})`,
    })),
    customData: `<language>${SITE.lang}</language>`,
  });
}
