import type { APIRoute } from "astro";
import { absUrl } from "../site";

// Note: crawlers only read robots.txt at the domain root (adarshnair.online/robots.txt),
// which is served by the main portfolio site. Add the Sitemap line there too.
export const GET: APIRoute = () =>
  new Response(`User-agent: *\nAllow: /\n\nSitemap: ${absUrl("/sitemap.xml")}\n`, {
    headers: { "Content-Type": "text/plain" },
  });
