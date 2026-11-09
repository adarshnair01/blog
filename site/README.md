# Blog (Astro)

The blog at https://adarshnair.online/blog/ is now built with [Astro](https://astro.build).
Content still lives at the repo root, so writing is unchanged:

| What | Where |
| --- | --- |
| Posts | `../_posts/YYYY-MM-DD-slug.md` (front matter: `title`, `date`, `excerpt`, `tags`) |
| Projects | `../_projects/*.md` |
| CV / publications | `../_data/cv.yml` |
| Site settings (nav, socials, analytics, giscus) | `src/site.ts` |

Post URLs keep the old Jekyll shape `/blog/blog/<year>/<slug>/`.

```bash
npm install
npm run dev      # http://localhost:4321/blog/  (search index not available in dev)
npm run build    # static site + Pagefind search index in dist/
npm run preview  # serve dist/ with search
```

Pushing to `master` deploys `dist/` to the `gh-pages` branch via `.github/workflows/deploy.yml`.

## What's in it

- **SEO:** canonical URLs, Open Graph/Twitter tags, JSON-LD (`BlogPosting`, `BreadcrumbList`, `Blog`, `ProfilePage`, `WebSite` + search action), `sitemap.xml` with `lastmod`, RSS at `/blog/feed.xml`, year/tag/category archives, server-rendered math (KaTeX) and code highlighting, near-zero JS.
- **Interactive:** ⌘K / `/` full-text search (Pagefind), Explore page (keyword, topic, month timeline, sort; state in the URL), reading progress, sticky table of contents with scroll-spy, copy-code buttons, share, related posts, dark mode, lazy Mermaid diagrams and Giscus comments, hover prefetch and smooth page transitions.

## Monetization setup

- **Affiliates:** edit `../_data/affiliates.yml` (instructions at the top). Set your Amazon Associates tag in `src/site.ts → affiliates.amazonTag`. Cards appear on `/toolkit/`, at the end of every post (matched to the post's tags), mid-article, in the desktop sidebar and on the home page. Links use `rel="sponsored"`.
- **AdSense:** set `src/site.ts → adsense.client` (`ca-pub-…`). Optionally add ad-unit slot IDs for in-article, end-of-post and sidebar placements. The script loads only after the first interaction or 4s idle. `ads.txt` must be served from the **domain root** (`adarshnair.online/ads.txt`, i.e. the portfolio site).
- **Analytics:** GA4 `G-HQ1CHRXL4D` loads only on `adarshnair.online`. Custom events: `affiliate_click` (item_id, placement), `outbound_click`, `search`. Mark `affiliate_click` as a key event in GA → Admin → Events.
