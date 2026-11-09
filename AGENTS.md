# Agent guidelines

Personal blog of Adarsh Nair, served at https://adarshnair.online/blog/. Built with **Astro** (static output) in `site/`; content lives at the repo root. The old Jekyll/al-folio setup has been removed.

## Layout

- `_posts/YYYY-MM-DD-slug.md`: posts. Only files with a date prefix are published; posts dated in the future are skipped until that date.
- `_projects/*.md`, `_data/cv.yml`, `_data/affiliates.yml`: projects, CV/publications, affiliate cards.
- `site/src/site.ts`: nav, socials, GA4 ID, AdSense publisher/slot IDs, Amazon tag, giscus.
- `site/src/lib/posts.ts`: post loading, URL slugs, related posts.
- `site/src/pages/`: routes. `site/src/styles/global.css`: all styling (CSS variables on `:root`, dark mode via `[data-theme]`).

## Commands

```bash
cd site
npm install
npm run dev      # http://localhost:4321/blog/
npm run build    # astro build + pagefind index -> site/dist
npm run preview  # serve dist (search works here)
```

## Rules

- **Never change post URLs.** They must stay `/blog/blog/<year>/<slug>/` (base `/blog` + Jekyll's old permalink); the slug is the filename after the date, run through Jekyll-style slugify. Changing this breaks search rankings and backlinks.
- **Lockfile:** after changing dependencies, regenerate `site/package-lock.json` cleanly and check it resolves for Linux (`npx npm@11 ci --dry-run --os=linux --cpu=x64 --libc=glibc`). A macOS-only lockfile once broke CI.
- **Domain root files** (`ads.txt`, `robots.txt`, AdSense verification tag) are served by the separate portfolio repo (`adarshnair01.github.io`), not this one.
- Affiliate links must keep `rel="sponsored"` and the disclosure; ads stay labelled "Advertisement".
- Run `npm run build` before committing; deploy is `.github/workflows/deploy.yml` on push to `master`.
