# Adarsh Nair - Data Science Portfolio

<div align="center">

**Senior Data Science Manager | ML Engineer**

[View Portfolio](https://adarshnair.online/blog)

---

[![Deploy site](https://github.com/adarshnair01/blog/actions/workflows/deploy.yml/badge.svg)](https://github.com/adarshnair01/blog/actions/workflows/deploy.yml)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-adarshnair01-blue?logo=linkedin)](https://www.linkedin.com/in/adarshnair01/)
[![GitHub](https://img.shields.io/badge/GitHub-adarshnair01-black?logo=github)](https://github.com/adarshnair01)

</div>

## 🚀 About This Repo

This repository contains my personal portfolio and blog, built with [Astro](https://astro.build). It showcases my professional experience, data science projects, and technical writings.

### Key Sections:

- **[Blog](https://adarshnair.online/blog/)**: Deep dives into Machine Learning, AI, and Data Science.
- **[Projects](https://adarshnair.online/blog/projects/)**: Detailed write-ups of internal and open-source data science projects.
- **[CV](https://adarshnair.online/blog/cv/)**: My professional resume, including experience at Yum! Digital & Technology and Kvantum Inc.
- **[Repositories](https://adarshnair.online/blog/repositories/)**: Key open-source projects from my GitHub.

---

## 🛠 Featured Projects

| Project                           | Description                                                              | Year |
| :-------------------------------- | :----------------------------------------------------------------------- | :--- |
| **Fine-tuning LLM for Intents**   | Custom LLaMA 3/Mistral fine-tuning for high-accuracy intent recognition. | 2026 |
| **WhatsApp Chatbot using LLMs**   | Enterprise-grade RAG-based AI support bot integrated with WhatsApp.      | 2025 |
| **Recommendation Engine**         | Two-Tower TFRS architecture for scalable product personalization.        | 2022 |
| **Forecasting for Pharma Client** | Hierarchical time series SKU forecasting reducing inventory waste.       | 2020 |
| **Voice of Consumer**             | NLP-powered sentiment and trend analysis for social listening.           | 2019 |

---

## 🚀 Deployment & Development

The site lives in [`site/`](site/) and reads content from the repo root:

| What | Where |
| --- | --- |
| Blog posts | `_posts/YYYY-MM-DD-slug.md` (front matter: `title`, `date`, `excerpt`, `tags`) |
| Projects | `_projects/*.md` |
| CV and publications | `_data/cv.yml` |
| Affiliate recommendations | `_data/affiliates.yml` |
| Site settings (nav, analytics, AdSense, affiliates) | `site/src/site.ts` |

### Local development

Requires Node.js 22.23+ (24 recommended).

```bash
cd site
npm install
npm run dev        # http://localhost:4321/blog/ (search works only after a build)
npm run build      # static site + search index in site/dist/
npm run preview    # serve the build, including search
```

### Deployment

Every push to `master` that touches content or `site/` runs [`deploy.yml`](.github/workflows/deploy.yml), which builds the site and publishes `site/dist/` to the `gh-pages` branch. It also rebuilds daily, so posts dated in the future go live on their date.

Live at [https://adarshnair.online/blog/](https://adarshnair.online/blog/). More detail in [`site/README.md`](site/README.md).
