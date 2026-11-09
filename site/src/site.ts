export const SITE = {
  title: "Adarsh Nair",
  blogName: "Data Science Chronicles",
  description:
    "A deep dive into machine learning, AI, and data science — practical explainers, opinion and field notes by Adarsh Nair.",
  author: "Adarsh Nair",
  jobTitle: "Senior Data Science Manager",
  origin: "https://adarshnair.online",
  base: "/blog",
  lang: "en",
  locale: "en_US",
  postsPerPage: 12,
  googleAnalytics: "G-HQ1CHRXL4D",
  // Google Search Console / Bing "HTML tag" verification codes (content="..." value only).
  googleSiteVerification: "",
  bingSiteVerification: "",

  // Google AdSense. Paste your publisher ID ("ca-pub-1234567890123456") to turn ads on.
  // With only `client` set, Google's Auto ads decides placement. To control placement,
  // create "Display" ad units in AdSense and paste their slot IDs below.
  adsense: {
    client: "ca-pub-6421889451287368",
    slots: {
      inArticle: "", // inside the post, after the intro
      endOfPost: "", // after the article, before related posts
      sidebar: "", // under the table of contents (desktop)
    },
  },

  // Affiliate settings. Items live in ../_data/affiliates.yml.
  affiliates: {
    amazonTag: "dschronicles-20", // your Amazon Associates tracking ID, e.g. "adarshnair-21"
    amazonDomain: "www.amazon.com", // or "www.amazon.in" for Amazon India Associates
    disclosure:
      "Some links are affiliate links: if you buy through them I may earn a small commission, at no extra cost to you. I only recommend things I've used or would recommend to my own team.",
  },
  ogImage: "/assets/img/prof_pic_color.png",
  socials: {
    email: "adarshnair01@gmail.com",
    github: "https://github.com/adarshnair01",
    linkedin: "https://www.linkedin.com/in/adarshnair01/",
    scholar: "https://scholar.google.com/citations?user=qc6CJjYAAAAJ",
    website: "https://adarshnair.online",
  },
  giscus: {
    repo: "adarshnair01/blog",
    repoId: "MDkwOTpSZXBvc2l0b3J5MTQ4MjE1MDE0",
    category: "Q&A",
    categoryId: "DIC_kwDOCNWU5s4C4bJo",
  },
  nav: [
    { href: "/", label: "Blog" },
    { href: "/explore/", label: "Explore" },
    { href: "/toolkit/", label: "Toolkit" },
    { href: "/projects/", label: "Projects" },
    { href: "/publications/", label: "Publications" },
    { href: "/cv/", label: "CV" },
    { href: "/about/", label: "About" },
  ],
} as const;

/** Prefix an internal path with the base path. */
export function url(path = "/"): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${SITE.base}${p}`.replace(/\/{2,}/g, "/");
}

/** Absolute URL (for canonical, OG, feeds, sitemap). */
export function absUrl(path = "/"): string {
  return new URL(url(path), SITE.origin).href;
}
