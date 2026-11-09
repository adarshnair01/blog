import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

// Posts stay where they are (../_posts) so writing a new post works exactly as before.
// Only files named YYYY-MM-DD-slug.md are published, matching Jekyll's rule.
const posts = defineCollection({
  loader: glob({
    pattern: "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-*.md",
    base: "../_posts",
    generateId: ({ entry }) => entry.replace(/\.md$/, ""),
  }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    excerpt: z.string().optional(),
    description: z.string().optional(),
    author: z.string().default("Adarsh Nair"),
    tags: z.union([z.array(z.coerce.string()), z.string()]).optional(),
    categories: z.union([z.array(z.coerce.string()), z.string()]).optional(),
    draft: z.boolean().optional(),
  }),
});

const projects = defineCollection({
  loader: glob({
    pattern: "*.md",
    base: "../_projects",
    generateId: ({ entry }) => entry.replace(/\.md$/, ""),
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    img: z.string().optional(),
    importance: z.number().default(99),
    category: z.string().optional(),
  }),
});

export const collections = { posts, projects };
