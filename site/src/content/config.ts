/**
 * Astro Content Collection schema.
 *
 * Each newsletter issue lives under `src/content/newsletters/<slug>.mdx`
 * with this frontmatter — written by `openmark.publish.orchestrator._write_mdx`.
 */

import { defineCollection, z } from 'astro:content';

const newsletters = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    kicker: z.string().optional(),
    language: z.string().default('en'),
    preheader: z.string().optional(),
    published_at: z.string().or(z.date()).transform(v => new Date(v)),
    draft: z.boolean().optional().default(false),
  }),
});

export const collections = { newsletters };
