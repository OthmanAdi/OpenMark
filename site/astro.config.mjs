import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from '@astrojs/tailwind';

// SITE is the canonical URL of the deployed site. Override via the
// PUBLISH_BASE_URL env var when running `astro build` from CI so the
// sitemap + RSS feed use the right absolute URLs.
const SITE = process.env.PUBLISH_BASE_URL || 'https://openmark.dev';

export default defineConfig({
  site: SITE,
  output: 'static',
  integrations: [
    mdx(),
    sitemap(),
    tailwind({ applyBaseStyles: false }),
  ],
  markdown: {
    shikiConfig: {
      theme: 'github-light',
      wrap: true,
    },
  },
});
