# OG image placeholder

This site's social card image (`/og.png`) is referenced by every page's
`<meta property="og:image">` tag. It's not committed yet because designing it
properly is part of the deferred design pass (P2.5).

Specs to hit when generating it:
- 1200 × 630 px (Twitter / LinkedIn / Facebook all support this aspect)
- < 200 KB
- Strong text contrast (visible on dark + light backgrounds)
- Include: OpenMark wordmark + 1-line tagline
- File: `site/public/og.png` (Astro serves it at `/og.png`)

Until this file exists, every social share will 404 the image and fall back to
the parent host's default crawl preview (often the favicon).
