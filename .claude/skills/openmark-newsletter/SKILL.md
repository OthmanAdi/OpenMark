---
name: openmark-newsletter
description: Compose a newsletter draft from Ahmad's OpenMark bookmarks. Use whenever Ahmad asks to draft, compose, write, or assemble a newsletter — phrases like "write me a newsletter on X", "make a newsletter about Y", "newsletter draft on Z", "put together a newsletter about W", "compose a newsletter", "draft my newsletter". Also use when Ahmad asks for a weekly write-up, roundup, or curated post. Pulls from OpenMark first, then WebFetches the chosen sources for quote-grade detail. Triggers on "newsletter", "draft a newsletter", "compose", "roundup", "weekly post", "curated post".
metadata:
  type: composition
---

# OpenMark — Newsletter Composer

Ahmad's newsletter voice: punchy, opinionated, no-fluff. Pulls only from things he's already saved (with optional WebFetch enrichment). Citations are the spine — never invent a source.

## House style

Three things matter:

1. **Personal voice, not LinkedIn corporate.** Sentence fragments OK. Light swearing OK (matches Ahmad's tone). Avoid: "in today's fast-paced world", "leveraging", "synergies", "delve into".
2. **Show the work.** Every claim either cites a bookmark URL or names a person/company. No floating assertions.
3. **One angle per issue.** Pick a thesis on input, build the issue around it. Don't write a generic "AI news this week" — write "this week, X mattered, here's why."

## Newsletter structure (use this template)

```
# [Title — punchy, ≤ 9 words, specific]

[Hook: 1–2 sentences. Why does this issue exist? What's the thesis?]

## What happened

[3–5 short paragraphs. Each paragraph cites at least one URL from OpenMark.
Use bold to highlight the names/orgs/products that matter.]

## Why it matters

[1–2 paragraphs. Ahmad's take. Connect the dots between the saves.]

## What I'm reading

[5–7 bullets. Title — URL — one-line why. These are the bookmarks the reader
should click. Pick from the highest-similarity OpenMark hits.]

## One more thing

[Optional. A single weird/funny/spiky bookmark that doesn't fit the thesis
but is too good to leave out. One sentence intro + URL.]

---
Sources cited above:
[Flat list of all URLs you referenced, in order of first mention.]
```

## Workflow

Before you draft anything, run this sequence. Use TodoWrite to plan it.

### 1. Get the topic + time window from Ahmad

If he didn't specify, ASK once: "What's the angle? And are we pulling from the last week / last month / all-time?"

### 2. Pull source material (parallel tool calls in one message)

For topic T and window W:

- `mcp__openmark__search_semantic(query=T, n=20)` — the core net
- ONE of (pick smartest):
  - `mcp__openmark__search_by_community(query=T, n=15)` — cluster view, broad topic
  - `mcp__openmark__search_by_category(category=<canonical>, query=T, n=15)` — if T maps to a category
- `mcp__openmark__find_recent(days=W, query=T, n=15)` — fresh saves filter (only works for LinkedIn nodes today; the call is cheap, run it anyway)
- `mcp__openmark__search_linkedin(query=T, n=10)` — LinkedIn has takes; takes make newsletters interesting

### 3. Pick 8–12 anchor bookmarks

Dedupe by URL. Score each by:
- Cross-strategy hits (in 2+ results) = +2
- `similarity ≥ 0.7` = +1
- `bm_score ≥ 5` = +1 (Ahmad rated it highly when he saved it)
- Has LinkedIn `author` field = +1 (human voice, good for quotes)

Take the top 10. These are your anchors.

### 4. Enrich the top 5 with WebFetch

For the top 5 anchors:

`WebFetch(url=<url>, prompt="Extract: (1) the single best quotable line (verbatim), (2) any numbers/dates/names that ground it, (3) the author or org behind it. Plain prose, no lists.")`

If WebFetch fails (paywall, 404, JS-only), drop that anchor and move on. Don't retry.

### 5. Draft

Fill the template. Rules while drafting:

- **What happened** paragraphs MUST cite. Inline links: `[descriptive phrase](URL)`.
- **What I'm reading** is the spine. 5–7 items. Each has Title — URL — one-line why.
- **Sources cited** at the bottom is a flat list — every URL you used, exactly as it appeared in tool output. No invented URLs. Ever.

### 6. Save and surface

Save the draft to `drafts/newsletter-YYYY-MM-DD-<slug>.md` in the project root. Then output a short summary to Ahmad:

```
Draft saved: drafts/newsletter-2026-05-14-context-engineering.md
Word count: 612
Sources used: 11 OpenMark hits + 4 WebFetch extracts
```

Don't paste the full draft into chat unless Ahmad asks. He'll read the file.

## What NOT to do

- Do NOT invent statistics. If a claim needs a number, it comes from WebFetch output or a saved bookmark, with the URL right next to it.
- Do NOT write "I think" or "in my opinion." This is Ahmad's voice. Write declaratively.
- Do NOT include URLs you didn't see in tool output. Never. Not even as "you might also like."
- Do NOT pad with "in conclusion" / "to wrap up" / "the bottom line is." End on the strongest line.

## Voice calibration examples

| Generic | Ahmad's voice |
|---|---|
| "There has been significant progress in agent frameworks recently." | "Agent frameworks finally stopped pretending. Three releases this week made that obvious." |
| "Many companies are exploring RAG architectures." | "Everyone's still building the same RAG stack from 2024. Two saves this week argue we should stop." |
| "I would recommend looking into ..." | "Check this: [link]." |

When in doubt: shorter sentence, more specific noun.
