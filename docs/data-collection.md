# Data Collection Guide

Everything you need to collect your saved content from each source before running the ingest pipeline.

---

## 1. Raindrop.io

OpenMark pulls **all your Raindrop collections automatically** via the official REST API. You just need a token.

**Steps:**
1. Go to [app.raindrop.io/settings/integrations](https://app.raindrop.io/settings/integrations)
2. Under "For Developers" → click **Create new app**
3. Copy the **Test token** (permanent, no expiry)
4. Add to `.env`:
   ```
   RAINDROP_TOKEN=your-token-here
   ```

The pipeline fetches every collection, every sub-collection, and every unsorted raindrop automatically. No manual export needed.

---

## 2. Browser Bookmarks (Edge / Chrome / Firefox)

Export your bookmarks as an HTML file in the Netscape bookmark format (all browsers support this).

**Edge:**
`Settings → Favourites → ··· (three dots) → Export favourites` → save as `favorites.html`

**Chrome:**
`Bookmarks Manager (Ctrl+Shift+O) → ··· → Export bookmarks` → save as `bookmarks.html`

**Firefox:**
`Bookmarks → Manage Bookmarks → Import and Backup → Export Bookmarks to HTML`

**After exporting:**
- Place the HTML file(s) in your `raindrop-mission` folder (or wherever `RAINDROP_MISSION_DIR` points)
- The pipeline (`merge.py`) looks for `favorites_*.html` and `bookmarks_*.html` patterns
- It parses the Netscape format and extracts URLs + titles + folder structure

> **Tip:** Export fresh before every ingest to capture new bookmarks.

---

## 3. LinkedIn Saved Posts

LinkedIn has no public API for saved posts. OpenMark uses LinkedIn's internal **Voyager GraphQL API** — the same API the LinkedIn web app uses internally.

**This is the exact endpoint used:**
```
https://www.linkedin.com/voyager/api/graphql
  ?variables=(start:0,count:10,paginationToken:null,
    query:(flagshipSearchIntent:SEARCH_MY_ITEMS_SAVED_POSTS))
  &queryId=voyagerSearchDashClusters.05111e1b90ee7fea15bebe9f9410ced9
```

**How to get your session cookie:**

1. Log into LinkedIn in your browser
2. Open DevTools (`F12`) → **Application** tab → **Cookies** → `https://www.linkedin.com`
3. Find the cookie named `li_at` — copy its value
4. Also find `JSESSIONID` — copy its value (used as CSRF token, format: `ajax:XXXXXXXXXXXXXXXXXX`)

**Run the fetch script:**
```bash
python raindrop-mission/linkedin_fetch.py
```
Paste your `li_at` value when prompted.

**Output:** `raindrop-mission/linkedin_saved.json` — 1,260 saved posts with author, content, and URL.

**Pagination:** LinkedIn returns 10 posts per page. The script detects end of results when no `nextPageToken` is returned. With 1,260 posts that's ~133 pages.

> **Important:** The `queryId` (`voyagerSearchDashClusters.05111e1b90ee7fea15bebe9f9410ced9`) is hardcoded in LinkedIn's JavaScript bundle and can change with LinkedIn deployments. If the script returns 0 results, intercept a fresh request from your browser's Network tab — filter for `voyagerSearchDashClusters`, copy the new `queryId`.

> **Personal use only.** This method is not officially supported by LinkedIn. Do not use for scraping at scale.

---

## 4. YouTube

Uses the official **YouTube Data API v3** via OAuth 2.0. Collects liked videos, watch later playlist, and any saved playlists.

**One-time setup:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g. "OpenMark")
3. Enable **YouTube Data API v3** (APIs & Services → Enable APIs)
4. Create credentials: **OAuth 2.0 Client ID** → Desktop App
5. Download the JSON file — rename it to `client_secret.json` and place it in `raindrop-mission/`
6. Go to **OAuth consent screen** → Test users → add your Google account email

**Run the fetch script:**
```bash
python raindrop-mission/youtube_fetch.py
```
A browser window opens for Google sign-in. After auth, a token is cached locally — you won't need to auth again.

**Output:** `raindrop-mission/youtube_MASTER.json` with:
- `liked_videos` — videos you've liked (up to ~3,200 via API limit)
- `watch_later` — requires Google Takeout (see below)
- `playlists` — saved playlists

**Watch Later via Google Takeout:**
YouTube's API does not expose Watch Later directly. Export it via [takeout.google.com](https://takeout.google.com):
- Select only **YouTube** → **Playlists** → Download
- Extract the CSV file named `Watch later-videos.csv`
- Place it in `raindrop-mission/`
- The `youtube_organize.py` script fetches video titles via API and includes them in `youtube_MASTER.json`

---

## 5. daily.dev Bookmarks

daily.dev does not provide a public API. Use the included browser console script to extract bookmarks directly from the page.

**Steps:**
1. Go to [app.daily.dev](https://app.daily.dev) → **Bookmarks**
2. Scroll all the way down to load all bookmarks
3. Open DevTools → **Console** tab
4. Paste and run `raindrop-mission/dailydev_console_script.js`
5. The script copies a JSON array to your clipboard
6. Paste into a file named `dailydev_bookmarks.json` in `raindrop-mission/`

> The script filters for `/posts/` URLs only — it ignores profile links, squad links, and other noise.

---

## Summary

| Source | Method | Output file |
|--------|--------|-------------|
| Raindrop | REST API (auto) | pulled live |
| Edge/Chrome bookmarks | HTML export | `favorites.html` / `bookmarks.html` |
| LinkedIn saved posts | Voyager GraphQL + session cookie | `linkedin_saved.json` |
| YouTube liked/playlists | YouTube Data API v3 + OAuth | `youtube_MASTER.json` |
| YouTube watch later | Google Takeout CSV | included in `youtube_MASTER.json` |
| daily.dev bookmarks | Browser console script | `dailydev_bookmarks.json` |

Once all files are in place, run:
```bash
python scripts/ingest.py
```
