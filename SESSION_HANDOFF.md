# OpenMark — session handoff

**Written:** 2026-05-15 ~20:55 local.
**For:** the next Claude Code session that loads this repo.
**Action:** read this top to bottom before touching anything. Sections 1+2 are state, 3 is what was done today, 4 is what's pending, 5 is quick-restart recipes.

---

## 1. Live runtime state (as of handoff)

| Component | Status | Where |
|---|---|---|
| Neo4j Desktop | Running | `bolt://127.0.0.1:7687`, database `openmark` |
| OpenMark Gradio UI | Running, **`grok-4.3` via Azure Foundry, reasoning=high** | `http://127.0.0.1:7860` |
| Ollama | DOWN (not needed in Azure mode) | would be `:11434` |
| Open-Generative-AI Studio | DOWN | would be `:3000` |
| Copilot-API proxy | DELETED (task removed earlier) | was `:4141` |

Neo4j contents: **13,831 bookmarks** · 5,469 tags · 19 categories · 0 communities (Louvain blocked, see §4).

The OpenMark UI process PID can be found via `netstat -ano | findstr :7860`. Started with `python openmark/ui/app.py` after the .env flip described in §5.

---

## 2. Current `.env` shape (don't print, don't commit)

Critical keys for the agent route (.env is gitignored, lives at repo root):

```
AGENT_PROVIDER=azure
AZURE_DEPLOYMENT_EXECUTOR=grok-4.3
AZURE_DEPLOYMENT_CLASSIFIER=grok-4.3
AZURE_REASONING_EXECUTOR=high
NEO4J_DATABASE=openmark
TAVILY_API_KEY=tvly-dev-…   (rotate when you can — was pasted in chat today)
BONSAI_URL=http://localhost:11434/v1   (unused in Azure mode)
BONSAI_MODEL=hermes3:8b                 (unused in Azure mode)
OPENMARK_PORT default = 7860            (no key in .env; shell override possible)
```

To switch route:

- **Local Hermes 3** via Ollama: set `AGENT_PROVIDER=local`, `ollama serve`, restart UI.
- **Azure codex 5.3**: change `AZURE_DEPLOYMENT_EXECUTOR=gpt-5.3-codex` and `AZURE_DEPLOYMENT_CLASSIFIER=gpt-5-mini`.
- **Azure grok-4-20-reasoning**: change `AZURE_DEPLOYMENT_EXECUTOR=grok-4-20-reasoning`. Costs latency (4-agent debate).
- **Dual side-by-side Grok**: launch two python processes with different shell env (`$env:AZURE_DEPLOYMENT_EXECUTOR='…'; $env:OPENMARK_PORT='7861'; python ui/app.py`). The `load_dotenv(override=False)` + empty-string scrub in `openmark/config.py` makes shell env win over `.env`.

---

## 3. What today shipped (commits)

Three commits today in chronological order, all pushed to `OthmanAdi/OpenMark@main`:

1. **`9419b4b`** — `feat(agent): web research stack — Tavily + DDG + GitHub + Reddit + Crawl`
   - Added `openmark/agent/web.py` with 6 backends: Tavily search/extract/crawl, DuckDuckGo fallback, GitHub repo intel, Reddit JSON.
   - 6 new `@tool` wrappers in `openmark/agent/tools.py` (now 21 tools total, was 15).
   - Mirrored in `openmark/mcp/server.py`.
   - New skill `.claude/skills/openmark-repo-research/SKILL.md` — 4-step recipe github_repo_intel → parallel web_search+reddit_search → web_fetch top picks → compose report.
   - Tavily MCP added to `.mcp.json` for Claude Code (not in-app agent).

2. **`f1c5fc1`** — `feat(agent): Grok routing on Azure Foundry + per-process env + empty-string env scrub`
   - `openmark/agent/llms.py` adds `_azure_grok()` for xAI Grok deployments. Foundry needs explicit `model=<deployment>` in the request body (otherwise `Invalid JSON data: model: invalid type: null`). All modern Grok (4.1/4.3/4.20-reasoning) accept `reasoning_effort` top-level.
   - `openmark/config.py` switched to `load_dotenv(override=False)` so shell env beats `.env` (needed for dual-process launches). BUT empty-string shell vars would clobber real `.env` defaults, so a whitelist of OpenMark-critical keys is scrubbed of `''` before `load_dotenv` runs. This fixed the "0 bookmarks" / `IndexNotFound 'bookmark_embedding'` bug — stale terminal had `NEO4J_DATABASE=''` in os.environ.
   - `openmark/ui/app.py` reads `server_port` from `OPENMARK_PORT` env (default 7860).

3. (No third commit today — the Hermes/Ollama work was earlier, see commits `e5bb649` and `d892198`.)

**Other artifacts written today (NOT committed, intentionally local):**
- `drafts/shai-hulud-team-warning-de.md` and `…-en.md` — Teams warning posts about the Mini Shai-Hulud npm worm.
- `Documents/GenerativeAI-Studio/research/01-community-agents-and-templates.md` — outside OpenMark repo, lives in the OpenGenAI project.

---

## 4. Known gaps + open todos

- **Louvain communities not running.** Hygiene step 5 errors with `gds.graph.project ProcedureNotFound` — Neo4j GDS plugin not installed. Install GDS in Neo4j Desktop 2 to unlock community-based search.
- **LinkedIn URL tracking-param dedup is a recurring post-step.** Each `inject_linkedin_fresh.py` MUST be followed by `graph_hygiene.py`. See `memory/feedback_linkedin_dedup.md`. Permanent fix: strip the param in `openmark/pipeline/normalize.py` for `source='linkedin'`.
- **Tavily API key was pasted in chat today** (in `.env` as `TAVILY_API_KEY`). Rotate at `https://app.tavily.com/home` when convenient.
- **One Gradio banner line is cosmetic-wrong:** when route is `azure-grok`, the startup print says "Responses API" — actual route is Chat Completions. Confusing but harmless. To fix, edit the `print` in `openmark/agent/graph.py` `build_agent()` to branch on `_is_grok(deployment)`.
- **`@tanstack/react-router` floating-version pins** in `othman-hero`, `multica`, `sandcastle/docs`. Pin to fixed versions before next `npm install` (Shai-Hulud risk).

---

## 5. Quick-restart recipes

Each recipe assumes you're at `cd C:\Users\oasrvadmin\Documents\OpenMark` and Neo4j Desktop is running.

### Resume current state (Azure Foundry, grok-4.3)

```powershell
# Ensure .env has these:
#   AGENT_PROVIDER=azure
#   AZURE_DEPLOYMENT_EXECUTOR=grok-4.3
#   AZURE_REASONING_EXECUTOR=high
python openmark\ui\app.py
# Browser: http://127.0.0.1:7860
```

### Switch to local Hermes 3

```powershell
# .env edit:
#   AGENT_PROVIDER=local
ollama serve   # in a separate terminal
python openmark\ui\app.py
```

### Pull fresh LinkedIn + embed

```powershell
Set-Location C:\Users\oasrvadmin\Documents\raindrop-mission
python linkedin_fetch.py                         # writes linkedin_saved.json
Set-Location C:\Users\oasrvadmin\Documents\OpenMark
python scripts\inject_linkedin_fresh.py          # diff + embed + write
python scripts\graph_hygiene.py                  # MANDATORY post-step (LinkedIn dedup)
```

### Dual Grok side-by-side

```powershell
# Window A
$env:AGENT_PROVIDER='azure'; $env:AZURE_DEPLOYMENT_EXECUTOR='grok-4.3'; `
  $env:AZURE_DEPLOYMENT_CLASSIFIER='grok-4.3'; $env:OPENMARK_PORT='7860'; `
  python openmark\ui\app.py

# Window B (different PS session)
$env:AGENT_PROVIDER='azure'; $env:AZURE_DEPLOYMENT_EXECUTOR='grok-4-20-reasoning'; `
  $env:AZURE_DEPLOYMENT_CLASSIFIER='grok-4-20-reasoning'; $env:OPENMARK_PORT='7861'; `
  python openmark\ui\app.py
```

### Diagnostic scripts

| Script | Purpose |
|---|---|
| `scripts/check_config.py` | Verify which DB / model config sees after .env load |
| `scripts/check_neo4j.py` | Bookmark count, indexes, labels, current database |
| `scripts/check_mcp.py` | List MCP tools registered |
| `scripts/check_skills.py` | List discovered openmark-* skills |
| `scripts/check_tavily.py` | Smoke-test Tavily /search /extract /crawl |
| `scripts/check_web_tools.py` | End-to-end test all 6 web tools |
| `scripts/check_web_deps.py` | Pip deps audit (httpx, ddgs, markdownify, etc.) |

---

## 6. Memories index (load these for context)

Already saved at `C:\Users\oasrvadmin\.claude\projects\C--Users-oasrvadmin-Documents-OpenMark\memory\`:

- `project_openmark.md`
- `project_current_state.md`
- `project_agent_routing.md` (new today)
- `project_web_tools.md` (new today)
- `project_graphrag_plan.md`
- `project_pr136_review.md`
- `project_bio_bounty.md`
- `project_open_generative_ai.md`
- `feedback_style.md`
- `feedback_commits.md`
- `feedback_search_strategy.md`
- `feedback_kill_processes.md`
- `feedback_config_dotenv.md` (new today)
- `feedback_linkedin_dedup.md` (new today)
- `reference_github.md`
- `reference_geo_tools.md`
- `user_ahmad.md`

Index file: `MEMORY.md` (one-liner per entry, always loaded into context).

---

**Resume cleanly:** open Claude Code in this dir, type `/init` or just send a question. The new session will load `MEMORY.md` automatically and find this handoff via repo root. Then ask: *"what's the current state?"* and it'll know.
