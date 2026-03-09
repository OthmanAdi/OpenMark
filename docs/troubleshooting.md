# Troubleshooting

---

## pplx-embed fails to load

**Error:** `ImportError: cannot import name 'Module' from 'sentence_transformers.models'`

**Cause:** pplx-embed's custom `st_quantize.py` imports `Module` from `sentence_transformers.models`, which was removed in version 4.x.

**Fix:** Pin to the correct version:
```bash
pip install "sentence-transformers==3.3.1"
```

---

## pplx-embed crashes with 404 on chat templates

**Error:** `RemoteEntryNotFoundError: 404 ... additional_chat_templates does not exist`

**Cause:** `transformers 4.57+` added `list_repo_templates()` which looks for an `additional_chat_templates` folder in every model repo. pplx-embed predates this feature and doesn't have the folder.

**Fix:** Already handled automatically in `openmark/embeddings/local.py` via a monkey-patch applied before model loading. If you see this error outside of OpenMark, apply:
```python
from transformers.utils import hub as _hub
import transformers.tokenization_utils_base as _tub
_orig = _hub.list_repo_templates
def _safe(*a, **kw):
    try: return _orig(*a, **kw)
    except Exception: return []
_hub.list_repo_templates = _safe
_tub.list_repo_templates = _safe
```

---

## Neo4j connection error: "Unable to retrieve routing information"

**Cause:** Using `neo4j://` URI (routing protocol) with a single local Neo4j instance.

**Fix:** Use `bolt://` instead:
```env
NEO4J_URI=bolt://127.0.0.1:7687
```

---

## Neo4j error: "Database does not exist"

**Cause:** The database name in `.env` doesn't match what's in Neo4j Desktop.

**Fix:** Open `http://localhost:7474`, check what databases exist:
```cypher
SHOW DATABASES
```
Update `NEO4J_DATABASE` in `.env` to match.

---

## LinkedIn script returns 0 results or 404

**Cause:** LinkedIn's internal `queryId` changes when they deploy new JavaScript bundles.

**Fix:**
1. Open LinkedIn in your browser → go to Saved Posts
2. Open DevTools → Network tab → filter for `voyagerSearchDashClusters`
3. Click one of the requests → copy the full URL
4. Extract the new `queryId` value
5. Update `linkedin_fetch.py` with the new `queryId`

---

## YouTube OAuth "Access Blocked: App not verified"

**Cause:** Your Google Cloud app is in testing mode and your account isn't listed as a test user.

**Fix:**
1. Google Cloud Console → OAuth consent screen
2. Scroll to "Test users" → Add users → add your Google account email
3. Re-run `youtube_fetch.py`

---

## ChromaDB ingest is slow

On CPU with local pplx-embed, embedding 8K items takes ~20 minutes. This is normal.

**Options:**
- Use Azure instead: `python scripts/ingest.py --provider azure` (~5 min, ~€0.30)
- The ingest is resumable — if interrupted, re-run and it skips already-ingested items

---

## SIMILAR_TO step takes too long

Building SIMILAR_TO edges queries ChromaDB for every bookmark's top-5 neighbors, then writes to Neo4j. For 8K items on CPU this takes ~25-40 minutes.

**Skip it:**
```bash
python scripts/ingest.py --skip-similar
```
The app works without SIMILAR_TO edges. You only lose the `find_similar_bookmarks` agent tool and cross-topic graph traversal.

---

## Windows UnicodeEncodeError in terminal

**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Cause:** Windows terminal (cmd/PowerShell) defaults to cp1252 encoding which can't handle emoji or some Unicode characters in bookmark titles.

**Fix:** Run from Windows Terminal (supports UTF-8) or add to the top of the script:
```python
import sys
sys.stdout.reconfigure(encoding='utf-8')
```
All OpenMark scripts already include this.

---

## gradio not found on Python 3.13

gradio 6.6.0 is installed on Python 3.14 by default on this machine. If using Python 3.13:
```bash
C:\Python313\python -m pip install gradio
```
