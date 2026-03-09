# HuggingFace Publishing Guide

OpenMark publishes two things on HuggingFace:
1. **Space** — live Gradio demo at `OthmanAdi/OpenMark`
2. **Dataset** — the categorized bookmarks at `OthmanAdi/openmark-bookmarks`

---

## Prerequisites

You need a HuggingFace account and a **write-access token**:
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token → **Write** access
3. Add to your `.env`:
   ```
   HF_TOKEN=hf_your_token_here
   ```

---

## 1. HuggingFace Space (Gradio Demo)

The Space hosts the Gradio UI publicly (or privately until you're ready).

**Create the Space:**
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()
api = HfApi(token=os.getenv('HF_TOKEN'))
api.create_repo(
    repo_id='OthmanAdi/OpenMark',
    repo_type='space',
    space_sdk='gradio',
    private=True,
)
print('Space created: https://huggingface.co/spaces/OthmanAdi/OpenMark')
"
```

**Push the code to the Space:**
```bash
python -c "
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()
api = HfApi(token=os.getenv('HF_TOKEN'))
api.upload_folder(
    folder_path='.',
    repo_id='OthmanAdi/OpenMark',
    repo_type='space',
    ignore_patterns=['.env', 'data/chroma_db/*', '__pycache__/*', '.git/*'],
)
"
```

> **Note:** The Space version requires your ChromaDB and Neo4j data to be pre-loaded. For a public demo, you would host a sample dataset. For private use, the full local setup is better.

---

## 2. HuggingFace Dataset

The dataset card publishes your 8,000+ categorized bookmarks as a reusable dataset for RAG experiments.

**What's in the dataset:**
- URL, title, category (19 categories), tags, score (1-10), source
- Sources: Raindrop, Edge browser, LinkedIn, YouTube, daily.dev
- ~8,007 unique items after deduplication

**Create the dataset repo:**
```bash
python -c "
from huggingface_hub import HfApi
import os, json
from dotenv import load_dotenv
load_dotenv()
api = HfApi(token=os.getenv('HF_TOKEN'))

# Create private dataset repo
api.create_repo(
    repo_id='OthmanAdi/openmark-bookmarks',
    repo_type='dataset',
    private=True,
)

# Upload dataset card
api.upload_file(
    path_or_fileobj='docs/dataset_card.md',
    path_in_repo='README.md',
    repo_id='OthmanAdi/openmark-bookmarks',
    repo_type='dataset',
)

# Upload the data (RAINDROP_MISSION_DIR/CATEGORIZED.json)
api.upload_file(
    path_or_fileobj=os.path.join(os.getenv('RAINDROP_MISSION_DIR'), 'CATEGORIZED.json'),
    path_in_repo='data/bookmarks.json',
    repo_id='OthmanAdi/openmark-bookmarks',
    repo_type='dataset',
)
print('Dataset created: https://huggingface.co/datasets/OthmanAdi/openmark-bookmarks')
"
```

---

## Making Public

When you're ready to go public, flip visibility:
```bash
python -c "
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()
api = HfApi(token=os.getenv('HF_TOKEN'))

# Make Space public
api.update_repo_visibility('OthmanAdi/OpenMark', private=False, repo_type='space')

# Make Dataset public
api.update_repo_visibility('OthmanAdi/openmark-bookmarks', private=False, repo_type='dataset')
print('Both are now public.')
"
```
