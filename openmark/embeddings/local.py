"""
Local pplx-embed embedding provider.
Uses:
  - perplexity-ai/pplx-embed-v1-0.6b        for queries
  - perplexity-ai/pplx-embed-context-v1-0.6b for documents

Two patches applied at import time:
  1. transformers 4.57 crashes on models without additional_chat_templates folder → catch 404
  2. pplx-embed's st_quantize.py imports sentence_transformers.models.Module (removed in 3.x) → add it back
"""

# ── Patch 1: transformers 4.57 list_repo_templates 404 crash ─
from transformers.utils import hub as _hub
import transformers.tokenization_utils_base as _tub
_orig_lrt = _hub.list_repo_templates
def _safe_lrt(*a, **kw):
    try:
        return _orig_lrt(*a, **kw)
    except Exception:
        return []
_hub.list_repo_templates = _safe_lrt
_tub.list_repo_templates = _safe_lrt

# ── Patch 2: sentence_transformers.models.Module missing ─────
import torch.nn as _nn
import sentence_transformers.models as _st_models
if not hasattr(_st_models, "Module"):
    _st_models.Module = _nn.Module

from sentence_transformers import SentenceTransformer
import numpy as np
from openmark.embeddings.base import EmbeddingProvider
from openmark import config


class LocalEmbedder(EmbeddingProvider):
    def __init__(self):
        print("Loading pplx-embed query model...")
        self._query_model = SentenceTransformer(config.PPLX_QUERY_MODEL, trust_remote_code=True)
        print("Loading pplx-embed document model...")
        self._doc_model = SentenceTransformer(config.PPLX_DOC_MODEL, trust_remote_code=True)
        print("Local embedder ready.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._doc_model.encode(texts, batch_size=32, show_progress_bar=True)
        return embeddings.astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._query_model.encode([text])
        return embedding[0].astype(float).tolist()

    @property
    def dimension(self) -> int:
        return 1024
