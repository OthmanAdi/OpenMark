"""
Optional cross-encoder rerank pass.

Default OFF. Enable with `OPENMARK_RERANK=1` in `.env`. When on, every
retrieval tool that wraps `vector_search` / `hybrid_search` reranks the
top-N pool (default 50) down to the requested `n` via bge-reranker-v2-m3.

The reranker is ~2.27 GB on first download and runs on CPU at ~3-5 s per 50
short-text pairs (or ~150-300 ms on CUDA with fp16). If the dependency
isn't installed (`pip install FlagEmbedding`), the env flag is ignored and
the caller gets the original pool back untouched — no crash, no silent
ranking change.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

log = logging.getLogger("openmark.agent.rerank")


def is_enabled() -> bool:
    return os.getenv("OPENMARK_RERANK", "").strip() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def _get_reranker():
    """Load FlagReranker lazily. Returns None if FlagEmbedding isn't installed
    or the model fails to load — caller treats `None` as "pass through"."""
    try:
        from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]
    except Exception as e:
        log.info(f"[rerank] FlagEmbedding not installed ({e!r}); rerank disabled")
        return None
    try:
        model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
        log.info("[rerank] FlagReranker bge-reranker-v2-m3 ready")
        return model
    except Exception as e:
        log.info(f"[rerank] FlagReranker load failed ({e!r}); rerank disabled")
        return None


def rerank_rows(
    query: str,
    rows: list[dict],
    text_keys: tuple[str, ...] = ("doc_text", "title"),
    top_k: int = 10,
    char_cap: int = 512,
) -> list[dict]:
    """Cross-encoder rerank a list of bookmark rows down to top_k.

    text_keys is a fallback chain — for each row, the first non-empty value
    is the text scored against the query. doc_text > title is the default
    because doc_text bundles category + tags + excerpt and ranks better.

    If the reranker isn't loaded or rows is empty, returns rows unchanged
    (truncated to top_k). Never raises.
    """
    if not rows:
        return rows
    if not is_enabled():
        return rows[:top_k]
    model = _get_reranker()
    if model is None:
        return rows[:top_k]

    def _pick(r: dict) -> str:
        for k in text_keys:
            v = r.get(k)
            if v:
                return (v or "")[:char_cap]
        return ""

    pairs = [[query, _pick(r)] for r in rows]
    try:
        scores = model.compute_score(pairs, normalize=True)
    except Exception as e:
        log.info(f"[rerank] compute_score failed ({e!r}); returning original order")
        return rows[:top_k]

    if not isinstance(scores, list):
        scores = [scores]
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for row, score in ranked[:top_k]:
        annotated = dict(row)
        annotated["rerank_score"] = float(score)
        out.append(annotated)
    return out
