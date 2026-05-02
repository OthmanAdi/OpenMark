"""
Perplexity pplx-embed local embedder.

Uses two models from the same 1024-dim cosine vector space:
  - pplx-embed-context-v1-0.6B  (doc storage — RAG-optimised)
  - pplx-embed-v1-0.6B          (query encoding — general purpose)

Both models are MIT-licensed, run fully locally via sentence-transformers.
First run downloads ~1.2 GB from HuggingFace (both 0.6B weights).

Requires:  sentence-transformers>=3.3.1   trust_remote_code=True
"""

from sentence_transformers import SentenceTransformer
from openmark.embeddings.base import EmbeddingProvider
from openmark import config


class PplxEmbedder(EmbeddingProvider):
    def __init__(self):
        doc_model_id   = config.PPLX_DOC_MODEL
        query_model_id = config.PPLX_QUERY_MODEL

        print(f"Loading pplx doc model:   {doc_model_id}")
        self._doc_model = SentenceTransformer(doc_model_id, trust_remote_code=True)

        if query_model_id == doc_model_id:
            # Same weights — reuse, don't load twice
            self._query_model = self._doc_model
            print(f"pplx query model: shared with doc model")
        else:
            print(f"Loading pplx query model: {query_model_id}")
            self._query_model = SentenceTransformer(query_model_id, trust_remote_code=True)

        print(f"pplx embedder ready ({self.dimension}-dim)")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        batch_size = 256
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = self._doc_model.encode(batch, show_progress_bar=False)
            results.extend(vecs.tolist())
            done = min(i + batch_size, len(texts))
            if done % 1000 == 0 or done == len(texts):
                print(f"  pplx embedded {done}/{len(texts)}")
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._query_model.encode([text], show_progress_bar=False)[0].tolist()

    @property
    def dimension(self) -> int:
        return config.pplx_dimension()
