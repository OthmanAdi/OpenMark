"""
Azure AI Foundry embedding provider.
Uses text-embedding-ada-002 (or whatever deployment is configured).
"""

from openai import AzureOpenAI
from openmark.embeddings.base import EmbeddingProvider
from openmark import config


class AzureEmbedder(EmbeddingProvider):
    def __init__(self):
        self._client = AzureOpenAI(
            azure_endpoint=config.AZURE_ENDPOINT,
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
        )
        self._deployment = config.AZURE_DEPLOYMENT_EMBED
        print(f"Azure embedder ready — deployment: {self._deployment}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._deployment,
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self._embed(batch))
            print(f"  Azure embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    @property
    def dimension(self) -> int:
        return {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }.get(self._deployment, 1536)
