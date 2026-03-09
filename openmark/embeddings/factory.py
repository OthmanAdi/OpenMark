from openmark import config
from openmark.embeddings.base import EmbeddingProvider


def get_embedder() -> EmbeddingProvider:
    """Return the configured embedding provider based on EMBEDDING_PROVIDER env var."""
    provider = config.EMBEDDING_PROVIDER.lower()
    if provider == "local":
        from openmark.embeddings.local import LocalEmbedder
        return LocalEmbedder()
    elif provider == "azure":
        from openmark.embeddings.azure import AzureEmbedder
        return AzureEmbedder()
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: '{provider}'. Use 'local' or 'azure'.")
