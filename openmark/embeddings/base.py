from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base — swap local pplx-embed or Azure without changing any other code."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...
