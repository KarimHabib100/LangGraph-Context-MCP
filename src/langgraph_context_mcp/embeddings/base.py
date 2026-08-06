"""Abstract embedding provider interface.

Every embedding path in this project goes through this interface. The v1 default — and the only
implementation that ships — is local and offline: ``nomic-embed-text-v1.5`` via ``fastembed``
(COR-002 / DEC-003). A cloud provider may be added later behind this same interface, but it must
never become the default path.

``embed()`` embeds *documents* (the node chunks written to the index); ``embed_query()`` embeds a
*search query*. They are kept separate because retrieval models are commonly trained with
distinct task prefixes for the two sides, and mixing them measurably degrades ranking. The
default ``embed_query()`` simply delegates to ``embed()``, so a provider with no such distinction
overrides nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Turns text into dense vectors. Implementations must be deterministic for a given input."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents, returning one vector per input text, in input order.

        Callers batch on purpose: an implementation is free to run the whole list through the
        model in a single pass. An empty input list returns an empty list without loading a model.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of every vector this provider returns. Must not require loading the model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model, persisted alongside the index."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Defaults to treating a query like a document. Override when the model expects a
        query-side task prefix (``nomic-embed-text-v1.5`` does — see ``NomicEmbeddingProvider``).
        """
        return self.embed([query])[0]
