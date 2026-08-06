"""Local, offline embedding provider: ``nomic-embed-text-v1.5`` via ``fastembed``.

This is the default and only embedding path in v1 (DEC-003 / COR-002). It runs an ONNX model on
CPU with no API key and no network access — except the one-time model download on first use,
after which the tool operates fully offline.

The model is loaded lazily on the first ``embed()`` call, never at import time: importing
``langgraph_context_mcp`` must stay fast and side-effect-free, and a user running
``--help`` should not pay for an ONNX session (or trigger a download).

``nomic-embed-text-v1.5`` is trained with task-instruction prefixes and ``fastembed`` does not
apply them for us, so this provider adds them itself: documents are embedded with
``search_document:`` and queries with ``search_query:``. Using the same prefix on both sides
measurably degrades retrieval, which is why ``EmbeddingProvider`` separates the two calls.
"""

from __future__ import annotations

import logging
import os
import threading

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# The model this project standardizes on (claude.md's ALLOWED table, DEC-003).
DEFAULT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Its vector width. Hardcoded for the default so `dimension` never has to import fastembed or
# touch the network; an overridden model's width is looked up from fastembed's static catalogue.
DEFAULT_MODEL_DIMENSION = 768

# Optional operator override (claude.md's ENVIRONMENT VARIABLES table). Unset = the default above.
MODEL_ENV_VAR = "LANGGRAPH_CONTEXT_EMBEDDING_MODEL"

# Task-instruction prefixes required by the nomic-embed-text family.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class NomicEmbeddingProvider(EmbeddingProvider):
    """``EmbeddingProvider`` backed by a locally-run ONNX model through ``fastembed``."""

    def __init__(self, model_name: str | None = None) -> None:
        """Configure the provider. Does not load, download, or touch the model.

        ``model_name`` defaults to ``LANGGRAPH_CONTEXT_EMBEDDING_MODEL`` if set, else
        ``nomic-embed-text-v1.5``.
        """
        self._model_name = model_name or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL_NAME
        self._model = None
        self._dimension: int | None = (
            DEFAULT_MODEL_DIMENSION if self._model_name == DEFAULT_MODEL_NAME else None
        )
        # MCP SDK v2 runs sync tool handlers on worker threads (DEC-010), so two searches can
        # race on first use. Loading the model twice would be wasteful, not incorrect — the lock
        # keeps it to once.
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Vector width, resolved without loading the model.

        For the default model this is a constant. For an overridden model it comes from
        ``fastembed``'s static model catalogue, which is a plain in-memory list — no download.
        """
        if self._dimension is None:
            self._dimension = self._lookup_dimension(self._model_name)
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed node chunks as documents. Returns one vector per input text, in input order."""
        if not texts:
            return []
        model = self._ensure_model()
        prefixed = [DOCUMENT_PREFIX + text for text in texts]
        return [list(map(float, vector)) for vector in model.embed(prefixed)]

    def embed_query(self, query: str) -> list[float]:
        """Embed a natural-language search query, with the model's query-side prefix."""
        model = self._ensure_model()
        vectors = [list(map(float, vector)) for vector in model.embed([QUERY_PREFIX + query])]
        return vectors[0]

    # ----------------------------------------------------------------------------------------
    # Lazy model loading
    # ----------------------------------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                # Imported here, not at module scope: this pulls in onnxruntime and may download
                # the model on first ever use.
                from fastembed import TextEmbedding

                logger.info("Loading embedding model %s (first use)", self._model_name)
                self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @staticmethod
    def _lookup_dimension(model_name: str) -> int:
        from fastembed import TextEmbedding

        for description in TextEmbedding.list_supported_models():
            if description["model"].lower() == model_name.lower():
                return int(description["dim"])
        raise ValueError(
            f"Unknown embedding model {model_name!r}. Set {MODEL_ENV_VAR} to a model supported "
            f"by fastembed, or unset it to use {DEFAULT_MODEL_NAME}."
        )
