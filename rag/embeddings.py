"""Embedding model wrapper using sentence-transformers."""



from typing import List

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class EmbeddingModel:
    """Singleton wrapper around sentence-transformers for embeddings."""

    _instance = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        logger.info("loading_embedding_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("embedding_model_loaded", model=model_name)

    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-small-en-v1.5") -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into dense embeddings."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)
