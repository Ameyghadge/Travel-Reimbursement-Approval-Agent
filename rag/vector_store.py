"""FAISS vector store for policy chunk retrieval."""



from typing import List, Optional

import numpy as np
import faiss
import structlog
from rag.embeddings import EmbeddingModel

logger = structlog.get_logger()


class FAISSStore:
    """FAISS IndexFlatL2 store for small policy corpus."""

    def __init__(self, embedding_model: EmbeddingModel):
        self._embedding_model = embedding_model
        self._index = None
        self._chunks: List[str] = []
        self._metadata: List[dict] = []

    @property
    def is_initialized(self) -> bool:
        return self._index is not None

    def build_index(self, chunks: List[str], metadata: Optional[List[dict]] = None) -> None:
        """Build FAISS index from text chunks."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        self._chunks = chunks
        self._metadata = metadata or [{"section": "unknown"} for _ in chunks]

        embeddings = self._embedding_model.encode(chunks)
        dimension = embeddings.shape[1]

        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings)

        logger.info("faiss_index_built", num_chunks=len(chunks), dimension=dimension)

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Search for top-k most relevant chunks."""
        if not self.is_initialized:
            raise RuntimeError("FAISS index not initialized. Call build_index() first.")

        query_embedding = self._embedding_model.encode([query])
        distances, indices = self._index.search(query_embedding, min(top_k, len(self._chunks)))

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            results.append({
                "content": self._chunks[idx],
                "section": self._metadata[idx].get("section", "unknown"),
                "relevance_score": round(1.0 / (1.0 + float(dist)), 4),
            })

        return results
