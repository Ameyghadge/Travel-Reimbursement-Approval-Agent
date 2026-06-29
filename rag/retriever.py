"""Policy retriever: loads, chunks, and retrieves policy sections."""



import re
from pathlib import Path
from typing import List, Tuple

import structlog
from rag.vector_store import FAISSStore

logger = structlog.get_logger()


class PolicyRetriever:
    """Chunks the travel policy and retrieves relevant sections via FAISS."""

    def __init__(self, policy_path: str, store: FAISSStore):
        self._policy_path = Path(policy_path)
        self._store = store
        self._initialized = False

    def initialize(self) -> None:
        """Load policy, chunk it, build FAISS index. Idempotent."""
        if self._initialized:
            return

        if not self._policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self._policy_path}")

        raw_text = self._policy_path.read_text(encoding="utf-8")
        chunks, metadata = self._chunk_policy(raw_text)

        self._store.build_index(chunks, metadata)
        self._initialized = True
        logger.info("policy_retriever_initialized", chunks=len(chunks))

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """Retrieve top-k relevant policy chunks for a query."""
        if not self._initialized:
            self.initialize()
        return self._store.search(query, top_k)

    def _chunk_policy(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> Tuple[List[str], List[dict]]:
        """Split policy by markdown headers, then by size."""
        sections = re.split(r"\n(?=## )", text)
        chunks = []
        metadata = []

        for section in sections:
            lines = section.strip().split("\n")
            # Extract section title
            title = lines[0].strip("# ").strip() if lines else "General"
            content = section.strip()

            if len(content) <= chunk_size:
                chunks.append(content)
                metadata.append({"section": title})
            else:
                # Split large sections with overlap
                for i in range(0, len(content), chunk_size - overlap):
                    chunk = content[i : i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                        metadata.append({"section": title})

        return chunks, metadata
