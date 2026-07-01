"""
Retrieval service: loads catalog.json, builds/loads a FAISS index over
sentence-transformer embeddings of each item's description, and exposes
a hybrid search (semantic + keyword boost) with hard metadata filtering.

Index build is offline / one-time (on startup, cached to disk). Search is
pure Python + numpy/faiss, no LLM call involved — that's Call-free by design.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas import CatalogItem

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "catalog.faiss"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this


class RetrievalService:
    def __init__(self) -> None:
        self.catalog: list[CatalogItem] = []
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load catalog + embedding model, build (or load cached) FAISS index."""
        raw = json.loads(CATALOG_PATH.read_text())
        self.catalog = [CatalogItem(**item) for item in raw]

        self.model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)

        texts = [self._embed_text(item) for item in self.catalog]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normalized vecs
        index.add(embeddings)
        self.index = index

        # Cache to disk so restarts are instant (optional, safe to skip in dev)
        try:
            faiss.write_index(self.index, str(INDEX_PATH))
        except Exception:
            pass  # non-fatal; index just gets rebuilt in-memory next boot

    @staticmethod
    def _embed_text(item: CatalogItem) -> str:
        return f"""
    Name: {item.name}
    Description:
    {item.description}
    Test Type:
    {item.test_type.value}
    Level:
    {item.level.value}
    Duration:
    {item.duration_minutes}
"""

    # -- search --------------------------------------------------------

    def search(
        self,
        query: str,
        test_type: list[str] | None = None,
        level: str | None = None,
        max_duration_minutes: int | None = None,
        top_k: int = 10,
    ) -> list[CatalogItem]:
        """
        Hybrid retrieve:
          1. Hard metadata filter (test_type / level / duration) on the catalog.
          2. Semantic search via FAISS over the filtered subset.
          3. Keyword boost for exact-term matches (e.g. "Java" must not get
             lost among fuzzy semantic neighbors like "JavaScript").
        """
        if self.model is None or self.index is None:
            raise RuntimeError("RetrievalService.load() must be called before search()")

        candidates = self._apply_hard_filters(test_type, level, max_duration_minutes)
        if not candidates:
            return []

        candidate_idx = [self.catalog.index(c) for c in candidates]

        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")

        # Search the full index, then keep only candidate_idx (simplest correct
        # approach for a catalog this size; swap to IndexIDMap for huge catalogs)
        k = min(len(self.catalog), max(top_k * 3, 20))
        scores, indices = self.index.search(query_vec, k)

        scored: list[tuple[float, int]] = []
        candidate_set = set(candidate_idx)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx not in candidate_set:
                continue
            boosted = score + self._keyword_boost(query, self.catalog[idx])
            scored.append((boosted, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scored[:top_k]]
        return [self.catalog[i] for i in top_indices]

    def _apply_hard_filters(
        self,
        test_type: list[str] | None,
        level: str | None,
        max_duration_minutes: int | None,
    ) -> list[CatalogItem]:
        results = self.catalog
        if test_type:
            wanted = set(test_type)
            results = [c for c in results if c.test_type.value in wanted]
        if level and level != "all_levels":
            results = [
                c for c in results
                if c.level.value == level or c.level.value == "all_levels"
            ]
        if max_duration_minutes:
            results = [
                c for c in results
                if c.duration_minutes is None or c.duration_minutes <= max_duration_minutes
            ]
        return results

    @staticmethod
    def _keyword_boost(query: str, item: CatalogItem) -> float:
        """Small additive boost when a query token exact-matches the item name.
        Prevents 'Java' queries from being swamped by 'JavaScript' etc."""
        query_tokens = set(re.findall(r"[a-zA-Z0-9+#.]+", query.lower()))
        name_tokens = set(re.findall(r"[a-zA-Z0-9+#.]+", item.name.lower()))
        overlap = query_tokens & name_tokens
        return 0.25 * len(overlap)

# Singleton instance used by the FastAPI app
retrieval_service = RetrievalService()
