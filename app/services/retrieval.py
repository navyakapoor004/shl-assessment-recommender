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

EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"


class RetrievalService:

    def __init__(self) -> None:
        self.catalog: list[CatalogItem] = []
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None

    # -----------------------------------------------------

    def load(self) -> None:
        """Load catalog and FAISS index."""

        raw = json.loads(CATALOG_PATH.read_text())
        self.catalog = [CatalogItem(**item) for item in raw]

        # If index already exists, don't rebuild
        if INDEX_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            return

        print("Building FAISS index (first startup only)...")

        self.model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)

        texts = [self._embed_text(item) for item in self.catalog]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, str(INDEX_PATH))

        # free memory
        del embeddings
        del self.model
        self.model = None

    # -----------------------------------------------------

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

    # -----------------------------------------------------

    def search(
        self,
        query: str,
        test_type: list[str] | None = None,
        level: str | None = None,
        max_duration_minutes: int | None = None,
        top_k: int = 10,
    ) -> list[CatalogItem]:

        if self.index is None:
            raise RuntimeError("RetrievalService.load() must be called first.")

        if self.model is None:
            self.model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)

        candidates = self._apply_hard_filters(
            test_type,
            level,
            max_duration_minutes,
        )

        if not candidates:
            return []

        candidate_idx = [self.catalog.index(c) for c in candidates]

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        k = min(len(self.catalog), max(top_k * 3, 20))

        scores, indices = self.index.search(query_vec, k)

        scored = []

        candidate_set = set(candidate_idx)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if idx not in candidate_set:
                continue

            score += self._keyword_boost(query, self.catalog[idx])

            scored.append((score, idx))

        scored.sort(reverse=True)

        return [self.catalog[idx] for _, idx in scored[:top_k]]

    # -----------------------------------------------------

    def _apply_hard_filters(
        self,
        test_type,
        level,
        max_duration_minutes,
    ):

        results = self.catalog

        if test_type:
            wanted = set(test_type)
            results = [
                c
                for c in results
                if c.test_type.value in wanted
            ]

        if level and level != "all_levels":
            results = [
                c
                for c in results
                if c.level.value in (level, "all_levels")
            ]

        if max_duration_minutes:
            results = [
                c
                for c in results
                if (
                    c.duration_minutes is None
                    or c.duration_minutes <= max_duration_minutes
                )
            ]

        return results

    # -----------------------------------------------------

    @staticmethod
    def _keyword_boost(query: str, item: CatalogItem):

        query_tokens = set(
            re.findall(r"[a-zA-Z0-9+#.]+", query.lower())
        )

        name_tokens = set(
            re.findall(r"[a-zA-Z0-9+#.]+", item.name.lower())
        )

        return 0.25 * len(query_tokens & name_tokens)


retrieval_service = RetrievalService()