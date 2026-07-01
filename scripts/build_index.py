"""
One-time offline index build. Useful to pre-warm data/catalog.faiss before
deploying, so the server's first startup is fast.

    python scripts/build_index.py
"""
from app.services.retrieval import retrieval_service

if __name__ == "__main__":
    retrieval_service.load()
    print(f"Indexed {len(retrieval_service.catalog)} catalog items.")
