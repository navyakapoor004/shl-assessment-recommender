import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.services.retrieval import retrieval_service

if __name__ == "__main__":
    retrieval_service.load()
    print(f"Indexed {len(retrieval_service.catalog)} catalog items.")